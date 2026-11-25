"""Score explanations of ML surrogates against simulator ground truth (EnergyPlus villa, scheduled power cuts).

Truth (from EnergyPlus, results/*.parquet):
  Sobol first-order / total indices (independent inputs) ........ global
  interventional partial dependence on a grid, IND/COR background  main effects
  baseline Shapley values for 50 instances per design ............ local
Surrogates: LightGBM, MLP, ridge, trained on IND (independent) or COR (correlated stock) designs.
Explainers: TreeSHAP interventional / path-dependent, permutation SHAP, PDP, ALE, SHAP dependence.
Outputs -> ../results/xai_*.csv, ../results/xai_store.pkl
"""
import sys, time, warnings, numpy as np, pandas as pd, lightgbm as lgb, shap
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import run_designs as RD

warnings.filterwarnings("ignore")
R = RD.R; NAMES = RD.NAMES; D = RD.D; LO, HI = RD.LO, RD.HI
TARGETS = ["Tpeak_FF", "DH28_FF", "t_exceed_FF", "rebound_kW", "E_cool_kWh"]
BASE = (LO + HI) / 2

# ------------------------------------------------------------------ ground truth
def sobol_truth(nboot=200):
    df = pd.read_parquet(R / "saltelli.parquet"); rows = []
    for tg in TARGETS:
        fA = df[df.block == "A"].sort_values("row")[tg].values; fB = df[df.block == "B"].sort_values("row")[tg].values
        rng = np.random.default_rng(0); N = len(fA)
        for n in NAMES:
            fAB = df[df.block == f"AB_{n}"].sort_values("row")[tg].values
            def est(ix):
                v = np.var(np.r_[fA[ix], fB[ix]])
                return np.mean(fB[ix] * (fAB[ix] - fA[ix])) / v, 0.5 * np.mean((fA[ix] - fAB[ix]) ** 2) / v
            s1, st = est(np.arange(N)); bs = np.array([est(rng.integers(0, N, N)) for _ in range(nboot)])
            rows.append({"target": tg, "feature": n, "S1": s1, "ST": st, "S1_lo": np.percentile(bs[:, 0], 2.5), "S1_hi": np.percentile(bs[:, 0], 97.5),
                         "ST_lo": np.percentile(bs[:, 1], 2.5), "ST_hi": np.percentile(bs[:, 1], 97.5)})
    return pd.DataFrame(rows)

def local_truth(tag):
    """baseline Shapley from permutation sampling: average marginal contribution of each added feature."""
    df = pd.read_parquet(R / f"local_truth_{tag}.parquet"); inst = pd.read_parquet(R / f"local_instances_{tag}.parquet")
    out = {}
    for tg in TARGETS:
        phi = np.zeros((len(inst), D)); se = np.zeros((len(inst), D))
        for k, g in df.groupby("inst"):
            contrib = {j: [] for j in range(D)}
            for m, gp in g.groupby("perm"):
                v = gp[tg].values; added = gp["added"].values
                for pos in range(1, len(v)): contrib[added[pos]].append(v[pos] - v[pos - 1])
            phi[k] = [np.mean(contrib[j]) for j in range(D)]
            se[k] = [np.std(contrib[j], ddof=1) / np.sqrt(len(contrib[j])) for j in range(D)]
        out[tg] = (phi, se)
    return inst[NAMES].values, out

def pdp_truth(tag):
    df = pd.read_parquet(R / f"pdp_truth_{tag}.parquet")
    return df.groupby(["feature", "value"])[TARGETS].mean().reset_index()

# ------------------------------------------------------------------ surrogates
def load_design(tag):
    df = pd.read_parquet(R / f"train_{tag}.parquet")
    if "error" in df: df = df[df["error"].isna()]
    return df[NAMES].values, df[TARGETS]

def make_model(kind):
    if kind == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.9, verbose=-1, n_jobs=14, random_state=0)
    if kind == "MLP":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 128, 64), alpha=1e-4, learning_rate_init=1e-3, max_iter=3000,
                                                            early_stopping=True, n_iter_no_change=50, random_state=0))
    if kind == "Ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))

class YScaled:
    """wrap a model so that it is fitted on standardised y (helps the MLP) but predicts in physical units."""
    def __init__(self, m): self.m = m
    def fit(self, X, y): self.mu, self.sd = y.mean(), y.std(); self.m.fit(X, (y - self.mu) / self.sd); return self
    def predict(self, X): return self.m.predict(X) * self.sd + self.mu

# ------------------------------------------------------------------ explainers
def ale_curve(f, X, i, grid):
    """first-order ALE on the given grid (uncentred cumulative, then centred by data-weighted mean)."""
    x = X[:, i]; eff = np.zeros(len(grid)); cnt = np.zeros(len(grid) - 1)
    for k in range(len(grid) - 1):
        sel = (x >= grid[k]) & (x <= grid[k + 1]) if k == 0 else (x > grid[k]) & (x <= grid[k + 1])
        if sel.sum() == 0: continue
        lo, hi = X[sel].copy(), X[sel].copy(); lo[:, i] = grid[k]; hi[:, i] = grid[k + 1]
        eff[k + 1] = np.mean(f(hi) - f(lo)); cnt[k] = sel.sum()
    acc = np.cumsum(eff); mid = (acc[:-1] + acc[1:]) / 2
    return acc - (np.sum(mid * cnt) / max(cnt.sum(), 1))

def pdp_curve(f, BG, i, grid):
    out = []
    for v in grid:
        Z = BG.copy(); Z[:, i] = v; out.append(f(Z).mean())
    return np.array(out)

def shap_dependence_curve(phi, X, i, grid):
    """binned mean SHAP value of feature i, on the grid points (nearest-bin)."""
    edges = (grid[:-1] + grid[1:]) / 2; b = np.digitize(X[:, i], edges)
    return np.array([phi[b == k, i].mean() if (b == k).any() else np.nan for k in range(len(grid))])

def centre(c): return c - np.nanmean(c)

def curve_err(est, truth):
    e, t = centre(est), centre(truth); rng = np.ptp(t)
    return float(np.sqrt(np.nanmean((e - t) ** 2)) / rng) if rng > 1e-9 else np.nan

def local_scores(phi_s, phi_t, se_t):
    cos = np.sum(phi_s * phi_t, 1) / (np.linalg.norm(phi_s, axis=1) * np.linalg.norm(phi_t, axis=1) + 1e-12)
    big = np.abs(phi_t) > np.maximum(0.05 * np.abs(phi_t).sum(1, keepdims=True), 2 * se_t)
    sign = np.where(big, np.sign(phi_s) == np.sign(phi_t), np.nan)
    top = (np.abs(phi_s).argmax(1) == np.abs(phi_t).argmax(1))
    nmae = np.abs(phi_s - phi_t).sum(1) / (np.abs(phi_t).sum(1) + 1e-12)
    return {"cosine": np.mean(cos), "sign_agree_%": 100 * np.nanmean(sign), "top_feature_agree_%": 100 * top.mean(), "rel_L1_error": np.mean(nmae)}

# ------------------------------------------------------------------ main experiment
def run(n_train_list=(250, 500, 1000, 3000), kinds=("LightGBM", "MLP", "Ridge")):
    sob = sobol_truth(); sob.to_csv(R / "xai_sobol_truth.csv", index=False)
    pdpT = {tag: pdp_truth(tag) for tag in ("IND", "COR")}
    loc = {tag: local_truth(tag) for tag in ("IND", "COR")}
    BGs = {"IND": RD.sample_ind(150, 31), "COR": RD.sample_cor(150, 32)}
    rows_acc, rows_glob, rows_main, rows_loc, store = [], [], [], [], {}
    for tag in ("IND", "COR"):
        X, Y = load_design(tag); Xte, Yte = X[3000:], Y.iloc[3000:]
        Xother, Yother = load_design("IND" if tag == "COR" else "COR")
        Xl, truthL = loc[tag]
        for tg in TARGETS:
            st = sob[sob.target == tg].set_index("feature").loc[NAMES]
            for kind in kinds:
                for n in (n_train_list if kind == "LightGBM" else (3000,)):
                    t0 = time.time(); y = Y[tg].values[:n]
                    m = YScaled(make_model(kind)).fit(X[:n], y) if kind != "LightGBM" else make_model(kind).fit(X[:n], y)
                    f = m.predict
                    acc = {"design": tag, "target": tg, "model": kind, "n_train": n,
                           "R2_test": r2_score(Yte[tg], f(Xte)), "R2_other_design": r2_score(Yother[tg], f(Xother)),
                           "RMSE_test": float(np.sqrt(np.mean((f(Xte) - Yte[tg].values) ** 2)))}
                    rows_acc.append(acc)
                    # ---- global: interventional SHAP on 500 test rows, background 200 training rows
                    bg = X[:200]; xs = Xte[:500]
                    if kind == "LightGBM":
                        phi_int = shap.TreeExplainer(m, data=bg, feature_perturbation="interventional").shap_values(xs, check_additivity=False)
                        phi_path = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent").shap_values(xs, check_additivity=False)
                    else:
                        phi_int = shap.PermutationExplainer(f, shap.maskers.Independent(bg, max_samples=200), seed=0)(xs, max_evals=2 * D + 1 + 2 * D * 10, silent=True).values
                        phi_path = None
                    imp = {"SHAP_interventional": np.abs(phi_int).mean(0)}
                    if phi_path is not None: imp["SHAP_path_dependent"] = np.abs(phi_path).mean(0)
                    for en, v in imp.items():
                        rho = spearmanr(v, st.ST.values).statistic
                        top3 = len(set(np.argsort(-v)[:3]) & set(np.argsort(-st.ST.values)[:3]))
                        share = v / v.sum(); sh_t = st.ST.values / st.ST.values.sum()
                        rows_glob.append({**{k: acc[k] for k in ("design", "target", "model", "n_train")}, "explainer": en,
                                          "spearman_vs_SobolST": rho, "top3_overlap": top3, "share_L1_vs_SobolST": float(np.abs(share - sh_t).sum()),
                                          **{f"share_{nm}": s for nm, s in zip(NAMES, share)}})
                    # ---- main effects vs interventional PDP truth (same background points)
                    pt = pdpT[tag]; BG = BGs[tag]
                    for i, nm in enumerate(NAMES):
                        tcur = pt[pt.feature == nm].sort_values("value"); grid = tcur.value.values; truth = tcur[tg].values
                        curves = {"PDP": pdp_curve(f, BG, i, grid), "ALE": ale_curve(f, X[:n], i, grid),
                                  "SHAP_dependence": shap_dependence_curve(phi_int, xs, i, grid)}
                        for en, c in curves.items():
                            rows_main.append({**{k: acc[k] for k in ("design", "target", "model", "n_train")}, "feature": nm, "method": en,
                                              "rel_RMSE": curve_err(c, truth), "truth_range": float(np.ptp(truth))})
                        if n == 3000: store[(tag, tg, kind, nm)] = {"grid": grid, "truth": truth, **curves}
                    # ---- local: baseline Shapley of the surrogate (exact) vs simulator
                    phi_t, se_t = truthL[tg]
                    if kind == "LightGBM":
                        phi_s = shap.TreeExplainer(m, data=BASE[None, :], feature_perturbation="interventional").shap_values(Xl, check_additivity=False)
                    else:
                        phi_s = shap.ExactExplainer(f, shap.maskers.Independent(BASE[None, :]))(Xl).values
                    rows_loc.append({**{k: acc[k] for k in ("design", "target", "model", "n_train")}, **local_scores(phi_s, phi_t, se_t),
                                     "truth_mean_abs": float(np.abs(phi_t).mean())})
                    gl = {"mean_abs_phi_true": np.abs(phi_t).mean(0), "mean_abs_phi_sur": np.abs(phi_s).mean(0)}
                    rows_glob.append({**{k: acc[k] for k in ("design", "target", "model", "n_train")}, "explainer": "baseline_Shapley_vs_simulator",
                                      "spearman_vs_SobolST": spearmanr(gl["mean_abs_phi_sur"], gl["mean_abs_phi_true"]).statistic,
                                      "top3_overlap": len(set(np.argsort(-gl["mean_abs_phi_sur"])[:3]) & set(np.argsort(-gl["mean_abs_phi_true"])[:3])),
                                      "share_L1_vs_SobolST": float(np.abs(gl["mean_abs_phi_sur"] / gl["mean_abs_phi_sur"].sum() - gl["mean_abs_phi_true"] / gl["mean_abs_phi_true"].sum()).sum())})
                    if n == 3000: store[("local", tag, tg, kind)] = {"phi_sur": phi_s, "phi_true": phi_t, "se_true": se_t, "X": Xl,
                                                                      "phi_int": phi_int, "xs": xs, "phi_path": phi_path}
                    print(f"{tag} {tg} {kind} n={n}: R2={acc['R2_test']:.3f} other={acc['R2_other_design']:.3f} "
                          f"rho_ST={rows_glob[-2 if phi_path is None else -3]['spearman_vs_SobolST']:.2f} local cos={rows_loc[-1]['cosine']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    for nm, rows in (("accuracy", rows_acc), ("global", rows_glob), ("main_effects", rows_main), ("local", rows_loc)):
        pd.DataFrame(rows).to_csv(R / f"xai_{nm}.csv", index=False)
    pd.to_pickle(store, R / "xai_store.pkl")

if __name__ == "__main__":
    run()
