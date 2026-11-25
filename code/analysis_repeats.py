"""Repeated-training analysis of explanation fidelity (no new simulations).

stage sobol   : centred Saltelli first-order and Jansen total Sobol indices, 200 bootstrap resamples
stage repeats : 5 random 3000/1000 splits x {IND, COR, STR, STR-mass} x 5 outputs x 3 surrogates;
                accuracy, transfer R2, global and local fidelity, per-input errors and signs, main-effect slope ratios
stage baselines : model-free baseline explanations
stage ovb     : omitted-variable-bias prediction of slope ratios (STR, mass omitted) vs observed
Outputs -> ../results/p1_*"""
import sys, time, warnings, numpy as np, pandas as pd, shap
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import xai as X, run_designs as RD

warnings.filterwarnings("ignore")
R = X.R; NAMES = X.NAMES; D = X.D; TARGETS = X.TARGETS; BASE = X.BASE; LO, HI = X.LO, X.HI
CASES = {"IND": ("IND", []), "COR": ("COR", []), "STR": ("STR", []), "STR-mass": ("STR", ["mass"])}
BGS = {"IND": RD.sample_ind(150, 31), "COR": RD.sample_cor(150, 32), "STR": RD.sample_str(150, 33)}
SEEDS = range(5)

# ------------------------------------------------------------------ truth
def sobol_centred(nboot=200):
    df = pd.read_parquet(R / "saltelli.parquet"); rows = []
    for tg in TARGETS:
        fA = df[df.block == "A"].sort_values("row")[tg].values; fB = df[df.block == "B"].sort_values("row")[tg].values
        rng = np.random.default_rng(0); N = len(fA)
        for n in NAMES:
            fAB = df[df.block == f"AB_{n}"].sort_values("row")[tg].values
            def est(ix):
                mu = np.mean(np.r_[fA[ix], fB[ix]]); a, b, ab = fA[ix] - mu, fB[ix] - mu, fAB[ix] - mu
                v = np.var(np.r_[a, b])
                return np.mean(b * (ab - a)) / v, 0.5 * np.mean((a - ab) ** 2) / v
            s1, st = est(np.arange(N)); bs = np.array([est(rng.integers(0, N, N)) for _ in range(nboot)])
            rows.append({"target": tg, "feature": n, "S1": s1, "ST": st,
                         "S1_lo": np.percentile(bs[:, 0], 2.5), "S1_hi": np.percentile(bs[:, 0], 97.5),
                         "ST_lo": np.percentile(bs[:, 1], 2.5), "ST_hi": np.percentile(bs[:, 1], 97.5)})
    return pd.DataFrame(rows)

def local_truth_pairs(design):
    """baseline Shapley by permutation sampling; SE from the 16 antithetic-pair means."""
    df = pd.read_parquet(R / f"local_truth_{design}.parquet"); inst = pd.read_parquet(R / f"local_instances_{design}.parquet")
    out = {}
    for tg in TARGETS:
        phi = np.zeros((len(inst), D)); se = np.zeros((len(inst), D))
        for k, g in df.groupby("inst"):
            C = np.zeros((g.perm.max() + 1, D))
            for m, gp in g.groupby("perm"):
                v = gp[tg].values; added = gp["added"].values
                for pos in range(1, len(v)): C[m, added[pos]] = v[pos] - v[pos - 1]
            pairs = (C[0::2] + C[1::2]) / 2
            phi[k] = C.mean(0); se[k] = pairs.std(0, ddof=1) / np.sqrt(len(pairs))
        out[tg] = (phi, se)
    return inst[NAMES].values, out

# ------------------------------------------------------------------ metrics
def local_metrics(ps, pt, se, cols):
    """ps: surrogate phi over observed cols; pt, se: simulator phi over all inputs."""
    pto, seo = pt[:, cols], se[:, cols]
    cos = np.sum(ps * pto, 1) / (np.linalg.norm(ps, axis=1) * np.linalg.norm(pto, axis=1) + 1e-12)
    rel = np.abs(ps - pto).sum(1) / (np.abs(pto).sum(1) + 1e-12)
    crit = (np.abs(pto) > 0.05 * np.abs(pt).sum(1, keepdims=True)) & (np.abs(pto) > 2 * seo)
    per = {}
    for j, c in enumerate(cols):
        k = crit[:, j]
        per[NAMES[c]] = {"n_sign": int(k.sum()), "n_sign_correct": int((np.sign(ps[k, j]) == np.sign(pto[k, j])).sum()),
                         "abs_err": float(np.abs(ps[:, j] - pto[:, j]).mean()), "true_abs": float(np.abs(pto[:, j]).mean())}
    top3 = len(set(np.argsort(-np.abs(ps).mean(0))[:3]) & set(np.argsort(-np.abs(pto).mean(0))[:3]))
    return cos, rel, per, spearmanr(np.abs(ps).mean(0), np.abs(pto).mean(0)).statistic, top3

def slope(grid, y):
    ok = ~np.isnan(y); return np.polyfit(grid[ok], y[ok], 1)[0]

def fit_model(kind, Xtr, y, seed):
    m = X.make_model(kind)
    if kind == "LightGBM": m.set_params(random_state=seed); return m.fit(Xtr, y)
    if kind == "MLP": m.set_params(mlpregressor__random_state=seed)
    return X.YScaled(m).fit(Xtr, y)

# ------------------------------------------------------------------ repeats
def stage_repeats(kinds=("LightGBM", "MLP", "Ridge")):
    sob = pd.read_csv(R / "p1_sobol_truth.csv")
    data = {d: X.load_design(d) for d in ("IND", "COR", "STR")}
    truthL = {d: local_truth_pairs(d) for d in ("IND", "COR", "STR")}
    pdpT = {d: X.pdp_truth(d) for d in ("IND", "COR", "STR")}
    rows, per_rows, dwell_rows, slope_rows = [], [], [], []
    for case, (design, drop) in CASES.items():
        cols = [i for i in range(D) if NAMES[i] not in drop]
        Xd, Yd = data[design]; Xo, Yo = data["COR" if design == "IND" else "IND"]
        Xl, TL = truthL[design]; BG = BGS[design][:, cols]; pt_curves = pdpT[design]
        for seed in SEEDS:
            perm = np.random.default_rng(100 + seed).permutation(len(Xd)); tr, te = perm[:3000], perm[3000:]
            Xtr, Xte = Xd[tr][:, cols], Xd[te][:, cols]
            for tg in TARGETS:
                st = sob[sob.target == tg].set_index("feature").loc[[NAMES[c] for c in cols]]
                phi_t, se_t = TL[tg]
                for kind in kinds:
                    t0 = time.time(); y = Yd[tg].values
                    m = fit_model(kind, Xtr, y[tr], seed); f = m.predict
                    key = {"case": case, "seed": seed, "target": tg, "model": kind}
                    r = {**key, "R2_test": r2_score(y[te], f(Xte)), "R2_transfer": r2_score(Yo[tg].values, f(Xo[:, cols]))}
                    bg = Xtr[:100]; xs = Xte[:500]
                    if kind == "LightGBM":
                        phi_g = shap.TreeExplainer(m, data=bg, feature_perturbation="interventional").shap_values(xs, check_additivity=False)
                        phi_s = shap.TreeExplainer(m, data=BASE[cols][None, :], feature_perturbation="interventional").shap_values(Xl[:, cols], check_additivity=False)
                    else:
                        phi_g = shap.PermutationExplainer(f, shap.maskers.Independent(bg, max_samples=100), seed=seed)(xs, max_evals=2 * len(cols) + 1 + 20 * len(cols), silent=True).values
                        phi_s = shap.ExactExplainer(f, shap.maskers.Independent(BASE[cols][None, :]))(Xl[:, cols]).values
                    g = np.abs(phi_g).mean(0)
                    r["rho_SHAP_vs_SobolST"] = spearmanr(g, st.ST.values).statistic
                    r["top3_SHAP_vs_SobolST"] = len(set(np.argsort(-g)[:3]) & set(np.argsort(-st.ST.values)[:3]))
                    cos, rel, per, rho_l, top3_l = local_metrics(phi_s, phi_t, se_t, cols)
                    r.update({"cosine": cos.mean(), "rel_L1": rel.mean(), "rho_local_global": rho_l, "top3_local_global": top3_l})
                    rows.append(r)
                    for nm, v in per.items(): per_rows.append({**key, "feature": nm, **v})
                    for k_ in range(len(cos)): dwell_rows.append({**key, "dwelling": k_, "cosine": cos[k_], "rel_L1": rel[k_]})
                    for j, c in enumerate(cols):
                        nm = NAMES[c]; tc = pt_curves[pt_curves.feature == nm].sort_values("value"); grid = tc.value.values; truth = tc[tg].values
                        ts = slope(grid, truth)
                        if abs(ts) < 1e-12: continue
                        for mth, cur in (("PDP", X.pdp_curve(f, BG, j, grid)), ("ALE", X.ale_curve(f, Xtr, j, grid)), ("SHAP_dependence", X.shap_dependence_curve(phi_g, xs, j, grid))):
                            slope_rows.append({**key, "feature": nm, "method": mth, "slope_ratio": slope(grid, cur) / ts, "curve_err": X.curve_err(cur, truth),
                                               "true_slope": ts, "true_range": float(np.ptp(truth))})
                    print(f"{case} s{seed} {tg} {kind}: R2={r['R2_test']:.3f} tr={r['R2_transfer']:.3f} cos={r['cosine']:.3f} relL1={r['rel_L1']:.3f} ({time.time()-t0:.0f}s)", flush=True)
            for nm, rr in (("p1_runs", rows), ("p1_per_input", per_rows), ("p1_dwellings", dwell_rows), ("p1_slopes", slope_rows)):
                pd.DataFrame(rr).to_parquet(R / f"{nm}.parquet")

def stage_baselines():
    """no-model explanations: global ranking by |Spearman(x_i, y)|; local phi_i = univariate OLS slope * (x_i - b_i).
    Also metric sensitivity: true phi with envelope attributions zeroed or sign-flipped."""
    sob = pd.read_csv(R / "p1_sobol_truth.csv"); rows = []
    for case, (design, drop) in CASES.items():
        cols = [i for i in range(D) if NAMES[i] not in drop]; Xd, Yd = X.load_design(design); Xl, TL = local_truth_pairs(design)
        for seed in SEEDS:
            perm = np.random.default_rng(100 + seed).permutation(len(Xd)); tr = perm[:3000]
            for tg in TARGETS:
                y = Yd[tg].values[tr]; Xt = Xd[tr][:, cols]; st = sob[sob.target == tg].set_index("feature").loc[[NAMES[c] for c in cols]]
                g = np.array([abs(spearmanr(Xt[:, j], y).statistic) for j in range(len(cols))])
                b = np.array([np.polyfit(Xt[:, j], y, 1)[0] for j in range(len(cols))])
                ps = (Xl[:, cols] - BASE[cols]) * b
                phi_t, se_t = TL[tg]; cos, rel, per, rho_l, top3_l = local_metrics(ps, phi_t, se_t, cols)
                rows.append({"case": case, "seed": seed, "target": tg, "model": "No model (marginal)", "rho_SHAP_vs_SobolST": spearmanr(g, st.ST.values).statistic,
                             "top3_SHAP_vs_SobolST": len(set(np.argsort(-g)[:3]) & set(np.argsort(-st.ST.values)[:3])),
                             "cosine": cos.mean(), "rel_L1": rel.mean(), "rho_local_global": rho_l, "top3_local_global": top3_l})
    sens = []
    for design in ("IND", "STR"):
        Xl, TL = local_truth_pairs(design)
        for tg in ("Tpeak_FF", "DH28_FF"):
            pt = TL[tg][0]; env = [NAMES.index(n) for n in ("U_wall", "U_roof", "ACH")]
            for lab, mod in (("envelope zeroed", 0.0), ("envelope sign-flipped", -1.0)):
                ps = pt.copy(); ps[:, env] *= mod
                cos = np.sum(ps * pt, 1) / (np.linalg.norm(ps, axis=1) * np.linalg.norm(pt, axis=1)); rel = np.abs(ps - pt).sum(1) / np.abs(pt).sum(1)
                sens.append({"design": design, "target": tg, "perturbation": lab, "cosine": cos.mean(), "rel_L1": rel.mean()})
    pd.DataFrame(rows).to_csv(R / "p1_baselines.csv", index=False); pd.DataFrame(sens).to_csv(R / "p1_metric_sensitivity.csv", index=False)
    print(pd.DataFrame(sens).round(3).to_string())

def stage_ovb():
    """slope_j(surrogate without mass) ~= beta_j + beta_mass * gamma_j, gamma from OLS of mass on the observed inputs (STR data)."""
    Xd, _ = X.load_design("STR"); mi = NAMES.index("mass"); obs = [i for i in range(D) if i != mi]
    A = np.c_[np.ones(len(Xd)), Xd[:, obs]]; gam = np.linalg.lstsq(A, Xd[:, mi], rcond=None)[0][1:]
    pdpT = X.pdp_truth("STR"); sl = pd.read_parquet(R / "p1_slopes.parquet"); rows = []
    for tg in ("Tpeak_FF", "DH28_FF", "E_cool_kWh", "rebound_kW"):
        beta = {}
        for n in NAMES:
            tc = pdpT[pdpT.feature == n].sort_values("value"); beta[n] = slope(tc.value.values, tc[tg].values)
        for j, i in enumerate(obs):
            n = NAMES[i]
            if abs(beta[n]) < 1e-9: continue
            pred = 1 + beta["mass"] * gam[j] / beta[n]
            o = sl[(sl.case == "STR-mass") & (sl.target == tg) & (sl.feature == n)]
            for kind, g in o.groupby("model"):
                rows.append({"target": tg, "feature": n, "model": kind, "gamma": gam[j], "beta": beta[n], "beta_mass": beta["mass"],
                             "predicted_ratio": pred, "observed_ratio_median": g.slope_ratio.median(), "observed_ratio_lo": g.slope_ratio.min(), "observed_ratio_hi": g.slope_ratio.max()})
    d = pd.DataFrame(rows); d.to_csv(R / "p1_ovb.csv", index=False)
    print(d[d.feature.isin(["U_wall", "U_roof", "ACH", "setpoint", "WWR"])].round(3).to_string())

if __name__ == "__main__":
    for st in sys.argv[1:]:
        if st == "sobol":
            s = sobol_centred(); s.to_csv(R / "p1_sobol_truth.csv", index=False)
            for tg in TARGETS:
                d = s[s.target == tg].sort_values("ST", ascending=False).head(5)
                print(tg, f"sumS1={s[s.target == tg].S1.sum():.2f} sumST={s[s.target == tg].ST.sum():.2f} |", " ; ".join(f"{r.feature} S1={r.S1:.2f}[{r.S1_lo:.2f},{r.S1_hi:.2f}] ST={r.ST:.2f}[{r.ST_lo:.2f},{r.ST_hi:.2f}]" for r in d.itertuples()))
        else:
            globals()["stage_" + st]()
