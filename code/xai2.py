"""Stress test: strongly dependent housing stock (STR), with and without an unobserved driver (thermal mass).

Same truth objects as xai.py (Sobol, interventional PDP with STR background, simulator baseline Shapley for
50 STR instances). Surrogates trained on STR with all 12 inputs, or with 'mass' removed from the inputs
(as when thermal mass is not recorded in a stock survey).
Outputs -> ../results/xai2_*.csv, ../results/xai2_store.pkl
"""
import time, warnings, numpy as np, pandas as pd, shap
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import xai as X, run_designs as RD

warnings.filterwarnings("ignore")
R = X.R; NAMES = X.NAMES; D = X.D; TARGETS = X.TARGETS; BASE = X.BASE
ENV = ["U_wall", "U_roof", "ACH", "mass"]

def run(kinds=("LightGBM", "MLP", "Ridge"), n=3000):
    sob = pd.read_csv(R / "xai_sobol_truth.csv")
    Xall, Y = X.load_design("STR"); Xte_all = Xall[3000:]
    Xind, Yind = X.load_design("IND")
    pdpT = X.pdp_truth("STR"); Xl_all, truthL = X.local_truth("STR"); BG_all = RD.sample_str(150, 33)
    rows_acc, rows_glob, rows_main, rows_loc, rows_feat, store = [], [], [], [], [], {}
    for case, drop in (("STR all inputs", []), ("STR mass unobserved", ["mass"])):
        cols = [i for i in range(D) if NAMES[i] not in drop]; names = [NAMES[i] for i in cols]
        Xtr, Xte, Xl, BG, Xo = Xall[:n][:, cols], Xte_all[:, cols], Xl_all[:, cols], BG_all[:, cols], Xind[:, cols]
        for tg in TARGETS:
            st = sob[sob.target == tg].set_index("feature")
            for kind in kinds:
                t0 = time.time(); y = Y[tg].values[:n]
                m = X.make_model(kind).fit(Xtr, y) if kind == "LightGBM" else X.YScaled(X.make_model(kind)).fit(Xtr, y)
                f = m.predict
                acc = {"case": case, "target": tg, "model": kind, "R2_test": r2_score(Y[tg].values[3000:], f(Xte)),
                       "R2_independent_design": r2_score(Yind[tg], f(Xo))}
                rows_acc.append(acc); key = {k: acc[k] for k in ("case", "target", "model")}
                bg = Xtr[:200]; xs = Xte[:500]
                if kind == "LightGBM":
                    phi_int = shap.TreeExplainer(m, data=bg, feature_perturbation="interventional").shap_values(xs, check_additivity=False)
                    phi_path = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent").shap_values(xs, check_additivity=False)
                else:
                    phi_int = shap.PermutationExplainer(f, shap.maskers.Independent(bg, max_samples=200), seed=0)(xs, max_evals=2 * len(cols) + 1 + 20 * len(cols), silent=True).values
                    phi_path = None
                # local truth and surrogate baseline Shapley
                phi_t, se_t = truthL[tg]
                if kind == "LightGBM":
                    phi_s = shap.TreeExplainer(m, data=BASE[cols][None, :], feature_perturbation="interventional").shap_values(Xl, check_additivity=False)
                else:
                    phi_s = shap.ExactExplainer(f, shap.maskers.Independent(BASE[cols][None, :]))(Xl).values
                rows_loc.append({**key, **X.local_scores(phi_s, phi_t[:, cols], se_t[:, cols])})
                for j, nm in enumerate(names):
                    big = np.abs(phi_t[:, cols[j]]) > 2 * se_t[:, cols[j]]
                    rows_feat.append({**key, "feature": nm, "true_mean_abs_phi": float(np.abs(phi_t[:, cols[j]]).mean()),
                                      "sur_mean_abs_phi": float(np.abs(phi_s[:, j]).mean()),
                                      "sign_agree_%": float(100 * np.mean(np.sign(phi_s[big, j]) == np.sign(phi_t[big, cols[j]]))) if big.any() else np.nan,
                                      "corr_sur_true": float(np.corrcoef(phi_s[:, j], phi_t[:, cols[j]])[0, 1]),
                                      "corr_error_with_true_mass_phi": float(np.corrcoef(phi_s[:, j] - phi_t[:, cols[j]], phi_t[:, NAMES.index("mass")])[0, 1]) if drop else np.nan,
                                      "SobolST": float(st.loc[nm, "ST"]),
                                      "SHAP_int_share": float(np.abs(phi_int[:, j]).mean() / np.abs(phi_int).mean(0).sum()),
                                      "SHAP_path_share": float(np.abs(phi_path[:, j]).mean() / np.abs(phi_path).mean(0).sum()) if phi_path is not None else np.nan,
                                      "true_local_share": float(np.abs(phi_t[:, cols[j]]).mean() / np.abs(phi_t).mean(0).sum())})
                gl_true = np.abs(phi_t).mean(0)[cols]
                for en, ph in (("SHAP_interventional", phi_int), ("SHAP_path_dependent", phi_path)):
                    if ph is None: continue
                    v = np.abs(ph).mean(0)
                    rows_glob.append({**key, "explainer": en, "spearman_vs_SobolST": spearmanr(v, st.loc[names, "ST"].values).statistic,
                                      "spearman_vs_true_local": spearmanr(v, gl_true).statistic})
                # main effects of the envelope inputs vs interventional PDP truth (STR background)
                for j, nm in enumerate(names):
                    tcur = pdpT[pdpT.feature == nm].sort_values("value"); grid = tcur.value.values; truth = tcur[tg].values
                    curves = {"PDP": X.pdp_curve(f, BG, j, grid), "ALE": X.ale_curve(f, Xtr, j, grid), "SHAP_dependence": X.shap_dependence_curve(phi_int, xs, j, grid)}
                    if phi_path is not None: curves["SHAP_path_dependence"] = X.shap_dependence_curve(phi_path, xs, j, grid)
                    for en, c in curves.items():
                        tr_slope = np.polyfit(grid, truth, 1)[0]; es_slope = np.polyfit(grid[~np.isnan(c)], c[~np.isnan(c)], 1)[0]
                        rows_main.append({**key, "feature": nm, "method": en, "rel_RMSE": X.curve_err(c, truth), "truth_range": float(np.ptp(truth)),
                                          "slope_sign_agree": bool(np.sign(tr_slope) == np.sign(es_slope)), "slope_ratio": float(es_slope / tr_slope) if abs(tr_slope) > 1e-12 else np.nan})
                    store[(case, tg, kind, nm)] = {"grid": grid, "truth": truth, **curves}
                store[("local", case, tg, kind)] = {"phi_sur": phi_s, "phi_true": phi_t, "cols": cols, "phi_int": phi_int, "phi_path": phi_path, "xs": xs}
                print(f"{case} | {tg} | {kind}: R2={acc['R2_test']:.3f} | local cos={rows_loc[-1]['cosine']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    for nm, rows in (("accuracy", rows_acc), ("global", rows_glob), ("main_effects", rows_main), ("local", rows_loc), ("features", rows_feat)):
        pd.DataFrame(rows).to_csv(R / f"xai2_{nm}.csv", index=False)
    pd.to_pickle(store, R / "xai2_store.pkl")

if __name__ == "__main__":
    run()
