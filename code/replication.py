"""Replication of the core experiment for a second climate (Karachi, hot-humid) or a second building type
(single-storey lightweight house, Kuwait climate).

Run with EP_CLIMATE=KHI EP_RESULTS_SUB=karachi so that every simulation uses the Karachi TMYx file and every
result is written to ../results/karachi/. The input samples are identical to those of the Kuwait study
(same seeds), so the two climates are compared on the same dwellings.

sims     : STR training pool (4,000), STR interventional PDP truth (16,200), STR baseline-Shapley truth (20,800),
           IND transfer sample (4,000) and the retrofit example (250)
analysis : STR and STR-mass surrogates (LightGBM, MLP, ridge; 3 repeats): accuracy, transfer gap, local fidelity,
           per-input signs, main-effect slope ratios (mean of PDP and ALE), omitted-variable prediction (Eq. 8)
signs    : wall-U sign accuracy per dwelling with a bootstrap over dwellings
retrofit : retrofit benefits from EnergyPlus vs surrogates with and without mass (reuses analysis_omission.stage_retrofit)
summary  : side-by-side numbers for Kuwait and Karachi -> ../results/khi_summary.csv
"""
import os, sys, time, warnings, numpy as np, pandas as pd, shap
CASE_OK = (os.environ.get("EP_CLIMATE") == "KHI" and os.environ.get("EP_RESULTS_SUB") == "karachi") or \
          (os.environ.get("EP_BTYPE") == "LIGHT" and os.environ.get("EP_RESULTS_SUB") == "light" and os.environ.get("EP_CLIMATE", "KWT") == "KWT")
assert CASE_OK, "set EP_CLIMATE=KHI EP_RESULTS_SUB=karachi (second climate) or EP_BTYPE=LIGHT EP_RESULTS_SUB=light (second building type)"
from sklearn.metrics import r2_score
import xai as X, run_designs as RD, analysis_repeats as P1, analysis_omission as P2

warnings.filterwarnings("ignore")
R = X.R; NAMES = X.NAMES; D = X.D; BASE = X.BASE
TGS = ["Tpeak_FF", "DH28_FF", "E_cool_kWh", "rebound_kW"]
ENV = ["U_wall", "U_roof", "ACH"]
SEEDS = range(3)


def stage_sims():
    RD.stage_strong()
    RD.run_matrix(RD.sample_ind(4000, 21), "train IND").to_parquet(R / "train_IND.parquet")
    old = P2.old_dwellings(); rows = [old]
    for f, v in P2.MEASURES.values():
        Z = old.copy(); Z[:, NAMES.index(f)] = v; rows.append(Z)
    df = RD.run_matrix(np.vstack(rows), "retrofit")
    df["measure"] = np.repeat(["Base"] + list(P2.MEASURES), len(old)); df["dwelling"] = np.tile(np.arange(len(old)), len(P2.MEASURES) + 1)
    df.to_parquet(R / "p2_retrofit.parquet")


def stage_analysis():
    Xd, Yd = X.load_design("STR"); Xo, Yo = X.load_design("IND")
    Xl, TL = P1.local_truth_pairs("STR"); pdpT = X.pdp_truth("STR"); BGfull = P1.BGS["STR"]
    rows, per_rows, slope_rows, store = [], [], [], {}
    for case, drop in (("STR", []), ("STR-mass", ["mass"])):
        cols = [i for i in range(D) if NAMES[i] not in drop]
        for seed in SEEDS:
            perm = np.random.default_rng(100 + seed).permutation(len(Xd)); tr, te = perm[:3000], perm[3000:]
            Xtr, Xte = Xd[tr][:, cols], Xd[te][:, cols]
            for tg in TGS:
                phi_t, se_t = TL[tg]; y = Yd[tg].values
                for kind in ("LightGBM", "MLP", "Ridge"):
                    t0 = time.time(); m = P1.fit_model(kind, Xtr, y[tr], seed); f = m.predict
                    key = {"case": case, "seed": seed, "target": tg, "model": kind}
                    r = {**key, "R2_test": r2_score(y[te], f(Xte)), "R2_transfer": r2_score(Yo[tg].values, f(Xo[:, cols]))}
                    if kind == "LightGBM":
                        phi_s = shap.TreeExplainer(m, data=BASE[cols][None, :], feature_perturbation="interventional").shap_values(Xl[:, cols], check_additivity=False)
                    else:
                        phi_s = shap.ExactExplainer(f, shap.maskers.Independent(BASE[cols][None, :]))(Xl[:, cols]).values
                    store[(case, seed, tg, kind)] = phi_s
                    cos, rel, per, rho_l, top3 = P1.local_metrics(phi_s, phi_t, se_t, cols)
                    r.update({"cosine": cos.mean(), "rel_L1": rel.mean(), "rho_local_global": rho_l}); rows.append(r)
                    for nm, v in per.items(): per_rows.append({**key, "feature": nm, **v})
                    for j, c in enumerate(cols):
                        nm = NAMES[c]; tc = pdpT[pdpT.feature == nm].sort_values("value"); grid = tc.value.values; truth = tc[tg].values
                        ts = P1.slope(grid, truth)
                        if abs(ts) < 1e-12: continue
                        sr = [P1.slope(grid, X.pdp_curve(f, BGfull[:, cols], j, grid)) / ts, P1.slope(grid, X.ale_curve(f, Xtr, j, grid)) / ts]
                        slope_rows.append({**key, "feature": nm, "slope_ratio": float(np.mean(sr)), "true_slope": ts})
                    print(f"{case} s{seed} {tg} {kind}: R2={r['R2_test']:.3f} tr={r['R2_transfer']:.3f} relL1={r['rel_L1']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_parquet(R / "khi_runs.parquet"); pd.DataFrame(per_rows).to_parquet(R / "khi_per_input.parquet")
    sl = pd.DataFrame(slope_rows); sl.to_parquet(R / "khi_slopes.parquet"); pd.to_pickle(store, R / "khi_local_store.pkl")
    # omitted-variable prediction: gamma from the STR inputs (identical to Kuwait), beta from the Karachi simulator
    mi = NAMES.index("mass"); obs = [i for i in range(D) if i != mi]
    gam = np.linalg.lstsq(np.c_[np.ones(len(Xd)), Xd[:, obs]], Xd[:, mi], rcond=None)[0][1:]; ov = []
    for tg in TGS:
        beta = {n: P1.slope(pdpT[pdpT.feature == n].sort_values("value").value.values, pdpT[pdpT.feature == n].sort_values("value")[tg].values) for n in NAMES}
        for j, i in enumerate(obs):
            n = NAMES[i]
            if abs(beta[n]) < 1e-9: continue
            o = sl[(sl.case == "STR-mass") & (sl.target == tg) & (sl.feature == n)]
            for kind, g in o.groupby("model"):
                ov.append({"target": tg, "feature": n, "model": kind, "gamma": gam[j], "beta": beta[n], "beta_mass": beta["mass"],
                           "predicted_ratio": 1 + beta["mass"] * gam[j] / beta[n], "observed_ratio_median": g.slope_ratio.median(),
                           "observed_ratio_lo": g.slope_ratio.min(), "observed_ratio_hi": g.slope_ratio.max()})
    pd.DataFrame(ov).to_csv(R / "khi_ovb.csv", index=False)


def stage_signs():
    store = pd.read_pickle(R / "khi_local_store.pkl"); Xl, TL = P1.local_truth_pairs("STR"); out = []
    for case in ("STR", "STR-mass"):
        cols = [i for i in range(D) if not (case == "STR-mass" and NAMES[i] == "mass")]
        for tg in ("Tpeak_FF", "DH28_FF"):
            pt, se = TL[tg]
            for kind in ("LightGBM", "MLP", "Ridge"):
                for f in ENV + ([] if case == "STR-mass" else ["mass"]):
                    c = NAMES.index(f); j = cols.index(c)
                    crit = (np.abs(pt[:, c]) > 0.05 * np.abs(pt).sum(1)) & (np.abs(pt[:, c]) > 2 * se[:, c])
                    if crit.sum() == 0: continue
                    share = np.mean([np.sign(store[(case, s, tg, kind)][crit, j]) == np.sign(pt[crit, c]) for s in SEEDS], axis=0)
                    rng = np.random.default_rng(0); bs = [share[rng.integers(0, len(share), len(share))].mean() for _ in range(2000)]
                    out.append({"case": case, "target": tg, "model": kind, "feature": f, "n_dwellings": int(crit.sum()),
                                "correct_sign_%": 100 * share.mean(), "ci_lo_%": 100 * np.percentile(bs, 2.5), "ci_hi_%": 100 * np.percentile(bs, 97.5)})
    d = pd.DataFrame(out); d.to_csv(R / "khi_signs.csv", index=False); print(d.round(1).to_string())


def stage_retrofit():
    P2.stage_retrofit()


def stage_summary():
    K = X.V2 / "results" if False else RD.V.V2 / "results"
    run = pd.read_parquet(R / "khi_runs.parquet"); sl = pd.read_parquet(R / "khi_slopes.parquet"); ov = pd.read_csv(R / "khi_ovb.csv")
    run["gap"] = run.R2_test - run.R2_transfer
    print("Karachi accuracy and gap (median over repeats):\n", run.groupby(["target", "case", "model"])[["R2_test", "R2_transfer", "gap", "rel_L1"]].median().round(3).to_string())
    print("Karachi envelope slope ratios:\n", sl[sl.feature.isin(ENV + ["mass"]) & sl.target.isin(["Tpeak_FF", "DH28_FF", "E_cool_kWh"])].groupby(["target", "case", "model", "feature"]).slope_ratio.median().unstack("feature").round(2).to_string())
    e = ov[ov.feature.isin(ENV)]
    print("Karachi OVB: r =", round(np.corrcoef(e.predicted_ratio, e.observed_ratio_median)[0, 1], 3), "MAE =", round((e.predicted_ratio - e.observed_ratio_median).abs().mean(), 3),
          "| by model:", e.assign(ae=(e.predicted_ratio - e.observed_ratio_median).abs()).groupby("model").ae.mean().round(3).to_dict())
    print(e[e.target == "Tpeak_FF"].drop_duplicates("feature")[["feature", "gamma", "beta", "beta_mass", "predicted_ratio"]].round(3).to_string())
    rf = pd.read_csv(R / "p2_retrofit_effects.csv")
    print("Karachi retrofit (median over repeats):\n", rf.groupby(["target", "measure", "source"]).effect_mean.median().unstack("source").round(2).to_string())


if __name__ == "__main__":
    for st in sys.argv[1:]: globals()["stage_" + st]()
