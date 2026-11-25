"""Supplementary checks.

excess  : Sobol indices of the excess peak temperature T_peak - T_set (existing Saltelli runs)
signs   : per-dwelling sign accuracy with a bootstrap over dwellings
budget  : disjoint, repeated subsampling of the simulator references (50 draws)
extins  : wall-insulation retrofit with external insulation for the 50 old villas (200 runs)
numbers : transfer gaps, curve errors, per-input errors, paired standard errors, retrofit ranks
Outputs -> ../results/r2_*"""
import sys, warnings, numpy as np, pandas as pd, shap
from multiprocessing import Pool
import xai as X, analysis_repeats as P1, analysis_omission as P2, villa as V, run_designs as RD

warnings.filterwarnings("ignore")
R = X.R; NAMES = X.NAMES; D = X.D; BASE = X.BASE
WI = NAMES.index("U_wall")

def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def stage_excess():
    df = pd.read_parquet(R / "saltelli.parquet").copy(); df["Tex"] = df.Tpeak_FF - df.setpoint
    rows = []
    fA = df[df.block == "A"].sort_values("row").Tex.values; fB = df[df.block == "B"].sort_values("row").Tex.values; N = len(fA)
    rng = np.random.default_rng(0)
    for n in NAMES:
        fAB = df[df.block == f"AB_{n}"].sort_values("row").Tex.values
        def est(ix):
            mu = np.mean(np.r_[fA[ix], fB[ix]]); a, b, ab = fA[ix] - mu, fB[ix] - mu, fAB[ix] - mu; v = np.var(np.r_[a, b])
            return np.mean(b * (ab - a)) / v, 0.5 * np.mean((a - ab) ** 2) / v
        s1, st = est(np.arange(N)); bs = np.array([est(rng.integers(0, N, N)) for _ in range(200)])
        rows.append({"feature": n, "S1": s1, "ST": st, "ST_lo": np.percentile(bs[:, 1], 2.5), "ST_hi": np.percentile(bs[:, 1], 97.5)})
    d = pd.DataFrame(rows).sort_values("ST", ascending=False); d.to_csv(R / "r2_excess_sobol.csv", index=False)
    print(d.round(3).head(8).to_string(), "\nsum S1", d.S1.sum().round(3), "sum ST", d.ST.sum().round(3))

def stage_signs():
    """refit STR and STR-mass (5 splits as in P1) and sweep levels (3 splits as in P2); keep per-dwelling attributions."""
    Xl, TL = P1.local_truth_pairs("STR"); rows, store = [], {}
    for case, drop, design, seeds, splitbase in [("STR", None, 0.85, range(5), 100), ("STR-mass", "mass", 0.85, range(5), 100)] + \
            [(f"rho={r}", "mass", r, range(3), 200) for r in P2.LEVELS]:
        Xd, Yd = P2.design(design); cols = [i for i in range(D) if NAMES[i] != drop]
        tgs = ("Tpeak_FF", "DH28_FF") if not case.startswith("rho") else ("Tpeak_FF",)
        for tg in tgs:
            phi_t, se_t = TL[tg]
            crit = (np.abs(phi_t) > 0.05 * np.abs(phi_t).sum(1, keepdims=True)) & (np.abs(phi_t) > 2 * se_t)
            for kind in ("LightGBM", "MLP", "Ridge"):
                ok = []
                for seed in seeds:
                    perm = np.random.default_rng(splitbase + seed).permutation(len(Xd)); tr = perm[:3000]
                    m = P1.fit_model(kind, Xd[tr][:, cols], Yd[tg].values[tr], seed)
                    ps = P2.explain_local(m, kind, Xl, cols); store[(case, tg, kind, seed)] = ps
                    ok.append(np.sign(ps) == np.sign(phi_t[:, cols]))
                ok = np.mean(ok, 0)                                   # per-dwelling share of seeds with correct sign
                for j, c in enumerate(cols):
                    k = crit[:, c]
                    if k.sum() == 0: continue
                    mu, lo, hi = boot_ci(ok[k, j])
                    rows.append({"case": case, "target": tg, "model": kind, "feature": NAMES[c], "n_dwellings": int(k.sum()),
                                 "correct_sign_%": 100 * mu, "ci_lo_%": 100 * lo, "ci_hi_%": 100 * hi})
                print(case, tg, kind, "done", flush=True)
    pd.DataFrame(rows).to_csv(R / "r2_signs.csv", index=False); pd.to_pickle(store, R / "r2_local_store.pkl")

def stage_budget(ndraw=50):
    df = pd.read_parquet(R / "local_truth_STR.parquet"); rng = np.random.default_rng(7); out = []
    store = pd.read_pickle(R / "r2_local_store.pkl")
    # contributions per (inst, perm, feature) for both targets
    C = {}
    for tg in ("Tpeak_FF", "DH28_FF"):
        arr = np.zeros((50, 32, D))
        for (k, m), gp in df.groupby(["inst", "perm"]):
            v = gp[tg].values; a = gp["added"].values
            for pos in range(1, len(v)): arr[k, m, a[pos]] = v[pos] - v[pos - 1]
        C[tg] = arr
    for draw in range(ndraw):
        inst = rng.choice(50, 20, replace=False); pairs = rng.choice(16, 4, replace=False)
        perm_sub = np.r_[2 * pairs, 2 * pairs + 1]; perm_rest = np.setdiff1d(np.arange(32), perm_sub)
        for tg in ("Tpeak_FF", "DH28_FF"):
            red = C[tg][inst][:, perm_sub].mean(1); rest = C[tg][inst][:, perm_rest].mean(1)
            rel = (np.abs(red - rest).sum(1) / np.abs(rest).sum(1)).mean()
            row = {"draw": draw, "target": tg, "rel_diff_reduced_vs_disjoint": rel, "wallU_sign_agree_%": 100 * np.mean(np.sign(red[:, WI]) == np.sign(rest[:, WI]))}
            # verdict: wall-U sign accuracy of STR-mass surrogates judged with the reduced vs the full reference
            full = C[tg].mean(1)
            for kind in ("LightGBM", "MLP", "Ridge"):
                ps = store[("STR-mass", tg, kind, 0)][inst][:, 0]
                row[f"{kind}_wallU_acc_reduced_%"] = 100 * np.mean(np.sign(ps) == np.sign(red[:, WI]))
                row[f"{kind}_wallU_acc_full_%"] = 100 * np.mean(np.sign(ps) == np.sign(full[inst][:, WI]))
            out.append(row)
    loc = pd.DataFrame(out)
    p = pd.read_parquet(R / "pdp_truth_STR.parquet"); sl = pd.read_parquet(R / "p1_slopes.parquet"); pout = []
    for draw in range(ndraw):
        bg = rng.choice(150, 30, replace=False); rest = np.setdiff1d(np.arange(150), bg)
        for tg in ("Tpeak_FF", "DH28_FF"):
            for f in ("U_wall", "U_roof", "ACH"):
                g = p[p.feature == f]
                a = g[g.bg.isin(bg)].groupby("value")[tg].mean(); b = g[g.bg.isin(rest)].groupby("value")[tg].mean(); full = g.groupby("value")[tg].mean()
                sa, sb, sf = (P1.slope(x.index.values, x.values) for x in (a, b, full))
                obs = sl[(sl.case == "STR-mass") & (sl.target == tg) & (sl.feature == f)].groupby("model").slope_ratio.median()
                pout.append({"draw": draw, "target": tg, "feature": f, "slope_reduced_over_disjoint": sa / sb,
                             **{f"{m}_ratio_with_reduced_ref": float(obs[m] * sf / sa) for m in obs.index}, **{f"{m}_ratio_with_full_ref": float(obs[m]) for m in obs.index}})
    pdp = pd.DataFrame(pout); loc.to_csv(R / "r2_budget_local.csv", index=False); pdp.to_csv(R / "r2_budget_pdp.csv", index=False)
    print(loc.groupby("target")[["rel_diff_reduced_vs_disjoint", "wallU_sign_agree_%"]].describe(percentiles=[.05, .5, .95]).round(3).T.to_string())
    for kind in ("LightGBM", "MLP", "Ridge"):
        d = (loc[f"{kind}_wallU_acc_reduced_%"] - loc[f"{kind}_wallU_acc_full_%"]).abs()
        print(kind, "abs diff in wall-U sign accuracy (pp): median", d.median(), "95th", np.percentile(d, 95))
    print(pdp.groupby(["target", "feature"]).slope_reduced_over_disjoint.describe(percentiles=[.05, .95]).round(3).to_string())

def _sim(p): return V.simulate(p)

def stage_extins():
    old = P2.old_dwellings(); jobs = []
    for ext in (0, 1):
        for meas in ("Base", "Wall"):
            for x in old:
                p = dict(zip(NAMES, map(float, x))); p["ext_ins"] = ext
                if meas == "Wall": p["U_wall"] = 0.30
                jobs.append((ext, meas, p))
    with Pool(14) as pool: res = pool.map(_sim, [j[2] for j in jobs])
    d = pd.DataFrame([{**{"ext": e, "measure": m}, **r} for (e, m, _), r in zip(jobs, res)])
    rows = []
    for e in (0, 1):
        b = d[(d.ext == e) & (d.measure == "Base")].reset_index(drop=True); w = d[(d.ext == e) & (d.measure == "Wall")].reset_index(drop=True)
        rows.append({"insulation": "external" if e else "internal (as in the study)", **{tg: (w[tg] - b[tg]).mean() for tg in ("Tpeak_FF", "DH28_FF", "E_cool_kWh")}})
    o = pd.DataFrame(rows); o.to_csv(R / "r2_extins.csv", index=False); print(o.round(3).to_string())

def stage_numbers():
    r1 = pd.read_parquet(R / "p1_runs.parquet"); sw = pd.read_parquet(R / "p2_sweep.parquet"); sd = pd.read_parquet(R / "p2_samedir.parquet")
    r1["gap"] = r1.R2_test - r1.R2_transfer
    print("P1 gaps (test - transfer), median over seeds:\n", r1.groupby(["target", "case", "model"]).gap.median().unstack("case").round(3).to_string())
    for nm, d in (("sweep", sw), ("samedir", sd)):
        d = d.drop_duplicates(["case", "omitted", "seed", "target", "model"]).copy(); d["gap"] = d.R2_test - d.R2_transfer_IND
        grp = ["rho", "omitted"] if nm == "sweep" else ["omitted"]
        print(nm, "gaps:\n", d.groupby(["target"] + grp + ["model"]).gap.median().unstack("model").round(3).to_string())
    sl = pd.read_parquet(R / "p1_slopes.parquet"); sob = pd.read_csv(R / "p1_sobol_truth.csv")
    c = sl.merge(sob[["target", "feature", "ST"]], on=["target", "feature"]); c = c[(c.case == "IND") & (c.ST > 0.05) & (c.target != "t_exceed_FF")]
    m = c.groupby(["target", "model", "method"]).curve_err.median(); print("IND curve error, median over inputs (S_T>0.05) and repeats, by output x model x method:", m.min().round(3), m.max().round(3))
    pi = pd.read_parquet(R / "p1_per_input.parquet"); imp = pi[pi.case.isin(["IND", "COR", "STR"]) & pi.target.isin(["Tpeak_FF", "DH28_FF", "E_cool_kWh", "rebound_kW"]) & pi.model.isin(["LightGBM", "MLP"])]
    imp = imp.merge(sob[["target", "feature", "ST"]], on=["target", "feature"]); imp = imp[imp.ST > 0.05]
    imp["norm_err"] = imp.abs_err / imp.true_abs
    print("per-input normalised error (inputs S_T>0.05, LightGBM/MLP, IND/COR/STR): median", imp.norm_err.median().round(3), "IQR", imp.norm_err.quantile([.25, .75]).round(3).tolist(), "max", imp.norm_err.max().round(3))
    print(imp.groupby("feature").norm_err.median().round(3).to_dict())
    sw["sr"] = sw[["slope_ratio_PDP", "slope_ratio_ALE"]].mean(axis=1)
    k = sw[(sw.omitted == "none") & sw.model.isin(["MLP", "Ridge"]) & sw.feature.isin(["U_wall", "U_roof", "ACH"]) & (sw.target == "Tpeak_FF")]
    print("mass recorded, MLP/Ridge envelope slope ratios Tpeak (per input, medians over seeds):", k.groupby(["rho", "model", "feature"]).sr.median().agg(["min", "max"]).round(2).tolist())
    for d in ("IND", "COR", "STR"):
        Xl, TL = P1.local_truth_pairs(d)
        for tg in ("Tpeak_FF", "DH28_FF", "E_cool_kWh", "rebound_kW", "t_exceed_FF"):
            phi, se = TL[tg]; big = np.abs(phi) > 0.05 * np.abs(phi).sum(1, keepdims=True)
            print("paired SE", d, tg, f"median SE/|phi| = {np.median(se[big] / np.abs(phi[big])):.4f}")
    rf = pd.read_csv(R / "p2_retrofit_effects.csv")
    g = rf[rf.source != "EnergyPlus"].copy()
    g["rank"] = g.groupby(["target", "source", "seed"]).effect_mean.rank()
    print(g.groupby(["target", "measure", "source"]).agg(med=("effect_mean", "median"), lo=("effect_mean", "min"), hi=("effect_mean", "max"), rank_mode=("rank", lambda x: x.mode().iloc[0]), rank_n=("rank", lambda x: (x == x.mode().iloc[0]).sum())).round(3).to_string())

if __name__ == "__main__":
    for st in sys.argv[1:]: globals()["stage_" + st]()
