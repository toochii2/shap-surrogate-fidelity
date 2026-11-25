"""Omitted-variable experiments and retrofit example.

sims     : EnergyPlus runs for the correlation sweep (mass-envelope rho = 0, 0.3, 0.6, 0.95; 0.85 = STR)
           and the retrofit example (50 old dwellings x base + 4 measures)
sweep    : bias against rho, with and without mass; Eq. (8) prediction; transfer R2; sign accuracy
samedir  : infiltration omitted instead of mass (STR)
retrofit : retrofit benefits from EnergyPlus vs surrogates with and without mass
budget   : subsampled simulator references
Outputs -> ../results/p2_*"""
import sys, time, warnings, numpy as np, pandas as pd, shap
from sklearn.metrics import r2_score
import xai as X, run_designs as RD, analysis_repeats as P1

warnings.filterwarnings("ignore")
R = X.R; NAMES = X.NAMES; D = X.D; BASE = X.BASE
LEVELS = [0.0, 0.3, 0.6, 0.85, 0.95]
TGS = ["Tpeak_FF", "DH28_FF", "E_cool_kWh"]
ENV = ["U_wall", "U_roof", "ACH"]

def pairs_for(rho):
    p = dict(RD.CORR_PAIRS_STRONG)
    p[("mass", "U_wall")] = rho; p[("mass", "U_roof")] = rho; p[("mass", "ACH")] = round(0.94 * rho, 3)
    return p

def design(rho):
    if rho == 0.85: return X.load_design("STR")
    df = pd.read_parquet(R / f"p2_train_rho{int(round(rho * 100)):02d}.parquet")
    if "error" in df: df = df[df["error"].isna()]
    return df[NAMES].values, df[X.TARGETS]

# ------------------------------------------------------------------ simulations
def old_dwellings(n=50):
    cand = RD.sample_str(4000, 51); z = (cand - RD.LO) / (RD.HI - RD.LO)
    score = z[:, NAMES.index("U_wall")] + z[:, NAMES.index("U_roof")] + z[:, NAMES.index("ACH")]
    return cand[np.argsort(-score)[:n]]

MEASURES = {"Wall insulation (U 0.30)": ("U_wall", 0.30), "Roof insulation (U 0.20)": ("U_roof", 0.20),
            "Airtightening (0.3 ach)": ("ACH", 0.30), "Solar-control glazing (SHGC 0.25)": ("SHGC", 0.25)}

def stage_sims():
    for rho in LEVELS:
        if rho == 0.85: continue
        X_ = RD.sample_cor(4000, 60 + int(rho * 100), pairs_for(rho))
        RD.run_matrix(X_, f"sweep rho={rho}").to_parquet(R / f"p2_train_rho{int(round(rho * 100)):02d}.parquet")
    old = old_dwellings(); rows = [old]
    for f, v in MEASURES.values():
        Z = old.copy(); Z[:, NAMES.index(f)] = v; rows.append(Z)
    df = RD.run_matrix(np.vstack(rows), "retrofit")
    df["measure"] = np.repeat(["Base"] + list(MEASURES), len(old)); df["dwelling"] = np.tile(np.arange(len(old)), len(MEASURES) + 1)
    df.to_parquet(R / "p2_retrofit.parquet")

# ------------------------------------------------------------------ analyses
def fit(kind, Xtr, y, seed): return P1.fit_model(kind, Xtr, y, seed)

def explain_local(m, kind, Xl, cols):
    if kind == "LightGBM":
        return shap.TreeExplainer(m, data=BASE[cols][None, :], feature_perturbation="interventional").shap_values(Xl[:, cols], check_additivity=False)
    return shap.ExactExplainer(m.predict, shap.maskers.Independent(BASE[cols][None, :]))(Xl[:, cols]).values

def omission_run(Xd, Yd, drop, label, seeds, kinds=("LightGBM", "MLP", "Ridge"), tgs=TGS):
    """train with/without `drop`; slope ratios on STR background vs STR truth; signs on STR reference dwellings; transfer to IND."""
    pdpT = X.pdp_truth("STR"); BG = P1.BGS["STR"]; Xl, TL = P1.local_truth_pairs("STR"); Xi, Yi = X.load_design("IND")
    rows, per = [], []
    for omit in (False, True):
        cols = [i for i in range(D) if not (omit and NAMES[i] == drop)]
        for seed in seeds:
            perm = np.random.default_rng(200 + seed).permutation(len(Xd)); tr, te = perm[:3000], perm[3000:]
            for tg in tgs:
                y = Yd[tg].values; phi_t, se_t = TL[tg]
                for kind in kinds:
                    m = fit(kind, Xd[tr][:, cols], y[tr], seed); f = m.predict
                    key = {"case": label, "omitted": drop if omit else "none", "seed": seed, "target": tg, "model": kind,
                           "R2_test": r2_score(y[te], f(Xd[te][:, cols])), "R2_transfer_IND": r2_score(Yi[tg].values, f(Xi[:, cols]))}
                    ps = explain_local(m, kind, Xl, cols); cos, rel, pin, _, _ = P1.local_metrics(ps, phi_t, se_t, cols)
                    key.update({"rel_L1": rel.mean(), "cosine": cos.mean()})
                    for j, c in enumerate(cols):
                        nm = NAMES[c]; tc = pdpT[pdpT.feature == nm].sort_values("value"); grid = tc.value.values; ts = P1.slope(grid, tc[tg].values)
                        if abs(ts) < 1e-9: continue
                        rows.append({**key, "feature": nm, "slope_ratio_PDP": P1.slope(grid, X.pdp_curve(f, BG[:, cols], j, grid)) / ts,
                                     "slope_ratio_ALE": P1.slope(grid, X.ale_curve(f, Xd[tr][:, cols], j, grid)) / ts,
                                     "n_sign": pin[nm]["n_sign"], "n_sign_correct": pin[nm]["n_sign_correct"]})
                    print(f"{label} omit={omit} s{seed} {tg} {kind}: R2={key['R2_test']:.3f} tr={key['R2_transfer_IND']:.3f} relL1={key['rel_L1']:.3f}", flush=True)
    return pd.DataFrame(rows)

def ovb_pred(Xd, drop, tgs=TGS):
    pdpT = X.pdp_truth("STR"); mi = NAMES.index(drop); obs = [i for i in range(D) if i != mi]
    gam = np.linalg.lstsq(np.c_[np.ones(len(Xd)), Xd[:, obs]], Xd[:, mi], rcond=None)[0][1:]; out = []
    for tg in tgs:
        beta = {n: P1.slope(pdpT[pdpT.feature == n].sort_values("value").value.values, pdpT[pdpT.feature == n].sort_values("value")[tg].values) for n in NAMES}
        for j, i in enumerate(obs):
            if abs(beta[NAMES[i]]) > 1e-9:
                out.append({"target": tg, "feature": NAMES[i], "gamma": gam[j], "predicted_ratio": 1 + beta[drop] * gam[j] / beta[NAMES[i]]})
    return pd.DataFrame(out)

def stage_sweep():
    res, pred = [], []
    for rho in LEVELS:
        Xd, Yd = design(rho)
        d = omission_run(Xd, Yd, "mass", f"rho={rho}", seeds=range(3)); d["rho"] = rho; res.append(d)
        p = ovb_pred(Xd, "mass"); p["rho"] = rho; p["empirical_corr_mass_Uwall"] = np.corrcoef(Xd[:, NAMES.index("mass")], Xd[:, 0])[0, 1]; pred.append(p)
        pd.concat(res).to_parquet(R / "p2_sweep.parquet"); pd.concat(pred).to_csv(R / "p2_sweep_ovb.csv", index=False)

def stage_samedir():
    Xd, Yd = X.load_design("STR")
    d = omission_run(Xd, Yd, "ACH", "STR", seeds=range(3), tgs=["Tpeak_FF", "DH28_FF", "E_cool_kWh", "rebound_kW"]); d.to_parquet(R / "p2_samedir.parquet")
    ovb_pred(Xd, "ACH", ["Tpeak_FF", "DH28_FF", "E_cool_kWh", "rebound_kW"]).to_csv(R / "p2_samedir_ovb.csv", index=False)

def stage_retrofit():
    rf = pd.read_parquet(R / "p2_retrofit.parquet"); Xd, Yd = X.load_design("STR"); rows = []
    base = rf[rf.measure == "Base"].sort_values("dwelling")
    for tg in ("Tpeak_FF", "DH28_FF", "E_cool_kWh"):
        for ms in MEASURES:
            after = rf[rf.measure == ms].sort_values("dwelling")
            rows.append({"target": tg, "measure": ms, "source": "EnergyPlus", "seed": -1, "effect_mean": (after[tg].values - base[tg].values).mean()})
        for omit in (False, True):
            cols = [i for i in range(D) if not (omit and NAMES[i] == "mass")]
            for seed in range(3):
                perm = np.random.default_rng(300 + seed).permutation(len(Xd)); tr = perm[:3000]
                for kind in ("LightGBM", "MLP", "Ridge"):
                    m = fit(kind, Xd[tr][:, cols], Yd[tg].values[tr], seed); b = m.predict(base[NAMES].values[:, cols])
                    for ms in MEASURES:
                        a = m.predict(rf[rf.measure == ms].sort_values("dwelling")[NAMES].values[:, cols])
                        rows.append({"target": tg, "measure": ms, "source": f"{kind}, {'mass omitted' if omit else 'all inputs'}", "seed": seed, "effect_mean": (a - b).mean()})
    d = pd.DataFrame(rows); d.to_csv(R / "p2_retrofit_effects.csv", index=False)
    print(d.groupby(["target", "measure", "source"]).effect_mean.median().unstack("source").round(3).to_string())

def stage_budget():
    """subsample the simulator references: 20 dwellings x 8 permutations (local) and 30 background cases (main effects)."""
    out = []
    df = pd.read_parquet(R / "local_truth_STR.parquet"); inst = pd.read_parquet(R / "local_instances_STR.parquet")
    keep_inst = np.arange(20); keep_perm = np.arange(8)
    sub = df[df.inst.isin(keep_inst) & df.perm.isin(keep_perm)]
    full = P1.local_truth_pairs("STR")[1]
    for tg in ("Tpeak_FF", "DH28_FF"):
        phi = np.zeros((20, D))
        for k, g in sub.groupby("inst"):
            C = np.zeros((8, D))
            for m, gp in g.groupby("perm"):
                v = gp[tg].values; a = gp["added"].values
                for pos in range(1, len(v)): C[m, a[pos]] = v[pos] - v[pos - 1]
            phi[k] = C.mean(0)
        pf = full[tg][0][:20]
        out.append({"target": tg, "check": "local reference, 20 dwellings x 8 permutations (2,080 runs)",
                    "rel_L1_vs_full_reference": float((np.abs(phi - pf).sum(1) / np.abs(pf).sum(1)).mean()),
                    "wall_U_sign_agreement_%": float(100 * np.mean(np.sign(phi[:, 0]) == np.sign(pf[:, 0])))})
    p = pd.read_parquet(R / "pdp_truth_STR.parquet")
    for tg in ("Tpeak_FF", "DH28_FF"):
        for f in ENV + ["mass"]:
            g = p[p.feature == f]; full_s = P1.slope(*g.groupby("value")[tg].mean().reset_index().values.T)
            red = g[g.bg < 30].groupby("value")[tg].mean().reset_index(); red_s = P1.slope(red.value.values, red[tg].values)
            out.append({"target": tg, "check": f"main effect of {f}, 30 background cases (3,240 runs for 12 inputs)", "slope_ratio_reduced_vs_full": red_s / full_s})
    d = pd.DataFrame(out); d.to_csv(R / "p2_budget.csv", index=False); print(d.round(3).to_string())

if __name__ == "__main__":
    for st in sys.argv[1:]: globals()["stage_" + st]()
