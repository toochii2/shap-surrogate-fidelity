"""Per-input envelope fidelity and detection of omitted inputs (existing results only).

envelope : per-input error and slope ratio of wall U, roof U and infiltration (ESM Table S2)
detect   : envelope slope error vs test-transfer gap and physics sign check, every omission case (ESM Table S3)
gamma    : Eq. (8) with gamma from survey-size subsamples (n = 50, 100, 200) and beta_mass +/- 30%
ridge    : relative attribution error of ridge vs flexible surrogates under correlation
misc     : orientation sign accuracy, Eq. (8) error by model, infiltration ratio at rho = 0
Outputs -> ../results/r3_*"""
import sys, numpy as np, pandas as pd
import xai as X

R = X.R; NAMES = X.NAMES; D = X.D
ENV = ["U_wall", "U_roof", "ACH"]; HEAT = ["Tpeak_FF", "DH28_FF"]


def stage_envelope():
    pi = pd.read_parquet(R / "p1_per_input.parquet"); sl = pd.read_parquet(R / "p1_slopes.parquet")
    pi = pi[pi.feature.isin(ENV) & pi.target.isin(HEAT) & pi.case.isin(["IND", "COR", "STR"])].copy()
    pi["norm_err"] = pi.abs_err / pi.true_abs
    e = pi.groupby(["target", "feature", "case", "model"]).norm_err.median().rename("err_median")
    s = sl[sl.feature.isin(ENV) & sl.target.isin(HEAT) & sl.case.isin(["IND", "COR", "STR"]) & sl.method.isin(["PDP", "ALE"])]
    s = s.groupby(["target", "feature", "case", "model", "seed"]).slope_ratio.mean().groupby(["target", "feature", "case", "model"]).median().rename("slope_ratio")
    out = pd.concat([e, s], axis=1).reset_index(); out.to_csv(R / "r3_envelope.csv", index=False)
    print(out.round(2).to_string())
    for c in ("IND", "COR", "STR"):
        k = out[out.case == c]
        print(c, "err range", k.err_median.min().round(3), k.err_median.max().round(3), "| LGBM/MLP", k[k.model != "Ridge"].err_median.min().round(3), k[k.model != "Ridge"].err_median.max().round(3))


def stage_detect():
    sw = pd.read_parquet(R / "p2_sweep.parquet"); sd = pd.read_parquet(R / "p2_samedir.parquet")
    rows = []
    for d, label in ((sw, "mass"), (sd, "ACH")):
        d = d.copy(); d["sr"] = d[["slope_ratio_PDP", "slope_ratio_ALE"]].mean(1); d["gap"] = d.R2_test - d.R2_transfer_IND
        if "rho" not in d: d["rho"] = 0.85
        feats = ENV if label == "mass" else ["U_wall", "U_roof", "mass"]
        for (rho, om, tg, m), g in d.groupby(["rho", "omitted", "target", "model"]):
            g = g[g.feature.isin(feats)]
            if g.empty: continue
            med = g.groupby("feature").sr.median()
            rows.append({"omitted_input": label, "rho_m": rho, "recorded": om == "none", "target": tg, "model": m,
                         "max_bias": float((med - 1).abs().max()), "gap": float(g.drop_duplicates("seed").gap.median()),
                         "sign_flag": bool((med < 0).any())})
    o = pd.DataFrame(rows); o.to_csv(R / "r3_detect.csv", index=False)
    thr = o[o.recorded].gap.max(); print("max gap with every input recorded (false-alarm ceiling):", round(thr, 3))
    print(o[o.recorded].sort_values("gap").tail(3).round(3).to_string())
    om = o[~o.recorded].copy(); om["gap_flag"] = om.gap > thr
    for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 9)):
        k = om[(om.max_bias >= lo) & (om.max_bias < hi)]
        print(f"bias {lo}-{hi}: n={len(k)} gap-flagged={k.gap_flag.sum()} sign-flagged={k.sign_flag.sum()} either={(k.gap_flag | k.sign_flag).sum()}")
    tab = om.groupby(["omitted_input", "rho_m", "target"]).agg(bias_lo=("max_bias", "min"), bias_hi=("max_bias", "max"), gap_lo=("gap", "min"), gap_hi=("gap", "max"),
                                                              gap_flags=("gap_flag", "sum"), sign_flags=("sign_flag", "sum")).reset_index()
    tab.to_csv(R / "r3_detect_table.csv", index=False); print(tab.round(2).to_string())


def stage_gamma(ndraw=500):
    Xd, _ = X.load_design("STR"); mi = NAMES.index("mass"); obs = [i for i in range(D) if i != mi]
    ov = pd.read_csv(R / "p1_ovb.csv").drop_duplicates(["target", "feature"]).set_index(["target", "feature"])
    rng = np.random.default_rng(7); rows = []
    for n in (50, 100, 200, len(Xd)):
        for k in range(ndraw if n < len(Xd) else 1):
            ix = rng.choice(len(Xd), n, replace=False) if n < len(Xd) else np.arange(len(Xd))
            g = np.linalg.lstsq(np.c_[np.ones(n), Xd[ix][:, obs]], Xd[ix, mi], rcond=None)[0][1:]
            fac = rng.uniform(0.7, 1.3) if n < len(Xd) else 1.0
            for tg in HEAT:
                for f in ENV:
                    if (tg, f) not in ov.index: continue
                    b, bm = ov.loc[(tg, f), "beta"], ov.loc[(tg, f), "beta_mass"]
                    rows.append({"n": n, "draw": k, "target": tg, "feature": f, "pred": 1 + fac * bm * g[obs.index(NAMES.index(f))] / b})
    d = pd.DataFrame(rows)
    obsr = pd.read_csv(R / "p1_ovb.csv").groupby(["target", "feature"]).observed_ratio_median.agg(["min", "max"])
    q = d.groupby(["target", "feature", "n"]).pred.quantile([0.05, 0.5, 0.95]).unstack().reset_index()
    q = q.merge(obsr.reset_index(), on=["target", "feature"]); q.to_csv(R / "r3_gamma.csv", index=False); print(q.round(2).to_string())


def stage_ridge():
    r = pd.read_parquet(R / "p1_runs.parquet"); b = pd.read_csv(R / "p1_baselines.csv")
    k = r[r.case.isin(["COR", "STR"]) & (r.target != "t_exceed_FF")]
    print(k.groupby(["case", "model"]).rel_L1.agg(["min", "median", "max"]).round(3).to_string())
    print(b.head(3).to_string())


def stage_misc():
    pi = pd.read_parquet(R / "p1_per_input.parquet")
    a = pi[(pi.feature == "azimuth") & pi.case.isin(["IND", "COR", "STR"])].groupby(["model"]).agg(n=("n_sign", "sum"), c=("n_sign_correct", "sum"))
    print("azimuth sign accuracy:", (a.c / a.n).round(3).to_dict())
    ov = pd.read_csv(R / "p1_ovb.csv"); ov = ov[ov.feature.isin(ENV)]
    ov["ae"] = (ov.predicted_ratio - ov.observed_ratio_median).abs(); ov["ae0"] = (1 - ov.observed_ratio_median).abs()
    print("P1 OVB MAE by model:", ov.groupby("model").ae.mean().round(3).to_dict(), "null MAE (ratio=1):", ov.ae0.mean().round(3))
    sw = pd.read_parquet(R / "p2_sweep.parquet"); sw["sr"] = sw[["slope_ratio_PDP", "slope_ratio_ALE"]].mean(1)
    k = sw[(sw.rho == 0) & (sw.omitted == "mass") & sw.target.isin(HEAT)]
    print("rho=0, mass omitted, envelope ratios:", k.groupby("feature").sr.median().loc[ENV].round(2).to_dict())


if __name__ == "__main__":
    for st in sys.argv[1:]: globals()["stage_" + st]()
