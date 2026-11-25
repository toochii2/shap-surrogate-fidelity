"""Side-by-side key results for the main study (Kuwait villa), the second climate (Karachi villa) and the second building type
(Kuwait lightweight single-storey house) -> ../results/replication_compare.csv (ESM Table S4)."""
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
ENV = ["U_wall", "U_roof", "ACH"]

def rng(s, f="{:.2f}"):
    s = pd.Series(s).dropna(); lo, hi = s.min(), s.max()
    return f.format(lo) if abs(hi - lo) < 0.005 else f"{f.format(lo)} to {f.format(hi)}"

def fmt(x): return x.replace("-0.00", "0.00").replace("-", "−")

def climate(name):
    if name == "Kuwait":
        run = pd.read_parquet(R / "p1_runs.parquet"); run = run[run.case.isin(["STR", "STR-mass"])].rename(columns={"R2_transfer": "R2_tr"})
        sl = pd.read_parquet(R / "p1_slopes.parquet"); sl = sl[sl.method.isin(["PDP", "ALE"]) & sl.case.isin(["STR", "STR-mass"])]
        sl = sl.groupby(["case", "seed", "target", "model", "feature"]).slope_ratio.mean().reset_index()
        ov = pd.read_csv(R / "p1_ovb.csv"); sg = pd.read_csv(R / "r2_signs.csv"); rf = pd.read_csv(R / "p2_retrofit_effects.csv")
        ind = pd.read_parquet(R / "train_IND.parquet")
    else:
        K = R / {"Karachi": "karachi", "Light": "light"}[name]
        run = pd.read_parquet(K / "khi_runs.parquet").rename(columns={"R2_transfer": "R2_tr"}); sl = pd.read_parquet(K / "khi_slopes.parquet")
        ov = pd.read_csv(K / "khi_ovb.csv"); sg = pd.read_csv(K / "khi_signs.csv"); rf = pd.read_csv(K / "p2_retrofit_effects.csv")
        ind = pd.read_parquet(K / "train_IND.parquet")
    T = "Tpeak_FF"; run = run[run.target == T].copy(); run["gap"] = run.R2_test - run.R2_tr
    med = lambda d: d.groupby(["model", "feature"]).slope_ratio.median()
    a = med(sl[(sl.case == "STR") & (sl.target == T) & sl.feature.isin(ENV)]); m = med(sl[(sl.case == "STR-mass") & (sl.target == T) & sl.feature.isin(ENV)])
    o = ov[(ov.target == T)].drop_duplicates("feature").set_index("feature")
    e = ov[ov.feature.isin(ENV) & ov.target.isin(["Tpeak_FF", "DH28_FF", "E_cool_kWh"])]
    w = sg[(sg.case == "STR-mass") & (sg.target == T) & (sg.feature == "U_wall")].set_index("model")["correct_sign_%"]
    t = rf[rf.target == T].copy(); ep = t[t.source == "EnergyPlus"].set_index("measure").effect_mean
    t = t[t.source.str.contains("mass omitted")].copy(); t["rank"] = t.groupby(["source", "seed"]).effect_mean.rank()
    wall = "Wall insulation (U 0.30)"; last = int((t[t.measure == wall]["rank"] == 4).sum())
    ep_rank = int(ep.rank()[wall])
    return {
        "Summer daily temperature range; mean relative humidity": {"Kuwait": "14.3 K; 26%", "Karachi": "5.2 K; 70%", "Light": "14.3 K; 26%"}[name],
        "Mean (SD) *T*~peak~ in the independent sample (°C)": f"{ind.Tpeak_FF.mean():.1f} ({ind.Tpeak_FF.std():.1f})",
        "Simulator slope of *T*~peak~: mass (K per m² m⁻²); wall U (K per W m⁻² K⁻¹)": f"{o.loc['U_wall', 'beta_mass']:.2f}; {o.loc['U_wall', 'beta']:.2f}",
        "Test *R*², all inputs / mass omitted": f"{rng(run[run.case == 'STR'].R2_test)} / {rng(run[run.case == 'STR-mass'].R2_test)}",
        "Envelope slope ratio, all inputs: LightGBM / MLP and ridge": f"{rng(a.loc['LightGBM'])} / {rng(pd.concat([a.loc['MLP'], a.loc['Ridge']]))}",
        "Slope ratio without mass, wall U (Eq. 8 prediction)": f"{rng(m.xs('U_wall', level='feature'))} ({o.loc['U_wall', 'predicted_ratio']:.2f})",
        "Slope ratio without mass, roof U (Eq. 8 prediction)": f"{rng(m.xs('U_roof', level='feature'))} ({o.loc['U_roof', 'predicted_ratio']:.2f})",
        "Slope ratio without mass, infiltration (Eq. 8 prediction)": f"{rng(m.xs('ACH', level='feature'))} ({o.loc['ACH', 'predicted_ratio']:.2f})",
        "Eq. (8), three outputs × three inputs × three surrogates: *r*; MAE": f"{np.corrcoef(e.predicted_ratio, e.observed_ratio_median)[0, 1]:.2f}; {(e.predicted_ratio - e.observed_ratio_median).abs().mean():.2f}",
        "Wall-U sign correct without mass: LightGBM / MLP / ridge (%)": f"{w['LightGBM']:.0f} / {w['MLP']:.0f} / {w['Ridge']:.0f}",
        "Test–transfer gap, mass omitted / all inputs": f"{rng(run[run.case == 'STR-mass'].gap)} / {rng(run[run.case == 'STR'].gap)}",
        "Retrofit: EnergyPlus wall-insulation benefit (K; rank of 4)": f"{ep[wall]:.2f} ({ep_rank})",
        "Retrofit: surrogates without mass ranking wall insulation last": f"{last} of 9 repeats",
    }

if __name__ == "__main__":
    k, p = climate("Kuwait"), climate("Karachi")
    cols = {"Quantity": list(k), "Villa, Kuwait": [fmt(v) for v in k.values()], "Villa, Karachi": [fmt(v) for v in p.values()]}
    if (R / "light" / "khi_compare_ready").exists() or (R / "light" / "p2_retrofit_effects.csv").exists():
        h = climate("Light"); cols["Lightweight house, Kuwait"] = [fmt(v) for v in h.values()]
    d = pd.DataFrame(cols)
    d.to_csv(R / "replication_compare.csv", index=False); print(d.to_string())
