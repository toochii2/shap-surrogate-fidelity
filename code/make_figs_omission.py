"""Figure 8 (dose-response of omitted-variable bias; same-direction omission) and Table 4 (retrofit example)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import make_figs as MF

R, LAB, OUTS, C, W2, save, plab = MF.R, MF.LAB, MF.OUTS, MF.C, MF.W2, MF.save, MF.plab
T = MF.T

def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0), min(c + h, 1)

def fig9(tg="Tpeak_FF"):
    sw = pd.read_parquet(R / "p2_sweep.parquet"); pr = pd.read_csv(R / "p2_sweep_ovb.csv")
    sw["slope_ratio"] = sw[["slope_ratio_PDP", "slope_ratio_ALE"]].mean(1)
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.9), gridspec_kw={"wspace": 0.36, "hspace": 0.5})
    rhos = sorted(sw.rho.unique())
    for k, f in enumerate(["U_wall", "U_roof", "ACH"]):
        ax = axs[0, k]
        p = pr[(pr.target == tg) & (pr.feature == f)].sort_values("rho"); ax.plot(p.rho, p.predicted_ratio, color="#222", lw=1.6, label="OVB prediction (Eq. 8)")
        for j, m in enumerate(("LightGBM", "MLP", "Ridge")):
            for om, ls, mk in (("mass", "-", "o"), ("none", ":", "s")):
                g = sw[(sw.target == tg) & (sw.feature == f) & (sw.model == m) & (sw.omitted == om)].groupby("rho").slope_ratio
                med, lo, hi = g.median(), g.min(), g.max()
                ax.errorbar(med.index + (j - 1) * 0.012, med.values, yerr=[med - lo, hi - med], fmt=mk, ls=ls, ms=3, lw=0.8, color=C[m],
                            mfc=C[m] if om == "mass" else "white", elinewidth=0.5)
        ax.axhline(1, color="#999", lw=0.5, ls="--"); ax.axhline(0, color="#ccc", lw=0.5)
        ax.set_xlabel("Mass–envelope correlation $\\rho_m$"); ax.set_title(f"({'abc'[k]}) {LAB[f]}", loc="left")
        if k == 0: ax.set_ylabel(f"Slope ratio, effect on {OUTS[tg]}")
    hs = [Line2D([], [], color="#222", lw=1.6, label="OVB prediction (Eq. 8)")] + [Line2D([], [], marker="o", color=C[m], lw=0.8, label=f"{m}, mass omitted") for m in C] + \
         [Line2D([], [], marker="s", color="#777", mfc="white", ls=":", lw=0.8, label="mass recorded")]
    fig.legend(handles=hs, loc="upper center", ncol=5, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, 1.01))
    ax = axs[1, 0]; sg = pd.read_csv(R / "r2_signs.csv")
    for m in ("LightGBM", "MLP", "Ridge"):
        g = sg[sg.case.str.startswith("rho") & (sg.model == m) & (sg.feature == "U_wall") & (sg.target == tg)].copy()
        g["rho"] = g.case.str.replace("rho=", "").astype(float); g = g.sort_values("rho")
        p, lo, hi = g["correct_sign_%"].values, g["ci_lo_%"].values, g["ci_hi_%"].values
        ax.errorbar(g.rho, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)], marker="o", ms=3, lw=0.8, color=C[m], label=m, elinewidth=0.5)
    ax.axhline(50, color="#999", lw=0.5, ls=":"); ax.set_ylim(-5, 105); ax.set_xlabel("Mass–envelope correlation $\\rho_m$")
    ax.set_ylabel("Correct sign of wall-U attribution (%)"); ax.set_title("(d) Wall-U sign, mass omitted", loc="left")
    ax = axs[1, 1]
    for m in ("LightGBM", "MLP", "Ridge"):
        g = sw[(sw.target == tg) & (sw.model == m) & (sw.omitted == "mass")].drop_duplicates(["rho", "seed"]).groupby("rho")
        ax.plot(g.R2_test.median().index, g.R2_test.median().values, marker="o", ls="-", ms=3, lw=0.8, color=C[m])
        ax.plot(g.R2_transfer_IND.median().index, g.R2_transfer_IND.median().values, marker="o", ls="--", ms=3, lw=0.8, color=C[m], mfc="white")
    ax.set_xlabel("Mass–envelope correlation $\\rho_m$"); ax.set_ylabel("$R^2$ of surrogate without mass"); ax.set_title("(e) Test vs transfer accuracy", loc="left")
    ax.legend(handles=[Line2D([], [], color="#555", marker="o", lw=0.8, label="test (same stock)"), Line2D([], [], color="#555", marker="o", mfc="white", ls="--", lw=0.8, label="transfer (independent)")],
              frameon=False, fontsize=6, loc="upper left")
    ax = axs[1, 2]; sd = pd.read_parquet(R / "p2_samedir.parquet"); sp = pd.read_csv(R / "p2_samedir_ovb.csv")
    sd["slope_ratio"] = sd[["slope_ratio_PDP", "slope_ratio_ALE"]].mean(1); sd = sd[sd.omitted == "ACH"]
    mk = {"Tpeak_FF": "o", "DH28_FF": "s", "E_cool_kWh": "^", "rebound_kW": "D"}
    for (t, f, m), g in sd[sd.feature.isin(["U_wall", "U_roof", "mass", "WWR", "setpoint"])].groupby(["target", "feature", "model"]):
        pv = sp[(sp.target == t) & (sp.feature == f)].predicted_ratio
        if pv.empty: continue
        ax.errorbar(pv.iloc[0], g.slope_ratio.median(), yerr=[[g.slope_ratio.median() - g.slope_ratio.min()], [g.slope_ratio.max() - g.slope_ratio.median()]], fmt=mk[t], ms=3.2, color=C[m], elinewidth=0.5)
    lim = ax.get_xlim(); lo_, hi_ = min(lim[0], ax.get_ylim()[0]), max(lim[1], ax.get_ylim()[1]); ax.plot([lo_, hi_], [lo_, hi_], color="#999", lw=0.6, ls="--"); ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
    ax.set_xlabel("Predicted slope ratio (Eq. 8)"); ax.set_ylabel("Observed slope ratio"); ax.set_title("(f) Infiltration omitted (STR)", loc="left")
    ax.legend(handles=[Line2D([], [], marker=mk[t], ls="", color="#777", label=OUTS[t]) for t in mk], frameon=False, fontsize=6, loc="upper left")
    save(fig, "Figure8")

def table4():
    d = pd.read_csv(R / "p2_retrofit_effects.csv"); rows = []
    srcs = ["EnergyPlus", "LightGBM, all inputs", "MLP, all inputs", "Ridge, all inputs", "LightGBM, mass omitted", "MLP, mass omitted", "Ridge, mass omitted"]
    unit = {"Tpeak_FF": ("Δ*T*~peak~ (K)", 1), "DH28_FF": ("ΔDH~28~ (K h)", 1), "E_cool_kWh": ("Δ*E*~cool~ (kWh m⁻²)", 1 / 384.0)}
    order = ["Wall insulation (U 0.30)", "Roof insulation (U 0.20)", "Airtightening (0.3 ach)", "Solar-control glazing (SHGC 0.25)"]
    for tg, (lab, sc) in unit.items():
        med = {s_: {ms: d[(d.target == tg) & (d.measure == ms) & (d.source == s_)].effect_mean.median() * sc for ms in order} for s_ in srcs}
        rank = {s_: {ms: r + 1 for r, ms in enumerate(sorted(order, key=lambda m: med[s_][m]))} for s_ in srcs}
        for k, ms in enumerate(order):
            r = {"Output": lab if k == 0 else "", "Measure": ms.split(" (")[0]}
            for s_ in srcs:
                e = d[(d.target == tg) & (d.measure == ms) & (d.source == s_)].effect_mean * sc
                flip = "†" if (e.min() < 0 < e.max()) else ""
                r[s_] = f"{med[s_][ms]:+.2f}{flip} ({rank[s_][ms]})"
            rows.append(r)
    out = pd.DataFrame(rows); out.to_csv(T / "table4_retrofit.csv", index=False); return out

if __name__ == "__main__":
    import sys
    if "9" in sys.argv[1:]: fig9()
    if "t4" in sys.argv[1:]: print(table4().to_string())
