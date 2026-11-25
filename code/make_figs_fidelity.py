"""Figures 3, 4, 6, 7 and S1."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import make_figs as MF

R, NAMES, LAB, UNIT, OUTS, TG, C = MF.R, MF.NAMES, MF.LAB, MF.UNIT, MF.OUTS, MF.TG, MF.C
W2, save, plab = MF.W2, MF.save, MF.plab
CASES = ["IND", "COR", "STR", "STR-mass"]
CM = {**C, "No model (marginal)": "#9a9a9a"}
MK = {"LightGBM": "o", "MLP": "s", "Ridge": "^", "No model (marginal)": "x"}

def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h

def fig3():
    sob = pd.read_csv(R / "p1_sobol_truth.csv")
    fig, axs = plt.subplots(1, 5, figsize=(W2, 3.0), sharey=True); y = np.arange(len(NAMES))[::-1]
    for k, tg in enumerate(TG):
        s = sob[sob.target == tg].set_index("feature").loc[NAMES]; ax = axs[k]
        ax.barh(y + 0.18, s.ST.clip(lower=0), 0.36, color="#1f3b57", label="Total $S_T$")
        ax.barh(y - 0.18, s.S1.clip(lower=0), 0.36, color="#9fb6cd", label="First-order $S_1$")
        ax.errorbar(s.ST.clip(lower=0), y + 0.18, xerr=[(s.ST - s.ST_lo).clip(lower=0), (s.ST_hi - s.ST).clip(lower=0)], fmt="none", ecolor="#444", elinewidth=0.5, capsize=1)
        ax.errorbar(s.S1.clip(lower=0), y - 0.18, xerr=[(s.S1 - s.S1_lo).clip(lower=0), (s.S1_hi - s.S1).clip(lower=0)], fmt="none", ecolor="#777", elinewidth=0.5, capsize=1)
        ax.set_title(f"({'abcde'[k]}) {OUTS[tg]}", loc="left"); ax.set_xlim(0, max(0.55, float(s.ST_hi.max()) * 1.05)); ax.set_xlabel("Sobol index")
        ax.text(0.97, 0.22, f"Σ$S_1$ = {s.S1.sum():.2f}\nΣ$S_T$ = {s.ST.sum():.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2)
    axs[0].set_yticks(y); axs[0].set_yticklabels([LAB[n] for n in NAMES])
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.04))
    fig.subplots_adjust(wspace=0.12); save(fig, "Figure3")

def runs_all():
    r = pd.read_parquet(R / "p1_runs.parquet"); b = pd.read_csv(R / "p1_baselines.csv")
    return pd.concat([r, b], ignore_index=True)

def fig4():
    d = runs_all(); models = ["LightGBM", "MLP", "Ridge", "No model (marginal)"]
    fig, axs = plt.subplots(2, 5, figsize=(W2, 4.4), sharex=True, gridspec_kw={"hspace": 0.18, "wspace": 0.42})
    for c, tg in enumerate(TG):
        for r_, (col, lab, lim) in enumerate((("rho_local_global", "Rank agreement with\nsimulator mean |φ|", (0.3, 1.02)),
                                              ("rel_L1", "Relative attribution error\n(Σ|φ̂−φ| / Σ|φ|)", None))):
            ax = axs[r_, c]
            for j, m in enumerate(models):
                g = d[(d.target == tg) & (d.model == m)]
                med = [g[g.case == cs][col].median() for cs in CASES]; lo = [g[g.case == cs][col].min() for cs in CASES]; hi = [g[g.case == cs][col].max() for cs in CASES]
                x = np.arange(4) + (j - 1.5) * 0.16
                ax.errorbar(x, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt=MK[m], ms=3.2, color=CM[m], elinewidth=0.6, capsize=0, label=m,
                            mfc=CM[m] if m != "No model (marginal)" else "none")
            if lim: ax.set_ylim(*lim)
            elif tg == "t_exceed_FF": ax.set_yscale("log")
            else: ax.set_ylim(0, max(0.8, ax.get_ylim()[1]))
            ax.set_xticks(range(4)); ax.set_xticklabels(CASES, fontsize=6, rotation=30)
            if c == 0: ax.set_ylabel(lab, fontsize=7)
            if r_ == 0: ax.set_title(f"({'abcde'[c]}) {OUTS[tg]}", loc="left")
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    save(fig, "Figure4")

def fig6():
    s = pd.read_parquet(R / "p1_slopes.parquet"); s = s[s.feature.isin(["U_wall", "U_roof", "ACH"])]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
    for k, tg in enumerate(("Tpeak_FF", "DH28_FF", "E_cool_kWh")):
        ax = axs[k]; x = np.arange(4)
        for j, m in enumerate(("LightGBM", "MLP", "Ridge")):
            g = s[(s.target == tg) & (s.model == m)]
            per_seed = g.groupby(["case", "seed"]).slope_ratio.median()
            med = [per_seed[cs].median() for cs in CASES]; lo = [per_seed[cs].min() for cs in CASES]; hi = [per_seed[cs].max() for cs in CASES]
            ax.bar(x + (j - 1) * 0.26, med, 0.26, color=C[m], label=m)
            ax.errorbar(x + (j - 1) * 0.26, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor="#333", elinewidth=0.6, capsize=1.2)
        ax.axhline(1, color="#222", lw=0.7, ls="--"); ax.set_xticks(x); ax.set_xticklabels(CASES, fontsize=6.5)
        ax.set_title(f"({'abc'[k]}) {OUTS[tg]}", loc="left"); ax.set_ylim(min(0, ax.get_ylim()[0]), 1.4)
    axs[0].set_ylabel("Estimated / true main-effect slope\n(wall U, roof U, infiltration)")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    save(fig, "Figure6")

def fig7():
    d = pd.read_parquet(R / "p1_runs.parquet"); mk = {"IND": "o", "COR": "s", "STR": "^", "STR-mass": "X"}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.8), gridspec_kw={"wspace": 0.3})
    ax = axs[0]
    for (cs, m, tg), g in d.groupby(["case", "model", "target"]):
        ax.errorbar(g.R2_test.median(), g.cosine.median(), xerr=[[g.R2_test.median() - g.R2_test.min()], [g.R2_test.max() - g.R2_test.median()]],
                    yerr=[[g.cosine.median() - g.cosine.min()], [g.cosine.max() - g.cosine.median()]], fmt=mk[cs], ms=3.5, color=C[m],
                    mfc=C[m] if tg != "t_exceed_FF" else "white", elinewidth=0.5, alpha=0.9)
    ax.set_xlabel("Surrogate accuracy, test $R^2$ (median of 5 seeds)"); ax.set_ylabel("Local fidelity, cosine (median of 5 seeds)"); ax.set_ylim(0.6, 1.0)
    hs = [Line2D([], [], marker="o", ls="", color=C[k], label=k) for k in C] + [Line2D([], [], marker=mk[c], ls="", color="#777", label=c) for c in CASES] + \
         [Line2D([], [], marker="o", ls="", color="#777", mfc="white", label="$t_{2\\mathrm{K}}$ (open)")]
    ax.legend(handles=hs, frameon=False, fontsize=5.8, loc="lower right", ncol=2); plab(ax, "a", -0.24)
    l1 = pd.read_csv(R / "xai_local.csv"); l1 = l1[l1.model == "LightGBM"]
    ax = axs[1]
    for (cs, tg), g in l1.groupby(["design", "target"]):
        g = g.sort_values("n_train"); ax.plot(g.n_train, g.cosine, marker="o" if cs == "IND" else "s", ms=3, lw=0.8, color=plt.cm.tab10(TG.index(tg)), ls="-" if cs == "IND" else "--")
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks([250, 500, 1000, 3000]); ax.set_xticklabels(["250", "500", "1,000", "3,000"]); ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("Training runs (LightGBM, one seed)"); ax.set_ylabel("Local fidelity, cosine similarity")
    hs = [Line2D([], [], color=plt.cm.tab10(i), lw=1, label=OUTS[tg]) for i, tg in enumerate(TG)] + [Line2D([], [], color="#777", ls="-", label="IND"), Line2D([], [], color="#777", ls="--", label="COR")]
    ax.legend(handles=hs, frameon=False, fontsize=6, loc="lower right", ncol=2); plab(ax, "b", -0.2)
    save(fig, "FigureS1")

def fig8():
    tg = "Tpeak_FF"; fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0), gridspec_kw={"wspace": 0.32, "hspace": 0.45})
    for k, nm in enumerate(("U_wall", "U_roof")):
        ax = axs[0, k]; t = MF.curves("STR all inputs", tg, "LightGBM", nm); u = MF.curves("STR mass unobserved", tg, "LightGBM", nm)
        ax.plot(t["grid"], t["truth"] - t["truth"].mean(), color="#222", lw=1.8, label="EnergyPlus (truth)")
        ax.plot(t["grid"], t["PDP"] - t["PDP"].mean(), color="#1b7f6b", lw=1.0, label="PDP, all inputs")
        ax.plot(u["grid"], u["PDP"] - u["PDP"].mean(), color="#c0392b", lw=1.0, ls="--", label="PDP, mass omitted")
        ax.plot(u["grid"], u["ALE"] - u["ALE"].mean(), color="#c0392b", lw=0.8, ls=":", label="ALE, mass omitted")
        ax.set_xlabel(f"{LAB[nm]} ({UNIT[nm]})"); ax.set_ylabel("Centred effect on $T_\\mathrm{peak}$ (K)" if k == 0 else ""); plab(ax, "ab"[k], -0.22)
    axs[0, 0].legend(frameon=False, fontsize=6, loc="upper left")
    ov = pd.read_csv(R / "p1_ovb.csv"); ax = axs[1, 0]
    show = ov[ov.feature.isin(["U_wall", "U_roof", "ACH", "WWR", "setpoint", "SHGC"])]
    mk = {"Tpeak_FF": "o", "DH28_FF": "s", "E_cool_kWh": "^", "rebound_kW": "D"}
    for r in show.itertuples():
        ax.errorbar(r.predicted_ratio, r.observed_ratio_median, yerr=[[r.observed_ratio_median - r.observed_ratio_lo], [r.observed_ratio_hi - r.observed_ratio_median]],
                    fmt=mk[r.target], ms=3.5, color=C[r.model], elinewidth=0.5)
    lim = (min(-0.2, show.predicted_ratio.min(), show.observed_ratio_lo.min()) - 0.05, max(1.3, show.predicted_ratio.max(), show.observed_ratio_hi.max()) + 0.05)
    ax.plot(lim, lim, color="#999", lw=0.6, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Slope ratio predicted by omitted-variable bias"); ax.set_ylabel("Observed slope ratio")
    hs = [Line2D([], [], marker=mk[t], ls="", color="#777", label=OUTS[t]) for t in mk] + [Line2D([], [], marker="o", ls="", color=C[k], label=k) for k in C]
    ax.legend(handles=hs, frameon=False, fontsize=5.8, loc="upper left", ncol=2); plab(ax, "c", -0.22)
    sg = pd.read_csv(R / "r2_signs.csv"); ax = axs[1, 1]
    feats = ["U_wall", "U_roof", "ACH"]; x = np.arange(len(feats))
    for j, (cs, m) in enumerate([("STR", "LightGBM"), ("STR-mass", "LightGBM"), ("STR", "MLP"), ("STR-mass", "MLP"), ("STR-mass", "Ridge")]):
        g = sg[(sg.case == cs) & (sg.model == m) & (sg.target == tg)].set_index("feature").loc[feats]
        p, lo, hi = g["correct_sign_%"].values, g["ci_lo_%"].values, g["ci_hi_%"].values
        ax.errorbar(x + (j - 2) * 0.14, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)], fmt="o" if cs == "STR" else "X", ms=3.5, color=C[m],
                    mfc=C[m] if cs == "STR" else "white", elinewidth=0.6, label=f"{m}, {'all inputs' if cs == 'STR' else 'mass omitted'}")
    ax.axhline(50, color="#999", lw=0.6, ls=":"); ax.set_xticks(x); ax.set_xticklabels([LAB[f_] for f_ in feats]); ax.set_ylim(-5, 105)
    ax.set_ylabel("Dwellings with correct sign (%)"); ax.legend(frameon=False, fontsize=5.8, loc="lower right"); plab(ax, "d", -0.22)
    save(fig, "Figure7")

if __name__ == "__main__":
    import sys
    for w in sys.argv[1:] or ["3", "4", "6", "7", "8"]: globals()["fig" + w]()
    print("done")
