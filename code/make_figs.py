"""Figures 1, 2 and 5, and shared plotting settings. Output -> ../figs"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from pathlib import Path
import villa as V

V2 = V.V2; R = V2 / "results"; F = V2 / "figs"; T = V2 / "tables"; F.mkdir(exist_ok=True); T.mkdir(exist_ok=True)
NAMES = V.NAMES
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "savefig.dpi": 750, "figure.dpi": 150})
W2 = 17 / 2.54
LAB = {"U_wall": "Wall U-value", "U_roof": "Roof U-value", "WWR": "Window-to-wall ratio", "SHGC": "SHGC", "ACH": "Infiltration",
       "mass": "Internal mass", "azimuth": "Orientation", "setpoint": "Setpoint", "occupants": "Occupants", "plug": "Plug loads",
       "cut_start": "Cut start", "cut_dur": "Cut duration"}
UNIT = {"U_wall": "W m$^{-2}$ K$^{-1}$", "U_roof": "W m$^{-2}$ K$^{-1}$", "WWR": "–", "SHGC": "–", "ACH": "h$^{-1}$", "mass": "m$^2$ m$^{-2}$",
        "azimuth": "°", "setpoint": "°C", "occupants": "–", "plug": "W m$^{-2}$", "cut_start": "h", "cut_dur": "h"}
OUT = {"Tpeak_FF": "Peak operative temperature", "DH28_FF": "Degree-hours above 28 °C", "t_exceed_FF": "Time to 2 K rise",
       "rebound_kW": "Rebound cooling peak", "E_cool_kWh": "Seasonal cooling energy"}
OUTS = {"Tpeak_FF": "$T_\\mathrm{peak}$", "DH28_FF": "DH$_{28}$", "t_exceed_FF": "$t_{2\\mathrm{K}}$", "rebound_kW": "$Q_\\mathrm{reb}$", "E_cool_kWh": "$E_\\mathrm{cool}$"}
OUNIT = {"Tpeak_FF": "°C", "DH28_FF": "K h", "t_exceed_FF": "min", "rebound_kW": "kW", "E_cool_kWh": "kWh"}
TG = list(OUT)
C = {"LightGBM": "#1b7f6b", "MLP": "#d9822b", "Ridge": "#5b6fb5"}
CM = {"Truth": "#222222", "PDP": "#1b7f6b", "ALE": "#c0392b", "SHAP_dependence": "#5b6fb5", "SHAP_path_dependence": "#9b59b6"}
CASES = ["IND", "COR", "STR all inputs", "STR mass unobserved"]
CASE_LAB = {"IND": "Independent", "COR": "Moderate\ncorrelation", "STR all inputs": "Strong\ncorrelation", "STR mass unobserved": "Strong correlation,\nmass unobserved"}
CASE_SHORT = {"IND": "IND", "COR": "COR", "STR all inputs": "STR", "STR mass unobserved": "STR–mass"}

def save(fig, name):
    fig.savefig(F / f"{name}.png", bbox_inches="tight"); fig.savefig(F / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)

def plab(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")

# ------------------------------------------------------------------ data
sob = pd.read_csv(R / "xai_sobol_truth.csv")
acc1 = pd.read_csv(R / "xai_accuracy.csv"); acc2 = pd.read_csv(R / "xai2_accuracy.csv")
loc1 = pd.read_csv(R / "xai_local.csv"); loc2 = pd.read_csv(R / "xai2_local.csv")
glo1 = pd.read_csv(R / "xai_global.csv"); glo2 = pd.read_csv(R / "xai2_global.csv")
S1 = pd.read_pickle(R / "xai_store.pkl"); S2 = pd.read_pickle(R / "xai2_store.pkl")

def curves(case, tg, kind, nm):
    return S1[(case, tg, kind, nm)] if case in ("IND", "COR") else S2[(case, tg, kind, nm)]
def local(case, tg, kind):
    if case in ("IND", "COR"):
        d = S1[("local", case, tg, kind)]; return d["phi_sur"], d["phi_true"], list(range(len(NAMES)))
    d = S2[("local", case, tg, kind)]; return d["phi_sur"], d["phi_true"], d["cols"]

def slope_ratio(grid, est, truth):
    ok = ~np.isnan(est); ts = np.polyfit(grid, truth, 1)[0]
    return np.polyfit(grid[ok], est[ok], 1)[0] / ts if abs(ts) > 1e-12 else np.nan

# ------------------------------------------------------------------ Fig 1 workflow
def fig1():
    """Workflow figure; drawn in fig1_workflow.py (stacked-phase style)."""
    import fig1_workflow; fig1_workflow.fig1()


# ------------------------------------------------------------------ building schematic (Fig 2a), drawn from the EnergyPlus geometry
def building_schematic(ax, wwr=0.2):
    """Oblique (planometric) drawing of both building cases at the same scale, viewed from the south-east.
    Geometry and window sizes follow villa.geometry(): windows centred, sill 0.9 m, height <= 1.6 m, WWR 0.2 (base case)."""
    import math
    from matplotlib.patches import Polygon as Poly
    k, ang = 0.5, math.radians(35); ca, sa = k * math.cos(ang), k * math.sin(ang)
    edge, win = "#2b2b2b", "#8fbfe6"; H = 3.3
    cases = [  # x0, dv (vertical offset of the drawing), L, W, storeys, wall colours, roof colour, name, description
        (4.0, 18.0, 16.0, 12.0, 2, ("#ead9bb", "#d3bf9c"), "#a39580", "Villa (main study)", "2 storeys, 200 mm block walls,\nconcrete roof slab"),
        (0.0, 0.0, 24.0, 16.0, 1, ("#e3eaf2", "#c7d2de"), "#8795a4", "Lightweight house", "1 storey, cement-board walls,\nsteel-deck roof")]
    for x0, dv, L, W, n, (cs, ce), cr, name, desc in cases:
        P = lambda x, y, z: (x + ca * y, z + sa * y + dv)
        def face(pts, fc, lw=0.6, z=2): ax.add_patch(Poly([P(*q) for q in pts], closed=True, fc=fc, ec=edge, lw=lw, zorder=z))
        for st in range(n):
            z0 = st * H
            face([(x0, 0, z0), (x0 + L, 0, z0), (x0 + L, 0, z0 + H), (x0, 0, z0 + H)], cs)
            face([(x0 + L, 0, z0), (x0 + L, W, z0), (x0 + L, W, z0 + H), (x0 + L, 0, z0 + H)], ce)
            for side, width in (("S", L), ("E", W)):
                area = wwr * width * H; hw = min(1.6, area / (0.9 * width)); ww = area / hw; zb, zt = z0 + 0.9, z0 + 0.9 + hw
                a, b = (width - ww) / 2, (width + ww) / 2
                pts = [(x0 + a, 0, zb), (x0 + b, 0, zb), (x0 + b, 0, zt), (x0 + a, 0, zt)] if side == "S" else \
                      [(x0 + L, a, zb), (x0 + L, b, zb), (x0 + L, b, zt), (x0 + L, a, zt)]
                face(pts, win, lw=0.4, z=3)
        face([(x0, 0, n * H), (x0 + L, 0, n * H), (x0 + L, W, n * H), (x0, W, n * H)], cr)
        ax.text(x0, dv - 1.0, name, ha="left", va="top", fontsize=6.8, fontweight="bold", color=edge)
        ax.text(x0, dv - 3.6, desc, ha="left", va="top", fontsize=6.1, color="#444", linespacing=1.2)
        if n == 2:
            ax.text(x0 - 0.5, dv + H / 2, "GF", ha="right", va="center", fontsize=6.0, color=edge)
            ax.text(x0 - 0.5, dv + 1.5 * H, "FF", ha="right", va="center", fontsize=6.0, color=edge)
    ax.set_aspect("equal"); ax.set_xlim(-3.0, 33.0); ax.set_ylim(-8.8, 29.5); ax.axis("off")


# ------------------------------------------------------------------ Fig 2 model and response
def fig2():
    out, ts = V.simulate(V.BASE, keep=True)
    top = ts[("FF", "Zone Operative Temperature")].reshape(-1, 144); tair = ts[("FF", "Zone Mean Air Temperature")].reshape(-1, 144)
    q = (ts[("FF_IDEAL", "Zone Ideal Loads Supply Air Total Cooling Rate")] + ts[("GF_IDEAL", "Zone Ideal Loads Supply Air Total Cooling Rate")]).reshape(-1, 144) / 1000
    epw = pd.read_csv(V.EPW, skiprows=8, header=None); tout = epw[6].values.reshape(365, 24)[151:243]
    d = int(np.argmax(top[:, 72:96].max(1)))
    fig = plt.figure(figsize=(W2, 2.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1, 1], wspace=0.16, hspace=0.32,
                          left=0.01, right=0.985, top=0.93, bottom=0.14)
    ax = fig.add_subplot(gs[:, 0])
    building_schematic(ax)
    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    h = np.arange(1, 145) / 6
    a1 = fig.add_subplot(gs[0, 1]); a1.axvspan(12, 14, color="#d9822b", alpha=0.15, lw=0)
    a1.plot(h, top[d], color="#c0392b", lw=1.1, label="Operative, FF"); a1.plot(h, tair[d], color="#c0392b", lw=0.8, ls="--", label="Air, FF")
    a1.plot(np.arange(1, 25), tout[d], color="#555", lw=0.9, ls=":", label="Outdoor air")
    a1.set_xlim(8, 20); a1.set_ylabel("Temperature (°C)"); a1.tick_params(labelbottom=False)
    a1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.06, 1.0), ncol=3, fontsize=6.5, handlelength=1.8, columnspacing=1.2, borderaxespad=0.1)
    a1.text(13, a1.get_ylim()[0] + 0.6, "cut", ha="center", va="bottom", fontsize=6.5, color="#a0561c"); plab(a1, "b", -0.12, 1.02)
    a2 = fig.add_subplot(gs[1, 1], sharex=a1); a2.axvspan(12, 14, color="#d9822b", alpha=0.15, lw=0)
    a2.plot(h, q[d], color="#1b7f6b", lw=1.1); a2.set_xlim(8, 20); a2.set_xlabel("Hour of day"); a2.set_ylabel("Cooling load (kW)")
    i = np.argmax(q[d, 84:90]) + 84; a2.annotate("rebound peak", (h[i], q[d, i]), (h[i] + 1.2, q[d, i] - 6), fontsize=6.5, va="center", arrowprops=dict(arrowstyle="-", lw=0.5))
    plab(a2, "c", -0.12, 1.02)
    for a in (a1, a2): a.yaxis.set_label_coords(-0.075, 0.5)
    save(fig, "Figure2")
    return out, d




# ------------------------------------------------------------------ Fig 5 main effects, IND vs STR
def fig5():
    feats = ["setpoint", "mass", "U_roof", "ACH"]; tg = "Tpeak_FF"
    fig, axs = plt.subplots(2, 4, figsize=(W2, 3.35), gridspec_kw={"hspace": 0.62, "wspace": 0.34})
    for r, case in enumerate(("IND", "STR all inputs")):
        for c, nm in enumerate(feats):
            ax = axs[r, c]; d = curves(case, tg, "LightGBM", nm); g = d["grid"].copy()
            xs = g / 60 if nm.startswith("cut") else g
            ax.plot(xs, d["truth"] - d["truth"].mean(), color=CM["Truth"], lw=1.8, label="EnergyPlus (truth)")
            for m, ls in (("PDP", "-"), ("ALE", "--"), ("SHAP_dependence", ":")):
                e = d[m] - np.nanmean(d[m]); ax.plot(xs, e, color=CM[m], lw=1.0, ls=ls, label={"SHAP_dependence": "SHAP dependence"}.get(m, m))
            ax.set_xlabel(f"{LAB[nm]} ({UNIT[nm]})"); ax.set_title(f"{'Independent' if case == 'IND' else 'Strong correlation'}", fontsize=7, loc="left", color="#555")
            if c == 0: ax.set_ylabel("Effect on $T_\\mathrm{peak}$ (K)")
            plab(ax, "abcdefgh"[r * 4 + c], -0.3, 1.13)
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    save(fig, "Figure5")






if __name__ == "__main__":
    import sys
    for w in sys.argv[1:] or ["1", "2", "5"]:
        if w == "1": fig1()
        if w == "2": fig2()
        if w == "5": fig5()
