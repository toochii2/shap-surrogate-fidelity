"""Figure 1: study workflow, in the stacked-phase style common in Building Simulation framework figures.

Two routes start from the same simulation set-up: the surrogate route (sampling -> surrogates -> explanations), which is
what practitioners do, and the simulator route (ground truth). They meet in the fidelity assessment.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
import make_figs as MF

INK = "#1d2733"; GREY = "#5a6570"
COL = {"sim": ("#1f3b57", "#eef2f7"), "sur": ("#1b7f6b", "#eaf5f2"), "ref": ("#b03a2e", "#fbeeec"), "fid": ("#8a5a14", "#fbf3e6")}
FS_HEAD, FS_TITLE, FS_TXT = 7.8, 7.2, 6.8


def band(ax, x, y, w, h, key, label):
    edge, fill = COL[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fill, ec=edge, lw=0.8, ls=(0, (4, 2))))
    tw = 5.0 + 1.62 * len(label)
    ax.add_patch(FancyBboxPatch((x + 2.0, y + h - 2.6), tw, 5.2, boxstyle="round,pad=0,rounding_size=1.2", fc=edge, ec=edge, lw=0))
    ax.text(x + 2.0 + tw / 2, y + h, label, ha="center", va="center", color="white", fontsize=FS_HEAD, fontweight="bold")
    return x + 2.0 + tw


LS = 3.8   # line spacing inside cards (mm)


def card(ax, x, y, w, h, title, lines, key, dark=False, icon=None):
    edge = COL[key][0]; fg = "white" if dark else INK; tc = "white" if dark else edge
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=edge if dark else "white", ec=edge, lw=0.7))
    dx = 2.4 if icon else 0.0
    ax.text(x + w / 2 + dx, y + h - 2.9, title, ha="center", va="center", fontsize=FS_TITLE, fontweight="bold", color=tc)
    if icon: icon(ax, x + 2.6, y + h - 2.9, edge)
    ax.plot([x + 2, x + w - 2], [y + h - 5.3, y + h - 5.3], color=("white" if dark else edge), lw=0.4, alpha=0.6)
    for k, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 8.5 - LS * k, l, ha="center", va="center", fontsize=FS_TXT, color=fg)


def arrow(ax, p0, p1, color=GREY, lw=1.0, text=None, toff=(1.8, 0), ha="left"):
    ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.26", color=color, lw=lw, mutation_scale=10, shrinkA=0, shrinkB=0))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + toff[0], (p0[1] + p1[1]) / 2 + toff[1], text, fontsize=FS_TXT - 0.4, color=color, ha=ha, va="center", style="italic")


# ---- small title icons (about 4 mm), centred vertically on (x, yc)
def ic_building(ax, x, yc, c):
    ax.add_patch(Rectangle((x, yc - 1.8), 2.6, 3.6, fc="white", ec=c, lw=0.6))
    ax.add_patch(Rectangle((x + 2.6, yc - 1.8), 2.4, 2.0, fc="white", ec=c, lw=0.6))
    for wy in (-1.1, 0.5):
        ax.add_patch(Rectangle((x + 0.6, yc + wy), 0.6, 0.6, fc="#9fc3e6", ec=c, lw=0.3)); ax.add_patch(Rectangle((x + 1.5, yc + wy), 0.6, 0.6, fc="#9fc3e6", ec=c, lw=0.3))
    ax.add_patch(Rectangle((x + 3.4, yc - 1.1), 0.7, 0.6, fc="#9fc3e6", ec=c, lw=0.3))


def ic_sun(ax, x, yc, c):
    import numpy as np
    cx = x + 2.0
    ax.add_patch(matplotlib.patches.Circle((cx, yc), 0.95, fc="#f0b429", ec="#c98a0b", lw=0.4))
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot([cx + 1.35 * np.cos(t), cx + 1.95 * np.cos(t)], [yc + 1.35 * np.sin(t), yc + 1.95 * np.sin(t)], color="#c98a0b", lw=0.5)


def ic_bolt(ax, x, yc, c):
    pts = [(1.6, 2.0), (0.4, -0.1), (1.4, -0.1), (0.9, -2.0), (2.6, 0.5), (1.6, 0.5), (2.3, 2.0)]
    ax.add_patch(Polygon([(x + 0.6 + px, yc + py) for px, py in pts], closed=True, fc="#f0b429", ec="#c98a0b", lw=0.4))


def ic_sliders(ax, x, yc, c):
    for k, (yy, kx) in enumerate(((1.3, 1.0), (0.0, 2.6), (-1.3, 1.7))):
        ax.plot([x + 0.2, x + 3.8], [yc + yy, yc + yy], color=c, lw=0.6)
        ax.add_patch(matplotlib.patches.Circle((x + 0.2 + kx, yc + yy), 0.42, fc="white", ec=c, lw=0.6))


def villa_icon(ax, x, y, s=1.0, c="#1f3b57"):
    ax.add_patch(Rectangle((x, y), 6 * s, 6 * s, fc="white", ec=c, lw=0.7))
    ax.plot([x, x + 6 * s], [y + 3 * s, y + 3 * s], color=c, lw=0.5)
    for wx in (1.0, 3.8):
        for wy in (0.9, 3.9):
            ax.add_patch(Rectangle((x + wx * s, y + wy * s), 1.2 * s, 1.2 * s, fc="#9fc3e6", ec=c, lw=0.35))


def house_icon(ax, x, y, s=1.0, c="#1f3b57"):
    ax.add_patch(Rectangle((x, y), 8 * s, 3.4 * s, fc="white", ec=c, lw=0.7))
    ax.add_patch(Polygon([(x - 0.4 * s, y + 3.4 * s), (x + 8.4 * s, y + 3.4 * s), (x + 8 * s, y + 4.3 * s), (x, y + 4.3 * s)], closed=True, fc=c, ec=c, lw=0.4))
    for wx in (1.2, 5.6):
        ax.add_patch(Rectangle((x + wx * s, y + 1.0 * s), 1.2 * s, 1.2 * s, fc="#9fc3e6", ec=c, lw=0.35))


def fig1():
    W, H = 170.0, 101.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ---- 1 simulation set-up
    Y1, H1 = 74.0, 23.5
    band(ax, 2, Y1, 166, H1, "sim", "1  Simulation set-up (EnergyPlus 26.2)")
    cw, ch = 38.5, 18.4; cy = Y1 + 1.4
    xs = [5 + k * (cw + 2.5) for k in range(4)]
    card(ax, xs[0], cy, cw, ch, "Building cases", ["2-storey heavy villa (main)", "1-storey lightweight house", "384 m² floor area"], "sim", icon=ic_building)
    card(ax, xs[1], cy, cw, ch, "Climates (TMYx)", ["Kuwait, hot-dry (main)", "Karachi, hot-humid", "1 Jun–31 Aug, 10-min step"], "sim", icon=ic_sun)
    card(ax, xs[2], cy, cw, ch, "Daily power cut", ["start 11:00–17:00, 1–4 h", "cooling, lights, plugs off", "windows closed"], "sim", icon=ic_bolt)
    card(ax, xs[3], cy, cw, ch, "12 inputs → 5 outputs", ["envelope, operation, cut", "$T_\\mathrm{peak}$, DH$_{28}$, $t_{2\\mathrm{K}}$", "$Q_\\mathrm{reb}$, $E_\\mathrm{cool}$"], "sim", icon=ic_sliders)

    # ---- surrogate route (left) and simulator route (right)
    Y2, H2 = 33.0, 29.5
    band(ax, 2, Y2, 109, H2, "sur", "Surrogate route (practice)")
    band(ax, 115, Y2, 53, H2, "ref", "Simulator route (truth)")
    gap = 6.0; sw = (103.0 - 2 * gap) / 3; sh = 23.0; sy = Y2 + 2.0
    sx = [5 + k * (sw + gap) for k in range(3)]
    card(ax, sx[0], sy, sw, sh, "2  Sampling designs", ["IND, COR, STR", "STR–mass: no mass", "$\\rho_m$ sweep 0–0.95", "4,000 runs each"], "sur")
    card(ax, sx[1], sy, sw, sh, "3  Surrogates", ["LightGBM, MLP, ridge", "3,000 train, 1,000 test", "3–5 repeats", "test, transfer $R^2$"], "sur")
    card(ax, sx[2], sy, sw, sh, "4  Explanations", ["TreeSHAP and", "permutation SHAP", "PDP, ALE, dependence", "local Shapley $\\hat\\varphi$"], "sur")
    for k in range(2):
        arrow(ax, (sx[k] + sw + 0.7, sy + sh / 2), (sx[k + 1] - 0.7, sy + sh / 2), color=COL["sur"][0])
    rx, rw = 118.0, 47.0
    card(ax, rx, sy, rw, sh, "Reference quantities", ["Sobol $S_i$, $S_{T,i}$ ($N$ = 2,048)", "interventional main effects", "baseline Shapley $\\varphi$ (50 × 32)", "retrofit runs (4 measures)"], "ref")
    arrow(ax, (sx[1] + sw / 2, Y1 - 0.4), (sx[1] + sw / 2, Y2 + H2 + 0.4), color=COL["sim"][0], text="training samples")
    arrow(ax, (163.5, Y1 - 0.4), (163.5, Y2 + H2 + 0.4), color=COL["sim"][0], text="interventions", ha="right", toff=(-1.8, 0))

    # ---- 5 fidelity assessment
    Y3, H3 = 2.0, 22.0

    band(ax, 2, Y3, 166, H3, "fid", "5  Fidelity assessment")
    fy, fh = Y3 + 2.0, 16.5; g1, g2 = 2.0, 7.0; fw = (160.0 - 3 * g1 - g2) / 5
    fx = [5 + k * (fw + g1) for k in range(4)]; fx.append(fx[3] + fw + g2)
    card(ax, fx[0], fy, fw, fh, "Global", ["rank agreement", "SHAP shares vs Sobol"], "fid")
    card(ax, fx[1], fy, fw, fh, "Main effects", ["slope ratio (Eq. 5)", "curve error"], "fid")
    card(ax, fx[2], fy, fw, fh, "Local", ["relative error $\\varepsilon$ (Eq. 6)", "sign accuracy"], "fid")
    card(ax, fx[3], fy, fw, fh, "Omitted driver", ["bias formula (Eq. 8)", "test–transfer gap"], "fid")
    card(ax, fx[4], fy, fw, fh, "Decision and rules", ["retrofit ranking", "practical rules (Table 3)"], "fid", dark=True)
    arrow(ax, (fx[3] + fw + 0.8, fy + fh / 2), (fx[4] - 0.8, fy + fh / 2), color=COL["fid"][0])
    xe = sx[2] + sw / 2
    arrow(ax, (xe, sy - 0.4), (xe, Y3 + H3 + 0.4), color=COL["sur"][0], text="explained effects")
    arrow(ax, (rx + rw / 2, sy - 0.4), (rx + rw / 2, Y3 + H3 + 0.4), color=COL["ref"][0], text="true effects")
    MF.save(fig, "Figure1")


if __name__ == "__main__":
    fig1()
