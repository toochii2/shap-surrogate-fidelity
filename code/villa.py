"""Parametric two-storey Kuwaiti villa in EnergyPlus 26.2 with daily scheduled power cuts.

One simulation = 1 June - 31 August, 10-min time step, Kuwait International Airport TMYx (2011-2025).
Every day the power is cut from `cut_start` for `cut_dur` minutes: the ideal cooling system, lights and
plug loads are off; occupants stay. Outputs are event-level indicators averaged over the 92 cut days.
"""
import os, sqlite3, shutil, subprocess, tempfile, numpy as np
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
_ep = os.environ.get("ENERGYPLUS_EXE") or next((str(p) for p in sorted((V2 / "tools").glob("EnergyPlus-26.2*/energyplus"))), "energyplus")
EP = Path(_ep)                                     # set ENERGYPLUS_EXE, put EnergyPlus 26.2 in ../tools, or have it on PATH
CLIMATE = os.environ.get("EP_CLIMATE", "KWT")      # KWT (main study) or KHI (Karachi, second climate)
CLIM = {
    "KWT": dict(epw="KWT_FA_Kuwait.Intl.AP.405820_TMYx.2011-2025.epw",
                site="Site:Location, Kuwait Intl AP, 29.227, 47.969, 3.0, 62.8;",
                ground="18, 19, 21, 24, 27, 29, 31, 32, 31, 28, 24, 20;   ! hot-arid shallow ground (assumption)"),
    "KHI": dict(epw="PAK_SD_Karachi-Jinnah.Intl.AP.417800_TMYx.2011-2025.epw",
                site="Site:Location, Karachi Jinnah Intl AP, 24.907, 67.161, 5.0, 30.5;",
                ground="24, 22, 21, 22, 24, 27, 30, 32, 33, 32, 30, 27;   ! 0.5 m ground temperatures of the weather file"),
}[CLIMATE]
EPW = V2 / "weather" / CLIM["epw"]
BTYPE = os.environ.get("EP_BTYPE", "VILLA")         # VILLA (main study: two-storey heavy villa) or LIGHT (single-storey lightweight house)

# name: (low, high, unit, description)
PARAMS = {
    "U_wall":    (0.30, 1.75, "W/m2K", "external wall U-value"),
    "U_roof":    (0.20, 1.50, "W/m2K", "roof U-value"),
    "WWR":       (0.10, 0.40, "-",     "window-to-wall ratio (all facades)"),
    "SHGC":      (0.25, 0.75, "-",     "glazing solar heat gain coefficient"),
    "ACH":       (0.20, 1.50, "1/h",   "infiltration air changes per hour"),
    "mass":      (0.50, 3.00, "m2/m2", "internal concrete mass area per floor area"),
    "azimuth":   (0.0, 360.0, "deg",   "building orientation (north axis)"),
    "setpoint":  (20.0, 26.0, "degC",  "cooling setpoint"),
    "occupants": (3.0, 10.0, "-",      "number of occupants"),
    "plug":      (2.0, 8.0, "W/m2",    "plug-load density"),
    "cut_start": (660, 1020, "min",    "cut start (minutes after midnight, 11:00-17:00)"),
    "cut_dur":   (60, 240, "min",      "cut duration"),
}
NAMES = list(PARAMS)
L, W, H = (16.0, 12.0, 3.3) if BTYPE == "VILLA" else (24.0, 16.0, 3.3)   # footprint (m) and storey height; 384 m2 floor area in both
AREA = L * W                       # floor area per zone (VILLA: 2 x 192 m2; LIGHT: 1 x 384 m2)

def _r10(x): return int(round(float(x) / 10.0) * 10)

def day_list(name, limits, values):
    v = ",\n    ".join(f"{x:g}" for x in values)
    return (f"Schedule:Day:List,\n    {name}_day, {limits}, No, 10,\n    {v};\n"
            f"Schedule:Week:Compact,\n    {name}_week, For: AllDays, {name}_day;\n"
            f"Schedule:Year,\n    {name}, {limits}, {name}_week, 1, 1, 12, 31;\n")

def profiles(cut_start, cut_dur):
    m = np.arange(10, 1441, 10)                      # end minute of each 10-min interval
    power = np.where((m > cut_start) & (m <= cut_start + cut_dur), 0.0, 1.0)
    hr = (m - 10) / 60.0
    occ = np.where(hr < 7, 1.0, np.where(hr < 14, 0.5, 0.9))
    light = np.where(hr < 6, 0.05, np.where(hr < 18, 0.15, np.where(hr < 23, 0.8, 0.3))) * power
    plug = np.where(hr < 7, 0.4, np.where(hr < 14, 0.5, np.where(hr < 23, 1.0, 0.5))) * power
    return power, occ, light, plug

def wall_layers(U, kind):
    """insulation thickness giving the target U-value (inner and outer surface resistances included)."""
    if BTYPE == "LIGHT":
        other = (0.012 / 0.25 + 0.0125 / 0.25 + 0.13 + 0.04) if kind == "wall" else (0.001 / 50.0 + 0.0125 / 0.25 + 0.10 + 0.04)
    elif kind == "wall":
        other = 0.02 / 0.72 + 0.20 / 0.60 + 0.015 / 0.50 + 0.13 + 0.04
    else:
        other = 0.03 / 1.0 + 0.05 / 1.4 + 0.20 / 1.4 + 0.015 / 0.5 + 0.10 + 0.04
    t = 0.035 * (1.0 / U - other)
    return max(t, 0.001)

def surface(name, stype, cons, zone, bc, bco, verts, sun=True):
    v = ",\n    ".join(f"{x:.3f},{y:.3f},{z:.3f}" for x, y, z in verts)
    se = "SunExposed, WindExposed" if sun else "NoSun, NoWind"
    return f"BuildingSurface:Detailed,\n    {name}, {stype}, {cons}, {zone}, , {bc}, {bco}, {se}, autocalculate, 4,\n    {v};\n"

def window(name, wall, verts):
    v = ",\n    ".join(f"{x:.3f},{y:.3f},{z:.3f}" for x, y, z in verts)
    return f"FenestrationSurface:Detailed,\n    {name}, Window, Glazing, {wall}, , autocalculate, , 1, 4,\n    {v};\n"

def geometry(wwr):
    out = []
    for k, (zn, z0) in enumerate((("GF", 0.0), ("FF", H)) if BTYPE == "VILLA" else (("FF", 0.0),)):
        z1 = z0 + H
        walls = {"S": [(0, 0, z1), (0, 0, z0), (L, 0, z0), (L, 0, z1)],
                 "E": [(L, 0, z1), (L, 0, z0), (L, W, z0), (L, W, z1)],
                 "N": [(L, W, z1), (L, W, z0), (0, W, z0), (0, W, z1)],
                 "W": [(0, W, z1), (0, W, z0), (0, 0, z0), (0, 0, z1)]}
        for f, vs in walls.items():
            wn = f"{zn}_wall_{f}"; out.append(surface(wn, "Wall", "ExtWall", zn, "Outdoors", "", vs))
            width = L if f in "SN" else W; area = wwr * width * H
            hw = min(1.6, area / (0.9 * width)); ww = area / hw; zb, zt = z0 + 0.9, z0 + 0.9 + hw
            a, b = (width - ww) / 2, (width + ww) / 2
            if f == "S": wv = [(a, 0, zt), (a, 0, zb), (b, 0, zb), (b, 0, zt)]
            elif f == "E": wv = [(L, a, zt), (L, a, zb), (L, b, zb), (L, b, zt)]
            elif f == "N": wv = [(b, W, zt), (b, W, zb), (a, W, zb), (a, W, zt)]
            else: wv = [(0, b, zt), (0, b, zb), (0, a, zb), (0, a, zt)]
            out.append(window(f"{zn}_win_{f}", wn, wv))
        floor = [(0, 0, z0), (0, W, z0), (L, W, z0), (L, 0, z0)]
        ceil = [(0, W, z1), (0, 0, z1), (L, 0, z1), (L, W, z1)]
        if BTYPE == "LIGHT":
            out.append(surface("FF_floor", "Floor", "GroundFloor", "FF", "Ground", "", floor, sun=False))
            out.append(surface("FF_roof", "Roof", "Roof", "FF", "Outdoors", "", ceil))
        elif zn == "GF":
            out.append(surface("GF_floor", "Floor", "GroundFloor", "GF", "Ground", "", floor, sun=False))
            out.append(surface("GF_ceiling", "Ceiling", "InterFloor", "GF", "Surface", "FF_floor", ceil, sun=False))
        else:
            out.append(surface("FF_floor", "Floor", "InterFloorRev", "FF", "Surface", "GF_ceiling", floor, sun=False))
            out.append(surface("FF_roof", "Roof", "Roof", "FF", "Outdoors", "", ceil))
    return "".join(out)

def make_idf(p):
    p = dict(p); cs, cd = _r10(p["cut_start"]), _r10(p["cut_dur"])
    power, occ, light, plug = profiles(cs, cd)
    tw, tr = wall_layers(p["U_wall"], "wall"), wall_layers(p["U_roof"], "roof")
    s = [f"""Version, 26.2;
SimulationControl, No, No, No, No, Yes, No, 1;
Building, Villa, {p['azimuth']:.1f}, City, 0.04, 0.4, FullExterior, 25, 6;
Timestep, 6;
RunPeriod, Summer, 6, 1, , 8, 31, , , No, No, No, Yes, Yes;
{CLIM['site']}
Site:GroundTemperature:BuildingSurface, {CLIM['ground']}
GlobalGeometryRules, UpperLeftCorner, Counterclockwise, Relative;
ScheduleTypeLimits, Fraction, 0, 1, Continuous;
ScheduleTypeLimits, AnyNumber;
Material, Plaster, Smooth, 0.02, 0.72, 1860, 840, 0.9, 0.6, 0.6;
Material, Block, MediumRough, 0.20, 0.60, 1200, 840, 0.9, 0.6, 0.6;
Material, WallIns, MediumRough, {tw:.4f}, 0.035, 25, 1400, 0.9, 0.6, 0.6;
Material, InPlaster, Smooth, 0.015, 0.50, 1300, 840, 0.9, 0.5, 0.5;
Material, Tiles, MediumRough, 0.03, 1.0, 2000, 840, 0.9, 0.6, 0.6;
Material, Screed, MediumRough, 0.05, 1.4, 2100, 840, 0.9, 0.6, 0.6;
Material, RoofIns, MediumRough, {tr:.4f}, 0.035, 25, 1400, 0.9, 0.6, 0.6;
Material, Slab, MediumRough, 0.20, 1.4, 2300, 840, 0.9, 0.6, 0.6;
Material, MassConc, MediumRough, 0.10, 1.4, 2300, 840, 0.9, 0.6, 0.6;
Material, Soil, Rough, 0.30, 1.0, 1800, 1000, 0.9, 0.6, 0.6;
WindowMaterial:SimpleGlazingSystem, GlazMat, 2.8, {p['SHGC']:.3f}, 0.6;
Material, CementBoard, MediumRough, 0.012, 0.25, 1000, 1000, 0.9, 0.6, 0.6;
Material, Gypsum, Smooth, 0.0125, 0.25, 900, 1000, 0.9, 0.5, 0.5;
Material, SteelDeck, Smooth, 0.001, 50.0, 7800, 500, 0.9, 0.6, 0.6;
{"Construction, ExtWall, CementBoard, WallIns, Gypsum;" if BTYPE == "LIGHT" else "Construction, ExtWall, " + ("Plaster, WallIns, Block" if p.get("ext_ins", 0) else "Plaster, Block, WallIns") + ", InPlaster;"}
{"Construction, Roof, SteelDeck, RoofIns, Gypsum;" if BTYPE == "LIGHT" else "Construction, Roof, Tiles, Screed, RoofIns, Slab, InPlaster;"}
Construction, InterFloor, Slab, Screed;
Construction, InterFloorRev, Screed, Slab;
Construction, GroundFloor, Soil, Slab, Screed;
Construction, MassCons, MassConc;
Construction, Glazing, GlazMat;
"""]
    s.append(day_list("Power", "Fraction", power)); s.append(day_list("Occ", "Fraction", occ))
    s.append(day_list("LightSch", "Fraction", light)); s.append(day_list("PlugSch", "Fraction", plug))
    s.append(day_list("Activity", "AnyNumber", np.full(144, 120.0))); s.append(day_list("SP", "AnyNumber", np.full(144, p["setpoint"])))
    s.append(day_list("CtrlType", "AnyNumber", np.full(144, 2.0))); s.append(day_list("AlwaysOff", "Fraction", np.zeros(144)))
    s.append(geometry(p["WWR"]))
    nz = 2 if BTYPE == "VILLA" else 1
    for zn in (("GF", "FF") if BTYPE == "VILLA" else ("FF",)):
        s.append(f"""Zone, {zn}, 0, 0, 0, 0, 1, 1, {H}, autocalculate;
InternalMass, {zn}_mass, MassCons, {zn}, , {p['mass'] * AREA:.1f};
People, {zn}_people, {zn}, Occ, People, {p['occupants'] / nz:.2f}, , , 0.3, autocalculate, Activity;
Lights, {zn}_lights, {zn}, LightSch, Watts/Area, , 5.0, , 0, 0.4, 0.2;
ElectricEquipment, {zn}_plug, {zn}, PlugSch, Watts/Area, , {p['plug']:.2f}, , 0, 0.3, 0;
ZoneInfiltration:DesignFlowRate, {zn}_inf, {zn}, , AirChanges/Hour, , , , {p['ACH']:.3f}, 1, 0, 0, 0;
ThermostatSetpoint:SingleCooling, {zn}_sp, SP;
ZoneControl:Thermostat, {zn}_tstat, {zn}, CtrlType, ThermostatSetpoint:SingleCooling, {zn}_sp;
ZoneHVAC:IdealLoadsAirSystem, {zn}_ideal, Power, {zn}_supply, , , 50, 13, 0.015, 0.0077, NoLimit, , , NoLimit, , , AlwaysOff, , None, , None;
ZoneHVAC:EquipmentList, {zn}_eq, SequentialLoad, ZoneHVAC:IdealLoadsAirSystem, {zn}_ideal, 1, 1, , ;
ZoneHVAC:EquipmentConnections, {zn}, {zn}_eq, {zn}_supply, , {zn}_node, {zn}_return;
""")
    s.append("""Output:Variable, *, Zone Mean Air Temperature, Timestep;
Output:Variable, *, Zone Operative Temperature, Timestep;
Output:Variable, *, Zone Ideal Loads Supply Air Total Cooling Rate, Timestep;
Output:SQLite, Simple;
""")
    return "".join(s)

def read_sql(path):
    con = sqlite3.connect(path)
    dd = {(k, n): i for i, k, n in con.execute("select ReportDataDictionaryIndex, KeyValue, Name from ReportDataDictionary")}
    tidx = np.array([r[0] for r in con.execute(
        "select TimeIndex from Time where (WarmupFlag is null or WarmupFlag=0) and IntervalType=-1 order by TimeIndex")])
    out = {}
    for (k, n), i in dd.items():
        v = dict(con.execute("select TimeIndex, Value from ReportData where ReportDataDictionaryIndex=?", (i,)).fetchall())
        out[(k.upper(), n)] = np.array([v[t] for t in tidx if t in v])
    con.close(); return out

def indicators(ts, p):
    """event indicators per cut day (upper floor, operative temperature unless stated), averaged over the 92 days."""
    cs, cd = _r10(p["cut_start"]), _r10(p["cut_dur"])
    top = ts[("FF", "Zone Operative Temperature")].reshape(-1, 144)
    tair = ts[("FF", "Zone Mean Air Temperature")].reshape(-1, 144)
    gf = ("GF", "Zone Operative Temperature") in ts
    tgf = ts[("GF", "Zone Operative Temperature")].reshape(-1, 144) if gf else np.full_like(top, np.nan)
    q = (ts[("FF_IDEAL", "Zone Ideal Loads Supply Air Total Cooling Rate")] + (ts[("GF_IDEAL", "Zone Ideal Loads Supply Air Total Cooling Rate")] if gf else 0)).reshape(-1, 144) / 1000.0
    m = np.arange(10, 1441, 10)
    win = (m > cs) & (m <= min(cs + cd + 120, 1440)); reb = (m > cs + cd) & (m <= min(cs + cd + 60, 1440))
    i0 = int(np.where(m == cs)[0][0])                                   # last cooled step before the cut
    x = top[:, win] > (top[:, i0] + 2.0)[:, None]
    t_exc = np.where(x.any(1), x.argmax(1) * 10 + 10, cd + 120).mean()
    return {"Tpeak_FF": top[:, win].max(1).mean(),                       # operative, upper floor
            "DH28_FF": (np.clip(top[:, win] - 28, 0, None).sum(1) / 6).mean(),
            "t_exceed_FF": t_exc,                                       # min until operative rises 2 K above its value at the cut
            "rebound_kW": q[:, reb].max(1).mean() if reb.any() else np.nan,
            "E_cool_kWh": q.sum() / 6.0,
            "Tpeak_GF": tgf[:, win].max(1).mean(),
            "Tpeak_FF_air": tair[:, win].max(1).mean(),
            "Tpeak_FF_hottest": top[:, win].max(1).max(),
            "n_days": top.shape[0]}

def simulate(p, keep=False):
    d = Path(tempfile.mkdtemp(prefix="ep_", dir=TMP))
    try:
        (d / "in.idf").write_text(make_idf(p))
        r = subprocess.run([str(EP), "-w", str(EPW), "-d", str(d), str(d / "in.idf")], capture_output=True, text=True)
        err = (d / "eplusout.err").read_text() if (d / "eplusout.err").exists() else r.stdout[-2000:]
        if r.returncode != 0 or "Fatal" in err:
            raise RuntimeError(err[-3000:])
        ts = read_sql(d / "eplusout.sql")
        out = indicators(ts, p); out["warnings"] = err.count("** Warning **")
        return (out, ts) if keep else out
    finally:
        if not keep: shutil.rmtree(d, ignore_errors=True)

TMP = Path(os.environ.get("EP_TMP", str(V2 / "ep_runs")))
TMP.mkdir(parents=True, exist_ok=True)
BASE = {"U_wall": 0.57, "U_roof": 0.4, "WWR": 0.2, "SHGC": 0.5, "ACH": 0.6, "mass": 1.5, "azimuth": 0.0,
        "setpoint": 24.0, "occupants": 6, "plug": 4.0, "cut_start": 720, "cut_dur": 120}

if __name__ == "__main__":
    import time; t0 = time.time(); out, ts = simulate(BASE, keep=True)
    print(f"{time.time()-t0:.1f}s", {k: round(v, 3) for k, v in out.items()})
    for k, v in ts.items(): print(k, len(v), round(float(v.min()), 2), round(float(v.max()), 2))
