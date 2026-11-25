"""Simulation designs for the XAI-fidelity study (EnergyPlus, parallel).

stage saltelli : Sobol ground truth, independent uniform inputs. N base samples -> N*(D+2) runs (A, B, AB_i)
stage train    : surrogate training/test pools: IND (independent, Sobol QMC) and COR (correlated housing stock)
stage pdp      : simulator partial dependence (interventional main effects) on a grid, for IND and COR backgrounds
stage local    : simulator baseline-Shapley values (permutation sampling) for explained instances
Results -> ../results/*.parquet
"""
import sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import qmc, norm
import villa as V

import os
R = V.V2 / "results" / os.environ.get("EP_RESULTS_SUB", ""); R.mkdir(exist_ok=True)
NAMES = V.NAMES; D = len(NAMES)
LO = np.array([V.PARAMS[n][0] for n in NAMES]); HI = np.array([V.PARAMS[n][1] for n in NAMES])
TARGETS = ["Tpeak_FF", "DH28_FF", "t_exceed_FF", "rebound_kW", "E_cool_kWh", "Tpeak_GF", "Tpeak_FF_hottest"]
NPROC = 14

def scale(u): return LO + u * (HI - LO)

def _one(row):
    try:
        out = V.simulate(dict(zip(NAMES, row)))
    except Exception as e:
        out = {"error": str(e)[-300:]}
    return out

def run_matrix(X, tag):
    t0 = time.time()
    with Pool(NPROC) as pool:
        res = pool.map(_one, [list(map(float, r)) for r in X], chunksize=8)
    df = pd.concat([pd.DataFrame(X, columns=NAMES), pd.DataFrame(res)], axis=1)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"{tag}: {len(X)} runs in {time.time()-t0:.0f}s, errors {n_err}", flush=True)
    return df

# ---------------------------------------------------------------- designs
CORR_PAIRS = {("U_wall", "U_roof"): 0.8, ("U_wall", "ACH"): 0.7, ("U_roof", "ACH"): 0.7,
              ("occupants", "plug"): 0.6, ("WWR", "SHGC"): 0.4}
# strongly dependent housing stock: building vintage drives envelope, leakiness and (inversely) insulation-free heavy mass
CORR_PAIRS_STRONG = {("U_wall", "U_roof"): 0.95, ("U_wall", "ACH"): 0.90, ("U_roof", "ACH"): 0.90,
                     ("mass", "U_wall"): 0.85, ("mass", "U_roof"): 0.85, ("mass", "ACH"): 0.80,
                     ("occupants", "plug"): 0.6, ("WWR", "SHGC"): 0.4}
def corr_matrix(pairs=None):
    pairs = CORR_PAIRS if pairs is None else pairs
    C = np.eye(D)
    for (a, b), r in pairs.items():
        i, j = NAMES.index(a), NAMES.index(b); C[i, j] = C[j, i] = r
    return C

def sample_ind(n, seed):
    return scale(qmc.Sobol(D, scramble=True, seed=seed).random(n))

def sample_cor(n, seed, pairs=None):
    C = corr_matrix(pairs); w, v = np.linalg.eigh(C); C = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T   # nearest PSD
    rng = np.random.default_rng(seed); Z = rng.multivariate_normal(np.zeros(D), C, size=n)
    return scale(norm.cdf(Z))
def sample_str(n, seed): return sample_cor(n, seed, CORR_PAIRS_STRONG)

def stage_saltelli(N=2048):
    AB = qmc.Sobol(2 * D, scramble=True, seed=11).random(N); A, B = scale(AB[:, :D]), scale(AB[:, D:])
    blocks = [A, B] + [np.where(np.arange(D) == i, B, A) for i in range(D)]
    X = np.vstack(blocks); df = run_matrix(X, "saltelli")
    df["block"] = np.repeat(["A", "B"] + [f"AB_{n}" for n in NAMES], N); df["row"] = np.tile(np.arange(N), D + 2)
    df.to_parquet(R / "saltelli.parquet")

def stage_train(n=4000):
    run_matrix(sample_ind(n, 21), "train IND").to_parquet(R / "train_IND.parquet")
    run_matrix(sample_cor(n, 22), "train COR").to_parquet(R / "train_COR.parquet")

GRID = 9
def pdp_grid(i): return np.linspace(LO[i], HI[i], GRID)

def stage_pdp(nbg=150):
    for tag, bg in (("IND", sample_ind(nbg, 31)), ("COR", sample_cor(nbg, 32))):
        rows, meta = [], []
        for i, n in enumerate(NAMES):
            for v in pdp_grid(i):
                X = bg.copy(); X[:, i] = v; rows.append(X); meta += [(n, v, k) for k in range(nbg)]
        df = run_matrix(np.vstack(rows), f"pdp {tag}")
        df[["feature", "value", "bg"]] = pd.DataFrame(meta)
        df.to_parquet(R / f"pdp_truth_{tag}.parquet")

def stage_local(n_inst=50, M=32):
    """baseline Shapley of the simulator: phi_i(x; b) with baseline b = centre of the input space."""
    b = (LO + HI) / 2
    for tag, pool in (("IND", sample_ind(n_inst, 41)), ("COR", sample_cor(n_inst, 42))):
        rng = np.random.default_rng(43); rows, meta = [], []
        for k, x in enumerate(pool):
            for m in range(M):
                perm = rng.permutation(D) if m % 2 == 0 else perm[::-1]      # antithetic pairs
                z = b.copy(); rows.append(z.copy()); meta.append((k, m, -1))
                for j in perm:
                    z[j] = x[j]; rows.append(z.copy()); meta.append((k, m, j))
        df = run_matrix(np.vstack(rows), f"local {tag}")
        df[["inst", "perm", "added"]] = pd.DataFrame(meta)
        pd.DataFrame(pool, columns=NAMES).to_parquet(R / f"local_instances_{tag}.parquet")
        df.to_parquet(R / f"local_truth_{tag}.parquet")

def stage_strong(n=4000, nbg=150, n_inst=50, M=32):
    """strongly dependent stock (STR): training pool, interventional PDP truth and local baseline-Shapley truth."""
    run_matrix(sample_str(n, 23), "train STR").to_parquet(R / "train_STR.parquet")
    bg = sample_str(nbg, 33); rows, meta = [], []
    for i, nm in enumerate(NAMES):
        for v in pdp_grid(i):
            X = bg.copy(); X[:, i] = v; rows.append(X); meta += [(nm, v, k) for k in range(nbg)]
    df = run_matrix(np.vstack(rows), "pdp STR"); df[["feature", "value", "bg"]] = pd.DataFrame(meta); df.to_parquet(R / "pdp_truth_STR.parquet")
    b = (LO + HI) / 2; pool = sample_str(n_inst, 43); rng = np.random.default_rng(44); rows, meta = [], []
    for k, x in enumerate(pool):
        for m in range(M):
            perm = rng.permutation(D) if m % 2 == 0 else perm[::-1]
            z = b.copy(); rows.append(z.copy()); meta.append((k, m, -1))
            for j in perm:
                z[j] = x[j]; rows.append(z.copy()); meta.append((k, m, j))
    df = run_matrix(np.vstack(rows), "local STR"); df[["inst", "perm", "added"]] = pd.DataFrame(meta)
    pd.DataFrame(pool, columns=NAMES).to_parquet(R / "local_instances_STR.parquet"); df.to_parquet(R / "local_truth_STR.parquet")

if __name__ == "__main__":
    for st in sys.argv[1:]: globals()["stage_" + st]()
