"""F-ANALOG valid measurement v2 — training-only consistent pool.

Fixes the zero-pool-reachability bug: the candidate pool is built FROM the
training structures themselves (same fp scheme for candidates and analogs), so
a held-out train molecule IS a reachable candidate and the analog MRR is valid.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors
RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import analog_accel as A

TRAIN = os.path.join(ROOT, "train.parquet")
N_HOLDOUT = 60
SEED = 11
TOPN = 25
PPM_WIN = 10.0
ANALOG_WIN = 200.0
N_ANALOG = 100
SIM_POWER = 4.0


def _gens():
    return (rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096),
            rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096),
            rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6))


def fp_pack(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    g2, g3, rk = _gens()
    v = np.concatenate([g2.GetFingerprintAsNumPy(m).astype(np.uint8),
                        g3.GetFingerprintAsNumPy(m).astype(np.uint8),
                        rk.GetFingerprintAsNumPy(m).astype(np.uint8)])
    return np.packbits(v)


def fp_worker(smi):
    return fp_pack(smi)


_POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.float64)

def packed_tanimoto(a, b):
    """Robust byte-AND Tanimoto via a 256-entry popcount table (no uint64
    reinterpret — that breaks on non-8-multiple padding). b: (nb, nbytes)."""
    a = np.ascontiguousarray(a, np.uint8)
    b = np.ascontiguousarray(b, np.uint8)
    inter = _POP[np.bitwise_and(a[None, :], b)].sum(axis=1)
    na = int(_POP[a].sum())
    nb = _POP[b].sum(axis=1)
    u = na + nb - inter
    return np.where(u > 0, inter / u, 0.0)


def main():
    t0 = time.time()
    L = A.Lib(TRAIN)
    print(f"[lib+clean] {L.n} spectra cleaned {time.time()-t0:.0f}s", flush=True)

    # ---- build training structure pool (cached) ----
    POOL_CACHE = os.path.join(ROOT, "data", "trainpool.npz")
    if os.path.exists(POOL_CACHE):
        z = np.load(POOL_CACHE, allow_pickle=True)
        pool_fp = z["fp"]; pool_mass = z["mass"]; pool_keys = z["keys"]
        o = np.argsort(pool_mass, kind="stable")
        pool_fp, pool_keys, pool_mass = pool_fp[o], pool_keys[o], pool_mass[o]
        print(f"[pool] cached: {len(pool_mass):,} structures (fp {pool_fp.shape}) {time.time()-t0:.0f}s", flush=True)
    else:
        t = pq.read_table(TRAIN, columns=["inchikey14", "normalized_smiles"])
        df = t.to_pandas().dropna().drop_duplicates("inchikey14")
        with Pool(8) as mp:
            fpm = mp.map(fp_worker, df.normalized_smiles.tolist(), chunksize=500)
        ok = [i for i, x in enumerate(fpm) if x is not None]
        pool_fp = np.stack([fpm[i] for i in ok]).astype(np.uint8)
        pool_keys = df.inchikey14.values[ok]
        pool_mass = np.array([Descriptors.ExactMolWt(Chem.MolFromSmiles(s)) for s in df.normalized_smiles.values[ok]])
        o = np.argsort(pool_mass, kind="stable")
        pool_fp, pool_keys, pool_mass = pool_fp[o], pool_keys[o], pool_mass[o]
        np.savez_compressed(POOL_CACHE, fp=pool_fp, mass=pool_mass, keys=pool_keys.astype(object))
        print(f"[pool] built+cached: {len(pool_mass):,} structures (fp {pool_fp.shape}) {time.time()-t0:.0f}s", flush=True)

    # ---- holdout: train molecules that ARE in the pool (all of them now) ----
    rng = np.random.default_rng(SEED)
    pool_key_set = set(pool_keys.tolist())
    if 'df' not in dir():
        t = pq.read_table(TRAIN, columns=["inchikey14"])
        df = t.to_pandas().drop_duplicates("inchikey14")
    reach = [k for k in df.inchikey14.values if k in pool_key_set]
    chosen = set(rng.choice(reach, size=min(N_HOLDOUT, len(reach)), replace=False).tolist())
    print(f"[holdout] {len(chosen)} reachable molecules held out {time.time()-t0:.0f}s", flush=True)

    # reps (one spectrum per structure) sorted by neutral mass
    best = {}
    for r in range(L.n):
        k = L.ik[r]
        if k and np.isfinite(L.nm[r]) and k not in best:
            best[k] = r
    rep = np.array(sorted(best.values()))
    nmr = L.nm[rep]; oor = np.argsort(nmr)
    rep = rep[oor]; nmr = nmr[oor]
    print(f"[rep] {len(rep)} representatives {time.time()-t0:.0f}s", flush=True)

    key2pool = {k: i for i, k in enumerate(pool_keys.tolist())}
    mrr = 0.0; hit = 0; q = 0
    for r in rep:
        k = str(L.ik[r])
        if k not in chosen:
            continue
        q += 1
        target = float(L.nm[r])
        a, b = L.off[r], L.off[r + 1]
        qm, qp = L.cmz[a:b], L.cp[a:b]
        lo = np.searchsorted(nmr, target - ANALOG_WIN, "left")
        hi = np.searchsorted(nmr, target + ANALOG_WIN, "right")
        if hi <= lo:
            continue
        rep_rows = rep[lo:hi]
        shifts = (target - nmr[lo:hi]).astype(np.float32)
        sims = A._search_shift_batch(qm, qp, rep_rows.astype(np.int64),
                                     L.coff.astype(np.int64),
                                     L.cmz, L.cp, A.MZ_TOL, shifts)
        agg = {}
        for j in range(len(rep_rows)):
            ka = str(L.ik[rep_rows[j]])
            if sims[j] > agg.get(ka, -1.0): agg[ka] = float(sims[j])
        analogs = sorted(agg.items(), key=lambda x: -x[1])[:N_ANALOG]

        half = PPM_WIN * 1e-6 * target
        cand = np.where(np.abs(pool_mass - target) <= half)[0]
        if len(cand) == 0: continue
        cf = pool_fp[cand]
        best_score = np.zeros(len(cand))
        for ak, s in analogs:
            pi = key2pool.get(ak)
            if pi is None: continue
            tan = packed_tanimoto(pool_fp[pi], cf)
            best_score = np.maximum(best_score, float(s) ** SIM_POWER * tan)
        order = np.argsort(-best_score)
        rank = None
        for pos in range(min(TOPN, len(order))):
            if str(pool_keys[cand[order[pos]]]) == k: rank = pos + 1; break
        if rank:
            mrr += 1.0 / rank; hit += 1
        if q % 10 == 0:
            print(f"  [{time.time()-t0:6.0f}s] q={q} mr_sofar={mrr/q:.4f} hit={hit}/{q}", flush=True)
    mrr /= max(1, q); h25 = hit / max(1, q)
    print("=" * 60)
    print(f"[F-ANALOG] valid tr-pool | MRR@25 = {mrr:.4f} | hit@25 = {h25:.4f} | n={q} | target>=0.45 (winner ~0.52)")
    print("=" * 60)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "fanalog_measured.txt"), "w").write(
        f"mode=valid-train-pool\nMRR@25={mrr:.6f}\nhit@25={h25:.6f}\nn_queries={q}\n")


if __name__ == "__main__":
    main()