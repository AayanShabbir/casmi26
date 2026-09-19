"""F-RANKER — combined-evidence GBDT candidate ranker (CASMI26).

Steps: for each query (train molecule), gather per-candidate features from the
analog batch (library sim, analog sim value + rank, tanimoto, sim^4*tan, counts,
ppm error). Train a HistGradientBoosting ranker on non-holdout queries labelled
by "is this candidate the truth?", then rank the holdout candidates and measure
MRR@25. GATE: must beat the F-ANALOG incumbent 0.2125.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import HistGradientBoostingClassifier
RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import analog_accel as A

TRAIN = os.path.join(ROOT, "train.parquet")
POOL_CACHE = os.path.join(ROOT, "data", "trainpool.npz")
N_TRAINQ = 120      # non-holdout molecules whose candidates become training rows
N_HOLDOUT = 60      # holdout molecules for eval (SEED 11, same as F-ANALOG)
SEED = 11
PPM_WIN = 10.0
ANALOG_WIN = 200.0
N_ANALOG = 100
SIM_POWER = 4.0
NFEAT = 9


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
    a = np.ascontiguousarray(a, np.uint8)
    b = np.ascontiguousarray(b, np.uint8)
    inter = _POP[np.bitwise_and(a[None, :], b)].sum(axis=1)
    na = int(_POP[a].sum())
    nb = _POP[b].sum(axis=1)
    u = na + nb - inter
    return np.where(u > 0, inter / u, 0.0)


def load_pool():
    z = np.load(POOL_CACHE, allow_pickle=True)
    fp = z["fp"]; mass = z["mass"]; keys = z["keys"]
    o = np.argsort(mass, kind="stable")
    return fp[o], mass[o], keys[o]


def query_features(L, key2pool, pool_fp, pool_keys, pool_mass, nmr, rep, r):
    """Return (Xf[nc,NFEAT], y[nc], target, cand_idx, qkey)."""
    k = str(L.ik[r]); target = float(L.nm[r])
    a, b = L.off[r], L.off[r + 1]
    qm, qp = L.cmz[a:b], L.cp[a:b]
    lo = np.searchsorted(nmr, target - ANALOG_WIN, "left")
    hi = np.searchsorted(nmr, target + ANALOG_WIN, "right")
    if hi <= lo:
        return None
    rep_rows = rep[lo:hi]
    shifts = (target - nmr[lo:hi]).astype(np.float32)
    sims = A._search_shift_batch(qm, qp, rep_rows.astype(np.int64),
                                 L.coff.astype(np.int64), L.cmz, L.cp, A.MZ_TOL, shifts)
    agg = {}
    for j in range(len(rep_rows)):
        ka = str(L.ik[rep_rows[j]])
        if sims[j] > agg.get(ka, -1.0):
            agg[ka] = float(sims[j])
    analogs = sorted(agg.items(), key=lambda x: -x[1])[:N_ANALOG]
    # unique analog keys + their sims
    akeys = [x[0] for x in analogs]; asims = np.array([x[1] for x in analogs])
    half = PPM_WIN * 1e-6 * target
    cand = np.where(np.abs(pool_mass - target) <= half)[0]
    if len(cand) == 0:
        return None
    cf = pool_fp[cand]
    nc = len(cand)
    # per-candidate: max sim to any analog, its tanimoto, combined, count, rank
    max_sim = np.zeros(nc); bk_tan = np.zeros(nc); combined = np.zeros(nc); cnt = np.zeros(nc, np.int32)
    for j, ak in enumerate(akeys):
        pi = key2pool.get(ak)
        if pi is None:
            continue
        t = packed_tanimoto(pool_fp[pi], cf)   # (nc,)
        s = float(asims[j])
        upd = s ** SIM_POWER * t
        gt = t > bk_tan
        max_sim[gt] = s
        bk_tan[gt] = t[gt]
        np.maximum(combined, upd, out=combined)
        cnt += (t > 0).astype(np.int32)
    # lib sim = max over analogs that ARE the candidate key itself
    lib_sim = np.zeros(nc)
    for j, ak in enumerate(akeys):
        pi = key2pool.get(ak)
        if pi is None:
            continue
        m = np.where(pool_keys[cand] == np.asarray(pool_keys[pi]))[0]
        if len(m):
            lib_sim[m] = max(lib_sim[m], float(asims[j]))
    # rank features: normalize
    rankoftop = np.zeros(nc)
    for c in range(nc):
        # fractional rank of this candidate's max_sim among all peaks (1=best)
        rankoftop[c] = 1.0 - (np.sum(max_sim > max_sim[c]) / max(nc, 1))
    ppm = (pool_mass[cand] - target) / target * 1e6
    lcand = np.log(nc + 1.0)
    Xf = np.column_stack([lib_sim, max_sim, bk_tan, combined, rankoftop, cnt / max(1, len(akeys)),
                          np.abs(ppm), np.full(nc, lcand), np.log1p(cnt)])
    y = np.zeros(nc, dtype=np.int8)
    qkidx = np.where(pool_keys[cand] == np.asarray(k))[0]
    if len(qkidx):
        y[qkidx[0]] = 1
    return Xf.astype(np.float32), y, target, cand, k


def main():
    t0 = time.time()
    L = A.Lib(TRAIN)
    print(f"[lib+clean] {L.n} spectra {time.time()-t0:.0f}s", flush=True)
    pool_fp, pool_mass, pool_keys = load_pool()
    print(f"[pool] {len(pool_mass):,} cached {time.time()-t0:.0f}s", flush=True)
    key2pool = {k: i for i, k in enumerate(pool_keys.tolist())}
    # reps
    best = {}
    for r in range(L.n):
        k = L.ik[r]
        if k and np.isfinite(L.nm[r]) and k not in best:
            best[k] = r
    rep = np.array(sorted(best.values())); nmr = L.nm[rep]; oor = np.argsort(nmr)
    rep = rep[oor]; nmr = nmr[oor]
    print(f"[rep] {len(rep)} representatives {time.time()-t0:.0f}s", flush=True)
    # holdout / train-query split
    rng = np.random.default_rng(SEED)
    pool_key_set = set(pool_keys.tolist())
    t2 = pq.read_table(TRAIN, columns=["inchikey14"]); df = t2.to_pandas().drop_duplicates("inchikey14")
    reach = [k for k in df.inchikey14.values if k in pool_key_set]
    # identical holdout to F-ANALOG (SEED 11, rng.choice 60) so the gate compares like-for-like
    eval_keys = set(rng.choice(reach, size=min(N_HOLDOUT, len(reach)), replace=False).tolist())
    eval_keys_remaining = [k for k in reach if k not in eval_keys]
    train_keys = eval_keys_remaining[:N_TRAINQ]
    print(f"[split] holdout={len(eval_keys)} trainq={len(train_keys)} {time.time()-t0:.0f}s", flush=True)
    # ---- training rows ---- 
    Xtr = []; ytr = []
    for r in rep:
        k = str(L.ik[r])
        if k not in set(train_keys):
            continue
        res = query_features(L, key2pool, pool_fp, pool_keys, pool_mass, nmr, rep, r)
        if res is None:
            continue
        Xf, y, _, _, _ = res
        Xtr.append(Xf); ytr.append(y)
    if not Xtr:
        print("[F-RANKER] no training rows!"); return
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    print(f"[train] {Xtr.shape[0]} rows, {ytr.sum()} positives {time.time()-t0:.0f}s", flush=True)
    # ---- train (2 priors x 2 seeds bagging) ----
    RANKERS = []
    for w1 in (0.5, 0.6):
        W = np.where(ytr == 1, w1, 1.0 - w1)
        for sd in (0, 1):
            m = HistGradientBoostingClassifier(random_state=sd, max_depth=6, max_iter=500,
                                               learning_rate=0.03, min_samples_leaf=80,
                                               l2_regularization=1.0)
            m.fit(Xtr, ytr, sample_weight=W)
            RANKERS.append(m)
    def proba(X):
        return np.mean([m.predict_proba(X)[:, 1] for m in RANKERS], axis=0)
    print(f"[fit] {len(RANKERS)} HistGB {time.time()-t0:.0f}s", flush=True)
    # ---- eval on holdout ----
    mrr = 0.0; hit = 0; q = 0
    for r in rep:
        k = str(L.ik[r])
        if k not in eval_keys:
            continue
        res = query_features(L, key2pool, pool_fp, pool_keys, pool_mass, nmr, rep, r)
        if res is None:
            continue
        Xf, y, _, cand, kk = res
        p = proba(Xf)
        order = np.argsort(-p)
        rank = None
        for pos in range(min(TOPN, len(order))):
            if str(pool_keys[cand[order[pos]]]) == kk:
                rank = pos + 1; break
        q += 1
        if rank:
            mrr += 1.0 / rank; hit += 1
        if q % 20 == 0:
            print(f"  [{time.time()-t0:6.0f}s] q={q} mr_sofar={mrr/q:.4f} hit={hit}/{q}", flush=True)
    mrr /= max(1, q); h25 = hit / max(1, q)
    print("=" * 60)
    print(f"[F-RANKER] MRR@25 = {mrr:.4f} | hit@25 = {h25:.4f} | n={q} | gate>0.2125 (F-ANALOG)")
    print("=" * 60)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "franker_measured.txt"), "w").write(
        f"mode=gbdt-rank\nMRR@25={mrr:.6f}\nhit@25={h25:.6f}\nn_queries={q}\n")


TOPN = 25
if __name__ == "__main__":
    main()