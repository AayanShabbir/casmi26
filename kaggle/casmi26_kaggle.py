"""CAASMI26 self-contained Kaggle submission script (Code Competition path).

Builds the F-RANKER pipeline start-to-finish from the raw parquet files and
writes submission.csv. Self-contained: inlines analog_accel._clean + neutral_mass
(negative-adduct fix included), f_speed.search_shift_prange, and f_ranker
feature-extraction/training/inference. Intended to run inside a Kaggle CPU
notebook (<9h budget; this pipeline is ~10-15 min). For local validation it
reuses data/trainpool.npz when present and TRAIN/TEST point at local files.

Usage:
  python casmi26_kaggle.py                     # Kaggle default /kaggle/input paths
  TRAIN=train.parquet TEST=test.parquet OUT=submission.csv python casmi26_kaggle.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors
RDLogger.DisableLog("rdApp.*")

# ---------------- config ----------------
COMP = os.environ.get("COMP", "enveda-casmi26-molecule-id-mass-spectra")
KAGGLE_IN = os.environ.get("KAGGLE_IN", f"/kaggle/input/{COMP}")
TRAIN = os.environ.get("TRAIN", f"{KAGGLE_IN}/train.parquet")
TEST  = os.environ.get("TEST",  f"{KAGGLE_IN}/test.parquet")
OUT   = os.environ.get("OUT",   "submission.csv")
LOCAL_POOL = os.environ.get("LOCAL_POOL", "")   # reuse cached pool for local validation
N_TRAINQ = 120; N_HOLDOUT = 60; SEED = 11
PPM_WIN = 10.0; ANALOG_WIN = 200.0; N_ANALOG = 100; SIM_POWER = 4.0
TOPN = 25; NFEAT = 20

PROTON = 1.00728; INT_FLOOR = 0.002; MAX_PEAKS = 256; MZ_TOL = 0.01
ADDUCT_NEUT = {"[M+H]+": PROTON, "[M+NH4]+": 18.0383, "[M+Na]+": 22.9892,
               "[M+H-H2O]+": PROTON - 18.0106, "[M+2H]2+": 2*PROTON,
               "[M-H]-": -PROTON, "[M+CH2O2-H]-": -(46.0255 - PROTON),
               "[M+K]+": 38.9637, "[M+Cl]-": -34.9693}

from numba import njit, prange

# ---------------- cleaning (analog_accel) ----------------
def neutral_mass(mz, adduct):
    mz = np.asarray(mz, np.float64); out = np.full(len(mz), np.nan)
    for a, d in ADDUCT_NEUT.items():
        m = np.asarray(adduct, dtype=object) == a
        out[m] = mz[m] - d
    return out

@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk):
    n = len(mz)
    if n == 0: return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx: mx = it[i]
    if mx <= 0: return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx; idx = np.empty(n, np.int64); c = 0
    for i in range(n):
        if it[i] >= thr: idx[c] = i; c += 1
    if c == 0: return np.empty(0, np.float32), np.empty(0, np.float32)
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c): v[i] = it[idx[i]]
        o = np.argsort(v); idx2 = np.empty(topk, np.int64)
        for j in range(topk): idx2[j] = idx[o[c - topk + j]]
        idx = idx2; c = topk
    om = np.empty(c, np.float32); oi = np.empty(c, np.float32)
    mz2 = np.empty(c, np.float64); ii = np.empty(c, np.float64)
    for j in range(c): mz2[j] = mz[idx[j]]; ii[j] = it[idx[j]]
    o = np.argsort(mz2); s = 0.0
    for j in range(c): om[j] = mz2[o[j]]; oi[j] = ii[o[j]]; s += ii[o[j]]
    if s <= 0: return np.empty(0, np.float32), np.empty(0, np.float32)
    p = np.empty(c, np.float64)
    for j in range(c): p[j] = ii[o[j]] / s
    S = 0.0
    for j in range(c):
        if p[j] > 0: S -= p[j] * np.log(p[j])
    if S < 3.0:
        w = 0.25 + 0.25 * S; s2 = 0.0
        for j in range(c): p[j] = p[j] ** w; s2 += p[j]
        if s2 > 0:
            for j in range(c): p[j] /= s2
    return om.astype(np.float32), p.astype(np.float32)

# ---------------- fast mass-shift sim (f_speed) ----------------
@njit(cache=True, fastmath=True)
def _entropy_sim_inline(qmz, qp, cmz, cp, tol):
    n = len(qmz); m = len(cmz)
    if n == 0 or m == 0: return 0.0
    SA = 0.0
    for x in range(n):
        if qp[x] > 0.0: SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0.0: SB -= cp[x] * np.log(cp[x])
    buf = np.empty(n + m, np.float64); b = 0
    i = 0; j = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol: buf[b] = qp[i]; i += 1; b += 1
        elif d > tol: buf[b] = cp[j]; j += 1; b += 1
        else: buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n: buf[b] = qp[i]; i += 1; b += 1
    while j < m: buf[b] = cp[j]; j += 1; b += 1
    tot = 0.0
    for x in range(b): tot += buf[x]
    if tot <= 0: return 0.0
    SAB = 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0.0: SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)

@njit(cache=True, parallel=True, fastmath=True)
def search_shift_prange(qmz, qp, rep_rows, coff, cmz, cp, tol, shifts):
    out = np.empty(len(rep_rows), np.float32)
    for k in prange(len(rep_rows)):
        r = rep_rows[k]; a, b = coff[r], coff[r + 1]
        if b <= a: out[k] = 0.0; continue
        s_direct = _entropy_sim_inline(qmz, qp, cmz[a:b], cp[a:b], tol)
        sh = shifts[k]
        if -0.001 < sh < 0.001: out[k] = s_direct; continue
        smz = np.empty(b - a, np.float32)
        for j in range(b - a): smz[j] = cmz[a + j] + sh
        s_shift = _entropy_sim_inline(qmz, qp, smz, cp[a:b], tol)
        out[k] = s_direct if s_direct > s_shift else s_shift
    return out

# ---------------- pool + fingerprints (f_ranker) ----------------
def _gens():
    return (rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096),
            rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096),
            rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6))

def fp_pack(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    g2, g3, rk = _gens()
    v = np.concatenate([g2.GetFingerprintAsNumPy(m).astype(np.uint8),
                        g3.GetFingerprintAsNumPy(m).astype(np.uint8),
                        rk.GetFingerprintAsNumPy(m).astype(np.uint8)])
    return np.packbits(v)

def fp_worker(smi): return fp_pack(smi)

_POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.float64)
def packed_tanimoto(a, b):
    a = np.ascontiguousarray(a, np.uint8); b = np.ascontiguousarray(b, np.uint8)
    inter = _POP[np.bitwise_and(a[None, :], b)].sum(axis=1)
    na = int(_POP[a].sum()); nb = _POP[b].sum(axis=1); u = na + nb - inter
    return np.where(u > 0, inter / u, 0.0)

def build_or_load_pool():
    if LOCAL_POOL and os.path.exists(LOCAL_POOL):
        z = np.load(LOCAL_POOL, allow_pickle=True)
        fp = z["fp"]; mass = z["mass"]; keys = z["keys"]
        o = np.argsort(mass, kind="stable")
        return fp[o], mass[o], keys[o]
    from multiprocessing import Pool
    t0 = time.time()
    t = pq.read_table(TRAIN, columns=["inchikey14", "normalized_smiles"])
    df = t.to_pandas().dropna().drop_duplicates("inchikey14")
    smis = df.normalized_smiles.tolist()
    print(f"[pool-fp] {len(smis):,} structures {time.time()-t0:.0f}s", flush=True)
    with Pool(os.cpu_count()) as mp:
        fpm = mp.map(fp_worker, smis, chunksize=500)
    ok = [i for i, x in enumerate(fpm) if x is not None]
    fp = np.stack([fpm[i] for i in ok]).astype(np.uint8)
    keys = df.inchikey14.values[ok]
    mass = np.array([Descriptors.ExactMolWt(Chem.MolFromSmiles(df.normalized_smiles.values[i]))
                     for i in ok])
    o = np.argsort(mass, kind="stable")
    return fp[o], mass[o], keys[o]

# ---------------- Library wrapper (cleaned spectra) ----------------
class Lib:
    def __init__(self, path, key_col="inchikey14"):
        if key_col == "inchikey14":
            t = pq.read_table(path, columns=[key_col, "adduct", "precursor_mz",
                                             "ms2_mzs", "ms2_normalized_intensities"])
        else:
            t = pq.read_table(path, columns=[key_col, "adduct", "precursor_mz",
                                             "ms2_mzs", "ms2_normalized_intensities"])
        self.ik = np.asarray(t.column(key_col).cast("string").to_pylist(), dtype=object)
        add = np.asarray(t.column("adduct").cast("string").to_pylist(), dtype=object)
        prec = t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
        self.nm = neutral_mass(prec, add)
        mzc = t.column("ms2_mzs").combine_chunks(); itc = t.column("ms2_normalized_intensities").combine_chunks()
        self.off = mzc.offsets.to_numpy().astype(np.int64)
        self.mz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        self.it = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        self.n = len(self.ik)
        off = np.empty(self.n + 1, np.int64); off[0] = 0
        for r in range(self.n):
            a, b = self.off[r], self.off[r + 1]
            cm, cp = _clean(self.mz[a:b].astype(np.float32), self.it[a:b].astype(np.float32), INT_FLOOR, MAX_PEAKS)
            off[r + 1] = off[r] + len(cm)
        total = int(off[-1]); cmzs = np.empty(total, np.float32); cps = np.empty(total, np.float32)
        for r in range(self.n):
            a, b = self.off[r], self.off[r + 1]
            cm, cp = _clean(self.mz[a:b].astype(np.float32), self.it[a:b].astype(np.float32), INT_FLOOR, MAX_PEAKS)
            s, e = off[r], off[r + 1]; cmzs[s:e] = cm; cps[s:e] = cp
        self.cmz = cmzs; self.cp = cps; self.coff = off

# ---------------- feature extraction (f_ranker.features_for_query) ----------------
def features_for_query(qm, qp, target, rep, nmr, acoff, acmz, acp, aik,
                       key2pool, pool_fp, pool_keys, pool_mass, qkey=None):
    lo = np.searchsorted(nmr, target - ANALOG_WIN, "left")
    hi = np.searchsorted(nmr, target + ANALOG_WIN, "right")
    if hi <= lo: return None
    rep_rows = rep[lo:hi]; shifts = (target - nmr[lo:hi]).astype(np.float32)
    sims = search_shift_prange(qm, qp, rep_rows.astype(np.int64),
                               acoff.astype(np.int64), acmz, acp, MZ_TOL, shifts)
    agg = {}
    for j in range(len(rep_rows)):
        ka = str(aik[rep_rows[j]])
        if sims[j] > agg.get(ka, -1.0): agg[ka] = float(sims[j])
    analogs = sorted(agg.items(), key=lambda x: -x[1])[:N_ANALOG]
    akeys = [x[0] for x in analogs]; asims = np.array([x[1] for x in analogs])
    half = PPM_WIN * 1e-6 * target
    cand = np.where(np.abs(pool_mass - target) <= half)[0]
    if len(cand) == 0: return None
    cf = pool_fp[cand]; nc = len(cand)
    max_sim = np.zeros(nc); bk_tan = np.zeros(nc); combined = np.zeros(nc); cnt = np.zeros(nc, np.int32)
    max_sim2 = np.zeros(nc); max_sim4 = np.zeros(nc); sum_sim2 = np.zeros(nc); sum_sim4 = np.zeros(nc)
    n_hi = np.zeros(nc, np.int32); n_mid = np.zeros(nc, np.int32); n_lo = np.zeros(nc, np.int32)
    for j, ak in enumerate(akeys):
        pi = key2pool.get(ak)
        if pi is None: continue
        t = packed_tanimoto(pool_fp[pi], cf); s = float(asims[j]); s2 = s * s; s4 = s2 * s2
        upd = s ** SIM_POWER * t; gt = t > bk_tan
        max_sim[gt] = s; bk_tan[gt] = t[gt]
        np.maximum(combined, upd, out=combined)
        ov = t > 0; cnt += ov.astype(np.int32)
        np.maximum(max_sim2, np.where(ov, s2, 0.0), out=max_sim2)
        np.maximum(max_sim4, np.where(ov, s4, 0.0), out=max_sim4)
        sum_sim2 += np.where(ov, s2, 0.0); sum_sim4 += np.where(ov, s4, 0.0)
        n_hi += (ov & (s >= 0.8)).astype(np.int32)
        n_mid += (ov & (s >= 0.6) & (s < 0.8)).astype(np.int32)
        n_lo += (ov & (s >= 0.4) & (s < 0.6)).astype(np.int32)
    lib_sim = np.zeros(nc)
    for j, ak in enumerate(akeys):
        pi = key2pool.get(ak)
        if pi is None: continue
        m = np.where(pool_keys[cand] == np.asarray(pool_keys[pi]))[0]
        if len(m): lib_sim[m] = max(lib_sim[m], float(asims[j]))
    rankoftop = np.zeros(nc); ecdf_tan = np.zeros(nc); ecdf_comb = np.zeros(nc)
    for c in range(nc):
        rankoftop[c] = 1.0 - (np.sum(max_sim > max_sim[c]) / max(nc, 1))
        ecdf_tan[c] = np.sum(bk_tan < bk_tan[c]) / max(nc, 1)
        ecdf_comb[c] = np.sum(combined < combined[c]) / max(nc, 1)
    ppm = (pool_mass[cand] - target) / target * 1e6
    lcand = np.log(nc + 1.0); ovn = np.maximum(cnt, 1.0)
    Xf = np.column_stack([lib_sim, max_sim, bk_tan, combined, rankoftop, cnt / max(1, len(akeys)),
                          np.abs(ppm), np.full(nc, lcand), np.log1p(cnt),
                          ecdf_tan, ecdf_comb, max_sim2, max_sim4, sum_sim2 / ovn, sum_sim4 / ovn,
                          np.log1p(n_hi), np.log1p(n_mid), np.log1p(n_lo),
                          np.floor(np.abs(ppm)), (lib_sim > 0).astype(np.float32)])
    assert Xf.shape[1] == NFEAT, (Xf.shape[1], NFEAT)
    y = np.zeros(nc, dtype=np.int8)
    if qkey is not None:
        qkidx = np.where(pool_keys[cand] == np.asarray(qkey))[0]
        if len(qkidx): y[qkidx[0]] = 1
    return Xf.astype(np.float32), y, target, cand, qkey

def query_features(L, key2pool, pool_fp, pool_keys, pool_mass, nmr, rep, r):
    k = str(L.ik[r]); target = float(L.nm[r])
    a, b = L.off[r], L.off[r + 1]
    qm, qp = L.cmz[a:b], L.cp[a:b]
    return features_for_query(qm, qp, target, rep, nmr, L.coff, L.cmz, L.cp, L.ik,
                              key2pool, pool_fp, pool_keys, pool_mass, qkey=k)

# ---------------- build reps ----------------
def build_reps(L):
    best = {}
    for r in range(L.n):
        k = L.ik[r]
        if k and np.isfinite(L.nm[r]) and k not in best: best[k] = r
    rep = np.array(sorted(best.values())); nmr = L.nm[rep]; o = np.argsort(nmr)
    return rep[o], nmr[o], best

# ---------------- main ----------------
def main():
    t0 = time.time()
    L = Lib(TRAIN, "inchikey14")
    print(f"[lib+clean] {L.n} spectra {time.time()-t0:.0f}s", flush=True)
    pool_fp, pool_mass, pool_keys = build_or_load_pool()
    print(f"[pool] {len(pool_mass):,} candidates {time.time()-t0:.0f}s", flush=True)
    key2pool = {k: i for i, k in enumerate(pool_keys.tolist())}
    rep, nmr, _ = build_reps(L)
    print(f"[rep] {len(rep)} reps {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(SEED)
    pool_key_set = set(pool_keys.tolist())
    t2 = pq.read_table(TRAIN, columns=["inchikey14"]); df = t2.to_pandas().drop_duplicates("inchikey14")
    reach = [k for k in df.inchikey14.values if k in pool_key_set]
    eval_keys = set(rng.choice(reach, size=min(N_HOLDOUT, len(reach)), replace=False).tolist())
    train_keys = [k for k in reach if k not in eval_keys][:N_TRAINQ]
    tkeys = set(train_keys)

    Xtr = []; ytr = []
    for r in rep:
        k = str(L.ik[r])
        if k not in tkeys: continue
        res = query_features(L, key2pool, pool_fp, pool_keys, pool_mass, nmr, rep, r)
        if res is None: continue
        Xf, y, _, _, _ = res
        Xtr.append(Xf); ytr.append(y)
    if not Xtr: raise SystemExit("no training rows")
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    print(f"[train] {Xtr.shape[0]} rows, {ytr.sum()} positives {time.time()-t0:.0f}s", flush=True)

    RANKERS = []
    for w1 in (0.5, 0.6):
        W = np.where(ytr == 1, w1, 1.0 - w1)
        for sd in (0, 1, 2, 3):
            m = HistGradientBoostingClassifier(random_state=sd, max_depth=6, max_iter=500,
                                               learning_rate=0.03, min_samples_leaf=80,
                                               l2_regularization=1.0)
            m.fit(Xtr, ytr, sample_weight=W)
            RANKERS.append(m)
    def proba(X): return np.mean([m.predict_proba(X)[:, 1] for m in RANKERS], axis=0)
    print(f"[fit] {len(RANKERS)} HistGB {time.time()-t0:.0f}s", flush=True)

    # ---- test inference ----
    QT = Lib(TEST, "molecule_id")
    trep_i, _, _ = build_reps(QT)
    nmols = len(trep_i)
    print(f"[test-reps] {nmols} molecules {time.time()-t0:.0f}s", flush=True)
    analog_args = dict(rep=rep, nmr=nmr, acoff=L.coff, acmz=L.cmz, acp=L.cp, aik=L.ik,
                       key2pool=key2pool, pool_fp=pool_fp, pool_keys=pool_keys, pool_mass=pool_mass)
    rows = []; missing = 0; n_feat = 0; no_cand = 0
    for i, r in enumerate(trep_i):
        qid = str(QT.ik[r]); qtarget = float(QT.nm[r])
        a, b = QT.off[r], QT.off[r + 1]
        qm, qp = QT.cmz[a:b], QT.cp[a:b]
        res = features_for_query(qm, qp, qtarget, **analog_args, qkey=qid)
        if res is None:
            if QT.off[r] == QT.off[r + 1]: no_cand += 1
            missing += 1; rows.append((qid, [])); continue
        Xf, y, _, cand, _ = res
        p = proba(Xf); order = np.argsort(-p)
        keys = [str(pool_keys[cand[j]]) for j in order[:TOPN]]
        n_feat += 1; rows.append((qid, keys))
        if (i + 1) % 100 == 0: print(f"  [{time.time()-t0:.0f}s] {i+1}/{nmols}", flush=True)

    # key -> SMILES map from train
    t3 = pq.read_table(TRAIN, columns=["inchikey14", "normalized_smiles"])
    kdf = t3.to_pandas().dropna(subset=["inchikey14"]).drop_duplicates("inchikey14", keep="first")
    key2smiles = dict(zip(kdf["inchikey14"].astype(str), kdf["normalized_smiles"].astype(str)))

    written = []; pad = []
    for qid, keys in rows:
        smis = []
        for k in keys:
            s = key2smiles.get(k)
            if s: smis.append(s)
        if not smis and pad: smis = [pad[0]] * TOPN
        elif len(smis) < TOPN and pad: smis = smis + [pad[0]] * (TOPN - len(smis))
        while len(smis) < TOPN: smis.append(pad[0] if pad else "CCO")
        if smis: pad = smis
        written.append((qid, ";".join(smis[:TOPN])))

    with open(OUT, "w") as f:
        f.write("molecule_id,smiles\n")
        for qid, smi in written: f.write(f"{qid},{smi}\n")
    print(f"[F-SUBMIT] wrote {OUT} ({len(written)} rows) | features={n_feat} no_features={missing} empty_spectra={no_cand} | {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()