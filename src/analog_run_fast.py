"""F-ANALOG fast runner — Numba entropy + precomputed cleaning + partial holdout + progress."""
from __future__ import annotations
import os, sys, time, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import analog_accel as A

TRAIN = os.path.join(ROOT, "train.parquet")
DATA = os.path.join(ROOT, "data")
N_HOLDOUT = 60
SEED = 11
TOPN = 25
PPM_WIN = 10.0
ANALOG_WIN = 200.0
N_ANALOG = 100
SIM_POWER = 4.0


def load_pool(data_dir):
    cf = np.load(os.path.join(data_dir, "coco_fp.npy"), mmap_mode="r")
    cm = pickle.load(open(os.path.join(data_dir, "coco_meta.pkl"), "rb"))
    mass = np.load(os.path.join(data_dir, "coco_mass.npy"))
    return dict(fp=cf, mass=np.asarray(mass, np.float64),
                keys=np.asarray(cm["keys"], dtype=object))


def packed_tanimoto(a, b):
    """b: (nb, 867) uint8 packed. bitmap AND/OR popcount via uint64."""
    ap = np.pad(a, (0, (-len(a)) % 8)).astype(np.uint8)
    bp = np.pad(b, ((0, 0), (0, (-b.shape[1]) % 8))).astype(np.uint8)
    a64 = np.frombuffer(ap.tobytes(), dtype=np.uint64)
    b64 = np.frombuffer(bp.tobytes(), dtype=np.uint64).reshape(b.shape[0], -1)
    inter = np.sum(np.bitwise_and(a64[None, :], b64).view(np.uint8).astype(np.int64), axis=1)
    na = int(np.count_nonzero(ap))
    nb = np.count_nonzero(bp, axis=1)
    union = na + nb - inter
    return np.where(union > 0, inter / union, 0.0)


def main():
    t0 = time.time()
    L = A.Lib(TRAIN)
    print(f"[lib+clean] {L.n} spectra cleaned {time.time()-t0:.0f}s", flush=True)
    pool = load_pool(DATA)
    print(f"[pool] {len(pool['mass'])} structures {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()

    rng = np.random.default_rng(SEED)
    # valid holdout = train molecules that EXIST in the COCONUT candidate pool
    # (else the correct answer is never among candidates -> trivially MRR 0).
    pool_key_set = set(pool["keys"].tolist())
    cand_mols = [k for k in np.unique(L.ik) if k in pool_key_set]
    chosen = set(rng.choice(cand_mols, size=min(N_HOLDOUT, len(cand_mols)), replace=False).tolist())
    print(f"[holdout] {len(cand_mols)} pool-reachable molecules, hold out {len(chosen)}", flush=True)
    mask_valid = np.isfinite(L.nm)
    nkept = int(np.sum(mask_valid))

    # reps (one cleaned spectrum per structure), sorted by neutral mass
    best = {}
    nz = np.diff(L.off)
    for r in range(L.n):
        k = L.ik[r]
        if k and k not in best and mask_valid[r]:
            best[k] = r
    rep = np.array(sorted(best.values()))
    nmr = L.nm[rep]; orr = np.argsort(nmr)
    rep = rep[orr]; nmr = nmr[orr]
    print(f"[rep] {len(rep)} representatives {time.time()-t0:.0f}s", flush=True)

    mrr = 0.0; hit = 0; q = 0
    seen = set()
    for qi, r in enumerate(rep):
        k = L.ik[r]
        if k not in chosen or k in seen: continue
        seen.add(k); q += 1
        target = float(nmr[qi] if False else L.nm[r])
        # cleaned query
        a, b = L.off[r], L.off[r + 1]
        qm = L.cmz[a:b]; qp = L.cp[a:b]
        # analog reps in mass window
        lo = np.searchsorted(nmr, target - ANALOG_WIN, "left")
        hi = np.searchsorted(nmr, target + ANALOG_WIN, "right")
        agg = {}
        for j in range(lo, hi):
            ra = rep[j]; shift = target - float(nmr[j])
            ca, cb = L.off[ra], L.off[ra + 1]
            cm = L.cmz[ca:cb]; cp = L.cp[ca:cb]
            if len(cm) == 0: continue
            s = A._mass_shift_sim(qm, qp, cm, cp, A.MZ_TOL, np.float32(shift))
            ka = L.ik[ra]
            if s > agg.get(ka, -1.0): agg[ka] = s
        analogs = sorted(agg.items(), key=lambda x: -x[1])[:N_ANALOG]
        # candidate pool window
        half = PPM_WIN * 1e-6 * target
        cand = np.where(np.abs(pool["mass"] - target) <= half)[0]
        if len(cand) == 0: continue
        cf = pool["fp"][cand].astype(np.uint8)
        best_score = np.full(len(cand), 0.0)
        for ak, s in analogs:
            m = np.where(pool["keys"] == ak)[0]
            if len(m) == 0: continue
            t = packed_tanimoto(pool["fp"][m[0]].astype(np.uint8), cf)
            best_score = np.maximum(best_score, float(s) ** SIM_POWER * t)
        order = np.argsort(-best_score)
        rank = None
        for pos in range(min(TOPN, len(order))):
            if pool["keys"][cand[order[pos]]] == k: rank = pos + 1; break
        if rank:
            mrr += 1.0 / rank; hit += 1
        if q % 10 == 0:
            print(f"  [{time.time()-t0:6.0f}s] q={q}/{len(seen)} mr_sofar={mrr/(q):.4f} hit={hit}/{q}", flush=True)
    mrr /= max(1, q); h25 = hit / max(1, q)
    print("=" * 56)
    print(f"[F-ANALOG] fast, Numba | MRR@25 = {mrr:.4f} | hit@25 = {h25:.4f} | n={q} | target>=0.45 (winner ~0.52)")
    print("=" * 56)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "fanalog_measured.txt"), "w").write(
        f"mode=numba-fast\nMRR@25={mrr:.6f}\nhit@25={h25:.6f}\nn_queries={q}\n")


if __name__ == "__main__":
    main()