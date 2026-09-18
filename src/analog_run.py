"""F-ANALOG for CASMI26 — mass-shifted analog propagation, single channel.

Reproduces the winner's analog single-channel backbone (target Class-2 MRR ~0.50):
    score(c) = max over analogs a:  sim(a)^PPM_POW * Tanimoto(fp_c, fp_a)

Analog = library structure whose spectrum, shifted by (query_neutral - analog_neutral),
entropy-similarity-matches the query. Pools: COCONUT (data/coco_fp.npy bitpacked)
candidates ranked by analog evidence.

Held-out: pick natural-product-like molecules, REMOVE all their spectra from the
library, query from remaining mass-window, MRR@25 by inchikey14 (fair Class-2).

Usage: .venv/bin/python src/analog_run.py
"""
from __future__ import annotations
import os, sys, time, pickle
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import analog as A  # noqa: E402

TRAIN = os.path.join(ROOT, "train.parquet")
DATA = os.path.join(ROOT, "data")

N_HOLDOUT = 250
SEED = 11
TOPN = 25

def main():
    t0 = time.time()
    # library
    L = A.load_library(TRAIN)
    print(f"[lib] {L['n']} spectra, {time.time()-t0:.0f}s", flush=True)
    # pool
    pool = A.load_pool(DATA)
    print(f"[pool] {len(pool['mass'])} structures, {time.time()-t0:.0f}s", flush=True)
    # representatives (one best spectra per structure)
    rep, rep_key, rep_nm, rep_fp = A.build_rep(L, pool)
    print(f"[rep] {len(rep)} representatives, {time.time()-t0:.0f}s", flush=True)

    # fair Class-2 holdout: pick molecules, REMOVE their spectra from library
    rng = np.random.default_rng(SEED)
    ik = np.asarray(L["ik"], dtype=object)
    uniq = np.unique(ik)
    chosen = set(rng.choice(uniq, size=N_HOLDOUT, replace=False).tolist())
    idx_valid = np.array([i for i in range(L["n"]) if ik[i] not in chosen])
    print(f"[holdout] {N_HOLDOUT} molecules, {len(idx_valid)} kept spectra", flush=True)

    mrr = 0.0; hit = 0; queries = 0
    seen = set()
    order = np.argsort(np.where(L["nm_valid"], L["nm"], 1e18))
    for i in order:
        k = ik[i]
        if k not in chosen or k in seen:
            continue
        seen.add(k)
        target = float(L["nm"][i])
        scores = A.analog_rank(L, pool, rep, rep_key, rep_nm, rep_fp, i, target, idx_valid)
        ranked_keys = [k2 for k2, _ in scores]
        rank = None
        for pos, kk in enumerate(ranked_keys[:TOPN], 1):
            if kk == k:
                rank = pos; break
        if rank:
            mrr += 1.0 / rank; hit += 1
        queries += 1
    mrr /= max(1, queries); h25 = hit / max(1, queries)
    out = f"F-ANALOG single-channel | MRR@25 = {mrr:.4f} | hit@25 = {h25:.4f} | n={queries} | target>=0.45\n"
    print("="*60); print(out); print("="*60)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "fanalog_measured.txt"), "w").write(out)

if __name__ == "__main__":
    main()