"""CASMI26 F-BASE runner.

Holds out 400 train molecules (seed 0), queries each held-out spectrum
against the built train library index inside a tight neutral-mass window
(+-8.5 ppm on NEUTRAL mass), cosine-ranks candidate spectra, and reports
MRR@25 and hit@25 by inchikey14.

Usage:  .venv/bin/python src/run.py
"""
from __future__ import annotations

import os
import time

import numpy as np

import lib

TRAIN = "train.parquet"
IDX_CACHE = "src/index_cache.npz"
N_HOLDOUT = 400
SEED = 0
PPM = 8.5
K = 25
# EXCLUDE_MODE:
#   'self'  -> exclude only the query's own row (keep same-molecule spectra as valid targets)
#   'twins' -> also exclude spectra sharing inchikey14 + rounded precursor (identical siblings)
EXCLUDE_MODE = os.environ.get("EXCLUDE_MODE", "self")


def build_holdout():
    import pyarrow.parquet as pq

    rng = np.random.default_rng(SEED)
    f = pq.ParquetFile(TRAIN)
    ik_all = []
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=["inchikey14"])
        ik_all.append(t.column("inchikey14").to_numpy(zero_copy_only=False))
    ik_all = np.concatenate(ik_all)
    uniq = np.unique(ik_all)
    chosen = rng.choice(uniq, size=N_HOLDOUT, replace=False)
    return set(chosen.tolist())


def main():
    t0 = time.time()
    idx = None
    if os.path.exists(IDX_CACHE):
        idx = lib.SpectralIndex.load(IDX_CACHE)
        print(f"[F-BASE] loaded cached index: {idx.n_spec} spectra, "
              f"nnz={idx.indices.size} ({t0 - 0:.0f}s)")
    else:
        print(f"[F-BASE] building index from {TRAIN} ...")
        idx = lib.load_and_build_index(TRAIN)
        print(f"[F-BASE] index built: {idx.n_spec} spectra, "
              f"nnz={idx.indices.size}, {time.time()-t0:.1f}s")
        idx.save(IDX_CACHE)
        print(f"[F-BASE] index cached -> {IDX_CACHE}")

    holdout = build_holdout()
    rng = np.random.default_rng(SEED)
    seen = set()
    queries = []
    for r, ik in enumerate(idx.inchikey14):
        if ik in holdout and ik not in seen:
            seen.add(ik)
            queries.append((r, ik))
        if len(queries) == N_HOLDOUT:
            break

    print(f"[F-BASE] {len(queries)} holdout queries (exclude_mode={EXCLUDE_MODE}), querying ...")

    mrr_sum = 0.0
    hit = 0
    n_done = 0
    tq0 = time.time()
    for r, qik in queries:
        qb = idx.indices[idx.indptr[r]:idx.indptr[r + 1]]
        qv = idx.values[idx.indptr[r]:idx.indptr[r + 1]]
        if qb.size == 0:
            n_done += 1
            continue
        neutral = idx.neutral_mass[r]
        cand = idx.candidates_in_window(neutral, PPM)
        if EXCLUDE_MODE == "twins":
            q_prec = np.round(idx.neutral_mass[r] + lib.PROTON, 2)
            cand = [
                c for c in cand
                if c != r and not (
                    idx.inchikey14[c] == qik
                    and np.round(idx.neutral_mass[c] + lib.PROTON, 2) == q_prec
                )
            ]
        else:  # 'self'
            cand = [c for c in cand if c != r]
        scored = idx.rank_spectra(qb, qv, cand, k=K)
        rank = None
        for pos, (ik_c, sc) in enumerate(scored[:K], start=1):
            if ik_c == qik:
                rank = pos
                break
        if rank is not None:
            mrr_sum += 1.0 / rank
            hit += 1
        n_done += 1

    mrr = mrr_sum / n_done
    h25 = hit / n_done
    print("=" * 56)
    print(f"[F-BASE] holdout  | MRR@25 = {mrr:.4f} | hit@25 = {h25:.4f} "
          f"({100*h25:.2f}%)")
    print(f"[F-BASE] gate     | target >= 0.108  | {'PASS' if mrr >= 0.108 else 'FAIL'}")
    print("=" * 56)
    with open(f"docs/fbase_measured_{EXCLUDE_MODE}.txt", "w") as fh:
        fh.write(f"mode={EXCLUDE_MODE}\nMRR@25={mrr:.6f}\nhit@25={h25:.6f}\n"
                 f"n_queries={n_done}\nbuild_time_s={t0}\nquery_time_s={time.time()-tq0:.1f}\n")
    return mrr


if __name__ == "__main__":
    main()