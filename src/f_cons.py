"""F-CONS — consensus clustering over analog candidates (CASMI26 Phase 2).

Aggregates the per-query candidate window by structural similarity (Tanimoto on
the pool fingerprints) so that near-identical candidates do not over-populate
the top-k. Ranking is read off cluster representatives (at most one per tight
cluster), then mapped back to the candidate set. Reuses the packed Tanimoto
bytes code from f_ranker; no data/ pool mutation, monotone, reproducible.
"""
from __future__ import annotations
import numpy as np
from f_ranker import packed_tanimoto  # reuse the repo's Tanimoto bytes code


def cluster_candidates(pool_fp, cand_indices, threshold=0.7):
    """Union-find consensus clustering of the candidate window.

    pool_fp      : packed fp bytes array for the whole pool (len P).
    cand_indices : int array of pool indices forming the candidate window.
    threshold    : Tanimoto >= threshold connects two candidates (tight cluster).

    Returns (cluster_ids, tan_matrix):
      cluster_ids : int array len == len(cand_indices), a component id per
                    candidate (0..k-1, arbitrary but deterministic order).
      tan_matrix  : (k, k) float Tanimoto between candidates (for inspection).
    """
    idx = np.asarray(cand_indices, dtype=np.int64)
    k = len(idx)
    cf = np.ascontiguousarray(pool_fp[idx], np.uint8)  # (k, nbytes)
    # fully-vectorized pairwise Tanimoto reusing packed_tanimoto row-wise
    tan = np.zeros((k, k), np.float32)
    for i in range(k):
        tan[i] = packed_tanimoto(cf[i], cf)  # (k,)
    tan = (tan + tan.T) * 0.5  # symmeterize (input is symmetric by construction)
    np.fill_diagonal(tan, 1.0)
    # union-find over tan >= threshold
    parent = np.arange(k)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(k):
        for j in range(i + 1, k):
            if tan[i, j] >= threshold:
                union(i, j)
    roots = np.array([find(i) for i in range(k)])
    # relabel to compact 0..C-1 by first appearance (deterministic)
    seen = {}
    n = 0
    for r in roots:
        if r not in seen:
            seen[r] = n
            n += 1
    cid = np.array([seen[r] for r in roots])
    return cid, tan


def diverse_rank(scores, cluster_ids, topk=25):
    """Greedy diversity-preserving rank: at most one candidate per tight cluster.

    scores       : float array len k, higher = better (already model proba).
    cluster_ids  : int array len k (from cluster_candidates).
    topk         : number of positions to keep (default 25).

    Returns a list (len <= topk) of candidate positions [0..k) in final rank
    order. Walking score-descending, we accept a candidate only when its
    cluster has no accepted member yet; stop when topk accepted. Every
    accepted candidate is the top-scoring representative of its cluster.
    """
    order = np.argsort(-np.asarray(scores, np.float64), kind="stable")
    chosen = []
    taken = set()
    for pos in order:
        c = int(cluster_ids[pos])
        if c in taken:
            continue
        taken.add(c)
        chosen.append(int(pos))
        if len(chosen) >= topk:
            break
    return chosen