"""CASMI26 F-BASE library module.

Loads train.parquet spectra, applies entropy-weighted peak cleaning,
and builds a fast numpy/sparse retrievable index over 0.1 Da bin vectors.

Recipe (F-BASE, measured Ch1 floor 0.108):
  - floor intensity 0.002
  - top 256 peaks by intensity
  - +200 Da window around precursor m/z (keep peaks <= precursor_mz + 200)
  - entropy sharpening (Li & Fiehn, SpectralEntropy: power = 0.25 + 0.25*S when S<3)
  - neutral mass = precursor_mz - 1.00728
  - 0.1 Da m/z bin vectors, L2-normalized
  - retrieval: tight neutral-mass window (ppm +-8.5 on NEUTRAL mass), cosine rank
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

PROTON = 1.00728
BIN_W = 0.1
MZ_MAX = 2000.0
N_BINS = int(MZ_MAX / BIN_W) + 1  # 20001
FLOOR = 0.002
TOP_PEAKS = 256
PRECURSOR_WINDOW = 200.0


def _entropy_weights(intensities: np.ndarray) -> np.ndarray:
    """Li & Fiehn spectral-entropy intensity weighting."""
    if intensities.size == 0:
        return intensities
    p = intensities.astype(np.float64)
    p = p / p.sum()
    s = float(-np.sum(p * np.log(p)))
    if s < 3.0:
        w = 0.25 + 0.25 * s
        out = np.power(p, w)
        out = out / out.sum()
        return out
    return p


def clean_spectrum(mz, intensity, precursor_mz):
    """Return (bin_indices, values) for a cleaned spectrum, or (None, None)."""
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.size == 0:
        return None, None
    # +200 Da window around precursor: keep fragment peaks up to precursor+200
    keep = (mz >= 0.0) & (mz <= precursor_mz + PRECURSOR_WINDOW)
    mz = mz[keep]
    intensity = intensity[keep]
    if mz.size == 0:
        return None, None
    # floor
    keep = intensity >= FLOOR
    mz = mz[keep]
    intensity = intensity[keep]
    if mz.size == 0:
        return None, None
    # top 256
    if mz.size > TOP_PEAKS:
        idx = np.argsort(intensity)[::-1][:TOP_PEAKS]
        mz = mz[idx]
        intensity = intensity[idx]
    # entropy sharpening
    intensity = _entropy_weights(intensity)
    # 0.1 Da bin
    bins = np.floor(mz / BIN_W).astype(np.int64)
    # merge duplicate bins by summation
    order = np.argsort(bins)
    bins = bins[order]
    intensity = intensity[order]
    uniq, first = np.unique(bins, return_index=True)
    # group sums
    boundaries = np.append(first, len(bins))
    sums = np.add.reduceat(intensity, boundaries[:-1])
    return uniq, sums


class SpectralIndex:
    """CSR-like sparse library index over 0.1 Da bin vectors."""

    def __init__(self):
        self.inchikey14 = []          # per spectrum
        self.neutral_mass = []        # per spectrum (float64)
        self.indptr = [0]             # len = n_spec + 1
        self.indices = []             # bin indices
        self.values = []              # weights
        self.n_spec = 0

    def add(self, inchikey14, neutral_mass, bins, values):
        if bins is None:
            bins = np.zeros(0, dtype=np.int64)
            values = np.zeros(0, dtype=np.float64)
        # L2 normalize
        n = np.sqrt(np.dot(values, values))
        if n > 0:
            values = values / n
        self.inchikey14.append(inchikey14)
        self.neutral_mass.append(neutral_mass)
        self.indices.append(bins.astype(np.int32))
        self.values.append(values.astype(np.float32))
        self.indptr.append(self.indptr[-1] + len(bins))
        self.n_spec += 1

    def finalize(self):
        self.neutral_mass = np.asarray(self.neutral_mass, dtype=np.float64)
        self.indptr = np.asarray(self.indptr, dtype=np.int64)
        self.indices = np.concatenate(self.indices).astype(np.int32)
        self.values = np.concatenate(self.values).astype(np.float32)
        # sort library by neutral mass for window retrieval
        self.order = np.argsort(self.neutral_mass, kind="stable")
        self.sorted_mass = self.neutral_mass[self.order]
        return self

    def save(self, path):
        np.savez_compressed(
            path,
            inchikey14=np.array(self.inchikey14, dtype=object),
            neutral_mass=self.neutral_mass,
            indptr=self.indptr,
            indices=self.indices,
            values=self.values,
            order=self.order,
            sorted_mass=self.sorted_mass,
        )

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        idx = cls()
        idx.inchikey14 = list(z["inchikey14"])
        idx.neutral_mass = z["neutral_mass"]
        idx.indptr = z["indptr"]
        idx.indices = z["indices"]
        idx.values = z["values"]
        idx.order = z["order"]
        idx.sorted_mass = z["sorted_mass"]
        idx.n_spec = len(idx.inchikey14)
        return idx

    def candidates_in_window(self, neutral_mass, ppm):
        half = ppm * 1e-6 * neutral_mass
        lo = neutral_mass - half
        hi = neutral_mass + half
        i = int(np.searchsorted(self.sorted_mass, lo, side="left"))
        j = int(np.searchsorted(self.sorted_mass, hi, side="right"))
        return self.order[i:j]

    def rank_spectra(self, query_bins, query_values, cand_rows, k=25):
        """Cosine rank candidate spectra. Return list of (inchikey14, score)."""
        qv = query_values.astype(np.float32)
        qn = float(np.sqrt(np.dot(qv, qv)))
        if qn == 0:
            return []
        qv = qv / qn
        # map query bins -> value for fast lookup
        qmap = np.zeros(N_BINS, dtype=np.float32)
        qmap[query_bins] = qv
        scored = []
        for r in cand_rows:
            s, e = self.indptr[r], self.indptr[r + 1]
            if s == e:
                continue
            bins = self.indices[s:e]
            vals = self.values[s:e]
            score = float(np.dot(vals, qmap[bins]))
            scored.append((self.inchikey14[r], score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


def load_and_build_index(
    parquet_path: str,
    chunk_cols=("inchikey14", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"),
) -> SpectralIndex:
    f = pq.ParquetFile(parquet_path)
    idx = SpectralIndex()
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=list(chunk_cols))
        ik = t.column("inchikey14").to_numpy(zero_copy_only=False)
        pm = t.column("precursor_mz").to_numpy(zero_copy_only=False)
        mzs = t.column("ms2_mzs").to_pylist()
        ints = t.column("ms2_normalized_intensities").to_pylist()
        for i in range(len(ik)):
            neutral = pm[i] - PROTON
            b, v = clean_spectrum(mzs[i], ints[i], pm[i])
            idx.add(ik[i], neutral, b, v)
    return idx.finalize()
