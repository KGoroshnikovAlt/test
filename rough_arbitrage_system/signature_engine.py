"""
signature_engine.py — Lead-Lag transform and truncated path signature.

The path signature truncated at level L for a piecewise-linear path is computed
exactly from the per-segment increments using Chen's identity:

    Sig(X * Y) = Sig(X) ⊗ Sig(Y),

and on a single linear piece with increment Δ ∈ R^d:

    Sig_segment_k = Δ^{⊗ k} / k!     (tensor exponential).

We deliberately implement this from scratch (rather than depending on iisignature
or signatory) so the backtest has zero native build-time dependencies — the rough
path theory needed here is small (d=2, L=3) and exact.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from config import Config


# --------------------------------------------------------------------------- #
# Lead-Lag transformation                                                     #
# --------------------------------------------------------------------------- #
def lead_lag(z: Sequence[float]) -> np.ndarray:
    """Lead-Lag transform of a 1-D series.

    Returns a 2-D path of shape (2*n + 1, 2) for an input of length n + 1.
    Channel 0 is the lead, channel 1 is the lag. The construction alternates
    "lead advances" and "lag catches up" steps; this prevents the tree-like
    cancellation (Hambly–Lyons) that would otherwise zero out the signature
    of a 1-D oscillating spread.
    """
    z = np.asarray(z, dtype=float).ravel()
    n = len(z) - 1
    if n < 1:
        raise ValueError("lead_lag requires at least two points")
    out = np.empty((2 * n + 1, 2), dtype=float)
    out[0] = (z[0], z[0])
    for k in range(1, n + 1):
        out[2 * k - 1] = (z[k], z[k - 1])     # lead advances
        out[2 * k]     = (z[k], z[k])         # lag catches up
    return out


# --------------------------------------------------------------------------- #
# Truncated path signature                                                    #
# --------------------------------------------------------------------------- #
def _segment_exp(delta: np.ndarray, L: int) -> List[np.ndarray]:
    """Tensor exponential of a single increment: [1, Δ, Δ⊗Δ/2!, Δ⊗Δ⊗Δ/3!, ...]."""
    sig = [np.array(1.0)]
    current = np.array(1.0)
    factorial = 1
    for k in range(1, L + 1):
        if k == 1:
            current = delta.copy()
        else:
            current = np.multiply.outer(current, delta)
        factorial *= k
        sig.append(current / factorial)
    return sig


def _chen_product(a: List[np.ndarray], b: List[np.ndarray], L: int) -> List[np.ndarray]:
    """Concatenation product in the truncated tensor algebra T^L(R^d)."""
    out: List[np.ndarray] = []
    for k in range(L + 1):
        accum: np.ndarray | None = None
        for i in range(k + 1):
            j = k - i
            ai, bj = a[i], b[j]
            if i == 0:
                contrib = float(ai) * bj
            elif j == 0:
                contrib = ai * float(bj)
            else:
                contrib = np.multiply.outer(ai, bj)
            accum = contrib if accum is None else accum + contrib
        out.append(accum)
    return out


def path_signature(path: np.ndarray, L: int) -> List[np.ndarray]:
    """Truncated signature of a piecewise-linear path (n_points, d) up to order L."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be 2-D (n_points, d)")
    if path.shape[0] < 2:
        # constant path
        return [np.array(1.0)] + [np.zeros((path.shape[1],) * k) for k in range(1, L + 1)]

    sig = [np.array(1.0)] + [np.zeros((path.shape[1],) * k) for k in range(1, L + 1)]
    for t in range(path.shape[0] - 1):
        delta = path[t + 1] - path[t]
        seg = _segment_exp(delta, L)
        sig = _chen_product(sig, seg, L)
    return sig


def flatten_signature(sig: List[np.ndarray], include_constant: bool = False) -> np.ndarray:
    """Flatten the level-wise signature into a 1-D vector (no level-0 by default)."""
    start = 0 if include_constant else 1
    parts = [sig[k].ravel() for k in range(start, len(sig))]
    return np.concatenate(parts) if parts else np.zeros(0)


def signature_dim(d: int, L: int, include_constant: bool = False) -> int:
    """Length of flatten_signature for dimension d and truncation L."""
    total = sum(d ** k for k in range(0, L + 1))
    return total if include_constant else total - 1


# --------------------------------------------------------------------------- #
# Convenience: full pipeline on a 1-D spread window                           #
# --------------------------------------------------------------------------- #
def spread_signature(z_window: Sequence[float], cfg: Config) -> np.ndarray:
    """Lead-Lag + signature, returned as a flat numpy vector of length 14 (d=2, L=3)."""
    ll = lead_lag(z_window)
    sig = path_signature(ll, cfg.SIGNATURE_ORDER)
    return flatten_signature(sig, include_constant=False)
