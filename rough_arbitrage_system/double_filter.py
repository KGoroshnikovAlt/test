"""
double_filter.py — geometric admissibility test on the rough path.

Two second-order invariants gate every entry:

  * Segmented Levy area  SS_{s,t} = (A(0,t1) + A(t1,t2) + ... + A(t_{K-1}, T)) / K
    where A(u,v) = 0.5 * (∫ X dY - ∫ Y dX) on [u, v].
    Captures who is *leading* the spread (Nasdaq vs S&P 500).

  * Increment covariance  C_{s,t} = sum_k (ΔX_k * ΔY_k)
    Captures co-movement; turning sharply negative breaks the arbitrage premise.

Both quantities are turned into rolling z-scores; entries are blocked when either
z-score leaves ±cfg.LEVY_Z_THRESHOLD / ±cfg.COVAR_Z_THRESHOLD.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from config import Config


def levy_area(path: np.ndarray) -> float:
    """Signed Levy area of a 2-D piecewise-linear path.

    A = 0.5 * sum_t (X_t * dY_t - Y_t * dX_t),
    using the trapezoidal mid-point of the segment.
    """
    p = np.asarray(path, dtype=float)
    if p.shape[0] < 2 or p.shape[1] != 2:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)
    mid_x = 0.5 * (x[:-1] + x[1:])
    mid_y = 0.5 * (y[:-1] + y[1:])
    return 0.5 * float(np.sum(mid_x * dy - mid_y * dx))


def segmented_levy_area(path: np.ndarray, K: int) -> float:
    """Average signed Levy area across K equal-length sub-paths."""
    p = np.asarray(path, dtype=float)
    n = p.shape[0]
    if n < K + 1:
        return levy_area(p)
    edges = np.linspace(0, n - 1, K + 1, dtype=int)
    areas = []
    for i in range(K):
        seg = p[edges[i]: edges[i + 1] + 1]
        if seg.shape[0] >= 2:
            areas.append(levy_area(seg))
    return float(np.mean(areas)) if areas else 0.0


def increment_covariance(path: np.ndarray) -> float:
    """Sum-of-products of channel increments — empirical quadratic covariation."""
    p = np.asarray(path, dtype=float)
    if p.shape[0] < 2 or p.shape[1] != 2:
        return 0.0
    dx = np.diff(p[:, 0])
    dy = np.diff(p[:, 1])
    return float(np.sum(dx * dy))


@dataclass
class FilterState:
    """Rolling-mean / rolling-std tracker for z-scoring SS and C online."""
    window: int = 500
    _ss: deque = None
    _cv: deque = None

    def __post_init__(self):
        self._ss = deque(maxlen=self.window)
        self._cv = deque(maxlen=self.window)

    def update(self, ss: float, cv: float) -> None:
        self._ss.append(float(ss))
        self._cv.append(float(cv))

    def _z(self, buf: deque, x: float) -> float:
        if len(buf) < 30:
            return 0.0
        arr = np.fromiter(buf, dtype=float)
        sd = float(arr.std(ddof=1))
        if sd < 1e-12:
            return 0.0
        return (x - float(arr.mean())) / sd

    def z_levy(self, ss: float) -> float:
        return self._z(self._ss, ss)

    def z_covar(self, cv: float) -> float:
        return self._z(self._cv, cv)


def admissible(z_levy: float, z_covar: float, cfg: Config) -> bool:
    """True iff both geometric invariants are within their statistical bands."""
    if abs(z_levy) > cfg.LEVY_Z_THRESHOLD:
        return False
    if z_covar < -cfg.COVAR_Z_THRESHOLD:
        # only a sharp negative co-movement breaks the arbitrage model
        return False
    return True
