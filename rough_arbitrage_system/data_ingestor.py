"""
data_ingestor.py — historical / synthetic minute bars for backtest.

In production this module wraps MetaTrader5 or cTrader. For backtest mode it
either loads CSV files (columns: timestamp, ndx, spx) or generates a synthetic
cointegrated pair using a slow random walk plus a mean-reverting Ornstein–
Uhlenbeck spread. The synthetic generator is deterministic given a seed and is
sufficient to exercise the full pipeline end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BarSeries:
    timestamps: np.ndarray     # int64 minute index
    ndx: np.ndarray            # float, #USNDAQ100 mid price
    spx: np.ndarray            # float, #USSPX500 mid price

    def __len__(self) -> int:
        return len(self.timestamps)


def load_csv(path: str | Path) -> BarSeries:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("timestamp") or cols.get("time") or cols.get("date")
    ndx_col = cols.get("ndx") or cols.get("#usndaq100") or cols.get("nasdaq")
    spx_col = cols.get("spx") or cols.get("#usspx500") or cols.get("sp500")
    if ts_col is None or ndx_col is None or spx_col is None:
        raise ValueError("CSV must contain timestamp, ndx and spx columns")
    ts = pd.to_datetime(df[ts_col]).astype("int64").to_numpy() // 60_000_000_000
    return BarSeries(
        timestamps=ts,
        ndx=df[ndx_col].to_numpy(dtype=float),
        spx=df[spx_col].to_numpy(dtype=float),
    )


def generate_synthetic(
    n_minutes: int = 20_000,
    seed: int = 7,
    ndx_start: float = 24_000.0,
    spx_start: float = 6_000.0,
    drift: float = 0.0,
    vol: float = 8.0,                 # NDX log-return scale per minute (in points)
    beta_true: float = 0.25,          # true hedge ratio NDX -> SPX
    spread_kappa: float = 0.05,       # OU mean reversion
    spread_sigma: float = 4.0,        # OU shock
    regime_shift_at: Optional[int] = None,
) -> BarSeries:
    """Synthetic NDX/SPX minute bars with a stationary spread.

    The construction:
      d log NDX_t = drift + vol * dW1_t
      log SPX_t   = log(spx_start) + beta_true * (log NDX_t - log ndx_start) + ε_t / spx_start
      ε_t follows an OU process so the resulting spread is mean reverting.
    """
    rng = np.random.default_rng(seed)
    dW = rng.standard_normal(n_minutes) * vol + drift
    ndx_log = np.log(ndx_start) + np.cumsum(dW / ndx_start)
    ndx = np.exp(ndx_log)

    eps = np.empty(n_minutes)
    eps[0] = 0.0
    for t in range(1, n_minutes):
        eps[t] = eps[t - 1] - spread_kappa * eps[t - 1] + spread_sigma * rng.standard_normal()
        if regime_shift_at is not None and t == regime_shift_at:
            eps[t] += 200.0  # injected dislocation
    spx_log = np.log(spx_start) + beta_true * (ndx_log - np.log(ndx_start)) + eps / spx_start
    spx = np.exp(spx_log)

    ts = np.arange(n_minutes, dtype=np.int64)
    return BarSeries(timestamps=ts, ndx=ndx, spx=spx)
