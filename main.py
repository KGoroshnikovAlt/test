"""
main.py
=======
Сквозной бэктест системы статистического арбитража на грубых путях.

Структурные улучшения относительно базовой версии:
    * Калман работает по логарифмам цен — это даёт стационарную пару (на
      сырых ценах NQ/ES спред — белый шум, AR(1) ≈ 0).
    * Rolling-z нормировка спреда устраняет эффект медленных режимных сдвигов
      и адаптируется к смене волатильности (в отличие от фиксированной σ
      обучающей выборки).
    * Тренировка LS-решателя на «условных» путях, чьё стартовое Z ∈ хвосте
      распределения. Без этого регрессия учит безусловное E[V] ≈ 0 и всегда
      советует ждать до горизонта.
    * Per-trade stop-loss по |ΔZ| ≥ STOP_LOSS_Z·σ ограничивает разовый убыток.

Pipeline на минуту:
    1. Калман по log-ценам: β_t, α_t, Z_t = log y − β_t log x − α_t.
    2. Rolling μ_t, σ_t спреда (окно ROLLING_ZSTD_WINDOW). z_norm = (Z−μ)/σ.
    3. Lead-Lag(окно Z) → 2D-путь → двойной геометрический фильтр.
    4. Если позиция закрыта: |z_norm| > ENTRY_Z_THRESHOLD ⇒ вход.
    5. Если позиция открыта: либо stop_optimal (payoff > continuation), либо
       stop_loss (|ΔZ| > STOP_LOSS_Z σ), либо horizon.
    6. Risk manager: маржа, лоты по β, drawdown floor.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import CFG
from data_ingestor import MarketBars, load_bars
from kalman_filter import DynamicHedgeKalman
from signature_engine import lead_lag
from double_filter import DoubleGeometricFilter
from solver_ls import LongstaffSchwartzSignature
from risk_manager import RiskManager, Side


@dataclass
class Trade:
    side: Side
    entry_bar: int
    exit_bar: int
    entry_z: float
    exit_z: float
    pnl: float
    reason: str


@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    spread: np.ndarray
    beta: np.ndarray
    trades: List[Trade] = field(default_factory=list)
    halted: bool = False

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve[-1])

    def summary(self) -> dict:
        n = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl > 0)
        gross = sum(t.pnl for t in self.trades)
        durations = [t.exit_bar - t.entry_bar for t in self.trades]
        from collections import Counter
        reasons = Counter(t.reason for t in self.trades)
        return {
            "n_trades": n,
            "wins": wins,
            "win_rate": (wins / n) if n else 0.0,
            "gross_pnl": gross,
            "avg_pnl": (gross / n) if n else 0.0,
            "avg_duration_min": float(np.mean(durations)) if durations else 0.0,
            "final_equity": self.final_equity,
            "max_equity": float(self.equity_curve.max()) if self.equity_curve.size else 0.0,
            "min_equity": float(self.equity_curve.min()) if self.equity_curve.size else 0.0,
            "halted_by_drawdown": self.halted,
            "close_reasons": dict(reasons),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_mean_std(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Скользящие μ, σ по окну. На первых `window` точках возвращает накопленные
    значения (expanding), затем — скользящие. Используется для нормировки Z.
    """
    n = x.size
    mu = np.empty(n)
    sd = np.empty(n)
    csum = np.cumsum(x)
    csum2 = np.cumsum(x * x)
    for t in range(n):
        if t < window:
            k = t + 1
            m = csum[t] / k
            v = csum2[t] / k - m * m
        else:
            s1 = csum[t] - csum[t - window]
            s2 = csum2[t] - csum2[t - window]
            m = s1 / window
            v = s2 / window - m * m
        mu[t] = m
        sd[t] = np.sqrt(max(v, 1e-24))
    return mu, sd


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    bars: Optional[MarketBars] = None,
    train_fraction: float = 0.4,
    entry_z_threshold: float = CFG.ENTRY_Z_THRESHOLD,
    stop_loss_z: float = CFG.STOP_LOSS_Z,
    use_log_prices: bool = CFG.USE_LOG_PRICES,
    rolling_window: int = CFG.ROLLING_ZSTD_WINDOW,
    horizon: int = CFG.LS_HORIZON,
    enable_geo_filter: bool = True,
    verbose: bool = True,
) -> BacktestResult:
    if bars is None:
        bars = load_bars(seed=CFG.RANDOM_SEED)
    y = bars.price_y
    x = bars.price_x
    n_total = len(bars)
    n_train = int(train_fraction * n_total)
    if verbose:
        print(f"[backtest] bars={n_total}, train={n_train}, test={n_total - n_train}")

    # ------------------------------------------------------------ 1. Калман
    obs_y = np.log(y) if use_log_prices else y
    obs_x = np.log(x) if use_log_prices else x
    kalman = DynamicHedgeKalman()
    z_full = np.empty(n_total)
    beta_full = np.empty(n_total)
    for t in range(n_total):
        z_full[t] = kalman.step(float(obs_x[t]), float(obs_y[t]))
        beta_full[t] = kalman.beta

    # ------------------------------------------------------------ 2. Rolling нормировка
    mu_roll, sd_roll = _rolling_mean_std(z_full, rolling_window)
    z_norm_full = (z_full - mu_roll) / sd_roll

    if verbose:
        print(f"[backtest] Z std (train): {z_full[:n_train].std():.6f}, "
              f"AR(1)={float(np.corrcoef(z_full[:n_train-1], z_full[1:n_train])[0,1]):.3f}")
        # доля времени, когда |z_norm| > threshold (в обучающей части)
        zn_tr = z_norm_full[:n_train]
        frac_extreme = float(np.mean(np.abs(zn_tr) > entry_z_threshold))
        print(f"[backtest] fraction |z_norm|>{entry_z_threshold} in train: {frac_extreme:.4f}")

    # ------------------------------------------------------------ 3. Обучение LS на условных путях
    z_train = z_full[:n_train]
    z_norm_train = z_norm_full[:n_train]

    # все возможные стартовые позиции
    n_starts = z_train.size - horizon
    valid_starts = np.arange(n_starts)

    # фильтр: для "short"-решателя — Z_0 в верхнем хвосте; для "long" — в нижнем
    long_mask = z_norm_train[valid_starts] < -entry_z_threshold
    short_mask = z_norm_train[valid_starts] > entry_z_threshold

    long_starts = valid_starts[long_mask]
    short_starts = valid_starts[short_mask]
    if verbose:
        print(f"[backtest] conditional starts: long={long_starts.size}, short={short_starts.size}")

    def _windows(starts: np.ndarray) -> np.ndarray:
        if starts.size == 0:
            # запасной вариант — без фильтра, чтобы не падать
            starts = valid_starts[::max(1, n_starts // CFG.LS_TRAINING_PATHS)]
        if starts.size > CFG.LS_TRAINING_PATHS:
            idx = np.linspace(0, starts.size - 1, CFG.LS_TRAINING_PATHS, dtype=int)
            starts = starts[idx]
        out = np.empty((starts.size, horizon + 1))
        for j, s in enumerate(starts):
            out[j] = z_train[s : s + horizon + 1]
        return out

    long_paths = _windows(long_starts)
    short_paths = _windows(short_starts)
    if verbose:
        print(f"[backtest] LS training: long_paths={long_paths.shape}, short_paths={short_paths.shape}")

    solver_long = LongstaffSchwartzSignature(horizon=horizon).fit(long_paths, side="long", verbose=verbose)
    solver_short = LongstaffSchwartzSignature(horizon=horizon).fit(short_paths, side="short", verbose=verbose)

    # ------------------------------------------------------------ 4. Онлайн-симуляция
    rm = RiskManager()
    geo = DoubleGeometricFilter(window_size=CFG.WINDOW_SIZE)

    equity_curve = np.empty(n_total - n_train)
    trades: List[Trade] = []
    active_solver: Optional[LongstaffSchwartzSignature] = None
    entry_window_start = 0

    for t_local, t in enumerate(range(n_train, n_total)):
        price_y = float(y[t])
        price_x = float(x[t])
        z_t = float(z_full[t])
        z_norm_t = float(z_norm_full[t])
        sd_t = float(sd_roll[t])

        # ---------------- drawdown gate ----------------
        if rm.enforce_drawdown(price_y, price_x):
            equity_curve[t_local:] = rm.equity(price_y, price_x)
            if verbose:
                print(f"[backtest] HALT at bar {t} (equity={rm.equity(price_y, price_x):.2f})")
            break

        # ---------------- decision on existing position ----------------
        if rm.position is not None:
            pos = rm.position
            n_open = pos.bars_open(t)
            window_z = z_full[entry_window_start : t + 1]

            sign_factor = 1.0 if pos.side == "long_spread" else -1.0
            payoff_z = sign_factor * (pos.entry_z - z_t)

            cont = active_solver.continuation_value(window_z, n_open) if active_solver else -np.inf

            close_reason: Optional[str] = None
            # 1) стоп-лосс: |ΔZ| против входа > STOP_LOSS_Z текущих σ
            if pos.entry_z_sigma > 0 and (-payoff_z) > stop_loss_z * pos.entry_z_sigma:
                close_reason = "stop_loss"
            elif n_open >= horizon:
                close_reason = "horizon"
            elif n_open >= 1 and payoff_z >= cont:
                close_reason = "stop_optimal"

            if close_reason is not None:
                side_taken = pos.side
                entry_bar = pos.entry_bar
                entry_z_val = pos.entry_z
                pnl = rm.close_position(price_y, price_x)
                trades.append(Trade(
                    side=side_taken,
                    entry_bar=entry_bar,
                    exit_bar=t,
                    entry_z=entry_z_val,
                    exit_z=z_t,
                    pnl=pnl,
                    reason=close_reason,
                ))
                active_solver = None

        # ---------------- entry decision ----------------
        if rm.position is None and not rm.halted:
            # rolling-z должно «прогреться»
            if t >= rolling_window // 2:
                window = z_full[max(0, t - CFG.WINDOW_SIZE + 1) : t + 1]
                allow = True
                if enable_geo_filter and window.size >= CFG.WINDOW_SIZE:
                    path2d = lead_lag(window)
                    fstats = geo.update(path2d)
                    allow = fstats.allow_entry

                want_short = z_norm_t > entry_z_threshold      # шортим спред
                want_long = z_norm_t < -entry_z_threshold      # лонгуем спред

                if allow and (want_long or want_short):
                    side: Side = "long_spread" if want_long else "short_spread"
                    opened = rm.open_position(
                        side=side,
                        beta=beta_full[t],
                        price_y=price_y,
                        price_x=price_x,
                        current_bar=t,
                        entry_z=z_t,
                        entry_z_sigma=sd_t,
                    )
                    if opened:
                        entry_window_start = t
                        active_solver = solver_long if side == "long_spread" else solver_short

        equity_curve[t_local] = rm.equity(price_y, price_x)

    return BacktestResult(
        equity_curve=equity_curve,
        spread=z_full[n_train:],
        beta=beta_full[n_train:],
        trades=trades,
        halted=rm.halted,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        bars = load_bars(csv_y=sys.argv[1], csv_x=sys.argv[2])
    else:
        bars = load_bars()
    print(f"Loaded {len(bars)} M1 bars.  y[{bars.price_y[0]:.2f}->{bars.price_y[-1]:.2f}], "
          f"x[{bars.price_x[0]:.2f}->{bars.price_x[-1]:.2f}]")
    result = run_backtest(bars)
    print("\n=== BACKTEST SUMMARY ===")
    for k, v in result.summary().items():
        print(f"  {k:>22}: {v}")
