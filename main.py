"""
main.py
=======
Сквозной бэктест системы статистического арбитража на грубых путях.

Pipeline на каждой минуте:
    1. Калман: обновление β_t, α_t, расчёт чистого спреда Z_t.
    2. Lead-Lag окно длины WINDOW_SIZE → 2D-путь.
    3. Двойной геометрический фильтр (площадь Леви + ковариация).
    4. Если позиция закрыта и фильтр пропускает — открываем по знаку Z.
    5. Если позиция открыта — сравниваем платёж с continuation value
       (выход алгоритма Лонгстаффа–Шварца на сигнатурах).
    6. Risk manager контролирует маржу и просадку.
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
        return {
            "n_trades": n,
            "wins": wins,
            "win_rate": (wins / n) if n else 0.0,
            "gross_pnl": gross,
            "avg_pnl": (gross / n) if n else 0.0,
            "avg_duration_min": float(np.mean(durations)) if durations else 0.0,
            "final_equity": self.final_equity,
            "halted_by_drawdown": self.halted,
        }


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    bars: Optional[MarketBars] = None,
    train_fraction: float = 0.4,
    entry_z_threshold: float = 1.5,
    verbose: bool = True,
) -> BacktestResult:
    """
    Запускает офлайн-обучение и онлайн-симуляцию на одной серии M1-баров.

    Параметры
    ---------
    bars : MarketBars | None
        Если None — генерируем синтетику через data_ingestor.load_bars().
    train_fraction : float
        Доля баров, отведённая под обучение Калмана и LS-решателя.
    entry_z_threshold : float
        Порог входа по нормированному Z (число σ скользящего окна).
    """
    if bars is None:
        bars = load_bars(seed=CFG.RANDOM_SEED)
    y = bars.price_y
    x = bars.price_x
    n_total = len(bars)
    n_train = int(train_fraction * n_total)

    if verbose:
        print(f"[backtest] bars={n_total}, train={n_train}, test={n_total - n_train}")

    # ------------------------------------------------------------ 1. Калман
    kalman = DynamicHedgeKalman()
    z_full = np.empty(n_total)
    beta_full = np.empty(n_total)
    for t in range(n_total):
        z_full[t] = kalman.step(float(x[t]), float(y[t]))
        beta_full[t] = kalman.beta

    # ------------------------------------------------------------ 2. Обучение LS
    z_train = z_full[:n_train]
    horizon = CFG.LS_HORIZON
    train_windows = LongstaffSchwartzSignature.build_training_windows(
        z_train, horizon=horizon, stride=max(1, horizon // 6),
    )
    # ограничиваем количеством LS_TRAINING_PATHS
    if train_windows.shape[0] > CFG.LS_TRAINING_PATHS:
        idx = np.linspace(0, train_windows.shape[0] - 1, CFG.LS_TRAINING_PATHS, dtype=int)
        train_windows = train_windows[idx]

    if verbose:
        print(f"[backtest] LS training windows: {train_windows.shape}")

    solver_long = LongstaffSchwartzSignature().fit(train_windows, side="long", verbose=verbose)
    solver_short = LongstaffSchwartzSignature().fit(train_windows, side="short", verbose=verbose)

    # ------------------------------------------------------------ 3. Простор для входа: σ от Z в обучающей выборке
    z_mu = float(z_train.mean())
    z_sigma = float(z_train.std(ddof=1) + 1e-12)

    # ------------------------------------------------------------ 4. Онлайн-симуляция
    rm = RiskManager()
    geo = DoubleGeometricFilter()

    equity_curve = np.empty(n_total - n_train)
    trades: List[Trade] = []
    active_solver: Optional[LongstaffSchwartzSignature] = None
    entry_window_start = 0

    for t_local, t in enumerate(range(n_train, n_total)):
        price_y = float(y[t])
        price_x = float(x[t])
        z_t = float(z_full[t])

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

            entry_z = z_full[pos.entry_bar]
            sign_factor = 1.0 if pos.side == "long_spread" else -1.0
            payoff_z = sign_factor * (entry_z - z_t)

            cont = active_solver.continuation_value(window_z, n_open) if active_solver else -np.inf

            close_reason: Optional[str] = None
            if n_open >= CFG.LS_HORIZON:
                close_reason = "horizon"
            elif payoff_z >= cont:
                close_reason = "stop_optimal"

            if close_reason is not None:
                side_taken = pos.side
                entry_bar = pos.entry_bar
                pnl = rm.close_position(price_y, price_x)
                trades.append(Trade(
                    side=side_taken,
                    entry_bar=entry_bar,
                    exit_bar=t,
                    entry_z=entry_z,
                    exit_z=z_t,
                    pnl=pnl,
                    reason=close_reason,
                ))
                active_solver = None

        # ---------------- entry decision ----------------
        if rm.position is None and not rm.halted:
            window = z_full[max(0, t - CFG.WINDOW_SIZE + 1) : t + 1]
            if window.size >= CFG.WINDOW_SIZE:
                path2d = lead_lag(window)
                fstats = geo.update(path2d)

                z_norm = (z_t - z_mu) / z_sigma
                want_short = z_norm > entry_z_threshold     # спред высок => шортим спред
                want_long = z_norm < -entry_z_threshold     # спред низок => лонгуем спред

                if fstats.allow_entry and (want_long or want_short):
                    side: Side = "long_spread" if want_long else "short_spread"
                    if rm.open_position(side, beta_full[t], price_y, price_x, t):
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
    bars = load_bars()
    print(f"Loaded {len(bars)} M1 bars.  y[{bars.price_y[0]:.2f}->{bars.price_y[-1]:.2f}], "
          f"x[{bars.price_x[0]:.2f}->{bars.price_x[-1]:.2f}]")
    result = run_backtest(bars)
    print("\n=== BACKTEST SUMMARY ===")
    for k, v in result.summary().items():
        print(f"  {k:>22}: {v}")
