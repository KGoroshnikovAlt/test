"""
main.py — end-to-end backtest harness.

Pipeline per minute:

    bar -> Kalman -> Z_t
         -> 30-min rolling window
         -> Lead-Lag + signature S^L(Z)
         -> Levy-area / increment-covariance double filter
         -> Longstaff-Schwartz continuation value
         -> RiskManager (margin, drawdown, lot sizing, execution)

Two phases:

    OFFLINE — first half of the data trains the LS regression weights;
    ONLINE  — second half is the actual paper trade, walked one minute at a time.

Run:  python main.py
"""
from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from config import CONFIG, Config
from data_ingestor import BarSeries, generate_synthetic, load_csv
from double_filter import (
    FilterState,
    admissible,
    increment_covariance,
    segmented_levy_area,
)
from kalman_filter import KalmanSpread, run_kalman_series
from risk_manager import RiskManager
from signature_engine import lead_lag, spread_signature
from solver_ls import LongstaffSchwartz, _signature_length


# --------------------------------------------------------------------------- #
# Offline training                                                            #
# --------------------------------------------------------------------------- #
def train_ls(z: np.ndarray, cfg: Config, episode_len: int = 60, stride: int = 15) -> LongstaffSchwartz:
    """Slice the offline Z series into episodes and fit Longstaff-Schwartz weights."""
    W = cfg.WINDOW_SIZE
    F = _signature_length(cfg)
    episodes_sig: list[np.ndarray] = []
    episodes_pay: list[np.ndarray] = []

    for start in range(W, len(z) - episode_len, stride):
        sigs = np.empty((episode_len, F))
        pays = np.empty(episode_len)
        # entry price baseline = Z at the moment of synthetic "entry"
        z_entry = z[start - 1]
        side = -np.sign(z_entry) if abs(z_entry) > 1e-12 else 1.0
        for n in range(episode_len):
            t = start + n
            window = z[t - W: t]
            sigs[n] = spread_signature(window, cfg)
            # payoff: short the spread when Z is large positive, long when large negative
            pays[n] = side * (z_entry - z[t])
        episodes_sig.append(sigs)
        episodes_pay.append(pays)

    if not episodes_sig:
        raise RuntimeError("Not enough offline data to train Longstaff-Schwartz")
    signatures = np.stack(episodes_sig)
    payoffs = np.stack(episodes_pay)
    ls = LongstaffSchwartz(cfg=cfg).fit(signatures, payoffs)
    return ls


# --------------------------------------------------------------------------- #
# Online simulation                                                           #
# --------------------------------------------------------------------------- #
def backtest(bars: BarSeries, cfg: Config = CONFIG, verbose: bool = True) -> dict:
    n = len(bars)
    split = n // 2

    # --- offline Kalman to build Z series for training ---------------------- #
    kf_series = run_kalman_series(bars.ndx[:split], bars.spx[:split], cfg)
    z_offline = kf_series["Z"]
    sigma_z_offline = float(np.std(z_offline[cfg.WINDOW_SIZE:]) + 1e-9)

    if verbose:
        print(f"[offline] split={split}, sigma(Z)={sigma_z_offline:.6g}")

    ls = train_ls(z_offline, cfg)

    # --- online walk-forward ------------------------------------------------ #
    kf = KalmanSpread(cfg)
    # warm-start with offline filter state for continuity
    kf.theta = np.array([kf_series["beta"][-1], kf_series["alpha"][-1]])

    rm = RiskManager(cfg=cfg)
    fs = FilterState(window=500)

    window: deque = deque(maxlen=cfg.WINDOW_SIZE)
    z_history: list[float] = []
    sigma_z = sigma_z_offline
    sigma_update_every = 200

    equity_curve = np.empty(n - split)
    n_entries = 0
    n_admissibility_blocks = 0
    n_panic = 0
    z_entry_value = 0.0
    minute_in_trade = 0

    for k, idx in enumerate(range(split, n)):
        y = float(bars.ndx[idx])
        x = float(bars.spx[idx])
        Z, beta, alpha = kf.step(y, x)
        z_history.append(Z)
        window.append(Z)

        # periodic recalibration of the entry-z normaliser
        if k > 0 and k % sigma_update_every == 0:
            sigma_z = float(np.std(z_history[-cfg.WINDOW_SIZE * 10 :]) + 1e-9)

        # mark to market every minute
        rm.mark(y, x)
        equity_curve[k] = rm.equity

        # drawdown gate
        panic_rec = rm.maybe_panic_close(idx, y, x)
        if panic_rec is not None:
            n_panic += 1
        if rm.panic:
            continue

        if len(window) < cfg.WINDOW_SIZE:
            continue

        # signature on the rolling 30-minute spread window
        z_arr = np.fromiter(window, dtype=float)
        path = lead_lag(z_arr)
        ss = segmented_levy_area(path, cfg.LEVY_SEGMENTS)
        cv = increment_covariance(path)
        fs.update(ss, cv)
        z_ll = fs.z_levy(ss)
        z_cov = fs.z_covar(cv)
        sig_vec = spread_signature(z_arr, cfg)

        if rm.position is None:
            # entry logic: large |Z|, signature admissible, sufficient margin
            z_norm = Z / sigma_z
            if abs(z_norm) >= cfg.ENTRY_Z_THRESHOLD and admissible(z_ll, z_cov, cfg):
                if rm.can_open(y, x, beta):
                    rm.open(idx, y, x, beta, z_sign=int(np.sign(Z)))
                    z_entry_value = Z
                    minute_in_trade = 0
                    n_entries += 1
            elif not admissible(z_ll, z_cov, cfg):
                n_admissibility_blocks += 1
        else:
            # exit logic via Longstaff-Schwartz
            minute_in_trade += 1
            # current paper payoff measured in spread units
            payoff_spread = (-np.sign(z_entry_value)) * (z_entry_value - Z)
            stop = ls.should_stop(minute_in_trade, sig_vec, payoff_spread)
            # also force a flat at the end of the LS horizon
            forced = minute_in_trade >= 59
            # break-out exit: if spread has reverted close to zero
            reverted = (z_entry_value > 0 and Z <= 0.1 * z_entry_value) or (
                z_entry_value < 0 and Z >= 0.1 * z_entry_value
            )
            if stop or forced or reverted:
                reason = "ls" if stop else ("forced" if forced else "reverted")
                rm.close(idx, y, x, reason=reason)

        if verbose and k % cfg.LOG_EVERY == 0 and k > 0:
            print(
                f"[t={idx}] Z={Z:+.3f}  β={beta:.3f}  equity={rm.equity:.2f}  "
                f"trades={len(rm.trade_log)}  open={'Y' if rm.position else 'N'}"
            )

    # ensure no position is left open
    if rm.position is not None:
        rm.close(n - 1, float(bars.ndx[-1]), float(bars.spx[-1]), reason="eod")

    stats = {
        "final_equity": rm.equity,
        "realised_pnl": rm.realised,
        "n_trades": len(rm.trade_log),
        "n_entries": n_entries,
        "n_admissibility_blocks": n_admissibility_blocks,
        "n_panic": n_panic,
        "max_equity": float(np.max(equity_curve)) if len(equity_curve) else cfg.START_CAPITAL,
        "min_equity": float(np.min(equity_curve)) if len(equity_curve) else cfg.START_CAPITAL,
        "equity_curve": equity_curve,
        "trades": rm.trade_log,
    }
    return stats


def _summary(stats: dict, cfg: Config) -> str:
    lines = [
        "===== Rough Arbitrage Backtest =====",
        f"  start capital    : {cfg.START_CAPITAL:.2f}",
        f"  final equity     : {stats['final_equity']:.2f}",
        f"  realised PnL     : {stats['realised_pnl']:+.2f}",
        f"  trades closed    : {stats['n_trades']}",
        f"  entry attempts   : {stats['n_entries']}",
        f"  admissibility blocks: {stats['n_admissibility_blocks']}",
        f"  drawdown stops   : {stats['n_panic']}",
        f"  equity max / min : {stats['max_equity']:.2f} / {stats['min_equity']:.2f}",
    ]
    if stats["trades"]:
        wins = sum(1 for t in stats["trades"] if t["pnl"] > 0)
        pnls = [t["pnl"] for t in stats["trades"]]
        lines.append(f"  win rate         : {wins}/{len(pnls)} = {wins / len(pnls):.1%}")
        lines.append(f"  avg trade pnl    : {np.mean(pnls):+.4f}")
        lines.append(f"  best / worst     : {max(pnls):+.2f} / {min(pnls):+.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="optional CSV of bars")
    parser.add_argument("--minutes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shock-at", type=int, default=None,
                        help="inject a regime shock at this minute (synthetic only)")
    args = parser.parse_args()

    bars = load_csv(args.csv) if args.csv else generate_synthetic(
        n_minutes=args.minutes, seed=args.seed, regime_shift_at=args.shock_at,
    )
    stats = backtest(bars, CONFIG)
    print(_summary(stats, CONFIG))


if __name__ == "__main__":
    main()
