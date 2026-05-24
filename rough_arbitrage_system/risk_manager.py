"""
risk_manager.py — capital, margin and drawdown control.

FxPro CFD math used throughout:

    Notional      = lot_size * price
    Margin_req    = Notional / leverage
    PnL           = (price_now - price_open) * lot_size * tick_value_per_lot / tick_size
                    (sign-adjusted by trade direction)

The manager exposes:

  * sizing of the hedge ratio in lots, given the live Kalman beta
  * margin admissibility checks
  * mark-to-market equity tracking
  * a hard "panic close" trigger at the 150-USD drawdown floor
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from config import Config, FxProSpec


class Side(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class Leg:
    spec: FxProSpec
    side: Side
    lot: float
    entry_price: float

    def pnl(self, current_price: float) -> float:
        move = (current_price - self.entry_price) * self.side.value
        return move * self.lot * self.spec.tick_value_per_lot / self.spec.tick_size

    def margin(self, current_price: float, leverage: int) -> float:
        return self.lot * current_price / leverage


@dataclass
class Position:
    """A market-neutral pair: long one index, short the other (or vice versa)."""
    ndx_leg: Leg
    spx_leg: Leg
    open_minute: int

    def mtm(self, p_ndx: float, p_spx: float) -> float:
        return self.ndx_leg.pnl(p_ndx) + self.spx_leg.pnl(p_spx)


@dataclass
class RiskManager:
    cfg: Config
    equity: float = field(init=False)
    peak_equity: float = field(init=False)
    realised: float = 0.0
    position: Optional[Position] = None
    panic: bool = False                       # latched after a drawdown stop
    trade_log: List[dict] = field(default_factory=list)

    def __post_init__(self):
        self.equity = self.cfg.START_CAPITAL
        self.peak_equity = self.cfg.START_CAPITAL

    # --- sizing --------------------------------------------------------------
    def hedge_lots(self, beta: float) -> tuple[float, float]:
        """Lots on (NDX, SPX) for the current dynamic hedge ratio beta."""
        ndx_lot = self.cfg.BASE_LOT_NDX
        # Round to nearest lot step >= min_lot
        raw = ndx_lot * abs(beta)
        spec = self.cfg.SPX
        spx_lot = max(spec.min_lot, round(raw / spec.lot_step) * spec.lot_step)
        return ndx_lot, spx_lot

    def required_margin(self, p_ndx: float, p_spx: float, beta: float) -> float:
        ndx_lot, spx_lot = self.hedge_lots(beta)
        return (ndx_lot * p_ndx + spx_lot * p_spx) / self.cfg.LEVERAGE

    # --- entry / exit --------------------------------------------------------
    def can_open(self, p_ndx: float, p_spx: float, beta: float) -> bool:
        if self.panic or self.position is not None:
            return False
        return self.required_margin(p_ndx, p_spx, beta) <= self.equity * 0.5

    def open(self, minute: int, p_ndx: float, p_spx: float, beta: float, z_sign: int) -> None:
        """When Z is positive (NDX rich vs SPX), short the spread: short NDX, long SPX."""
        ndx_lot, spx_lot = self.hedge_lots(beta)
        ndx_side = Side.SHORT if z_sign > 0 else Side.LONG
        spx_side = Side.LONG if z_sign > 0 else Side.SHORT
        ndx_leg = Leg(self.cfg.NDX, ndx_side, ndx_lot, p_ndx)
        spx_leg = Leg(self.cfg.SPX, spx_side, spx_lot, p_spx)
        self.position = Position(ndx_leg, spx_leg, minute)

    def close(self, minute: int, p_ndx: float, p_spx: float, reason: str) -> Optional[dict]:
        if self.position is None:
            return None
        pnl = self.position.mtm(p_ndx, p_spx)
        self.realised += pnl
        rec = {
            "open_minute": self.position.open_minute,
            "close_minute": minute,
            "pnl": pnl,
            "reason": reason,
            "ndx_side": self.position.ndx_leg.side.name,
            "ndx_lot": self.position.ndx_leg.lot,
            "spx_lot": self.position.spx_leg.lot,
        }
        self.trade_log.append(rec)
        self.position = None
        return rec

    # --- equity / drawdown ---------------------------------------------------
    def mark(self, p_ndx: float, p_spx: float) -> float:
        floating = self.position.mtm(p_ndx, p_spx) if self.position else 0.0
        self.equity = self.cfg.START_CAPITAL + self.realised + floating
        self.peak_equity = max(self.peak_equity, self.equity)
        return self.equity

    def drawdown_breached(self) -> bool:
        return self.equity <= self.cfg.EQUITY_FLOOR

    def maybe_panic_close(self, minute: int, p_ndx: float, p_spx: float) -> Optional[dict]:
        """If equity has fallen through the floor, force-flatten and latch panic."""
        if self.drawdown_breached() and self.position is not None:
            rec = self.close(minute, p_ndx, p_spx, reason="drawdown")
            self.panic = True
            return rec
        if self.drawdown_breached():
            self.panic = True
        return None
