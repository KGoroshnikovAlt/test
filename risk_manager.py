"""
risk_manager.py
===============
Контроль маржи, расчёт лотов и принудительный аварийный выход.

Спецификации FxPro (CFD на индексы):
    * минимальный лот:   0.01
    * шаг лота:          0.01
    * плечо:             1:200
    * стоимость 1 тика на 1 лот = 1 USD при шаге цены 0.01

Требуемая маржа на 1 лот при цене P:
    margin = (lot_size * P) / leverage.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal

from config import CFG

Side = Literal["long_spread", "short_spread", "flat"]


@dataclass
class Position:
    side: Side                              # направление сделки по СПРЕДУ
    lot_y: float                            # лот по Nasdaq (с учётом знака)
    lot_x: float                            # лот по S&P 500 (с учётом знака)
    entry_price_y: float
    entry_price_x: float
    entry_bar: int                          # индекс минутного бара входа
    margin_locked: float                    # маржа, заблокированная под сделку
    entry_z: float = 0.0                    # значение Z в момент входа
    entry_z_sigma: float = 1.0              # текущая σ Z в момент входа (для стопа)

    def bars_open(self, current_bar: int) -> int:
        return current_bar - self.entry_bar


@dataclass
class RiskManager:
    """
    Лёгкий риск-менеджер, поддерживающий одну активную позицию по спреду.

    Учёт PnL ведётся в "точках * стоимость тика": для CFD на индексы при шаге
    0.01 стоимость 1 пункта индекса на 1 лот = 100 USD (так как один тик = 1
    USD, а в пункте 100 тиков).
    """
    capital: float = CFG.START_CAPITAL
    leverage: int = CFG.LEVERAGE
    min_lot: float = CFG.MIN_LOT
    lot_step: float = CFG.LOT_STEP
    tick_size: float = CFG.TICK_SIZE
    tick_value: float = CFG.TICK_VALUE_PER_LOT
    equity_floor: float = CFG.EQUITY_FLOOR
    position: Optional[Position] = field(default=None)
    realized_pnl: float = 0.0
    halted: bool = False                    # сработал ли drawdown constraint

    # --------------------------------------------------------- helpers
    @property
    def pnl_multiplier(self) -> float:
        """
        USD за 1.00 пункта цены на 1.00 лота.
        По спецификации FxPro: tick=0.01 ↔ 1 USD/лот  =>  множитель = 100.
        """
        return self.tick_value / self.tick_size

    def _round_lot(self, lot: float) -> float:
        # округление к ближайшему допустимому шагу с минимальной квантой
        n_steps = round(lot / self.lot_step)
        return max(self.min_lot, n_steps * self.lot_step)

    @staticmethod
    def _required_margin(lot: float, price: float, leverage: int) -> float:
        return abs(lot) * price / leverage

    # --------------------------------------------------------- sizing
    def size_position(self, beta: float, price_y: float, price_x: float) -> tuple[float, float]:
        """
        Возвращает пару объёмов (lot_y, lot_x) в положительной нотации.
        Хеджирующий принцип: на 0.01 лота Nasdaq берём β · 0.01 лота S&P 500.
        """
        lot_y = self.min_lot
        lot_x = self._round_lot(self.min_lot * abs(beta))
        return lot_y, lot_x

    # --------------------------------------------------------- open / close
    def open_position(
        self,
        side: Side,
        beta: float,
        price_y: float,
        price_x: float,
        current_bar: int,
        entry_z: float = 0.0,
        entry_z_sigma: float = 1.0,
    ) -> bool:
        if self.halted or self.position is not None or side == "flat":
            return False

        lot_y_abs, lot_x_abs = self.size_position(beta, price_y, price_x)

        # знак: long_spread = long Y, short X;  short_spread = short Y, long X
        sign_y = +1.0 if side == "long_spread" else -1.0
        sign_x = -1.0 if side == "long_spread" else +1.0

        margin_y = self._required_margin(lot_y_abs, price_y, self.leverage)
        margin_x = self._required_margin(lot_x_abs, price_x, self.leverage)
        total_margin = margin_y + margin_x

        # запас: не открываем, если маржа > 25% свободного капитала
        if total_margin > 0.25 * self.equity(price_y, price_x):
            return False

        self.position = Position(
            side=side,
            lot_y=sign_y * lot_y_abs,
            lot_x=sign_x * lot_x_abs,
            entry_price_y=price_y,
            entry_price_x=price_x,
            entry_bar=current_bar,
            margin_locked=total_margin,
            entry_z=entry_z,
            entry_z_sigma=entry_z_sigma,
        )
        return True

    def unrealized_pnl(self, price_y: float, price_x: float) -> float:
        if self.position is None:
            return 0.0
        p = self.position
        m = self.pnl_multiplier
        pnl_y = p.lot_y * (price_y - p.entry_price_y) * m
        pnl_x = p.lot_x * (price_x - p.entry_price_x) * m
        return pnl_y + pnl_x

    def close_position(self, price_y: float, price_x: float) -> float:
        if self.position is None:
            return 0.0
        pnl = self.unrealized_pnl(price_y, price_x)
        self.realized_pnl += pnl
        self.position = None
        return pnl

    def equity(self, price_y: float, price_x: float) -> float:
        return self.capital + self.realized_pnl + self.unrealized_pnl(price_y, price_x)

    # --------------------------------------------------------- drawdown gate
    def enforce_drawdown(self, price_y: float, price_x: float) -> bool:
        """
        Если эквити < 850 USD — аварийный выход и блокировка дальнейшей торговли.
        Возвращает True, если сработал аварийный выход.
        """
        if self.halted:
            return True
        if self.equity(price_y, price_x) <= self.equity_floor:
            if self.position is not None:
                self.close_position(price_y, price_x)
            self.halted = True
            return True
        return False
