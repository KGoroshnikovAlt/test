"""
config.py
=========
Глобальные параметры системы статистического арбитража.

Спецификация контрактов соответствует условиям брокера FxPro для CFD на индексы
#USNDAQ100 и #USSPX500: минимальный лот 0.01, шаг 0.01, плечо 1:200, стоимость
тика на 1 лот = 1 USD.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Config:
    # ---------- Капитал и риск ----------
    START_CAPITAL: float = 1000.0          # стартовый депозит, USD
    LEVERAGE: int = 200                    # плечо 1:200
    MAX_DRAWDOWN_PCT: float = 0.15         # 15 % => принудительный выход на 850 USD
    EQUITY_FLOOR: float = field(init=False)

    # ---------- Спецификация инструментов ----------
    SYMBOL_Y: str = "USNDAQ100"            # ведомый (Nasdaq)
    SYMBOL_X: str = "USSPX500"             # ведущий (S&P 500)
    MIN_LOT: float = 0.01
    LOT_STEP: float = 0.01
    TICK_SIZE: float = 0.01
    TICK_VALUE_PER_LOT: float = 1.0        # USD за тик на 1 лот

    # ---------- Параметры микроструктуры ----------
    WINDOW_SIZE: int = 30                  # длина скользящего окна, минут
    SIGNATURE_ORDER: int = 3               # уровень усечения сигнатуры L
    LEVY_SEGMENTS: int = 3                 # число сегментов K для площади Леви

    # ---------- Калман ----------
    KALMAN_Q_BETA: float = 1e-5            # дисперсия шума процесса по beta
    KALMAN_Q_ALPHA: float = 1e-5           # дисперсия шума процесса по alpha
    KALMAN_R: float = 1.0                  # дисперсия шума измерения
    KALMAN_P0: float = 1.0                 # начальная дисперсия состояния

    # ---------- Геометрические фильтры ----------
    LEVY_Z_THRESHOLD: float = 1.96         # ±1.96 σ для площади Леви
    COVAR_Z_THRESHOLD: float = -1.96       # глубокий минус ковариации блокирует вход

    # ---------- Longstaff–Schwartz ----------
    LS_TRAINING_PATHS: int = 2000          # число обучающих траекторий
    LS_HORIZON: int = 30                   # максимальный горизонт удержания, минут
    LS_RIDGE: float = 1e-3                 # L2-регуляризация регрессии

    # ---------- Сценарий "log-spread" и стоп-лосс ----------
    USE_LOG_PRICES: bool = True            # Калман по log P → стационарная связка
    ROLLING_ZSTD_WINDOW: int = 1440        # окно для скользящих μ/σ Z, минут (≈ 1 сутки)
    ENTRY_Z_THRESHOLD: float = 2.0         # порог входа по rolling-z
    STOP_LOSS_Z: float = 2.5               # стоп-лосс: |Z - Z_entry| ≥ 2.5σ → выход

    # ---------- Прочее ----------
    RANDOM_SEED: int = 42

    def __post_init__(self) -> None:
        # уровень принудительного аварийного выхода (Drawdown Constraint)
        object.__setattr__(
            self, "EQUITY_FLOOR",
            self.START_CAPITAL * (1.0 - self.MAX_DRAWDOWN_PCT),
        )

    # Удобный атрибут: вектор размерностей сигнатуры по уровням для пути в R^d
    def signature_levels_dims(self, d: int = 2) -> Tuple[int, ...]:
        return tuple(d ** k for k in range(1, self.SIGNATURE_ORDER + 1))

    def signature_total_dim(self, d: int = 2) -> int:
        return sum(self.signature_levels_dims(d))


CFG = Config()
