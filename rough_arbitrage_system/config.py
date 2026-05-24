"""
config.py — system parameters, risk limits, and FxPro CFD specifications.

All parameters are centralised so the rest of the system stays declarative.
Lot/tick values mirror FxPro CFD specs for #USNDAQ100 and #USSPX500.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FxProSpec:
    """Contract specification for a single CFD on FxPro."""
    symbol: str
    tick_size: float = 0.01
    tick_value_per_lot: float = 1.0  # USD profit per 1.0 price move per 1 lot
    min_lot: float = 0.01
    lot_step: float = 0.01


@dataclass(frozen=True)
class Config:
    # --- account ---
    LEVERAGE: int = 200
    START_CAPITAL: float = 1000.0
    MAX_DRAWDOWN_PCT: float = 0.15            # 15% -> 150 USD hard stop
    EQUITY_FLOOR: float = 850.0               # explicit floor for clarity

    # --- trajectory / signature ---
    WINDOW_SIZE: int = 30                     # minutes in rolling window
    SIGNATURE_ORDER: int = 3                  # truncation level L
    LEVY_SEGMENTS: int = 3                    # K segments for Levy area

    # --- entry filter thresholds (z-score on the geometric invariants) ---
    LEVY_Z_THRESHOLD: float = 1.96
    COVAR_Z_THRESHOLD: float = 1.96

    # --- Kalman filter hyperparameters ---
    # Q is the state-noise covariance for [beta, alpha], R is observation noise.
    KALMAN_Q_BETA: float = 1e-5
    KALMAN_Q_ALPHA: float = 1e-5
    KALMAN_R: float = 1e-3
    KALMAN_P0: float = 1.0
    KALMAN_BETA0: float = 1.0
    KALMAN_ALPHA0: float = 0.0

    # --- Longstaff-Schwartz ---
    LS_PAYOFF_THRESHOLD: float = 0.0          # only act when expected payoff > this
    LS_REG_LAMBDA: float = 1e-4               # ridge regularization

    # --- contract specs (FxPro CFDs) ---
    NDX: FxProSpec = field(default_factory=lambda: FxProSpec(symbol="#USNDAQ100"))
    SPX: FxProSpec = field(default_factory=lambda: FxProSpec(symbol="#USSPX500"))

    # --- execution / signal ---
    ENTRY_Z_THRESHOLD: float = 1.5            # |Z_t / sigma_Z| to consider entry
    BASE_LOT_NDX: float = 0.01                # we always trade 0.01 on the lead leg
    TICK_SIZE: float = 0.01

    # --- backtest output ---
    LOG_EVERY: int = 500
    PLOT_RESULTS: bool = False                # set True for matplotlib output


CONFIG = Config()
