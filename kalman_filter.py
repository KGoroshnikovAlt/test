"""
kalman_filter.py
================
Динамический фильтр Калмана для оценки коэффициента хеджирования
бета и смещения альфа между двумя индексами.

Модель состояния:
    theta_t = theta_{t-1} + w_t,            w_t ~ N(0, Q)
    y_t     = H_t theta_t + v_t,             v_t ~ N(0, R)

где
    theta_t = [beta_t, alpha_t]^T,
    H_t     = [x_t, 1].

Возвращаемый «чистый» спред:
    Z_t = y_t - beta_{t|t} x_t - alpha_{t|t}.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import CFG


@dataclass
class KalmanState:
    theta: np.ndarray   # (2,)  оценка [beta, alpha]
    P: np.ndarray       # (2,2) ковариация ошибки оценки


class DynamicHedgeKalman:
    """
    Калмановский фильтр для динамической оценки коэффициента хеджирования.

    Параметры
    ---------
    q_beta, q_alpha : float
        Диагональные элементы Q (шум процесса). Чем больше, тем быстрее модель
        реагирует на изменения структуры, но тем шумнее остаток.
    r : float
        Дисперсия шума наблюдения.
    """

    def __init__(
        self,
        q_beta: float = CFG.KALMAN_Q_BETA,
        q_alpha: float = CFG.KALMAN_Q_ALPHA,
        r: float = CFG.KALMAN_R,
        p0: float = CFG.KALMAN_P0,
        theta0: Optional[np.ndarray] = None,
    ) -> None:
        self.Q = np.diag([q_beta, q_alpha]).astype(np.float64)
        self.R = float(r)
        self._state = KalmanState(
            theta=np.zeros(2) if theta0 is None else np.asarray(theta0, dtype=np.float64),
            P=p0 * np.eye(2, dtype=np.float64),
        )

    # ------------------------------------------------------------------ state
    @property
    def beta(self) -> float:
        return float(self._state.theta[0])

    @property
    def alpha(self) -> float:
        return float(self._state.theta[1])

    @property
    def state(self) -> KalmanState:
        return self._state

    # ------------------------------------------------------------------ step
    def step(self, x: float, y: float) -> float:
        """
        Один шаг фильтра: вход — пара измерений (x_t, y_t), выход — остаток Z_t.

        Используется стандартное двухшаговое разложение Predict / Update.
        """
        s = self._state

        # --- 1. Predict ---
        # Состояние эволюционирует как random walk: theta_pred = theta_prev.
        theta_pred = s.theta
        P_pred = s.P + self.Q

        # --- 2. Update ---
        H = np.array([x, 1.0], dtype=np.float64)             # (2,)
        y_hat = float(H @ theta_pred)                         # скаляр
        innovation = y - y_hat                                # пред-fit остаток
        S = float(H @ P_pred @ H) + self.R                    # дисперсия инновации
        K = (P_pred @ H) / S                                  # калмановский gain (2,)

        theta_new = theta_pred + K * innovation
        # Joseph form для численной устойчивости ковариации
        I = np.eye(2)
        P_new = (I - np.outer(K, H)) @ P_pred @ (I - np.outer(K, H)).T + np.outer(K, K) * self.R

        self._state = KalmanState(theta=theta_new, P=P_new)

        # post-fit остаток — собственно очищенный спред Z_t
        z_t = y - float(H @ theta_new)
        return z_t

    # ------------------------------------------------------------------ batch
    def run(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Пакетный прогон по двум одинаковой длины массивам. Возвращает массив Z_t.
        Полезно для офлайн-этапа Лонгстаффа–Шварца, чтобы получить полную
        историю спреда одной операцией.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        assert x.shape == y.shape
        z = np.empty_like(x)
        for t in range(x.size):
            z[t] = self.step(float(x[t]), float(y[t]))
        return z
