"""
double_filter.py
================
Двойной геометрический фильтр входа.

1) Сегментированная площадь Леви SS_{s,t}
   Для двумерного пути (X, Y) на отрезке [s, t] площадь Леви есть
        A(X, Y)_{s,t} = (1/2) ∫_s^t (X_u - X_s) dY_u - (Y_u - Y_s) dX_u,
   что для уровня 2 сигнатуры эквивалентно
        A = (1/2) (S^{1,2} - S^{2,1}).
   Делим окно на K сегментов и берём сумму |A_k|·sign(A_k) либо вектор A_k;
   стандартизуем по скользящей σ и сравниваем с порогом ±1.96.

2) Ковариация приращений C_{s,t}
   C_{s,t} = Σ ΔX_i · ΔY_i. Если стандартизованная C проваливается ниже
   -1.96 σ — индексы движутся в противофазе, арбитражная связка ломается.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np

from config import CFG


@dataclass
class FilterStats:
    levy_z: float       # z-score сегментированной площади Леви (макс по сегментам)
    covar_z: float      # z-score ковариации приращений
    allow_entry: bool   # True => фильтр пропускает вход


class DoubleGeometricFilter:
    """
    Скользящие статистики площади Леви и ковариации приращений с
    адаптивными порогами (±1.96 σ относительно скользящего окна истории).
    """

    def __init__(
        self,
        window_size: int = CFG.WINDOW_SIZE,
        segments: int = CFG.LEVY_SEGMENTS,
        history_size: int = 500,
        levy_z_threshold: float = CFG.LEVY_Z_THRESHOLD,
        covar_z_threshold: float = CFG.COVAR_Z_THRESHOLD,
    ) -> None:
        self.window_size = window_size
        self.segments = segments
        self.levy_z_threshold = levy_z_threshold
        self.covar_z_threshold = covar_z_threshold
        # история скаляров — используется для статистических порогов
        self._levy_hist: Deque[float] = deque(maxlen=history_size)
        self._cov_hist: Deque[float] = deque(maxlen=history_size)

    # ------------------------------------------------------------ primitives
    @staticmethod
    def _segmented_levy_area(path2d: np.ndarray, K: int) -> np.ndarray:
        """
        Возвращает массив длины K с площадями Леви на K равных подсегментах
        двумерного пути.
        """
        N = path2d.shape[0]
        if N < 2:
            return np.zeros(K)
        edges = np.linspace(0, N - 1, K + 1, dtype=int)
        areas = np.empty(K, dtype=np.float64)
        for k in range(K):
            a, b = edges[k], edges[k + 1]
            if b - a < 1:
                areas[k] = 0.0
                continue
            seg = path2d[a:b + 1]
            # классическая формула Леви для дискретной траектории
            x0, y0 = seg[0]
            dx = np.diff(seg[:, 0])
            dy = np.diff(seg[:, 1])
            # ∫ (X - X_s) dY  — используем "левую" точку каждого шага
            xs = seg[:-1, 0] - x0
            ys = seg[:-1, 1] - y0
            areas[k] = 0.5 * float(np.sum(xs * dy - ys * dx))
        return areas

    @staticmethod
    def _increment_covariance(path2d: np.ndarray) -> float:
        if path2d.shape[0] < 2:
            return 0.0
        d = np.diff(path2d, axis=0)
        return float(np.sum(d[:, 0] * d[:, 1]))

    # ------------------------------------------------------------ scoring
    def update(self, path2d: np.ndarray) -> FilterStats:
        """
        Принимает текущее окно как 2D-путь (например, выход Lead-Lag(Z))
        и возвращает статистики плюс решение о допустимости входа.
        """
        levy_segments = self._segmented_levy_area(path2d, self.segments)
        # для скаляризации берём максимальную по модулю площадь сегмента —
        # она сигнализирует о локальной аномалии лидерства
        levy_scalar = float(levy_segments[np.argmax(np.abs(levy_segments))])

        cov_scalar = self._increment_covariance(path2d)

        self._levy_hist.append(levy_scalar)
        self._cov_hist.append(cov_scalar)

        levy_z = self._zscore(levy_scalar, self._levy_hist)
        cov_z = self._zscore(cov_scalar, self._cov_hist)

        allow = (
            abs(levy_z) < self.levy_z_threshold
            and cov_z > self.covar_z_threshold
        )
        return FilterStats(levy_z=levy_z, covar_z=cov_z, allow_entry=allow)

    @staticmethod
    def _zscore(value: float, history: Deque[float]) -> float:
        if len(history) < 30:
            return 0.0
        arr = np.fromiter(history, dtype=np.float64)
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=1))
        if sigma < 1e-12:
            return 0.0
        return (value - mu) / sigma
