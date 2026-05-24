"""
solver_ls.py
============
Немарковское обобщение алгоритма Лонгстаффа–Шварца на признаках сигнатуры.

Идея
----
В задаче оптимального остановления нужно сравнить немедленный платёж g(τ) с
ожидаемой будущей стоимостью V(τ+1, ...). Для пути, у которого нет марковского
свойства (а спред Z_t — именно такой, благодаря fBM-подобной памяти), нам
нужны *функционалы пути*, а не точечные базисные функции.

Сигнатура пути даёт универсальный базис для непрерывных функционалов на
вариационных путях: любая регулярная функция от траектории приближается как
линейная комбинация компонент сигнатуры. Поэтому регрессоры =
[1, sig(path)] полностью описывают «информационное состояние» к моменту t.

Алгоритм
--------
Offline (fit):
    Для n = T-1, T-2, ..., 0:
        X_i = [1, signature(Z_i[0..n])]
        alpha_n = argmin_α Σ_i (V_{n+1, i} - <α, X_i>)^2 + λ ‖α‖²
        V_{n, i} = max(payoff_n_i, <alpha_n, X_i>)

Online (predict):
    Continuation_t = <alpha_t, [1, signature(Z_current[0..t])]>.
    Если payoff_t > Continuation_t — закрываем сделку.
"""

from __future__ import annotations
from typing import List

import numpy as np

from config import CFG
from signature_engine import (
    signature_of_spread,
    signature_dim,
    cumulative_signatures_of_spread,
)


class LongstaffSchwartzSignature:
    """
    Регрессионный решатель оптимального остановления на сигнатурах.

    Параметры
    ---------
    order : int
        Уровень усечения сигнатуры (= L).
    horizon : int
        Максимальное число минут удержания позиции.
    ridge : float
        Коэффициент L2-регуляризации (Tikhonov).
    """

    def __init__(
        self,
        order: int = CFG.SIGNATURE_ORDER,
        horizon: int = CFG.LS_HORIZON,
        ridge: float = CFG.LS_RIDGE,
    ) -> None:
        self.order = order
        self.horizon = horizon
        self.ridge = ridge
        # размерность пути после Lead-Lag = 2  =>  sig имеет sum(2^k) компонент
        self._sig_dim = signature_dim(order, 2)
        self.alphas: List[np.ndarray] = []  # длины horizon, каждое (sig_dim+1,)
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        training_spreads: np.ndarray,
        side: str = "long",
        verbose: bool = False,
    ) -> "LongstaffSchwartzSignature":
        """
        training_spreads : ndarray (N, horizon+1)
            Семейство возможных траекторий спреда после входа.
        side : 'long' или 'short'
            Направление сделки. 'long' = ставим на падение спреда (вошли при
            высоком Z), 'short' — наоборот. Платёж определяется как
                payoff_n = sign_factor * (Z_0 - Z_n).
        """
        assert training_spreads.ndim == 2
        N, Tp1 = training_spreads.shape
        T = Tp1 - 1
        if T < self.horizon:
            raise ValueError(
                f"Training paths must have length >= horizon+1 ({self.horizon + 1})."
            )
        T = self.horizon                                # обучаемся ровно на горизонте
        paths = training_spreads[:, : self.horizon + 1]

        sign_factor = 1.0 if side == "long" else -1.0
        z0 = paths[:, 0:1]
        payoff = sign_factor * (z0 - paths)             # (N, T+1)

        # Терминальная стоимость = ликвидация на горизонте
        V = payoff[:, T].copy()
        self.alphas = [None] * T

        D = self._sig_dim + 1                           # +1 для свободного члена
        I_ridge = self.ridge * np.eye(D)

        # Предварительно вычисляем cumulative-сигнатуры каждой обучающей траектории —
        # на инкрементальном Chen's identity. Стоимость: O(N * 2T) сегментов.
        if verbose:
            print(f"  precomputing cumulative signatures for {N} paths...")
        cum_sigs = [cumulative_signatures_of_spread(paths[i], self.order) for i in range(N)]

        for n in range(T - 1, -1, -1):
            X = np.zeros((N, D), dtype=np.float64)
            X[:, 0] = 1.0
            if n >= 1:
                for i in range(N):
                    sig = cum_sigs[i][n]
                    if sig is not None:
                        X[i, 1:] = sig
            # n == 0: окно из одной точки — оставляем только интерцепт.

            A = X.T @ X + I_ridge
            b = X.T @ V
            alpha = np.linalg.solve(A, b)
            self.alphas[n] = alpha

            continuation = X @ alpha
            V = np.maximum(payoff[:, n], continuation)

            if verbose and (n % 10 == 0 or n == T - 1):
                print(f"  LS fit step n={n:3d}  E[V]={V.mean():+.6f}  ‖α‖={np.linalg.norm(alpha):.3e}")

        self._fitted = True
        return self

    # ------------------------------------------------------------ continuation
    def continuation_value(self, z_window: np.ndarray, n: int) -> float:
        """
        Текущее окно спреда Z[0..n] -> ожидаемая стоимость продолжения.
        Если n ≥ horizon, возвращаем -∞ (форсируем выход).
        """
        if not self._fitted:
            raise RuntimeError("Solver is not fitted. Call .fit() first.")
        if n >= self.horizon:
            return -np.inf
        if n < 0:
            return 0.0
        x = np.zeros(self._sig_dim + 1, dtype=np.float64)
        x[0] = 1.0
        if n >= 1 and z_window.size >= 2:
            x[1:] = signature_of_spread(z_window[: n + 1], self.order)
        return float(x @ self.alphas[n])

    # --------------------------------------------------------- helper datasets
    @staticmethod
    def build_training_windows(z_series: np.ndarray, horizon: int, stride: int = 1) -> np.ndarray:
        """
        Нарезает длинный массив спреда z_series на перекрывающиеся окна длины
        horizon+1. Удобно для синтеза training-датасета из исторического Z.
        """
        z_series = np.asarray(z_series, dtype=np.float64).ravel()
        n_total = z_series.size
        if n_total < horizon + 1:
            raise ValueError("Series too short for given horizon.")
        starts = np.arange(0, n_total - horizon, stride)
        out = np.empty((starts.size, horizon + 1), dtype=np.float64)
        for j, s in enumerate(starts):
            out[j] = z_series[s : s + horizon + 1]
        return out
