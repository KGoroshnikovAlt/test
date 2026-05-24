"""
signature_engine.py
===================
Lead-Lag трансформация и расчёт усечённой сигнатуры пути.

Сигнатура пути X: [0, T] → R^d уровня L — это конкатенация итерированных
интегралов
    S^(k)(X)_{0,T}^{i_1...i_k} = ∫_{0<t_1<...<t_k<T} dX^{i_1}_{t_1} ... dX^{i_k}_{t_k},
для k = 1, ..., L и i_j ∈ {1, ..., d}.

Реализация — на чистом numpy через Chen's identity:
  для каждого кусочно-линейного сегмента с приращением v ∈ R^d сигнатура
  считается аналитически: S^(k)_w(v) = (v_{w_1} ⋯ v_{w_k}) / k!, после чего
  сегментные сигнатуры перемножаются в усечённой тензорной алгебре.

Это полностью эквивалентно iisignature.sig(path, order) для кусочно-линейных
путей и не требует внешних бинарных зависимостей.
"""

from __future__ import annotations
from functools import lru_cache
from itertools import product
from math import factorial
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Lead-Lag
# ---------------------------------------------------------------------------

def lead_lag(z: np.ndarray) -> np.ndarray:
    """
    Lead-Lag трансформация одномерного ряда z[0..N-1].

    На выходе путь в R^2 длины 2N-1 со следующей структурой по сетке t = k/2:
        path[2i]   = (z_i,   z_i)
        path[2i+1] = (z_{i+1}, z_i)
    То есть на каждом полу-шаге увеличивается только одна координата ("lead"
    идёт впереди, "lag" — позади), что предотвращает древовидное зануление
    итерированных интегралов на осциллирующих участках.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    n = z.size
    if n < 2:
        raise ValueError("Lead-Lag transform requires at least 2 points.")
    path = np.empty((2 * n - 1, 2), dtype=np.float64)
    path[0::2, 0] = z          # точки 2i: lead = z_i
    path[0::2, 1] = z          # точки 2i: lag  = z_i
    path[1::2, 0] = z[1:]      # точки 2i+1: lead = z_{i+1}
    path[1::2, 1] = z[:-1]     # точки 2i+1: lag  = z_i
    return path


# ---------------------------------------------------------------------------
# Индексация слов в плоском представлении сигнатуры
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def signature_dim(order: int, d: int) -> int:
    return sum(d ** k for k in range(1, order + 1))


@lru_cache(maxsize=64)
def _level_offsets(order: int, d: int) -> Tuple[int, ...]:
    """Кумулятивный сдвиг для каждого уровня k = 1..order в плоском векторе."""
    offsets = []
    s = 0
    for k in range(1, order + 1):
        offsets.append(s)
        s += d ** k
    return tuple(offsets)


@lru_cache(maxsize=64)
def _words_of_length(k: int, d: int) -> Tuple[Tuple[int, ...], ...]:
    return tuple(product(range(d), repeat=k))


def _word_index(word: Tuple[int, ...], order: int, d: int) -> int:
    k = len(word)
    offset = _level_offsets(order, d)[k - 1]
    idx_within = 0
    for digit in word:
        idx_within = idx_within * d + digit
    return offset + idx_within


# ---------------------------------------------------------------------------
# Сигнатура одного кусочно-линейного сегмента
# ---------------------------------------------------------------------------

def _segment_signature(increment: np.ndarray, order: int) -> np.ndarray:
    """
    Аналитическая сигнатура отрезка прямой с приращением v ∈ R^d:
        S^(k)_w(v) = (Π_i v_{w_i}) / k!.
    """
    v = np.asarray(increment, dtype=np.float64).ravel()
    d = v.size
    sig = np.zeros(signature_dim(order, d), dtype=np.float64)
    for k in range(1, order + 1):
        denom = float(factorial(k))
        for word in _words_of_length(k, d):
            prod_val = 1.0
            for digit in word:
                prod_val *= v[digit]
            sig[_word_index(word, order, d)] = prod_val / denom
    return sig


# ---------------------------------------------------------------------------
# Tensor-product / Chen's identity в усечённой тензорной алгебре
# ---------------------------------------------------------------------------

def _chen_product(S: np.ndarray, T: np.ndarray, order: int, d: int) -> np.ndarray:
    """
    Произведение S ⊗ T в T^L(R^d), где предполагается, что нулевой уровень
    обоих сомножителей равен 1.

    Формула:  (S ⊗ T)_w = Σ_{w = u · v} S_u T_v,  где S_ε = T_ε = 1.
    """
    R = np.zeros_like(S)
    for k in range(1, order + 1):
        for word in _words_of_length(k, d):
            acc = 0.0
            # перебор всех расщеплений word на u и v
            for split in range(k + 1):
                u = word[:split]
                v = word[split:]
                s_u = 1.0 if not u else S[_word_index(u, order, d)]
                t_v = 1.0 if not v else T[_word_index(v, order, d)]
                acc += s_u * t_v
            R[_word_index(word, order, d)] = acc
    return R


# ---------------------------------------------------------------------------
# Сигнатура всего кусочно-линейного пути
# ---------------------------------------------------------------------------

def path_signature(path: np.ndarray, order: int) -> np.ndarray:
    """
    Сигнатура кусочно-линейного пути path: ndarray (N, d) до уровня order.
    """
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2:
        raise ValueError("path must have shape (N>=2, d).")
    d = path.shape[1]
    increments = np.diff(path, axis=0)

    # начальная сигнатура — сегмент 0
    sig = _segment_signature(increments[0], order)
    # последовательно умножаем на сигнатуры остальных сегментов
    for i in range(1, len(increments)):
        seg = _segment_signature(increments[i], order)
        sig = _chen_product(sig, seg, order, d)
    return sig


# ---------------------------------------------------------------------------
# Удобная высокоуровневая функция: сигнатура спреда через Lead-Lag
# ---------------------------------------------------------------------------

def signature_of_spread(z_window: np.ndarray, order: int) -> np.ndarray:
    """
    Полный pipeline для одномерного спреда: Lead-Lag → 2D-путь → сигнатура.
    Возвращает плоский вектор размерности sum(2^k для k=1..L).
    Для L=3 это ровно 2 + 4 + 8 = 14 коэффициентов.
    """
    path = lead_lag(z_window)
    return path_signature(path, order)


def cumulative_signatures_of_spread(z_path: np.ndarray, order: int) -> list:
    """
    Возвращает list `cum` такой, что `cum[n] = signature_of_spread(z_path[:n+1], order)`
    для n = 1, ..., len(z_path) - 1. На индексе n = 0 хранится None (сигнатура
    одной точки не определена).

    Считается инкрементально через Chen's identity, что даёт значительный
    выигрыш в скорости при оффлайн-обучении Лонгстаффа–Шварца, где сигнатуры
    префиксов одной и той же траектории нужны на каждом шаге обратной индукции.
    """
    z_path = np.asarray(z_path, dtype=np.float64).ravel()
    N = z_path.size
    ll = lead_lag(z_path)                        # (2N-1, 2)
    increments = np.diff(ll, axis=0)             # (2N-2, 2)
    d = 2
    cum: list = [None] * N
    sig = None
    for k in range(2 * (N - 1)):
        seg = _segment_signature(increments[k], order)
        sig = seg if sig is None else _chen_product(sig, seg, order, d)
        # после k+1 сегментов = после 2(n+1)-1 - 1 = 2n+1 - 1 = 2n сегментов
        # => n = (k+1)/2, целое только когда (k+1) чётное.
        if (k + 1) % 2 == 0:
            n = (k + 1) // 2
            cum[n] = sig.copy()
    return cum
