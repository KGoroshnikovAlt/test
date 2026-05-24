"""
data_ingestor.py
================
Поставщик минутных баров для бэктеста.

В production-режиме сюда подключается MetaTrader5 / cTrader; в режиме бэктеста
мы либо читаем CSV (если задан путь), либо синтезируем коинтегрированную пару
индексов USNDAQ100 / USSPX500 с реалистичными параметрами:

    log P_x_t = log P_x_{t-1} + mu_x dt + sigma_x sqrt(dt) eps_x_t
    log P_y_t = beta * log P_x_t + alpha + u_t
    u_t      = phi * u_{t-1} + sigma_u sqrt(dt) eps_u_t      (OU/AR(1))

Это даёт пару с дрейфующим спредом, идеально подходящую для тестирования
динамического фильтра Калмана и алгоритма оптимального остановления.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MarketBars:
    """Контейнер для синхронизированных минутных баров двух инструментов."""
    timestamps: pd.DatetimeIndex
    price_y: np.ndarray   # цена закрытия Nasdaq
    price_x: np.ndarray   # цена закрытия S&P 500

    def __len__(self) -> int:
        return len(self.timestamps)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"y": self.price_y, "x": self.price_x},
            index=self.timestamps,
        )


def load_from_csv(path_y: str | Path, path_x: str | Path) -> MarketBars:
    """
    Загружает два CSV-файла с колонками ['time', 'close'] и приводит их
    к общему индексу по времени (inner join).
    """
    df_y = pd.read_csv(path_y, parse_dates=["time"]).set_index("time")
    df_x = pd.read_csv(path_x, parse_dates=["time"]).set_index("time")
    df = df_y[["close"]].rename(columns={"close": "y"}).join(
        df_x[["close"]].rename(columns={"close": "x"}), how="inner"
    ).dropna()
    return MarketBars(
        timestamps=df.index,
        price_y=df["y"].to_numpy(dtype=np.float64),
        price_x=df["x"].to_numpy(dtype=np.float64),
    )


def synthesize_cointegrated_pair(
    n_bars: int = 60 * 24 * 30,        # ~ 30 торговых дней по 1 мин
    start_price_y: float = 24_000.0,    # типичный уровень NDX
    start_price_x: float = 5_800.0,     # типичный уровень SPX
    true_beta: float = 3.5,             # ведомый ≈ 3.5 * ведущий (в log-пространстве)
    mu_x: float = 0.04,                 # годовой дрейф
    sigma_x: float = 0.18,              # годовая волатильность ведущего
    sigma_u: float = 0.0008,            # волатильность остатка (минутная)
    phi: float = 0.985,                 # коэффициент возврата к среднему (AR(1))
    seed: int = 42,
) -> MarketBars:
    """
    Возвращает синтезированную пару с заведомо известной коинтеграцией.
    Параметры по умолчанию подобраны так, чтобы дать стационарный спред с
    периодом возврата к среднему порядка 60–90 минут.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / (252 * 24 * 60)          # шаг = 1 минута, выраженный в годах

    # Лог-доходности ведущего (Brownian motion)
    eps_x = rng.standard_normal(n_bars)
    log_returns_x = mu_x * dt + sigma_x * np.sqrt(dt) * eps_x
    log_x = np.cumsum(log_returns_x) + np.log(start_price_x)

    # AR(1)-остаток (OU-процесс в дискретном времени)
    eps_u = rng.standard_normal(n_bars)
    u = np.empty(n_bars)
    u[0] = 0.0
    for t in range(1, n_bars):
        u[t] = phi * u[t - 1] + sigma_u * eps_u[t]

    # log P_y = beta * log P_x + alpha + u, alpha подобрана так, чтобы цена
    # стартовала рядом со start_price_y
    alpha = np.log(start_price_y) - true_beta * np.log(start_price_x)
    log_y = true_beta * log_x + alpha + u

    timestamps = pd.date_range(
        start="2024-01-02 09:30:00", periods=n_bars, freq="1min"
    )
    return MarketBars(
        timestamps=timestamps,
        price_y=np.exp(log_y),
        price_x=np.exp(log_x),
    )


def load_bars(
    csv_y: Optional[str] = None,
    csv_x: Optional[str] = None,
    **synth_kwargs,
) -> MarketBars:
    """Универсальная точка входа: CSV если заданы оба пути, иначе синтетика."""
    if csv_y and csv_x:
        return load_from_csv(csv_y, csv_x)
    return synthesize_cointegrated_pair(**synth_kwargs)
