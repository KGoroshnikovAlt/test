# Rough Path Statistical Arbitrage — Backtest

Минимальная end-to-end реализация системы арбитража на CFD-индексах
`#USNDAQ100` / `#USSPX500` через теорию грубых путей.

## Состав

| Модуль | Назначение |
|---|---|
| `config.py` | Параметры капитала, риска, окон и LS-решателя |
| `data_ingestor.py` | Загрузка CSV либо синтез коинтегрированной пары M1-баров |
| `kalman_filter.py` | Динамическая оценка β_t, α_t и чистый спред Z_t |
| `signature_engine.py` | Lead-Lag + усечённая сигнатура L=3 на чистом numpy |
| `double_filter.py` | Сегментированная площадь Леви + ковариация приращений |
| `solver_ls.py` | Лонгстафф–Шварц на сигнатурах (per-step ridge регрессия) |
| `risk_manager.py` | Маржа, лоты, drawdown-floor 850 USD |
| `main.py` | Сквозной бэктест |

## Запуск

```bash
pip install -r requirements.txt
python main.py
```

По умолчанию `data_ingestor.synthesize_cointegrated_pair` генерирует ~30 дней
1-минутных баров. Чтобы подставить свои данные, передайте `csv_y`/`csv_x` в
`load_bars()`.

## Параметры по умолчанию

* Стартовый капитал — 1000 USD, плечо 1:200, лот 0.01.
* Окно траектории — 30 мин, сигнатура — порядок 3 (14 коэффициентов).
* Горизонт удержания — 60 мин, training paths — 2000.
* Drawdown floor — 850 USD (15 % просадки).

## Что дальше

* `data_ingestor.py` — точка интеграции с MetaTrader5 / cTrader.
* `solver_ls.py` — обучение можно перенести в офлайн-скрипт и
  сериализовать `solver.alphas` в `.npz`.
* `double_filter.py` — пороги ±1.96 σ можно затянуть/расслабить под
  волатильностный режим.
