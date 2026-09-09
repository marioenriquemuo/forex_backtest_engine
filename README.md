# Forex Backtest Engine

This project is a **backtest engine** for forex (and a few metals). A backtest is a test of trading rules on **old price data**. It is not live trading, and it does not place real orders.

This repository has **the engine only**. It does not include trading strategies. You write a strategy yourself and pass it to the engine.

**How to use it:** see [USAGE.md](USAGE.md) (step-by-step, with a small example).

## What the engine does

You give it:

1. Bid and Ask OHLC bars (open, high, low, close for both sides of the spread).
2. A strategy that, on each bar, can return zero or more **orders**.
3. Account settings such as start balance, leverage, slippage, and fees.

It then walks through time, bar by bar, and reports closed trades, an equity curve, and simple performance numbers.

## Timing: no future data

A strategy only sees bars **up to the current bar**. It cannot read tomorrow.

The signal is taken on the **close of bar t**. The fill happens from the **open of bar t+1**. That delay is on purpose. Filling at the same close that created the signal would make results look better than real life.

Higher-timeframe data is also delayed until that higher bar is finished. The engine does not let you use a 4-hour candle before it has closed.

## Bid and Ask

The engine does **not** use one mid price.

- A **buy** enters on the **Ask**.
- A **sell** enters on the **Bid**.
- A long position exits on the **Bid**.
- A short position exits on the **Ask**.

So the spread is part of the result. If you mix Bid and Ask, or ignore the spread, the test is not honest.

Price times use the **America/New_York** timezone.

## How fills work (pessimistic)

The broker rules try to be **strict**, not kind:

- Extra **slippage** can move the fill against you.
- If stop-loss and take-profit are both hit in the same bar, **stop-loss wins**.
- If the market **gaps** through your stop or your pending price, the fill is at the **open**, not at your exact price.
- A limit order fills only if the bar actually trades through that price.

These rules can make results worse. That is the point: a backtest that is too kind is not useful.

## Money and risk

The portfolio tracks:

- **Balance** and **equity** (equity includes open profit and loss).
- **Commission** (per lot and/or a percent of notional).
- **Swap** (overnight fee), charged at 17:00 New York, with a triple charge on Wednesday.
- **Margin**. If equity falls too far, positions are closed (stop-out).
- **Position size** from risk, if you set a stop-loss and do not set lots yourself.

Numbers you see are **net** of these costs when you turn them on. Do not treat a run with zero fees and zero slippage as a real-world result.

## Main parts

| File | Role |
| --- | --- |
| `engine.py` | Main loop (`OOEngine`). Runs strategies and the broker together. |
| `broker.py` | Order queue and fill rules. |
| `data_handler.py` | Loads Bid/Ask CSV data, pair names, pip size, lot size. |
| `portfolio.py` | Cash, positions, fees, swap, margin. |
| `orders.py` | Order, fill, and trade records. |
| `strategy.py` | Strategy interface. New code should use `on_bar`. |
| `engine_jit.py` | Faster replay of **already known** orders (same fill rules). |
| `metrics.py` | Profit, drawdown, Sharpe, win rate, and similar stats. |
| `reporting.py` | CSV/JSON logs and HTML charts. |
| `stress.py` | Extra tests: more slippage, rejected entries, bootstrap of returns. |

There are also small **shim** files (`backtest_fun_consistent.py` and similar). They keep old import names working. New code should use `OOEngine`, `DataHandler`, and `Strategy`.

## How to write a strategy

Subclass `Strategy` and implement `on_bar`. You receive:

- `window`: all bars from the start **through the current bar**.
- `account`: equity, balance, and open trades.

Return a list of `Order` objects (market, limit, stop, trailing stop, or exit). Return an empty list if you do nothing.

This repo does not ship a real strategy. The tests use tiny fake strategies only to check the engine.

## Data format

CSV data must have these columns:

`BidOpen`, `BidHigh`, `BidLow`, `BidClose`, `AskOpen`, `AskHigh`, `AskLow`, `AskClose`

You can also load separate Bid and Ask files with `DataHandler.from_bid_ask_csv`.

## Tests

From this folder:

```bash
python -m pytest
```

The tests check timing, Bid/Ask fills, fees, and that a strategy cannot see future bars.

## Honest limits

A backtest is a model. It can still be wrong if:

- your data has gaps or errors,
- you set fees and slippage to zero,
- your strategy was fitted too hard on the same data you test.

The engine tries not to hide those problems. It does not make a strategy profitable.
