# Forex Backtest Engine

This project is a **backtest engine** for forex (and a few metals). A backtest is a test of trading rules on **old price data**. It is not live trading, and it does not place real orders.

This repository has **the engine only**. It does not include trading strategies. You write a strategy yourself and pass it to the engine.

**Setup (Python lab):** see [SETUP.md](SETUP.md) (ELI5: Miniconda + `fx-server37`).  
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

During `on_bar` the engine also:

- gives you a **copy** of that prefix (`window`), so changing it does not change later prices
- cuts `handler.data` and `strategy.data` to “now”
- still calls the strategy when you are at the open-trade cap; new entries are dropped, `EXIT` still queues

**Higher timeframes:** pass `htf_rules` (for example `("4H",)`) into `OOEngine`. Then read `account["htf"]["4H"]`. Those frames use completed candles only (`align_htf_pit`: shift one HTF bar, then forward-fill). An H4 labelled 00:00 (H1 00:00–03:00) is first visible on the **04:00** H1 bar; a signal there still fills at the **05:00** open.

Do not cache a full-sample resample in `__init__` and then take `.iloc[-1]` as “now”. Do not call `resample_ohlc(window)` and treat a forming 4h bar as closed. Use `account["htf"]` or `handler.resample_htf` **inside** `on_bar`. Details: [USAGE.md](USAGE.md#multi-timeframe-no-future-bars).

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
| `engine.py` | Main loop (`OOEngine`). t+1 fills, optional `htf_rules`, PIT fence. |
| `broker.py` | Order queue and fill rules. |
| `data_handler.py` | Bid/Ask CSV, pip/lot specs, as-of prefix, `pit_htf` / `resample_htf`. |
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

- `window`: all bars from the start **through the current bar** (a copy).
- `account`: equity, balance, open trades, and `htf` (empty unless you set `htf_rules`).

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

The tests check timing, Bid/Ask fills, fees, forming vs completed 4h bars, and that a strategy cannot see future bars.

## Honest limits

A backtest is a model. It can still be wrong if:

- your data has gaps or errors,
- you set fees and slippage to zero,
- your strategy was fitted too hard on the same data you test,
- you build your own higher-timeframe series from the **full** file and read future rows.

Sharpe and Sortino are yearly numbers. Pass bar timestamps (`index=index_from_result(result)`). The engine counts **how many of your bars fit in a calendar year**. Do not use 252 unless each equity point is a trading day. With no `index` and no `periods_per_year`, those ratios stay 0.

HTML trade charts may **draw** a few bars after the exit (`plot_trade` lookforward). That is a picture only, not a fill.

The engine tries not to hide those problems. It does not make a strategy profitable.
