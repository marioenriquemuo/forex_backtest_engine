# How to use this engine (ELI5)

Think of three people:

1. **You** write the rules (the strategy).
2. The **data** is an old video of prices.
3. The **engine** is the referee. It watches the video in order, applies your rules, and keeps score.

You do **not** trade real money. You replay the past.

This file is a step-by-step. Do the steps in order the first time.

---

## 0. Setup checklist (do this first)

1. Open a terminal **inside this folder** (`BacktestEngine`).
   Imports look like `from engine import OOEngine`. They only work if Python can see these files.
2. Use the Python that already has `pandas` (this research env is Python 3.7 + pandas 1.3). You need `plotly` only if you want HTML charts.
3. Check the engine is healthy:

```bash
python -m pytest
```

If tests pass, the engine works on this machine. If `python` is missing, try `python3` or activate your conda env first.

4. Put your price file somewhere you can find (see the CSV shape below).
5. Write a small script in **this same folder** (for example `run_once.py`).
6. Run it from this folder:

```bash
python run_once.py
```

If you run the script from another folder, Python will say it cannot import `engine`. Either `cd` here, or add this folder to `PYTHONPATH`.

---

## 1. How to call the engine

You always do the same four things:

1. Load prices into a `DataHandler`.
2. Write a `Strategy` with `on_bar`.
3. Build `OOEngine` with **matching pair names** for data and strategy.
4. Call `.run()`.

That is the whole call:

```python
from data_handler import DataHandler
from engine import OOEngine
from strategy import Strategy

handler = DataHandler.from_csv("eurusd.csv", pair="EUR/USD")
engine = OOEngine(
    handlers={"EUR/USD": handler},
    strategies={"EUR/USD": MyStrategy()},
    start_balance=100000.0,
    htf_rules=("4H",),  # optional; omit if you only use the chart timeframe
)
result = engine.run()
```

The keys `"EUR/USD"` must match. The engine cleans names (`eur-usd` becomes `EUR/USD`), but it is simpler if you write them the same way every time.

### Settings you can pass to `OOEngine`

These are the real arguments. Defaults are shown.

| Argument | Default | Meaning |
| --- | --- | --- |
| `handlers` | required | Dict: pair name → `DataHandler` |
| `strategies` | required | Dict: pair name → `Strategy` |
| `start_balance` | required | Starting cash |
| `slippage_pips` | `0.0` | Extra pips against you on market fills |
| `max_active_trades_per_pair` | `2` | Cap on open trades per pair. The strategy is still called; new entries are dropped, `EXIT` still queues |
| `leverage` | `30.0` | Margin leverage |
| `stop_out_fraction` | `0.5` | If equity drops below `used_margin * 0.5`, all positions close (`Margin`) |
| `commission_per_lot` | `0.0` | Cash fee per lot at open |
| `commission_pct` | `0.0` | Fee as a fraction of notional at open |
| `swap_long_per_lot` | `0.0` | Overnight fee for longs (17:00 New York; triple on Wednesday) |
| `swap_short_per_lot` | `0.0` | Overnight fee for shorts |
| `max_risk_per_trade` | `0.01` | If you omit `lots`, size is 1% of balance × `risk_ratio` on the order |
| `contract_size` | `None` | Override lot size; otherwise the pair table is used (FX is usually 100000) |
| `extra_slippage_pips` | `0.0` | Extra random-run slippage (used by stress tests) |
| `reject_entry_rate` | `0.0` | Chance to skip an entry (needs `rng`) |
| `rng` | `None` | Random generator for rejects / stress |
| `htf_rules` | `()` | Extra timeframes to attach on `account["htf"]`, e.g. `("4H", "1D")`. Point-in-time only |

**Easy first run:** set `start_balance`, leave fees at zero, and set `lots` on the order yourself. Then turn on slippage and commission when you want a harder test.

You can run **more than one pair**. Give each pair its own handler and its own strategy object. The engine walks a shared time index.

---

## 2. What the engine is expecting

### 2.1 Price data

The engine wants **Bid and Ask** candles, not one mid price.

**CSV file** (`DataHandler.from_csv`):

- A `Date` column (parsed as UTC, then converted to America/New_York).
- These eight columns, with **exactly** these names:

`BidOpen`, `BidHigh`, `BidLow`, `BidClose`, `AskOpen`, `AskHigh`, `AskLow`, `AskClose`

Optional: `Volume`.

Example CSV:

```text
Date,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose
2024-01-02 15:00:00+00:00,1.1000,1.1010,1.0990,1.1005,1.1002,1.1012,1.0992,1.1007
2024-01-02 16:00:00+00:00,1.1005,1.1020,1.1000,1.1010,1.1007,1.1022,1.1002,1.1012
2024-01-02 17:00:00+00:00,1.1010,1.1015,1.0995,1.1000,1.1012,1.1017,1.0997,1.1002
```

Load it:

```python
handler = DataHandler.from_csv(
    "eurusd.csv",
    pair="EUR/USD",
    start="2024-01-01",   # optional
    end="2024-12-31",     # optional
)
```

If Bid and Ask live in two files:

```python
handler = DataHandler.from_bid_ask_csv("bid.csv", "ask.csv", pair="EUR/USD")
```

If you already have a pandas DataFrame, you can skip the CSV:

```python
handler = DataHandler(df, pair="EUR/USD")
```

`df` must have a datetime index and the same eight Bid/Ask columns.

If a required column is missing, the engine **raises an error**. It does not guess.

### 2.2 A strategy (`on_bar`)

Subclass `Strategy`. On every bar the engine calls:

```python
orders = strategy.on_bar(window, account)
```

**`window`**

- All bars from the start **through the current bar**.
- Last row = this bar. There is no next bar in `window`.
- It is a **copy**. Changing `window` does not change the engine’s stored prices.
- Use `window.iloc[-1]` for “now”. Do not try `window.iloc[len(window)]`.

During this call the engine also cuts `handler.data` and `strategy.data` to the same last timestamp, then puts the full series back afterwards.

**`account`** (a dict)

| Key | What it is |
| --- | --- |
| `equity` | Cash + floating P&L |
| `balance` | Closed cash |
| `active_trades` | List of open-trade dicts (`Type`, `Entry`, `SL`, `TP`, `Lots`, `Pair`, …) |
| `positions` | The same opens as `Position` objects |
| `htf` | Dict of higher-timeframe frames, same LTF index, **completed bars only**. Empty if you omit `htf_rules` |

**Return value**

- A **list** of `Order` objects.
- `[]` means “do nothing this bar”.
- Anything that is not an `Order` is ignored.

The engine queues those orders on bar `t`. It tries to fill them from the **open of bar t+1**. So a market buy on bar 0 is **not** filled at bar 0 close.

### 2.3 An `Order`

```python
from orders import BUY, SELL, MARKET, LIMIT, STOP, TRAILING_STOP, EXIT, Order

Order(
    side=BUY,              # BUY is "B", SELL is "S"
    order_type=MARKET,     # or LIMIT, STOP, TRAILING_STOP, EXIT
    lots=0.10,             # optional if you set sl (then size comes from risk)
    sl=1.0950,             # stop loss price
    tp=1.1100,             # take profit price (optional)
    price=None,            # needed for LIMIT / STOP
    trail_pips=None,       # for TRAILING_STOP
    max_hold_bars=None,    # close after N bars
    expire_bars=None,      # pending order dies after N bars
    risk_ratio=1.0,        # scales risk when lots is omitted
    rr_ratio=None,         # if set and tp is missing, tp is built from sl distance
    pair="EUR/USD",        # optional; engine fills this from the dict key
    reason="my signal",    # label stored on the trade
)
```

Simple rules:

- `side` must be `"B"` or `"S"` (use `BUY` / `SELL`).
- For a **long**, `sl` must be **below** entry. For a **short**, `sl` must be **above** entry. Bad stops are rejected.
- If `lots` is `None`, you **must** give `sl`. Size uses `max_risk_per_trade`.
- `LIMIT` / `STOP` need `price`. They fill only if the next bars trade through that price.

### 2.4 Simple example (copy and run)

Save this as `run_once.py` in this folder. It uses three fake hourly bars. You do not need a CSV for this first test.

```python
import pandas as pd

from data_handler import DataHandler
from engine import OOEngine
from metrics import compute_metrics
from orders import BUY, MARKET, Order
from reporting import export_logs
from strategy import Strategy


class BuyOnce(Strategy):
    """On the first bar, ask for one small buy. Then do nothing."""

    def __init__(self):
        super(BuyOnce, self).__init__()
        self.done = False

    def on_bar(self, window, account):
        if self.done:
            return []
        self.done = True
        last = window.iloc[-1]
        # Signal on this close; fill is at the *next* Ask open.
        return [
            Order(
                side=BUY,
                order_type=MARKET,
                lots=0.10,
                sl=float(last["BidClose"]) - 0.0100,
                tp=float(last["BidClose"]) + 0.0200,
                reason="buy-once-demo",
            )
        ]


def make_tiny_data():
    idx = pd.date_range(
        "2024-01-02 10:00", periods=3, freq="H", tz="America/New_York"
    )
    rows = [
        # bid o/h/l/c , ask o/h/l/c
        [1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007],
        [1.1005, 1.1005, 1.1005, 1.1005, 1.1007, 1.1007, 1.1007, 1.1007],
        [1.0900, 1.1300, 1.0890, 1.1200, 1.0902, 1.1302, 1.0892, 1.1202],
    ]
    cols = [
        "BidOpen", "BidHigh", "BidLow", "BidClose",
        "AskOpen", "AskHigh", "AskLow", "AskClose",
    ]
    return pd.DataFrame(rows, index=idx, columns=cols)


if __name__ == "__main__":
    pair = "EUR/USD"
    handler = DataHandler(make_tiny_data(), pair=pair)
    engine = OOEngine(
        handlers={pair: handler},
        strategies={pair: BuyOnce()},
        start_balance=100000.0,
        slippage_pips=0.0,
        max_active_trades_per_pair=2,
    )
    result = engine.run()

    print("final balance:", result.balance)
    print("n trades:", len(result.trades))
    if result.trades:
        t = result.trades[0]
        print("entry:", t["Entry"], "exit:", t["Exit_Price"], "why:", t["Exit_Reason"])
        print("pnl:", t["Result"])

    stats = compute_metrics(result.trades, result.equity_curve, start_balance=100000.0)
    print("net profit:", stats["total_net_profit"])
    print("win rate:", stats["win_rate"])

    paths = export_logs(result, "out")
    print("files:", paths)
```

What this example does, in order:

1. Bar 0: strategy sees the first candle and queues a buy.
2. Bar 1: broker fills the buy at **AskOpen** `1.1007` (not at bar 0 close).
3. Bar 2: Bid opens at `1.0900`, already through the stop `1.0905`, so the exit is the **Bid open** (gap rule), not the exact stop.

That is enough to prove you called the engine correctly. Then replace `make_tiny_data()` with `DataHandler.from_csv(...)` and replace `BuyOnce` with your real rules.

### 2.5 Higher timeframe (optional)

```python
engine = OOEngine(
    handlers={pair: handler},
    strategies={pair: MyStrategy()},
    start_balance=100000.0,
    htf_rules=("4H",),
)

# inside on_bar:
h4 = account["htf"]["4H"]
h4_close = h4.iloc[-1]["BidClose"]  # NaN until that 4h candle is finished
```

Use this dict (or `handler.resample_htf("4H")` **inside** `on_bar`). Do not resample the whole file once at start-up and keep it.

---

## 3. What the engine is delivering

`engine.run()` returns an `EngineResult`.

### 3.1 `result.trades` — closed trades (list of dicts)

Each dict looks like this:

| Key | Meaning |
| --- | --- |
| `Date` | Entry time |
| `Exit_Date` | Exit time |
| `Entry_Bar_Index` / `Exit_Bar_Index` | Bar numbers |
| `Type` | `"B"` or `"S"` |
| `Pair` | Pair name |
| `Entry` / `Exit_Price` | Fill prices |
| `SL` / `TP` | Stop and target |
| `Lots` | Size |
| `Result` | Net P&L for that trade (includes commission/swap on the trade) |
| `Balance` | Account balance after the close |
| `Exit_Reason` | Why it closed (`SL`, `TP`, and other engine reasons) |
| `Entry_Reason` | Your `order.reason` |
| `Commission` | Fee charged at open |
| `Swap` | Overnight charge |
| `Slippage_Paid_Pips` | Slippage tracked on the trade |
| `MAE` / `MFE` | Worst / best excursion while open |
| `Status` | `"Closed"` |
| `Features` | Whatever you put on the order (or `None`) |

Open trades that never close are **not** in this list. They still affect `equity_curve` while they are open.

### 3.2 `result.equity_curve` — score over time

A list of floats. One point per processed bar (starts with `start_balance`). This is **equity**, not only closed balance: open P&L is included.

### 3.3 `result.balance` — cash at the end

Final closed cash. If a trade is still open at the last bar, equity and balance can differ.

### 3.4 `result.queue_log` — what you asked for

A list of dicts: each order the strategy queued (`created_bar`, `side`, `order_type`, prices, lots, pair, reason). Use this to check “did my strategy fire?” even if nothing filled.

### 3.5 `result.closed` and `result.handlers`

- `closed`: same trades as objects (`ClosedTrade`), if you need attributes instead of dicts.
- `handlers`: the `DataHandler`s you passed in (useful for plots).

`result.fills` is currently unused by `run()` (it stays empty). Use `trades` and `queue_log` instead.

### 3.6 Numbers and files (optional next step)

```python
from metrics import compute_metrics
from reporting import export_logs, plot_dashboard, plot_trade

stats = compute_metrics(result.trades, result.equity_curve, start_balance=100000.0)
# stats keys include: total_net_profit, cagr, payoff_ratio, expectancy_currency,
# expectancy_pips, max_drawdown, max_drawdown_pct, ulcer_index, sharpe, sortino,
# calmar, trades, win_rate, mae_mean, mfe_mean

paths = export_logs(result, "out")
# writes: out/trade_log.csv, out/equity_curve.csv, out/order_queue.json, out/run_summary.json

plot_dashboard(result, "out/dash.html")
if result.trades:
    plot_trade(handler, result.trades[0], "out/trade.html")
    # lookforward=5 draws extra candles after the exit for the picture only
```

Those HTML files need `plotly`.

---

## Easy-use reminders

1. Work **in this folder**.
2. Pair names in `handlers` and `strategies` must match.
3. CSV must have `Date` plus eight Bid/Ask columns. Missing columns fail loudly.
4. `on_bar` must return a **list of `Order`**. Returning a dict does nothing.
5. A signal on bar `t` fills from bar `t+1` open. If you expect the same-bar close, you will think the engine is “wrong”.
6. Buys fill on **Ask**. Sells fill on **Bid**.
7. If stop and target hit in the same bar, **stop wins**.
8. If you leave `lots=None`, you must set `sl`, and size follows `max_risk_per_trade`.
9. `max_active_trades_per_pair` blocks **new entries** when you are at the cap. `EXIT` still works. Raise the cap while you debug.
10. First run: tiny data, one order, print `result.trades` and `result.queue_log`. Then use your CSV.
11. For 4h/daily filters, set `htf_rules` and read `account["htf"]`. Do not keep a full-file HTF cache.

### Multi-timeframe (no future bars)

The engine will not let a higher timeframe see a candle before that candle is **finished**.

- Pass `htf_rules=("4H",)` (and/or `"1D"`, etc.) into `OOEngine`.
- In `on_bar`, read `account["htf"]["4H"].iloc[-1]`. That row is aligned to the current LTF bar.
- An H4 bar labelled 00:00 uses H1 00:00–03:00 and is first visible on the **04:00** H1 bar (`align_htf_pit`: shift one HTF bar, then forward-fill). A signal there still fills at the **next** H1 open (05:00).
- During `on_bar`, `handler.data` and `strategy.data` stop at the current bar. After the call they are restored.
- Do **not** cache `handler.resample_htf(...)` on the full file in `__init__` and then take `.iloc[-1]` as “now”.
- Do **not** call `resample_ohlc(window)` and treat a forming 4h bar (01:00 / 02:00 / 03:00) as a closed 4h candle. `account["htf"]` and `handler.resample_htf` already apply the PIT shift.

`plot_trade(..., lookforward=5)` may draw later candles. That is a chart only. Do not use it as signal input.

The engine does not include a trading strategy. Your `on_bar` **is** the strategy.
