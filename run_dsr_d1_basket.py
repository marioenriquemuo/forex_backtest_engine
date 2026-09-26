#!/usr/bin/env python
"""DSR + random search on Cross-Currency Basket Matrix D1 data.

Data: forex-research cleaned D1 Bid/Ask CSVs (same source as
investigations/Cross-Currency Basket Matrix). Saturday sessions dropped
(Sunday week-open kept), matching L99 D1 clock notes.

Run from BacktestEngine root:
  python run_dsr_d1_basket.py
"""

from __future__ import print_function

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_handler import BID_ASK_COLS, DataHandler, instrument_spec
from engine import OOEngine
from metrics import index_from_result
from orders import BUY, MARKET, SELL, Order
from stress import random_param_search
from strategy import Strategy

# Paths from Cross-Currency Basket Matrix config layout
FOREX_RESEARCH = Path("/home/mario/dev/forex-research")
CLEANED = FOREX_RESEARCH / "data" / "cleaned"
BASKET = FOREX_RESEARCH / "investigations" / "Cross-Currency Basket Matrix"

# Demo universe: liquid G10 majors from the basket list
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
START = "2015-01-01"
END = "2019-12-31"
N_TRIALS = 24
DSR_THRESHOLD = 0.90


def slug(pair):
    return pair.strip().replace("/", "-")


def d1_csv(pair):
    return CLEANED / ("D1-complete-%s.csv" % slug(pair))


def load_d1(pair, start=START, end=END):
    """Load D1 Bid/Ask; drop Saturday NY like phase2_d1.py."""
    path = d1_csv(pair)
    if not path.is_file():
        raise FileNotFoundError("Missing D1 CSV: %s" % path)
    handler = DataHandler.from_csv(path, pair=pair, start=start, end=end, freq=None)
    df = handler._data.copy()
    # Index is America/New_York after DataHandler
    df = df[df.index.dayofweek != 5]
    return DataHandler(df, pair=pair, freq=None)


def _wilder_atr(high, low, close, period=14):
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


class D1Breakout(Strategy):
    """D1 breakout: enter on close beyond lookback range; SL = ATR*mult; TP via rr."""

    def __init__(self, lookback, atr_mult, rr, max_hold, pair="EUR/USD"):
        super(D1Breakout, self).__init__()
        self.lookback = int(lookback)
        self.atr_mult = float(atr_mult)
        self.rr = float(rr)
        self.max_hold = int(max_hold)
        self.pair = pair

    def on_bar(self, window, account):
        need = self.lookback + 20
        if len(window) < need:
            return []
        active = account.get("active_trades") or []
        if any(t.get("Pair") == self.pair and t.get("Status") == "Active" for t in active):
            return []

        mid_h = (window["BidHigh"] + window["AskHigh"]) / 2.0
        mid_l = (window["BidLow"] + window["AskLow"]) / 2.0
        mid_c = (window["BidClose"] + window["AskClose"]) / 2.0
        atr = _wilder_atr(mid_h, mid_l, mid_c, 14)
        atr_now = float(atr.iloc[-1])
        if not np.isfinite(atr_now) or atr_now <= 0:
            return []

        # Prior lookback excludes current bar (no same-bar lookahead on the signal).
        prior = window.iloc[-(self.lookback + 1) : -1]
        hi = float(prior["BidHigh"].max())
        lo = float(prior["BidLow"].min())
        close_bid = float(window["BidClose"].iloc[-1])
        close_ask = float(window["AskClose"].iloc[-1])

        orders = []
        if close_bid > hi:
            # Long: fill next AskOpen; SL below by atr_mult*ATR
            entry_guess = close_ask  # pessimistic placeholder for SL geometry
            sl = entry_guess - self.atr_mult * atr_now
            orders.append(
                Order(
                    side=BUY,
                    order_type=MARKET,
                    lots=0.1,
                    sl=sl,
                    rr_ratio=self.rr,
                    max_hold_bars=self.max_hold,
                    pair=self.pair,
                    reason="D1BreakoutLong",
                )
            )
        elif close_ask < lo:
            entry_guess = close_bid
            sl = entry_guess + self.atr_mult * atr_now
            orders.append(
                Order(
                    side=SELL,
                    order_type=MARKET,
                    lots=0.1,
                    sl=sl,
                    rr_ratio=self.rr,
                    max_hold_bars=self.max_hold,
                    pair=self.pair,
                    reason="D1BreakoutShort",
                )
            )
        return orders


def make_run_fn(pair, df_cache):
    def run_fn(params):
        # Fresh handler + strategy every trial
        handler = DataHandler(df_cache.copy(), pair=pair, freq=None)
        strat = D1Breakout(
            lookback=params["lookback"],
            atr_mult=params["atr_mult"],
            rr=params["rr"],
            max_hold=params["max_hold"],
            pair=pair,
        )
        result = OOEngine(
            {pair: handler},
            {pair: strat},
            start_balance=100000.0,
            slippage_pips=1.0,
            max_risk_per_trade=10.0,
            max_active_trades_per_pair=1,
            commission_per_lot=0.0,
        ).run()
        return result

    return run_fn


def main():
    print("=== DSR D1 run — Cross-Currency Basket Matrix ===")
    print("basket dir:", BASKET)
    print("cleaned D1:", CLEANED)
    print("window:", START, "->", END)
    print("pairs:", PAIRS)
    print("n_trials:", N_TRIALS, "| DSR gate:", DSR_THRESHOLD)
    print()

    space = {
        "lookback": [10, 20, 40],
        "atr_mult": [1.0, 1.5, 2.0],
        "rr": [1.5, 2.0, 3.0],
        "max_hold": [5, 10, 20],
    }
    # 3^4 = 81 combos; sample N_TRIALS unique

    summaries = []
    for pair in PAIRS:
        print("----", pair, "----")
        handler0 = load_d1(pair)
        df_cache = handler0._data.copy()
        print("  bars:", len(df_cache), "| first:", df_cache.index[0], "| last:", df_cache.index[-1])
        idx = df_cache.index
        # ~252 D1 sessions / year
        out = random_param_search(
            make_run_fn(pair, df_cache),
            space,
            n_trials=N_TRIALS,
            rng=np.random.default_rng(42),
            index=idx,
            dsr_threshold=DSR_THRESHOLD,
        )
        dsr = out["dsr"]
        print("  N run:", out["n_trials"])
        print("  best:", out["best_params"])
        print("  best Sharpe:", round(out["best_sharpe"], 4))
        print("  DSR:", round(dsr["dsr"], 4), "| n_obs:", dsr["n_obs"])
        print("  passes?", out["passes"])
        summaries.append(
            {
                "pair": pair,
                "n_trials": out["n_trials"],
                "best_params": out["best_params"],
                "best_sharpe": out["best_sharpe"],
                "dsr": dsr["dsr"],
                "passes": out["passes"],
                "n_trades_proxy": len(out["best_equity"]) if out["best_equity"] else 0,
            }
        )
        print()

    print("=== summary ===")
    for s in summaries:
        flag = "KEEP" if s["passes"] else "REJECT"
        print(
            "%s  DSR=%.4f  Sharpe=%.4f  %s  best=%s"
            % (s["pair"], s["dsr"], s["best_sharpe"], flag, s["best_params"])
        )
    n_pass = sum(1 for s in summaries if s["passes"])
    print()
    print("%d / %d pairs passed DSR >= %.2f" % (n_pass, len(summaries), DSR_THRESHOLD))


if __name__ == "__main__":
    main()
