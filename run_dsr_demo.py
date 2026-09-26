#!/usr/bin/env python
"""Demo: random knob search + DSR gate (see DSR.md).

Run from the repo root:
  python run_dsr_demo.py
"""

from __future__ import print_function

import numpy as np

from data_handler import DataHandler
from engine import OOEngine
from metrics import dsr_from_equity
from orders import BUY, MARKET, Order
from stress import random_param_search
from strategy import Strategy
from tests.helpers import flat_bar, make_bars


# --- Shared fake market (same for every trial) ---

def _demo_bars():
    # Signal bar, fill bar, then path that tags both SL and TP zones depending on knobs.
    rows = [flat_bar(1.1000)]
    rows.append(
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002)
    )
    # Wide range bar: BidLow 1.085, BidHigh 1.130
    rows.append(
        (1.1000, 1.1300, 1.0850, 1.1200, 1.1002, 1.1302, 1.0852, 1.1202)
    )
    for mid in (1.1180, 1.1150, 1.1120, 1.1100, 1.1080):
        rows.append(flat_bar(mid))
    return make_bars(rows)


class _ParamFire(Strategy):
    """Fire one buy on bar 0; SL/TP come from constructor knobs."""

    def __init__(self, sl_pips, rr, pair="EUR/USD"):
        super(_ParamFire, self).__init__()
        self.sl_pips = float(sl_pips)
        self.rr = float(rr)
        self.pair = pair
        self._done = False

    def on_bar(self, window, account):
        if self._done or len(window) - 1 != 0:
            return []
        self._done = True
        # Entry expected next open ~1.1002; place SL/TP from that guess.
        entry_guess = 1.1002
        pip = 0.0001
        sl = entry_guess - self.sl_pips * pip
        tp = entry_guess + self.sl_pips * pip * self.rr
        return [
            Order(
                side=BUY,
                order_type=MARKET,
                lots=0.1,
                sl=sl,
                tp=tp,
                pair=self.pair,
                reason="demo",
            )
        ]


def run_fn(params):
    """Fresh handler + strategy + engine every trial (required for honest search)."""
    df = _demo_bars()
    pair = "EUR/USD"
    handler = DataHandler(df, pair=pair)
    strat = _ParamFire(sl_pips=params["sl_pips"], rr=params["rr"], pair=pair)
    result = OOEngine(
        {pair: handler},
        {pair: strat},
        start_balance=100000.0,
        slippage_pips=0.0,
        max_risk_per_trade=10.0,
        max_active_trades_per_pair=10,
    ).run()
    return result


def main():
    space = {
        "sl_pips": [50, 100, 150, 200],
        "rr": [1.0, 1.5, 2.0, 2.5],
    }
    # 4 * 4 = 16 combos → full grid (n_trials large enough).
    n_trials = 40
    rng = np.random.default_rng(42)

    print("=== DSR + random param search demo ===")
    print("knob lists:", space)
    print("n_trials asked:", n_trials)
    print("pass rule: DSR >= 0.90")
    print()

    out = random_param_search(
        run_fn,
        space,
        n_trials=n_trials,
        rng=rng,
        dsr_threshold=0.90,
    )

    print("N actually run:", out["n_trials"])
    print("best knobs:", out["best_params"])
    print("best Sharpe (annualised if ppy known):", round(out["best_sharpe"], 4))
    dsr = out["dsr"]
    print("DSR:", round(dsr["dsr"], 4))
    print("sr_star (luck bar):", round(dsr["sr_star"], 6))
    print("n_obs (return bars):", dsr["n_obs"])
    print("passes?", out["passes"])
    print()

    # Show every trial briefly
    print("--- all trials (params -> Sharpe) ---")
    for t in sorted(out["trials"], key=lambda x: -x["sharpe"]):
        print(" ", t["params"], "Sharpe=", round(t["sharpe"], 4))

    print()
    if out["passes"]:
        print("KEEP this setup (DSR >= 0.90)")
    else:
        print("REJECT this setup (DSR < 0.90)")
        print("(Tiny demo sample → DSR often fails the gate. That is expected.)")

    # Second check: same best equity scored with a LIE about N (don't do this for real)
    if out["best_equity"] is not None:
        honest = dsr_from_equity(out["best_equity"], n_trials=out["n_trials"])
        lie = dsr_from_equity(out["best_equity"], n_trials=1)
        print()
        print("--- why honest N matters ---")
        print("DSR with N=%d (honest):" % out["n_trials"], round(honest["dsr"], 4))
        print("DSR with N=1 (cheat):  ", round(lie["dsr"], 4))


if __name__ == "__main__":
    main()
