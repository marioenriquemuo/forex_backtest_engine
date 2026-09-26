"""Regressions for logic-audit fixes (multi-pair, SL/TP, JIT, metrics)."""

import numpy as np
import pandas as pd

from data_handler import DataHandler
from engine import OOEngine
from engine_jit import run_jit
from metrics import compute_metrics, _years_from_trades
from orders import BUY, MARKET, Order, Fill, tp_on_correct_side
from portfolio import PortfolioManager
from stress import perturb_execution
from strategy import Strategy
from tests.helpers import FireOnce, flat_bar, make_bars


class _CountBars(Strategy):
    def __init__(self):
        super(_CountBars, self).__init__()
        self.n = 0

    def on_bar(self, window, account):
        self.n += 1
        return []


def test_tp_on_correct_side():
    assert tp_on_correct_side(True, 1.10, 1.12)
    assert not tp_on_correct_side(True, 1.10, 1.09)
    assert tp_on_correct_side(False, 1.10, 1.08)
    assert not tp_on_correct_side(False, 1.10, 1.12)
    assert tp_on_correct_side(True, 1.10, None)


def test_wrong_side_tp_rejected():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
    ]
    df = make_bars(rows)
    order = Order(
        side=BUY, order_type=MARKET, lots=0.1, sl=1.0900, tp=1.0995, pair="EUR/USD"
    )
    eng = OOEngine(
        {"EUR/USD": DataHandler(df, pair="EUR/USD")},
        {"EUR/USD": FireOnce(0, order)},
        start_balance=100000.0,
        slippage_pips=0.0,
        max_risk_per_trade=10.0,
        max_active_trades_per_pair=10,
    )
    result = eng.run()
    assert len(result.trades) == 0
    assert eng.portfolio.positions == []


def test_fill_then_book_reject_cancels_not_fill():
    """Wrong-side SL: fill attempted but open_position fails — no fill, no position."""
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        flat_bar(1.10),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.1100, pair="EUR/USD")
    eng = OOEngine(
        {"EUR/USD": DataHandler(df, pair="EUR/USD")},
        {"EUR/USD": FireOnce(0, order)},
        start_balance=100000.0,
        slippage_pips=0.0,
        max_risk_per_trade=10.0,
        max_active_trades_per_pair=10,
    )
    result = eng.run()
    assert len(result.trades) == 0
    assert eng.portfolio.positions == []
    assert eng.brokers["EUR/USD"].working == []


def test_multipair_does_not_close_eur_on_gbp_bar():
    eur = make_bars([flat_bar(1.10), flat_bar(1.10), flat_bar(1.10), flat_bar(1.10)])
    gbp_rows = [
        flat_bar(1.25),
        (1.25, 1.25, 1.00, 1.20, 1.2502, 1.2502, 1.0002, 1.2002),
        flat_bar(1.25),
        flat_bar(1.25),
    ]
    gbp = make_bars(gbp_rows)
    order = Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.05, tp=1.20, pair="EUR/USD")

    class Idle(Strategy):
        def on_bar(self, window, account):
            return []

    eng = OOEngine(
        {"EUR/USD": DataHandler(eur, pair="EUR/USD"), "GBP/USD": DataHandler(gbp, pair="GBP/USD")},
        {"EUR/USD": FireOnce(0, order), "GBP/USD": Idle()},
        start_balance=100000.0,
        slippage_pips=0.0,
        max_risk_per_trade=10.0,
        max_active_trades_per_pair=10,
    )
    result = eng.run()
    # EUR long must not SL on GBP BidLow=1.00
    open_eur = [p for p in eng.portfolio.positions if p.pair == "EUR/USD"]
    closed_eur = [t for t in result.trades if t["Pair"] == "EUR/USD"]
    assert len(open_eur) == 1 or (len(closed_eur) == 1 and closed_eur[0]["Exit_Reason"] != "SL")
    if closed_eur:
        assert closed_eur[0]["Exit_Price"] != 1.00


def test_jit_rejects_wrong_side_sl_like_oo():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        (1.1000, 1.1300, 1.0700, 1.1200, 1.1002, 1.1302, 1.0702, 1.1202),
    ]
    df = make_bars(rows)
    handler = DataHandler(df, pair="EUR/USD")
    order = Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.1100, tp=1.1200, pair="EUR/USD")
    order.created_bar = 0
    oo = OOEngine(
        {"EUR/USD": handler},
        {"EUR/USD": FireOnce(0, Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.1100, tp=1.1200, pair="EUR/USD"))},
        start_balance=100000.0,
        slippage_pips=0.0,
        max_risk_per_trade=10.0,
        max_active_trades_per_pair=10,
    ).run()
    jit = run_jit(handler, [order], start_balance=100000.0, slippage_pips=0.0)
    assert len(oo.trades) == 0
    assert len(jit["trades"]) == 0


def test_jit_max_hold_matches_oo():
    rows = [flat_bar(1.10) for _ in range(5)]
    df = make_bars(rows)
    handler = DataHandler(df, pair="EUR/USD")
    order = Order(
        side=BUY,
        order_type=MARKET,
        lots=0.1,
        sl=1.0500,
        tp=1.2000,
        max_hold_bars=1,
        pair="EUR/USD",
    )
    oo = OOEngine(
        {"EUR/USD": DataHandler(df, pair="EUR/USD")},
        {"EUR/USD": FireOnce(0, order)},
        start_balance=100000.0,
        slippage_pips=0.0,
        max_risk_per_trade=10.0,
        max_active_trades_per_pair=10,
    ).run()
    frozen = Order(
        side=BUY,
        order_type=MARKET,
        lots=0.1,
        sl=1.0500,
        tp=1.2000,
        max_hold_bars=1,
        pair="EUR/USD",
        created_bar=0,
    )
    jit = run_jit(handler, [frozen], start_balance=100000.0, slippage_pips=0.0)
    assert len(oo.trades) == 1
    assert oo.trades[0]["Exit_Reason"] == "Time"
    assert len(jit["trades"]) == 1
    assert jit["trades"][0]["Exit_Reason"] == "Time"
    assert abs(oo.trades[0]["Result"] - jit["trades"][0]["Result"]) < 1e-8


def test_cagr_years_uses_min_max_not_row_order():
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    trades = [
        {"Date": idx[5], "Exit_Date": idx[6], "Result": 1.0, "Lots": 1.0},
        {"Date": idx[0], "Exit_Date": idx[9], "Result": 1.0, "Lots": 1.0},
    ]
    df = pd.DataFrame(trades)
    years = _years_from_trades(df)
    expected = (idx[9] - idx[0]).total_seconds() / (365.25 * 24 * 3600.0)
    assert abs(years - expected) < 1e-12
    # Row-order iloc[0]->iloc[-1] would be idx[5]->idx[9], shorter.
    wrong = (idx[9] - idx[5]).total_seconds() / (365.25 * 24 * 3600.0)
    assert years > wrong


def test_metrics_net_of_commission():
    trades = [
        {"Result": 10.0, "Commission": 15.0, "Lots": 1.0, "MAE": 0.0, "MFE": 0.0},
    ]
    m = compute_metrics(trades, [100.0, 95.0], start_balance=100.0, years=1.0)
    assert m["win_rate"] == 0.0
    assert abs(m["expectancy_currency"] - (-5.0)) < 1e-9


def test_floating_pnl_includes_swap():
    pm = PortfolioManager(start_balance=100000.0, swap_long_per_lot=-5.0)
    order = Order(side=BUY, order_type=MARKET, lots=1.0, sl=1.05, pair="EUR/USD")
    bar = make_bars([flat_bar(1.10)]).iloc[0]
    fill = Fill(bar_index=0, time=bar.name, price=1.1002, lots=1.0, side=BUY, reason="M", pair="EUR/USD")
    pos = pm.open_position(order, fill, bar)
    pos.swap = -20.0
    bars = {"EUR/USD": bar}
    fp = pm.floating_pnl(bars)
    expected = (float(bar["BidClose"]) - pos.entry) * pos.lots * pos.contract_size + pos.swap
    assert abs(fp - expected) < 1e-9


def test_perturb_isolates_strategy_state():
    rows = [flat_bar(1.10) for _ in range(4)]
    df = make_bars(rows)
    handler = DataHandler(df, pair="EUR/USD")
    strat = _CountBars()
    perturb_execution(
        {"EUR/USD": handler},
        {"EUR/USD": strat},
        start_balance=100000.0,
        n_runs=3,
        max_extra_slip_pips=0.0,
        reject_rate=0.0,
        rng=np.random.default_rng(0),
        slippage_pips=0.0,
        max_risk_per_trade=10.0,
        max_active_trades_per_pair=10,
    )
    # Original strategy must not accumulate bars from all runs.
    assert strat.n == 0
