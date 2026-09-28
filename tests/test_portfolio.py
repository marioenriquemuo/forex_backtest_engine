"""Accounting: spread, commission, swap, margin."""

import pandas as pd

from data_handler import DataHandler
from engine import OOEngine
from orders import BUY, MARKET, Order
from tests.helpers import FireOnce, flat_bar, make_bars


def _run(df, strategy, **kwargs):
    pair = "EUR/USD"
    handler = DataHandler(df, pair=pair)
    kw = dict(
        start_balance=100000.0,
        slippage_pips=0.0,
        max_active_trades_per_pair=10,
        max_risk_per_trade=10.0,
        leverage=30.0,
    )
    kw.update(kwargs)
    eng = OOEngine({pair: handler}, {pair: strategy}, **kw)
    return eng.run(), eng


def test_spread_is_bid_ask_difference_no_static_spread():
    rows = [
        flat_bar(1.10, spread=0.0004),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1004, 1.1004, 1.1004, 1.1004),
        (1.0900, 1.0900, 1.0900, 1.0900, 1.0904, 1.0904, 1.0904, 1.0904),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=MARKET, lots=1.0, sl=1.0900, tp=None, pair="EUR/USD")
    result, _ = _run(df, FireOnce(0, order))
    t = result.trades[0]
    assert t["Entry"] == 1.1004
    assert t["Exit_Price"] == 1.0900
    expected = 1.0 * (1.0900 - 1.1004) * 100000.0
    assert abs(t["Result"] - expected) < 1e-6


def test_commission_per_lot_deducted_at_open():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        (1.0900, 1.0900, 1.0800, 1.0850, 1.0902, 1.0902, 1.0802, 1.0852),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=MARKET, lots=2.0, sl=1.0950, pair="EUR/USD")
    result, eng = _run(df, FireOnce(0, order), commission_per_lot=7.0)
    t = result.trades[0]
    assert t["Commission"] == 14.0
    assert abs(eng.portfolio.balance - (100000.0 - 14.0 + t["Result"])) < 1e-6


def test_commission_pct_on_notional():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000),
        (1.0900, 1.0900, 1.0800, 1.0850, 1.0900, 1.0900, 1.0800, 1.0850),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=MARKET, lots=1.0, sl=1.0950, pair="EUR/USD")
    result, _ = _run(df, FireOnce(0, order), commission_pct=0.0001)
    t = result.trades[0]
    assert abs(t["Commission"] - (0.0001 * 1.0 * 100000.0 * 1.1000)) < 1e-6


def test_swap_at_1700_ny_triple_on_wednesday():
    idx = pd.DatetimeIndex(
        [
            "2024-01-02 16:00",
            "2024-01-02 17:00",
            "2024-01-03 17:00",
            "2024-01-04 16:00",
        ],
        tz="America/New_York",
    )
    # Tuesday 17:00, Wednesday 17:00. Position opened Tuesday 17:00 bar.
    rows = [flat_bar(1.10) for _ in range(4)]
    df = pd.DataFrame(
        [dict(zip(
            ["BidOpen", "BidHigh", "BidLow", "BidClose", "AskOpen", "AskHigh", "AskLow", "AskClose"],
            r,
        )) for r in rows],
        index=idx,
    )
    df["Volume"] = 1.0
    df.iloc[3, df.columns.get_loc("BidLow")] = 1.0500
    order = Order(side=BUY, order_type=MARKET, lots=1.0, sl=1.0600, pair="EUR/USD")
    result, _ = _run(df, FireOnce(0, order), swap_long_per_lot=-5.0)
    t = result.trades[0]
    # Rollover Tue 17:00 after open? Signal bar 0 (16:00), fill bar 1 (17:00 Tue).
    # apply_swap on fill bar: last_rollover None, hour>=17, apply once (Tue *1 = -5)
    # Wed 17:00: *3 = -15. Total -20.
    assert t["Swap"] == -20.0


def test_margin_stop_out_liquidates():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000),
        (1.1000, 1.1000, 0.0100, 0.0200, 1.1002, 1.1002, 0.0102, 0.0202),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=MARKET, lots=50.0, sl=0.0010, pair="EUR/USD")
    result, eng = _run(
        df,
        FireOnce(0, order),
        start_balance=10000.0,
        leverage=30.0,
        stop_out_fraction=0.5,
        max_risk_per_trade=1e9,
    )
    assert any(t["Exit_Reason"] == "Margin" for t in result.trades) or eng.portfolio.is_blown_up
    assert eng.portfolio.positions == []
