"""Golden-bar contracts for pessimistic execution."""

import pandas as pd

from data_handler import DataHandler
from engine import OOEngine
from orders import BUY, LIMIT, MARKET, SELL, STOP, TRAILING_STOP, Order
from tests.helpers import FireOnce, PeekFuture, flat_bar, make_bars


def _run(df, strategy, pair="EUR/USD", **kwargs):
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


def test_t_plus_1_fill_at_next_open_not_signal_close():
    rows = [
        (1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007),
        (1.2000, 1.2010, 1.1990, 1.2005, 1.2002, 1.2012, 1.1992, 1.2007),
        (1.2000, 1.2010, 1.1990, 1.2005, 1.2002, 1.2012, 1.1992, 1.2007),
        (1.0500, 1.0510, 1.0000, 1.0100, 1.0502, 1.0512, 1.0002, 1.0102),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.0000, tp=None, pair="EUR/USD")
    strat = FireOnce(0, order)
    result, _ = _run(df, strat)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t["Entry"] == 1.2002
    assert t["Entry"] != 1.1005
    assert t["Entry"] != 1.1007
    assert t["Exit_Reason"] == "SL"


def test_strategy_window_cannot_read_future_bar():
    df = make_bars([flat_bar(1.10) for _ in range(5)])
    strat = PeekFuture()
    _run(df, strat)
    assert strat.saw_future is False


def test_long_enters_at_ask_sl_tp_on_bid():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        (1.1000, 1.1300, 1.0900, 1.1200, 1.1002, 1.1302, 1.0902, 1.1202),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.0950, tp=1.1250, pair="EUR/USD")
    result, _ = _run(df, FireOnce(0, order))
    t = result.trades[0]
    assert t["Entry"] == 1.1002
    assert t["Exit_Reason"] == "SL"
    assert t["Exit_Price"] == 1.0950


def test_short_enters_at_bid_sl_tp_on_ask():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        (1.0900, 1.1100, 1.0800, 1.0850, 1.0902, 1.1102, 1.0802, 1.0852),
    ]
    df = make_bars(rows)
    order = Order(side=SELL, order_type=MARKET, lots=0.1, sl=1.1050, tp=1.0850, pair="EUR/USD")
    result, _ = _run(df, FireOnce(0, order))
    t = result.trades[0]
    assert t["Entry"] == 1.1000
    assert t["Exit_Reason"] == "SL"
    assert t["Exit_Price"] == 1.1050


def test_sl_wins_when_same_bar_hits_sl_and_tp():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        (1.1000, 1.1300, 1.0700, 1.1200, 1.1002, 1.1302, 1.0702, 1.1202),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.0900, tp=1.1200, pair="EUR/USD")
    result, _ = _run(df, FireOnce(0, order))
    assert result.trades[0]["Exit_Reason"] == "SL"
    assert result.trades[0]["Exit_Price"] == 1.0900


def test_same_bar_instant_death_after_buy_stop():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1120, 1.0900, 1.0910, 1.1002, 1.1122, 1.0902, 1.0912),
        flat_bar(1.09),
    ]
    df = make_bars(rows)
    order = Order(
        side=BUY,
        order_type=STOP,
        lots=0.1,
        price=1.1050,
        sl=1.0950,
        tp=1.1300,
        pair="EUR/USD",
    )
    result, _ = _run(df, FireOnce(0, order))
    t = result.trades[0]
    assert t["Entry"] == 1.1050
    assert t["Exit_Reason"] == "SL"
    assert t["Exit_Price"] == 1.0950
    assert t["Entry_Bar_Index"] == t["Exit_Bar_Index"]


def test_same_bar_death_uses_open_if_open_already_through_sl():
    rows = [
        flat_bar(1.10),
        (1.0800, 1.1120, 1.0700, 1.0750, 1.1060, 1.1122, 1.0702, 1.0752),
        flat_bar(1.07),
    ]
    df = make_bars(rows)
    order = Order(
        side=BUY,
        order_type=STOP,
        lots=0.1,
        price=1.1050,
        sl=1.0950,
        tp=1.1300,
        pair="EUR/USD",
    )
    result, _ = _run(df, FireOnce(0, order))
    t = result.trades[0]
    assert t["Entry"] == 1.1060
    assert t["Exit_Reason"] == "SL"
    assert t["Exit_Price"] == 1.0800


def test_weekend_gap_sl_fills_at_open_not_sl():
    idx = pd.DatetimeIndex(
        [
            "2024-01-05 16:00",
            "2024-01-05 17:00",
            "2024-01-08 09:00",
        ],
        tz="America/New_York",
    )
    df = pd.DataFrame(
        [
            dict(zip(
                ["BidOpen", "BidHigh", "BidLow", "BidClose", "AskOpen", "AskHigh", "AskLow", "AskClose"],
                flat_bar(1.10),
            )),
            dict(zip(
                ["BidOpen", "BidHigh", "BidLow", "BidClose", "AskOpen", "AskHigh", "AskLow", "AskClose"],
                (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
            )),
            dict(zip(
                ["BidOpen", "BidHigh", "BidLow", "BidClose", "AskOpen", "AskHigh", "AskLow", "AskClose"],
                (1.0800, 1.0820, 1.0780, 1.0810, 1.0802, 1.0822, 1.0782, 1.0812),
            )),
        ],
        index=idx,
    )
    df["Volume"] = 1.0
    order = Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.0900, tp=1.1500, pair="EUR/USD")
    result, _ = _run(df, FireOnce(0, order))
    t = result.trades[0]
    assert t["Exit_Reason"] == "SL"
    assert t["Exit_Price"] == 1.0800
    assert t["Exit_Price"] != 1.0900


def test_gap_through_buy_stop_fills_at_open():
    rows = [
        flat_bar(1.10),
        (1.1100, 1.1120, 1.1090, 1.1110, 1.1102, 1.1122, 1.1092, 1.1112),
        flat_bar(1.11),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=STOP, lots=0.1, price=1.1050, sl=1.0800, pair="EUR/USD")
    result, eng = _run(df, FireOnce(0, order))
    assert eng.portfolio.positions[0].entry == 1.1102


def test_limit_does_not_fill_if_bar_never_trades_through():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1010, 1.0990, 1.1000, 1.1002, 1.1012, 1.0992, 1.1002),
        (1.1000, 1.1010, 1.0990, 1.1000, 1.1002, 1.1012, 1.0992, 1.1002),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=LIMIT, lots=0.1, price=1.0900, sl=1.0800, pair="EUR/USD")
    result, eng = _run(df, FireOnce(0, order))
    assert result.trades == []
    assert len(eng.brokers["EUR/USD"].working) == 1


def test_buy_limit_fills_when_ask_trades_through():
    rows = [
        flat_bar(1.10),
        (1.0970, 1.1010, 1.0940, 1.0960, 1.0972, 1.1012, 1.0942, 1.0962),
        (1.0500, 1.0510, 1.0400, 1.0410, 1.0502, 1.0512, 1.0402, 1.0412),
    ]
    df = make_bars(rows)
    order = Order(side=BUY, order_type=LIMIT, lots=0.1, price=1.0960, sl=1.0800, pair="EUR/USD")
    result, _ = _run(df, FireOnce(0, order))
    assert result.trades[0]["Entry"] == 1.0960


def test_trailing_stop_ratchets_only_in_favor():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        (1.1000, 1.1100, 1.0990, 1.1090, 1.1002, 1.1102, 1.0992, 1.1092),
        (1.1070, 1.1080, 1.1040, 1.1065, 1.1072, 1.1082, 1.1042, 1.1067),
    ]
    df = make_bars(rows)
    order = Order(
        side=BUY,
        order_type=TRAILING_STOP,
        lots=0.1,
        sl=1.0900,
        trail_pips=50,
        pair="EUR/USD",
    )
    result, _ = _run(df, FireOnce(0, order))
    t = result.trades[0]
    assert t["Exit_Reason"] == "SL"
    assert abs(t["Exit_Price"] - 1.1050) < 1e-9
