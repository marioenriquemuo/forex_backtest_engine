"""Engine PIT fence: forming HTF, full-sample resample, truncated data, window copy, EXIT at cap."""

import pandas as pd

from data_handler import DataHandler, resample_ohlc
from engine import OOEngine
from orders import BUY, EXIT, MARKET, Order
from strategy import Strategy
from tests.helpers import make_bars


def _h1_8():
    mids = [1.10 + i * 0.001 for i in range(8)]
    rows = []
    for m in mids:
        bid, ask = m, m + 0.0002
        rows.append((bid, bid + 0.0001, bid - 0.0001, bid, ask, ask + 0.0001, ask - 0.0001, ask))
    return make_bars(rows, start="2024-01-02 00:00")


def _run(df, strategy, **kwargs):
    pair = "EUR/USD"
    handler = DataHandler(df, pair=pair)
    kw = dict(
        start_balance=100000.0,
        slippage_pips=0.0,
        max_active_trades_per_pair=10,
        max_risk_per_trade=10.0,
        htf_rules=("4H",),
    )
    kw.update(kwargs)
    result = OOEngine({pair: handler}, {pair: strategy}, **kw).run()
    return result, handler


class HtfRecorder(Strategy):
    def __init__(self):
        super(HtfRecorder, self).__init__()
        self.closes = []
        self.fired = False

    def on_bar(self, window, account):
        h4 = account["htf"]["4H"]
        val = h4.iloc[-1]["BidClose"] if len(h4) else float("nan")
        self.closes.append((len(window) - 1, window.index[-1], val))
        if self.fired or pd.isna(val):
            return []
        self.fired = True
        return [
            Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.0000, pair="EUR/USD")
        ]


class ResampleSpy(Strategy):
    def __init__(self, handler, end_close):
        super(ResampleSpy, self).__init__()
        self.handler = handler
        self.end_close = float(end_close)
        self.saw_end = False

    def on_bar(self, window, account):
        last = self.handler.resample_htf("4H").iloc[-1]["BidClose"]
        if pd.notna(last) and abs(float(last) - self.end_close) < 1e-12:
            self.saw_end = True
        return []


class DataSpy(Strategy):
    def __init__(self, full):
        super(DataSpy, self).__init__(full)
        self.ok = True

    def on_bar(self, window, account):
        if self.data.index[-1] != window.index[-1]:
            self.ok = False
        if len(self.data) != len(window):
            self.ok = False
        return []


class MutateWindow(Strategy):
    def on_bar(self, window, account):
        window["BidClose"] = 999.0
        return []


class BuyThenExit(Strategy):
    def __init__(self):
        super(BuyThenExit, self).__init__()
        self.sent_entry = False

    def on_bar(self, window, account):
        if account.get("active_trades"):
            return [Order(side=BUY, order_type=EXIT, pair="EUR/USD")]
        if self.sent_entry:
            return []
        self.sent_entry = True
        return [
            Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.0000, pair="EUR/USD")
        ]


def test_htf_not_visible_until_04_and_fill_is_t_plus_1():
    df = _h1_8()
    h4 = resample_ohlc(df, "4H")
    h4_midnight = h4.loc[h4.index[0], "BidClose"]
    pair = "EUR/USD"
    handler = DataHandler(df, pair=pair)
    strat = HtfRecorder()
    eng = OOEngine(
        {pair: handler},
        {pair: strat},
        start_balance=100000.0,
        slippage_pips=0.0,
        max_active_trades_per_pair=10,
        max_risk_per_trade=10.0,
        htf_rules=("4H",),
    )
    eng.run()
    for i, ts, val in strat.closes[:4]:
        assert i < 4
        assert pd.isna(val) or val != h4_midnight
    i4, ts4, val4 = strat.closes[4]
    assert i4 == 4
    assert ts4 == handler._data.index[4]
    assert val4 == h4_midnight
    assert len(eng.portfolio.positions) == 1
    pos = eng.portfolio.positions[0]
    assert pos.entry_bar == 5
    assert pos.entry == float(handler._data.iloc[5]["AskOpen"])


def test_resample_htf_during_on_bar_does_not_see_end_of_sample_h4():
    df = _h1_8()
    handler = DataHandler(df, pair="EUR/USD")
    end_close = resample_ohlc(handler._data, "4H").iloc[-1]["BidClose"]
    spy = ResampleSpy(handler, end_close)
    OOEngine(
        {"EUR/USD": handler},
        {"EUR/USD": spy},
        start_balance=100000.0,
        htf_rules=("4H",),
        max_active_trades_per_pair=10,
    ).run()
    assert spy.saw_end is False


def test_strategy_data_truncated_to_window():
    df = _h1_8()
    spy = DataSpy(df.copy())
    _run(df, spy)
    assert spy.ok is True
    assert len(spy.data) == len(df)


def test_mutating_window_does_not_change_handler_data():
    df = _h1_8()
    result, handler = _run(df, MutateWindow())
    assert not (handler._data["BidClose"] == 999.0).any()
    assert handler._asof is None


def test_exit_still_queued_when_at_max_active():
    df = _h1_8()
    result, _ = _run(
        df,
        BuyThenExit(),
        max_active_trades_per_pair=1,
        htf_rules=(),
    )
    assert len(result.trades) == 1
    assert result.trades[0]["Exit_Reason"] == "Exit"
    assert any(q["order_type"] == EXIT for q in result.queue_log)
