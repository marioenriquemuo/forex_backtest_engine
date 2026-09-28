"""HTF resample PIT + instrument pip/lot table."""

import pandas as pd

from data_handler import (
    DataHandler,
    align_htf_pit,
    instrument_spec,
    normalize_pair,
    resample_ohlc,
)
from tests.helpers import make_bars


def test_instrument_pip_table():
    assert instrument_spec("EUR/USD")["pip_size"] == 0.0001
    assert instrument_spec("eur-usd")["pip_size"] == 0.0001
    assert instrument_spec("USD/JPY")["pip_size"] == 0.01
    assert instrument_spec("XAU/USD")["pip_size"] == 0.01
    assert instrument_spec("XAU/USD")["contract_size"] == 100.0
    assert instrument_spec("FOO/JPY")["pip_size"] == 0.01
    assert normalize_pair(" eur-usd ") == "EUR/USD"


def test_htf_4h_not_visible_until_bar_completes():
    # 8 H1 bars 00:00–07:00. H4 labelled 00:00 uses 00,01,02,03 and completes at 04:00.
    mids = [1.10 + i * 0.001 for i in range(8)]
    rows = []
    for m in mids:
        bid, ask = m, m + 0.0002
        rows.append((bid, bid + 0.0001, bid - 0.0001, bid, ask, ask + 0.0001, ask - 0.0001, ask))
    df = make_bars(rows, start="2024-01-02 00:00")
    h4 = resample_ohlc(df, "4H")
    aligned = align_htf_pit(h4, df.index)
    h4_midnight = h4.loc[h4.index[0], "BidClose"]
    # H1 00:00, 01:00, 02:00, 03:00 must not see the completed 00:00 H4 close.
    for ts in df.index[:4]:
        val = aligned.loc[ts, "BidClose"]
        assert pd.isna(val) or val != h4_midnight
    # From 04:00 the completed 00:00 H4 bar is visible.
    assert aligned.loc[df.index[4], "BidClose"] == h4_midnight


def test_htf_bid_ask_resampled_independently():
    rows = []
    for i in range(4):
        bid = 1.10 + i * 0.001
        ask = bid + 0.0005
        rows.append((bid, bid + 0.002, bid - 0.002, bid + 0.0005, ask, ask + 0.002, ask - 0.002, ask + 0.0005))
    df = make_bars(rows, start="2024-01-02 00:00")
    h4 = resample_ohlc(df, "4H")
    row = h4.iloc[0]
    assert row["BidOpen"] == df["BidOpen"].iloc[0]
    assert row["BidClose"] == df["BidClose"].iloc[3]
    assert row["BidHigh"] == df["BidHigh"].max()
    assert row["BidLow"] == df["BidLow"].min()
    assert row["AskOpen"] == df["AskOpen"].iloc[0]
    assert row["AskClose"] == df["AskClose"].iloc[3]


def test_datahandler_window_is_prefix_only():
    df = make_bars([(1.1, 1.11, 1.09, 1.1, 1.1002, 1.1102, 1.0902, 1.1002)] * 5)
    h = DataHandler(df, pair="EUR/USD")
    w = h.window(2)
    assert len(w) == 3
    assert w.index[-1] == h.data.index[2]
