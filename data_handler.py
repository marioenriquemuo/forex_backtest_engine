"""Bid/Ask data loading, HTF resample, instrument pip/lot specs."""

from pathlib import Path

import pandas as pd

BID_ASK_COLS = [
    "BidOpen",
    "BidHigh",
    "BidLow",
    "BidClose",
    "AskOpen",
    "AskHigh",
    "AskLow",
    "AskClose",
]

_FX_CONTRACT = 100000.0

# Explicit pip/lot table. Fallback is quote-currency inference, not substring hacks.
INSTRUMENT_SPECS = {
    "EUR/USD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "GBP/USD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "AUD/USD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "NZD/USD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "USD/CAD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "USD/CHF": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "USD/JPY": {"pip_size": 0.01, "contract_size": _FX_CONTRACT},
    "EUR/JPY": {"pip_size": 0.01, "contract_size": _FX_CONTRACT},
    "GBP/JPY": {"pip_size": 0.01, "contract_size": _FX_CONTRACT},
    "AUD/JPY": {"pip_size": 0.01, "contract_size": _FX_CONTRACT},
    "CAD/JPY": {"pip_size": 0.01, "contract_size": _FX_CONTRACT},
    "CHF/JPY": {"pip_size": 0.01, "contract_size": _FX_CONTRACT},
    "NZD/JPY": {"pip_size": 0.01, "contract_size": _FX_CONTRACT},
    "EUR/GBP": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "EUR/CHF": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "EUR/AUD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "EUR/CAD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "EUR/NZD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "GBP/CHF": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "GBP/CAD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "GBP/AUD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "GBP/NZD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "AUD/CAD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "AUD/CHF": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "AUD/NZD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "NZD/CAD": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "NZD/CHF": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "CAD/CHF": {"pip_size": 0.0001, "contract_size": _FX_CONTRACT},
    "XAU/USD": {"pip_size": 0.01, "contract_size": 100.0},
    "XAG/USD": {"pip_size": 0.001, "contract_size": 5000.0},
}


def normalize_pair(pair):
    return str(pair).strip().upper().replace("-", "/").replace("_", "/")


def instrument_spec(pair, contract_size=None):
    p = normalize_pair(pair)
    spec = dict(INSTRUMENT_SPECS.get(p) or _infer_spec(p))
    if contract_size is not None:
        spec["contract_size"] = float(contract_size)
    spec["pair"] = p
    return spec


def _infer_spec(pair):
    p = normalize_pair(pair)
    if "/" in p:
        base, quote = p.split("/", 1)
    else:
        base, quote = p, ""
    if quote == "JPY" or base == "XAU":
        pip = 0.01
    elif base == "XAG":
        pip = 0.001
    else:
        pip = 0.0001
    contract = 100.0 if base == "XAU" else (5000.0 if base == "XAG" else _FX_CONTRACT)
    return {"pip_size": pip, "contract_size": contract}


def _to_ny_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex")
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df = df.copy()
    df.index = idx.tz_convert("America/New_York")
    return df.sort_index()


def _require_cols(df, cols, src):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError("Missing columns in %s: %s" % (src, miss))


def _read_ohlc_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("CSV not found: %s" % path)
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("Missing Date column in %s" % path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.dropna(subset=["Date"]).set_index("Date")
    return df.sort_index()


def fill_missing_timestamps(df, freq):
    """Insert weekday missing stamps and forward-fill OHLC. Does not invent weekends."""
    if df.empty:
        return df
    full = pd.date_range(df.index[0], df.index[-1], freq=freq, tz=df.index.tz)
    full = full[full.dayofweek < 5]
    out = df.reindex(full)
    ohlc = [c for c in BID_ASK_COLS if c in out.columns]
    out[ohlc] = out[ohlc].ffill()
    if "Volume" in out.columns:
        out["Volume"] = out["Volume"].fillna(0.0)
    return out.dropna(subset=[c for c in ("BidOpen", "AskOpen") if c in out.columns])


def resample_ohlc(df, rule):
    """Synthesize HTF Bid/Ask independently. Open-labelled, left-closed.

    An H4 bar labelled T contains [T, T+4h) and is complete at T+4h.
    """
    agg = {
        "BidOpen": "first",
        "BidHigh": "max",
        "BidLow": "min",
        "BidClose": "last",
        "AskOpen": "first",
        "AskHigh": "max",
        "AskLow": "min",
        "AskClose": "last",
    }
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    use = {k: v for k, v in agg.items() if k in df.columns}
    rule = str(rule)
    try:
        pd.tseries.frequencies.to_offset(rule)
    except ValueError:
        swapped = rule.replace("h", "H") if "h" in rule else rule.replace("H", "h")
        rule = swapped
    out = df.resample(rule, label="left", closed="left").agg(use)
    need = [c for c in ("BidOpen", "AskOpen") if c in out.columns]
    return out.dropna(subset=need)


def align_htf_pit(htf_df, base_index):
    """Shift completed HTF bars so they are visible only from the next HTF open."""
    return htf_df.shift(1).reindex(base_index, method="ffill")


class DataHandler:
    """Synchronized Bid/Ask series, chronological bars, optional weekday ffill.

    During OOEngine.on_bar, _asof is set so .data and resample_htf see a prefix only.
    pit_htf / resample_htf apply align_htf_pit (completed HTF bars, never forming).
    """

    def __init__(self, df, pair="EUR/USD", freq=None, contract_size=None):
        _require_cols(df, BID_ASK_COLS, pair)
        self.pair = normalize_pair(pair)
        cols = list(BID_ASK_COLS)
        if "Volume" in df.columns:
            cols.append("Volume")
        data = df[cols].copy()
        if "Volume" not in data.columns:
            data["Volume"] = 0.0
        data = _to_ny_index(data)
        if freq:
            data = fill_missing_timestamps(data, freq)
        self._data = data
        self._asof = None
        self.spec = instrument_spec(self.pair, contract_size=contract_size)

    @property
    def data(self):
        if self._asof is None:
            return self._data
        return self._data.iloc[: self._asof + 1].copy()

    @classmethod
    def from_csv(cls, path, pair="EUR/USD", start=None, end=None, freq=None, contract_size=None):
        df = _read_ohlc_csv(path)
        _require_cols(df, BID_ASK_COLS, path)
        df = cls._slice(df, start, end)
        return cls(df, pair=pair, freq=freq, contract_size=contract_size)

    @classmethod
    def from_bid_ask_csv(
        cls, bid_path, ask_path, pair="EUR/USD", start=None, end=None, freq=None, contract_size=None
    ):
        bid = _read_ohlc_csv(bid_path)
        ask = _read_ohlc_csv(ask_path)
        bid_cols = ["BidOpen", "BidHigh", "BidLow", "BidClose"]
        ask_cols = ["AskOpen", "AskHigh", "AskLow", "AskClose"]
        if all(c in bid.columns for c in bid_cols):
            bid = bid[bid_cols]
        else:
            _require_cols(bid, ["Open", "High", "Low", "Close"], bid_path)
            bid = bid.rename(columns={c: "Bid%s" % c for c in ("Open", "High", "Low", "Close")})
            bid = bid[bid_cols]
        if all(c in ask.columns for c in ask_cols):
            ask = ask[ask_cols]
        else:
            _require_cols(ask, ["Open", "High", "Low", "Close"], ask_path)
            ask = ask.rename(columns={c: "Ask%s" % c for c in ("Open", "High", "Low", "Close")})
            ask = ask[ask_cols]
        merged = bid.join(ask, how="outer").sort_index()
        merged[bid_cols] = merged[bid_cols].ffill()
        merged[ask_cols] = merged[ask_cols].ffill()
        merged = merged.dropna(subset=BID_ASK_COLS)
        merged = cls._slice(merged, start, end)
        return cls(merged, pair=pair, freq=freq, contract_size=contract_size)

    @staticmethod
    def _slice(df, start, end):
        if start is not None:
            s = pd.Timestamp(start)
            if s.tzinfo is None:
                s = s.tz_localize("America/New_York")
            df = df[df.index.tz_convert("America/New_York") >= s]
        if end is not None:
            e = pd.Timestamp(end)
            if e.tzinfo is None:
                e = e.tz_localize("America/New_York")
            df = df[df.index.tz_convert("America/New_York") <= e]
        return df

    def __len__(self):
        return len(self._data)

    def bar(self, i):
        return self._data.iloc[i]

    def window(self, t):
        """Bars [0..t] inclusive. Copy so strategies cannot mutate later bars."""
        end = t
        if self._asof is not None:
            end = min(int(t), int(self._asof))
        return self._data.iloc[: end + 1].copy()

    def pit_htf(self, rule):
        """Completed HTF only, aligned to the full LTF index (shift-1 then ffill)."""
        src = self._data
        return align_htf_pit(resample_ohlc(src, rule), src.index)

    def resample_htf(self, rule):
        """PIT HTF on currently visible data (prefix during on_bar)."""
        src = self.data
        return align_htf_pit(resample_ohlc(src, rule), src.index)
