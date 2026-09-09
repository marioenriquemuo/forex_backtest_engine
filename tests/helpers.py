import pandas as pd

from strategy import Strategy

BID_ASK = [
    "BidOpen",
    "BidHigh",
    "BidLow",
    "BidClose",
    "AskOpen",
    "AskHigh",
    "AskLow",
    "AskClose",
]


def make_bars(rows, start="2024-01-02 10:00", freq="h"):
    """rows: dicts or 8-tuples (bid o/h/l/c, ask o/h/l/c)."""
    records = []
    for row in rows:
        if isinstance(row, dict):
            rec = {k: float(row[k]) for k in BID_ASK}
            rec["Volume"] = float(row.get("Volume", 1.0))
        else:
            rec = dict(zip(BID_ASK, (float(x) for x in row)))
            rec["Volume"] = 1.0
        records.append(rec)
    idx = pd.date_range(start, periods=len(records), freq="H", tz="America/New_York")
    return pd.DataFrame(records, index=idx)


def flat_bar(mid, spread=0.0002, width=0.0004):
    half = spread / 2.0
    bid = mid - half
    ask = mid + half
    w = width / 2.0
    return (
        bid,
        bid + w,
        bid - w,
        bid,
        ask,
        ask + w,
        ask - w,
        ask,
    )


class FireOnce(Strategy):
    def __init__(self, at_bar, order):
        super(FireOnce, self).__init__()
        self.at_bar = at_bar
        self.order = order
        self.window_lengths = []
        self.peek_failed = True

    def on_bar(self, window, account):
        self.window_lengths.append(len(window))
        try:
            window.iloc[len(window)]
            self.peek_failed = False
        except Exception:
            self.peek_failed = True
        if len(window) - 1 == self.at_bar:
            return [self.order]
        return []


class PeekFuture(Strategy):
    def __init__(self):
        super(PeekFuture, self).__init__()
        self.saw_future = False

    def on_bar(self, window, account):
        try:
            window.iloc[len(window)]
            self.saw_future = True
        except Exception:
            pass
        return []
