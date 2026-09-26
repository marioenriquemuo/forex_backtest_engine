"""Order, fill, and position records."""

from dataclasses import dataclass, field
from typing import Any, Optional

MARKET = "market"
LIMIT = "limit"
STOP = "stop"
TRAILING_STOP = "trailing_stop"
EXIT = "exit"

BUY = "B"
SELL = "S"


def sl_on_correct_side(is_buy, entry, sl):
    if sl is None:
        return True
    if is_buy:
        return sl < entry
    return sl > entry


def tp_on_correct_side(is_buy, entry, tp):
    if tp is None:
        return True
    if is_buy:
        return tp > entry
    return tp < entry


@dataclass
class Order:
    side: str
    order_type: str = MARKET
    lots: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    price: Optional[float] = None
    trail_pips: Optional[float] = None
    max_hold_bars: Optional[int] = None
    expire_bars: Optional[int] = None
    risk_ratio: float = 1.0
    rr_ratio: Optional[float] = None
    pair: str = ""
    reason: str = ""
    features: Any = None
    created_bar: Optional[int] = None


@dataclass
class Fill:
    bar_index: int
    time: Any
    price: float
    lots: float
    side: str
    reason: str
    pair: str = ""


@dataclass
class Position:
    side: str
    entry: float
    lots: float
    sl: Optional[float]
    tp: Optional[float]
    pair: str
    entry_bar: int
    entry_time: Any
    trail_distance: Optional[float] = None
    max_hold_bars: Optional[int] = None
    swap: float = 0.0
    commission: float = 0.0
    mae: float = 0.0
    mfe: float = 0.0
    reason: str = ""
    features: Any = None
    pip_size: float = 0.0001
    contract_size: float = 100000.0


@dataclass
class ClosedTrade:
    pair: str
    side: str
    entry: float
    exit_price: float
    lots: float
    sl: Optional[float]
    tp: Optional[float]
    entry_time: Any
    exit_time: Any
    entry_bar: int
    exit_bar: int
    pnl: float
    swap: float
    commission: float
    exit_reason: str
    mae: float
    mfe: float
    balance: float
    slippage_paid: float = 0.0
    reason: str = ""
    features: Any = None
    extra: dict = field(default_factory=dict)

    def as_legacy_dict(self):
        return {
            "Date": self.entry_time,
            "Entry_Bar_Index": self.entry_bar,
            "Type": self.side,
            "Entry": self.entry,
            "SL": self.sl,
            "TP": self.tp,
            "Lots": self.lots,
            "Status": "Closed",
            "Entry_Reason": self.reason,
            "Pair": self.pair,
            "Swap": self.swap,
            "Features": self.features,
            "Max_Hold_Bars": self.extra.get("max_hold_bars"),
            "Exit_Date": self.exit_time,
            "Exit_Price": self.exit_price,
            "Exit_Bar_Index": self.exit_bar,
            "Result": self.pnl,
            "Balance": self.balance,
            "Exit_Reason": self.exit_reason,
            "Slippage_Paid_Pips": self.slippage_paid,
            "Commission": self.commission,
            "MAE": self.mae,
            "MFE": self.mfe,
        }
