"""Strategy ABC and adapter for generate_signal(current_bar_index) strategies."""

from abc import ABC

from orders import EXIT, LIMIT, MARKET, Order


class Strategy(ABC):
    def __init__(self, data=None):
        self.data = data

    def on_bar(self, window, account):
        """Return a list of Order objects. window is bars [0..t] only (a copy).

        account may include "htf" from OOEngine(htf_rules=...).
        """
        sig = self.generate_signal(len(window), active_trades=account.get("active_trades"))
        return _as_orders(sig, "")

    def generate_signal(self, current_bar_index, active_trades=None):
        """Legacy hook. New strategies should override on_bar."""
        return None


def _as_orders(sig, pair):
    if not sig:
        return []
    if not isinstance(sig, list):
        sig = [sig]
    out = []
    for s in sig:
        if not s:
            continue
        if s.get("Type") == "Exit":
            out.append(
                Order(
                    side=s.get("Side") or "B",
                    order_type=EXIT,
                    pair=s.get("Pair") or pair,
                    reason=s.get("Reason", "Exit"),
                )
            )
            continue
        limit = bool(s.get("Limit_Entry"))
        out.append(
            Order(
                side=s["Side"],
                order_type=LIMIT if limit else MARKET,
                lots=s.get("Lots"),
                sl=s.get("SL"),
                tp=s.get("TP"),
                price=s.get("Entry_Price") if limit else None,
                max_hold_bars=s.get("Max_Hold_Bars"),
                risk_ratio=float(s.get("risk_ratio", 1.0)),
                rr_ratio=s.get("rr_ratio"),
                pair=s.get("Pair") or pair,
                reason=s.get("Reason", "Signal"),
                features=s.get("Features"),
            )
        )
    return out


class LegacyStrategyAdapter(Strategy):
    """Maps generate_signal(k) with sig_idx=k-1 onto on_bar(window[-1] = signal bar)."""

    def __init__(self, legacy_strategy, pair=""):
        super(LegacyStrategyAdapter, self).__init__(getattr(legacy_strategy, "data", None))
        self.legacy = legacy_strategy
        self.pair = pair

    def on_bar(self, window, account):
        if window is None or len(window) == 0:
            return []
        inner = self.legacy
        if hasattr(inner, "account_equity"):
            inner.account_equity = account.get("equity")
        ts = window.index[-1]
        data = getattr(inner, "data", None)
        if data is None or data.empty:
            return _as_orders(
                inner.generate_signal(len(window), active_trades=account.get("active_trades")),
                self.pair,
            )
        if ts not in data.index:
            return []
        loc = data.index.get_loc(ts)
        if hasattr(loc, "stop"):
            loc = int(loc.stop) - 1
        k = int(loc) + 1
        sig = inner.generate_signal(k, active_trades=account.get("active_trades"))
        return _as_orders(sig, self.pair)

    def generate_signal(self, current_bar_index, active_trades=None):
        return self.legacy.generate_signal(current_bar_index, active_trades=active_trades)
