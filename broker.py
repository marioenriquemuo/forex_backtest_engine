"""BrokerSimulator and shared pessimistic fill rules.

Fill helpers take scalars so the Numba loop can call the same logic.
"""

from typing import List

from data_handler import normalize_pair
from orders import (
    BUY,
    EXIT,
    LIMIT,
    MARKET,
    STOP,
    TRAILING_STOP,
    Fill,
    Order,
)


def market_entry_price(is_buy, ask_open, bid_open, slippage):
    if is_buy:
        return ask_open + slippage
    return bid_open - slippage


def resolve_long_exit(bid_open, bid_high, bid_low, sl, tp):
    """SL before TP. Gap through at Open fills at Open."""
    if sl is not None and bid_open <= sl:
        return bid_open, "SL"
    if tp is not None and bid_open >= tp:
        return bid_open, "TP"
    if sl is not None and bid_low <= sl:
        return sl, "SL"
    if tp is not None and bid_high >= tp:
        return tp, "TP"
    return None, None


def resolve_short_exit(ask_open, ask_high, ask_low, sl, tp):
    if sl is not None and ask_open >= sl:
        return ask_open, "SL"
    if tp is not None and ask_open <= tp:
        return ask_open, "TP"
    if sl is not None and ask_high >= sl:
        return sl, "SL"
    if tp is not None and ask_low <= tp:
        return tp, "TP"
    return None, None


def resolve_exit(is_buy, bar, sl, tp):
    if is_buy:
        return resolve_long_exit(
            bar["BidOpen"], bar["BidHigh"], bar["BidLow"], sl, tp
        )
    return resolve_short_exit(
        bar["AskOpen"], bar["AskHigh"], bar["AskLow"], sl, tp
    )


def buy_stop_fill(stop, ask_open, ask_high):
    if ask_high < stop:
        return None
    if ask_open >= stop:
        return ask_open
    return stop


def buy_limit_fill(limit, ask_open, ask_low):
    if ask_low > limit:
        return None
    if ask_open <= limit:
        return ask_open
    return limit


def sell_stop_fill(stop, bid_open, bid_low):
    if bid_low > stop:
        return None
    if bid_open <= stop:
        return bid_open
    return stop


def sell_limit_fill(limit, bid_open, bid_high):
    if bid_high < limit:
        return None
    if bid_open >= limit:
        return bid_open
    return limit


def pending_fill_price(order, bar):
    is_buy = order.side == BUY
    px = order.price
    if px is None:
        return None
    if order.order_type == STOP:
        if is_buy:
            return buy_stop_fill(px, bar["AskOpen"], bar["AskHigh"])
        return sell_stop_fill(px, bar["BidOpen"], bar["BidLow"])
    if order.order_type == LIMIT:
        if is_buy:
            return buy_limit_fill(px, bar["AskOpen"], bar["AskLow"])
        return sell_limit_fill(px, bar["BidOpen"], bar["BidHigh"])
    return None


def update_trailing_sl(is_buy, sl, trail_distance, bid_high, ask_low):
    if trail_distance is None or trail_distance <= 0:
        return sl
    if is_buy:
        candidate = bid_high - trail_distance
        if sl is None:
            return candidate
        return max(sl, candidate)
    candidate = ask_low + trail_distance
    if sl is None:
        return candidate
    return min(sl, candidate)


class BrokerSimulator:
    """Queues orders on bar t and executes them from bar t+1 Open onward."""

    def __init__(self, slippage=0.0):
        self.slippage = float(slippage)
        self.working: List[Order] = []
        self.queue_log: List[dict] = []

    def enqueue(self, orders, created_bar):
        placed = []
        for order in orders or []:
            order.created_bar = created_bar
            self.working.append(order)
            placed.append(
                {
                    "created_bar": created_bar,
                    "side": order.side,
                    "order_type": order.order_type,
                    "price": order.price,
                    "sl": order.sl,
                    "tp": order.tp,
                    "lots": order.lots,
                    "pair": order.pair,
                    "reason": order.reason,
                }
            )
        self.queue_log.extend(placed)
        return placed

    def _expire(self, bar_index):
        keep = []
        for order in self.working:
            if order.expire_bars is None or order.created_bar is None:
                keep.append(order)
                continue
            if bar_index - order.created_bar > order.expire_bars:
                continue
            keep.append(order)
        self.working = keep

    def process_bar(
        self,
        bar,
        bar_index,
        portfolio,
        reject_mask=None,
        extra_slippage=0.0,
        pair=None,
    ):
        """Execute working orders, then manage open positions for `pair` only."""
        self._expire(bar_index)
        fills = []
        still = []
        extras = []
        for order in self.working:
            if order.order_type == EXIT:
                extras.append(order)
                continue
            if reject_mask is not None and reject_mask(order, bar_index):
                # Reject-once cancel (caller should decide reject at enqueue).
                continue
            extra = extra_slippage if order.order_type in (MARKET, TRAILING_STOP) else 0.0
            fill = self._try_fill_order(order, bar, bar_index, extra)
            if fill is None:
                still.append(order)
                continue
            pos = portfolio.open_position(order, fill, bar)
            if pos is None:
                # Fill price ok but book rejected (bad SL/TP/risk) — cancel, do not drop as filled.
                continue
            fills.append(fill)
            self._same_bar_death(pos, bar, bar_index, portfolio)
        self.working = extras + still
        self._process_exits(bar, bar_index, portfolio, pair=pair)
        self._manage_positions(bar, bar_index, portfolio, pair=pair)
        return fills

    def _try_fill_order(self, order, bar, bar_index, extra_slip=0.0):
        is_buy = order.side == BUY
        if order.order_type in (MARKET, TRAILING_STOP):
            price = market_entry_price(
                is_buy, bar["AskOpen"], bar["BidOpen"], self.slippage + extra_slip
            )
            return Fill(
                bar_index=bar_index,
                time=bar.name,
                price=price,
                lots=order.lots or 0.0,
                side=order.side,
                reason=order.reason or "Market",
                pair=order.pair,
            )
        price = pending_fill_price(order, bar)
        if price is None:
            return None
        return Fill(
            bar_index=bar_index,
            time=bar.name,
            price=price,
            lots=order.lots or 0.0,
            side=order.side,
            reason=order.reason or order.order_type,
            pair=order.pair,
        )

    def _same_bar_death(self, pos, bar, bar_index, portfolio):
        is_buy = pos.side == BUY
        px, reason = resolve_exit(is_buy, bar, pos.sl, pos.tp)
        if px is None:
            portfolio.touch_excursion(pos, bar)
            return
        portfolio.close_position(pos, px, reason, bar, bar_index)

    def _pair_match(self, pos_pair, broker_pair):
        if not broker_pair:
            return True
        return normalize_pair(pos_pair or "") == normalize_pair(broker_pair)

    def _manage_positions(self, bar, bar_index, portfolio, pair=None):
        for pos in list(portfolio.positions):
            if not self._pair_match(pos.pair, pair):
                continue
            is_buy = pos.side == BUY
            if pos.max_hold_bars is not None:
                if bar_index - pos.entry_bar >= int(pos.max_hold_bars):
                    px = bar["BidOpen"] if is_buy else bar["AskOpen"]
                    portfolio.close_position(pos, px, "Time", bar, bar_index)
                    continue
            px, reason = resolve_exit(is_buy, bar, pos.sl, pos.tp)
            if px is not None:
                portfolio.close_position(pos, px, reason, bar, bar_index)
                continue
            if pos.trail_distance:
                pos.sl = update_trailing_sl(
                    is_buy,
                    pos.sl,
                    pos.trail_distance,
                    bar["BidHigh"],
                    bar["AskLow"],
                )
            portfolio.touch_excursion(pos, bar)

    def _process_exits(self, bar, bar_index, portfolio, pair=None):
        still = []
        for order in self.working:
            if order.order_type != EXIT:
                still.append(order)
                continue
            closed_any = False
            for pos in list(portfolio.positions):
                if not self._pair_match(pos.pair, pair):
                    continue
                if order.pair and pos.pair and normalize_pair(order.pair) != normalize_pair(pos.pair):
                    continue
                if order.side and pos.side != order.side:
                    continue
                is_buy = pos.side == BUY
                px = bar["BidOpen"] if is_buy else bar["AskOpen"]
                portfolio.close_position(pos, px, "Exit", bar, bar_index)
                closed_any = True
            if not closed_any:
                still.append(order)
        self.working = still
