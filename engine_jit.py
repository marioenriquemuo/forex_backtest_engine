"""Array-based execution loop. Numba-jitted when available.

Replays frozen orders against Bid/Ask arrays using the same fill rules as BrokerSimulator.
Supports market/limit/stop/trail, SL/TP side checks, and max_hold Time exits.
No EXIT orders, commission_pct, swap, or margin stop-out — use OOEngine for those.
"""

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return deco


OT_MARKET = 0
OT_LIMIT = 1
OT_STOP = 2
OT_TRAIL = 3

SIDE_BUY = 1
SIDE_SELL = -1

REASON_SL = 1
REASON_TP = 2
REASON_TIME = 3


@njit
def _resolve_long(bid_open, bid_high, bid_low, sl, tp):
    nan = np.nan
    if sl == sl and bid_open <= sl:
        return bid_open, 1
    if tp == tp and bid_open >= tp:
        return bid_open, 2
    if sl == sl and bid_low <= sl:
        return sl, 1
    if tp == tp and bid_high >= tp:
        return tp, 2
    return nan, 0


@njit
def _resolve_short(ask_open, ask_high, ask_low, sl, tp):
    nan = np.nan
    if sl == sl and ask_open >= sl:
        return ask_open, 1
    if tp == tp and ask_open <= tp:
        return ask_open, 2
    if sl == sl and ask_high >= sl:
        return sl, 1
    if tp == tp and ask_low <= tp:
        return tp, 2
    return nan, 0


@njit
def _market_px(is_buy, ask_open, bid_open, slip):
    if is_buy == 1:
        return ask_open + slip
    return bid_open - slip


@njit
def _pending_px(is_buy, otype, price, ask_open, ask_high, ask_low, bid_open, bid_high, bid_low):
    nan = np.nan
    if otype == OT_STOP:
        if is_buy == 1:
            if ask_high < price:
                return nan
            if ask_open >= price:
                return ask_open
            return price
        if bid_low > price:
            return nan
        if bid_open <= price:
            return bid_open
        return price
    if otype == OT_LIMIT:
        if is_buy == 1:
            if ask_low > price:
                return nan
            if ask_open <= price:
                return ask_open
            return price
        if bid_high < price:
            return nan
        if bid_open >= price:
            return bid_open
        return price
    return nan


@njit
def _trail(is_buy, sl, trail, bid_high, ask_low):
    if trail != trail or trail <= 0:
        return sl
    if is_buy == 1:
        cand = bid_high - trail
        if sl != sl:
            return cand
        return sl if sl > cand else cand
    cand = ask_low + trail
    if sl != sl:
        return cand
    return sl if sl < cand else cand


@njit
def _sl_tp_ok(is_buy, px, sl, tp):
    if sl == sl:
        if is_buy == 1 and sl >= px:
            return 0
        if is_buy == 0 and sl <= px:
            return 0
    if tp == tp:
        if is_buy == 1 and tp <= px:
            return 0
        if is_buy == 0 and tp >= px:
            return 0
    return 1


@njit
def jit_loop(
    bid_o, bid_h, bid_l, bid_c,
    ask_o, ask_h, ask_l, ask_c,
    created, side, otype, price, sl, tp, lots, trail, max_hold,
    slip, extra_slip, reject, rng,
    start_balance, contract_size, commission_per_lot,
):
    n = bid_o.shape[0]
    n_ord = created.shape[0]
    live = np.zeros(n_ord, dtype=np.int8)
    filled = np.zeros(n_ord, dtype=np.int8)
    entry = np.zeros(n_ord, dtype=np.float64)
    entry_bar = np.full(n_ord, -1, dtype=np.int64)
    cur_sl = sl.copy()
    cur_tp = tp.copy()
    closed_pnl = np.zeros(n_ord, dtype=np.float64)
    closed_reason = np.zeros(n_ord, dtype=np.int8)
    balance = start_balance
    for t in range(n):
        for i in range(n_ord):
            if filled[i] == 1 or live[i] == 1:
                continue
            if created[i] >= t:
                continue
            if reject[i] == 1:
                filled[i] = 1
                continue
            is_buy = 1 if side[i] == SIDE_BUY else 0
            px = nan = np.nan
            extra = extra_slip[i] if extra_slip.shape[0] == n_ord else 0.0
            if otype[i] == OT_MARKET or otype[i] == OT_TRAIL:
                px = _market_px(is_buy, ask_o[t], bid_o[t], slip + extra)
            else:
                px = _pending_px(
                    is_buy, otype[i], price[i],
                    ask_o[t], ask_h[t], ask_l[t],
                    bid_o[t], bid_h[t], bid_l[t],
                )
            if px != px:
                continue
            if _sl_tp_ok(is_buy, px, sl[i], tp[i]) == 0:
                filled[i] = 1
                continue
            filled[i] = 1
            live[i] = 1
            entry[i] = px
            entry_bar[i] = t
            balance -= lots[i] * commission_per_lot
            if otype[i] == OT_TRAIL and (cur_sl[i] != cur_sl[i]) and trail[i] == trail[i]:
                cur_sl[i] = px - trail[i] if is_buy == 1 else px + trail[i]
            if is_buy == 1:
                xpx, reason = _resolve_long(bid_o[t], bid_h[t], bid_l[t], cur_sl[i], cur_tp[i])
            else:
                xpx, reason = _resolve_short(ask_o[t], ask_h[t], ask_l[t], cur_sl[i], cur_tp[i])
            if reason != 0:
                pnl = lots[i] * contract_size * ((xpx - px) if is_buy == 1 else (px - xpx))
                balance += pnl
                closed_pnl[i] = pnl
                closed_reason[i] = reason
                live[i] = 0
        for i in range(n_ord):
            if live[i] != 1:
                continue
            is_buy = 1 if side[i] == SIDE_BUY else 0
            if max_hold[i] == max_hold[i]:
                if t - entry_bar[i] >= int(max_hold[i]):
                    xpx = bid_o[t] if is_buy == 1 else ask_o[t]
                    pnl = lots[i] * contract_size * (
                        (xpx - entry[i]) if is_buy == 1 else (entry[i] - xpx)
                    )
                    balance += pnl
                    closed_pnl[i] = pnl
                    closed_reason[i] = REASON_TIME
                    live[i] = 0
                    continue
            if is_buy == 1:
                xpx, reason = _resolve_long(bid_o[t], bid_h[t], bid_l[t], cur_sl[i], cur_tp[i])
            else:
                xpx, reason = _resolve_short(ask_o[t], ask_h[t], ask_l[t], cur_sl[i], cur_tp[i])
            if reason != 0:
                pnl = lots[i] * contract_size * (
                    (xpx - entry[i]) if is_buy == 1 else (entry[i] - xpx)
                )
                balance += pnl
                closed_pnl[i] = pnl
                closed_reason[i] = reason
                live[i] = 0
                continue
            cur_sl[i] = _trail(is_buy, cur_sl[i], trail[i], bid_h[t], ask_l[t])
    return balance, closed_pnl, closed_reason, entry


_TYPE_MAP = {"market": OT_MARKET, "limit": OT_LIMIT, "stop": OT_STOP, "trailing_stop": OT_TRAIL}
_REASON_NAME = {REASON_SL: "SL", REASON_TP: "TP", REASON_TIME: "Time"}


def orders_to_arrays(orders, n_bars, reject_rate=0.0, extra_slippage=0.0, rng=None):
    n = len(orders)
    created = np.zeros(n, dtype=np.int64)
    side = np.zeros(n, dtype=np.int64)
    otype = np.zeros(n, dtype=np.int64)
    price = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    tp = np.full(n, np.nan)
    lots = np.zeros(n, dtype=np.float64)
    trail = np.full(n, np.nan)
    max_hold = np.full(n, np.nan)
    extra = np.zeros(n, dtype=np.float64)
    reject = np.zeros(n, dtype=np.int64)
    for i, o in enumerate(orders):
        if getattr(o, "order_type", None) == "exit":
            raise ValueError("JIT does not support EXIT orders; use OOEngine")
        created[i] = int(o.created_bar if o.created_bar is not None else -1)
        side[i] = SIDE_BUY if o.side == "B" else SIDE_SELL
        otype[i] = _TYPE_MAP.get(o.order_type, OT_MARKET)
        if o.price is not None:
            price[i] = float(o.price)
        if o.sl is not None:
            sl[i] = float(o.sl)
        if o.tp is not None:
            tp[i] = float(o.tp)
        lots[i] = float(o.lots or 0.0)
        if o.trail_pips:
            trail[i] = float(o.trail_pips)
        if o.max_hold_bars is not None:
            max_hold[i] = float(o.max_hold_bars)
        extra[i] = float(extra_slippage)
        if reject_rate > 0 and rng is not None and rng.random() < reject_rate:
            reject[i] = 1
    return created, side, otype, price, sl, tp, lots, trail, max_hold, extra, reject


def run_jit(handler, orders, start_balance, slippage_pips=0.0, extra_slippage_pips=0.0,
            reject_entry_rate=0.0, rng=None, commission_per_lot=0.0):
    df = handler.data
    spec = handler.spec
    pip = spec["pip_size"]
    arrays = orders_to_arrays(
        orders, len(df),
        reject_rate=reject_entry_rate,
        extra_slippage=extra_slippage_pips * pip,
        rng=rng,
    )
    created, side, otype, price, sl, tp, lots, trail, max_hold, extra, reject = arrays
    trail_px = trail.copy()
    trail_px = np.where(np.isnan(trail_px), trail_px, trail_px * pip)
    bal, pnl, reason, entry = jit_loop(
        df["BidOpen"].to_numpy(np.float64),
        df["BidHigh"].to_numpy(np.float64),
        df["BidLow"].to_numpy(np.float64),
        df["BidClose"].to_numpy(np.float64),
        df["AskOpen"].to_numpy(np.float64),
        df["AskHigh"].to_numpy(np.float64),
        df["AskLow"].to_numpy(np.float64),
        df["AskClose"].to_numpy(np.float64),
        created, side, otype, price, sl, tp, lots, trail_px, max_hold,
        slippage_pips * pip, extra, reject,
        np.zeros(1, dtype=np.float64),
        float(start_balance), spec["contract_size"], float(commission_per_lot),
    )
    trades = []
    for i in range(len(orders)):
        if reason[i] == 0:
            continue
        trades.append(
            {
                "Entry": float(entry[i]),
                "Result": float(pnl[i]),
                "Exit_Reason": _REASON_NAME.get(int(reason[i]), "Unknown"),
                "Lots": float(lots[i]),
            }
        )
    return {"balance": float(bal), "trades": trades, "pnl": pnl, "entry": entry, "reason": reason}
