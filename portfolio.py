"""Equity, floating PnL, commissions, rollover swap, margin stop-outs."""

from typing import List, Optional

import pandas as pd

from orders import BUY, ClosedTrade, Position, sl_on_correct_side, tp_on_correct_side
from data_handler import instrument_spec, normalize_pair


def _ny(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("America/New_York")
    else:
        t = t.tz_convert("America/New_York")
    return t


class PortfolioManager:
    def __init__(
        self,
        start_balance,
        leverage=30.0,
        stop_out_fraction=0.5,
        commission_per_lot=0.0,
        commission_pct=0.0,
        swap_long_per_lot=0.0,
        swap_short_per_lot=0.0,
        max_risk_per_trade=1.0,
        default_contract_size=100000.0,
    ):
        self.start_balance = float(start_balance)
        self.balance = float(start_balance)
        self.leverage = float(leverage)
        self.stop_out_fraction = float(stop_out_fraction)
        self.commission_per_lot = float(commission_per_lot)
        self.commission_pct = float(commission_pct)
        self.swap_long_per_lot = float(swap_long_per_lot)
        self.swap_short_per_lot = float(swap_short_per_lot)
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.default_contract_size = float(default_contract_size)
        self.positions: List[Position] = []
        self.closed: List[ClosedTrade] = []
        self.equity_curve = [float(start_balance)]
        self.last_rollover_date = None
        self.is_blown_up = False
        self._last_mark = {}

    def lots_for_risk(self, entry, sl, risk_ratio, pip_size, contract_size):
        dist = abs(float(entry) - float(sl))
        if dist <= 0:
            return None
        distance_pips = max(dist / pip_size, 1.0)
        pip_value = contract_size * pip_size
        risk_amount = self.balance * self.max_risk_per_trade * float(risk_ratio)
        lot = risk_amount / (distance_pips * pip_value)
        return max(round(lot, 2), 0.01)

    def open_position(self, order, fill, bar):
        if self.is_blown_up:
            return None
        pair = normalize_pair(order.pair or fill.pair or "")
        spec = instrument_spec(pair or "EUR/USD", contract_size=self.default_contract_size)
        pip_size = spec["pip_size"]
        contract_size = spec["contract_size"]
        is_buy = order.side == BUY
        sl = order.sl
        tp = order.tp
        if order.rr_ratio is not None and tp is None and sl is not None and order.rr_ratio > 0:
            dist = abs(fill.price - sl)
            tp = fill.price + dist * order.rr_ratio if is_buy else fill.price - dist * order.rr_ratio
        if not sl_on_correct_side(is_buy, fill.price, sl):
            return None
        if not tp_on_correct_side(is_buy, fill.price, tp):
            return None
        lots = order.lots
        if lots is None:
            if sl is None:
                return None
            lots = self.lots_for_risk(fill.price, sl, order.risk_ratio, pip_size, contract_size)
        lots = float(lots)
        if lots <= 0:
            return None
        if sl is not None:
            max_risk = self.balance * self.max_risk_per_trade * float(order.risk_ratio)
            realized = abs(fill.price - sl) * lots * contract_size
            if realized > max_risk + 1e-9:
                return None
        trail_distance = None
        if order.trail_pips:
            trail_distance = float(order.trail_pips) * pip_size
        if order.order_type == "trailing_stop" and trail_distance:
            if sl is None:
                sl = fill.price - trail_distance if is_buy else fill.price + trail_distance
        commission = lots * self.commission_per_lot
        if self.commission_pct:
            commission += self.commission_pct * lots * contract_size * fill.price
        self.balance -= commission
        pos = Position(
            side=order.side,
            entry=float(fill.price),
            lots=lots,
            sl=sl,
            tp=tp,
            pair=pair,
            entry_bar=fill.bar_index,
            entry_time=fill.time,
            trail_distance=trail_distance,
            max_hold_bars=order.max_hold_bars,
            swap=0.0,
            commission=commission,
            reason=order.reason,
            features=order.features,
            pip_size=pip_size,
            contract_size=contract_size,
        )
        fill.lots = lots
        self.positions.append(pos)
        return pos

    def touch_excursion(self, pos, bar):
        is_buy = pos.side == BUY
        if is_buy:
            adverse = pos.entry - float(bar["BidLow"])
            favorable = float(bar["BidHigh"]) - pos.entry
        else:
            adverse = float(bar["AskHigh"]) - pos.entry
            favorable = pos.entry - float(bar["AskLow"])
        pos.mae = max(pos.mae, adverse)
        pos.mfe = max(pos.mfe, favorable)

    def close_position(self, pos, exit_price, reason, bar, bar_index):
        if pos not in self.positions:
            return None
        is_buy = pos.side == BUY
        if is_buy:
            pnl = pos.lots * (exit_price - pos.entry) * pos.contract_size
        else:
            pnl = pos.lots * (pos.entry - exit_price) * pos.contract_size
        pnl += pos.swap
        self.balance += pnl
        self.touch_excursion(pos, bar)
        trade = ClosedTrade(
            pair=pos.pair,
            side=pos.side,
            entry=pos.entry,
            exit_price=float(exit_price),
            lots=pos.lots,
            sl=pos.sl,
            tp=pos.tp,
            entry_time=pos.entry_time,
            exit_time=bar.name,
            entry_bar=pos.entry_bar,
            exit_bar=bar_index,
            pnl=pnl,
            swap=pos.swap,
            commission=pos.commission,
            exit_reason=reason,
            mae=pos.mae,
            mfe=pos.mfe,
            balance=self.balance,
            reason=pos.reason,
            features=pos.features,
            extra={"max_hold_bars": pos.max_hold_bars},
        )
        self.positions.remove(pos)
        self.closed.append(trade)
        return trade

    def floating_pnl(self, bars_by_pair):
        total = 0.0
        for pos in self.positions:
            bar = bars_by_pair.get(pos.pair)
            if bar is not None:
                self._last_mark[pos.pair] = bar
            else:
                bar = self._last_mark.get(pos.pair)
            if bar is None:
                continue
            if pos.side == BUY:
                diff = float(bar["BidClose"]) - pos.entry
            else:
                diff = pos.entry - float(bar["AskClose"])
            total += diff * pos.lots * pos.contract_size + pos.swap
        return total

    def equity(self, bars_by_pair):
        return self.balance + self.floating_pnl(bars_by_pair)

    def used_margin(self, bars_by_pair):
        if self.leverage <= 0:
            return 0.0
        total = 0.0
        for pos in self.positions:
            bar = bars_by_pair.get(pos.pair)
            price = pos.entry
            if bar is not None:
                price = float(bar["BidClose"] if pos.side == BUY else bar["AskClose"])
            total += (pos.lots * pos.contract_size * price) / self.leverage
        return total

    def apply_swap(self, ts):
        t = _ny(ts)
        if t.hour < 17:
            return
        day = t.date()
        if self.last_rollover_date is not None and day <= self.last_rollover_date:
            return
        if not self.positions:
            self.last_rollover_date = day
            return
        mult = 3.0 if t.weekday() == 2 else 1.0
        for pos in self.positions:
            rate = self.swap_long_per_lot if pos.side == BUY else self.swap_short_per_lot
            pos.swap += pos.lots * rate * mult
        self.last_rollover_date = day

    def check_margin(self, bars_by_pair, bar_index):
        used = self.used_margin(bars_by_pair)
        if used <= 0:
            return False
        eq = self.equity(bars_by_pair)
        if eq >= used * self.stop_out_fraction:
            return False
        for pos in list(self.positions):
            bar = bars_by_pair.get(pos.pair)
            if bar is None:
                continue
            px = float(bar["BidClose"] if pos.side == BUY else bar["AskClose"])
            self.close_position(pos, px, "Margin", bar, bar_index)
        self.is_blown_up = True
        return True

    def snapshot_equity(self, bars_by_pair):
        eq = self.equity(bars_by_pair)
        self.equity_curve.append(eq)
        return eq

    def active_trade_dicts(self):
        rows = []
        for pos in self.positions:
            rows.append(
                {
                    "Type": pos.side,
                    "Entry": pos.entry,
                    "SL": pos.sl,
                    "TP": pos.tp,
                    "Lots": pos.lots,
                    "Pair": pos.pair,
                    "Status": "Active",
                    "Date": pos.entry_time,
                    "Entry_Bar_Index": pos.entry_bar,
                    "Max_Hold_Bars": pos.max_hold_bars,
                    "Features": pos.features,
                    "Swap": pos.swap,
                }
            )
        return rows
