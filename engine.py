"""Object-oriented backtest engine. Signal on close t, fill from open t+1.

Optional htf_rules attach completed higher-timeframe frames on account["htf"].
During on_bar, window/handler.data/strategy.data are prefixes only.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from broker import BrokerSimulator
from data_handler import DataHandler, normalize_pair
from orders import EXIT, Order
from portfolio import PortfolioManager
from strategy import Strategy


@dataclass
class EngineResult:
    trades: List[dict]
    equity_curve: List[float]
    balance: float
    fills: list = field(default_factory=list)
    queue_log: list = field(default_factory=list)
    handlers: dict = field(default_factory=dict)
    closed: list = field(default_factory=list)


class OOEngine:
    def __init__(
        self,
        handlers: Dict[str, DataHandler],
        strategies: Dict[str, Strategy],
        start_balance,
        slippage_pips=0.0,
        max_active_trades_per_pair=2,
        leverage=30.0,
        stop_out_fraction=0.5,
        commission_per_lot=0.0,
        commission_pct=0.0,
        swap_long_per_lot=0.0,
        swap_short_per_lot=0.0,
        max_risk_per_trade=0.01,
        contract_size=None,
        extra_slippage_pips=0.0,
        reject_entry_rate=0.0,
        rng=None,
        htf_rules=(),
    ):
        self.handlers = {normalize_pair(k): v for k, v in handlers.items()}
        self.strategies = {normalize_pair(k): v for k, v in strategies.items()}
        self.max_active = int(max_active_trades_per_pair)
        self.extra_slippage_pips = float(extra_slippage_pips)
        self.reject_entry_rate = float(reject_entry_rate)
        self.rng = rng
        self.htf_rules = tuple(htf_rules or ())
        self._htf = {
            pair: {rule: handler.pit_htf(rule) for rule in self.htf_rules}
            for pair, handler in self.handlers.items()
        }
        self.portfolio = PortfolioManager(
            start_balance=start_balance,
            leverage=leverage,
            stop_out_fraction=stop_out_fraction,
            commission_per_lot=commission_per_lot,
            commission_pct=commission_pct,
            swap_long_per_lot=swap_long_per_lot,
            swap_short_per_lot=swap_short_per_lot,
            max_risk_per_trade=max_risk_per_trade,
            default_contract_size=float(contract_size or 100000.0),
        )
        self.brokers = {}
        for pair, handler in self.handlers.items():
            slip = float(slippage_pips) * handler.spec["pip_size"]
            self.brokers[pair] = BrokerSimulator(slippage=slip)
        self._bar_index = {pair: -1 for pair in self.handlers}
        self._notify_closed_from = 0

    def _union_index(self):
        idxs = [h._data.index for h in self.handlers.values()]
        out = idxs[0]
        for idx in idxs[1:]:
            out = out.union(idx)
        return out.sort_values()

    def _htf_asof(self, pair, t):
        out = {}
        for rule, frame in self._htf.get(pair, {}).items():
            out[rule] = frame.iloc[: t + 1].copy()
        return out

    @staticmethod
    def _truncate_strategy_data(strat, ts):
        restored = []
        objs = [strat]
        legacy = getattr(strat, "legacy", None)
        if legacy is not None and legacy is not strat:
            objs.append(legacy)
        for obj in objs:
            data = getattr(obj, "data", None)
            if not isinstance(data, pd.DataFrame) or data.empty:
                continue
            restored.append((obj, data))
            obj.data = data.loc[:ts].copy()
        return restored

    @staticmethod
    def _restore_strategy_data(restored):
        for obj, data in restored:
            obj.data = data

    def _notify_legacy(self):
        newly = self.portfolio.closed[self._notify_closed_from :]
        self._notify_closed_from = len(self.portfolio.closed)
        for trade in newly:
            strat = self.strategies.get(trade.pair)
            inner = getattr(strat, "legacy", strat)
            d = trade.as_legacy_dict()
            if hasattr(inner, "learn_outcome") and d.get("Features") is not None:
                inner.learn_outcome(d["Features"], 1 if trade.pnl > 0 else 0)
            if hasattr(inner, "on_trade_closed"):
                try:
                    inner.on_trade_closed(d)
                except Exception:
                    pass

    def _maybe_reject(self, order, bar_index):
        if self.reject_entry_rate <= 0 or self.rng is None:
            return False
        if order.order_type == "exit":
            return False
        return float(self.rng.random()) < self.reject_entry_rate

    def run(self):
        all_fills = []
        for ts in self._union_index():
            bars_by_pair = {}
            for pair, handler in self.handlers.items():
                if ts not in handler._data.index:
                    continue
                self._bar_index[pair] += 1
                t = self._bar_index[pair]
                bar = handler.bar(t)
                bars_by_pair[pair] = bar
                broker = self.brokers[pair]
                extra = self.extra_slippage_pips * handler.spec["pip_size"]
                broker.process_bar(
                    bar,
                    t,
                    self.portfolio,
                    reject_mask=self._maybe_reject,
                    extra_slippage=extra,
                )

            if not bars_by_pair:
                continue
            any_bar = next(iter(bars_by_pair.values()))
            self.portfolio.apply_swap(any_bar.name)
            self.portfolio.check_margin(bars_by_pair, max(self._bar_index.values()))
            self._notify_legacy()

            if not self.portfolio.is_blown_up:
                for pair, bar in bars_by_pair.items():
                    t = self._bar_index[pair]
                    n_active = sum(1 for p in self.portfolio.positions if p.pair == pair)
                    at_cap = n_active >= self.max_active
                    handler = self.handlers[pair]
                    handler._asof = t
                    restored = []
                    try:
                        window = handler.window(t)
                        account = {
                            "equity": self.portfolio.equity(bars_by_pair),
                            "balance": self.portfolio.balance,
                            "active_trades": self.portfolio.active_trade_dicts(),
                            "positions": list(self.portfolio.positions),
                            "htf": self._htf_asof(pair, t),
                        }
                        restored = self._truncate_strategy_data(
                            self.strategies[pair], window.index[-1]
                        )
                        orders = self.strategies[pair].on_bar(window, account) or []
                    finally:
                        self._restore_strategy_data(restored)
                        handler._asof = None
                    if not isinstance(orders, list):
                        orders = [orders]
                    cleaned = []
                    for o in orders:
                        if not isinstance(o, Order):
                            continue
                        o.pair = o.pair or pair
                        if at_cap and o.order_type != EXIT:
                            continue
                        cleaned.append(o)
                    self.brokers[pair].enqueue(cleaned, created_bar=t)

            self.portfolio.snapshot_equity(bars_by_pair)

        return EngineResult(
            trades=[t.as_legacy_dict() for t in self.portfolio.closed],
            equity_curve=list(self.portfolio.equity_curve),
            balance=self.portfolio.balance,
            fills=all_fills,
            queue_log=sum((b.queue_log for b in self.brokers.values()), []),
            handlers=self.handlers,
            closed=list(self.portfolio.closed),
        )
