"""Shim: old import path. New code should use OOEngine / Strategy / DataHandler."""

from data_handler import DataHandler
from engine import OOEngine
from strategy import LegacyStrategyAdapter, Strategy

__all__ = [
    "Strategy",
    "load_data_consistent",
    "ConsistentBacktestEngine",
]


def load_data_consistent(PAIR, URL_DATA, start_date, end_date):
    handler = DataHandler.from_csv(
        URL_DATA, pair=PAIR, start=start_date, end=end_date
    )
    return handler.data.copy()


class ConsistentBacktestEngine(object):
    """Single-pair wrapper around OOEngine for leftover callers."""

    def __init__(
        self,
        strategy,
        data,
        start_balance,
        risk_reward_ratio,
        max_risk_per_trade,
        pip_size,
        contract_size,
        max_active_trades=2,
        slippage_pips=1.0,
    ):
        pair = getattr(strategy, "pair_name", None) or getattr(strategy, "pair", None) or "EUR/USD"
        self.handler = DataHandler(data, pair=pair, contract_size=contract_size)
        if not hasattr(strategy, "on_bar") or strategy.__class__.on_bar is Strategy.on_bar:
            wrapped = LegacyStrategyAdapter(strategy, pair=pair)
        else:
            wrapped = strategy
        self._engine = OOEngine(
            {pair: self.handler},
            {pair: wrapped},
            start_balance=start_balance,
            slippage_pips=slippage_pips,
            max_active_trades_per_pair=max_active_trades,
            max_risk_per_trade=max_risk_per_trade,
            contract_size=contract_size,
        )
        self.strategy = strategy
        self.data = data
        self.start_balance = start_balance
        self.current_balance = start_balance
        self.risk_reward_ratio = risk_reward_ratio
        self.max_risk_per_trade = max_risk_per_trade
        self.pip_size = pip_size
        self.contract_size = contract_size
        self.max_active_trades = max_active_trades
        self.trades_active = []
        self.trades_closed = []
        self.is_blown_up = False

    def run_backtest(self):
        result = self._engine.run()
        self.current_balance = result.balance
        self.trades_closed = result.trades
        self.trades_active = self._engine.portfolio.active_trade_dicts()
        self.is_blown_up = self._engine.portfolio.is_blown_up
        self.equity_curve = result.equity_curve
        return result
