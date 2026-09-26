"""Shim: multi-pair runner on the modular OOEngine."""

import pandas as pd

from data_handler import DataHandler, instrument_spec, normalize_pair
from engine import OOEngine
from strategy import LegacyStrategyAdapter, Strategy


def load_and_prepare_all_data_consistent(pairs, strategy_params, start_date, end_date):
    all_dfs = []
    print("   ...Loading data for %d pairs..." % len(pairs))
    for pair in pairs:
        url = strategy_params[pair]["URL_DATA"]
        df = DataHandler.from_csv(url, pair=pair, start=start_date, end=end_date).data
        df = df.copy()
        df["Pair"] = pair
        all_dfs.append(df)
    if not all_dfs:
        raise ValueError("No data loaded for pairs: %s" % pairs)
    master_df = pd.concat(all_dfs).sort_index()
    master_df["pair_index"] = master_df.groupby("Pair").cumcount()
    return master_df


class MultiPairBacktestEngineConsistent(object):
    def __init__(self, master_data, pairs, strategy_params, config, strategy_class):
        self.master_data = master_data
        self.pairs = [normalize_pair(p) for p in pairs]
        self.config = config
        self.strategy_class = strategy_class
        self.current_balance = config["START_BALANCE"]
        self.equity_curve = [self.current_balance]
        self.trade_log = []
        self.engines = {}
        self.strategies = {}
        handlers = {}
        wrapped = {}
        contract = config.get("CONTRACT_SIZE")
        for pair in pairs:
            key = normalize_pair(pair)
            pair_data = master_data[master_data["Pair"] == pair].copy()
            if pair_data.empty:
                pair_data = master_data[master_data["Pair"] == key].copy()
            p_params = strategy_params[pair].copy()
            p_params.pop("URL_DATA", None)
            p_params["RR"] = config["RISK_REWARD_RATIO"]
            strat = strategy_class(pair_data, **p_params)
            self.strategies[key] = strat
            handlers[key] = DataHandler(pair_data, pair=key, contract_size=contract)
            if strat.__class__.on_bar is Strategy.on_bar:
                wrapped[key] = LegacyStrategyAdapter(strat, pair=key)
            else:
                wrapped[key] = strat
            spec = instrument_spec(key, contract_size=contract)
            self.engines[key] = {"pip_size": spec["pip_size"], "contract_size": spec["contract_size"]}
        self._oo = OOEngine(
            handlers,
            wrapped,
            start_balance=config["START_BALANCE"],
            slippage_pips=config.get("SLIPPAGE_PIPS", 1.0),
            max_active_trades_per_pair=config["MAX_ACTIVE_TRADES_PER_PAIR"],
            max_risk_per_trade=config["MAX_RISK_PER_TRADE"],
            contract_size=contract,
            leverage=config.get("LEVERAGE", 30.0),
            commission_per_lot=config.get("COMMISSION_PER_LOT", 0.0),
            commission_pct=config.get("COMMISSION_PCT", 0.0),
            swap_long_per_lot=config.get("SWAP_LONG_PER_LOT", 0.0),
            swap_short_per_lot=config.get("SWAP_SHORT_PER_LOT", 0.0),
        )
        self.result = None

    def run_backtest(self):
        self.result = self._oo.run()
        self.trade_log = self.result.trades
        self.equity_curve = self.result.equity_curve
        self.current_balance = self.result.balance
        return pd.DataFrame(self.trade_log)
