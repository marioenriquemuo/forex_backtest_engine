"""BacktestEngine public surface.

OOEngine(htf_rules=...) puts PIT higher-timeframe frames on account["htf"].
"""

from broker import BrokerSimulator
from data_handler import DataHandler, align_htf_pit, instrument_spec, resample_ohlc
from engine import EngineResult, OOEngine
from engine_jit import run_jit
from metrics import (
    compute_metrics,
    deflated_sharpe_ratio,
    dsr_from_equity,
    index_from_result,
    probabilistic_sharpe_ratio,
)
from orders import EXIT, LIMIT, MARKET, STOP, TRAILING_STOP, Order
from portfolio import PortfolioManager
from reporting import export_logs, plot_dashboard, plot_trade
from strategy import LegacyStrategyAdapter, Strategy
from stress import moving_block_bootstrap, perturb_execution, random_param_search, sample_param_sets
