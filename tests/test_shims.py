"""Old import paths still resolve."""

def test_legacy_shim_imports():
    from backtest_fun_consistent import Strategy, load_data_consistent, ConsistentBacktestEngine
    from multi_pair_engine_consistent import (
        MultiPairBacktestEngineConsistent,
        load_and_prepare_all_data_consistent,
    )
    from portfolio_runner_consistent import PortfolioBacktestRunnerConsistent
    assert Strategy is not None
    assert callable(load_data_consistent)
    assert ConsistentBacktestEngine is not None
    assert MultiPairBacktestEngineConsistent is not None
    assert callable(load_and_prepare_all_data_consistent)
    assert PortfolioBacktestRunnerConsistent is not None
