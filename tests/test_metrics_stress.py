"""Metrics, JIT parity, block bootstrap, execution perturbation."""

import numpy as np

from data_handler import DataHandler
from engine import OOEngine
from engine_jit import run_jit
from metrics import compute_metrics, max_drawdown, ulcer_index
from orders import BUY, MARKET, Order
from stress import moving_block_bootstrap
from strategy import Strategy
from tests.helpers import FireOnce, flat_bar, make_bars


class Capture(Strategy):
    def __init__(self, inner):
        super(Capture, self).__init__()
        self.inner = inner
        self.orders = []

    def on_bar(self, window, account):
        orders = self.inner.on_bar(window, account) or []
        t = len(window) - 1
        for o in orders:
            o.created_bar = t
            self.orders.append(o)
        return orders


def test_metrics_hand_computed_five_trades():
    trades = [
        {"Result": 100.0, "Lots": 1.0, "MAE": 0.001, "MFE": 0.002},
        {"Result": -50.0, "Lots": 1.0, "MAE": 0.002, "MFE": 0.0005},
        {"Result": 80.0, "Lots": 1.0, "MAE": 0.001, "MFE": 0.003},
        {"Result": -40.0, "Lots": 1.0, "MAE": 0.0015, "MFE": 0.0004},
        {"Result": 30.0, "Lots": 1.0, "MAE": 0.0005, "MFE": 0.001},
    ]
    eq = [1000.0, 1100.0, 1050.0, 1130.0, 1090.0, 1120.0]
    m = compute_metrics(
        trades, eq, start_balance=1000.0, years=1.0, pip_size=0.0001, contract_size=100000.0
    )
    assert abs(m["total_net_profit"] - 120.0) < 1e-9
    assert abs(m["expectancy_currency"] - 24.0) < 1e-9
    assert abs(m["payoff_ratio"] - (70.0 / 45.0)) < 1e-9
    dd, dd_pct, dur = max_drawdown(eq)
    assert abs(m["max_drawdown"] - dd) < 1e-9
    assert abs(m["max_drawdown_pct"] - dd_pct) < 1e-9
    assert m["max_drawdown_duration"] == dur
    assert abs(m["ulcer_index"] - ulcer_index(eq)) < 1e-9
    assert abs(m["mae_mean"] - 0.0012) < 1e-9
    assert m["cagr"] != 0.0
    assert "calmar" in m


def test_oo_vs_jit_parity():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        (1.1000, 1.1300, 1.0700, 1.1200, 1.1002, 1.1302, 1.0702, 1.1202),
    ]
    df = make_bars(rows)
    handler = DataHandler(df, pair="EUR/USD")
    order = Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.0900, tp=1.1200, pair="EUR/USD")
    inner = FireOnce(0, order)
    cap = Capture(inner)
    oo = OOEngine(
        {"EUR/USD": handler},
        {"EUR/USD": cap},
        start_balance=100000.0,
        slippage_pips=0.0,
        max_risk_per_trade=10.0,
        max_active_trades_per_pair=10,
    )
    result = oo.run()
    jit = run_jit(handler, cap.orders, start_balance=100000.0, slippage_pips=0.0)
    assert len(result.trades) == 1
    assert len(jit["trades"]) == 1
    assert abs(result.trades[0]["Entry"] - jit["trades"][0]["Entry"]) < 1e-12
    assert result.trades[0]["Exit_Reason"] == jit["trades"][0]["Exit_Reason"]
    assert abs(result.trades[0]["Result"] - jit["trades"][0]["Result"]) < 1e-8


def test_perturbation_zero_matches_and_slippage_changes_fill():
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        (1.0900, 1.0900, 1.0800, 1.0850, 1.0902, 1.0902, 1.0802, 1.0852),
    ]
    df = make_bars(rows)
    handler = DataHandler(df, pair="EUR/USD")
    order = Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.0850, pair="EUR/USD")
    kw = dict(
        start_balance=100000.0,
        slippage_pips=0.0,
        max_risk_per_trade=10.0,
        max_active_trades_per_pair=10,
    )
    base = OOEngine({"EUR/USD": handler}, {"EUR/USD": FireOnce(0, order)}, **kw).run()
    same = OOEngine(
        {"EUR/USD": handler},
        {"EUR/USD": FireOnce(0, order)},
        extra_slippage_pips=0.0,
        reject_entry_rate=0.0,
        **kw
    ).run()
    slipped = OOEngine(
        {"EUR/USD": handler},
        {"EUR/USD": FireOnce(0, order)},
        extra_slippage_pips=2.0,
        reject_entry_rate=0.0,
        **kw
    ).run()
    assert abs(base.trades[0]["Entry"] - same.trades[0]["Entry"]) < 1e-12
    assert slipped.trades[0]["Entry"] > base.trades[0]["Entry"]


def test_block_bootstrap_preserves_streak_more_than_iid():
    rets = np.array([0.01, 0.01, 0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
    n_sim = 400
    block_size = 5
    n = len(rets)
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(1)

    def _ww_mean(seqs):
        totals = []
        for s in seqs:
            pos = s > 0
            totals.append(sum(1 for i in range(len(s) - 1) if pos[i] and pos[i + 1]))
        return float(np.mean(totals))

    block_seqs = []
    max_start = n - block_size
    for _ in range(n_sim):
        out = []
        while len(out) < n:
            blk = int(rng_a.integers(0, max_start + 1))
            out.extend(list(rets[blk : blk + block_size]))
        block_seqs.append(np.array(out[:n]))
    iid_seqs = [rng_b.permutation(rets) for _ in range(n_sim)]
    assert _ww_mean(block_seqs) > _ww_mean(iid_seqs)

    finals, _ = moving_block_bootstrap(rets, 50, 3, 1000.0, np.random.default_rng(2))
    assert len(finals) == 50
