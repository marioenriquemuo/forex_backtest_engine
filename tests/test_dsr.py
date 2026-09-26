"""DSR and random parameter search tests."""

import math

import numpy as np

from metrics import (
    _norm_cdf,
    _norm_ppf,
    deflated_sharpe_ratio,
    dsr_from_equity,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    return_moments,
    sharpe_std_error,
)
from stress import random_param_search, sample_param_sets


def test_norm_ppf_cdf_roundtrip():
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        z = _norm_ppf(p)
        assert abs(_norm_cdf(z) - p) < 1e-4


def test_psr_n1_high_sharpe_near_one():
    # Long strong drift, low vol → high per-bar SR → PSR vs 0 near 1.
    rng = np.random.default_rng(0)
    rets = 0.002 + 0.001 * rng.standard_normal(500)
    mom = return_moments(rets)
    psr = probabilistic_sharpe_ratio(mom["sr"], mom["n_obs"], mom["skew"], mom["kurtosis"], 0.0)
    assert psr > 0.95


def test_dsr_falls_when_n_trials_grows():
    rng = np.random.default_rng(1)
    # Mild edge so DSR is not already capped at 1.0 for both N.
    rets = 0.0003 + 0.01 * rng.standard_normal(400)
    mom = return_moments(rets)
    d1 = deflated_sharpe_ratio(mom["sr"], mom["n_obs"], mom["skew"], mom["kurtosis"], 1)
    d200 = deflated_sharpe_ratio(mom["sr"], mom["n_obs"], mom["skew"], mom["kurtosis"], 200)
    assert d200["sr_star"] > d1["sr_star"]
    assert d1["dsr"] >= d200["dsr"]
    assert d1["dsr"] - d200["dsr"] > 1e-6


def test_expected_max_sharpe_zero_for_one_trial():
    assert expected_max_sharpe(1, 0.1) == 0.0
    assert expected_max_sharpe(10, 0.1) > 0.0


def test_dsr_from_equity_matches_moments():
    eq = [100.0 * ((1.001) ** i) for i in range(200)]
    out = dsr_from_equity(eq, n_trials=5, periods_per_year=252.0)
    assert 0.0 <= out["dsr"] <= 1.0
    assert out["n_trials"] == 5
    assert out["n_obs"] == 199


def test_sample_param_sets_full_grid_when_small():
    space = {"a": [1, 2], "b": [10, 20]}
    sets = sample_param_sets(space, n_trials=100, rng=np.random.default_rng(0))
    assert len(sets) == 4
    keys = {(s["a"], s["b"]) for s in sets}
    assert keys == {(1, 10), (1, 20), (2, 10), (2, 20)}


def test_sample_param_sets_unique_cap():
    space = {"a": list(range(10)), "b": list(range(10))}
    sets = sample_param_sets(space, n_trials=12, rng=np.random.default_rng(2))
    assert len(sets) == 12
    assert len({(s["a"], s["b"]) for s in sets}) == 12


def test_random_param_search_picks_best_and_uses_n():
    def run_fn(params):
        # Higher "edge" → steeper equity.
        edge = float(params["edge"])
        return [100.0 * ((1.0 + edge) ** i) for i in range(120)]

    space = {"edge": [0.0001, 0.0005, 0.002]}
    out = random_param_search(
        run_fn,
        space,
        n_trials=10,
        rng=np.random.default_rng(3),
        periods_per_year=252.0,
        dsr_threshold=0.90,
    )
    assert out["n_trials"] == 3  # full grid of 3
    assert out["best_params"]["edge"] == 0.002
    assert out["dsr"]["n_trials"] == 3
    assert "passes" in out
    assert out["best_sharpe"] == max(t["sharpe"] for t in out["trials"])


def test_sharpe_std_error_normal_null():
    # Under SR=0, skew=0, kurt=3: se = 1/sqrt(n-1)
    se = sharpe_std_error(0.0, 101, 0.0, 3.0)
    assert abs(se - 1.0 / math.sqrt(100.0)) < 1e-12
