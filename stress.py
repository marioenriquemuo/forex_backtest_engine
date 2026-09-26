"""Block bootstrap, execution perturbation, and random parameter search.

random_param_search: see DSR.md.
"""

import copy
import itertools

import numpy as np

from engine import EngineResult, OOEngine
from metrics import dsr_from_equity, sharpe_from_returns, equity_returns


def moving_block_bootstrap(rets, n_sim, block_size, start_bal, rng):
    n = len(rets)
    if n == 0:
        return np.array([]), np.array([])
    if block_size < 1:
        block_size = 1
    max_start = max(0, n - block_size)
    finals = np.empty(n_sim, dtype=float)
    max_dds = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        bal = float(start_bal)
        peak = bal
        mdd = 0.0
        pos = 0
        while pos < n:
            blk = int(rng.integers(0, max(1, max_start + 1)))
            chunk = rets[blk : blk + block_size]
            for r in chunk:
                bal *= 1.0 + float(r)
                pos += 1
                if bal > peak:
                    peak = bal
                if peak > 0:
                    dd = (bal - peak) / peak
                    if dd < mdd:
                        mdd = dd
                if pos >= n:
                    break
        finals[i] = bal
        max_dds[i] = mdd
    return finals, max_dds


def iid_shuffle(rets, n_sim, start_bal, rng):
    n = len(rets)
    finals = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        shuffled = rng.permutation(rets)
        bal = float(start_bal)
        for r in shuffled:
            bal *= 1.0 + float(r)
        finals[i] = bal
    return finals


def perturb_execution(
    handlers,
    strategies,
    start_balance,
    n_runs,
    max_extra_slip_pips,
    reject_rate,
    rng,
    **engine_kwargs
):
    """Re-run the OO loop with extra market slippage and random entry rejects."""
    results = []
    for _ in range(int(n_runs)):
        extra = float(rng.uniform(0.0, max_extra_slip_pips))
        eng = OOEngine(
            {k: copy.deepcopy(h) for k, h in handlers.items()},
            {k: copy.deepcopy(s) for k, s in strategies.items()},
            start_balance,
            extra_slippage_pips=extra,
            reject_entry_rate=reject_rate,
            rng=rng,
            **engine_kwargs
        )
        results.append(eng.run())
    return results


def _space_keys_values(space):
    keys = list(space.keys())
    values = [list(space[k]) for k in keys]
    return keys, values


def _combo_count(values):
    n = 1
    for v in values:
        n *= max(len(v), 0)
    return n


def sample_param_sets(space, n_trials, rng):
    """Unique param dicts from discrete lists. Full grid if small enough. See DSR.md."""
    if not space:
        return [{}]
    keys, values = _space_keys_values(space)
    if any(len(v) == 0 for v in values):
        return []
    n_trials = int(n_trials)
    total = _combo_count(values)
    if total <= n_trials:
        out = []
        for combo in itertools.product(*values):
            out.append(dict(zip(keys, combo)))
        return out
    seen = set()
    out = []
    # Cap attempts so we do not spin forever on tiny spaces with collisions.
    attempts = 0
    max_attempts = max(n_trials * 50, 1000)
    while len(out) < n_trials and attempts < max_attempts:
        attempts += 1
        combo = tuple(rng.choice(v) for v in values)
        if combo in seen:
            continue
        seen.add(combo)
        out.append(dict(zip(keys, combo)))
    return out


def _equity_from_run(result):
    if isinstance(result, EngineResult):
        return list(result.equity_curve)
    if hasattr(result, "equity_curve"):
        return list(result.equity_curve)
    return list(result)


def random_param_search(
    run_fn,
    space,
    n_trials,
    rng,
    periods_per_year=None,
    index=None,
    dsr_threshold=0.90,
):
    """Try random knobs, pick best Sharpe, score with DSR. See DSR.md.

    run_fn(params) -> equity_curve list/array or EngineResult.
    Build a fresh strategy inside run_fn each call.
    """
    from metrics import infer_periods_per_year

    param_sets = sample_param_sets(space, n_trials, rng)
    trials = []
    best = None
    for params in param_sets:
        equity = _equity_from_run(run_fn(params))
        rets = equity_returns(equity)
        ppy = periods_per_year
        if ppy is None and index is not None:
            ppy = infer_periods_per_year(index, len(rets))
        sharpe = sharpe_from_returns(rets, ppy)
        row = {"params": dict(params), "sharpe": float(sharpe), "equity_curve": equity}
        trials.append(row)
        if best is None or row["sharpe"] > best["sharpe"]:
            best = row
    n_run = len(trials)
    if best is None:
        empty = {
            "dsr": 0.0,
            "sr": 0.0,
            "sr_star": 0.0,
            "n_trials": 0,
            "n_obs": 0,
            "skew": 0.0,
            "kurtosis": 3.0,
            "sr_std_error": 0.0,
            "sharpe_annualised": 0.0,
            "periods_per_year": float(periods_per_year or 0.0),
        }
        return {
            "trials": [],
            "best_params": None,
            "best_sharpe": 0.0,
            "best_equity": None,
            "dsr": empty,
            "passes": False,
            "n_trials": 0,
            "dsr_threshold": float(dsr_threshold),
        }
    dsr = dsr_from_equity(
        best["equity_curve"],
        n_trials=n_run,
        periods_per_year=periods_per_year,
        index=index,
    )
    passes = float(dsr["dsr"]) >= float(dsr_threshold)
    return {
        "trials": [{"params": t["params"], "sharpe": t["sharpe"]} for t in trials],
        "best_params": best["params"],
        "best_sharpe": best["sharpe"],
        "best_equity": best["equity_curve"],
        "dsr": dsr,
        "passes": passes,
        "n_trials": n_run,
        "dsr_threshold": float(dsr_threshold),
    }
