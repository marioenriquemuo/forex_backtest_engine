"""Block bootstrap of closed trades and parametric execution perturbation."""

import copy

import numpy as np

from engine import OOEngine


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
