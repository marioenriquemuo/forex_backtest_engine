"""Performance metrics from closed trades and the floating equity curve.

Sharpe/Sortino are annualised from the bar clock, not a stock-day 252.
Pass `index` (bar timestamps) or `periods_per_year`. With neither, those ratios stay 0.
Win rate / expectancy / payoff use Result net of Commission when present.

DSR / PSR: see DSR.md. Formulas follow Bailey & López de Prado (2014).
"""

import math

import numpy as np
import pandas as pd

_SECONDS_PER_YEAR = 365.25 * 24 * 3600.0
_EULER_MASCHERONI = 0.5772156649015329


def _trade_frame(trades):
    if isinstance(trades, pd.DataFrame):
        df = trades.copy()
    else:
        df = pd.DataFrame(trades)
    if df.empty:
        return df
    if "Result" not in df.columns and "pnl" in df.columns:
        df["Result"] = df["pnl"]
    return df


def calendar_years(t0, t1):
    a = pd.Timestamp(t0)
    b = pd.Timestamp(t1)
    if a.tzinfo is None:
        a = a.tz_localize("UTC")
    if b.tzinfo is None:
        b = b.tz_localize("UTC")
    sec = (b - a).total_seconds()
    if sec <= 0:
        return None
    return sec / _SECONDS_PER_YEAR


def infer_periods_per_year(index, n_returns):
    """Bars-per-year from calendar span of `index` and the number of equity returns."""
    if index is None or n_returns is None or int(n_returns) < 1:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    if len(idx) < 2:
        return None
    years = calendar_years(idx[0], idx[-1])
    if years is None or years <= 0:
        return None
    return float(n_returns) / float(years)


def index_from_result(result):
    """Union of handler bar times. Matches OOEngine's walk over time."""
    handlers = getattr(result, "handlers", None) or {}
    idxs = []
    for handler in handlers.values():
        data = getattr(handler, "_data", None)
        if data is None:
            data = getattr(handler, "data", None)
        if data is None or getattr(data, "index", None) is None or len(data.index) == 0:
            continue
        idxs.append(data.index)
    if not idxs:
        return None
    out = idxs[0]
    for idx in idxs[1:]:
        out = out.union(idx)
    return out.sort_values()


def max_drawdown(equity):
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0, 0.0, 0
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    dd_pct = np.where(peak > 0, dd / peak, 0.0)
    i = int(np.argmin(dd_pct))
    duration = 0
    max_dur = 0
    for p, e in zip(peak, eq):
        if e < p:
            duration += 1
            if duration > max_dur:
                max_dur = duration
        else:
            duration = 0
    return float(dd[i]), float(dd_pct[i]), int(max_dur)


def ulcer_index(equity):
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd_pct = np.where(peak > 0, (eq - peak) / peak, 0.0)
    return float(np.sqrt(np.mean(dd_pct * dd_pct)))


def _ann_factor(periods_per_year):
    if periods_per_year is None or float(periods_per_year) <= 0:
        return 0.0
    return math.sqrt(float(periods_per_year))


def _years_from_trades(df):
    """Calendar span from earliest entry to latest exit (not row order)."""
    if df is None or df.empty:
        return None
    if "Exit_Date" not in df.columns or "Date" not in df.columns:
        return None
    t0 = pd.to_datetime(df["Date"], utc=True, errors="coerce").min()
    t1 = pd.to_datetime(df["Exit_Date"], utc=True, errors="coerce").max()
    if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
        return None
    return calendar_years(t0, t1)


# --- DSR / PSR (Bailey & López de Prado 2014); see DSR.md ---


def _norm_cdf(x):
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.0))


def _norm_ppf(p):
    """Inverse standard normal CDF (Peter J. Acklam approximation)."""
    p = float(p)
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if p == 0.5:
        return 0.0
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614736e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def equity_returns(equity_curve):
    """Bar-to-bar percent returns from an equity curve. See DSR.md."""
    eq = pd.Series(list(equity_curve) if equity_curve is not None else [], dtype=float)
    if len(eq) < 2:
        return pd.Series(dtype=float)
    return eq.pct_change().dropna()


def sharpe_from_returns(rets, periods_per_year=None):
    """Per-bar Sharpe, or annualised if periods_per_year > 0. See DSR.md."""
    r = pd.Series(rets, dtype=float).dropna()
    if len(r) < 2:
        return 0.0
    vol = float(r.std(ddof=1))
    if vol <= 0:
        return 0.0
    sr = float(r.mean()) / vol
    af = _ann_factor(periods_per_year)
    if af > 0:
        return sr * af
    return sr


def return_moments(rets):
    """n_obs, per-bar Sharpe, skew, raw kurtosis (3 for normal)."""
    r = pd.Series(rets, dtype=float).dropna()
    n = int(len(r))
    if n < 2:
        return {"n_obs": n, "sr": 0.0, "skew": 0.0, "kurtosis": 3.0}
    vol = float(r.std(ddof=1))
    sr = float(r.mean()) / vol if vol > 0 else 0.0
    skew = float(r.skew()) if n >= 3 else 0.0
    # pandas Series.kurtosis is excess; Bailey γ4 is raw kurtosis.
    kurt = float(r.kurtosis()) + 3.0 if n >= 4 else 3.0
    if math.isnan(skew):
        skew = 0.0
    if math.isnan(kurt):
        kurt = 3.0
    return {"n_obs": n, "sr": sr, "skew": skew, "kurtosis": kurt}


def sharpe_std_error(sr, n_obs, skew, kurtosis):
    """Std error of the Sharpe estimator (non-normal). Bailey & LdP."""
    n = int(n_obs)
    if n <= 1:
        return float("inf")
    sr = float(sr)
    skew = float(skew)
    kurtosis = float(kurtosis)
    inside = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * (sr * sr)
    if inside <= 0:
        inside = 1e-12
    return math.sqrt(inside / float(n - 1))


def probabilistic_sharpe_ratio(sr, n_obs, skew, kurtosis, sr_benchmark=0.0):
    """PSR in [0, 1]: Prob(true SR > benchmark). See DSR.md."""
    se = sharpe_std_error(sr, n_obs, skew, kurtosis)
    if not math.isfinite(se) or se <= 0:
        return 0.0
    return float(_norm_cdf((float(sr) - float(sr_benchmark)) / se))


def expected_max_sharpe(n_trials, sr_std):
    """Expected best Sharpe among n_trials under a null with std sr_std."""
    n = int(n_trials)
    if n < 1 or sr_std is None or float(sr_std) <= 0:
        return 0.0
    if n == 1:
        return 0.0
    u1 = 1.0 - 1.0 / float(n)
    u2 = 1.0 - 1.0 / (float(n) * math.e)
    u1 = min(max(u1, 1e-12), 1.0 - 1e-12)
    u2 = min(max(u2, 1e-12), 1.0 - 1e-12)
    z = (1.0 - _EULER_MASCHERONI) * _norm_ppf(u1) + _EULER_MASCHERONI * _norm_ppf(u2)
    return float(sr_std) * float(z)


def deflated_sharpe_ratio(sr, n_obs, skew, kurtosis, n_trials, sr_benchmark=None):
    """DSR probability in [0, 1]. Gate at >= 0.90. See DSR.md.

    If sr_benchmark is None, use E[max Sharpe] given n_trials (true DSR).
    Internals use the same per-bar SR units as `return_moments`.
    """
    n_trials = int(n_trials)
    mom_se = sharpe_std_error(sr, n_obs, skew, kurtosis)
    if sr_benchmark is None:
        null_se = sharpe_std_error(0.0, n_obs, skew, kurtosis)
        sr_star = expected_max_sharpe(n_trials, null_se)
    else:
        sr_star = float(sr_benchmark)
    dsr = probabilistic_sharpe_ratio(sr, n_obs, skew, kurtosis, sr_benchmark=sr_star)
    return {
        "dsr": dsr,
        "sr": float(sr),
        "sr_star": float(sr_star),
        "n_trials": n_trials,
        "n_obs": int(n_obs),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
        "sr_std_error": float(mom_se),
    }


def dsr_from_equity(equity_curve, n_trials, periods_per_year=None, index=None):
    """DSR dict from an equity curve. n_trials = tries you actually ran. See DSR.md."""
    rets = equity_returns(equity_curve)
    mom = return_moments(rets)
    ppy = periods_per_year
    if ppy is None and index is not None:
        ppy = infer_periods_per_year(index, mom["n_obs"])
    out = deflated_sharpe_ratio(
        mom["sr"], mom["n_obs"], mom["skew"], mom["kurtosis"], n_trials
    )
    out["sharpe_annualised"] = sharpe_from_returns(rets, ppy)
    out["periods_per_year"] = float(ppy or 0.0)
    return out


def compute_metrics(
    trades,
    equity_curve,
    start_balance,
    pip_size=0.0001,
    contract_size=100000.0,
    years=None,
    periods_per_year=None,
    index=None,
):
    df = _trade_frame(trades)
    eq = list(equity_curve) if equity_curve is not None else []
    final = float(eq[-1]) if eq else float(start_balance)
    net = final - float(start_balance)
    n = int(len(df)) if df is not None and not df.empty else 0
    empty = {
        "total_net_profit": net,
        "cagr": 0.0,
        "payoff_ratio": 0.0,
        "expectancy_currency": 0.0,
        "expectancy_pips": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_duration": 0,
        "ulcer_index": 0.0,
        "mae_mean": 0.0,
        "mfe_mean": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "trades": n,
        "win_rate": 0.0,
        "periods_per_year": 0.0,
    }
    rets = pd.Series(eq, dtype=float).pct_change().dropna() if eq else pd.Series(dtype=float)
    ppy = periods_per_year
    if ppy is None:
        ppy = infer_periods_per_year(index, len(rets))
    if ppy is None:
        ppy = 0.0
    empty["periods_per_year"] = float(ppy)

    if years is None:
        if index is not None:
            idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
            if len(idx) >= 2:
                years = calendar_years(idx[0], idx[-1])
        if years is None:
            years = _years_from_trades(df)

    if n == 0:
        dd, dd_pct, dur = max_drawdown(eq)
        empty["max_drawdown"] = dd
        empty["max_drawdown_pct"] = dd_pct
        empty["max_drawdown_duration"] = dur
        empty["ulcer_index"] = ulcer_index(eq)
        if years is not None and years > 0 and start_balance > 0 and final > 0:
            empty["cagr"] = (final / float(start_balance)) ** (1.0 / float(years)) - 1.0
        vol = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
        mean_r = float(rets.mean()) if len(rets) else 0.0
        af = _ann_factor(ppy)
        empty["sharpe"] = (mean_r / vol) * af if vol > 0 and af > 0 else 0.0
        downside = rets[rets < 0]
        dvol = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        empty["sortino"] = (mean_r / dvol) * af if dvol > 0 and af > 0 else 0.0
        empty["calmar"] = (
            (empty["cagr"] / abs(empty["max_drawdown_pct"]))
            if empty["max_drawdown_pct"] < 0
            else 0.0
        )
        return empty

    # Net of commission (Result is gross pnl+swap; commission already left the cash balance).
    pnl = df["Result"].astype(float)
    if "Commission" in df.columns:
        pnl = pnl - df["Commission"].astype(float).fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    expectancy = float(pnl.mean())
    pip_value = float(contract_size) * float(pip_size)
    lots = df["Lots"].astype(float) if "Lots" in df.columns else pd.Series([1.0] * n)
    pips = pnl / (lots * pip_value).replace(0, np.nan)
    expectancy_pips = float(pips.mean()) if pips.notna().any() else 0.0

    cagr = 0.0
    if years is not None and years > 0 and start_balance > 0 and final > 0:
        cagr = (final / float(start_balance)) ** (1.0 / float(years)) - 1.0

    dd, dd_pct, dur = max_drawdown(eq)
    ulcer = ulcer_index(eq)
    mae = float(df["MAE"].astype(float).mean()) if "MAE" in df.columns else 0.0
    mfe = float(df["MFE"].astype(float).mean()) if "MFE" in df.columns else 0.0

    vol = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    mean_r = float(rets.mean()) if len(rets) else 0.0
    af = _ann_factor(ppy)
    sharpe = (mean_r / vol) * af if vol > 0 and af > 0 else 0.0
    downside = rets[rets < 0]
    dvol = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean_r / dvol) * af if dvol > 0 and af > 0 else 0.0
    calmar = (cagr / abs(dd_pct)) if dd_pct < 0 else 0.0

    return {
        "total_net_profit": net,
        "cagr": cagr,
        "payoff_ratio": payoff,
        "expectancy_currency": expectancy,
        "expectancy_pips": expectancy_pips,
        "max_drawdown": dd,
        "max_drawdown_pct": dd_pct,
        "max_drawdown_duration": dur,
        "ulcer_index": ulcer,
        "mae_mean": mae,
        "mfe_mean": mfe,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "trades": n,
        "win_rate": float((pnl > 0).mean()),
        "periods_per_year": float(ppy),
    }
