"""Performance metrics from closed trades and the floating equity curve.

Sharpe/Sortino are annualised from the bar clock, not a stock-day 252.
Pass `index` (bar timestamps) or `periods_per_year`. With neither, those ratios stay 0.
Win rate / expectancy / payoff use Result net of Commission when present.
"""

import math

import numpy as np
import pandas as pd

_SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


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
