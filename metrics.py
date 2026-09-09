"""Performance metrics from closed trades and the floating equity curve."""

import math

import numpy as np
import pandas as pd


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


def _ann_factor(n, periods_per_year):
    if n <= 1:
        return 0.0
    return math.sqrt(float(periods_per_year))


def compute_metrics(
    trades,
    equity_curve,
    start_balance,
    pip_size=0.0001,
    contract_size=100000.0,
    years=None,
    periods_per_year=252.0,
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
    }
    if n == 0:
        dd, dd_pct, dur = max_drawdown(eq)
        empty["max_drawdown"] = dd
        empty["max_drawdown_pct"] = dd_pct
        empty["max_drawdown_duration"] = dur
        empty["ulcer_index"] = ulcer_index(eq)
        return empty

    pnl = df["Result"].astype(float)
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

    if years is None:
        years = 1.0
        if "Exit_Date" in df.columns and "Date" in df.columns:
            t0 = pd.to_datetime(df["Date"].iloc[0], utc=True, errors="coerce")
            t1 = pd.to_datetime(df["Exit_Date"].iloc[-1], utc=True, errors="coerce")
            if pd.notna(t0) and pd.notna(t1) and t1 > t0:
                years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1.0 / 365.25)
    cagr = 0.0
    if years > 0 and start_balance > 0 and final > 0:
        cagr = (final / float(start_balance)) ** (1.0 / float(years)) - 1.0

    dd, dd_pct, dur = max_drawdown(eq)
    ulcer = ulcer_index(eq)
    mae = float(df["MAE"].astype(float).mean()) if "MAE" in df.columns else 0.0
    mfe = float(df["MFE"].astype(float).mean()) if "MFE" in df.columns else 0.0

    rets = pd.Series(eq, dtype=float).pct_change().dropna()
    vol = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    mean_r = float(rets.mean()) if len(rets) else 0.0
    sharpe = (mean_r / vol) * _ann_factor(len(rets), periods_per_year) if vol > 0 else 0.0
    downside = rets[rets < 0]
    dvol = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean_r / dvol) * _ann_factor(len(rets), periods_per_year) if dvol > 0 else 0.0
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
    }
