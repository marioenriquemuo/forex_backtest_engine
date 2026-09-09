"""Headless CSV/JSON exporters, trade plotter, dashboard charts."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def export_logs(result, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades = pd.DataFrame(result.trades)
    trades_path = out / "trade_log.csv"
    trades.to_csv(str(trades_path), index=False)
    eq = pd.DataFrame({"equity": result.equity_curve})
    eq_path = out / "equity_curve.csv"
    eq.to_csv(str(eq_path), index=False)
    queue_path = out / "order_queue.json"
    with queue_path.open("w") as fh:
        json.dump(result.queue_log, fh, indent=2, default=str)
    summary = {
        "balance": result.balance,
        "n_trades": int(len(result.trades)),
        "n_equity": int(len(result.equity_curve)),
    }
    with (out / "run_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    return {"trades": trades_path, "equity": eq_path, "queue": queue_path}


def plot_trade(handler, trade, out_html, lookback=20, lookforward=5):
    df = handler.data
    entry_i = int(trade.get("Entry_Bar_Index") or 0)
    exit_i = int(trade.get("Exit_Bar_Index") or entry_i)
    lo = max(0, entry_i - lookback)
    hi = min(len(df), exit_i + lookforward + 1)
    win = df.iloc[lo:hi]
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=win.index,
            open=win["BidOpen"],
            high=win["BidHigh"],
            low=win["BidLow"],
            close=win["BidClose"],
            name="Bid",
        )
    )
    fig.add_trace(
        go.Candlestick(
            x=win.index,
            open=win["AskOpen"],
            high=win["AskHigh"],
            low=win["AskLow"],
            close=win["AskClose"],
            name="Ask",
            increasing_line_color="#888",
            decreasing_line_color="#444",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trade["Date"]],
            y=[trade["Entry"]],
            mode="markers",
            name="Entry",
            marker=dict(color="blue", size=10, symbol="triangle-up"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trade["Exit_Date"]],
            y=[trade["Exit_Price"]],
            mode="markers",
            name="Exit",
            marker=dict(color="red", size=10, symbol="x"),
        )
    )
    if trade.get("SL") is not None:
        fig.add_hline(y=trade["SL"], line_dash="dash", annotation_text="SL")
    if trade.get("TP") is not None:
        fig.add_hline(y=trade["TP"], line_dash="dot", annotation_text="TP")
    mae_y = trade["Entry"] - trade["MAE"] if trade.get("Type") == "B" else trade["Entry"] + trade["MAE"]
    mfe_y = trade["Entry"] + trade["MFE"] if trade.get("Type") == "B" else trade["Entry"] - trade["MFE"]
    if trade.get("MAE") is not None:
        fig.add_trace(go.Scatter(x=[trade["Date"]], y=[mae_y], mode="markers", name="MAE"))
    if trade.get("MFE") is not None:
        fig.add_trace(go.Scatter(x=[trade["Date"]], y=[mfe_y], mode="markers", name="MFE"))
    fig.update_layout(title="Trade Bid/Ask with SL/TP and MAE/MFE", template="plotly_white")
    path = Path(out_html)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def plot_dashboard(result, out_html):
    eq = pd.Series(result.equity_curve, dtype=float)
    peak = eq.cummax()
    underwater = (eq - peak) / peak.replace(0, pd.NA)
    trades = pd.DataFrame(result.trades)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Equity", "Underwater drawdown", "PnL by hour", "PnL by weekday"),
    )
    fig.add_trace(go.Scatter(y=eq, name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(y=underwater.fillna(0.0), name="Underwater"), row=1, col=2)
    if not trades.empty and "Exit_Date" in trades.columns:
        ts = pd.to_datetime(trades["Exit_Date"], utc=True, errors="coerce")
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("America/New_York")
        trades = trades.copy()
        trades["hour"] = ts.dt.hour
        trades["dow"] = ts.dt.dayofweek
        by_hour = trades.groupby("hour")["Result"].sum()
        by_dow = trades.groupby("dow")["Result"].sum()
        fig.add_trace(go.Bar(x=by_hour.index, y=by_hour.values, name="Hour"), row=2, col=1)
        fig.add_trace(go.Heatmap(z=[by_dow.values], x=by_dow.index, name="DOW"), row=2, col=2)
    fig.update_layout(title="Backtest dashboard", template="plotly_white")
    path = Path(out_html)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path
