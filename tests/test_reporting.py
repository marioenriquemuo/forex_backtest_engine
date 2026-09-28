"""Exporter and plotter smoke tests."""

from pathlib import Path

from data_handler import DataHandler
from engine import OOEngine
from orders import BUY, MARKET, Order
from reporting import export_logs, plot_dashboard, plot_trade
from tests.helpers import FireOnce, flat_bar, make_bars


def test_export_and_plot_contain_fixture_trade(tmp_path):
    rows = [
        flat_bar(1.10),
        (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
        (1.1000, 1.1300, 1.0700, 1.1200, 1.1002, 1.1302, 1.0702, 1.1202),
    ]
    df = make_bars(rows)
    handler = DataHandler(df, pair="EUR/USD")
    order = Order(side=BUY, order_type=MARKET, lots=0.1, sl=1.0900, tp=1.1200, pair="EUR/USD")
    result = OOEngine(
        {"EUR/USD": handler},
        {"EUR/USD": FireOnce(0, order)},
        start_balance=100000.0,
        max_risk_per_trade=10.0,
        max_active_trades_per_pair=10,
    ).run()
    paths = export_logs(result, tmp_path)
    text = Path(paths["trades"]).read_text()
    assert str(result.trades[0]["Entry"])[:6] in text
    assert result.trades[0]["SL"] is not None
    trade_html = plot_trade(handler, result.trades[0], str(tmp_path / "trade.html"))
    html = Path(trade_html).read_text()
    assert "SL" in html
    dash = plot_dashboard(result, str(tmp_path / "dash.html"))
    assert Path(dash).is_file()
    assert "Equity" in Path(dash).read_text()
