import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar
from multi_pair_engine_consistent import MultiPairBacktestEngineConsistent, load_and_prepare_all_data_consistent

class PortfolioBacktestRunnerConsistent:
    def __init__(self, pairs, config, params, start, end, strategy_class):
        self.pairs = pairs
        self.config = config
        self.params = params
        self.start = start
        self.end = end
        self.strategy_class = strategy_class
        self.engine = None
        self.trade_log = pd.DataFrame()
        self.metrics = pd.DataFrame()

    def run(self):
        data = load_and_prepare_all_data_consistent(self.pairs, self.params, self.start, self.end)
        if data.empty:
            print("No data loaded.")
            return
            
        self.engine = MultiPairBacktestEngineConsistent(
            data, self.pairs, self.params, self.config, self.strategy_class
        )
        self.trade_log = self.engine.run_backtest()
        
        self.calc_metrics()
        self.generate_deep_analysis()

    def calc_metrics(self):
        if self.trade_log.empty: return
        
        df = self.trade_log
        final = self.engine.current_balance
        profit = final - self.config['START_BALANCE']
        
        wins = df[df['Result'] > 0]
        losses = df[df['Result'] < 0]
        
        win_rate = len(wins) / len(df) if len(df) > 0 else 0
        pf = wins['Result'].sum() / abs(losses['Result'].sum()) if not losses.empty else 0
        
        eq = pd.Series(self.engine.equity_curve)
        dd = (eq - eq.cummax()) / eq.cummax()
        max_dd = dd.min()
        
        self.metrics = pd.DataFrame({
            'Metric': ['Final Balance', 'Net Profit', 'Total Trades', 'Win Rate', 'Profit Factor', 'Max DD'],
            'Value': [final, profit, len(df), win_rate, pf, max_dd]
        })
        
        # --- RESTORED PRINT ---
        print("\n=== GENERAL METRICS ===")
        print(self.metrics.to_markdown())

    def generate_deep_analysis(self):
        if self.trade_log.empty: return
        
        print("\n" + "="*50)
        print("🔍 GENERATING DEEP ANALYSIS REPORT")
        print("="*50)
        
        trades = self.trade_log.copy()
        
        wins = trades[trades['Result'] > 0]
        losses = trades[trades['Result'] <= 0]
        
        avg_win = wins['Result'].mean() if not wins.empty else 0
        avg_loss = losses['Result'].mean() if not losses.empty else 0
        rr_real = abs(avg_win/avg_loss) if avg_loss != 0 else 0
        expectancy = (len(wins)/len(trades) * avg_win) + (len(losses)/len(trades) * avg_loss)
        
        trades['Win'] = np.where(trades['Result'] > 0, 1, 0)
        groups = trades['Win'].ne(trades['Win'].shift()).cumsum()
        streaks = trades.groupby(groups)['Win'].agg(['count', 'first'])
        
        max_win_streak = streaks[streaks['first'] == 1]['count'].max() if not streaks.empty else 0
        max_loss_streak = streaks[streaks['first'] == 0]['count'].max() if not streaks.empty else 0
        
        print(f"\n📊 ADVANCED STATISTICS:")
        print(f"💰 Avg Win:             ${avg_win:.2f}")
        print(f"💸 Avg Loss:            ${avg_loss:.2f}")
        print(f"⚖️  Real R:R Ratio:      {rr_real:.2f}")
        print(f"🔮 Expectancy:          ${expectancy:.2f} per trade")
        print(f"❄️ Max Loss Streak:     {max_loss_streak} trades")
        print(f"🔥 Max Win Streak:      {max_win_streak} trades")

        self._plot_trade_examples(trades)

    def _plot_monthly_heatmap(self, trades):
        pass # Disabled per request

    def _plot_trade_examples(self, trades, n=3):
        sorted_trades = trades.sort_values('Result', ascending=False)
        best = sorted_trades.head(n)
        worst = sorted_trades.tail(n)
        
        selection = pd.concat([best, worst])
        selection = selection[~selection.index.duplicated(keep='first')]
        
        print(f"\n🔍 VISUALIZING {len(selection)} TRADE EXAMPLES (Output Hidden for Experiment Mode)")
        # Trade plotting logic hidden for clean output as requested