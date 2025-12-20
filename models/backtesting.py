"""
Backtesting engine for NBA prop betting strategies.

Contains BetResult dataclass and Backtester class for historical performance simulation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd

from core.odds_utils import american_to_decimal


@dataclass
class BetResult:
    """Individual bet outcome."""
    date: datetime
    player: str
    prop_type: str
    side: str
    line: float
    odds: int
    projection: float
    edge: float
    actual: float
    won: bool
    profit: float
    units: float


class Backtester:
    """
    Backtesting engine for prop betting strategies.
    Tracks performance metrics and manages simulated bankroll.
    """

    def __init__(self, initial_bankroll: float = 1000, unit_size: float = 10):
        self.initial_bankroll = initial_bankroll
        self.unit_size = unit_size
        self.results: List[BetResult] = []
        self.bankroll_history = [initial_bankroll]

    def run_backtest(self, props_df: pd.DataFrame, game_logs: pd.DataFrame,
                     model, min_edge: float = 0.03,
                     min_confidence: float = 0.4) -> pd.DataFrame:
        """
        Run backtest on historical prop data.

        Args:
            props_df: DataFrame with prop lines and results
            game_logs: DataFrame with player game logs
            model: Prediction model to use
            min_edge: Minimum edge required to bet
            min_confidence: Minimum confidence to bet
        """

        self.results = []
        bankroll = self.initial_bankroll
        self.bankroll_history = [bankroll]

        # Sort by date
        props_df = props_df.sort_values('date')

        for _, prop in props_df.iterrows():
            # Get player history before this game
            player_history = game_logs[
                (game_logs['player'] == prop['player']) &
                (game_logs['date'] < prop['date'])
            ].sort_values('date')

            if len(player_history) < 10:
                continue

            # Get prediction
            if hasattr(model, 'predict') and model.name == 'Situational':
                game_context = {
                    'home': prop['home'],
                    'b2b': prop['b2b'],
                    'opp_def_rtg': prop['opp_def_rtg']
                }
                pred = model.predict(player_history, game_context,
                                    prop['prop_type'], prop['line'])
            else:
                pred = model.predict(player_history[prop['prop_type']], prop['line'])

            # Check if we should bet
            if pred.recommended_side == 'pass':
                continue
            if abs(pred.edge) < min_edge:
                continue
            if pred.confidence < min_confidence:
                continue

            # Place bet
            side = pred.recommended_side
            odds = prop['over_odds'] if side == 'over' else prop['under_odds']

            # Determine units based on edge (Kelly-lite)
            units = min(3, max(1, abs(pred.edge) * 20))
            stake = units * self.unit_size

            if stake > bankroll:
                continue  # Can't afford this bet

            # Determine outcome
            actual = prop['actual']
            if side == 'over':
                won = actual > prop['line']
            else:
                won = actual < prop['line']

            push = actual == prop['line']

            if push:
                profit = 0
            elif won:
                decimal_odds = american_to_decimal(odds)
                profit = stake * (decimal_odds - 1)
            else:
                profit = -stake

            bankroll += profit
            self.bankroll_history.append(bankroll)

            result = BetResult(
                date=prop['date'],
                player=prop['player'],
                prop_type=prop['prop_type'],
                side=side,
                line=prop['line'],
                odds=odds,
                projection=pred.projection,
                edge=pred.edge,
                actual=actual,
                won=won,
                profit=profit,
                units=units
            )
            self.results.append(result)

        return self._generate_results_df()

    def _generate_results_df(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'date': r.date,
                'player': r.player,
                'prop_type': r.prop_type,
                'side': r.side,
                'line': r.line,
                'odds': r.odds,
                'projection': r.projection,
                'edge': r.edge,
                'actual': r.actual,
                'won': r.won,
                'profit': r.profit,
                'units': r.units
            }
            for r in self.results
        ])

    def get_metrics(self) -> dict:
        """Calculate comprehensive performance metrics."""
        if not self.results:
            return {'error': 'No results to analyze'}

        results_df = self._generate_results_df()

        total_bets = len(results_df)
        wins = results_df['won'].sum()
        losses = total_bets - wins

        win_rate = wins / total_bets if total_bets > 0 else 0

        total_profit = results_df['profit'].sum()
        total_wagered = (results_df['units'] * self.unit_size).sum()

        roi = total_profit / total_wagered if total_wagered > 0 else 0

        # Calculate max drawdown
        peak = self.initial_bankroll
        max_drawdown = 0
        for bal in self.bankroll_history:
            if bal > peak:
                peak = bal
            drawdown = (peak - bal) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Profit by prop type
        profit_by_prop = results_df.groupby('prop_type')['profit'].sum().to_dict()
        winrate_by_prop = results_df.groupby('prop_type')['won'].mean().to_dict()

        # Profit by side
        profit_by_side = results_df.groupby('side')['profit'].sum().to_dict()

        # Streaks
        results_df['streak'] = (results_df['won'] != results_df['won'].shift()).cumsum()
        win_streaks = results_df[results_df['won']].groupby('streak').size()
        loss_streaks = results_df[~results_df['won']].groupby('streak').size()

        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 4),
            'total_profit': round(total_profit, 2),
            'total_wagered': round(total_wagered, 2),
            'roi': round(roi, 4),
            'final_bankroll': round(self.bankroll_history[-1], 2),
            'max_drawdown': round(max_drawdown, 4),
            'profit_by_prop': profit_by_prop,
            'winrate_by_prop': winrate_by_prop,
            'profit_by_side': profit_by_side,
            'longest_win_streak': win_streaks.max() if len(win_streaks) > 0 else 0,
            'longest_loss_streak': loss_streaks.max() if len(loss_streaks) > 0 else 0,
            'avg_odds': round(results_df['odds'].mean(), 1),
            'avg_edge': round(results_df['edge'].mean(), 4)
        }

    def print_report(self):
        """Print formatted backtest report."""
        metrics = self.get_metrics()

        if 'error' in metrics:
            print(metrics['error'])
            return

        print("\n" + "="*60)
        print("           NBA PROP BETTING BACKTEST REPORT")
        print("="*60)

        print(f"\n  OVERALL PERFORMANCE")
        print("-"*40)
        print(f"  Total Bets:        {metrics['total_bets']:,}")
        print(f"  Record:            {metrics['wins']}-{metrics['losses']}")
        print(f"  Win Rate:          {metrics['win_rate']*100:.1f}%")
        print(f"  ROI:               {metrics['roi']*100:+.2f}%")
        print(f"  Total Profit:      ${metrics['total_profit']:+,.2f}")
        print(f"  Total Wagered:     ${metrics['total_wagered']:,.2f}")

        print(f"\n  BANKROLL")
        print("-"*40)
        print(f"  Starting:          ${self.initial_bankroll:,.2f}")
        print(f"  Ending:            ${metrics['final_bankroll']:,.2f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']*100:.1f}%")

        print(f"\n  BY PROP TYPE")
        print("-"*40)
        for prop, profit in metrics['profit_by_prop'].items():
            wr = metrics['winrate_by_prop'].get(prop, 0)
            print(f"  {prop.upper():12} ${profit:+8.2f}  ({wr*100:.1f}% win)")

        print(f"\n  BY SIDE")
        print("-"*40)
        for side, profit in metrics['profit_by_side'].items():
            print(f"  {side.upper():12} ${profit:+8.2f}")

        print(f"\n  STREAKS")
        print("-"*40)
        print(f"  Best Win Streak:   {metrics['longest_win_streak']}")
        print(f"  Worst Loss Streak: {metrics['longest_loss_streak']}")

        print(f"\n  BET CHARACTERISTICS")
        print("-"*40)
        print(f"  Avg Odds:          {metrics['avg_odds']:.0f}")
        print(f"  Avg Edge:          {metrics['avg_edge']*100:.2f}%")

        print("\n" + "="*60 + "\n")
