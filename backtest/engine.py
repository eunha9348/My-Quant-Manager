import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from strategies.base import BaseStrategy

from .metrics import PerformanceMetrics


class BacktestEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        bc = cfg["backtest"]
        self.initial_capital = bc["initial_capital"]
        self.commission = bc["commission"]
        self.slippage = bc["slippage"]

    def run(self, strategy: BaseStrategy, prices: pd.Series) -> PerformanceMetrics:
        signals = strategy.generate_signals(prices)
        signals, prices = signals.align(prices, join="inner")

        capital = self.initial_capital
        position = 0
        equity_curve = []

        for i, (date, price) in enumerate(prices.items()):
            signal = signals.iloc[i]

            if signal == 1 and position == 0:
                cost = price * (1 + self.slippage + self.commission)
                shares = capital // cost
                capital -= shares * cost
                position = shares

            elif signal == -1 and position > 0:
                proceeds = price * (1 - self.slippage - self.commission)
                capital += position * proceeds
                position = 0

            equity_curve.append(capital + position * price)

        equity = pd.Series(equity_curve, index=prices.index)
        returns = equity.pct_change().dropna()
        return PerformanceMetrics(equity, returns)

    def plot(self, metrics: PerformanceMetrics, title: str = "Backtest Result") -> None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

        axes[0].plot(metrics.equity, label="Equity Curve")
        axes[0].set_title(title)
        axes[0].set_ylabel("Portfolio Value (KRW)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        rolling_max = metrics.equity.cummax()
        drawdown = (metrics.equity - rolling_max) / rolling_max
        axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.5, color="red", label="Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"backtest_result.png", dpi=150)
        plt.show()
        print("차트 저장: backtest_result.png")
