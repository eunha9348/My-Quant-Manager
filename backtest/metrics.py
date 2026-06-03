import numpy as np
import pandas as pd


class PerformanceMetrics:
    def __init__(self, equity_curve: pd.Series, returns: pd.Series, risk_free: float = 0.03):
        self.equity = equity_curve
        self.returns = returns.dropna()
        self.risk_free = risk_free

    @property
    def total_return(self) -> float:
        return (self.equity.iloc[-1] / self.equity.iloc[0]) - 1

    @property
    def annualized_return(self) -> float:
        n_years = len(self.returns) / 252
        return (1 + self.total_return) ** (1 / n_years) - 1

    @property
    def annualized_volatility(self) -> float:
        return self.returns.std() * np.sqrt(252)

    @property
    def sharpe_ratio(self) -> float:
        excess = self.annualized_return - self.risk_free
        return excess / self.annualized_volatility if self.annualized_volatility != 0 else 0

    @property
    def max_drawdown(self) -> float:
        rolling_max = self.equity.cummax()
        drawdown = (self.equity - rolling_max) / rolling_max
        return drawdown.min()

    @property
    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown)
        return self.annualized_return / mdd if mdd != 0 else 0

    @property
    def win_rate(self) -> float:
        return (self.returns > 0).sum() / len(self.returns)

    @property
    def profit_factor(self) -> float:
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return gains / losses if losses != 0 else float("inf")

    def summary(self) -> dict:
        return {
            "총 수익률": f"{self.total_return:.2%}",
            "연간 수익률": f"{self.annualized_return:.2%}",
            "연간 변동성": f"{self.annualized_volatility:.2%}",
            "샤프 비율": f"{self.sharpe_ratio:.2f}",
            "최대 낙폭(MDD)": f"{self.max_drawdown:.2%}",
            "칼마 비율": f"{self.calmar_ratio:.2f}",
            "승률": f"{self.win_rate:.2%}",
            "손익비": f"{self.profit_factor:.2f}",
        }

    def print_summary(self) -> None:
        print(f"\n{'='*40}")
        print(f"{'성과 요약':^40}")
        print(f"{'='*40}")
        for k, v in self.summary().items():
            print(f"  {k:<18}: {v:>10}")
        print(f"{'='*40}\n")
