"""
전략-포트폴리오 융합 엔진.

여러 (전략, 종목) 조합을 독립적으로 백테스트한 뒤,
각 전략의 수익률 시계열을 하나의 '자산'으로 취급하여
포트폴리오 최적화로 최종 비중을 결정합니다.
"""
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from strategies.base import BaseStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from portfolio.optimizer import PortfolioOptimizer


class StrategyPortfolio:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        optimization: str = "max_sharpe",  # max_sharpe | min_vol | risk_parity | equal_weight
        lookback_days: int = 252,           # 최적화에 사용할 과거 데이터 길이
        rebalance_freq: str = "monthly",    # daily | weekly | monthly | quarterly
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.optimization = optimization
        self.lookback_days = lookback_days
        self.rebalance_freq = rebalance_freq
        self.engine = BacktestEngine(config_path)
        self.optimizer = PortfolioOptimizer(max_weight=cfg["portfolio"]["max_position_size"])

    # ── 내부 유틸 ────────────────────────────────────────────────────
    def _resample_rule(self) -> str:
        return {"daily": "D", "weekly": "W-FRI", "monthly": "ME", "quarterly": "QE"}[self.rebalance_freq]

    def _get_weights(self, strategy_returns: pd.DataFrame) -> dict[str, float]:
        n = len(strategy_returns.columns)
        if self.optimization == "equal_weight":
            return {c: 1 / n for c in strategy_returns.columns}
        if self.optimization == "risk_parity":
            return self.optimizer.equal_risk_contribution(strategy_returns)
        if self.optimization == "min_vol":
            try:
                return self.optimizer.min_volatility(strategy_returns)
            except Exception:
                return self.optimizer.equal_risk_contribution(strategy_returns)
        # default: max_sharpe
        try:
            return self.optimizer.max_sharpe(strategy_returns)
        except Exception:
            return self.optimizer.equal_risk_contribution(strategy_returns)

    # ── 메인 실행 ────────────────────────────────────────────────────
    def run(
        self,
        specs: list[tuple[BaseStrategy, pd.Series]],
    ) -> "StrategyPortfolioResult":
        """
        Args:
            specs: [(전략, 가격시리즈), ...] 리스트
        """
        # 1. 각 전략 독립 백테스트 → 수익률 시리즈
        equity_curves: dict[str, pd.Series] = {}
        for strategy, prices in specs:
            label = f"{strategy.name}_{prices.name or 'asset'}"
            metrics = self.engine.run(strategy, prices)
            equity_curves[label] = metrics.equity

        # 2. 수익률 DataFrame 정렬
        ret_df = pd.DataFrame({k: v.pct_change() for k, v in equity_curves.items()}).dropna()

        # 3. 리밸런싱 날짜 결정
        rebal_dates = ret_df.resample(self._resample_rule()).last().index

        # 4. 시뮬레이션
        capital = self.cfg["backtest"]["initial_capital"]
        equity_list = []
        weights: dict[str, float] = {k: 1 / len(equity_curves) for k in equity_curves}

        for date, row in ret_df.iterrows():
            if date in rebal_dates:
                start = ret_df.index.get_loc(date)
                hist_start = max(0, start - self.lookback_days)
                hist = ret_df.iloc[hist_start:start]
                if len(hist) >= 20:
                    weights = self._get_weights(hist)

            daily_ret = sum(weights.get(k, 0) * row[k] for k in ret_df.columns)
            capital *= (1 + daily_ret)
            equity_list.append(capital)

        combined_equity = pd.Series(equity_list, index=ret_df.index)
        combined_returns = combined_equity.pct_change().dropna()

        return StrategyPortfolioResult(
            equity=combined_equity,
            returns=combined_returns,
            equity_curves=equity_curves,
            final_weights=weights,
            optimization=self.optimization,
        )


class StrategyPortfolioResult:
    def __init__(
        self,
        equity: pd.Series,
        returns: pd.Series,
        equity_curves: dict[str, pd.Series],
        final_weights: dict[str, float],
        optimization: str,
    ):
        self.equity = equity
        self.returns = returns
        self.equity_curves = equity_curves
        self.final_weights = final_weights
        self.optimization = optimization
        self._metrics = PerformanceMetrics(equity, returns)

    # PerformanceMetrics 위임
    def print_summary(self) -> None:
        print(f"\n{'='*45}")
        print(f"  전략 포트폴리오 결과 [{self.optimization}]")
        print(f"{'='*45}")
        self._metrics.print_summary()
        print("최종 전략 비중:")
        for k, w in sorted(self.final_weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 30)
            print(f"  {k:<35} {w:>5.1%}  {bar}")

    def plot(self, title: str = "Strategy Portfolio") -> None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                                 gridspec_kw={"height_ratios": [3, 2, 1]})

        # 통합 자산 곡선
        axes[0].plot(self.equity, color="navy", linewidth=2, label="Combined Portfolio")
        axes[0].set_title(title)
        axes[0].set_ylabel("Portfolio Value (KRW)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # 개별 전략 정규화
        for name, curve in self.equity_curves.items():
            norm = curve / curve.iloc[0]
            axes[1].plot(norm, alpha=0.7, label=name[:25])
        axes[1].set_title("개별 전략 정규화 수익")
        axes[1].set_ylabel("Normalized Return")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)

        # MDD
        rolling_max = self.equity.cummax()
        dd = (self.equity - rolling_max) / rolling_max
        axes[2].fill_between(dd.index, dd, 0, color="red", alpha=0.4, label="Drawdown")
        axes[2].set_ylabel("Drawdown")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("strategy_portfolio.png", dpi=150)
        plt.show()
        print("차트 저장: strategy_portfolio.png")
