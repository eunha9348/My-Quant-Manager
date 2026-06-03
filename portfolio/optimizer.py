import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt.risk_models import CovarianceShrinkage


class PortfolioOptimizer:
    """
    PyPortfolioOpt 기반 포트폴리오 최적화.
    주어진 수익률 DataFrame으로 최적 비중을 계산합니다.
    """

    def __init__(self, max_weight: float = 0.4, min_weight: float = 0.0):
        self.max_weight = max_weight
        self.min_weight = min_weight

    def _build_ef(self, returns: pd.DataFrame) -> EfficientFrontier:
        mu = expected_returns.mean_historical_return(returns, returns_data=True, frequency=252)
        S = CovarianceShrinkage(returns, returns_data=True, frequency=252).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(self.min_weight, self.max_weight))
        return ef

    def max_sharpe(self, returns: pd.DataFrame, risk_free: float = 0.03) -> dict[str, float]:
        ef = self._build_ef(returns)
        ef.max_sharpe(risk_free_rate=risk_free)
        return dict(ef.clean_weights())

    def min_volatility(self, returns: pd.DataFrame) -> dict[str, float]:
        ef = self._build_ef(returns)
        ef.min_volatility()
        return dict(ef.clean_weights())

    def max_quadratic_utility(self, returns: pd.DataFrame, risk_aversion: float = 2.0) -> dict[str, float]:
        ef = self._build_ef(returns)
        ef.max_quadratic_utility(risk_aversion=risk_aversion)
        return dict(ef.clean_weights())

    def equal_risk_contribution(self, returns: pd.DataFrame) -> dict[str, float]:
        """리스크 패리티 (동일 리스크 기여도)."""
        cov = returns.cov() * 252
        vol = np.sqrt(np.diag(cov.values))
        vol = np.where(vol < 1e-8, 1e-8, vol)  # 0 방지
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights /= weights.sum()
        return dict(zip(returns.columns, weights))

    def print_weights(self, weights: dict[str, float], method: str = "") -> None:
        title = f"포트폴리오 비중 ({method})" if method else "포트폴리오 비중"
        print(f"\n{title}")
        print("-" * 40)
        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 30)
            print(f"  {name:<30} {w:>6.1%}  {bar}")
        print()
