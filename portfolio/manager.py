import numpy as np
import pandas as pd
import yaml


class PortfolioManager:
    """여러 종목의 비중을 계산하고 리밸런싱을 관리합니다."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        pc = cfg["portfolio"]
        self.max_position = pc["max_position_size"]

    def equal_weight(self, tickers: list[str]) -> dict[str, float]:
        w = 1.0 / len(tickers)
        return {t: min(w, self.max_position) for t in tickers}

    def min_variance(self, returns: pd.DataFrame) -> dict[str, float]:
        cov = returns.cov() * 252
        n = len(returns.columns)
        # 단순 역분산 가중 (근사치)
        variances = np.diag(cov.values)
        inv_var = 1.0 / variances
        weights = inv_var / inv_var.sum()
        weights = np.clip(weights, 0, self.max_position)
        weights /= weights.sum()
        return dict(zip(returns.columns, weights))

    def risk_parity(self, returns: pd.DataFrame) -> dict[str, float]:
        cov = returns.cov() * 252
        vol = np.sqrt(np.diag(cov.values))
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
        weights = np.clip(weights, 0, self.max_position)
        weights /= weights.sum()
        return dict(zip(returns.columns, weights))

    def momentum_weight(self, returns: pd.DataFrame, lookback: int = 60) -> dict[str, float]:
        momentum = returns.iloc[-lookback:].sum()
        momentum = momentum.clip(lower=0)
        if momentum.sum() == 0:
            return self.equal_weight(list(returns.columns))
        weights = momentum / momentum.sum()
        weights = weights.clip(upper=self.max_position)
        weights /= weights.sum()
        return weights.to_dict()

    def print_weights(self, weights: dict[str, float]) -> None:
        print(f"\n{'종목':>12}  {'비중':>8}")
        print("-" * 24)
        for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 40)
            print(f"  {ticker:<12} {w:>6.1%}  {bar}")
        print()
