import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from strategies import RSIStrategy, MACDStrategy, MovingAverageCrossStrategy, MultiFactorStrategy
from portfolio.strategy_portfolio import StrategyPortfolio
from portfolio.optimizer import PortfolioOptimizer


def make_prices(n=500, seed=42, name="TEST"):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.02, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(100 * np.cumprod(1 + r), index=idx, name=name)


def test_portfolio_optimizer():
    prices = pd.DataFrame({
        "A": make_prices(seed=1).values,
        "B": make_prices(seed=2).values,
        "C": make_prices(seed=3).values,
    })
    returns = prices.pct_change().dropna()
    opt = PortfolioOptimizer(max_weight=0.5)

    w_rp = opt.equal_risk_contribution(returns)
    assert abs(sum(w_rp.values()) - 1.0) < 0.01

    w_mv = opt.min_volatility(returns)
    assert abs(sum(w_mv.values()) - 1.0) < 0.01


def test_strategy_portfolio_equal_weight():
    prices = make_prices(n=500, name="005930.KS")
    specs = [
        (RSIStrategy(), prices),
        (MACDStrategy(), prices),
        (MovingAverageCrossStrategy(), prices),
    ]
    sp = StrategyPortfolio(optimization="equal_weight", lookback_days=60, rebalance_freq="monthly")
    result = sp.run(specs)

    assert len(result.equity) > 0
    assert result.equity.iloc[-1] > 0
    assert abs(sum(result.final_weights.values()) - 1.0) < 0.05


def test_strategy_portfolio_risk_parity():
    prices = make_prices(n=500, name="005930.KS")
    specs = [
        (RSIStrategy(), prices),
        (MultiFactorStrategy(), prices),
    ]
    sp = StrategyPortfolio(optimization="risk_parity", lookback_days=60, rebalance_freq="monthly")
    result = sp.run(specs)
    assert result.equity.iloc[-1] > 0


if __name__ == "__main__":
    test_portfolio_optimizer()
    test_strategy_portfolio_equal_weight()
    test_strategy_portfolio_risk_parity()
    print("전략 포트폴리오 테스트 모두 통과!")
