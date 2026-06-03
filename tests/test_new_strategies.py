import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from strategies import RSIStrategy, MACDStrategy, DualMomentumStrategy, MultiFactorStrategy


def make_prices(n=300, seed=42, name="TEST"):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.02, n)
    s = pd.Series(100 * np.cumprod(1 + r), name=name)
    return s


def test_rsi_strategy():
    prices = make_prices()
    strategy = RSIStrategy(period=14, oversold=30, overbought=70)
    signals = strategy.generate_signals(prices)
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert len(signals) == len(prices)


def test_macd_strategy():
    prices = make_prices()
    strategy = MACDStrategy()
    signals = strategy.generate_signals(prices)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_dual_momentum_single():
    prices = make_prices()
    strategy = DualMomentumStrategy(lookback=60)
    signals = strategy.generate_signals(prices)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_dual_momentum_multi():
    p1 = make_prices(name="A")
    p2 = make_prices(seed=99, name="B")
    prices_df = pd.DataFrame({"A": p1.values, "B": p2.values})
    strategy = DualMomentumStrategy(lookback=60, top_n=1)
    weights = strategy.generate_signals(prices_df)
    assert isinstance(weights, pd.DataFrame)
    # 비중 합은 0 또는 1
    row_sums = weights.sum(axis=1)
    assert ((row_sums >= -0.01) & (row_sums <= 1.01)).all()


def test_multi_factor():
    prices = make_prices(n=300)
    strategy = MultiFactorStrategy(threshold=2)
    signals = strategy.generate_signals(prices)
    assert set(signals.unique()).issubset({-1, 0, 1})


if __name__ == "__main__":
    test_rsi_strategy()
    test_macd_strategy()
    test_dual_momentum_single()
    test_dual_momentum_multi()
    test_multi_factor()
    print("새 전략 테스트 모두 통과!")
