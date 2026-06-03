import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import MovingAverageCrossStrategy, MeanReversionStrategy, MomentumStrategy


def make_prices(n=300, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    return pd.Series(100 * np.cumprod(1 + returns), name="Close")


def test_ma_signals():
    prices = make_prices()
    strategy = MovingAverageCrossStrategy(short_window=20, long_window=60)
    signals = strategy.generate_signals(prices)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_mean_reversion_signals():
    prices = make_prices()
    strategy = MeanReversionStrategy(window=20, num_std=2.0)
    signals = strategy.generate_signals(prices)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_momentum_signals():
    prices = make_prices()
    strategy = MomentumStrategy(lookback=60)
    signals = strategy.generate_signals(prices)
    assert len(signals) == len(prices)


if __name__ == "__main__":
    test_ma_signals()
    test_mean_reversion_signals()
    test_momentum_signals()
    print("모든 전략 테스트 통과!")
