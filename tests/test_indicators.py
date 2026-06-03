import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.indicators import TechnicalIndicators as TI


def make_prices(n=100, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    prices = pd.Series(100 * np.cumprod(1 + returns), name="Close")
    return prices


def test_sma():
    prices = make_prices()
    sma = TI.sma(prices, 20)
    assert sma.iloc[19] == prices.iloc[:20].mean()
    assert pd.isna(sma.iloc[18])


def test_rsi_range():
    prices = make_prices()
    rsi = TI.rsi(prices, 14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_bollinger_bands():
    prices = make_prices()
    bb = TI.bollinger_bands(prices, 20)
    valid = bb.dropna()
    assert (valid["upper"] >= valid["middle"]).all()
    assert (valid["middle"] >= valid["lower"]).all()


def test_macd():
    prices = make_prices(200)
    macd = TI.macd(prices).dropna()
    assert "macd" in macd.columns
    assert "signal" in macd.columns
    assert "histogram" in macd.columns


if __name__ == "__main__":
    test_sma()
    test_rsi_range()
    test_bollinger_bands()
    test_macd()
    print("모든 테스트 통과!")
