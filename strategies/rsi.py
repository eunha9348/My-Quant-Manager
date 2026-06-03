import pandas as pd
import numpy as np
from .base import BaseStrategy
from utils.indicators import TechnicalIndicators as TI


class RSIStrategy(BaseStrategy):
    """RSI 과매수/과매도 기반 역추세 전략."""

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame | pd.Series) -> pd.Series:
        prices = data["Close"] if isinstance(data, pd.DataFrame) else data
        rsi = TI.rsi(prices, self.period)

        prev_rsi = rsi.shift(1)
        signals = pd.Series(0, index=prices.index)
        signals[(prev_rsi >= self.oversold) & (rsi < self.oversold)] = 1   # 과매도 진입
        signals[(prev_rsi <= self.overbought) & (rsi > self.overbought)] = -1  # 과매수 진입
        return signals

    def _params_str(self) -> str:
        return f"period={self.period}, oversold={self.oversold}, overbought={self.overbought}"
