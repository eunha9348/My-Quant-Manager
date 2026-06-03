import pandas as pd
import numpy as np
from .base import BaseStrategy
from utils.indicators import TechnicalIndicators as TI


class MultiFactorStrategy(BaseStrategy):
    """
    RSI + MACD + 이동평균 3가지 신호를 투표 방식으로 결합.
    2개 이상 동의하면 포지션 진입.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 40.0,
        rsi_overbought: float = 60.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        ma_short: int = 20,
        ma_long: int = 60,
        threshold: int = 2,  # 최소 동의 신호 수
    ):
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame | pd.Series) -> pd.Series:
        prices = data["Close"] if isinstance(data, pd.DataFrame) else data

        # RSI 신호
        rsi = TI.rsi(prices, self.rsi_period)
        rsi_signal = pd.Series(0, index=prices.index)
        rsi_signal[rsi < self.rsi_oversold] = 1
        rsi_signal[rsi > self.rsi_overbought] = -1

        # MACD 신호
        macd = TI.macd(prices, self.macd_fast, self.macd_slow, self.macd_signal)
        macd_signal = pd.Series(0, index=prices.index)
        macd_signal[macd["histogram"] > 0] = 1
        macd_signal[macd["histogram"] < 0] = -1

        # 이동평균 신호
        short_ma = TI.sma(prices, self.ma_short)
        long_ma = TI.sma(prices, self.ma_long)
        ma_signal = pd.Series(0, index=prices.index)
        ma_signal[short_ma > long_ma] = 1
        ma_signal[short_ma < long_ma] = -1

        # 투표
        vote = rsi_signal + macd_signal + ma_signal
        signals = pd.Series(0, index=prices.index)
        signals[vote >= self.threshold] = 1
        signals[vote <= -self.threshold] = -1
        return signals

    def _params_str(self) -> str:
        return f"rsi={self.rsi_period}, ma={self.ma_short}/{self.ma_long}, threshold={self.threshold}"
