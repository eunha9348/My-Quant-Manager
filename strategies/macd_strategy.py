import pandas as pd
from .base import BaseStrategy
from utils.indicators import TechnicalIndicators as TI


class MACDStrategy(BaseStrategy):
    """MACD 골든/데드 크로스 추세 추종 전략."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__()
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data: pd.DataFrame | pd.Series) -> pd.Series:
        prices = data["Close"] if isinstance(data, pd.DataFrame) else data
        macd = TI.macd(prices, self.fast, self.slow, self.signal)

        hist = macd["histogram"]
        prev_hist = hist.shift(1)

        signals = pd.Series(0, index=prices.index)
        signals[(prev_hist <= 0) & (hist > 0)] = 1   # 히스토그램 양전환 → 매수
        signals[(prev_hist >= 0) & (hist < 0)] = -1  # 히스토그램 음전환 → 매도
        return signals

    def _params_str(self) -> str:
        return f"fast={self.fast}, slow={self.slow}, signal={self.signal}"
