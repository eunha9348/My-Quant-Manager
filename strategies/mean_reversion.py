import numpy as np
import pandas as pd

from .base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """볼린저 밴드 기반 평균 회귀 전략. 하단 밴드 터치 시 매수, 상단 밴드 터치 시 매도."""

    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__()
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame | pd.Series) -> pd.Series:
        prices = data["Close"] if isinstance(data, pd.DataFrame) else data

        rolling_mean = prices.rolling(self.window).mean()
        rolling_std = prices.rolling(self.window).std()

        upper_band = rolling_mean + self.num_std * rolling_std
        lower_band = rolling_mean - self.num_std * rolling_std

        signals = pd.Series(0, index=prices.index)
        signals[prices < lower_band] = 1    # 하단 밴드 이탈 → 매수
        signals[prices > upper_band] = -1   # 상단 밴드 이탈 → 매도

        return signals

    def get_bands(self, prices: pd.Series) -> pd.DataFrame:
        mean = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std()
        return pd.DataFrame({
            "mean": mean,
            "upper": mean + self.num_std * std,
            "lower": mean - self.num_std * std,
        })

    def _params_str(self) -> str:
        return f"window={self.window}, std={self.num_std}"
