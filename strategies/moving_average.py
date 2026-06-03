import pandas as pd

from .base import BaseStrategy


class MovingAverageCrossStrategy(BaseStrategy):
    """단기 이동평균이 장기 이동평균을 상향 돌파하면 매수, 하향 돌파하면 매도."""

    def __init__(self, short_window: int = 20, long_window: int = 60):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame | pd.Series) -> pd.Series:
        prices = data["Close"] if isinstance(data, pd.DataFrame) else data

        short_ma = prices.rolling(self.short_window).mean()
        long_ma = prices.rolling(self.long_window).mean()

        signals = pd.Series(0, index=prices.index)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1

        # 신호가 바뀌는 시점만 거래 (포지션 변화)
        position = signals.diff().fillna(0)
        trades = pd.Series(0, index=prices.index)
        trades[position > 0] = 1   # 매수
        trades[position < 0] = -1  # 매도
        return trades

    def _params_str(self) -> str:
        return f"short={self.short_window}, long={self.long_window}"
