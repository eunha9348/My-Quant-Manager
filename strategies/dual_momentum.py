import pandas as pd
import numpy as np
from .base import BaseStrategy


class DualMomentumStrategy(BaseStrategy):
    """
    Gary Antonacci의 듀얼 모멘텀 전략.
    - 절대 모멘텀: 자산 수익률 > 0 (무위험 수익률 초과)
    - 상대 모멘텀: 여러 자산 중 모멘텀 상위 선택
    단일 종목 사용 시 절대 모멘텀만 적용.
    """

    def __init__(self, lookback: int = 252, top_n: int = 1):
        super().__init__()
        self.lookback = lookback  # 약 12개월
        self.top_n = top_n

    def generate_signals(self, data: pd.DataFrame | pd.Series) -> pd.Series:
        # 단일 종목: 절대 모멘텀만
        if isinstance(data, pd.Series):
            return self._absolute_momentum(data)

        prices = data if isinstance(data, pd.DataFrame) else data
        return self._dual_momentum(prices)

    def _absolute_momentum(self, prices: pd.Series) -> pd.Series:
        momentum = prices.pct_change(self.lookback)
        signals = pd.Series(0, index=prices.index)
        signals[momentum > 0] = 1
        signals[momentum <= 0] = -1
        return signals

    def _dual_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """여러 종목에 대한 듀얼 모멘텀 비중 반환."""
        momentum = prices.pct_change(self.lookback)
        # 절대 모멘텀 필터 (수익률 > 0) + 상대 모멘텀 (상위 top_n 선택)
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date in prices.index[self.lookback:]:
            row = momentum.loc[date]
            positive = row[row > 0]
            if len(positive) == 0:
                continue  # 전부 음수 → 현금 보유
            top = positive.nlargest(min(self.top_n, len(positive)))
            weights.loc[date, top.index] = 1.0 / len(top)
        return weights

    def _params_str(self) -> str:
        return f"lookback={self.lookback}, top_n={self.top_n}"
