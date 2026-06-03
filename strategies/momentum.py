import pandas as pd

from .base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """과거 N일 수익률이 양수면 매수, 음수면 매도하는 추세 추종 전략."""

    def __init__(self, lookback: int = 60, holding: int = 20):
        super().__init__()
        self.lookback = lookback
        self.holding = holding

    def generate_signals(self, data: pd.DataFrame | pd.Series) -> pd.Series:
        prices = data["Close"] if isinstance(data, pd.DataFrame) else data

        momentum = prices.pct_change(self.lookback)

        signals = pd.Series(0, index=prices.index)
        signals[momentum > 0] = 1
        signals[momentum < 0] = -1

        # holding 기간만큼 포지션 유지 (0을 NaN으로 바꿔 ffill 후 복원)
        signals = signals.replace(0, float("nan")).ffill(limit=self.holding).fillna(0).astype(int)
        return signals

    def _params_str(self) -> str:
        return f"lookback={self.lookback}, holding={self.holding}"
