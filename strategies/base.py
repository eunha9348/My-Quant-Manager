from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """모든 전략의 기본 클래스. generate_signals()를 구현해야 합니다."""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Returns:
            pd.Series: 날짜 인덱스, 값은 1(매수), -1(매도), 0(중립)
        """

    def __repr__(self) -> str:
        return f"{self.name}({self._params_str()})"

    def _params_str(self) -> str:
        return ""
