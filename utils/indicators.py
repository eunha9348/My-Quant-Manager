import numpy as np
import pandas as pd


class TechnicalIndicators:
    """순수 pandas/numpy로 구현한 기술적 지표 모음."""

    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        return prices.rolling(window).mean()

    @staticmethod
    def ema(prices: pd.Series, span: int) -> pd.Series:
        return prices.ewm(span=span, adjust=False).mean()

    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        mean = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        return pd.DataFrame({
            "upper": mean + num_std * std,
            "middle": mean,
            "lower": mean - num_std * std,
            "bandwidth": (2 * num_std * std) / mean,
        })

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        })

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(d_period).mean()
        return pd.DataFrame({"K": k, "D": d})

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical = (high + low + close) / 3
        return (typical * volume).cumsum() / volume.cumsum()
