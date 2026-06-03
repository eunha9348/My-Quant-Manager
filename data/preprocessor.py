import numpy as np
import pandas as pd


class Preprocessor:
    @staticmethod
    def returns(prices: pd.DataFrame | pd.Series, method: str = "log") -> pd.DataFrame | pd.Series:
        if method == "log":
            return np.log(prices / prices.shift(1)).dropna()
        return prices.pct_change().dropna()

    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.mean()) / df.std()

    @staticmethod
    def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std

    @staticmethod
    def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
        z = np.abs((df - df.mean()) / df.std())
        return df.where(z < z_thresh)

    @staticmethod
    def align(*dfs: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
        combined = pd.concat(dfs, axis=1, join="inner")
        n = len(dfs)
        cols_per = combined.shape[1] // n
        return tuple(combined.iloc[:, i * cols_per:(i + 1) * cols_per] for i in range(n))
