import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml


class DataFetcher:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg["data"]
        self.cache_dir = Path(self.cfg["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        tickers: list[str] | str,
        start: str | None = None,
        end: str | None = None,
        interval: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        start = start or self.cfg["default_start"]
        end = end or datetime.today().strftime("%Y-%m-%d")
        interval = interval or self.cfg["interval"]

        cache_key = f"{'_'.join(sorted(tickers))}_{start}_{end}_{interval}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if use_cache and cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(hours=24):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True)

        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product([data.columns, tickers])

        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

        return data

    def fetch_close(
        self,
        tickers: list[str] | str,
        start: str | None = None,
        end: str | None = None,
        interval: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        data = self.fetch(tickers, start, end, interval, use_cache)
        return data["Close"]

    def fetch_ohlcv(self, ticker: str, **kwargs) -> pd.DataFrame:
        data = self.fetch([ticker], **kwargs)
        ohlcv = data.xs(ticker, axis=1, level=1)
        return ohlcv
