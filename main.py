"""
My-Quant-Manager 빠른 시작 예제.
삼성전자(005930.KS) 데이터를 받아 이동평균 전략을 백테스트합니다.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import DataFetcher
from strategies import MovingAverageCrossStrategy, MeanReversionStrategy, MomentumStrategy
from backtest import BacktestEngine
from utils import TechnicalIndicators as TI


def main():
    print("데이터 다운로드 중...")
    fetcher = DataFetcher()
    prices = fetcher.fetch_close(["005930.KS"], start="2021-01-01")["005930.KS"]
    prices.name = "Close"
    print(f"  기간: {prices.index[0].date()} ~ {prices.index[-1].date()} ({len(prices)}일)")
    print(f"  현재가: {prices.iloc[-1]:,.0f}원\n")

    engine = BacktestEngine()

    strategies = {
        "이동평균 교차 (20/60)": MovingAverageCrossStrategy(20, 60),
        "볼린저 밴드 평균회귀": MeanReversionStrategy(20, 2.0),
        "모멘텀 (60일)": MomentumStrategy(60),
    }

    results = {}
    for name, strategy in strategies.items():
        print(f"[{name}] 백테스트 실행...")
        metrics = engine.run(strategy, prices)
        metrics.print_summary()
        results[name] = metrics

    print("RSI(14) 현재값:", f"{TI.rsi(prices, 14).iloc[-1]:.1f}")


if __name__ == "__main__":
    main()
