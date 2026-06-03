# My-Quant-Manager

개인 퀀트 트레이딩 전략 개발 및 백테스트 프레임워크.

## 프로젝트 구조

```
My-Quant-Manager/
├── config/config.yaml      # 초기 자본, 수수료, 유니버스 설정
├── data/                   # 데이터 수집 및 전처리
│   ├── fetcher.py          # yfinance 기반 OHLCV 수집 + 로컬 캐시
│   └── preprocessor.py     # 수익률 계산, 정규화, 이상치 제거
├── strategies/             # 트레이딩 전략
│   ├── base.py             # BaseStrategy 추상 클래스
│   ├── moving_average.py   # 이동평균 교차 (골든/데드 크로스)
│   ├── mean_reversion.py   # 볼린저 밴드 평균 회귀
│   └── momentum.py         # 모멘텀 추세 추종
├── backtest/               # 백테스트 엔진
│   ├── engine.py           # 수수료/슬리피지 적용 시뮬레이션
│   └── metrics.py          # 샤프/MDD/칼마/승률/손익비
├── portfolio/              # 포트폴리오 관리
│   └── manager.py          # 동일비중/최소분산/리스크패리티/모멘텀
├── utils/                  # 유틸리티
│   ├── indicators.py       # SMA/EMA/RSI/MACD/볼린저/ATR/스토캐스틱/VWAP
│   └── visualizer.py       # 캔들차트, 전략 비교 차트
├── notebooks/              # Jupyter 분석 노트북
├── tests/                  # 단위 테스트
└── main.py                 # 빠른 시작 예제
```

## 새 전략 추가하는 법

`strategies/base.py`의 `BaseStrategy`를 상속해서 `generate_signals()` 구현:

```python
from strategies.base import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    def generate_signals(self, data: pd.Series) -> pd.Series:
        # 1(매수), -1(매도), 0(중립) 시리즈 반환
        ...
```

## 백테스트 실행

```python
from data import DataFetcher
from backtest import BacktestEngine

fetcher = DataFetcher()
prices = fetcher.fetch_close(["005930.KS"], start="2021-01-01")["005930.KS"]

engine = BacktestEngine()
metrics = engine.run(MyStrategy(), prices)
metrics.print_summary()
```

## 설정 변경

`config/config.yaml`에서 조정:
- `initial_capital`: 초기 자본 (기본 1천만원)
- `commission`: 수수료율 (기본 0.15%)
- `slippage`: 슬리피지 (기본 0.1%)
- `universe`: 관심 종목 목록

## 개발 환경 설치

```bash
pip install -r requirements.txt
```

## 테스트 실행

```bash
python tests/test_indicators.py
python tests/test_strategies.py
```
