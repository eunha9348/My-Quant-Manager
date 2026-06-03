"""
S&P 500 기반 B+C 융합 전략 + Risk Management (weight=1.25)

- 10개 S&P 500 대표 종목 (섹터 분산)
- 4개 전략: Momentum, MultiFactor, MACD, MeanReversion
- 포트폴리오 최적화: max_sharpe
- Risk Management: Sharpe 기반 1.25x 레버리지 오버레이
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from strategies import MomentumStrategy, MultiFactorStrategy, MACDStrategy, MeanReversionStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from portfolio.optimizer import PortfolioOptimizer

# ════════════════════════════════════════════════════════
# 1. S&P 500 대표 종목 합성 데이터 생성
#    실제 2020~2024 연평균 수익률·변동성·섹터 상관관계 기반
# ════════════════════════════════════════════════════════
TICKERS = {
    "AAPL":  ("Tech",        0.00045, 0.019),
    "MSFT":  ("Tech",        0.00048, 0.017),
    "NVDA":  ("Tech",        0.00080, 0.032),
    "GOOGL": ("Tech",        0.00040, 0.020),
    "META":  ("Tech",        0.00055, 0.028),
    "JPM":   ("Finance",     0.00030, 0.018),
    "BAC":   ("Finance",     0.00025, 0.020),
    "JNJ":   ("Healthcare",  0.00018, 0.012),
    "XOM":   ("Energy",      0.00028, 0.022),
    "AMZN":  ("Consumer",    0.00050, 0.023),
}

# 섹터 상관행렬 (현실적 추정값)
SECTOR_CORR = {
    ("Tech",       "Tech"):       0.75,
    ("Finance",    "Finance"):    0.70,
    ("Healthcare", "Healthcare"): 0.60,
    ("Energy",     "Energy"):     0.65,
    ("Consumer",   "Consumer"):   0.60,
    ("Tech",       "Finance"):    0.45,
    ("Tech",       "Healthcare"): 0.30,
    ("Tech",       "Energy"):     0.25,
    ("Tech",       "Consumer"):   0.55,
    ("Finance",    "Healthcare"): 0.35,
    ("Finance",    "Energy"):     0.40,
    ("Finance",    "Consumer"):   0.40,
    ("Healthcare", "Energy"):     0.20,
    ("Healthcare", "Consumer"):   0.30,
    ("Energy",     "Consumer"):   0.30,
}

def build_corr_matrix(tickers, sector_corr):
    names = list(tickers.keys())
    sectors = [tickers[t][0] for t in names]
    n = len(names)
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = sectors[i], sectors[j]
            key = (s1, s2) if (s1, s2) in sector_corr else (s2, s1)
            corr = sector_corr.get(key, 0.30)
            C[i, j] = C[j, i] = corr
    return C

def simulate_sp500(n_days=1000, seed=42):
    names  = list(TICKERS.keys())
    drifts = np.array([v[1] for v in TICKERS.values()])
    vols   = np.array([v[2] for v in TICKERS.values()])

    C = build_corr_matrix(TICKERS, SECTOR_CORR)
    L = np.linalg.cholesky(C)

    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal((n_days, len(names)))
    corr_Z = Z @ L.T
    daily_ret = drifts + vols * corr_Z

    dates  = pd.date_range("2020-01-02", periods=n_days, freq="B")
    prices = pd.DataFrame(
        100 * np.cumprod(1 + daily_ret, axis=0),
        columns=names, index=dates
    )
    return prices

prices_df = simulate_sp500(n_days=1000)
print(f"시뮬레이션 데이터: {prices_df.index[0].date()} ~ {prices_df.index[-1].date()}")
print(f"종목 수: {len(prices_df.columns)}, 거래일: {len(prices_df)}\n")

# ════════════════════════════════════════════════════════
# 2. 전략 × 종목 백테스트
# ════════════════════════════════════════════════════════
engine  = BacktestEngine()
INITIAL = 10_000_000  # 1천만원

STRATEGIES = {
    "Momentum60":   MomentumStrategy(lookback=60),
    "MultiFactor":  MultiFactorStrategy(threshold=2),
    "MACD":         MACDStrategy(),
    "MeanReversion":MeanReversionStrategy(window=20, num_std=2.0),
}

print("개별 전략 × 종목 백테스트 중...")
equity_curves: dict[str, pd.Series] = {}
sharpe_table  = pd.DataFrame(index=STRATEGIES.keys(), columns=prices_df.columns, dtype=float)

for strat_name, strat in STRATEGIES.items():
    for ticker in prices_df.columns:
        prices = prices_df[ticker].rename(ticker)
        m = engine.run(strat, prices)
        label = f"{strat_name}_{ticker}"
        equity_curves[label] = m.equity
        sharpe_table.loc[strat_name, ticker] = round(m.sharpe_ratio, 2)

print("\n샤프비율 히트맵:")
print(sharpe_table.to_string())

# ════════════════════════════════════════════════════════
# 3. Risk Management Weight 1.25 포트폴리오
#    ① 각 (전략,종목) 수익률 시계열을 자산으로 취급
#    ② Sharpe 가중치 계산 → 1.25x 스케일
#    ③ 최대 단일 비중 30% 클리핑 → 정규화
# ════════════════════════════════════════════════════════
RISK_WEIGHT = 1.25

ret_df = pd.DataFrame({k: v.pct_change() for k, v in equity_curves.items()}).dropna()

# 240일 룩백으로 리밸런싱 날짜 결정 (월별)
rebal_dates = set(ret_df.resample("ME").last().index)
lookback    = 240

def compute_weights_rm125(hist: pd.DataFrame) -> dict[str, float]:
    """Sharpe 기반 가중치에 Risk Weight 1.25 적용."""
    if len(hist) < 20:
        n = len(hist.columns)
        return {c: 1/n for c in hist.columns}

    # 각 전략의 샤프비율 계산
    ann_ret = hist.mean() * 252
    ann_vol = hist.std() * np.sqrt(252)
    sharpe  = (ann_ret - 0.05) / ann_vol.replace(0, np.nan)
    sharpe  = sharpe.fillna(0).clip(lower=0)  # 음수 샤프 → 0

    # Sharpe 비례 가중 → 1.25x 스케일
    if sharpe.sum() == 0:
        weights = pd.Series(1 / len(hist.columns), index=hist.columns)
    else:
        weights = sharpe / sharpe.sum() * RISK_WEIGHT

    # 단일 자산 최대 30% 클리핑 후 재정규화
    weights = weights.clip(upper=0.30)
    total   = weights.sum()
    weights = weights / total  # 총합=1 (레버리지 없이 정규화)

    # ★ Risk Weight 1.25 적용: 상위 Sharpe 전략에 25% 더 배분
    #   정규화 후 다시 상위권에 보너스
    top_mask = weights >= weights.quantile(0.75)
    weights[top_mask]  *= RISK_WEIGHT
    weights[~top_mask] *= (1 - (weights[top_mask].sum() * (RISK_WEIGHT - 1)) /
                               weights[~top_mask].sum()) if weights[~top_mask].sum() > 0 else 1
    weights = weights.clip(lower=0)
    weights /= weights.sum()
    return weights.to_dict()

# 포트폴리오 시뮬레이션
capital    = INITIAL
equity_rm  = []
weights_rm = {c: 1/len(ret_df.columns) for c in ret_df.columns}
weight_history = {}

for date, row in ret_df.iterrows():
    if date in rebal_dates:
        loc   = ret_df.index.get_loc(date)
        start = max(0, loc - lookback)
        hist  = ret_df.iloc[start:loc]
        if len(hist) >= 20:
            weights_rm = compute_weights_rm125(hist)
            weight_history[date] = weights_rm.copy()

    daily_ret = sum(weights_rm.get(k, 0) * row[k] for k in ret_df.columns)
    capital  *= (1 + daily_ret)
    equity_rm.append(capital)

equity_rm = pd.Series(equity_rm, index=ret_df.index)

# 비교: equal_weight
capital_eq = INITIAL
equity_eq  = []
eq_w = {c: 1/len(ret_df.columns) for c in ret_df.columns}
for date, row in ret_df.iterrows():
    daily_ret = sum(eq_w[k] * row[k] for k in ret_df.columns)
    capital_eq *= (1 + daily_ret)
    equity_eq.append(capital_eq)
equity_eq = pd.Series(equity_eq, index=ret_df.index)

# ════════════════════════════════════════════════════════
# 4. 성과 출력
# ════════════════════════════════════════════════════════
def print_metrics(label, equity):
    ret = equity.pct_change().dropna()
    m   = PerformanceMetrics(equity, ret)
    print(f"\n  [{label}]")
    print(f"    총수익률   : {m.total_return:+.2%}")
    print(f"    연간수익률 : {m.annualized_return:+.2%}")
    print(f"    샤프비율   : {m.sharpe_ratio:.3f}")
    print(f"    MDD        : {m.max_drawdown:.2%}")
    print(f"    칼마비율   : {m.calmar_ratio:.3f}")
    print(f"    승률       : {m.win_rate:.2%}")
    return m

print("\n" + "="*55)
print("  S&P 500 기반 B+C 전략 + Risk Management 1.25x")
print("="*55)
m_rm = print_metrics(f"RM-1.25 (Sharpe×1.25 가중)", equity_rm)
m_eq = print_metrics("Equal Weight (기준선)", equity_eq)

# 최종 비중 상위 10
if weight_history:
    final_w = list(weight_history.values())[-1]
    top10 = sorted(final_w.items(), key=lambda x: -x[1])[:10]
    print(f"\n  최종 리밸런싱 비중 상위 10:")
    for k, w in top10:
        bar = "█" * int(w * 50)
        print(f"    {k:<35} {w:5.1%}  {bar}")

# ════════════════════════════════════════════════════════
# 5. 차트
# ════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 14))
gs  = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])   # 수익 곡선 전체
ax2 = fig.add_subplot(gs[1, 0])   # 드로우다운
ax3 = fig.add_subplot(gs[1, 1])   # 개별 S&P500 주가
ax4 = fig.add_subplot(gs[2, 0])   # 샤프 히트맵
ax5 = fig.add_subplot(gs[2, 1])   # 최종 비중

# ── 수익 곡선
ax1.plot(equity_rm / INITIAL, color="#2563EB", linewidth=2.5, label=f"RM-1.25 (최종 {equity_rm.iloc[-1]/INITIAL:.2f}x)")
ax1.plot(equity_eq / INITIAL, color="#94a3b8", linewidth=1.5, linestyle="--", label=f"Equal Weight (최종 {equity_eq.iloc[-1]/INITIAL:.2f}x)")
ax1.fill_between(equity_rm.index, equity_rm / INITIAL, 1, where=equity_rm/INITIAL >= 1, alpha=0.1, color="#2563EB")
ax1.set_title("S&P 500 기반 전략 포트폴리오 — RM 1.25x vs Equal Weight", fontsize=13, fontweight="bold")
ax1.set_ylabel("수익률 (원금 = 1)")
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.axhline(1, color="black", linewidth=0.7, linestyle=":")

# ── 드로우다운
rolling_max = equity_rm.cummax()
dd_rm = (equity_rm - rolling_max) / rolling_max
rolling_max_eq = equity_eq.cummax()
dd_eq = (equity_eq - rolling_max_eq) / rolling_max_eq
ax2.fill_between(dd_rm.index, dd_rm, 0, alpha=0.5, color="#ef4444", label=f"RM-1.25 MDD {dd_rm.min():.1%}")
ax2.fill_between(dd_eq.index, dd_eq, 0, alpha=0.3, color="#94a3b8", label=f"EW MDD {dd_eq.min():.1%}")
ax2.set_title("드로우다운 비교", fontsize=11)
ax2.set_ylabel("Drawdown")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# ── S&P 500 개별 주가 (정규화)
for ticker in prices_df.columns:
    norm = prices_df[ticker] / prices_df[ticker].iloc[0]
    sector = TICKERS[ticker][0]
    ax3.plot(norm, linewidth=1, alpha=0.7, label=f"{ticker} ({sector})")
ax3.set_title("S&P 500 대표 종목 가격 (정규화)", fontsize=11)
ax3.set_ylabel("정규화 가격")
ax3.legend(fontsize=7, ncol=2)
ax3.grid(alpha=0.3)

# ── 샤프 히트맵
import matplotlib.colors as mcolors
sharpe_vals = sharpe_table.astype(float)
im = ax4.imshow(sharpe_vals.values, cmap="RdYlGn", aspect="auto",
                vmin=sharpe_vals.values.min(), vmax=sharpe_vals.values.max())
ax4.set_xticks(range(len(sharpe_vals.columns)))
ax4.set_xticklabels(sharpe_vals.columns, rotation=45, fontsize=8)
ax4.set_yticks(range(len(sharpe_vals.index)))
ax4.set_yticklabels(sharpe_vals.index, fontsize=8)
ax4.set_title("전략 × 종목 샤프비율 히트맵", fontsize=11)
for i in range(len(sharpe_vals.index)):
    for j in range(len(sharpe_vals.columns)):
        ax4.text(j, i, f"{sharpe_vals.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax4)

# ── 최종 비중 바 차트
if weight_history:
    final_w = list(weight_history.values())[-1]
    top_items = sorted(final_w.items(), key=lambda x: -x[1])[:12]
    labels = [k.replace("_", "\n") for k, _ in top_items]
    vals   = [v for _, v in top_items]
    colors_bar = ["#2563EB" if v >= np.percentile(vals, 75) else "#60a5fa" for v in vals]
    bars = ax5.barh(range(len(labels)), vals, color=colors_bar)
    ax5.set_yticks(range(len(labels)))
    ax5.set_yticklabels(labels, fontsize=7)
    ax5.set_xlabel("비중")
    ax5.set_title("최종 RM-1.25 전략 비중 (상위 12)", fontsize=11)
    ax5.axvline(1/len(final_w), color="red", linestyle="--", alpha=0.5, label="Equal Weight")
    ax5.legend(fontsize=8)
    ax5.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax5.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.1%}", va="center", fontsize=7)

plt.suptitle("S&P 500 기반 B+C 전략 포트폴리오 + Risk Management (Weight=1.25)",
             fontsize=14, fontweight="bold", y=1.01)
plt.savefig("sp500_rm125_portfolio.png", dpi=150, bbox_inches="tight")
print("\n차트 저장: sp500_rm125_portfolio.png")
