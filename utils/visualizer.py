import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    @staticmethod
    def candlestick(ohlcv: pd.DataFrame, title: str = "", n: int = 60) -> None:
        data = ohlcv.tail(n).copy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        for i, (date, row) in enumerate(data.iterrows()):
            color = "red" if row["Close"] >= row["Open"] else "blue"
            ax1.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=1)
            ax1.bar(i, abs(row["Close"] - row["Open"]),
                    bottom=min(row["Open"], row["Close"]),
                    color=color, alpha=0.7, width=0.6)

        step = max(1, n // 10)
        ax1.set_xticks(range(0, len(data), step))
        ax1.set_xticklabels([str(d)[:10] for d in data.index[::step]], rotation=45)
        ax1.set_title(title)
        ax1.set_ylabel("Price")
        ax1.grid(alpha=0.3)

        ax2.bar(range(len(data)), data["Volume"], color="gray", alpha=0.5)
        ax2.set_ylabel("Volume")
        ax2.set_xticks(range(0, len(data), step))
        ax2.set_xticklabels([str(d)[:10] for d in data.index[::step]], rotation=45)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{title or 'chart'}.png", dpi=150)
        plt.show()

    @staticmethod
    def compare_equity(curves: dict[str, pd.Series], title: str = "Strategy Comparison") -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, curve in curves.items():
            normalized = curve / curve.iloc[0]
            ax.plot(normalized, label=name)
        ax.set_title(title)
        ax.set_ylabel("Normalized Return")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("strategy_comparison.png", dpi=150)
        plt.show()
