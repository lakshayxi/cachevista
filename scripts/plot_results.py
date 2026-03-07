import csv
import matplotlib.pyplot as plt
from pathlib import Path
from cachevista.config import load

cfg = load()
results_dir = Path(cfg["benchmark"]["results_dir"])
figures_dir = Path(cfg["benchmark"]["figures_dir"])
figures_dir.mkdir(exist_ok=True)

with open(results_dir / "benchmark.csv") as f:
    rows = list(csv.DictReader(f))

strategies = [r["strategy"] for r in rows]
hit_rates = [float(r["hit_rate"]) * 100 for r in rows]
avg_latencies = [float(r["avg_latency_ms"]) for r in rows]
colors = ["#94A3B8", "#60A5FA", "#1F3A8A"]


def bar_chart(values, ylabel, title, filename, unit=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(strategies, values, color=colors, width=0.5, edgecolor="white")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    max_val = max(values) if max(values) > 0 else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.01,
            f"{val:.2f}{unit}",
            ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150)
    plt.close()
    print(f"saved {filename}")


bar_chart(hit_rates, "Hit Rate (%)", "Cache Hit Rate by Strategy", "hit_rate.png", "%")
bar_chart(avg_latencies, "Avg Latency (ms)", "Average Latency per Query", "latency.png", "ms")