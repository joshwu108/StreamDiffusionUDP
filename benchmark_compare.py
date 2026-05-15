"""
TCP vs UDP Latency Comparison — StreamDiffusion Edge Computing Research.

Reads live-measured CSV files from the HTTP/TCP and UDP benchmarks and prints
a side-by-side statistical comparison with delta and percent-change columns.
All statistics are computed at runtime; no numbers are hardcoded.

Usage:
    python benchmark_compare.py \
        [--tcp-csv demo/realtime-txt2img/http_latency.csv] \
        [--udp-csv udp_latency.csv] \
        [--plot]

Required CSV columns:
    TCP CSV: request_num, e2e_ms, server_ms, net_ms, payload_kb
    UDP CSV: request_num, e2e_ms, server_ms, net_ms, payload_kb, chunks
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"CSV not found: {path}\nRun the benchmark first.")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"CSV is empty: {path}")
    return rows


def extract(rows: list[dict], key: str) -> list[float]:
    vals = []
    for r in rows:
        v = r.get(key)
        if v not in (None, ""):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return vals


def percentile(sorted_vals: list[float], p: float) -> float:
    idx = int(p * len(sorted_vals))
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "mean":   statistics.mean(vals),
        "median": statistics.median(vals),
        "p95":    percentile(s, 0.95),
        "p99":    percentile(s, 0.99),
        "stdev":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min":    min(vals),
        "max":    max(vals),
    }


def fmt_delta(tcp_val: float, udp_val: float) -> tuple[str, str]:
    delta = udp_val - tcp_val
    pct = (delta / tcp_val * 100) if tcp_val != 0 else float("nan")
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f}", f"{sign}{pct:.1f}%"


def print_comparison(tcp_rows: list[dict], udp_rows: list[dict]) -> None:
    tcp_e2e  = extract(tcp_rows, "e2e_ms")
    udp_e2e  = extract(udp_rows, "e2e_ms")
    tcp_srv  = extract(tcp_rows, "server_ms")
    udp_srv  = extract(udp_rows, "server_ms")
    tcp_net  = extract(tcp_rows, "net_ms")
    udp_net  = extract(udp_rows, "net_ms")
    tcp_pay  = extract(tcp_rows, "payload_kb")
    udp_pay  = extract(udp_rows, "payload_kb")

    te = compute_stats(tcp_e2e)
    ue = compute_stats(udp_e2e)

    W = 62
    print()
    print("=" * W)
    print("  TCP vs UDP Latency Comparison  —  StreamDiffusion Edge Research")
    print("=" * W)
    print(f"  HTTP/TCP requests: {len(tcp_e2e)}    UDP requests: {len(udp_e2e)}")
    print()

    # Header
    col = 14
    sep = "-" * col
    print(f"  {'Metric':<26} {'HTTP/TCP':>{col}} {'UDP':>{col}} {'Delta':>{col}} {'% Change':>{col}}")
    print(f"  {'-'*26} {sep} {sep} {sep} {sep}")

    def row(label, tcp_v, udp_v):
        d, p = fmt_delta(tcp_v, udp_v)
        print(f"  {label:<26} {tcp_v:>{col}.2f} {udp_v:>{col}.2f} {d:>{col}} {p:>{col}}")

    row("E2E Mean",        te["mean"],   ue["mean"])
    row("E2E Median",      te["median"], ue["median"])
    row("E2E p95",         te["p95"],    ue["p95"])
    row("E2E p99",         te["p99"],    ue["p99"])
    row("Jitter (StdDev)", te["stdev"],  ue["stdev"])
    row("Min",             te["min"],    ue["min"])
    row("Max",             te["max"],    ue["max"])

    tcp_tput = 1000 / te["mean"] if te["mean"] else 0
    udp_tput = 1000 / ue["mean"] if ue["mean"] else 0
    d, p = fmt_delta(tcp_tput, udp_tput)
    print(f"  {'Throughput (req/s)':<26} {tcp_tput:>{col}.2f} {udp_tput:>{col}.2f} {d:>{col}} {p:>{col}}")

    if tcp_pay and udp_pay:
        row("Payload mean (KB)", statistics.mean(tcp_pay), statistics.mean(udp_pay))

    print()

    # Server vs Network breakdown
    if tcp_srv and udp_srv and tcp_net and udp_net:
        print(f"  {'--- Latency Decomposition ---':<26}")
        print(f"  {'Metric':<26} {'HTTP/TCP':>{col}} {'UDP':>{col}} {'Delta':>{col}} {'% Change':>{col}}")
        print(f"  {'-'*26} {sep} {sep} {sep} {sep}")
        row("Server Mean (inference)",   statistics.mean(tcp_srv), statistics.mean(udp_srv))
        row("Network Overhead Mean",     statistics.mean(tcp_net), statistics.mean(udp_net))
        row("Server p99",                percentile(sorted(tcp_srv), 0.99), percentile(sorted(udp_srv), 0.99))
        row("Network p99",               percentile(sorted(tcp_net), 0.99), percentile(sorted(udp_net), 0.99))
        print()

    print("=" * W)

    # Histogram comparison
    buckets = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, float("inf"))]
    labels  = ["<50", "50-100", "100-150", "150-200", "200-300", "300-500", "500+"]

    def bucket_counts(vals):
        counts = [0] * len(labels)
        for v in vals:
            for j, (lo, hi) in enumerate(buckets):
                if lo <= v < hi:
                    counts[j] += 1
                    break
        return counts

    tc = bucket_counts(tcp_e2e)
    uc = bucket_counts(udp_e2e)
    max_count = max(max(tc), max(uc), 1)
    bar_width = 20

    print()
    print("  E2E Latency Distribution (ms)")
    bw5 = bar_width + 5
    print(f"  {'Range':>8}   {'HTTP/TCP':<{bw5}}   {'UDP':<{bw5}}")
    print(f"  {'-'*8}   {'-'*bw5}   {'-'*bw5}")
    for label, tv, uv in zip(labels, tc, uc):
        tb = "█" * int(tv / max_count * bar_width)
        ub = "█" * int(uv / max_count * bar_width)
        print(f"  {label:>8}   {tb:<{bar_width}} ({tv:3d})   {ub:<{bar_width}} ({uv:3d})")
    print()


def plot_comparison(tcp_rows: list[dict], udp_rows: list[dict], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib/numpy not available — skipping plot (pip install matplotlib)")
        return

    tcp_e2e = extract(tcp_rows, "e2e_ms")
    udp_e2e = extract(udp_rows, "e2e_ms")
    tcp_srv = extract(tcp_rows, "server_ms")
    udp_srv = extract(udp_rows, "server_ms")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("TCP vs UDP Latency — StreamDiffusion Edge Computing", fontsize=13)

    # Panel 1: CDF of E2E latency
    ax = axes[0]
    for vals, label, color in [(tcp_e2e, "HTTP/TCP", "#e74c3c"), (udp_e2e, "UDP", "#2ecc71")]:
        s = sorted(vals)
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, cdf, label=label, color=color, linewidth=2)
    ax.set_xlabel("E2E Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Key percentile bar chart
    ax = axes[1]
    metrics = ["Mean", "Median", "p95", "p99"]
    te = compute_stats(tcp_e2e)
    ue = compute_stats(udp_e2e)
    tcp_vals = [te["mean"], te["median"], te["p95"], te["p99"]]
    udp_vals = [ue["mean"], ue["median"], ue["p95"], ue["p99"]]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, tcp_vals, w, label="HTTP/TCP", color="#e74c3c", alpha=0.8)
    ax.bar(x + w/2, udp_vals, w, label="UDP", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Key Percentiles")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: Latency over time (request number)
    ax = axes[2]
    tcp_nums = [int(r["request_num"]) for r in tcp_rows if r.get("e2e_ms")]
    udp_nums = [int(r["request_num"]) for r in udp_rows if r.get("e2e_ms")]
    ax.plot(tcp_nums, tcp_e2e, label="HTTP/TCP", color="#e74c3c", alpha=0.7, linewidth=1)
    ax.plot(udp_nums, udp_e2e, label="UDP", color="#2ecc71", alpha=0.7, linewidth=1)
    ax.set_xlabel("Request Number")
    ax.set_ylabel("E2E Latency (ms)")
    ax.set_title("Latency Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if tcp_srv and udp_srv:
        ax.axhline(statistics.mean(tcp_srv), color="#e74c3c", linestyle="--", alpha=0.4, label="TCP server mean")
        ax.axhline(statistics.mean(udp_srv), color="#2ecc71", linestyle="--", alpha=0.4, label="UDP server mean")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Chart saved to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare TCP vs UDP benchmark CSVs")
    p.add_argument("--tcp-csv", default="demo/realtime-txt2img/http_latency.csv",
                   help="Path to HTTP/TCP benchmark CSV")
    p.add_argument("--udp-csv", default="udp_latency.csv",
                   help="Path to UDP benchmark CSV")
    p.add_argument("--plot", action="store_true",
                   help="Generate latency_comparison.png (requires matplotlib)")
    args = p.parse_args()

    tcp_rows = load_csv(Path(args.tcp_csv))
    udp_rows = load_csv(Path(args.udp_csv))

    print_comparison(tcp_rows, udp_rows)

    if args.plot:
        plot_comparison(tcp_rows, udp_rows, Path("latency_comparison.png"))


if __name__ == "__main__":
    main()
