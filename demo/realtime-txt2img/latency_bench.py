"""
HTTP/TCP Latency Benchmark for StreamDiffusion realtime-txt2img demo.

Usage:
    # Start the server first:
    #   cd demo/realtime-txt2img && python main.py
    python latency_bench.py [--url http://localhost:7860] [--n 100] [--warmup 10]

Outputs per-request latency stats and a CSV for further analysis.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


def run_benchmark(url: str, prompt: str, n: int, warmup: int) -> list[dict]:
    try:
        import httpx
    except ImportError:
        sys.exit("Install httpx: pip install httpx")

    endpoint = f"{url}/api/predict"
    payload = {"prompt": prompt}
    results = []

    print(f"Target: {endpoint}")
    print(f"Prompt: '{prompt}'")
    print(f"Warmup: {warmup}  Measured: {n}")
    print()

    # Reuse the same HTTP connection (keep-alive) so TCP handshake is amortized.
    with httpx.Client(timeout=60.0) as client:
        for i in range(warmup + n):
            t_send = time.perf_counter_ns()
            resp = client.post(endpoint, json=payload)
            t_recv = time.perf_counter_ns()
            resp.raise_for_status()

            e2e_ms = (t_recv - t_send) / 1e6
            body = resp.json()
            # base64 image size in bytes (characters ≈ bytes for ASCII)
            payload_kb = len(body.get("base64_image", "")) / 1024

            if i < warmup:
                print(f"  warmup {i+1:3d}: {e2e_ms:.1f} ms")
                continue

            results.append({
                "request_num": i - warmup + 1,
                "e2e_ms": e2e_ms,
                "payload_kb": payload_kb,
            })
            print(f"  req {i - warmup + 1:3d}: {e2e_ms:.1f} ms  payload={payload_kb:.1f} KB")

    return results


def print_stats(results: list[dict]) -> None:
    e2e = [r["e2e_ms"] for r in results]
    payload = [r["payload_kb"] for r in results]
    n = len(e2e)

    e2e_sorted = sorted(e2e)
    p95 = e2e_sorted[int(0.95 * n)]
    p99 = e2e_sorted[int(0.99 * n)]

    print()
    print("=" * 55)
    print("  HTTP/TCP Latency Results")
    print("=" * 55)
    print(f"  Requests:        {n}")
    print(f"  Mean E2E:        {statistics.mean(e2e):.2f} ms")
    print(f"  Median E2E:      {statistics.median(e2e):.2f} ms")
    print(f"  p95 E2E:         {p95:.2f} ms")
    print(f"  p99 E2E:         {p99:.2f} ms")
    print(f"  StdDev (jitter): {statistics.stdev(e2e):.2f} ms")
    print(f"  Min E2E:         {min(e2e):.2f} ms")
    print(f"  Max E2E:         {max(e2e):.2f} ms")
    print(f"  Throughput:      {1000 / statistics.mean(e2e):.2f} req/s")
    print(f"  Payload mean:    {statistics.mean(payload):.1f} KB (base64 JPEG)")
    print("=" * 55)

    # Buckets
    buckets = [0, 50, 100, 150, 200, 300, 500, float("inf")]
    labels = ["<50", "50-100", "100-150", "150-200", "200-300", "300-500", "500+"]
    counts = [0] * len(labels)
    for v in e2e:
        for j, upper in enumerate(buckets[1:]):
            if v < upper:
                counts[j] += 1
                break

    print("\n  Latency histogram (ms):")
    for label, count in zip(labels, counts):
        bar = "█" * count
        print(f"    {label:>8} ms: {bar} ({count})")


def save_csv(results: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP latency benchmark")
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--prompt", default="a cat sitting on a bench, photorealistic")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--csv", default="http_latency.csv")
    args = parser.parse_args()

    results = run_benchmark(args.url, args.prompt, args.n, args.warmup)
    print_stats(results)
    save_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()