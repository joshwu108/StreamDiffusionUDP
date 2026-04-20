"""
UDP Latency Benchmark Client for StreamDiffusion.

Sends N inference requests via UDP and measures:
  - E2E latency: client-side send → receive (all chunks)
  - Server latency: server_recv_ns → server_send_ns (embedded in response)
  - Network overhead: E2E - server_latency

Usage:
    # Start UDP server first:
    #   python demo/udp-txt2img/server.py
    python demo/udp-txt2img/client.py [--host 127.0.0.1] [--port 9999] [--n 100]
"""

import argparse
import csv
import socket
import statistics
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path


def _import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        return None


sys.path.insert(0, str(Path(__file__).parent))
from protocol import RequestPacket, ResponsePacket


def reassemble(chunks: dict[int, bytes], total: int) -> bytes:
    return b"".join(chunks[i] for i in range(total))


def run_benchmark(
    host: str,
    port: int,
    prompt: str,
    n: int,
    warmup: int,
    timeout_s: float,
    save_last_image: bool,
) -> list[dict]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout_s)
    server_addr = (host, port)

    results = []
    seq_id = 0
    Image = _import_pil()

    print(f"Target: udp://{host}:{port}")
    print(f"Prompt: '{prompt}'")
    print(f"Warmup: {warmup}  Measured: {n}  Timeout: {timeout_s}s")
    print()

    for i in range(warmup + n):
        seq_id += 1
        req = RequestPacket(
            seq_id=seq_id,
            timestamp_ns=time.perf_counter_ns(),
            prompt=prompt,
        )
        t_send_ns = time.perf_counter_ns()
        sock.sendto(req.pack(), server_addr)

        # Collect all chunks for this seq_id
        chunks: dict[int, bytes] = {}
        total_chunks = None
        server_recv_ns = None
        server_send_ns = None
        timed_out = False

        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                timed_out = True
                break

            try:
                resp = ResponsePacket.unpack(data)
            except Exception:
                continue

            if resp.seq_id != seq_id:
                continue  # stale packet from a previous request

            if server_recv_ns is None:
                server_recv_ns = resp.server_recv_ns
                server_send_ns = resp.server_send_ns
            total_chunks = resp.total_chunks
            chunks[resp.chunk_idx] = resp.data

            if len(chunks) == total_chunks:
                break

        t_recv_ns = time.perf_counter_ns()

        if timed_out:
            print(f"  {'warmup' if i < warmup else 'req':6s} {i+1:3d}: TIMEOUT")
            continue

        e2e_ms = (t_recv_ns - t_send_ns) / 1e6
        server_ms = (server_send_ns - server_recv_ns) / 1e6
        net_ms = e2e_ms - server_ms
        jpeg = reassemble(chunks, total_chunks)
        payload_kb = len(jpeg) / 1024

        if i < warmup:
            print(f"  warmup {i+1:3d}: {e2e_ms:.1f} ms  server={server_ms:.1f} ms")
            continue

        results.append({
            "request_num": i - warmup + 1,
            "e2e_ms": e2e_ms,
            "server_ms": server_ms,
            "net_ms": net_ms,
            "payload_kb": payload_kb,
            "chunks": total_chunks,
        })
        print(
            f"  req {i - warmup + 1:3d}: e2e={e2e_ms:.1f}ms  "
            f"server={server_ms:.1f}ms  net={net_ms:.1f}ms  "
            f"payload={payload_kb:.1f}KB"
        )

        if save_last_image and i == warmup + n - 1 and Image is not None:
            img = Image.open(BytesIO(jpeg))
            out_path = Path("udp_last_frame.jpg")
            img.save(out_path)
            print(f"\n  Last frame saved to {out_path}")

    sock.close()
    return results


def print_stats(results: list[dict]) -> None:
    if not results:
        print("No results collected.")
        return

    def stats(vals: list[float], label: str) -> None:
        s = sorted(vals)
        n = len(s)
        print(f"\n  {label}:")
        print(f"    Mean:   {statistics.mean(vals):.2f} ms")
        print(f"    Median: {statistics.median(vals):.2f} ms")
        print(f"    p95:    {s[int(0.95 * n)]:.2f} ms")
        print(f"    p99:    {s[int(0.99 * n)]:.2f} ms")
        print(f"    StdDev: {statistics.stdev(vals):.2f} ms")
        print(f"    Min:    {min(vals):.2f} ms")
        print(f"    Max:    {max(vals):.2f} ms")

    e2e = [r["e2e_ms"] for r in results]
    srv = [r["server_ms"] for r in results]
    net = [r["net_ms"] for r in results]
    pay = [r["payload_kb"] for r in results]

    print()
    print("=" * 55)
    print("  UDP Latency Results")
    print("=" * 55)
    print(f"  Requests measured: {len(results)}")

    stats(e2e, "End-to-End Latency")
    stats(srv, "Server-Side Latency (inference + encode)")
    stats(net, "Network Overhead (E2E - server)")

    print(f"\n  Throughput:        {1000 / statistics.mean(e2e):.2f} req/s")
    print(f"  Payload mean:      {statistics.mean(pay):.1f} KB (raw JPEG)")
    print("=" * 55)

    buckets = [0, 50, 100, 150, 200, 300, 500, float("inf")]
    labels = ["<50", "50-100", "100-150", "150-200", "200-300", "300-500", "500+"]
    counts = [0] * len(labels)
    for v in e2e:
        for j, upper in enumerate(buckets[1:]):
            if v < upper:
                counts[j] += 1
                break

    print("\n  E2E latency histogram (ms):")
    for label, count in zip(labels, counts):
        bar = "█" * count
        print(f"    {label:>8} ms: {bar} ({count})")


def save_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="UDP latency benchmark")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--prompt", default="a cat sitting on a bench, photorealistic")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timeout", type=float, default=5.0,
                   help="Per-request timeout in seconds")
    p.add_argument("--csv", default="udp_latency.csv")
    p.add_argument("--save-image", action="store_true",
                   help="Save the last received frame as udp_last_frame.jpg")
    args = p.parse_args()

    results = run_benchmark(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        n=args.n,
        warmup=args.warmup,
        timeout_s=args.timeout,
        save_last_image=args.save_image,
    )
    print_stats(results)
    save_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
