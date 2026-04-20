"""
UDP Inference Server for StreamDiffusion txt2img.

Architecture:
  - Single UDP socket (SOCK_DGRAM) on configurable host:port
  - One-worker ThreadPoolExecutor keeps the recv loop non-blocking during inference
  - No asyncio overhead: inference is sequential on a single GPU anyway
  - Embeds server_recv_ns and server_send_ns in every response for latency decomposition

Usage:
    cd /path/to/StreamDiffusionUDP
    python demo/udp-txt2img/server.py \
        --model KBlueLeaf/kohaku-v2.1 \
        --host 0.0.0.0 \
        --port 9999 \
        --acceleration xformers
"""

import argparse
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from protocol import fragment_jpeg, RequestPacket

from utils.wrapper import StreamDiffusionWrapper


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="StreamDiffusion UDP server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--model", default="KBlueLeaf/kohaku-v2.1")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--t-index-list", nargs="+", type=int, default=[32, 45])
    p.add_argument("--acceleration", default="xformers",
                   choices=["none", "xformers", "tensorrt"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--jpeg-quality", type=int, default=40)
    p.add_argument("--use-lcm-lora", action="store_true", default=True)
    return p.parse_args()


def pil_to_jpeg_bytes(image, quality: int) -> bytes:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def serve(args: argparse.Namespace) -> None:
    print(f"Loading model '{args.model}'...")
    wrapper = StreamDiffusionWrapper(
        model_id_or_path=args.model,
        t_index_list=args.t_index_list,
        mode="txt2img",
        output_type="pil",
        device="cuda",
        width=args.width,
        height=args.height,
        warmup=args.warmup,
        acceleration=args.acceleration,
        use_lcm_lora=args.use_lcm_lora,
        use_tiny_vae=True,
        cfg_type="none",
    )
    print("Model ready.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"UDP server listening on {args.host}:{args.port}")

    # One worker thread keeps inference sequential (single GPU) while the recv
    # loop stays responsive. Futures are submitted and forgotten — client
    # drives flow control via timeouts.
    executor = ThreadPoolExecutor(max_workers=1)

    def handle_request(data: bytes, addr: tuple, server_recv_ns: int) -> None:
        try:
            req = RequestPacket.unpack(data)
        except Exception as e:
            print(f"Bad packet from {addr}: {e}")
            return

        image = wrapper(prompt=req.prompt)
        jpeg_bytes = pil_to_jpeg_bytes(image, args.jpeg_quality)
        server_send_ns = time.perf_counter_ns()

        packets = fragment_jpeg(
            jpeg_bytes=jpeg_bytes,
            seq_id=req.seq_id,
            client_ts_ns=req.timestamp_ns,
            server_recv_ns=server_recv_ns,
            server_send_ns=server_send_ns,
        )
        for pkt in packets:
            sock.sendto(pkt, addr)

        inference_ms = (server_send_ns - server_recv_ns) / 1e6
        print(f"[seq {req.seq_id:05d}] {addr[0]}:{addr[1]}  "
              f"server={inference_ms:.1f}ms  "
              f"jpeg={len(jpeg_bytes)/1024:.1f}KB  "
              f"chunks={len(packets)}")

    try:
        while True:
            data, addr = sock.recvfrom(65535)
            server_recv_ns = time.perf_counter_ns()
            executor.submit(handle_request, data, addr, server_recv_ns)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        executor.shutdown(wait=False)
        sock.close()


if __name__ == "__main__":
    serve(build_args())
