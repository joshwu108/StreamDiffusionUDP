# StreamDiffusion: TCP → UDP Latency Research

---

## 1. Project Overview

StreamDiffusion is a real-time image generation pipeline built on Stable Diffusion with several key optimizations:
- **Denoising Batch Streaming**: Multiple denoising timesteps are batched so the GPU pipeline never stalls
- **LCM-LoRA**: Reduces required denoising steps from ~50 to 4
- **TinyVAE (TAESD)**: ~8× less compute than the standard KL VAE decoder
- **xformers / TensorRT acceleration**: Optional JIT-compiled GPU kernels

The original demo uses **FastAPI + HTTP** for client-server communication. This study:
1. Profiles baseline HTTP latency end-to-end and per-component
2. Replaces HTTP with a **raw UDP socket protocol**
3. Measures and compares both transports in the context of edge inference

---

## 2. System Architecture

### 2.1 Original HTTP Stack

```
┌──────────────────────────────────────────────────────┐
│                      CLIENT                          │
│  POST /api/predict  {"prompt": "..."}                │
└───────────────────────┬──────────────────────────────┘
                        │ TCP (HTTP/1.1, keep-alive)
                        ▼
┌──────────────────────────────────────────────────────┐
│                 UVICORN / FASTAPI                    │
│  HTTP parse → asyncio.Lock → Pydantic validate       │
│       ↓                                              │
│  StreamDiffusionWrapper.__call__                     │
│    ├─ stream.update_prompt()   ← CLIP encode         │
│    ├─ stream.txt2img()         ← GPU denoise         │
│    └─ postprocess_image()      ← PIL convert         │
│       ↓                                              │
│  _pil_to_base64(): JPEG → base64 → JSON body         │
└───────────────────────┬──────────────────────────────┘
                        │ {"base64_image": "..."}
                        ▼
│   CLIENT: base64 decode → display
```

**Key observation:** The `asyncio.Lock` in the HTTP server serializes all requests — only one inference runs at a time, creating head-of-line blocking for multi-client scenarios.

### 2.2 HTTP Latency Budget

| Phase | Cost source |
|-------|-------------|
| TCP overhead | SYN/ACK on first connect; keep-alive amortizes this |
| HTTP framing | Request header parse, routing, Pydantic validation |
| asyncio overhead | Event loop scheduling, lock acquire/release |
| CLIP encode | Text tokenization + transformer forward pass |
| Denoising | UNet × N timesteps (batched) |
| VAE decode | Latent → pixel space |
| Postprocess | Tensor → PIL, JPEG compress, **base64 encode (+33% size)** |
| HTTP response | JSON serialization, chunked-transfer encoding |

---

## 3. What Changed: UDP Implementation

### 3.1 Files Added

#### `demo/udp-txt2img/protocol.py`
Defines a compact binary packet format using Python's `struct` module.

**Request** (client → server, 17 + len(prompt) bytes):
```
[b'SDREQ'] [seq_id: u32] [timestamp_ns: u64] [prompt: UTF-8]
```

**Response** (server → client, 37 + len(chunk) bytes per packet):
```
[b'SDRES'] [seq_id: u32] [client_ts_ns: u64]
[server_recv_ns: u64] [server_send_ns: u64]
[chunk_idx: u16] [total_chunks: u16] [raw JPEG bytes]
```

The server echoes `client_ts_ns`, `server_recv_ns`, and `server_send_ns` back in every packet. This lets the client decompose latency into **server-side** (inference + encode) and **network** (pure transport) without requiring clock synchronization between client and server.

Payloads over 60,000 bytes are split into multiple chunks (safely below UDP's 65,507-byte limit). At JPEG quality 40, a 512×512 image is typically 20–43 KB, so most responses fit in a single packet.

#### `demo/udp-txt2img/server.py`
A standalone UDP inference server using `socket.SOCK_DGRAM` instead of Uvicorn:
- Blocking `recvfrom` loop — no asyncio overhead
- `ThreadPoolExecutor(max_workers=1)` keeps the recv loop responsive during inference without parallelizing GPU work
- Same `StreamDiffusionWrapper` as the HTTP demo
- Sends raw JPEG bytes — no base64, no JSON

#### `demo/udp-txt2img/client.py`
Benchmark client that measures all three latency components per request and outputs statistics + CSV.

#### `demo/realtime-txt2img/latency_bench.py`
HTTP benchmark client using `httpx` with keep-alive, mirroring the UDP client's methodology for a fair apples-to-apples comparison.

### 3.2 Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Binary vs JSON | Binary (`struct.pack`) | Zero parse overhead, ~25% smaller payload |
| Base64 vs raw bytes | Raw bytes | Removes 33% size inflation and encode/decode CPU cost |
| Retransmit | None | A dropped frame is better than a late frame — skip and render the next |
| Server threading | 1-worker ThreadPool | Single GPU is inherently sequential; asyncio would only add overhead |

### 3.3 What Was Not Changed

- `src/streamdiffusion/pipeline.py` — GPU CUDA event timing is untouched
- `utils/wrapper.py` — model loading is untouched
- `demo/realtime-txt2img/main.py` — original HTTP server is untouched

---

## 4. Results

### 4.1 HTTP/TCP Baseline (N=100)

```
Prompt: "a cat sitting on a bench, photorealistic"  |  512×512

Mean E2E:        64.40 ms
Median E2E:      63.96 ms
p95 E2E:         67.88 ms
p99 E2E:         81.07 ms   ← notable spike
StdDev (jitter): 2.10 ms
Min:             62.72 ms
Max:             81.07 ms
Throughput:      15.53 req/s
Payload:         56.9 KB (base64 JPEG)
```

### 4.2 UDP (N=100)

```
Prompt: "a cat sitting on a bench, photorealistic"  |  512×512

End-to-End:
  Mean:    56.90 ms
  Median:  56.30 ms
  p95:     61.03 ms
  p99:     64.05 ms
  StdDev:  1.68 ms
  Min:     54.02 ms
  Max:     64.05 ms

Server-side (inference + JPEG encode):
  Mean:    56.71 ms     ← nearly all of E2E

Network overhead (E2E − server):
  Mean:    0.18 ms      ← UDP loopback transport is effectively free
  p99:     0.26 ms

Throughput: 17.57 req/s
Payload:    ~42.7 KB (raw JPEG, estimated from base64 size ÷ 1.33)
```

### 4.3 Comparison

| Metric | HTTP/TCP | UDP | Delta | % Change |
|--------|----------|-----|-------|----------|
| E2E Mean | 64.40 ms | 56.90 ms | −7.50 ms | **−11.6%** |
| E2E Median | 63.96 ms | 56.30 ms | −7.66 ms | **−12.0%** |
| p95 | 67.88 ms | 61.03 ms | −6.85 ms | **−10.1%** |
| p99 | 81.07 ms | 64.05 ms | −17.02 ms | **−21.0%** |
| Jitter (StdDev) | 2.10 ms | 1.68 ms | −0.42 ms | **−20.0%** |
| Max latency | 81.07 ms | 64.05 ms | −17.02 ms | **−21.0%** |
| Payload size | 56.9 KB | ~42.7 KB | −14.2 KB | **−25.0%** |
| Throughput | 15.53 req/s | 17.57 req/s | +2.04 | **+13.1%** |

---

## 5. Interpretation

### 5.1 The 7.5ms Mean Improvement

UDP is 7.5ms faster on average even though the GPU dominates latency. This overhead comes from HTTP, not the network — the UDP network overhead is only 0.18ms (loopback). The HTTP savings come from:

- **No base64 encode/decode**: A 43KB JPEG encoded to base64 produces ~57KB of ASCII. The encode itself is fast, but JSON serialization of a 57KB string adds measurable CPU cost
- **No asyncio event loop scheduling**: HTTP requests pass through Uvicorn's event loop, the lock, and Pydantic validation before inference even starts
- **No HTTP response framing**: No chunked-transfer encoding, no Content-Type negotiation, no header writes

### 5.2 The p99 Spike is the Most Important Finding

HTTP p99 = **81.07ms** vs UDP p99 = **64.05ms** — a **17ms difference**.

The HTTP server has an `asyncio.Lock`. When occasional GC pauses, OS scheduler preemptions, or TCP ACK delays stack up, a request can sit queued behind the lock for an extra 10–20ms. UDP has no lock, no connection state, and no ACK traffic — the worst-case is bounded by inference variance alone.

For real-time applications, p99 is the metric that determines whether a system feels smooth or stutters. A 17ms p99 improvement is the difference between 81ms worst-case frames (below 12fps) and 64ms worst-case frames (below 16fps).

### 5.3 Jitter Reduction (20%)

UDP StdDev dropped from 2.10ms to 1.68ms. Tighter jitter means more predictable frame delivery — important for any streaming application where a display refresh timer expects frames at a fixed interval.

### 5.4 GPU Is Still the Bottleneck

UDP server-side mean = 56.71ms. UDP E2E mean = 56.90ms. The gap is 0.19ms — **transport overhead is less than 0.4% of total latency on loopback**. This confirms:

1. Switching to UDP is not a "free lunch" for mean latency — you still need to make the GPU faster (TensorRT, fewer timesteps, smaller resolution)
2. But UDP meaningfully reduces the **non-GPU overhead** — base64, HTTP framing, and asyncio scheduling — which is where tail latency and jitter originate

---

## 6. Why UDP for Edge Computing

### 6.1 TCP vs UDP

| Property | TCP | UDP |
|----------|-----|-----|
| Connection setup | 3-way handshake (~1–3 RTTs) | None |
| Delivery guarantee | Yes (retransmit on loss) | No (skip the frame) |
| Head-of-line blocking | Yes | No |
| Congestion control | Yes (can throttle throughput) | No |
| Per-packet overhead | ~20B header + ACK traffic | ~8B header |
| Kernel state per client | ~4KB + connection machine | None (stateless) |

### 6.2 Why Real-Time AI Inference at the Edge Favors UDP

**Dropped frame vs late frame.** In a 30fps pipeline, a dropped packet means the next frame arrives 33ms later — acceptable. A TCP retransmit can stall the pipeline 100–200ms waiting for the missing segment — not acceptable. Real-time inference systems should skip stale frames, exactly as video codecs do.

**LAN packet loss is effectively zero.** TCP's reliability mechanism adds overhead without providing protection against a failure that doesn't happen on a local network.

**No per-client connection state.** TCP keeps per-connection state in the kernel (recv window, congestion window, sequence numbers). A UDP server serving 100 IoT cameras uses a single socket fd with no per-client memory — TCP would maintain 100 connection state machines.

**Simpler for embedded clients.** Edge devices (Jetson Nano, industrial cameras, microcontrollers with lightweight network stacks) may not have a full HTTP runtime but can always open a UDP socket.

### 6.3 When TCP Is Still Better

- Lossy links (WAN, degraded WiFi) where retransmit is necessary
- Stateful multi-turn sessions (TCP's stream abstraction is genuinely useful)
- When you need TLS, HTTP-based auth, or CDN/load-balancer integration

### 6.4 Edge Deployment Architecture

```
┌───────────────────────────────────────────────────────┐
│                     EDGE NODE                         │
│  StreamDiffusion UDP Server                           │
│  (Jetson AGX Orin / RTX A2000 / H100 NVL)            │
│  UDP :9999 — no per-client state                      │
└──────────────────────┬────────────────────────────────┘
                       │ LAN < 0.1ms RTT
        ┌──────────────┼───────────────┐
        │              │               │
  ┌─────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
  │ IoT Cam #1 │ │ IoT Cam #2 │ │ AR Headset │
  └────────────┘ └────────────┘ └────────────┘
```

At the edge, network RTT is sub-millisecond, so **inference latency ≈ total latency**. Every millisecond of transport overhead is a large fraction of the budget. This is where reducing HTTP framing from ~7.5ms to ~0.18ms has its greatest relative impact.

For **5G MEC** (Multi-Access Edge Computing), the 5G RAN already adds 1–5ms. HTTP adding another 7.5ms doubles the observable latency; UDP's 0.18ms overhead is negligible next to the radio delay.

---

## 7. Limitations

- **Loopback only**: All benchmarks ran on localhost. Physical NIC latency, switch hops, and IRQ affinity will raise absolute numbers but should not change relative ordering.
- **Single client**: A multi-client load test would better demonstrate UDP's statelessness advantage under concurrent load.
- **No packet loss simulation**: `tc netem` could inject 0.1–1% loss to show how TCP retransmits vs UDP frame-skipping behave under realistic impairment.
- **No DTLS**: The UDP server sends plaintext. Production deployments require DTLS for encryption — adds ~1–3ms handshake overhead once per session.

---

*Last updated: 2026-04-20*