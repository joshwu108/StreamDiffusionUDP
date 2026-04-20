"""
Binary packet protocol for StreamDiffusion UDP transport.

Request (client → server):
  [b'SDREQ'] [seq_id: u32] [timestamp_ns: u64] [prompt: bytes]

Response (server → client, one packet per chunk):
  [b'SDRES'] [seq_id: u32] [client_ts_ns: u64]
  [server_recv_ns: u64] [server_send_ns: u64]
  [chunk_idx: u16] [total_chunks: u16] [jpeg_data: bytes]
"""

import struct
from dataclasses import dataclass

REQ_MAGIC = b"SDREQ"
RES_MAGIC = b"SDRES"

# Request: magic(5) + seq_id(4) + timestamp_ns(8) = 17 bytes fixed header
_REQ_HEADER_FMT = "!5sIQ"
_REQ_HEADER_SIZE = struct.calcsize(_REQ_HEADER_FMT)  # 17

# Response: magic(5) + seq_id(4) + client_ts(8) + srv_recv(8) + srv_send(8)
#           + chunk_idx(2) + total_chunks(2) = 37 bytes fixed header
_RES_HEADER_FMT = "!5sIQQQHH"
_RES_HEADER_SIZE = struct.calcsize(_RES_HEADER_FMT)  # 37

# Stay well under UDP's 65507-byte limit
MAX_CHUNK_BYTES = 60_000


@dataclass
class RequestPacket:
    seq_id: int
    timestamp_ns: int
    prompt: str

    def pack(self) -> bytes:
        prompt_bytes = self.prompt.encode("utf-8")
        header = struct.pack(_REQ_HEADER_FMT, REQ_MAGIC, self.seq_id, self.timestamp_ns)
        return header + prompt_bytes

    @staticmethod
    def unpack(data: bytes) -> "RequestPacket":
        if len(data) < _REQ_HEADER_SIZE:
            raise ValueError("Packet too short for request header")
        magic, seq_id, timestamp_ns = struct.unpack_from(_REQ_HEADER_FMT, data)
        if magic != REQ_MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        prompt = data[_REQ_HEADER_SIZE:].decode("utf-8")
        return RequestPacket(seq_id=seq_id, timestamp_ns=timestamp_ns, prompt=prompt)


@dataclass
class ResponsePacket:
    seq_id: int
    client_ts_ns: int
    server_recv_ns: int
    server_send_ns: int
    chunk_idx: int
    total_chunks: int
    data: bytes

    def pack(self) -> bytes:
        header = struct.pack(
            _RES_HEADER_FMT,
            RES_MAGIC,
            self.seq_id,
            self.client_ts_ns,
            self.server_recv_ns,
            self.server_send_ns,
            self.chunk_idx,
            self.total_chunks,
        )
        return header + self.data

    @staticmethod
    def unpack(data: bytes) -> "ResponsePacket":
        if len(data) < _RES_HEADER_SIZE:
            raise ValueError("Packet too short for response header")
        (magic, seq_id, client_ts_ns, server_recv_ns, server_send_ns, chunk_idx, total_chunks) = struct.unpack_from(_RES_HEADER_FMT, data)
        if magic != RES_MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        payload = data[_RES_HEADER_SIZE:]
        return ResponsePacket(
            seq_id=seq_id,
            client_ts_ns=client_ts_ns,
            server_recv_ns=server_recv_ns,
            server_send_ns=server_send_ns,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            data=payload,
        )


def fragment_jpeg(
    jpeg_bytes: bytes,
    seq_id: int,
    client_ts_ns: int,
    server_recv_ns: int,
    server_send_ns: int,
) -> list[bytes]:
    """Split jpeg_bytes into UDP-safe chunks and return list of packed packets."""
    chunks = [jpeg_bytes[i : i + MAX_CHUNK_BYTES] for i in range(0, max(len(jpeg_bytes), 1), MAX_CHUNK_BYTES)]
    total = len(chunks)
    return [
        ResponsePacket(
            seq_id=seq_id,
            client_ts_ns=client_ts_ns,
            server_recv_ns=server_recv_ns,
            server_send_ns=server_send_ns,
            chunk_idx=idx,
            total_chunks=total,
            data=chunk,
        ).pack()
        for idx, chunk in enumerate(chunks)
    ]
