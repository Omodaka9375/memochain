"""Fixed-size memory state: the 22 KB artifact.

11 heads x 32 keys x 32 values x int16 = 22,528 bytes, matching the Mina
protocol's constant-size chain figure. Each head is a fast-weight
associative matrix in Q14 fixed point (value = int / 2^14). The state
carries a full SHA-256 digest that acts as its block hash: any change to
the bytes changes the digest, which makes tampering detectable.

Exports are wrapped in a versioned Capsule envelope so old saved states
can never be silently misread after the format evolves.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

N_HEADS = 11
KEY_DIM = 32
VAL_DIM = 32
HEAD_PARAMS = KEY_DIM * VAL_DIM          # 1,024 per head
STATE_PARAMS = N_HEADS * HEAD_PARAMS     # 11,264 parameters
STATE_BYTES = STATE_PARAMS * 2           # 22,528 bytes ~= 22 KB

assert STATE_BYTES == 22_528

MAGIC = b"MCN1"
FORMAT_VERSION = 3  # v1: state; v2: + hot tail (int16 704 docs); v3: hot kind byte
# v1 header: magic, version, step, chain head, payload hash
_HEADER_STRUCT = struct.Struct("<4sHQ32s32s")
_HEADER_SIZE = _HEADER_STRUCT.size       # 78 bytes
# v2 adds: hot_count (u32), hot_dim (u32); payload hash covers state + hot bytes
_HEADER_V2_STRUCT = struct.Struct("<4sHQ32s32sII")
_HEADER_V2_SIZE = _HEADER_V2_STRUCT.size  # 86 bytes
# v3 adds: hot_kind (u8) after hot_dim; 0 = raw int16 704-dim docs,
# 1 = normalized float16 embeddings (hot_dim from header, usually 2560)
_HEADER_V3_STRUCT = struct.Struct("<4sHQ32s32sIIB")
_HEADER_V3_SIZE = _HEADER_V3_STRUCT.size  # 87 bytes

HOT_DOC_DIM = 2 * N_HEADS * KEY_DIM       # 704: doc space of the integer fold
HOT_KIND_RAW = 0   # int16 704-dim integer-fold docs
HOT_KIND_EMBED = 1  # float16 embeddings (student world, dim from header)


class MemoryState:
    """A fixed-size memory state. Never grows."""

    __slots__ = ("tokens",)

    def __init__(self, tokens: np.ndarray) -> None:
        if tokens.shape != (N_HEADS, KEY_DIM, VAL_DIM):
            raise ValueError(
                f"expected shape ({N_HEADS}, {KEY_DIM}, {VAL_DIM}), got {tokens.shape}"
            )
        self.tokens = np.ascontiguousarray(tokens.astype(np.int16))

    @classmethod
    def zero(cls) -> MemoryState:
        return cls(np.zeros((N_HEADS, KEY_DIM, VAL_DIM), dtype=np.int16))

    @classmethod
    def from_bytes(cls, raw: bytes) -> MemoryState:
        if len(raw) != STATE_BYTES:
            raise ValueError(f"expected {STATE_BYTES} bytes, got {len(raw)}")
        return cls(np.frombuffer(raw, dtype=np.int16).reshape(N_HEADS, KEY_DIM, VAL_DIM))

    def to_bytes(self) -> bytes:
        return self.tokens.tobytes()

    @property
    def nbytes(self) -> int:
        return int(self.tokens.nbytes)

    def digest(self) -> str:
        """Block hash: full 256-bit SHA-256 over the raw state bytes."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"MemoryState(heads={N_HEADS}x{KEY_DIM}x{VAL_DIM}, "
            f"nbytes={self.nbytes}, digest={self.digest()[:16]}...)"
        )


@dataclass(frozen=True)
class Capsule:
    """Versioned export envelope around a memory state.

    v1 layout: magic(4) | version(u16) | step(u64) | chain head(32) |
    payload sha256(32) | payload (22,528 bytes).

    v2 layout: same header plus hot_count(u32) | hot_dim(u32), and after
    the payload, hot_count raw document embeddings of hot_dim int16 values
    each. The payload hash covers state + hot bytes, so a bit flip anywhere
    in the file fails loudly. v1 files (state only) still load.

    v3 adds hot_kind(u8): 0 = raw int16 integer-fold docs (v2 semantics),
    1 = float16 embeddings from the student world (dim = hot_dim, usually
    2560). v2 and v1 files still load.
    """

    state: MemoryState
    step: int
    head: str  # receipt-chain head, 64 hex chars
    hot: tuple[np.ndarray, ...] = ()  # hot-tail entries (int16 or float16)
    hot_kind: int = HOT_KIND_RAW

    def to_bytes(self) -> bytes:
        payload = self.state.to_bytes()
        head = bytes.fromhex(self.head) if self.head else bytes(32)
        if len(head) != 32:
            raise ValueError(f"chain head must be empty or 32 bytes (64 hex chars), got {len(head)}")
        if self.hot:
            if self.hot_kind == HOT_KIND_RAW:
                hot_arrays = [np.ascontiguousarray(h, dtype=np.int16).reshape(-1) for h in self.hot]
                if any(h.shape[0] != HOT_DOC_DIM for h in hot_arrays):
                    raise ValueError(f"raw hot entries must have {HOT_DOC_DIM} int16 values")
                hot_dim = HOT_DOC_DIM
                dt = np.int16
            elif self.hot_kind == HOT_KIND_EMBED:
                hot_arrays = [np.ascontiguousarray(h, dtype=np.float16).reshape(-1) for h in self.hot]
                dims = {h.shape[0] for h in hot_arrays}
                if len(dims) != 1:
                    raise ValueError(f"embedding hot entries must share one dim, got {dims}")
                hot_dim = dims.pop()
                dt = np.float16
            else:
                raise ValueError(f"unknown hot_kind {self.hot_kind}")
            hot_bytes = b"".join(h.astype(dt).tobytes() for h in hot_arrays)
            full = (
                _HEADER_V3_STRUCT.pack(
                    MAGIC, FORMAT_VERSION, self.step, head,
                    hashlib.sha256(payload + hot_bytes).digest(),
                    len(hot_arrays), hot_dim, self.hot_kind,
                )
                + payload
                + hot_bytes
            )
            return full
        header = _HEADER_STRUCT.pack(
            MAGIC, 1, self.step, head, hashlib.sha256(payload).digest()
        )
        return header + payload

    @classmethod
    def from_bytes(cls, raw: bytes) -> Capsule:
        if len(raw) < _HEADER_SIZE + STATE_BYTES:
            raise ValueError(f"capsule too small: {len(raw)} bytes")
        magic, version, step, head, payload_hash = _HEADER_STRUCT.unpack(
            raw[:_HEADER_SIZE]
        )
        if magic != MAGIC:
            raise ValueError("not a MemoChain capsule (bad magic)")

        if version == 3:
            (_m, _v, _s, _h, _p, hot_count, hot_dim, hot_kind) = _HEADER_V3_STRUCT.unpack(
                raw[:_HEADER_V3_SIZE]
            )
            item_bytes = hot_dim * (2 if hot_kind in (HOT_KIND_RAW, HOT_KIND_EMBED) else 1)
            hot_bytes_len = hot_count * item_bytes
            expected = _HEADER_V3_SIZE + STATE_BYTES + hot_bytes_len
            if len(raw) != expected:
                raise ValueError(
                    f"capsule size mismatch: expected {expected}, got {len(raw)}"
                )
            payload = raw[_HEADER_V3_SIZE : _HEADER_V3_SIZE + STATE_BYTES]
            hot_raw = raw[_HEADER_V3_SIZE + STATE_BYTES :]
            if hashlib.sha256(payload + hot_raw).digest() != payload_hash:
                raise ValueError("capsule payload hash mismatch (corrupted or tampered)")
            if hot_kind == HOT_KIND_RAW:
                hot_flat = np.frombuffer(hot_raw, dtype=np.int16).reshape(hot_count, hot_dim)
            elif hot_kind == HOT_KIND_EMBED:
                hot_flat = np.frombuffer(hot_raw, dtype=np.float16).reshape(hot_count, hot_dim)
            else:
                raise ValueError(f"unknown hot_kind {hot_kind}")
            return cls(
                state=MemoryState.from_bytes(payload),
                step=step,
                head=head.hex() if head != bytes(32) else "",
                hot=tuple(hot_flat[i].copy() for i in range(hot_count)),
                hot_kind=hot_kind,
            )

        if version == 2:
            (_m, _v, _s, _h, _p, hot_count, hot_dim) = _HEADER_V2_STRUCT.unpack(
                raw[:_HEADER_V2_SIZE]
            )
            hot_bytes_len = hot_count * hot_dim * 2
            expected = _HEADER_V2_SIZE + STATE_BYTES + hot_bytes_len
            if len(raw) != expected:
                raise ValueError(
                    f"capsule size mismatch: expected {expected}, got {len(raw)}"
                )
            payload = raw[_HEADER_V2_SIZE : _HEADER_V2_SIZE + STATE_BYTES]
            hot_raw = raw[_HEADER_V2_SIZE + STATE_BYTES :]
            if hashlib.sha256(payload + hot_raw).digest() != payload_hash:
                raise ValueError("capsule payload hash mismatch (corrupted or tampered)")
            hot_flat = np.frombuffer(hot_raw, dtype=np.int16).reshape(hot_count, hot_dim)
            return cls(
                state=MemoryState.from_bytes(payload),
                step=step,
                head=head.hex() if head != bytes(32) else "",
                hot=tuple(hot_flat[i].copy() for i in range(hot_count)),
                hot_kind=HOT_KIND_RAW,
            )

        if version != 1:
            raise ValueError(f"unsupported capsule format version {version}")
        expected = _HEADER_SIZE + STATE_BYTES
        if len(raw) != expected:
            raise ValueError(
                f"capsule size mismatch: expected {expected}, got {len(raw)}"
            )
        payload = raw[_HEADER_SIZE:]
        if hashlib.sha256(payload).digest() != payload_hash:
            raise ValueError("capsule payload hash mismatch (corrupted or tampered)")
        return cls(
            state=MemoryState.from_bytes(payload),
            step=step,
            head=head.hex() if head != bytes(32) else "",
        )
