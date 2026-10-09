"""Shared helpers: digests, the pinned token regex, a self-contained PRNG, f16 I/O.

Pure standard library on purpose. Everything the unit tests and the manifest verifier
need runs without torch, so CI can check the committed artifacts without the model.
"""

from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The token-count rule shared with the Rust harnesses (M008 design): a run of ASCII
# alphanumerics or one other non-space character. Pool text is restricted to printable
# ASCII so that Python's Unicode-aware `\s` and Rust's ASCII classes cannot disagree.
TOKEN_PATTERN = r"[A-Za-z0-9]+|[^\sA-Za-z0-9]"
_TOKEN = re.compile(TOKEN_PATTERN)
_PRINTABLE_ASCII = re.compile(r"[ -~]+")


def count_tokens(text: str) -> int:
    if not _PRINTABLE_ASCII.fullmatch(text):
        raise ValueError(f"text is not printable ASCII: {text!r}")
    return len(_TOKEN.findall(text))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SplitMix64:
    """SplitMix64: trivial to reimplement bit-exactly in any language."""

    MASK = (1 << 64) - 1

    def __init__(self, salt: str):
        self.state = int.from_bytes(hashlib.sha256(salt.encode("utf-8")).digest()[:8], "little")

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def below(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        # Rejection sampling keeps the draw unbiased.
        limit = self.MASK + 1 - ((self.MASK + 1) % bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound

    def shuffle(self, items: list) -> None:
        for index in range(len(items) - 1, 0, -1):
            other = self.below(index + 1)
            items[index], items[other] = items[other], items[index]

    def sample(self, items: list, count: int) -> list:
        if count > len(items):
            raise ValueError("sample larger than population")
        pool = list(items)
        self.shuffle(pool)
        return pool[:count]


def read_f16_rows(path: Path, dim: int) -> list[tuple[float, ...]]:
    """Decode a little-endian IEEE half matrix; used by tests and the drivers' tooling."""
    data = Path(path).read_bytes()
    row_bytes = dim * 2
    if len(data) % row_bytes:
        raise ValueError(f"{path}: size {len(data)} is not a multiple of {row_bytes}")
    rows = []
    for offset in range(0, len(data), row_bytes):
        rows.append(struct.unpack(f"<{dim}e", data[offset : offset + row_bytes]))
    return rows
