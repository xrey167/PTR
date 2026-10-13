"""Value-free pod identity, mirrored byte for byte from `ptr-types` (`pod_identity.rs`).

The Rust side owns the definition. This module computes the same `PodIdentity` and `ClusterKey`
from the same descriptor so the exporter can bind a representation to them without a round trip
through Rust. The two sides are held together by `identity_vectors` in the shared fixture
(`crates/ptr-router/tests/fixtures/dragonfly-expected.json`): the Rust test recomputes every vector
and the Python test checks them against this module.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

SCHEMA_VERSION = 1
_IDENTITY_DOMAIN = "ptr.pod.identity"
_CLUSTER_DOMAIN = "ptr.pod.cluster-key"
_IDENTITY_PREFIX = "pod:"


def put_text(out: bytearray, value: str) -> None:
    """Length-prefixed (u64 little endian) UTF-8, as `ptr-types` `put_text`."""
    data = value.encode("utf-8")
    out += struct.pack("<Q", len(data))
    out += data


@dataclass(frozen=True)
class PodIdentityDescriptor:
    """The hard semantics a pod identity is derived from (no artifact, generation or value)."""

    domain: str
    kind: str
    semantic_role: str
    predicate: str | None = None
    aliases: frozenset[str] = field(default_factory=frozenset)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported identity schema {self.schema_version}")
        for name in ("domain", "kind", "semantic_role"):
            if not getattr(self, name):
                raise ValueError(f"descriptor field {name} is empty")
        if self.predicate is not None and not self.predicate:
            raise ValueError("descriptor field predicate is empty")
        if any(not alias for alias in self.aliases):
            raise ValueError("descriptor contains an empty alias")

    def _hard_semantics(self, out: bytearray) -> None:
        out += struct.pack("<I", self.schema_version)
        put_text(out, self.domain)
        put_text(out, self.kind)
        if self.predicate is None:
            out += b"\x00"
        else:
            out += b"\x01"
            put_text(out, self.predicate)
        put_text(out, self.semantic_role)

    def cluster_key(self) -> bytes:
        """Groups pods by domain, kind, predicate and role; aliases are excluded."""
        out = bytearray()
        put_text(out, _CLUSTER_DOMAIN)
        self._hard_semantics(out)
        return hashlib.sha256(bytes(out)).digest()

    def identity(self) -> str:
        """Stable identity of the pod: `pod:` and the hex of the identity hash."""
        out = bytearray()
        put_text(out, _IDENTITY_DOMAIN)
        self._hard_semantics(out)
        # Rust keeps aliases in a BTreeSet<String>, ordered by UTF-8 bytes; Python orders str by
        # code point, which is the same order.
        aliases = sorted(self.aliases)
        out += struct.pack("<Q", len(aliases))
        for alias in aliases:
            put_text(out, alias)
        return _IDENTITY_PREFIX + hashlib.sha256(bytes(out)).hexdigest()

    def as_json(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "kind": self.kind,
            "predicate": self.predicate,
            "semantic_role": self.semantic_role,
            "aliases": sorted(self.aliases),
        }

    @classmethod
    def from_json(cls, data: dict) -> "PodIdentityDescriptor":
        return cls(
            schema_version=data["schema_version"],
            domain=data["domain"],
            kind=data["kind"],
            predicate=data["predicate"],
            semantic_role=data["semantic_role"],
            aliases=frozenset(data["aliases"]),
        )
