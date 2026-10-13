"""Export a trained address in the format `ptr-router` reads (`PodRepresentation::from_artifact`).

Two files per pod, named after the pod identity with `:` replaced by `-` (a colon is not a legal
file name character everywhere):

* `dragonfly-<identity>.bin`: `dims` little-endian float32 values of `z`, then one float32 bias.
* `dragonfly-<identity>.json`: the manifest. `weights_sha256` is the SHA-256 of the `.bin` file and
  `digest` covers every field plus the weights, so any later edit is detected on load.

The digest layout is the one in `crates/ptr-router/src/learned.rs` (`digest_of`); the committed
fixture and its two tests (Rust and Python) keep the implementations together.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from .gate import STATUS_ACTIVATED
from .identity import put_text
from .train import TrainedAddress

SCHEMA = 1
KIND = "dragonfly-pod-address"
DIGEST_DOMAIN = "ptr.dragonfly.artifact.v1"


def weights_bytes(z: tuple[float, ...] | list[float], bias: float) -> bytes:
    """`z` then the bias, little-endian float32."""
    return struct.pack(f"<{len(z) + 1}f", *z, bias)


def artifact_digest(manifest: dict, weights: bytes) -> bytes:
    """SHA-256 over every manifest field and the weights, in the order `digest_of` hashes them."""
    gate = manifest["gate"]
    scoring = manifest["scoring"]
    out = bytearray()
    put_text(out, DIGEST_DOMAIN)
    out += struct.pack("<I", manifest["schema"])
    put_text(out, manifest["encoder"]["id"])
    put_text(out, manifest["encoder"]["revision"])
    out += struct.pack("<I", manifest["dims"])
    out += struct.pack("<Q", manifest["generation"])
    put_text(out, manifest["pod_id"])
    put_text(out, manifest["pod_identity"])
    out += bytes.fromhex(manifest["cluster_key"])
    out += struct.pack("<f", scoring["cosine_weight"])
    out += struct.pack("<f", scoring["metadata_weight"])
    out += struct.pack("<B", 1 if gate["status"] == STATUS_ACTIVATED else 0)
    out += struct.pack(
        "<HHHH",
        gate["alias_threshold_permille"],
        gate["alias_min_permille"],
        gate["negative_threshold_permille"],
        gate["negative_reject_permille"],
    )
    out += struct.pack("<II", gate["alias_count"], gate["negative_count"])
    out += weights
    return hashlib.sha256(bytes(out)).digest()


def build_manifest(trained: TrainedAddress, weights: bytes) -> dict:
    descriptor = trained.spec.descriptor
    gate = trained.gate
    manifest = {
        "schema": SCHEMA,
        "kind": KIND,
        "encoder": {"id": trained.encoder_id, "revision": trained.encoder_revision},
        "dims": trained.dims,
        "generation": trained.spec.generation,
        "pod_id": trained.spec.pod_id,
        "pod_identity": descriptor.identity(),
        "cluster_key": descriptor.cluster_key().hex(),
        "scoring": {
            "cosine_weight": trained.cosine_weight,
            "metadata_weight": trained.metadata_weight,
        },
        "gate": {
            "status": gate.status,
            "alias_threshold_permille": gate.alias_threshold_permille,
            "alias_min_permille": gate.alias_min_permille,
            "negative_threshold_permille": gate.negative_threshold_permille,
            "negative_reject_permille": gate.negative_reject_permille,
            "alias_count": gate.alias_count,
            "negative_count": gate.negative_count,
        },
        "weights_file": "",
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
    }
    return manifest


def artifact_stem(pod_identity: str) -> str:
    return "dragonfly-" + pod_identity.replace(":", "-")


def export_address(trained: TrainedAddress, out_dir: Path) -> tuple[Path, Path]:
    """Write the manifest and weights of `trained`; returns `(manifest_path, weights_path)`.

    A not-activated address is exported too (so the refusal can be tested end to end), with
    status `not-activated`; `ptr-router` never loads it.
    """
    weights = weights_bytes(trained.z, trained.bias)
    manifest = build_manifest(trained, weights)
    stem = artifact_stem(manifest["pod_identity"])
    manifest["weights_file"] = f"{stem}.bin"
    manifest["digest"] = artifact_digest(manifest, weights).hex()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    bin_path = out_dir / f"{stem}.bin"
    bin_path.write_bytes(weights)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_path, bin_path


def verify_artifact(json_path: Path) -> dict:
    """Re-read an exported pair and check both digests; returns the manifest."""
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    weights = (json_path.parent / manifest["weights_file"]).read_bytes()
    if len(weights) != (manifest["dims"] + 1) * 4:
        raise ValueError("weights file has the wrong length")
    if hashlib.sha256(weights).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("weights digest mismatch")
    if artifact_digest(manifest, weights).hex() != manifest["digest"]:
        raise ValueError("artifact digest mismatch")
    return manifest
