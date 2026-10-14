"""The synthetic world behind the shared Python/Rust fixture.

`build_world(out_dir)` trains three supplier pods over the deterministic hashed encoder and writes
their artifacts plus `dragonfly-expected.json`: identity vectors, each pod's gate status and probe
queries with the scores this package computes. The Rust test
(`crates/ptr-router/tests/dragonfly_artifact.rs`) loads the very same files and must reproduce
every number, which is what keeps the export format and the scoring formula bit-compatible.

Regenerate the committed copy with
`PYTHONPATH=training/src python3 -m ptr_training.dragonfly.fixture crates/ptr-router/tests/fixtures`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .encoder import HashedBagOfWordsEncoder, l2_normalize
from .export import artifact_stem, export_address
from .gate import round_f32, score
from .identity import PodIdentityDescriptor
from .train import PodTrainingSpec, TrainConfig, TrainedAddress, train_pod_address

ENCODER_DIMS = 32
PROJECT = "demo"

_QUESTIONS = {
    "mueller": [
        "How long is the delivery time of {name}?",
        "Is {name} late with the X12 shipment?",
        "What did {name} quote for the lead time?",
        "Delay of the procurement order at {name}",
        "Which delivery date did {name} confirm?",
    ],
    "schmidt": [
        "How long is the delivery time of {name}?",
        "Is {name} late with the X12 shipment?",
        "What did {name} quote for the lead time?",
        "Delay of the procurement order at {name}",
        "Which delivery date did {name} confirm?",
    ],
    "kowalski": [
        "How long is the delivery time of {name}?",
        "Is {name} late with the X12 shipment?",
        "What did {name} quote for the lead time?",
        "Delay of the procurement order at {name}",
        "Which delivery date did {name} confirm?",
    ],
}

_ALIASES = {
    "mueller": ["Mueller GmbH", "Muller Werke"],
    "schmidt": ["Schmidt AG", "Schmitt Logistik"],
    "kowalski": ["Kowalski Sp", "Kowalsky Trans"],
}
_NAMES = {"mueller": "Mueller GmbH", "schmidt": "Schmidt AG", "kowalski": "Kowalski Sp"}


def _descriptor(key: str) -> PodIdentityDescriptor:
    return PodIdentityDescriptor(
        domain="supply",
        kind="Retrieval",
        predicate="lead_time",
        semantic_role=f"supplier-{key}",
        aliases=frozenset(_ALIASES[key]),
    )


def _spec(key: str) -> PodTrainingSpec:
    positives = [q.format(name=_NAMES[key]) for q in _QUESTIONS[key]]
    negatives = [
        q.format(name=_NAMES[other])
        for other in _NAMES
        if other != key
        for q in _QUESTIONS[other]
    ]
    negative_aliases = [a for other in _ALIASES if other != key for a in _ALIASES[other]]
    return PodTrainingSpec(
        project=PROJECT,
        pod_id=f"supplier-{key}",
        descriptor=_descriptor(key),
        generation=3,
        positives=tuple(positives),
        negatives=tuple(negatives),
        negative_aliases=tuple(negative_aliases),
    )


PROBES = [
    "How long does Mueller GmbH need to deliver?",
    "Is Schmidt AG late with the shipment?",
    "Lead time at Kowalski Sp for the X12 part",
    "Which delivery date did Muller Werke confirm?",
    "Weather forecast for tomorrow",
]


def build_world(out_dir: Path) -> dict:
    """Train, export and describe the three pods; returns the expected-values document."""
    encoder = HashedBagOfWordsEncoder(ENCODER_DIMS)
    trained: dict[str, TrainedAddress] = {}
    trained["mueller"] = train_pod_address(_spec("mueller"), encoder)
    trained["schmidt"] = train_pod_address(_spec("schmidt"), encoder)
    # A pod whose training never moved: its gate does not pass, so Rust must refuse to load it.
    trained["kowalski"] = train_pod_address(
        _spec("kowalski"), encoder, TrainConfig(steps=1, learning_rate=0.0)
    )

    pods = []
    for key, address in trained.items():
        json_path, _ = export_address(address, out_dir)
        pods.append(
            {
                "project": address.spec.project,
                "pod_id": address.spec.pod_id,
                "pod_identity": address.spec.descriptor.identity(),
                "cluster_key": address.spec.descriptor.cluster_key().hex(),
                "generation": address.spec.generation,
                "status": address.gate.status,
                "manifest": json_path.name,
                "weights": artifact_stem(address.spec.descriptor.identity()) + ".bin",
            }
        )

    probes = []
    for text in PROBES:
        vector = [round_f32(v) for v in l2_normalize(encoder.encode(text))]
        probes.append(
            {
                "text": text,
                "vector": vector,
                "scores": {
                    address.spec.pod_id: score(
                        vector,
                        address.z,
                        address.bias,
                        cosine_weight=address.cosine_weight,
                        metadata_weight=address.metadata_weight,
                    )
                    for address in trained.values()
                },
            }
        )

    identity_vectors = [
        {
            "descriptor": descriptor.as_json(),
            "identity": descriptor.identity(),
            "cluster_key": descriptor.cluster_key().hex(),
        }
        for descriptor in [
            _descriptor("mueller"),
            _descriptor("schmidt"),
            PodIdentityDescriptor(
                domain="d",
                kind="K",
                semantic_role="r",
                predicate=None,
                aliases=frozenset(),
            ),
            PodIdentityDescriptor(
                domain="Straße",
                kind="Memory",
                semantic_role="rôle",
                predicate="größe",
                aliases=frozenset({"ß", "alias", "Äpfel"}),
            ),
        ]
    ]

    expected = {
        "encoder": {"id": encoder.encoder_id, "revision": encoder.encoder_revision},
        "dims": ENCODER_DIMS,
        "pods": pods,
        "probes": probes,
        "identity_vectors": identity_vectors,
    }
    (out_dir / "dragonfly-expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return expected


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: python -m ptr_training.dragonfly.fixture OUT_DIR", file=sys.stderr)
        return 2
    build_world(Path(arguments[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
