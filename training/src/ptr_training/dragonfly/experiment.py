"""Training step of experiment R015 (Dragonfly pod addressing): one synthetic world per seed.

`python3 -m ptr_training.dragonfly.experiment --seed S --out DIR` draws a world from the seed
(supplier pods with names, spelling variants and templated questions), trains and exports one
address per pod over the deterministic hashed encoder, and writes `world.json` for the Rust
harness (`ptr-bench dragonfly-addressing S`), which routes held-out questions with the learned
router and with two baselines. Held-out questions use phrasings the trainer never saw, so the
experiment measures whether the address generalises beyond its training templates, not whether it
memorised them. The encoder is a stand-in that needs no download; the claim under test is the
routing mechanism, not any language model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .encoder import HashedBagOfWordsEncoder, l2_normalize
from .export import export_address
from .gate import round_f32
from .identity import PodIdentityDescriptor
from .train import PodTrainingSpec, TrainConfig, train_pod_address

ENCODER_DIMS = 64
POD_COUNT = 6

_ONSETS = ["br", "ch", "dr", "fl", "gr", "kl", "kr", "pl", "sch", "st", "tr", "vol", "wen", "zi"]
_VOWELS = ["a", "e", "i", "o", "u", "ei", "au"]
_CODAS = ["n", "r", "l", "m", "k", "t", "ng", "ld", "rt", "s"]
_FORMS = ["GmbH", "AG", "Werke", "Logistik", "Trans", "KG"]

TRAIN_TEMPLATES = [
    "How long is the delivery time of {name}?",
    "Is {name} late with the shipment?",
    "What did {name} quote for the lead time?",
    "Delay of the procurement order at {name}",
    "Which delivery date did {name} confirm?",
    "Status of the open purchase order at {name}",
]
HELD_OUT_TEMPLATES = [
    "When will {name} deliver the parts?",
    "Any shipping problems with {name} this week?",
    "{name} lead time for the next order",
]
DISTRACTORS = [
    "Weather forecast for tomorrow in Hamburg",
    "How do I reset my password?",
    "Summarize the meeting notes from Monday",
    "What is the capital of France?",
]


def _word(rng: random.Random) -> str:
    return (
        rng.choice(_ONSETS)
        + rng.choice(_VOWELS)
        + rng.choice(_CODAS)
        + rng.choice(_VOWELS)
        + rng.choice(_CODAS)
    ).capitalize()


def _variant(name: str, rng: random.Random) -> str:
    """A near spelling of `name`: one interior letter doubled or dropped."""
    index = rng.randrange(1, len(name) - 1)
    if rng.random() < 0.5:
        return name[:index] + name[index] + name[index:]
    return name[:index] + name[index + 1 :]


def draw_world(seed: int) -> list[dict]:
    """The pods of one seed: distinct names, a trusted second spelling and a form suffix."""
    rng = random.Random(seed)
    pods: list[dict] = []
    seen: set[str] = set()
    while len(pods) < POD_COUNT:
        base = _word(rng)
        if base.casefold() in seen:
            continue
        variant = _variant(base, rng)
        if variant.casefold() in seen or variant.casefold() == base.casefold():
            continue
        seen.update({base.casefold(), variant.casefold()})
        form = rng.choice(_FORMS)
        pods.append(
            {
                "pod_id": f"supplier-{base.lower()}",
                "name": f"{base} {form}",
                "aliases": [f"{base} {form}", f"{variant} {form}"],
            }
        )
    return pods


def _spec(pod: dict, others: list[dict]) -> PodTrainingSpec:
    descriptor = PodIdentityDescriptor(
        domain="supply",
        kind="Retrieval",
        predicate="lead_time",
        semantic_role=pod["pod_id"],
        aliases=frozenset(pod["aliases"]),
    )
    positives = [t.format(name=pod["name"]) for t in TRAIN_TEMPLATES]
    negatives = [t.format(name=other["name"]) for other in others for t in TRAIN_TEMPLATES]
    negative_aliases = [a for other in others for a in other["aliases"]]
    return PodTrainingSpec(
        pod_id=pod["pod_id"],
        descriptor=descriptor,
        generation=1,
        positives=tuple(positives),
        negatives=tuple(negatives),
        negative_aliases=tuple(negative_aliases),
        description=f"Supplier {pod['name']} delivery and lead time",
    )


def build(seed: int, out_dir: Path, *, steps: int = 300) -> dict:
    """Train every pod of `seed`, export the artifacts and write `world.json`."""
    pods = draw_world(seed)
    encoder = HashedBagOfWordsEncoder(ENCODER_DIMS)
    entries = []
    for pod in pods:
        others = [p for p in pods if p is not pod]
        spec = _spec(pod, others)
        trained = train_pod_address(spec, encoder, TrainConfig(steps=steps))
        json_path, _ = export_address(trained, out_dir)
        description = [
            round_f32(v) for v in l2_normalize(encoder.encode(spec.description or pod["name"]))
        ]
        entries.append(
            {
                "pod_id": pod["pod_id"],
                "manifest": json_path.name,
                "status": trained.gate.status,
                "description_vector": description,
            }
        )

    probes = []
    for pod in pods:
        for template in HELD_OUT_TEMPLATES:
            for alias in pod["aliases"]:
                text = template.format(name=alias)
                probes.append({"text": text, "label": pod["pod_id"], "kind": "held-out"})
    for text in DISTRACTORS:
        probes.append({"text": text, "label": None, "kind": "distractor"})
    unseen = random.Random(seed + 1).sample(_ONSETS, 3)
    for onset in unseen:
        name = f"{onset.capitalize()}ovia {_FORMS[0]}"
        probes.append(
            {
                "text": HELD_OUT_TEMPLATES[0].format(name=name),
                "label": None,
                "kind": "unregistered-supplier",
            }
        )
    for probe in probes:
        probe["vector"] = [round_f32(v) for v in l2_normalize(encoder.encode(probe["text"]))]

    world = {
        "seed": seed,
        "encoder": {"id": encoder.encoder_id, "revision": encoder.encoder_revision},
        "dims": ENCODER_DIMS,
        "pods": entries,
        "probes": probes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "world.json").write_text(
        json.dumps(world, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return world


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        help="output directory (default: target/dragonfly-addressing/<seed>)",
    )
    parser.add_argument("--steps", type=int, default=300)
    arguments = parser.parse_args(argv)
    out = arguments.out or Path("target/dragonfly-addressing") / str(arguments.seed)
    world = build(arguments.seed, out, steps=arguments.steps)
    activated = sum(1 for pod in world["pods"] if pod["status"] == "activated")
    print(f"seed {arguments.seed}: {activated}/{len(world['pods'])} pods activated -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
