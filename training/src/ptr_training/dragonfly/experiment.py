"""Training step of experiment R015 (Dragonfly pod addressing): one synthetic world per seed.

`python3 -m ptr_training.dragonfly.experiment --seed S --out DIR` draws a world from the seed
(supplier pods with names, spelling variants and templated questions), trains and exports one
address per pod, and writes `world.json` for the Rust harness (`ptr-bench dragonfly-addressing S`,
which runs this module itself and consumes its output). Held-out questions use phrasings the
trainer never saw, so the experiment measures whether the address generalises beyond its training
templates, not whether it memorised them.

The encoder is the one the experiment's `config.toml` pins (`encoder_model`, `encoder_revision`).
While the pin is still a `must-be-pinned-...` placeholder, or when the pinned encoder cannot be
built (its optional extra is not installed), the run stops with exit status 2: a confirmatory run
never falls back to another encoder. `--smoke` selects the deterministic stand-in encoder
(`hashed-bow-64`) instead and marks the world, and so the harness output, as no evidence about any
language model.

The Python environment is part of the experiment. The Rust harness starts this module through
`uv run --locked --project training --extra dragonfly` for a confirmatory run, so the installed
packages are the ones `training/uv.lock` names. The module checks that again itself: a run without
`--smoke` stops with exit status 2 when the interpreter's installed `sentence-transformers`,
`torch`, `transformers` or `numpy` differ from the lock (or are missing), or when the interpreter
is not exactly the `python_version` the preregistration pins (the harness passes the same
version to `uv run --python`), and every run, smoke or
not, writes the interpreter version, those package versions, the locked versions and the SHA-256 of
`training/uv.lock` into `world.json` (`environment`). A smoke run only records the mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import sys
import tomllib
from pathlib import Path

from .encoder import Encoder, HashedBagOfWordsEncoder, l2_normalize
from .export import export_address
from .gate import round_f32
from .identity import PodIdentityDescriptor
from .train import PodTrainingSpec, TrainConfig, train_pod_address

ENCODER_DIMS = 64
POD_COUNT = 6
PROJECT = "demo"
NAMESPACE = "supply"
CONFIG = (
    Path(__file__).resolve().parents[4]
    / "experiments/runtime/R015-dragonfly-pod-addressing/config.toml"
)
PLACEHOLDER_PREFIX = "must-be-pinned"
LOCK = Path(__file__).resolve().parents[4] / "training/uv.lock"
#: The packages whose versions decide what a sentence encoder computes.
LOCKED_PACKAGES = ("sentence-transformers", "torch", "transformers", "numpy")


class EncoderUnavailable(Exception):
    """The encoder the preregistration names cannot be used; a confirmatory run stops."""


def python_matches(pin: str | None, version=None) -> bool:
    """Whether the running interpreter is exactly the pinned `X.Y.Z` (no range, no prefix)."""
    version = sys.version_info if version is None else version
    if not pin or not re.fullmatch(r"\d+\.\d+\.\d+", pin):
        return False
    return pin == f"{version[0]}.{version[1]}.{version[2]}"


def pinned_python(config: Path = CONFIG) -> str | None:
    """The interpreter version `config` pins (`python_version`), or `None` while it is unset or a
    `must-be-pinned-...` placeholder."""
    table = tomllib.loads(config.read_text(encoding="utf-8"))["preregistration"]
    pin = str(table.get("python_version", ""))
    return None if not pin or pin.startswith(PLACEHOLDER_PREFIX) else pin


def environment_report(
    lock: Path = LOCK,
    packages: tuple[str, ...] = LOCKED_PACKAGES,
    python_pin: str | None = None,
) -> dict:
    """The interpreter, the installed and the locked versions of `packages`, and the lock digest.

    `matches_lock` is true only when every package is installed at a version the lock names.
    """
    raw = lock.read_bytes()
    locked: dict[str, list[str]] = {}
    for package in tomllib.loads(raw.decode("utf-8")).get("package", []):
        locked.setdefault(package["name"], []).append(package["version"])
    report: dict = {}
    mismatches: list[str] = []
    for name in packages:
        try:
            installed: str | None = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        wanted = sorted(set(locked.get(name, [])))
        report[name] = {"installed": installed, "locked": wanted}
        if installed is None or installed not in wanted:
            mismatches.append(name)
    return {
        "python": sys.version.split()[0],
        "python_pinned": python_pin,
        "python_matches": python_matches(python_pin),
        "implementation": platform.python_implementation(),
        "uv_lock_sha256": hashlib.sha256(raw).hexdigest(),
        "packages": report,
        "matches_lock": not mismatches,
        "mismatches": mismatches,
    }


def pinned_encoder(config: Path = CONFIG) -> Encoder:
    """Build the encoder `config` pins, or raise [`EncoderUnavailable`] (never a stand-in)."""
    table = tomllib.loads(config.read_text(encoding="utf-8"))["preregistration"]
    model = str(table.get("encoder_model", ""))
    revision = str(table.get("encoder_revision", ""))
    if not model or not revision or model.startswith(PLACEHOLDER_PREFIX) or revision.startswith(
        PLACEHOLDER_PREFIX
    ):
        raise EncoderUnavailable(
            "the encoder is not pinned in the preregistration (encoder_model/encoder_revision); "
            "use --smoke for a run that is no evidence"
        )
    from .encoder import SentenceTransformerEncoder

    try:
        return SentenceTransformerEncoder(model, revision)
    except Exception as error:  # noqa: BLE001 - any failure to build the pinned encoder stops the run
        raise EncoderUnavailable(f"{type(error).__name__}: {error}") from error

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
        project=PROJECT,
        namespace=NAMESPACE,
        pod_id=pod["pod_id"],
        descriptor=descriptor,
        generation=1,
        positives=tuple(positives),
        negatives=tuple(negatives),
        negative_aliases=tuple(negative_aliases),
        description=f"Supplier {pod['name']} delivery and lead time",
    )


def build(
    seed: int,
    out_dir: Path,
    encoder: Encoder,
    *,
    steps: int = 300,
    smoke: bool = False,
    environment: dict | None = None,
) -> dict:
    """Train every pod of `seed` with `encoder`, export the artifacts and write `world.json`."""
    pods = draw_world(seed)
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
        "dims": encoder.dims,
        "project": PROJECT,
        "namespace": NAMESPACE,
        # A smoke world is produced with the stand-in encoder and is no evidence about any model.
        "smoke": smoke,
        "evidence": "none (stand-in encoder)" if smoke else "candidate",
        "environment": environment if environment is not None else environment_report(),
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
    parser.add_argument("--seed", type=int, help="the seed of the world (every run but --preflight)")
    parser.add_argument(
        "--out",
        type=Path,
        help="output directory (default: target/dragonfly-addressing/<seed>)",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the deterministic stand-in encoder; the world is marked as no evidence",
    )
    parser.add_argument("--config", type=Path, default=CONFIG, help="preregistration config.toml")
    parser.add_argument("--lock", type=Path, default=LOCK, help="the uv.lock the run must match")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="check the confirmatory environment (pinned encoder, interpreter, locked packages) "
        "and exit; builds no world and takes no seed",
    )
    arguments = parser.parse_args(argv)
    if arguments.preflight and (arguments.smoke or arguments.seed is not None):
        parser.error("--preflight takes neither --seed nor --smoke")
    if not arguments.preflight and arguments.seed is None:
        parser.error("--seed is required")
    if arguments.smoke:
        encoder: Encoder = HashedBagOfWordsEncoder(ENCODER_DIMS)
    else:
        try:
            encoder = pinned_encoder(arguments.config)
        except EncoderUnavailable as error:
            print(f"ptr_training.dragonfly.experiment: {error}", file=sys.stderr)
            return 2
    environment = environment_report(arguments.lock, python_pin=pinned_python(arguments.config))
    if not arguments.smoke and not environment["python_matches"]:
        print(
            "ptr_training.dragonfly.experiment: the interpreter is "
            f"{environment['python']}, the preregistration pins {environment['python_pinned']}; "
            "a confirmatory run needs exactly that version",
            file=sys.stderr,
        )
        return 2
    if not arguments.smoke and not environment["matches_lock"]:
        print(
            "ptr_training.dragonfly.experiment: the installed "
            f"{', '.join(environment['mismatches'])} differ from {arguments.lock.name}; "
            "run through `uv run --locked --project training --extra dragonfly`",
            file=sys.stderr,
        )
        return 2
    if arguments.preflight:
        # The pinned encoder must also produce a usable vector, not only load.
        try:
            probe = encoder.encode("preflight")
        except Exception as error:  # noqa: BLE001 - any failure of the model is a failed preflight
            print(f"ptr_training.dragonfly.experiment: the pinned encoder fails: {error}", file=sys.stderr)
            return 2
        if len(probe) != encoder.dims or not any(probe):
            print(
                "ptr_training.dragonfly.experiment: the pinned encoder returned an unusable vector",
                file=sys.stderr,
            )
            return 2
        print(
            "ptr_training.dragonfly.experiment: preflight passed "
            f"({encoder.encoder_id}@{encoder.encoder_revision}, python {environment['python']})",
            file=sys.stderr,
        )
        return 0
    out = arguments.out or Path("target/dragonfly-addressing") / str(arguments.seed)
    world = build(
        arguments.seed,
        out,
        encoder,
        steps=arguments.steps,
        smoke=arguments.smoke,
        environment=environment,
    )
    activated = sum(1 for pod in world["pods"] if pod["status"] == "activated")
    print(f"seed {arguments.seed}: {activated}/{len(world['pods'])} pods activated -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
