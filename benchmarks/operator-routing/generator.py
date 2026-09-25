#!/usr/bin/env python3
"""Operator-routing v1: the deterministic generator (generator_version 1).

Standard library only. Every byte of every split is a function of the benchmark
seed, the split name and the example index, through the splitmix64 stream
defined below; Python's `random` module is not used anywhere in this file.

    python3 benchmarks/operator-routing/generator.py            # write every split
    python3 benchmarks/operator-routing/generator.py --check    # regenerate, compare with splits.lock.json
    python3 benchmarks/operator-routing/generator.py --update-lock   # write the splits and re-pin the lock
    python3 benchmarks/operator-routing/generator.py --write-sample  # datasets/samples/operator_route_v1.jsonl

README.md in this directory is the specification this file implements: the PRNG,
the sampling order, the label rule and its constants, the counterfactual tags, the
JSONL and TSV formats and the digests. `score.py` re-derives every label with an
implementation that shares no code with this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = ROOT / "datasets/generated/operator_routing_v1"
LOCK_PATH = HERE / "splits.lock.json"
SAMPLE_PATH = ROOT / "datasets/samples/operator_route_v1.jsonl"
CODEBOOK_PATH = ROOT / "datasets/generated/codebook.json"

GENERATOR_VERSION = 1
BENCHMARK_SEED = 20260925
# The codebook this generator's integer codes were read from. A different
# fingerprint means the kernel's code assignment moved; the generator refuses to
# run rather than emit codes that mean something else.
CODEBOOK_FINGERPRINT = "2b6f8175a7a7bb648910e7acb17dcfcff5149834f2fe87355c5d2c4524c113a3"
CODEBOOK_VERSION = 1

# --------------------------------------------------------------------------- PRNG

MASK64 = (1 << 64) - 1
GOLDEN_GAMMA = 0x9E3779B97F4A7C15
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
TWO_POW_53 = 9007199254740992.0


def mix64(z: int) -> int:
    """The splitmix64 finalizer of crates/ptr-types/src/slot_encoding.rs."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def splitmix64(x: int) -> int:
    """One output of a splitmix64 generator whose state starts at `x`.

    Identical to `splitmix64(input)` in slot_encoding.rs: add the golden gamma
    (wrapping), then apply the finalizer.
    """
    return mix64((x + GOLDEN_GAMMA) & MASK64)


def fnv1a64(data: bytes, state: int = FNV_OFFSET) -> int:
    """Return the wrapping FNV-1a-64 digest, optionally continuing a prior state."""
    for byte in data:
        state ^= byte
        state = (state * FNV_PRIME) & MASK64
    return state


class Stream:
    """Classic splitmix64: state += gamma; output = mix64(state)."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        """Initialize the stream with the seed reduced to an unsigned 64-bit state."""
        self.state = seed & MASK64

    def next(self) -> int:
        """Advance the state once and return the next splitmix64 output."""
        self.state = (self.state + GOLDEN_GAMMA) & MASK64
        return mix64(self.state)

    def u01(self) -> float:
        """Draw a uniform value in [0, 1) from the next output's upper 53 bits."""
        # (next >> 11) is below 2^53, so the division is exact.
        return (self.next() >> 11) / TWO_POW_53

    def randbelow(self, n: int) -> int:
        """Draw an integer in [0, n) using the stream's float scaling rule."""
        # float64 product, then floor. (2^53 - 1) / 2^53 * n rounds below n for
        # every n used here, so the result is always in 0..n-1.
        return int(self.u01() * n)


def example_seed(split: str, index: int) -> int:
    """Derive an example's independent seed from its split name and index."""
    return splitmix64(BENCHMARK_SEED ^ fnv1a64(split.encode("utf-8")) ^ splitmix64(index))


# ----------------------------------------------------------------------- codebook


def load_codebook(path: Path = CODEBOOK_PATH) -> dict[str, dict[str, int]]:
    """Name -> code for each family, read from the committed artifact."""
    document = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(bytes.fromhex(document["canonical_bytes_hex"])).hexdigest()
    if digest != document["fingerprint_sha256"]:
        raise SystemExit(f"{path}: fingerprint is not the digest of canonical_bytes_hex")
    if document["fingerprint_sha256"] != CODEBOOK_FINGERPRINT:
        raise SystemExit(
            f"{path}: codebook fingerprint {document['fingerprint_sha256']} is not the "
            f"one generator_version {GENERATOR_VERSION} was written against "
            f"({CODEBOOK_FINGERPRINT}); bump the generator version deliberately"
        )
    if document["version"] != CODEBOOK_VERSION:
        raise SystemExit(f"{path}: codebook version {document['version']} is not {CODEBOOK_VERSION}")
    return {
        family["family"]: {member["name"]: member["code"] for member in family["members"]}
        for family in document["families"]
    }


CODEBOOK = load_codebook()
ROLE = CODEBOOK["semantic_role"]
EPI = CODEBOOK["epistemic_state"]
OP = CODEBOOK["reasoning_operator"]
VALIDITY = CODEBOOK["validity"]

N_ROLES = len(ROLE)  # 9
N_EPI = len(EPI)  # 6
N_OPS = len(OP)  # 11
ROLE_NAMES = sorted(ROLE, key=ROLE.get)
EPI_NAMES = sorted(EPI, key=EPI.get)
OP_NAMES = sorted(OP, key=OP.get)
VALIDITY_NAMES = sorted(VALIDITY, key=VALIDITY.get)
assert (N_ROLES, N_EPI, N_OPS, len(VALIDITY)) == (9, 6, 11, 4)

# ------------------------------------------------------------------------ design

REGIME_NAMES = ["tabular", "temporal", "interventional", "textual"]
INTERVENTIONAL = REGIME_NAMES.index("interventional")
BUDGET = [0.4, 0.7, 1.0]
EVIDENCE = ROLE["evidence"]
PROBABILISTIC = OP["probabilistic"]
LIVE = VALIDITY["live"]
NON_LIVE = [VALIDITY["superseded"], VALIDITY["revoked"], VALIDITY["disputed"]]

ROLE_PRIOR = [
    ROLE[name]
    for name in (
        "goal", "constraint", "claim", "claim", "evidence", "evidence",
        "resource", "capability", "relation", "procedure", "action",
    )
]
HELD_OUT = frozenset(
    (ROLE[r], EPI[e])
    for r, e in (
        ("goal", "verified"), ("constraint", "hypothesis"), ("claim", "observed"),
        ("evidence", "assumed"), ("resource", "inferred"), ("capability", "unknown"),
        ("relation", "verified"), ("procedure", "hypothesis"), ("action", "observed"),
    )
)

G = [0.0, 0.5, 1.0, 1.0, 1.25]  # by confidence bucket
_W = {"unknown": 0.30, "assumed": 0.55, "hypothesis": 0.80, "observed": 1.30, "inferred": 1.05, "verified": 1.55}
_U = {"unknown": 0.35, "assumed": 0.60, "hypothesis": 0.85, "observed": 0.0, "inferred": 0.0, "verified": 0.0}
W = [_W[name] for name in EPI_NAMES]
U = [_U[name] for name in EPI_NAMES]
LAMBDA = 1.0
_COST = {
    "semantic": 0.2, "deductive": 0.3, "probabilistic": 0.5, "statistical": 0.5,
    "temporal": 0.5, "causal": 0.6, "search": 0.6, "optimization": 0.8,
    "simulation": 0.8, "symbolic": 0.4, "external_pod": 0.9,
}
COST = [_COST[name] for name in OP_NAMES]
PRIMARY_WEIGHT = 2.0
SECONDARY_WEIGHT = 0.9
REGIME_OP = [OP["statistical"], OP["temporal"], OP["causal"], OP["semantic"]]  # op(R)
_REGIME = "op(R)"
_AFFINITY_TABLE = {
    "goal": ("search", "optimization"),
    "constraint": ("optimization", "symbolic"),
    "claim": ("deductive", _REGIME),
    "evidence": (_REGIME, "probabilistic"),
    "resource": ("external_pod", "search"),
    "capability": ("simulation", "external_pod"),
    "relation": ("causal", "deductive"),
    "procedure": ("symbolic", "simulation"),
    "action": ("external_pod", "temporal"),
}
TIE_TOL = 1e-9
SOFTMAX_TEMPERATURE = 0.5

N_SLOTS = 6
VOCAB = 216
PAD = 0
ATTR_BASE = 1
ATTR_STRIDE = 24
OFF_ROLE, OFF_EPI, OFF_CB, OFF_VALIDITY = 0, 9, 15, 20
REGIME_TOKEN = 145
BUDGET_TOKEN = 149
FILLER_TOKEN = 152
N_FILLER_TYPES = 64

TAG_ORDER = ["validity", "regime", "budget", "confidence", "epistemic", "heldout", "transfer"]


def _op_code(name: str, regime: int) -> int:
    """Resolve an affinity's operator name, substituting the regime operator."""
    return REGIME_OP[regime] if name == _REGIME else OP[name]


def primary_op(role: int, regime: int) -> int:
    """Return the primary operator code for a role under the given regime."""
    return _op_code(_AFFINITY_TABLE[ROLE_NAMES[role]][0], regime)


def secondary_op(role: int, regime: int) -> int:
    """Return the secondary operator code for a role under the given regime."""
    return _op_code(_AFFINITY_TABLE[ROLE_NAMES[role]][1], regime)


# AFFINITY[role][regime] is the 11-vector A[r, R].
AFFINITY = []
for _role in range(N_ROLES):
    _per_regime = []
    for _regime in range(4):
        vector = [0.0] * N_OPS
        vector[primary_op(_role, _regime)] += PRIMARY_WEIGHT
        vector[secondary_op(_role, _regime)] += SECONDARY_WEIGHT
        _per_regime.append(vector)
    AFFINITY.append(_per_regime)
# PENALTY[B][k] = LAMBDA * max(0, COST_k - BUDGET_B), in that float64 order.
PENALTY = [[LAMBDA * max(0.0, COST[k] - BUDGET[b]) for k in range(N_OPS)] for b in range(3)]


# -------------------------------------------------------------------------- splits

SPLITS = [
    "train", "val", "test_iid",
    "ood_compose_epi", "ood_compose_regime", "ood_distractors", "ood_validity", "ood_payload",
]
SPLIT_SIZES = {
    "train": 16000, "val": 2000, "test_iid": 3000,
    "ood_compose_epi": 3000, "ood_compose_regime": 3000, "ood_distractors": 3000,
    "ood_validity": 3000, "ood_payload": 3000,
}
TEST_SPLITS = SPLITS[2:]


def p_live(split: str) -> float:
    """Return the split's probability of sampling a live fact."""
    return 0.50 if split == "ood_validity" else 0.78


def n_fillers(split: str) -> int:
    """Return the filler-token count, including the distractor shift's increase."""
    return 46 if split == "ood_distractors" else 10


def entity_base(split: str) -> int:
    """Return the entity ID offset used to separate the payload-shift pool."""
    return 256 if split == "ood_payload" else 0


# ------------------------------------------------------------------------ sampling


def confidence(k: int) -> float:
    """The continuous confidence of draw k in 0..999: (k + 0.5) / 1000."""
    return (k + 0.5) / 1000.0


def bucket(c: float) -> int:
    """cb = floor(5c); never ambiguous because c = (k + 0.5) / 1000."""
    return int(5.0 * c)


class Example:
    __slots__ = ("split", "index", "regime", "budget", "facts", "tokens")

    def __init__(self, split, index, regime, budget, facts, tokens):
        """Store the sampled context, ordered fact tuples, and raw tokens."""
        self.split = split
        self.index = index
        self.regime = regime
        self.budget = budget
        # facts: tuples (role, epistemic, c, validity, entity), in slot order.
        self.facts = facts
        self.tokens = tokens

    @property
    def id(self) -> str:
        """Return the split name and zero-padded example index as a stable ID."""
        return f"{self.split}-{self.index:05d}"


def sample(split: str, index: int) -> Example:
    """Example `index` of `split`, drawn from its own splitmix64 stream."""
    stream = Stream(example_seed(split, index))
    live_p = p_live(split)
    base = entity_base(split)
    while True:
        # (1) regime and budget. ood_compose_regime fixes R without a draw.
        regime = INTERVENTIONAL if split == "ood_compose_regime" else stream.randbelow(4)
        budget = stream.randbelow(3)
        # (2) the Evidence ban.
        ban = regime == INTERVENTIONAL and split != "ood_compose_regime"
        # (3) two focus roles.
        focus = []
        for _ in range(2):
            while True:
                role = ROLE_PRIOR[stream.randbelow(11)]
                if ban and role == EVIDENCE:
                    continue
                break
            focus.append(role)
        # (4) six facts.
        facts = []
        for _ in range(N_SLOTS):
            while True:
                if stream.u01() < 0.5:
                    role = focus[stream.randbelow(2)]
                else:
                    role = ROLE_PRIOR[stream.randbelow(11)]
                epi = stream.randbelow(6)
                if split != "ood_compose_epi" and (role, epi) in HELD_OUT:
                    continue
                if ban and role == EVIDENCE:
                    continue
                break
            c = confidence(stream.randbelow(1000))
            if stream.u01() < live_p:
                validity = LIVE
            else:
                validity = NON_LIVE[stream.randbelow(3)]
            entity = stream.randbelow(256) + base
            facts.append((role, epi, c, validity, entity))
        # (5) whole-example acceptance.
        if not any(f[3] == LIVE for f in facts):
            continue
        if split == "ood_compose_epi" and not any(
            f[3] == LIVE and bucket(f[2]) > 0 and (f[0], f[1]) in HELD_OUT for f in facts
        ):
            continue
        if split == "ood_compose_regime" and not any(
            f[3] == LIVE and bucket(f[2]) > 0 and f[0] == EVIDENCE for f in facts
        ):
            continue
        break
    # Raw tokens: 24 attribute tokens in slot order, regime, budget, then fillers;
    # then one Fisher-Yates (Durstenfeld) pass from the same stream.
    tokens = []
    for i, (role, epi, c, validity, _entity) in enumerate(facts):
        base_tok = ATTR_BASE + ATTR_STRIDE * i
        tokens.append(base_tok + OFF_ROLE + role)
        tokens.append(base_tok + OFF_EPI + epi)
        tokens.append(base_tok + OFF_CB + bucket(c))
        tokens.append(base_tok + OFF_VALIDITY + validity)
    tokens.append(REGIME_TOKEN + regime)
    tokens.append(BUDGET_TOKEN + budget)
    for _ in range(n_fillers(split)):
        tokens.append(FILLER_TOKEN + stream.randbelow(N_FILLER_TYPES))
    for i in range(len(tokens) - 1, 0, -1):
        j = stream.randbelow(i + 1)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return Example(split, index, regime, budget, facts, tokens)


# ---------------------------------------------------------------------- label rule


def utilities(
    facts,
    regime: int,
    budget: int,
    *,
    admit_all: bool = False,
    gain_one: bool = False,
    flat_epistemic: bool = False,
    drop=None,
    evidence_regime: int | None = None,
) -> list[float]:
    """z_k of the label rule, and of each counterfactual used by the tags.

    The factual rule is `utilities(facts, R, B)`: facts in slot order, each Live
    fact with cb > 0 adding G[cb] * (W[e] * A[r,R]_k + U[e] * [k = Probabilistic])
    - LAMBDA * max(0, COST_k - BUDGET_B) to z_k.

    Counterfactuals: admit_all treats every fact as Live; gain_one sets G to 1 for
    every bucket (so cb = 0 facts contribute too); flat_epistemic sets W to 1 and U
    to 0; drop(fact) removes facts entirely (vote and penalty); evidence_regime
    makes Evidence facts vote as if the regime were that value.
    """
    z = [0.0] * N_OPS
    penalty = PENALTY[budget]
    for fact in facts:
        role, epi, c, validity, _entity = fact
        if validity != LIVE and not admit_all:
            continue
        if drop is not None and drop(fact):
            continue
        cb = bucket(c)
        g = 1.0 if gain_one else G[cb]
        if g == 0.0:
            continue
        r = evidence_regime if (evidence_regime is not None and role == EVIDENCE) else regime
        a = AFFINITY[role][r]
        w = 1.0 if flat_epistemic else W[epi]
        u = 0.0 if flat_epistemic else U[epi]
        for k in range(N_OPS):
            z[k] += g * (w * a[k] + (u if k == PROBABILISTIC else 0.0)) - penalty[k]
    return z


def decide(z: list[float]) -> tuple[int, bool]:
    """argmax with the rule's tie-break: within 1e-9 of the max, lowest COST, then lowest code."""
    best = max(z)
    tied = [k for k in range(N_OPS) if z[k] >= best - TIE_TOL]
    return min(tied, key=lambda k: (COST[k], k)), len(tied) > 1


def label_of(facts, regime: int, budget: int, **counterfactual) -> int:
    """Return the winning operator after applying any requested counterfactuals."""
    return decide(utilities(facts, regime, budget, **counterfactual))[0]


def margin_of(z: list[float]) -> float:
    """Return the gap between the two highest operator utilities."""
    ordered = sorted(z, reverse=True)
    return ordered[0] - ordered[1]


def tags_of(example: Example, label: int) -> list[str]:
    """Return ordered tags for counterfactuals that change the example's label."""
    facts, regime, budget = example.facts, example.regime, example.budget
    tags = set()
    if label_of(facts, regime, budget, admit_all=True) != label:
        tags.add("validity")
    if any(label_of(facts, r, budget) != label for r in range(4) if r != regime):
        tags.add("regime")
    if any(label_of(facts, regime, b) != label for b in range(3) if b != budget):
        tags.add("budget")
    if label_of(facts, regime, budget, gain_one=True) != label:
        tags.add("confidence")
    if label_of(facts, regime, budget, flat_epistemic=True) != label:
        tags.add("epistemic")
    if example.split == "ood_compose_epi":
        if label_of(facts, regime, budget, drop=lambda f: (f[0], f[1]) in HELD_OUT) != label:
            tags.add("heldout")
    if example.split == "ood_compose_regime":
        if all(p != label for p in no_transfer_labels(facts, regime, budget)):
            tags.add("transfer")
    return [t for t in TAG_ORDER if t in tags]


def no_transfer_labels(facts, regime: int, budget: int) -> list[int]:
    """Evidence ignored, or voting as tabular, temporal or textual."""
    labels = [label_of(facts, regime, budget, drop=lambda f: f[0] == EVIDENCE)]
    for r in range(4):
        if r != INTERVENTIONAL:
            labels.append(label_of(facts, regime, budget, evidence_regime=r))
    return labels


# ------------------------------------------------------------------------- records


def _clean(x: float, digits: int) -> float:
    """Round a value for serialization and normalize negative zero to zero."""
    return round(x, digits) + 0.0  # + 0.0 turns -0.0 into 0.0


def route_targets(z: list[float]) -> list[float]:
    """softmax(z / 0.5), in millionths, residual on the largest entry, so the
    written targets sum to exactly 1 and do not depend on libm's last bit."""
    scaled = [v / SOFTMAX_TEMPERATURE for v in z]
    top = max(scaled)
    weights = [math.exp(v - top) for v in scaled]
    total = sum(weights)
    micro = [int(round(w / total * 1_000_000)) for w in weights]
    largest = min(range(N_OPS), key=lambda k: (-micro[k], k))
    micro[largest] += 1_000_000 - sum(micro)
    return [m / 1_000_000 for m in micro]


def render_task(example: Example) -> str:
    """Describe the regime, budget, and ordered facts as a routing prompt."""
    parts = []
    for i, (role, epi, c, validity, entity) in enumerate(example.facts):
        parts.append(
            f"[{i}] {ROLE_NAMES[role]} ({EPI_NAMES[epi]}, confidence {c:.4f}, "
            f"{VALIDITY_NAMES[validity]}) about entity-{entity}"
        )
    return (
        f"Route this {REGIME_NAMES[example.regime]} problem to one reasoning operator "
        f"under a cost budget of {BUDGET[example.budget]:.1f}. Facts: " + "; ".join(parts) + "."
    )


def build(example: Example) -> tuple[dict, str, int]:
    """The JSONL record, the TSV line and the label code of one example."""
    z = utilities(example.facts, example.regime, example.budget)
    label, _tied = decide(z)
    targets = route_targets(z)
    record = {
        "id": example.id,
        "split": example.split,
        "generator_version": GENERATOR_VERSION,
        "task": render_task(example),
        "routes": [{"operator": OP_NAMES[k], "target": targets[k]} for k in range(N_OPS)],
        "cost_budget": BUDGET[example.budget],
        "type_codebook_version": CODEBOOK_VERSION,
        "type_codebook_fingerprint": CODEBOOK_FINGERPRINT,
        "regime": {"code": example.regime, "name": REGIME_NAMES[example.regime]},
        "budget": {"code": example.budget, "value": BUDGET[example.budget]},
        "slots": [
            {
                "index": i,
                "role": ROLE_NAMES[role],
                "epistemic": EPI_NAMES[epi],
                "confidence": c,
                "confidence_bucket": bucket(c),
                "validity": VALIDITY_NAMES[validity],
                "entity": entity,
                "provenance_bucket": i,
            }
            for i, (role, epi, c, validity, entity) in enumerate(example.facts)
        ],
        "raw_tokens": list(example.tokens),
        "label": {"operator": OP_NAMES[label], "code": label},
        "utility": [_clean(v, 10) for v in z],
        "margin": _clean(margin_of(z), 10),
        "tags": tags_of(example, label),
    }
    facts = ";".join(
        f"{role},{epi},{c:.4f},{validity},{entity}" for role, epi, c, validity, entity in example.facts
    )
    tsv = (
        f"{example.id}\t{label}\t{example.regime}\t{example.budget}\t"
        f"{','.join(str(t) for t in example.tokens)}\t{facts}\n"
    )
    return record, tsv, label


def jsonl_line(record: dict) -> str:
    """Serialize a record as compact ASCII JSON followed by one newline."""
    return json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n"


def render_split(split: str, n: int | None = None) -> tuple[bytes, bytes, str]:
    """(JSONL bytes, TSV bytes, label hex string) for the first n examples."""
    count = SPLIT_SIZES[split] if n is None else n
    jsonl, tsv, labels = [], [], []
    for index in range(count):
        record, line, label = build(sample(split, index))
        jsonl.append(jsonl_line(record))
        tsv.append(line)
        labels.append(format(label, "x"))
    return "".join(jsonl).encode("utf-8"), "".join(tsv).encode("utf-8"), "".join(labels)


# -------------------------------------------------------------------------- digests

PREFIX_N = 64


def hex64(value: int) -> str:
    """Format a digest as lowercase hexadecimal padded to 16 characters."""
    return f"{value:016x}"


def lock_document(rendered: dict[str, tuple[bytes, bytes, str]]) -> dict:
    """Build split sizes, hashes, and prefix digests from rendered split bytes."""
    data_state = FNV_OFFSET
    splits = {}
    for split in SPLITS:
        jsonl, tsv, labels = rendered[split]
        data_state = fnv1a64(tsv, data_state)
        splits[split] = {
            "n": SPLIT_SIZES[split],
            "jsonl": {
                "path": f"datasets/generated/operator_routing_v1/{split}.jsonl",
                "bytes": len(jsonl),
                "sha256": hashlib.sha256(jsonl).hexdigest(),
            },
            "tsv": {
                "path": f"datasets/generated/operator_routing_v1/{split}.tsv",
                "bytes": len(tsv),
                "sha256": hashlib.sha256(tsv).hexdigest(),
                "fnv1a64": hex64(fnv1a64(tsv)),
            },
            "label_fnv1a64": hex64(fnv1a64(labels.encode("ascii"))),
        }
    return {
        "generator_version": GENERATOR_VERSION,
        "benchmark_seed": BENCHMARK_SEED,
        "type_codebook_version": CODEBOOK_VERSION,
        "type_codebook_fingerprint": CODEBOOK_FINGERPRINT,
        "data_fnv1a64": hex64(data_state),
        "data_fnv1a64_definition": (
            "FNV-1a-64 (offset 0xcbf29ce484222325, prime 0x100000001b3) over the raw bytes "
            "of the eight TSV files concatenated in this order, with nothing between them: "
            + ", ".join(f"{s}.tsv" for s in SPLITS)
        ),
        "label_fnv1a64_definition": (
            "FNV-1a-64 over the ASCII string of one lowercase hex digit (label code 0-a) per "
            "example, in file order: the PRED-string form of the gold labels"
        ),
        "splits": splits,
        "prefix": {
            "n": PREFIX_N,
            "definition": "SHA-256 of the first n lines of each file, for fast stability tests",
            "splits": {
                split: {
                    "jsonl_sha256": hashlib.sha256(
                        b"".join(rendered[split][0].splitlines(keepends=True)[:PREFIX_N])
                    ).hexdigest(),
                    "tsv_sha256": hashlib.sha256(
                        b"".join(rendered[split][1].splitlines(keepends=True)[:PREFIX_N])
                    ).hexdigest(),
                }
                for split in SPLITS
            },
        },
    }


def dump_json(document: dict) -> str:
    """Serialize a document as indented JSON with a trailing newline."""
    return json.dumps(document, indent=2, sort_keys=False) + "\n"


def render_all(verbose: bool = True) -> dict[str, tuple[bytes, bytes, str]]:
    """Render every split in specification order, optionally reporting progress."""
    rendered = {}
    for split in SPLITS:
        rendered[split] = render_split(split)
        if verbose:
            print(f"  {split}: {SPLIT_SIZES[split]} examples", file=sys.stderr)
    return rendered


def compare_lock(expected: dict, actual: dict) -> list[str]:
    """Return field-level differences between stored and regenerated locks."""
    problems = []

    def walk(a, b, path):
        """Append recursive dictionary differences with their dotted field paths."""
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a:
                    problems.append(f"{path}{key}: absent from the lock")
                elif key not in b:
                    problems.append(f"{path}{key}: absent from the regenerated data")
                else:
                    walk(a[key], b[key], f"{path}{key}.")
        elif a != b:
            problems.append(f"{path.rstrip('.')}: lock {a!r} != regenerated {b!r}")

    walk(expected, actual, "")
    return problems


def check_on_disk(out: Path, lock: dict) -> list[str]:
    """Check existing split files against lock hashes and report missing files."""
    problems = []
    for split in SPLITS:
        for kind in ("jsonl", "tsv"):
            path = out / f"{split}.{kind}"
            if not path.is_file():
                print(f"note: {path} is not written; only the regenerated bytes were checked", file=sys.stderr)
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != lock["splits"][split][kind]["sha256"]:
                problems.append(f"{path}: sha256 {digest} != lock {lock['splits'][split][kind]['sha256']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Generate splits or samples, or verify regenerated data against the lock."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH, help="splits.lock.json path")
    parser.add_argument("--check", action="store_true", help="regenerate in memory and compare with the lock")
    parser.add_argument("--update-lock", action="store_true", help="write the splits and (re)write the lock")
    parser.add_argument("--write-sample", nargs="?", const=SAMPLE_PATH, type=Path, default=None,
                        help="write one record per split (example 0) to this JSONL path")
    args = parser.parse_args(argv)

    if args.write_sample is not None:
        lines = [jsonl_line(build(sample(split, 0))[0]) for split in SPLITS]
        args.write_sample.write_text("".join(lines), encoding="utf-8")
        print(f"wrote {args.write_sample}", file=sys.stderr)
        if not (args.check or args.update_lock):
            return 0

    rendered = render_all()
    actual = lock_document(rendered)

    if args.check:
        if not args.lock.is_file():
            print(f"error: {args.lock} does not exist", file=sys.stderr)
            return 1
        expected = json.loads(args.lock.read_text(encoding="utf-8"))
        problems = compare_lock(expected, actual) + check_on_disk(args.out, expected)
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"OK: every split matches {args.lock.name} (data FNV-1a-64 {actual['data_fnv1a64']})")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        jsonl, tsv, _labels = rendered[split]
        (args.out / f"{split}.jsonl").write_bytes(jsonl)
        (args.out / f"{split}.tsv").write_bytes(tsv)
    if args.update_lock:
        args.lock.write_text(dump_json(actual), encoding="utf-8")
        print(f"wrote {args.lock} (data FNV-1a-64 {actual['data_fnv1a64']})")
        return 0
    if args.lock.is_file():
        problems = compare_lock(json.loads(args.lock.read_text(encoding="utf-8")), actual)
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if problems:
            print("error: the written splits differ from the lock; nothing was re-pinned", file=sys.stderr)
            return 1
    print(f"wrote {len(SPLITS)} splits to {args.out} (data FNV-1a-64 {actual['data_fnv1a64']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
