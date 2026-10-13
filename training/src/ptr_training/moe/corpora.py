"""M011 corpora: pinned sources, partition rules, domain labels, selection digests.

Everything here is pure Python. `datasets` / `huggingface_hub` are imported only inside the
`load_*` functions that actually download, so tests run offline against in-memory fixtures that
follow each dataset's schema.

Schemas (checked against the dataset cards / files at the pinned revisions):

* `openai/gsm8k`, config `main`: splits `train` (7473) and `test` (1319); fields `question`,
  `answer` (reasoning, ends with `#### <number>`).
* `openai/openai_humaneval`, config `openai_humaneval`, split `test` (164); fields `task_id`,
  `prompt`, `canonical_solution`, `test`, `entry_point`.
* `gorilla-llm/Berkeley-Function-Calling-Leaderboard`, file `BFCL_v3_simple.json` (JSON lines):
  `id`, `question` (list of turns, each a list of `{role, content}`), `function` (list of
  `{name, description, parameters}`); ground truth in `possible_answer/BFCL_v3_simple.json`
  (JSON lines `{id, ground_truth: [{func_name: {param: [allowed values]}}]}`).

Partition rules (M011 config `corpus_split_rule`, `validation_rule`):

* GSM8K: `train` items with position mod 10 == 9 are the tuning validation, the rest train;
  `test` is the held-out evaluation.
* HumanEval and BFCL: index mod 3: 0 train, 1 tuning validation, 2 held-out evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ptr_training.moe import scorers

DOMAIN_OF = {"gsm8k": "math", "humaneval": "code", "bfcl_simple": "tool"}


@dataclass(frozen=True)
class CorpusPin:
    key: str  # gsm8k | humaneval | bfcl_simple
    repo: str
    revision: str
    domain: str
    config: str | None = None
    file: str | None = None


PINNED_CORPORA: dict[str, CorpusPin] = {
    "gsm8k": CorpusPin(
        "gsm8k", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "math", config="main"
    ),
    "humaneval": CorpusPin(
        "humaneval",
        "openai/openai_humaneval",
        "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "code",
        config="openai_humaneval",
    ),
    "bfcl_simple": CorpusPin(
        "bfcl_simple",
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
        "tool",
        file="BFCL_v3_simple.json",
    ),
}
VALIDATION_MODULUS_GSM8K = 10
VALIDATION_RESIDUE_GSM8K = 9
SPLIT_MODULUS_SINGLE = 3  # HumanEval / BFCL: 0 train, 1 validation, 2 evaluation


@dataclass(frozen=True)
class Example:
    corpus: str
    domain: str
    index: int  # position in the source split (the partition rules act on this)
    item_id: str
    prompt: str
    answer: str  # SFT completion target
    reference: dict = field(default_factory=dict, compare=False)  # scorer inputs


@dataclass(frozen=True)
class Partitions:
    train: list[Example]
    validation: list[Example]
    evaluation: list[Example]


# ----------------------------------------------------------------------------- example builders


def gsm8k_example(index: int, row: dict, split: str) -> Example:
    """GSM8K has two source splits whose row indices both start at 0, so the id names the split:
    `gsm8k:train:9` and `gsm8k:test:9` are different items."""
    return Example(
        corpus="gsm8k",
        domain="math",
        index=index,
        item_id=f"gsm8k:{split}:{index}",
        prompt=f"Question: {row['question']}\nAnswer:",
        answer=" " + row["answer"],  # leading space: BPE-natural token boundary
        reference={"final": scorers.gsm8k_reference(row["answer"])},
    )


def humaneval_example(index: int, row: dict) -> Example:
    return Example(
        corpus="humaneval",
        domain="code",
        index=index,
        item_id=row["task_id"],
        prompt=row["prompt"],
        answer=row["canonical_solution"],
        reference={
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        },
    )


def bfcl_example(index: int, row: dict, ground_truth: list[dict]) -> Example:
    turns = row["question"]
    if len(turns) != 1 or len(turns[0]) != 1:
        raise ValueError(f"BFCL simple item {row['id']} is not a single user turn")
    functions = json.dumps(row["function"], sort_keys=True, separators=(",", ":"))
    prompt = f"Functions: {functions}\nQuestion: {turns[0][0]['content']}\nCall:"
    return Example(
        corpus="bfcl_simple",
        domain="tool",
        index=index,
        item_id=row["id"],
        prompt=prompt,
        answer=" " + scorers.bfcl_target(ground_truth),
        reference={"ground_truth": ground_truth},
    )


# ----------------------------------------------------------------------------- partitions


def partition_gsm8k(train_rows: list[dict], test_rows: list[dict]) -> Partitions:
    train, validation = [], []
    for i, row in enumerate(train_rows):
        target = validation if i % VALIDATION_MODULUS_GSM8K == VALIDATION_RESIDUE_GSM8K else train
        target.append(gsm8k_example(i, row, "train"))
    evaluation = [gsm8k_example(i, row, "test") for i, row in enumerate(test_rows)]
    return Partitions(train, validation, evaluation)


def _partition_mod3(examples: list[Example]) -> Partitions:
    parts: list[list[Example]] = [[], [], []]
    for ex in examples:
        parts[ex.index % SPLIT_MODULUS_SINGLE].append(ex)
    return Partitions(parts[0], parts[1], parts[2])


def partition_humaneval(rows: list[dict]) -> Partitions:
    return _partition_mod3([humaneval_example(i, r) for i, r in enumerate(rows)])


def partition_bfcl(rows: list[dict], answers: list[dict]) -> Partitions:
    by_id = {a["id"]: a["ground_truth"] for a in answers}
    missing = [r["id"] for r in rows if r["id"] not in by_id]
    if missing:
        raise ValueError(f"BFCL items without ground truth: {missing[:5]}")
    return _partition_mod3([bfcl_example(i, r, by_id[r["id"]]) for i, r in enumerate(rows)])


# ----------------------------------------------------------------------------- digests


def _reference_digest(reference: dict) -> str:
    """sha256 of the scorer inputs as canonical JSON (sorted keys), so a changed HumanEval test
    or BFCL allowed answer changes the selection digest even when prompt and answer do not."""
    canon = json.dumps(reference, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr)
    return hashlib.sha256(canon.encode()).hexdigest()


def selection_digest(examples: list[Example]) -> str:
    """sha256 over canonical JSON of `[source index, item id, sha256(prompt NUL answer),
    sha256(reference)]` rows in selection (index) order."""
    rows = [
        [
            ex.index,
            ex.item_id,
            hashlib.sha256(f"{ex.prompt}\0{ex.answer}".encode()).hexdigest(),
            _reference_digest(ex.reference),
        ]
        for ex in examples
    ]
    canon = json.dumps(rows, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def corpus_report(key: str, parts: Partitions) -> dict:
    """Per-corpus counts, digests and the validation-overlap counter for run.json. A validation
    item overlaps when its id or its prompt text also occurs in training or in the held-out
    evaluation split (either way tuning would see an item it must not)."""
    def prompt_key(ex: Example) -> str:
        return hashlib.sha256(ex.prompt.encode()).hexdigest()

    others = parts.train + parts.evaluation
    other_ids = {ex.item_id for ex in others}
    other_prompts = {prompt_key(ex) for ex in others}
    overlapping = [
        ex for ex in parts.validation
        if ex.item_id in other_ids or prompt_key(ex) in other_prompts
    ]
    return {
        "domain": DOMAIN_OF[key],
        "counts": {
            "train": len(parts.train),
            "validation": len(parts.validation),
            "evaluation": len(parts.evaluation),
        },
        "selection_digest": {
            "train": selection_digest(parts.train),
            "validation": selection_digest(parts.validation),
            "evaluation": selection_digest(parts.evaluation),
        },
        "validation_overlap_count": len(overlapping),
        "training_outside_training_partition_count": 0,
    }


# ----------------------------------------------------------------------------- loaders (lazy)


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_gsm8k(pin: CorpusPin = PINNED_CORPORA["gsm8k"]) -> Partitions:
    from datasets import load_dataset

    ds = load_dataset(pin.repo, pin.config, revision=pin.revision)
    return partition_gsm8k([dict(r) for r in ds["train"]], [dict(r) for r in ds["test"]])


def load_humaneval(pin: CorpusPin = PINNED_CORPORA["humaneval"]) -> Partitions:
    from datasets import load_dataset

    ds = load_dataset(pin.repo, pin.config, revision=pin.revision, split="test")
    return partition_humaneval([dict(r) for r in ds])


def load_bfcl(pin: CorpusPin = PINNED_CORPORA["bfcl_simple"]) -> Partitions:
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> list[dict]:
        path = hf_hub_download(pin.repo, name, repo_type="dataset", revision=pin.revision)
        return _read_jsonl(path)

    return partition_bfcl(fetch(pin.file), fetch(f"possible_answer/{pin.file}"))


def load_pinned() -> dict[str, Partitions]:
    """Download (or read from the HF cache) the three corpora at their pinned revisions."""
    return {
        "gsm8k": load_gsm8k(),
        "humaneval": load_humaneval(),
        "bfcl_simple": load_bfcl(),
    }


# ----------------------------------------------------------------------------- fixtures


def fixture_partitions() -> dict[str, Partitions]:
    """Small deterministic in-memory corpora that follow the real schemas (offline tests and
    the tiny-model smoke run). The HumanEval canonical solutions pass their tests and the BFCL
    targets pass the AST check, so the scorers can be exercised end to end."""
    gsm_train = [
        {"question": f"Tom has {a} apples and buys {b} more. How many apples?",
         "answer": f"{a} + {b} = <<{a}+{b}={a + b}>>{a + b}\n#### {a + b}"}
        for a, b in ((i % 7 + 2, i % 5 + 1) for i in range(30))
    ]
    gsm_test = [
        {"question": f"Sam has {a} pens and loses {b}. How many pens?",
         "answer": f"{a} - {b} = <<{a}-{b}={a - b}>>{a - b}\n#### {a - b}"}
        for a, b in ((i + 6, i % 3 + 1) for i in range(6))
    ]
    humaneval = [
        {
            "task_id": f"Fixture/{i}",
            "prompt": f"def add_{i}(x):\n    \"\"\"Add {i} to x.\"\"\"\n",
            "canonical_solution": f"    return x + {i}\n",
            "test": f"def check(candidate):\n    assert candidate(1) == {1 + i}\n"
            f"    assert candidate(5) == {5 + i}\n",
            "entry_point": f"add_{i}",
        }
        for i in range(12)
    ]
    bfcl_rows, bfcl_answers = [], []
    for i in range(12):
        bfcl_rows.append(
            {
                "id": f"simple_{i}",
                "question": [[{"role": "user", "content": f"Double the number {i + 2}."}]],
                "function": [
                    {
                        "name": "math.double",
                        "description": "Double a number.",
                        "parameters": {
                            "type": "dict",
                            "properties": {"number": {"type": "integer", "description": "n"}},
                            "required": ["number"],
                        },
                    }
                ],
            }
        )
        bfcl_answers.append({"id": f"simple_{i}", "ground_truth": [{"math.double": {"number": [i + 2]}}]})
    return {
        "gsm8k": partition_gsm8k(gsm_train, gsm_test),
        "humaneval": partition_humaneval(humaneval),
        "bfcl_simple": partition_bfcl(bfcl_rows, bfcl_answers),
    }


def by_domain(parts: dict[str, Partitions], split: str) -> dict[str, list[Example]]:
    """{domain: examples of `split`} (`train` | `validation` | `evaluation`)."""
    return {DOMAIN_OF[k]: list(getattr(p, split)) for k, p in parts.items()}


__all__ = [
    "CorpusPin",
    "DOMAIN_OF",
    "Example",
    "PINNED_CORPORA",
    "Partitions",
    "by_domain",
    "corpus_report",
    "fixture_partitions",
    "load_pinned",
    "partition_bfcl",
    "partition_gsm8k",
    "partition_humaneval",
    "selection_digest",
]
