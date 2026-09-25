#!/usr/bin/env python3
"""Operator-routing v1: reference lines and the G0 data bands.

Reads the generated splits (the TSV twin), recomputes every label with
generator.py's rule, and writes research/falsification/A0-ablations-v1/references.json:

- label statistics per split (class shares, exact ties, margins, decisive rates);
- the reference lines: train-majority, count router (naive Bayes), unbound
  bag-of-attributes and bound additive multinomial logistic regressions (SGD),
  hand-primary, hand-weighted, the validity-blind rule, the raw-blind exact Bayes
  ceiling (training posterior), the no-transfer references and the nuisance-only
  predictor;
- every G0 data band with its measured value and pass/fail.

    python3 benchmarks/operator-routing/references.py [--data DIR] [--out FILE]

No model is needed. Exits 1 if any band fails or the data does not match
splits.lock.json. Nothing here changes a constant of the design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generator as gen  # noqa: E402  (references may share the generator's rule; score.py may not)

ROOT = gen.ROOT
DEFAULT_OUT = ROOT / "research/falsification/A0-ablations-v1/references.json"
EVAL_SPLITS = ["val", "test_iid", "ood_compose_epi", "ood_compose_regime",
               "ood_distractors", "ood_validity", "ood_payload"]
DECISIVE = ["validity", "regime", "budget", "confidence", "epistemic"]
N_OPS = gen.N_OPS

# Values the synthesized design measured on its prototype (Python `random`,
# n = 3000 per split, SE about 0.009), kept beside the recomputed ones.
PROTOTYPE = {
    "references": {
        "train_majority": {"test_iid": 0.200, "ood_compose_epi": 0.238, "ood_compose_regime": 0.069, "ood_validity": 0.195},
        "count_router": {"test_iid": 0.338, "ood_compose_epi": 0.306, "ood_compose_regime": 0.431, "ood_validity": 0.339},
        "unbound_bag_of_attributes": {"test_iid": 0.583, "ood_compose_epi": 0.575, "ood_compose_regime": 0.340, "ood_validity": 0.508},
        "bound_additive": {"test_iid": 0.651, "ood_compose_epi": 0.613, "ood_compose_regime": 0.377, "ood_validity": 0.652},
        "hand_primary": {"test_iid": 0.652, "ood_compose_epi": 0.648, "ood_compose_regime": 0.607, "ood_validity": 0.702},
        "hand_weighted": {"test_iid": 0.812, "ood_compose_epi": 0.790, "ood_compose_regime": 0.783, "ood_validity": 0.853},
        "validity_blind_rule": {"test_iid": 0.792, "ood_compose_epi": 0.826, "ood_compose_regime": 0.822, "ood_validity": 0.601},
        "raw_blind_exact_bayes": {"test_iid": 0.815, "ood_compose_epi": 0.834, "ood_compose_regime": 0.449, "ood_validity": 0.851},
    },
    "no_transfer": {
        "ood_compose_epi": {"ignore_heldout": 0.650, "heldout_as_inferred": 0.835},
        "ood_compose_regime": {"ignore_evidence": 0.468, "evidence_as_tabular": 0.496,
                               "evidence_as_temporal": 0.465, "evidence_as_textual": 0.479,
                               "causal_label_share": 0.534},
    },
    "label_stats_test_iid": {"majority_share": 0.200, "smallest_share": 0.053, "exact_tie_rate": 0.010,
                             "margin_below_0_1": 0.070, "mean_live_facts": 4.68,
                             "decisive": {"validity": 0.208, "regime": 0.221, "budget": 0.192,
                                          "confidence": 0.268, "epistemic": 0.271}},
    "ood_compose_epi_heldout_decisive": 0.35,
    "ood_compose_regime_regime_decisive": 0.57,
    "ood_compose_regime_transfer_share": 0.43,
    "ood_validity_exact_tie_rate": 0.041,
    "train_causal_share_under_interventional": 0.096,
}


# --------------------------------------------------------------------------- data


class Row:
    __slots__ = ("id", "split", "label", "regime", "budget", "tokens", "facts", "z", "tied", "margin", "tags")


def load(data_dir: Path, split: str, with_tags: bool) -> list[Row]:
    """Read a TSV split, verify its labels and size, and optionally derive tags."""
    rows = []
    for index, line in enumerate((data_dir / f"{split}.tsv").read_text(encoding="utf-8").splitlines()):
        ident, label, regime, budget, tokens, facts = line.split("\t")
        row = Row()
        row.id, row.split = ident, split
        row.label, row.regime, row.budget = int(label), int(regime), int(budget)
        row.tokens = [int(t) for t in tokens.split(",")]
        row.facts = []
        for chunk in facts.split(";"):
            r, e, c, v, ent = chunk.split(",")
            row.facts.append((int(r), int(e), float(c), int(v), int(ent)))
        row.z = gen.utilities(row.facts, row.regime, row.budget)
        derived, row.tied = gen.decide(row.z)
        if derived != row.label:
            raise SystemExit(f"{split}:{ident}: TSV label {row.label} but the rule gives {derived}")
        row.margin = gen.margin_of(row.z)
        if with_tags:
            example = gen.Example(split, index, row.regime, row.budget, row.facts, row.tokens)
            row.tags = set(gen.tags_of(example, row.label))
        else:
            row.tags = set()
        rows.append(row)
    if len(rows) != gen.SPLIT_SIZES[split]:
        raise SystemExit(f"{split}: {len(rows)} rows, expected {gen.SPLIT_SIZES[split]}")
    return rows


def data_identity(data_dir: Path) -> dict:
    """Compare all split hashes and the combined TSV digest with the lock."""
    lock = json.loads(gen.LOCK_PATH.read_text(encoding="utf-8"))
    files = {}
    ok = True
    state = gen.FNV_OFFSET
    for split in gen.SPLITS:
        for kind in ("jsonl", "tsv"):
            data = (data_dir / f"{split}.{kind}").read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            match = digest == lock["splits"][split][kind]["sha256"]
            ok &= match
            files[f"{split}.{kind}"] = {"sha256": digest, "matches_lock": match}
            if kind == "tsv":
                state = gen.fnv1a64(data, state)
    fnv = gen.hex64(state)
    ok &= fnv == lock["data_fnv1a64"]
    return {"data_fnv1a64": fnv, "lock_data_fnv1a64": lock["data_fnv1a64"],
            "all_files_match_lock": ok, "files": files}


# ---------------------------------------------------------------------- statistics


def label_stats(rows: list[Row]) -> dict:
    """Summarize class shares, ties, margins, live facts, and decisive subsets."""
    n = len(rows)
    counts = Counter(r.label for r in rows)
    shares = {gen.OP_NAMES[k]: counts.get(k, 0) / n for k in range(N_OPS)}
    majority = max(range(N_OPS), key=lambda k: (counts.get(k, 0), -k))
    smallest = min(range(N_OPS), key=lambda k: (counts.get(k, 0), k))
    out = {
        "n": n,
        "class_shares": shares,
        "majority_operator": gen.OP_NAMES[majority],
        "majority_share": counts.get(majority, 0) / n,
        "smallest_operator": gen.OP_NAMES[smallest],
        "smallest_share": counts.get(smallest, 0) / n,
        "exact_tie_rate": sum(r.tied for r in rows) / n,
        "margin_below_0_1": sum(r.margin < 0.1 for r in rows) / n,
        "clear_margin_rate": sum(r.margin >= 0.25 - gen.TIE_TOL for r in rows) / n,
        "mean_live_facts": sum(sum(f[3] == gen.LIVE for f in r.facts) for r in rows) / n,
    }
    if rows and rows[0].split != "train":
        out["decisive"] = {t: sum(t in r.tags for r in rows) / n for t in DECISIVE}
        if rows[0].split == "ood_compose_epi":
            out["heldout_decisive"] = sum("heldout" in r.tags for r in rows) / n
        if rows[0].split == "ood_compose_regime":
            out["transfer_share"] = sum("transfer" in r.tags for r in rows) / n
            transfer = [r for r in rows if "transfer" in r.tags]
            out["transfer_subset_causal_share"] = (
                sum(r.label == gen.OP["causal"] for r in transfer) / len(transfer) if transfer else None
            )
    return out


def accuracy(predict, rows: list[Row]) -> float:
    """Return the fraction of rows whose predicted operator matches the label."""
    return sum(predict(r) == r.label for r in rows) / len(rows)


# ---------------------------------------------------------------------- references


def train_majority(train: list[Row]):
    """Return a constant predictor and its training-majority operator code."""
    counts = Counter(r.label for r in train)
    best = max(range(N_OPS), key=lambda k: (counts.get(k, 0), -k))
    return lambda r: best, best


def count_router(train: list[Row]):
    """Naive Bayes over (role, R), e and cb of Live facts, plus R and B; Laplace 1."""
    n = len(train)
    cy = Counter()
    c_r = {y: Counter() for y in range(N_OPS)}
    c_b = {y: Counter() for y in range(N_OPS)}
    c_rho = {y: Counter() for y in range(N_OPS)}
    c_e = {y: Counter() for y in range(N_OPS)}
    c_cb = {y: Counter() for y in range(N_OPS)}
    n_slots = Counter()
    for r in train:
        y = r.label
        cy[y] += 1
        c_r[y][r.regime] += 1
        c_b[y][r.budget] += 1
        for role, e, c, v, _ in r.facts:
            if v != gen.LIVE:
                continue
            c_rho[y][(role, r.regime)] += 1
            c_e[y][e] += 1
            c_cb[y][gen.bucket(c)] += 1
            n_slots[y] += 1

    def lp(table, y, key, card, total):
        """Return a Laplace-smoothed log probability for one categorical feature."""
        return math.log((table[y][key] + 1) / (total + card))

    def predict(r: Row) -> int:
        """Choose the most likely operator from live-fact counts, regime, and budget."""
        best, best_y = None, None
        for y in range(N_OPS):
            if cy[y] == 0:
                continue
            s = math.log(cy[y] / n) + lp(c_r, y, r.regime, 4, cy[y]) + lp(c_b, y, r.budget, 3, cy[y])
            for role, e, c, v, _ in r.facts:
                if v != gen.LIVE:
                    continue
                s += lp(c_rho, y, (role, r.regime), 36, n_slots[y])
                s += lp(c_e, y, e, 6, n_slots[y])
                s += lp(c_cb, y, gen.bucket(c), 5, n_slots[y])
            if best is None or s > best:
                best, best_y = s, y
        return best_y

    return predict


def feats_bound(r: Row) -> list[tuple[int, float]]:
    """Bound additive: role, e and cb histograms of the Live facts (mean-normalised), R, B, bias."""
    live = [f for f in r.facts if f[3] == gen.LIVE]
    n = len(live)
    x = Counter()
    for role, e, c, _v, _ in live:
        x[role] += 1 / n
        x[9 + e] += 1 / n
        x[15 + gen.bucket(c)] += 1 / n
    x[20 + r.regime] += 1
    x[24 + r.budget] += 1
    x[27] += 1
    return list(x.items())


def feats_unbound(r: Row) -> list[tuple[int, float]]:
    """Unbound bag of attributes: role, e, cb and live/non-live histograms over all six facts, R, B, bias."""
    x = Counter()
    for role, e, c, v, _ in r.facts:
        x[role] += 1 / 6
        x[9 + e] += 1 / 6
        x[15 + gen.bucket(c)] += 1 / 6
        x[28 + (0 if v == gen.LIVE else 1)] += 1 / 6
    x[20 + r.regime] += 1
    x[24 + r.budget] += 1
    x[27] += 1
    return list(x.items())


def train_logistic(feats, train: list[Row], n_features=30, epochs=5, lr0=0.5, seed=1):
    """Multinomial logistic regression by plain SGD.

    Zero init; one example per step; lr = lr0 * (1 - step / total) + 0.01. Each
    epoch reshuffles the running index permutation with one Fisher-Yates pass drawn
    from a splitmix64 stream seeded with `seed` (the generator's PRNG, not Python's
    random). Ties in prediction go to the lowest code.
    """
    weights = [[0.0] * N_OPS for _ in range(n_features)]
    order = list(range(len(train)))
    stream = gen.Stream(seed)
    xs = [feats(r) for r in train]
    ys = [r.label for r in train]
    total = epochs * len(order)
    step = 0
    ops = range(N_OPS)
    for _ in range(epochs):
        for i in range(len(order) - 1, 0, -1):
            j = stream.randbelow(i + 1)
            order[i], order[j] = order[j], order[i]
        for i in order:
            lr = lr0 * (1 - step / total) + 0.01
            step += 1
            x = xs[i]
            z = [0.0] * N_OPS
            for f, v in x:
                w = weights[f]
                for k in ops:
                    z[k] += w[k] * v
            top = max(z)
            e = [math.exp(q - top) for q in z]
            s = sum(e)
            g = [q / s for q in e]
            g[ys[i]] -= 1.0
            for f, v in x:
                w = weights[f]
                for k in ops:
                    w[k] -= lr * g[k] * v

    def predict(r: Row) -> int:
        """Score sparse features with fitted weights, breaking ties by lowest code."""
        z = [0.0] * N_OPS
        for f, v in feats(r):
            w = weights[f]
            for k in ops:
                z[k] += w[k] * v
        top = max(z)
        return z.index(top)

    return predict


def hand_router(weighted: bool):
    """Build a primary-affinity router with optional confidence and state weights."""
    def predict(r: Row) -> int:
        """Vote for primary operators using live facts with nonzero confidence gain."""
        z = [0.0] * N_OPS
        for role, e, c, v, _ in r.facts:
            cb = gen.bucket(c)
            if v == gen.LIVE and cb > 0:
                z[gen.primary_op(role, r.regime)] += (gen.G[cb] * gen.W[e]) if weighted else 1.0
        return gen.decide(z)[0]

    return predict


def validity_blind(r: Row) -> int:
    """Apply the label rule while admitting facts of every validity state."""
    return gen.label_of(r.facts, r.regime, r.budget, admit_all=True)


class RawBlindBayes:
    """The exact posterior over (R, B) given every slot field, under the training
    generative model, and the Bayes decision it implies.

    Only (role, e) of each fact carries information about R: c, validity and the
    entity are drawn independently of R and B, and the whole-example rejection
    (no Live fact) depends only on validity. B is uniform and independent of
    everything in the slots. R enters only through the Evidence ban (R =
    interventional in a training-type split), which removes Evidence from the
    focus-role draw and from every fact draw. The likelihood of the six (role, e)
    pairs is therefore a sum over the 81 focus-role pairs of the focus prior times
    the product of per-fact rejection-sampling probabilities.
    """

    def __init__(self) -> None:
        """Precompute focus priors and fact probabilities with and without the evidence ban."""
        prior = [gen.ROLE_PRIOR.count(r) / len(gen.ROLE_PRIOR) for r in range(gen.N_ROLES)]
        self.focus = {}
        self.fact = {}
        for ban in (False, True):
            p = [0.0 if (ban and r == gen.EVIDENCE) else prior[r] for r in range(gen.N_ROLES)]
            s = sum(p)
            self.focus[ban] = [x / s for x in p]
            for f1 in range(gen.N_ROLES):
                for f2 in range(gen.N_ROLES):
                    q = {}
                    for r in range(gen.N_ROLES):
                        # role = focus[randbelow(2)] w.p. 1/2, else ROLE_PRIOR[randbelow(11)]
                        qr = 0.5 * ((r == f1) + (r == f2)) / 2 + 0.5 * prior[r]
                        for e in range(gen.N_EPI):
                            if (r, e) in gen.HELD_OUT or (ban and r == gen.EVIDENCE):
                                continue
                            q[(r, e)] = qr / gen.N_EPI
                    z = sum(q.values())
                    self.fact[(f1, f2, ban)] = {k: v / z for k, v in q.items()}

    def likelihood(self, pairs, ban: bool) -> float:
        """Marginalize role-state pair likelihoods over the latent focus-role pair."""
        fd = self.focus[ban]
        total = 0.0
        for f1 in range(gen.N_ROLES):
            if fd[f1] == 0.0:
                continue
            for f2 in range(gen.N_ROLES):
                if fd[f2] == 0.0:
                    continue
                d = self.fact[(f1, f2, ban)]
                p = fd[f1] * fd[f2]
                for pair in pairs:
                    p *= d.get(pair, 0.0)
                    if p == 0.0:
                        break
                total += p
        return total

    def posterior_regime(self, facts) -> list[float]:
        """Infer regime probabilities from facts, dropping impossible held-out pairs if needed."""
        pairs = [(f[0], f[1]) for f in facts]
        lik = {ban: self.likelihood(pairs, ban) for ban in (False, True)}
        weights = [0.25 * lik[r == gen.INTERVENTIONAL] for r in range(4)]
        if sum(weights) == 0.0:
            # Held-out pairs have probability 0 under every regime of the training
            # model, so they carry no information about R: drop them.
            pairs = [p for p in pairs if p not in gen.HELD_OUT]
            lik = {ban: self.likelihood(pairs, ban) for ban in (False, True)}
            weights = [0.25 * lik[r == gen.INTERVENTIONAL] for r in range(4)]
        s = sum(weights)
        return [w / s for w in weights] if s > 0 else [0.25] * 4

    def decide(self, r: Row) -> tuple[int, float]:
        """Return the Bayes operator and its posterior mass, marginalizing regime and budget."""
        post = self.posterior_regime(r.facts)
        mass = [0.0] * N_OPS
        for regime in range(4):
            if post[regime] == 0.0:
                continue
            for budget in range(3):
                mass[gen.label_of(r.facts, regime, budget)] += post[regime] / 3
        top = max(mass)
        tied = [k for k in range(N_OPS) if mass[k] >= top - 1e-12]
        return min(tied, key=lambda k: (gen.COST[k], k)), top


def nuisance_only(train: list[Row]):
    """Naive Bayes (Laplace 1) on nuisance fields only: the six entity ids (512
    possible values, both pools) and the filler tokens (64 values)."""
    n = len(train)
    cy = Counter(r.label for r in train)
    ent = {y: Counter() for y in range(N_OPS)}
    fil = {y: Counter() for y in range(N_OPS)}
    n_ent, n_fil = Counter(), Counter()
    for r in train:
        for f in r.facts:
            ent[r.label][f[4]] += 1
            n_ent[r.label] += 1
        for t in r.tokens:
            if t >= gen.FILLER_TOKEN:
                fil[r.label][t] += 1
                n_fil[r.label] += 1

    def predict(r: Row) -> int:
        """Predict an operator using only entity IDs and filler-token counts."""
        best, best_y = None, None
        for y in range(N_OPS):
            if cy[y] == 0:
                continue
            s = math.log(cy[y] / n)
            for f in r.facts:
                s += math.log((ent[y][f[4]] + 1) / (n_ent[y] + 512))
            for t in r.tokens:
                if t >= gen.FILLER_TOKEN:
                    s += math.log((fil[y][t] + 1) / (n_fil[y] + gen.N_FILLER_TYPES))
            if best is None or s > best:
                best, best_y = s, y
        return best_y

    return predict


# --------------------------------------------------------------------------- bands


def band(ident: str, description: str, measured: float, lower=None, upper=None) -> dict:
    """Record a measurement and whether it meets the optional inclusive bounds."""
    ok = (lower is None or measured >= lower) and (upper is None or measured <= upper)
    return {"id": ident, "description": description, "measured": measured,
            "lower": lower, "upper": upper, "pass": ok}


def main(argv: list[str] | None = None) -> int:
    """Fit reference predictors, write measured G0 bands, and return their status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=gen.DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    def log(message: str) -> None:
        """Print elapsed-time progress to standard error."""
        print(f"[{time.perf_counter() - started:6.1f}s] {message}", file=sys.stderr)

    identity = data_identity(args.data)
    log(f"data FNV-1a-64 {identity['data_fnv1a64']}, lock match {identity['all_files_match_lock']}")
    train = load(args.data, "train", with_tags=False)
    splits = {s: load(args.data, s, with_tags=True) for s in EVAL_SPLITS}
    log("loaded and re-derived every split")

    stats = {"train": label_stats(train)}
    interventional = [r for r in train if r.regime == gen.INTERVENTIONAL]
    stats["train"]["causal_share_under_interventional"] = (
        sum(r.label == gen.OP["causal"] for r in interventional) / len(interventional)
    )
    for s, rows in splits.items():
        stats[s] = label_stats(rows)

    predictors = {}
    majority_predict, majority_code = train_majority(train)
    predictors["train_majority"] = majority_predict
    predictors["count_router"] = count_router(train)
    log("count router fitted")
    predictors["unbound_bag_of_attributes"] = train_logistic(feats_unbound, train)
    log("unbound bag-of-attributes regression fitted")
    predictors["bound_additive"] = train_logistic(feats_bound, train)
    log("bound additive regression fitted")
    predictors["hand_primary"] = hand_router(weighted=False)
    predictors["hand_weighted"] = hand_router(weighted=True)
    predictors["validity_blind_rule"] = validity_blind
    bayes = RawBlindBayes()
    predictors["nuisance_only"] = nuisance_only(train)

    references = {name: {} for name in predictors}
    references["raw_blind_exact_bayes"] = {}
    raw_blind_expected = {}
    for s, rows in splits.items():
        for name, predict in predictors.items():
            references[name][s] = accuracy(predict, rows)
        decisions = [bayes.decide(r) for r in rows]
        references["raw_blind_exact_bayes"][s] = sum(d[0] == r.label for d, r in zip(decisions, rows)) / len(rows)
        raw_blind_expected[s] = sum(d[1] for d in decisions) / len(rows)
    log("references evaluated")

    epi = splits["ood_compose_epi"]
    held = [r for r in epi if "heldout" in r.tags]

    def ignore_heldout(r):
        """Apply the rule after removing held-out role-state combinations."""
        return gen.label_of(r.facts, r.regime, r.budget, drop=lambda f: (f[0], f[1]) in gen.HELD_OUT)

    def heldout_as_inferred(r):
        """Apply the rule after replacing held-out epistemic states with inferred."""
        facts = [(f[0], gen.EPI["inferred"] if (f[0], f[1]) in gen.HELD_OUT else f[1], f[2], f[3], f[4]) for f in r.facts]
        return gen.label_of(facts, r.regime, r.budget)

    reg = splits["ood_compose_regime"]
    transfer = [r for r in reg if "transfer" in r.tags]

    def ignore_evidence(r):
        """Apply the rule after removing all evidence facts."""
        return gen.label_of(r.facts, r.regime, r.budget, drop=lambda f: f[0] == gen.EVIDENCE)

    def evidence_as(regime):
        """Build a predictor that assigns evidence facts the supplied regime."""
        return lambda r: gen.label_of(r.facts, r.regime, r.budget, evidence_regime=regime)

    no_transfer = {
        "ood_compose_epi": {
            "ignore_heldout": accuracy(ignore_heldout, epi),
            "heldout_as_inferred": accuracy(heldout_as_inferred, epi),
            "heldout_subset_n": len(held),
            "ignore_heldout_on_heldout_subset": accuracy(ignore_heldout, held),
            "heldout_as_inferred_on_heldout_subset": accuracy(heldout_as_inferred, held),
        },
        "ood_compose_regime": {
            "ignore_evidence": accuracy(ignore_evidence, reg),
            "evidence_as_tabular": accuracy(evidence_as(0), reg),
            "evidence_as_temporal": accuracy(evidence_as(1), reg),
            "evidence_as_textual": accuracy(evidence_as(3), reg),
            "causal_label_share": sum(r.label == gen.OP["causal"] for r in reg) / len(reg),
            "transfer_subset_n": len(transfer),
            "max_no_transfer_on_transfer_subset": max(
                accuracy(p, transfer) for p in (ignore_evidence, evidence_as(0), evidence_as(1), evidence_as(3))
            ),
        },
    }
    log("no-transfer references evaluated")

    iid = stats["test_iid"]
    bands = [
        band("test_iid_majority", "share of the most frequent label on test_iid <= 0.25", iid["majority_share"], upper=0.25),
        band("train_min_operator_share", "every operator >= 0.03 of train labels", stats["train"]["smallest_share"], lower=0.03),
    ]
    for tag in DECISIVE:
        bands.append(band(f"iid_decisive_{tag}", f"{tag}-decisive rate on test_iid in [0.12, 0.40]",
                          iid["decisive"][tag], lower=0.12, upper=0.40))
    bands += [
        band("compose_epi_heldout_decisive", "heldout-decisive rate on ood_compose_epi >= 0.25",
             stats["ood_compose_epi"]["heldout_decisive"], lower=0.25),
        band("compose_regime_regime_decisive", "regime-decisive rate on ood_compose_regime >= 0.40",
             stats["ood_compose_regime"]["decisive"]["regime"], lower=0.40),
        band("iid_exact_ties", "exact-tie rate on test_iid <= 0.02", iid["exact_tie_rate"], upper=0.02),
        band("hand_weighted", "hand-weighted accuracy on test_iid in [0.74, 0.88]",
             references["hand_weighted"]["test_iid"], lower=0.74, upper=0.88),
        band("bound_additive", "bound additive accuracy on test_iid in [0.58, 0.72]",
             references["bound_additive"]["test_iid"], lower=0.58, upper=0.72),
        band("raw_blind_ceiling", "raw-blind exact Bayes accuracy on test_iid in [0.76, 0.86]",
             references["raw_blind_exact_bayes"]["test_iid"], lower=0.76, upper=0.86),
        band("nuisance_only", "nuisance-only predictor (entity ids, fillers) on test_iid <= train-majority + 0.02",
             references["nuisance_only"]["test_iid"], upper=references["train_majority"]["test_iid"] + 0.02),
    ]
    all_pass = all(b["pass"] for b in bands)

    document = {
        "suite": "operator-routing",
        "generator_version": gen.GENERATOR_VERSION,
        "benchmark_seed": gen.BENCHMARK_SEED,
        "type_codebook_version": gen.CODEBOOK_VERSION,
        "type_codebook_fingerprint": gen.CODEBOOK_FINGERPRINT,
        "data": identity,
        "train_majority_operator": gen.OP_NAMES[majority_code],
        "label_stats": stats,
        "references": references,
        "raw_blind_expected_accuracy": raw_blind_expected,
        "no_transfer": no_transfer,
        "bands": bands,
        "all_bands_pass": all_pass,
        "prototype": PROTOTYPE,
        "definitions": {
            "exact_tie": "two or more operators within 1e-9 of the maximum z",
            "decisive": "the label changes under the tag's counterfactual (README.md, Tags)",
            "raw_blind_exact_bayes": "Bayes decision under the exact training-model posterior over (R, B) given every slot field; accuracy against the true label",
            "raw_blind_expected_accuracy": "mean over examples of the posterior mass of the Bayes decision",
            "sgd": "zero init, 5 epochs, lr 0.5 * (1 - step/total) + 0.01, per-epoch Fisher-Yates from splitmix64 stream seeded 1",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {args.out}")

    print(f"{'reference':32s}" + "".join(f"{s[:18]:>20s}" for s in EVAL_SPLITS[1:]))
    for name, values in references.items():
        print(f"{name:32s}" + "".join(f"{values[s]:20.3f}" for s in EVAL_SPLITS[1:]))
    for b in bands:
        print(f"{'PASS' if b['pass'] else 'FAIL'}  {b['id']:34s} {b['measured']:.4f}  [{b['lower']}, {b['upper']}]")
    ok = all_pass and identity["all_files_match_lock"]
    print("all G0 data bands pass" if ok else "G0 data bands: FAILURE (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
