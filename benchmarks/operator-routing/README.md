# operator-routing

Reusable benchmark contract. Keep test data isolated from training.

This directory holds **operator-routing v1** (`generator_version = 1`, benchmark
seed `20260925`): a synthetic 11-way routing task over six typed facts, built for
the A0 mechanism study (research/falsification/A0-ablations-v1). This README is its
specification. Everything is Python standard library (3.11+) and deterministic:
the same bytes on every run.

| File | Role |
|---|---|
| `generator.py` | PRNG, sampling, label rule, tags, JSONL and TSV writers, `--check` |
| `score.py` | Independent label re-derivation (shares no code with the generator), `agree` and `score` modes |
| `references.py` | Every reference line and the G0 data bands; writes `research/falsification/A0-ablations-v1/references.json` |
| `suite.toml` | Metric definitions, split sizes, generator version, codebook fingerprint |
| `splits.lock.json` | SHA-256 of every generated file, the data FNV-1a-64, per-split label FNV-1a-64, prefix digests |
| `tests/` | `python3 -m unittest discover -s benchmarks/operator-routing/tests` |

## Regenerate and check

```sh
python3 benchmarks/operator-routing/generator.py            # write datasets/generated/operator_routing_v1/ (about 12 s)
python3 benchmarks/operator-routing/generator.py --check    # regenerate in memory, compare with splits.lock.json; exit 1 on any difference
python3 benchmarks/operator-routing/score.py agree          # score.py vs generator on 100% of records (about 6 s)
python3 benchmarks/operator-routing/references.py           # references + G0 bands -> references.json (about 15 s); exit 1 if a band fails
python3 -m unittest discover -s benchmarks/operator-routing/tests   # about 13 s with the data on disk, 4 s without
python3 benchmarks/operator-routing/score.py score experiments/model/M00x-*/results/run-*.json [--out FILE]
```

`generator.py` without flags refuses (exit 1) if what it wrote differs from the
lock; `--update-lock` re-pins it deliberately. `--write-sample` rewrites
`datasets/samples/operator_route_v1.jsonl` (example 0 of every split). The output
directory `datasets/generated/operator_routing_v1/` is git-ignored by
`/datasets/generated/*`; only the lock is committed.

## PRNG

All arithmetic is on unsigned 64-bit integers, wrapping.

```
GAMMA = 0x9E3779B97F4A7C15
mix64(z):  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
           z = (z ^ (z >> 27)) * 0x94D049BB133111EB
           return z ^ (z >> 31)                      # the finalizer of crates/ptr-types/src/slot_encoding.rs
stream(seed): state = seed
next():    state = state + GAMMA; return mix64(state)
splitmix64(x) = mix64(x + GAMMA)                     # one output of a stream whose state starts at x
                                                     # (identical to splitmix64() in slot_encoding.rs)
fnv1a64(bytes): h = 0xCBF29CE484222325; for b in bytes: h = (h ^ b) * 0x100000001B3
```

- Example `j` (0-based) of split `s` draws only from `stream(splitmix64(20260925 ^ fnv1a64(utf8(s)) ^ splitmix64(j)))`.
- `u01() = (next() >> 11) / 2^53`, an exact float64 division.
- `randbelow(n) = floor(u01() * n)`: one `next()`, a float64 multiply by `n` as a float64, then floor. It is always below `n`.
- Python's `random` module is not used.

Pinned vectors (tested): `stream(0)` gives `e220a8397b1dcdaf, 6e789e6aa1b965f4, 06c45d188009454f`;
`fnv1a64("train") = dee795a6c5087209`; `seed("train", 0) = 0e0645255755a741`, whose stream starts
`607d08c136d17281, 0a5ad4c5a0f9e481, c3ddcfb2e72a2caa`; `seed("test_iid", 1) = 81d60976a260ed72`;
`seed("ood_payload", 2999) = 6439903d016197d9`.

## Codes

All integer codes come from `datasets/generated/codebook.json` (version 1,
fingerprint `2b6f8175a7a7bb648910e7acb17dcfcff5149834f2fe87355c5d2c4524c113a3`); the
generator refuses to run against any other fingerprint. JSONL uses the codebook's
member names (`external_pod`, not `ExternalPod`).

| Family | Codes |
|---|---|
| role `r` | goal 0, constraint 1, claim 2, evidence 3, resource 4, capability 5, relation 6, procedure 7, action 8 |
| epistemic `e` | unknown 0, assumed 1, hypothesis 2, observed 3, inferred 4, verified 5 |
| operator `k` | semantic 0, deductive 1, probabilistic 2, statistical 3, temporal 4, causal 5, search 6, optimization 7, simulation 8, symbolic 9, external_pod 10 |
| validity | live 0, superseded 1, revoked 2, disputed 3 |
| regime `R` (design) | tabular 0, temporal 1, interventional 2, textual 3 |
| budget `B` (design) | 0 -> 0.4, 1 -> 0.7, 2 -> 1.0 |

## Sampling one example

In exactly this draw order, from the example's own stream:

1. `R = randbelow(4)`; in `ood_compose_regime`, `R = interventional` **without a draw**. `B = randbelow(3)`.
2. `ban = (R == interventional and split != ood_compose_regime)`.
3. Two focus roles, each `ROLE_PRIOR[randbelow(11)]`, redrawn while `ban` and the role is Evidence.
   `ROLE_PRIOR = [goal, constraint, claim, claim, evidence, evidence, resource, capability, relation, procedure, action]`.
4. Six facts `i = 0..5`. Each repeats until accepted:
   `role = focus[randbelow(2)] if u01() < 0.5 else ROLE_PRIOR[randbelow(11)]`; `e = randbelow(6)`;
   reject if `(role, e)` is in HELD_OUT (not checked in `ood_compose_epi`) or if `ban` and role is Evidence.
   Then, in this order: `c = (randbelow(1000) + 0.5) / 1000`; Live iff `u01() < p_live`
   (0.78; 0.50 in `ood_validity`), otherwise `validity = [superseded, revoked, disputed][randbelow(3)]`;
   `entity = randbelow(256)` (+256 in `ood_payload`).
5. Reject the whole example if no fact is Live, or if the split's acceptance rule fails
   (below). A rejected example restarts at step 1 **on the same, continuing stream**.
6. Raw tokens (below): fillers are drawn after acceptance, then the sequence is shuffled.

`cb = floor(5c)` is never ambiguous because `c = (k + 0.5)/1000` sits 0.0025 from
every bucket edge; `c` is written with 4 decimals and parses back to the same float64.

HELD_OUT `(role, e)`: (goal, verified), (constraint, hypothesis), (claim, observed),
(evidence, assumed), (resource, inferred), (capability, unknown), (relation, verified),
(procedure, hypothesis), (action, observed).

## Raw tokens

Vocabulary 216, shared by every arm.

| Ids | Meaning |
|---|---|
| 0 | PAD (raw-blind replaces every token with it) |
| `1 + 24i + offset`, i = 0..5 | fact i's attributes: role `0..8`, e `9..14`, cb `15..19`, validity `20..23` (codebook codes as offsets) |
| 145..148 | regime `145 + R` |
| 149..151 | budget `149 + B` |
| 152..215 | 64 fillers |

The unshuffled sequence is: for i = 0..5 the four tokens `role, e, cb, validity` of
fact i; then the regime token; then the budget token; then F fillers
`152 + randbelow(64)` (F = 10, T = 36; in `ood_distractors` F = 46, T = 72). It is
then shuffled in place by Fisher-Yates (Durstenfeld): for `i = T-1` down to `1`,
`j = randbelow(i + 1)`, swap `a[i]` and `a[j]`.

## Label rule

Computed in float64 over facts in slot order, starting from `z_k = 0.0`. Each fact
that is Live **and** has `cb > 0` ("usable") adds, for every k,

```
z_k += G[cb] * (W[e] * A[r,R]_k + (U[e] if k == probabilistic else 0.0)) - LAMBDA * max(0.0, COST_k - BUDGET_B)
```

in exactly that operation order. `label = argmax_k z_k`; every k with
`z_k >= max(z) - 1e-9` is tied; ties go to the lowest COST, then the lowest code. A
fact that is not Live or has `cb = 0` adds nothing, not even the budget penalty. With
no usable fact, all 11 operators tie at 0 and the label is semantic.

| Constant | Value |
|---|---|
| G (by cb 0..4) | 0, 0.5, 1.0, 1.0, 1.25 |
| W (by e) | unknown 0.30, assumed 0.55, hypothesis 0.80, observed 1.30, inferred 1.05, verified 1.55 |
| U (by e) | unknown 0.35, assumed 0.60, hypothesis 0.85, observed 0, inferred 0, verified 0 |
| LAMBDA | 1.0 |
| COST (by k) | semantic 0.2, deductive 0.3, probabilistic 0.5, statistical 0.5, temporal 0.5, causal 0.6, search 0.6, optimization 0.8, simulation 0.8, symbolic 0.4, external_pod 0.9 |
| BUDGET (by B) | 0.4, 0.7, 1.0 |
| A | 2.0 on the role's primary operator, 0.9 on its secondary, 0 elsewhere |

`op(R)` = (statistical, temporal, causal, semantic)[R].

| Role | Primary | Secondary |
|---|---|---|
| goal | search | optimization |
| constraint | optimization | symbolic |
| claim | deductive | op(R) |
| evidence | op(R) | probabilistic |
| resource | external_pod | search |
| capability | simulation | external_pod |
| relation | causal | deductive |
| procedure | symbolic | simulation |
| action | external_pod | temporal |

`margin` = the largest z minus the second largest (0 on an exact tie).

### Tags

A tag is set when the counterfactual changes the label (same tie rule):

| Tag | Counterfactual | Splits |
|---|---|---|
| validity | every fact is admitted (treated as Live); the cb > 0 gate still applies | all |
| regime | any other R (3 alternatives) | all |
| budget | any other B (2 alternatives) | all |
| confidence | G = 1 for every bucket, so cb = 0 facts become usable too (vote and penalty) | all |
| epistemic | W = 1 and U = 0 for every e | all |
| heldout | facts with a held-out pair are removed (no vote, no penalty) | `ood_compose_epi` only |
| transfer | the label differs from **all four** no-transfer treatments of Evidence: Evidence facts removed; or Evidence voting with `A[evidence, R']` for R' = tabular, temporal, textual (Claim still reads the true R; gain and penalty unchanged) | `ood_compose_regime` only |

## Splits

Every example of every split comes from its own stream, so splits are independent
and any prefix can be regenerated alone.

| Split | n | Change relative to train |
|---|---|---|
| train | 16,000 | - |
| val | 2,000 | none (IID) |
| test_iid | 3,000 | none (IID) |
| ood_compose_epi | 3,000 | HELD_OUT check skipped; accepted only if some Live fact with cb > 0 carries a held-out pair |
| ood_compose_regime | 3,000 | R = interventional (no draw), Evidence allowed (no ban); accepted only if some Live Evidence fact has cb > 0 |
| ood_distractors | 3,000 | F = 46 fillers, T = 72 |
| ood_validity | 3,000 | p_live = 0.50 |
| ood_payload | 3,000 | entity ids 256..511 |

S = 6 slots in every split.

## File formats

`datasets/generated/operator_routing_v1/<split>.jsonl` and `<split>.tsv`, UTF-8, LF,
one example per line, a final newline, no header, in example order.

**TSV** (what the Rust loader reads): six tab-separated fields

```
id \t label_code \t R \t B \t tok,tok,...,tok \t r,e,c,validity,entity;r,e,c,validity,entity;...
```

`id` is `<split>-<index, 5 digits>`; codes are the integers above; `c` has exactly 4
decimals (`%.4f`); tokens are in shuffled order; facts are in slot order.

**JSONL** (compact JSON, keys in this order; extends `datasets/samples/operator_route.jsonl`
and validates against `datasets/schemas/operator_route.schema.json`):
`id`, `split`, `generator_version` (1), `task` (rendered English), `routes` (all 11
operators in code order, `target = softmax(z / 0.5)` rounded to millionths with the
rounding residual on the largest entry, so targets sum to exactly 1), `cost_budget`,
`type_codebook_version` (integer 1), `type_codebook_fingerprint`, `regime`
`{code, name}`, `budget` `{code, value}`, `slots[6]` `{index, role, epistemic,
confidence, confidence_bucket, validity, entity, provenance_bucket}` (names from the
codebook; `provenance_bucket = index`), `raw_tokens`, `label` `{operator, code}`,
`utility[11]` (z rounded to 10 decimals), `margin` (rounded to 10 decimals), `tags`
(in the order validity, regime, budget, confidence, epistemic, heldout, transfer).

## Digests (splits.lock.json)

- `sha256` of every JSONL and TSV file, with its byte count.
- `data_fnv1a64`: FNV-1a-64 over the raw bytes of the eight TSV files **concatenated in
  the order** train, val, test_iid, ood_compose_epi, ood_compose_regime, ood_distractors,
  ood_validity, ood_payload, with nothing between them (each file already ends in `\n`).
  Written as 16 lowercase hex digits. Currently `0ad71688f09b0d0d`. The Rust entrypoint
  pins this literal.
- `label_fnv1a64` per split: FNV-1a-64 over the ASCII string of one lowercase hex digit
  (`0`-`9`, `a`) per example, in file order - the gold labels in PRED-string form.
- `tsv.fnv1a64` per file, for diagnosis.
- `prefix`: SHA-256 of the first 64 lines of each file, which the tests regenerate.

## Metrics (score.py)

`score.py score` reads run records (JSON files with a `stdout` string, as written by
`scripts/run_experiment.py`), finds `PRED <arm> <split> <hex>` lines (one hex digit per
example, case-insensitive, must be 0..a and exactly n long) and reports per (record,
arm, split):

- `route_accuracy` = mean [pred == label] (primary);
- `cost_adjusted_regret` = mean (z_label - z_pred), from its own re-derived z;
- `task_success` = mean [z_label - z_pred <= 0.25 + 1e-9];
- subset accuracies `{n, correct, accuracy}`: each decisive tag, `clear_margin`
  (margin >= 0.25 - 1e-9), `heldout` on ood_compose_epi, `transfer` on ood_compose_regime;
- `rust`: the binary's final row for that (arm, split) - a JSON row whose `arm` and
  `split` fields name it and that has a numeric `correct` - with `n, correct, accuracy,
  nll, ece15` passed through, and `rust_count_agrees` (gate G5). A disagreement makes
  the exit status 1.

`score.py agree` re-derives every label, utility, margin and tag from the TSV fields
and compares them, and every slot/token field, with the JSONL (100% of records).

## References and G0 bands (references.py)

Measured on the generated splits (prototype values from the design in parentheses).

| Reference | test_iid | compose_epi | compose_regime | validity |
|---|---|---|---|---|
| train-majority (deductive) | 0.194 (0.200) | 0.239 (0.238) | 0.073 (0.069) | 0.194 (0.195) |
| count router (naive Bayes) | 0.344 (0.338) | 0.303 (0.306) | 0.408 (0.431) | 0.335 (0.339) |
| unbound bag-of-attributes linear | 0.600 (0.583) | 0.594 (0.575) | 0.342 (0.340) | 0.484 (0.508) |
| bound additive linear | 0.651 (0.651) | 0.638 (0.613) | 0.362 (0.377) | 0.641 (0.652) |
| hand-primary | 0.644 (0.652) | 0.666 (0.648) | 0.592 (0.607) | 0.694 (0.702) |
| hand-weighted | 0.806 (0.812) | 0.806 (0.790) | 0.776 (0.783) | 0.851 (0.853) |
| validity-blind rule | 0.811 (0.792) | 0.825 (0.826) | 0.826 (0.822) | 0.574 (0.601) |
| raw-blind exact Bayes (training posterior) | 0.821 (0.815) | 0.852 (0.834) | 0.439 (0.449) | 0.852 (0.851) |
| nuisance-only (entity ids, fillers) | 0.171 | 0.187 | 0.079 | 0.158 |

No-transfer references: compose_epi ignore held-out 0.650 (0.650), held-out as inferred
0.839 (0.835); compose_regime ignore Evidence 0.460 (0.468), Evidence as tabular 0.484
(0.496), temporal 0.455 (0.465), textual 0.466 (0.479); Causal share 0.528 (0.534). Every
no-transfer reference scores 0 on the transfer subset (1,352 examples, 45.1%), and
ignore-held-out scores 0 on the heldout subset (1,050, 35.0%), by construction.

Every G0 data band passes: test_iid majority 0.194 (<= 0.25); smallest train operator
0.0515 (>= 0.03); IID decisive validity 0.189, regime 0.203, budget 0.210, confidence
0.262, epistemic 0.263 (each in [0.12, 0.40]); heldout-decisive on compose_epi 0.350
(>= 0.25); regime-decisive on compose_regime 0.587 (>= 0.40); IID exact ties 0.0153
(<= 0.02); hand-weighted 0.806 ([0.74, 0.88]); bound additive 0.651 ([0.58, 0.72]);
raw-blind ceiling 0.821 ([0.76, 0.86]); nuisance-only 0.171 (<= 0.194 + 0.02).
`references.json` carries every measured value unrounded, with its pass/fail.

## Decisions made while implementing

Where the design was silent or ambiguous, this is the reading taken:

1. **Regime draw in ood_compose_regime.** `R = interventional` consumes no draw (as the prototype).
2. **Rejection.** A rejected fact redraws from step 4's top; a rejected example restarts at step 1 on the same continuing stream. Fillers and the shuffle are drawn once, after acceptance.
3. **Draw order inside a fact.** role (u01, then randbelow), e, then c, Live/validity, entity - the order the design lists them.
4. **Non-Live validity** uses the codebook codes of superseded/revoked/disputed (1/2/3).
5. **Unshuffled token order** is fact-major (role, e, cb, validity per fact), then regime, budget, fillers; Fisher-Yates runs from the top index down.
6. **Confidence tag.** "G is set to 1" is read as G = 1 for every bucket including cb = 0, so cb = 0 Live facts become usable (vote and penalty), as in the prototype. The validity tag keeps the cb > 0 gate.
7. **Removal counterfactuals** (heldout; transfer's "Evidence ignored") remove the fact entirely, penalty included. "Evidence voting as R'" changes only Evidence's affinity; Claim still uses the true R.
8. **heldout and transfer** are computed only in their own splits.
9. **Tolerances.** Ties use `z_k >= max - 1e-9`. The same 1e-9 is applied to the 0.25 thresholds: clear margin is `margin >= 0.25 - 1e-9`, task success `regret <= 0.25 + 1e-9`, because margins that are exactly 0.25 in real arithmetic occur and float64 can land either side.
10. **JSONL numbers.** Route targets are rounded to millionths (residual on the largest entry) so the JSONL digest cannot depend on libm's last bit and the validator's sum-to-1 check holds; `utility` and `margin` are rounded to 10 decimals. The TSV holds only integers and 4-decimal `c`, so its FNV is platform-independent. Consumers recompute z; they never read these.
11. **Names.** Role, epistemic, validity and operator names are the codebook's member names; `regime` and `budget` are objects `{code, name}` and `{code, value}`; ids are `<split>-<5-digit index>`.
12. **Label FNV** (not defined by the design) is FNV-1a-64 over the gold labels as a PRED hex string; the data FNV is the plain concatenation in split order above.
13. **`--check`** compares regenerated bytes with the lock, and also the files on disk when they exist (a missing file is reported but not an error, so CI without the data can run it).
14. **"test_iid majority"** is the share of the most frequent label on test_iid (it equals the train-majority predictor's accuracy here: both are deductive, 0.194).
15. **SGD references.** "Seed 1" is a splitmix64 stream seeded 1 driving one Fisher-Yates reshuffle of a running permutation per epoch (not Python's `random`); zero init, lr = 0.5(1 - step/total) + 0.01, features as the prototype (bound: role/e/cb histograms of Live facts normalised by their count, plus R, B, bias; unbound: the same over all six facts plus a live/non-live histogram). Prediction ties go to the lowest code, as do count-router ties. hand-primary and hand-weighted use the rule's own tie-break (1e-9, lowest COST, then code), so with no usable fact they predict semantic.
16. **Raw-blind Bayes.** The posterior uses (role, e) only (c, validity, entity and the no-Live rejection are independent of R and B; B is uniform). Held-out pairs have zero likelihood under every regime of the training model and are dropped before normalising (compose_epi). The decision maximises posterior label mass; masses within 1e-12 tie and go to lowest COST, then code. The band uses accuracy against the true label; the expected accuracy (mean posterior mass of the decision) is also recorded (test_iid 0.821).
17. **Nuisance-only predictor.** Naive Bayes, Laplace 1, class prior plus the six entity ids (512 possible values) and every filler token (64 values), trained on train; the band compares its test_iid accuracy with the train-majority accuracy + 0.02.
18. **Bands** are inclusive and compared unrounded.
19. **References** are evaluated on val and all six test splits.

## Not in this directory

The Rust loader (model/burn-a0/examples/a0_ablation) re-derives labels from the TSV and
pins `data_fnv1a64`; the aggregator and the driver live under scripts/. The tests here
are not yet wired into CI or the Makefile.
