# Q002 evidence promotion: draft preregistration

Pre-freeze text of the preregistration of Q002; at the freeze it becomes the `preregistration_text` file of `config.toml`'s
`[preregistration]` table (renamed `PREREGISTRATION.md`, as S001's was) and its SHA-256 is frozen with the table. Q002 is
`planned`: nothing here is a result and no run is evidence. The decisions are made by `criteria.toml` and applied by
`aggregate.py`; this file gives their reasons, the definitions, the modelling choices, what the harness cannot produce and the
limits of the claim. Every criterion is the text of `experiment.toml` (hypothesis, falsification, metrics) and the pins of
`config.toml`'s `[pinned_thresholds]`; the harness is `bins/ptr-bench/src/experiments/q002/` (subcommand
`evidence-promotion <seed>`); the product under test is `crates/ptr-search` (`src/source.rs` and the typed promotion path of
`SearchHit` in `src/lib.rs`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command, set at the freeze):
`cargo run --release --locked -p ptr-bench -- evidence-promotion <seed>`

Scope (owner decisions of 2026-10-09, binding): synthetic structured deterministic facts as raw-event bytes; exact source is
the byte-exact quote at the claimed span of a digest-valid raw event plus a deterministic comparison of a structured claim with
that span; retrieval is a harness retriever with injected poison; the lifecycle authority is the real
`PtrRuntime::generation_validity`; task success is the correct answer or the correct abstention, also reported on the
answerable questions alone. The claim: driven through its public API, the typed promotion path of `ptr_search` never promotes a
poisoned hit to `VerifiedKnown`, refuses every adversarial stage skip with the refusal the stage rules prescribe, keeps the
honest hits (at least 950 per mille promoted) and answers at least as many questions correctly as direct retrieval-to-answer.
The existing boolean API (`observe(.., true)`, `promote_verified(true)`) is kept as the documented trusted-caller path and is the
negative control B2.

## 1. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written
`>= N`, a "> 0" violation `<= 0`). COMPLETED iff every seed meets every coverage rule and trips no falsifier **and the mutation
evidence is complete**: `results/mutations.json` is present for this checkout's code and plan and `killed == total ==` the
number of mutations `tests/mutations.toml` lists (24). FALSIFIED if a coverage rule (the run is vacuous) or a falsifier fails
on any seed; a falsification is never downgraded by the mutation evidence (`mutations_ok` is reported beside it). INVALID if a
record is missing, does not describe the frozen run, a counter is missing or mistyped, the exit status disagrees with the
rules, the baseline control B0 did not hallucinate, or the mutation evidence is missing or incomplete (INVALID says nothing
about the hypothesis). No rule changes after a run.

The harness exits 0 when every rule holds, 1 when one fails, 2 when it cannot run (section 7). **A defect of the code under test
is never exit 2**: a refusal with the wrong reason, a promotion, an accepted stage skip, a panic (every call into `ptr_search`
is guarded by `catch_unwind` and counted in `product_panics`) are counters and falsifiers.

Coverage (the run is vacuous if unmet), 25 rules:

- `poison_attempts.P1` .. `poison_attempts.P8` `>= 20` each: hits of that class that the mechanism refused with the refusal the
  generator expects for the class, counted from the call's return at the API boundary;
- `true_claims >= 100` (honest hits shown to the mechanism), `unanswerable_questions >= 50`;
- `stage_skip_by_kind.K1` .. `K10` `>= 20` each: adversarial attempts that got the refusal the stage rules prescribe and left
  the hit unchanged;
- `false_known_baseline_direct`, `false_known_gen_filter_only`, `false_known_bool_trust` `>= 1` each (the negative controls B0,
  B2 and the contrast B1: a baseline that never hallucinates on the corpus shows nothing);
- `oracle_label_mismatches <= 0`, `corpus_selection_digest_mismatches <= 0` (harness-integrity guards, section 5).

Falsifiers, 8 rules: `false_known_mechanism <= 0`; `stage_skip_accepted <= 0`; `stale_or_revoked_promoted <= 0`;
`stage_skip_unexpected <= 0`; `wrong_refusal_reason <= 0`; `product_panics <= 0`; `recall_permille >= 950` (owner decision);
`task_success_mechanism >= task_success_baseline` (owner decision; counted over all questions, correct answer or correct
abstention; the answerable-only reading is `answerable_success_permille`, reported and not judged).

Rules that hold by construction or duplicate another, named so that nobody reads them as evidence: `unanswerable_questions >=
50` counts generated questions; `true_claims >= 100` counts generated honest hits; `oracle_label_mismatches` and
`corpus_selection_digest_mismatches` guard the harness; `stale_or_revoked_promoted` overlaps `false_known_mechanism` on P6 to
P8 by design (a stale, revoked or disputed hit that is promoted is both); `recall_violation` restates the recall rule as a count
for the mutation checker; `recall_permille` is 1000 for a sound mechanism because the honest hits are exact (section 9).

## 2. The world

A seeded generator (`world.rs`; `experiments/rng.rs` SplitMix64) writes, per seed, 36 capsules: 26 healthy, 5 revoked, 5
disputed, each with one to three generations (balanced), each generation one raw event of one-line statements in the format
`subject: predicate is [not ]value`. Subjects are `ent-NNNN`, predicates one of six, values `v<predicate><kk>` (fixed width).
Every fact (subject, predicate) is owned by exactly one capsule, six per capsule. A healthy capsule's live generation is
authoritative; a revoked or disputed capsule's facts have no authoritative value. Older generations state a changed value with
probability 0.7 (stale). Quarantined (revoked and disputed) events also state four **mirror** statements about facts of healthy
capsules with a wrong value; live healthy events also state two **negated** statements about their own facts. Sixty entities
have a **twin** whose name only looks like theirs (an ASCII O for a zero, or a Cyrillic letter for a Latin one); both twins own
a fact of the same predicate, always in healthy capsules.

Questions (one per fact, in `world.rs`): 156 answerable (a fact of a healthy capsule: expected answer its live value), 60
quarantined (a fact of a revoked or disputed capsule: no authoritative answer, abstain), 40 absent (a fact nobody states:
abstain). Hits per question, ranked by a seeded score (honest 400..900, poison 300..950, so poison tops the ranking often):

| class | hit | what a correct mechanism says |
|---|---|---|
| honest | the live statement of the fact's owner, exact citation, exact claim (156) | promote |
| P1 | a source id no store holds (`raw-fab-NNNN`), fabricated claim; every absent question gets one (60) | `UnknownRawEvent` |
| P2 | wrong span: the span shifted by 1..3 bytes with the real statement as quote (24); the span reaching past the end (24); the right span with a doctored quote (value replaced, the first eight bytes intact) (24) | `QuoteMismatch`, `SpanOutOfRange`, `QuoteMismatch` |
| P3 | a corrupt replica of an event (a value changed, the original's digest kept), citing the changed bytes, claim equal to the changed value (40) | `DigestMismatch` |
| P4 | the statement about the question's twin, claimed under the question's subject (48) | `SubjectMismatch` |
| P5 | the right statement with a claim of another value (30); a negated statement claimed as affirmative (30) | `ValueMismatch`, `NegatedStatement` |
| P6 | statements of superseded generations, exact (60) | `StaleGeneration` |
| P7 | every statement (own and mirror) of the live generation of each revoked capsule, exact (50) | `RevokedGeneration` |
| P8 | the same for disputed capsules (50) | `DisputedGeneration` |

Each class has exactly one defect, so exactly one barrier refuses it; a class with fewer candidates than its quota gets all of
them. The corpus (shape, events, replicas, questions, hits with every field the generator decided) is hashed (SHA-256) and the
digest of each seed is pinned in `config.toml` (`corpus_digest_seed_N`); `corpus_selection_digest_mismatches` is 1 when a run at the
pinned shape differs (it is a harness guard against drift of the generator after the freeze; the freeze re-measures the digests
on the frozen commit).

## 3. The mechanism and the arms

The mechanism M, per hit, through the public API of `ptr_search` only: `SearchHit::new`, `mark_possible`;
`ExactSourceResolver::resolve(citation)` against the source (the store of record, plus the corrupt replicas, section 8);
`observe_resolved(resolved, validity)` with the lifecycle authority as `validity`; `verify_claim(claim, resolved)`;
`promote_verified_with(proof)`. M processes every hit of every question independently; the best-ranked promoted hit gives the
answer (its verified claim's value), no promoted hit means abstention.

- **B0 direct retrieval-to-answer** (the manifest baseline): the top hit of the question is the answer. Runs no code under test.
- **B1 lifecycle filter only**: `retain_live` over the hits, then the top survivor. A contrast, not a control.
- **B2 boolean trust**: today's API fed `true`: `mark_possible`, `observe(validity, true)`, `promote_verified(true)`, the first
  hit the lifecycle lets through is the answer. The API is documented as the trusted-caller path; B2 is what a caller does who
  hands it a retriever's output.
- **M** as above.

**False Known** is defined by the generator's label: a promoted hit is false iff its class is not honest (its source is not the
exact quote of an intact event, or its claim is not what the statement states, or its generation is not the live generation of a
healthy capsule). The baselines make one Known assertion per question (their top hit), the mechanism is counted on every hit it
promotes. `false_known_untrue_*` (internal) counts the subset whose claim is also false in the world: a P2 hit with the right
quote at a wrong span has a true claim and is still a false Known, because it has no exact source. The label, not the truth of
the claim, is what the mechanism's promise is about.

**The lifecycle authority** is a real `PtrRuntime` (`authority.rs`): the harness commits `CapsuleCommitted`,
`CapsuleSuperseded` and `Revoked` events and asks `generation_validity` for every question (owner decision). The runtime never
answers `Disputed`, so P8 is the one class whose lifecycle comes from the harness: the generations the generator marked
disputed are answered `Disputed` by an overlay and every other by the runtime. What P8 shows is that the mechanism refuses a
`Disputed` answer, not that a runtime produces one.

## 4. Oracle and cross-check

The truth is the generator's table: the class of every hit (with the refusal it must get) and the authoritative answer of every
question. It is cross-checked, before the arms run, by a **naive independent reading** of the finished corpus (`oracle.rs`,
which calls nothing of `ptr_search`): a hit is eligible iff its quote occurs in the pristine event text exactly at the cited span
(`match_indices`), the quote equals the canonical affirmative rendering of the claim (string equality), and the generator's own
lifecycle table says the generation is the live one of a healthy capsule; an answer is the value of the single affirmative
statement of the fact in the live events of healthy capsules. Each hit or question on which the reading and the label disagree is
one `oracle_label_mismatches`. This is a cross-check of two readings of one harness, not an external oracle; the grammar can only
express a disagreement as a failed coverage rule (the seed is vacuous), so a disagreement does not exit 2 and the freeze review
must see it at 0.

## 5. Metrics (JSON integers; grouped metrics are objects of integers)

| metric | definition |
|---|---|
| questions, answerable_questions, unanswerable_questions | generated questions (156 + 100 = 256) |
| hits_by_class.{H,P1..P8} | hits of the class shown to M |
| poison_attempts.P1..P8 | class-k hits M refused with the expected refusal (coverage) |
| true_claims, true_known_promoted, recall_permille | honest hits shown to M / promoted / 1000 x promoted / shown (falsifier >= 950); recall_violation = 1 iff below |
| false_known_mechanism (hard) | M-promoted hits with a non-eligible label |
| false_known_baseline_direct, _gen_filter_only, _bool_trust | the same for B0, B1, B2 (one Known per question) |
| stale_or_revoked_promoted (hard) | M-promoted hits of P6, P7, P8 |
| wrong_refusal_reason (hard) | poison hits M refused with another refusal than the class's (the pairs are in `internal.wrong_refusal.*`) |
| product_panics (hard) | calls into `ptr_search` that panicked (guarded; the hit then counts neither as refused nor as promoted) |
| stage_skip_attempts, stage_skip_by_kind.K1..K10 | adversarial attempts that got the prescribed refusal with the hit unchanged (section 6) |
| stage_skip_accepted (hard) | adversarial calls that returned `Ok` |
| stage_skip_unexpected (hard) | adversarial calls with another refusal, with the hit changed, or whose setup from honest hits failed |
| task_success_mechanism / _baseline / _gen_filter_only / _bool_trust | questions answered correctly or correctly abstained on, per arm (baseline = B0) |
| answerable_success_permille, answerable_success_baseline_permille | descriptive: successes on answerable questions of M and B0 |
| abstentions_correct | M abstentions on unanswerable questions (descriptive) |
| oracle_label_mismatches, corpus_selection_digest_mismatches (hard) | section 4, section 2 |
| resolver_calls, lifecycle_queries | calls observed at the boundary (descriptive) |

`hard_failures` is the sum of the nine hard counters named in `tests/mutations.toml`; the wall-clock time is in `timing_ns` and
never judged.

## 6. Adversarial stage-skip attempts

On honest hits (and, for K9, K10, hits of generations that are not live), through the public API, each kind 24 times (a
seeded cycle through the honest hits): K1 promote a retrieval candidate with a valid proof; K2 promote possible evidence that
was never observed; K3 observe a candidate that was never marked possible; K4 observe an already observed hit with another
resolution; K5 step an observed or a known hit back with `mark_possible`; K6 promote an observed hit with a valid proof made for
another source (a forged resolution); K7 promote a known hit again; K8 promote with a proof a hit observed on the caller's
word (no source held); K9 launder a hit of a superseded, revoked or disputed generation with the valid resolution of an honest
one; K10 promote that hit afterwards. Each expects `WrongStage`, `ClaimProofMismatch`, `SourceNotResolved` or the lifecycle
refusal of the hit's generation, with the hit's stage, source and claim unchanged. An attempt is counted only on that outcome.
`ResolvedSource` and `ClaimProof` cannot be constructed outside `ptr_search` (private fields, compile-fail doctests); the
attempts therefore replay or mis-pair genuine proofs, they do not forge one. There is no run-time negative control of this
counter besides the mutation plan (the stage checks removed, the proof held against the wrong source).

## 7. Exit 2 (the harness cannot run), and nothing else

A shape that does not generate; the runtime or the store refusing the world's own setup (`PtrRuntime::commit`,
`KnowledgeStore::append_raw`: neither is under test); the negative control B0 producing no false Known (it runs no code under
test); an argument that does not parse. B2 runs the boolean API of the code under test, so its failing to hallucinate is not exit
2: it fails the coverage rule `false_known_bool_trust >= 1`. Anything the code under test does wrong is a counter.

## 8. Weak metrics and what is not measured

- Recall is 1000 for a sound mechanism: honest hits are exact by construction, so the 950 floor guards against over-refusal, not
  against garbled evidence; no claim about a real retriever's hits.
- P3 cites **corrupt replicas** served beside the store by a `RawEventSource` double, because `KnowledgeStore::append_raw`
  re-checks digests and a store cannot hold such an event; the resolver's digest check is defence in depth against a damaged
  store and this is the only way to reach it.
- A proof is **not tied to the capsule** a hit names (the store records no event-to-capsule mapping): a hit paired with the
  resolution of another capsule's event is not caught by the typed path; the lifecycle is asked about the hit's own capsule
  generation, which is why P6 to P8 are refused whatever event they cite. Neither a stage-skip kind nor a poison class pairs a hit
  with a foreign capsule's event, so Q002 says nothing about that gap.
- The lifecycle is read once per hit at observation time; `promote_verified_with` does not ask again (documented in the API).
- The negative controls (B0, B2) demonstrate that the corpus can fool a direct or boolean-trusting caller, not that M catches a
  defective implementation; that evidence is the mutation plan.
- Defence in depth: the resolver vouches for the quote, the verifier for the claim; a defect in one is absorbed by the other
  when a poison class reaches both. The plan contains the mutations the classes reach through exactly one barrier; a mutation
  that weakens a barrier no class tests through alone (an empty or out-of-range span accepted) is equivalent and not listed.
- `task_success_gen_filter_only`, `task_success_bool_trust`, the answerable-only permille, abstentions and wall time are
  descriptive.

## 9. Mutation plan

`tests/mutations.toml` (`python3 scripts/mutation_check.py Q002`; `--check` verifies the anchors; invoked as
`ptr-bench evidence-promotion 36 17`, the pinned world at seed 17, `"scale":"custom"`) plants 24 defects one at a time: 19 in
`ptr-search` (the resolver skips the digest check / compares only a quote prefix / ignores the offset / accepts unknown raw
events / panics on a span past the end; the verifier skips the subject compare / the value compare / treats negation as a match /
reports a subject mismatch as a value mismatch / over-refuses honest claims; `observe` accepts a Revoked, a Disputed, a
Superseded generation / misreports the wrong stage / has no stage check; `mark_possible` has no stage check; `promote_verified_with`
has no stage check / ignores the proof's source / promotes without a held source) and 5 in the harness (the authority revokes the
wrong generation; the label table calls a lookalike eligible; the K6 driver uses the hit's own proof; the generator drifts from
the pinned corpus; the expected refusal of P7 is wrong). Every hard counter is expected by at least one; a mutation counts as
killed only when the run exits 1 and one of its expected counters fired. COMPLETED needs `killed == total`.

## 10. Open points for review

See the report that accompanied this draft: the freeze re-measures the corpus digests, a decision on whether to keep the digest
pin (it makes every generator change a re-pin), and whether the proof should be tied to the capsule (an API change).
