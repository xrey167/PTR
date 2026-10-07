# R012 approximate-taint-containment: DRAFT preregistration

Status: DRAFT. Not frozen, not in `experiments/preregistration.toml`, no results. It becomes
`PREREGISTRATION.md` (a `file` key of `config.toml`'s `[preregistration]` table, with `criteria.toml` and
`aggregate.py`) at the freeze, which happens only after the harness smoke test passes. The entrypoint
(`experiment.toml`) is empty until then. Format and pin pattern: R005.

Scope: the Approximate half split out of R005 on 2026-10-06. The producer is a harness-only stub
(`bins/ptr-bench/src/experiments/r012/stub.rs`, subcommand `approximate-taint-containment <seed>`) that
runs a genuine cold prefill on the reference backend and labels the report `KvReuseMode::Approximate`.
This experiment tests containment of a labelled Approximate result, not any real Approximate producer.

## 1. Definitions

- **Hop**: one derivation step, that is one run whose input includes a segment derived from a previous
  run's output (a Derived segment minted by the runtime from that output). Hop 1 is the Approximate
  run's own output being re-fed; `candidate_taint_hops` is the longest chain of such steps, checked
  with the taint intact at its end. The Approximate run itself is hop 0 and is not counted.
- **Attempt after persistence, before recompute**: a coverage counter (`*_attempts`) counts an attempt
  only if (a) the `KvTaintCommitted` record for the subject has been appended to a real `FileLedger`
  (the append returned) and (b) no cold Exact recomputation with the same recipe digest has completed.
  Attempts made earlier or later are executed but not counted.

## 2. Pinned values (copied from R005 where shared)

seeds 17, 29, 43, 71, 101; harness `approximate-taint-containment`; max_new_tokens 8; intermediate_size
128; rope_theta 10000.0; rms_norm_eps 1e-6; QK-norm disabled; base checkpoint "none (seeded random)";
tokenizer ByteTokenizer 256; corpus_generator_seed 20240607; tiny model digest as in `config.toml`.
Chosen here (not copied): prompt_tokens 64, segment_tokens 16, hops_per_chain 3 (R005's main prompt is
256 tokens; 64 keeps one seed under about 10 s on CPU, and every R012 claim is about taint, not length).
Corpus selection digest and the two `_sha256` keys are written at the freeze. Per-seed role: the seed
orders the cases' prompts; the corpus is fixed by `corpus_generator_seed`.

## 3. Owner decisions of 2026-10-07 (binding, verbatim)

1. Clearing: ONLY by a cold exact recomputation (kv_report == None) with the same recipe and clean
   inputs; always clears the Derivation subject; clears Content and Action subjects only on
   byte-identical recomputed digests. Echo never clears.
2. Raw rule: tainted content resubmitted as Raw keeps its taint (fail closed, false positives
   accepted). Raw is admitted only when it equals the request's own ingress record; otherwise
   MisclassifiedRawSegment. Exact whole-segment digest only; paraphrase/substring evasion NOT detected
   and stated in the preregistration.
3. Bound: 65,536 live taint subjects; when full, new Approximate ingest refused before anything is
   appended; never evict.
4. Design accepted as proposed: new ledger event KvTaintCommitted (tag 29) written taint-first; Pod
   outputs carry kv_taint plus runtime lookup by request id/dependencies;
   StateDelta/VerifiedResult/ActionProposal refused, Observation/Candidate/ToolResult/
   EnvironmentObservation stored as tainted candidates, Hypothesis tainted; merge/SemDB promotion of
   tainted content refused even under Triage 1.0; KV catalog stays Exact-only via a guard (no KvSalt
   change, protects R005); stub Approximate producer harness-only.

### Stated limits (what this experiment does NOT show)

- Paraphrase, substring, re-tokenisation or whitespace edits of tainted text are NOT detected: the
  content index matches only the exact digest of a whole segment. A model that rewrites tainted text
  produces untainted-looking Raw input; this is a known evasion and is not tested as a pass.
- False positives (an honest user typing identical text) are tainted: fail-closed, accepted.
- The runtime trusts the backend's `KvReuseReport`; a backend that mislabels an Approximate run as
  Exact is outside the claim. The stub is the only Approximate producer; real ones (layered
  batch-variant, P3, lossy tiers) are out of scope.
- A build predating tag 29 refuses to open such a log (fail-closed).

## 4. Cases (per seed, real `FileLedger` in a temp dir) and metrics

| Case | What runs | Metrics produced | Falsifier (any seed) |
|---|---|---|---|
| C1 action admission | stub Approximate run emits an ActionReady; ingest; `authorize_action` and execution check | approximate_action_admission_attempts (>=1), approximate_run_actions_admitted (must be 0) | admitted > 0; attempts < 1 |
| C2 restart and compaction | after the record: reopen with `open_durable`, and `export_compacted_snapshot` then `restore_compacted`; re-attempt C1 and a candidate reload | tainted_candidate_reload_attempts (>=1), taint_lost (must be 0) | taint_lost > 0; reload attempts < 1 |
| C3 candidate path, 3 hops | Approximate result persisted as a candidate, re-fed as Derived over 3 hops into otherwise-Exact runs; each hop is an admission attempt | approximate_candidate_path_attempts (>=1), approximate_candidate_exact_admissions (must be 0), candidate_taint_hops (>=2, target 3) | exact admission > 0; hops < 2 |
| C4 SemDB promotion | `admit_pod_output` with StateDelta and VerifiedResult, `promote_verified_pod_output`, `merge_branch` (Triage 1.0 and Reviewed), host write of tainted content | approximate_semdb_promotion_attempts (>=1), approximate_semdb_promotions (must be 0) | promotions > 0; attempts < 1 |
| C5 every Pod kind | all 8 `PodOutputKind`s from a tainted request: 4 refused with the typed error, 4 stored as tainted candidates (Hypothesis tainted), then promotion refused after restart | approximate_pod_result_promotion_attempts (>=1), approximate_pod_result_promotions_accepted (must be 0) | accepted > 0; attempts < 1; any kind unclassified |
| C6 exact echo | an Exact (non-cold or cold) pass whose input includes the tainted Derived segment | exact_echo_cases (>=1), taint_cleared_by_echo (must be 0) | cleared > 0; cases < 1 |
| C7 fresh cold recompute (positive control) | cold Exact run, same recipe digest, clean inputs: clears the Derivation subject, and Content/Action only on identical digests; the action is then authorized; a different-input or different-operation recompute does not clear | fresh_exact_recomputation_controls (>=1) | controls < 1 (vacuous) |
| C8 Raw laundering | tainted output resubmitted as Raw in a new request, over >=2 hops | approximate_output_reintroduced_as_raw_attempts (>=1), laundered_exact_admissions (must be 0) | laundered > 0; attempts < 1 |
| C9 misclassified Raw | a semantic/ingress value of another request presented as Raw; a forged Derived id | misclassified_revocable_as_raw_attempts (>=1), misclassified_raw_admissions (must be 0) | admissions > 0; attempts < 1 |
| pins | corpus selection digest, procedure parameters, tiny-model config digest, stub name and version | selection_digest_mismatches, procedure_mismatches, model_pin_mismatches (all 0) | any > 0 |

Every config metric is mapped above; the 21 metrics of `experiment.toml` are exactly the union.

### Metrics that cannot be produced honestly, or are weaker than their name

- `model_pin_mismatches`: base checkpoint, tokenizer revision and MoE topology are constants
  ("none", no external revision), so only the tiny-model config digest and stub identity are
  checked. It cannot detect anything beyond what `pin::verify` (as in R005) already compares.
- `taint_lost`: counts loss against the harness's own record of what it persisted. A backend that
  lies in its report, or a log tampered with outside the runtime, is not covered.
- `approximate_semdb_promotions` and `laundered_exact_admissions` measure the checked paths in C4/C8
  only. They say nothing about a paraphrased re-entry (see stated limits).
- `misclassified_revocable_as_raw_attempts`: counts the ingress-equality rule (owner decision 2). The
  optional snapshot-value scan of the plan is not preregistered unless the implementation lands.

## 5. Decision rule (to be mirrored by `criteria.toml` and `aggregate.py` at the freeze)

COMPLETED iff every seed meets every coverage rule and no falsifier trips; FALSIFIED if a coverage
rule or falsifier fails on any seed (every seed must pass); INVALID if a record is missing, does not
describe the frozen run, or a counter is missing or mistyped. No rule changes after a run.

## 6. Implementation prerequisites (not yet in the code at the base commit)

Ledger event tag 29, persistent registry, derivation and segment admission, Pod and SemDB checks,
the stub and the harness (plan WP1 to WP5). Until they exist, no case can run; the freeze waits for
the smoke test.
