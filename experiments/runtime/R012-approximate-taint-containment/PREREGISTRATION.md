# R012 approximate-taint-containment: preregistration

Frozen text of the preregistration of R012. It is a `file` key of `config.toml`'s `[preregistration]` table, so its
SHA-256 is frozen with the table. Nothing in it is a result. The decisions are made by `criteria.toml` (frozen the same
way) and applied by `aggregate.py`; this file gives their reasons, the definitions, the modelling choices, what the
harness cannot produce and the limits of the claim. The harness is
`bins/ptr-bench/src/experiments/r012/` (subcommand `approximate-taint-containment <seed>`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command):
`cargo run --release --locked -p ptr-bench -- approximate-taint-containment <seed>`

Scope: containment of a labelled Approximate-KV result only (a taint-containment claim, not a forward-compute or a
KV-memory claim). R012 is the Approximate half split out of R005 on 2026-10-06 by owner decision; the Exact half stays in
R005. The producer is a harness-only stub (`r012/stub.rs`, name `r012-stub-approximate-producer`, version `1`): it runs a
genuine cold prefill of the R005 tiny model on the reference backend (`ReferenceKvReuseBackend::infer_cold`: no lookup,
no insert), adds a deterministic `ActionReady` whose payload is the generated text, and labels the report
`KvReuseMode::Approximate` with `KvReusePolicy::AllowApproximate` and a hit of one chunk. The label is the stub's, not a
measurement: this experiment tests whether the runtime contains a result so labelled, not any real Approximate producer
(layered batch-variant, P3 and lossy tiers are out of scope). The code under test is `PtrRuntime` over a durable
`FileLedger` (one in a temporary directory per case), driven only through its public API: persistence, a restart with
`open_durable` and a compacted snapshot export and restore are part of the cases.

## 1. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (the rules are the
manifest's falsification text in the grammar of `criteria.toml`: a "< N" prerequisite is written `>= N`, a "> 0"
violation `<= 0`). COMPLETED iff every seed meets every coverage rule and no falsifier trips; FALSIFIED if a coverage
rule (the run is vacuous) or a falsifier fails on any seed; INVALID if a record is missing, does not describe the frozen
run, or a counter is missing or mistyped, or the exit status disagrees with the rules (INVALID says nothing about the
hypothesis). `aggregate.py` is R010's with this experiment's names. The harness exits 0 when every rule holds, 1 when one
fails, 2 when it cannot run a case: a refusal that is not the expected typed error, a fixture that does not build, a
clean control that fails or a chain that stops for a reason other than a measured violation. No rule changes after a run.

## 2. Definitions

- **Hop**: one derivation step, that is one run whose input includes a segment derived from a previous run's output (a
  `Derived` segment minted by the runtime from that output). The Approximate run itself is hop 0 and is not counted.
  `candidate_taint_hops` is the longest chain of hops in which every hop's output stayed Approximate and its action stayed
  refused at `authorize_action`.
- **Attempt after persistence, before recompute**: a coverage counter (`*_attempts`) counts an attempt only if (a) a
  `KvTaintCommitted` record has been appended to the ledger (the append returned; after a compaction, before it) and (b) no
  cold Exact recomputation of the same recipe has completed in the case. An attempt made outside that window is executed
  and counted under `internal.attempts_outside_window`, never as an attempt.
- **Violation counters** (`*_admitted`, `*_promotions`, `*_accepted`, `taint_lost`, `misclassified_raw_admissions`,
  `laundered_exact_admissions`, `taint_cleared_by_echo`) count what the runtime did wrong. The sub-case that produced each
  is listed under `subcases` in the result.
- An Exact run (cold or warm) under a request id that was already tainted, whose action is then authorized, counts under
  `approximate_run_actions_admitted`: the Request subject is never cleared, so a reused tainted request id keeps tainting
  what runs under it (C1, sub-case "C1 request id reuse").
- The five Pod kinds stored as tainted candidates or a hypothesis (Observation, Candidate, ToolResult,
  EnvironmentObservation, Hypothesis) count as `approximate_pod_result_promotion_attempts` (they are offered to the Pod
  boundary); they are never an accepted promotion, and a later `promote_verified_pod_output` or merge of what they stored
  is a further attempt, counted the same way.
- A reload attempt (`tainted_candidate_reload_attempts`) and a hop attempt (`approximate_candidate_path_attempts`) count
  only when the runtime admitted the segments. A refusal of a Derived segment the persistent registry holds (a tainted one),
  at a reload or at any hop, is exit 2, not a counted attempt: the runtime must recognise tainted segments across a restart
  and a compaction.

## 3. Pins the config left open, with reasons

| value | pinned | reason / how verified |
|---|---|---|
| seeds / harness | 17, 29, 43, 71, 101 / `approximate-taint-containment` | `config.toml` `[preregistration]` |
| max_new_tokens / intermediate_size / rope_theta / rms_norm_eps / QK-norm | 8 / 128 / 10000.0 / 1e-6 / disabled | inherited from R005; reproduced by `layered_kv::pins` |
| base checkpoint / tokenizer / topology | none (seeded random) / ByteTokenizer, vocabulary 256 / dense | constants; see section 6 |
| prompt_tokens | 64 | every request the harness builds from the corpus has 64 tokens (a Raw segment of the whole ingress text, or an own-text Raw segment completed by one Derived segment) and every stub run generates 8 tokens; requests that by design carry model output as their whole Raw text (C8, and the "a model output" case of C9) or a forged extra segment have the size of that content and are not size-checked; R005's main prompt is 256 tokens, 64 keeps one seed under about 10 s on CPU and every R012 claim is about taint, not length; a request of another size is a `procedure_mismatches` |
| segment_tokens | 16 | the backend's prefill chunk (`KV_CHUNK_TOKENS`); the stub's hit is one chunk |
| hops_per_chain | 3 | the candidate chain of C3; `candidate_taint_hops` must reach it at the pinned scale unless a measured violation broke the chain (else exit 2); the criterion is the manifest's 2 |
| corpus | sequence `i` of `corpus_generator_seed` 20240607, `i` in `0..48`, 64 tokens each; a prompt is the text whose character `k` is `'!' + (token_k mod 94)` (valid UTF-8, one byte token per character) | the text form is needed because a Raw segment must equal the request's own text ingress record |
| corpus selection digest | `sha256:9f48b499e6ec8735a762d8c746fbf6972d02c86c0b17f351a6680856350e3eef` (domain `ptr.r012.corpus-selection.v1`, 48 sequences) | `params.rs` pins it; a test reproduces it; written into the table at the freeze |
| per-seed role | the seed orders the nine cases and assigns corpus prompts to them (five slots per case); the corpus is fixed by `corpus_generator_seed` | `corpus::order`; the assignment is recorded as `prompt_indexes` |
| producer identity | `r012-stub-approximate-producer/1`, generation parameters `max_new_tokens=8;prompt_tokens=64;greedy;byte-tokenizer` | bound into every recipe digest, also for the cold recomputation (the same operation by the same producer pipeline) |
| model pins | tiny_model_config digest, tokenizer, shape, chain id, cold run reproduced by an independent stack, and the stub's: same content as the independent cold run, reports Approximate, pinned identity | `layered_kv::pins` and `cases::stub_pin_checks` |

## 4. Owner decisions of 2026-10-07 (binding, verbatim)

1. Clearing: ONLY by a cold exact recomputation (kv_report == None) with the same recipe and clean inputs; always clears
   the Derivation subject; clears Content and Action subjects only on byte-identical recomputed digests. Echo never clears.
2. Raw rule: tainted content resubmitted as Raw keeps its taint (fail closed, false positives accepted). Raw is admitted
   only when it equals the request's own ingress record; otherwise MisclassifiedRawSegment. Exact whole-segment digest only;
   paraphrase/substring evasion NOT detected and stated in the preregistration.
3. Bound: 65,536 live taint subjects; when full, new Approximate ingest refused before anything is appended; never evict.
4. Design accepted as proposed: new ledger event KvTaintCommitted (tag 29) written taint-first; Pod outputs carry kv_taint
   plus runtime lookup by request id/dependencies; StateDelta/VerifiedResult/ActionProposal refused,
   Observation/Candidate/ToolResult/EnvironmentObservation stored as tainted candidates, Hypothesis tainted; merge/SemDB
   promotion of tainted content refused even under Triage 1.0; KV catalog stays Exact-only via a guard (no KvSalt change,
   protects R005); stub Approximate producer harness-only.

## 5. Cases (per seed, a real `FileLedger` per case) and metrics

| Case | What runs | Metrics produced | Falsifier (any seed) |
|---|---|---|---|
| C1 action admission | the stub's Approximate run is ingested as a derivation hop, through `ingest_model_run_for` and through `run_model_run_resumable`; each action is offered to `authorize_action` and to `prepare_execution_once`; a request id that was already tainted is reused for a fresh request and an Exact cold run, whose action is offered too | approximate_action_admission_attempts (>=1), approximate_run_actions_admitted (must be 0; an `EffectAttempted` record or an executor call for a tainted action counts as admitted) | admitted > 0; attempts < 1 |
| C2 restart and compaction | after the record: reopen with `open_durable` (twice at the pinned scale), then `export_compacted_snapshot` and `restore_compacted`; each time re-attempt C1's offers and reload the persisted candidate (its derived segment and its text as the request's own Raw text) | tainted_candidate_reload_attempts (>=1), taint_lost (must be 0) | taint_lost > 0; reload attempts < 1 |
| C3 candidate path, 3 hops | the Approximate result persisted as a candidate (its derived segment) is re-fed into otherwise-Exact runs over 3 hops (hops 1 and 3 cold, hop 2 with an Exact warm report), each hop a genuine inference on the 64-token request; the runtime is reopened after hop 1 and compacted after hop 2; each hop is an admission attempt | approximate_candidate_path_attempts (>=1), approximate_candidate_exact_admissions (must be 0), candidate_taint_hops (>=2, target 3) | exact admission > 0; hops < 2 |
| C4 SemDB promotion | `admit_pod_output` with a StateDelta and a VerifiedResult, `promote_verified_pod_output` (by the tainted request and by tainted content), `merge_pod_hypothesis` and `merge_branch` under every `MergeAuthority` (Triage 1.0, Reviewed, and at the pinned scale Triage 0.5; `preview_merge` too), the host write as text and as payload, the host write of a plain Approximate run's text; clean controls on every path (promote by request and by content, the host write as text and as payload, `merge_branch` under Triage 1.0, `preview_merge`, `merge_pod_hypothesis`, clean StateDelta, VerifiedResult and Hypothesis admissions), the ones that need no Pod scope repeated after the restart | approximate_semdb_promotion_attempts (>=1), approximate_semdb_promotions (must be 0) | promotions > 0; attempts < 1; a clean control that fails is exit 2 |
| C5 every Pod kind | all 8 `PodOutputKind`s from a tainted request (a Pod each): the 3 refused kinds with the typed error before the verifier runs, the 5 stored kinds as tainted candidates or hypothesis; the stored payloads promoted by `promote_verified_pod_output` and the hypothesis merged under every authority, before and after a restart; a Pod from a clean request whose manifest binds a tainted stored candidate and whose output depends on it | approximate_pod_result_promotion_attempts (>=1), approximate_pod_result_promotions_accepted (must be 0) | accepted > 0; attempts < 1; any kind with another outcome is exit 2 |
| C6 exact echo | an Exact pass whose input includes the tainted Derived segment: a genuine cold run, the adversarial echo that reproduces the tainted run's own text and action under a cold report, and the same under an Exact warm report; again after a restart | exact_echo_cases (>=1), taint_cleared_by_echo (must be 0) | cleared > 0; cases < 1 |
| C7 fresh cold recompute (positive control) | a cold Exact run, new request id, same Raw ingress text, same recipe: clears the Derivation, Content and Action subjects (identical bytes); the action is then authorized, prepared and executed once, and still so after a restart; negative controls on a fresh base each: another input, another producer, other generation parameters, a warm Exact run, another Approximate run must not clear | fresh_exact_recomputation_controls (>=1) | controls < 1 (vacuous); a positive control that fails is exit 2; a negative control that clears counts as taint_lost |
| C8 Raw laundering | tainted output resubmitted as the whole Raw text of a new request, over 2 hops through the derived path; through `run_model_run_resumable` (the server path); the output of a plain Approximate run (`ingest_model_run_for`) both ways; a whitespace-edited text is probed and recorded as a limit, not a pass | approximate_output_reintroduced_as_raw_attempts (>=1), laundered_exact_admissions (must be 0) | laundered > 0; attempts < 1 |
| C9 misclassified Raw | another request's ingress text, the tainted request's ingress text, a committed semantic value, a model output and the text of a request with no ingress record, each presented as Raw (through `admit_segments` and through a hop; nothing may be appended); forged Derived ids refused with `UnknownDerivedSegment`, and an id naming the tainted derivation with other content or generation admitted as Approximate at most | misclassified_revocable_as_raw_attempts (>=1), misclassified_raw_admissions (must be 0) | admissions > 0; attempts < 1 |
| pins | corpus selection digest, procedure parameters (the backend's shape, the prompt size, the generated tokens), tiny-model config digest, stub identity, label and content | selection_digest_mismatches, procedure_mismatches, model_pin_mismatches (all 0) | any > 0 |

The 21 metrics of `experiment.toml` are exactly the union. Differences from the draft, made before any run: the
sub-cases of C1 (plain and server paths, request id reuse), C4 (plain-run text), C5 (binding to a tainted stored candidate)
and C8 (plain-run output, server path) were added because they are paths a laundering or promotion could take; each
counts into the same metrics. A violation in them is a violation. The result lists the sub-cases with their violations.

## 6. How each metric is measured, and what is weaker than its name

- Every stub run is a genuine inference on the request's real tokens (a Derived segment contributes its UTF-8 bytes as
  tokens); only the report and the action are the stub's.
- `taint_lost`: a persisted taint subject (the action, the candidate segment, its text) that reads Exact after a reopen or a
  compacted restore, plus a negative recompute control that cleared. It compares with what this harness itself persisted:
  a backend that lies in its report, or a log tampered with outside the runtime, is not covered. Weak.
- `model_pin_mismatches`: base checkpoint, tokenizer revision and topology are constants ("none", no external revision,
  dense), so only the tiny-model config digest, the tokenizer, the shape, the chain id, the backend's and the stub's cold run
  against an independent stack, and the stub's identity and label are checked. It cannot detect more than `pins::base_checks`
  already compares. Weak.
- `approximate_semdb_promotions` and `laundered_exact_admissions` measure the checked paths of C4 and C8 only; they say
  nothing about a paraphrased re-entry (section 7).
- `misclassified_revocable_as_raw_attempts` counts the ingress-equality rule (owner decision 2). No snapshot-value scan is
  preregistered: a committed semantic value presented as Raw is refused because it is not the request's own ingress text.
- `taint_cleared_by_echo`: an echo is cleared if a Clear record is committed, the tainted action is authorized, the tainted
  derived segment is admitted Exact, or the echo's own output is Exact.
- `candidate_taint_hops`: hop 2 carries an Exact report over the same genuine content (a stand-in: R005 shows real Exact
  reuse is bitwise equal to cold), so the runtime's handling of an Exact warm report is exercised without a real warm path.

## 7. Stated limits (what this experiment does NOT show)

- Exact whole-segment digest only. Paraphrase, substring, re-tokenisation and whitespace edits of tainted text are NOT
  detected: the content index matches the exact digest of a whole segment. A model that rewrites tainted text produces
  untainted-looking Raw input; this is a known evasion and is not tested as a pass (the result records the whitespace probe as
  `limit_whitespace_edit_detected`, information only). False positives (an honest user typing identical text) are tainted:
  fail closed, accepted.
- The runtime trusts the backend's `KvReuseReport`; a backend that mislabels an Approximate run as Exact is outside the
  claim. The stub is the only Approximate producer.
- The Exact-only KV catalog guard is structural (Approximate sessions never look up or insert: `kv_infer_session.rs`,
  `kv_infer_layered.rs`), not a check inside the catalog; the harness's stub runs the cold path and never touches the catalog,
  so no catalog behaviour is measured.
- Cleared taint leaves tombstones in the materialized state (the ledger keeps the Record and the Clear).
- A ledger writer who knows a recipe could forge a Clear: the model is a trusted ledger, and the recompute digest is not
  verifiable on replay.
- The Request subject is never cleared. Pod-side taint records (PodOutput and the payload's Content subject) are permanent:
  no cold recomputation of a Pod output exists to reproduce their recipe.
- A clean minted derivation is remembered in memory only, so a clean derived segment held across a restart is refused (fail
  closed); a tainted one is recognised from the persistent registry.
- A build that predates ledger tag 29 refuses to open such a log (fail closed). The registry is bounded at 65,536 live
  subjects and refuses new Approximate ingest when full; the bound is not exercised by this experiment (no run fills it).
- Production loops do not yet call `ingest_derived_run`; the experiment shows the mechanism, not that every production path
  uses it.

## 8. Files

`PREREGISTRATION.md` (this frozen text), `criteria.toml` (the rules, with their grammar; a harness test pins them to the rules
the harness judges, in order, and checks that every counter they name resolves in the harness's JSON), `aggregate.py`
(applies them to the five seed records; verdict COMPLETED, FALSIFIED or INVALID; no static evidence), `config.toml`
`[preregistration]` (the frozen table). The harness is `bins/ptr-bench` (`approximate-taint-containment <seed>`, one JSON
object on stdout). A run at the reduced scale (the smoke test) says `"scale":"reduced"` and is rejected by `aggregate.py`.
