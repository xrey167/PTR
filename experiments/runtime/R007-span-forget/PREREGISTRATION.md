# R007 span-forget: preregistration

Pre-freeze text of the preregistration of R007; at the freeze it becomes the `preregistration_text` file of
`config.toml`'s `[preregistration]` table and its SHA-256 is frozen with the table, as R012's was. R007 is `planned`:
nothing here is a result and no run is evidence. The decisions are made by `criteria.toml` and applied by
`aggregate.py`; this file gives their reasons, the definitions, the modelling choices, what the harness cannot produce and
the limits of the claim. Every criterion is the verbatim text of `experiment.toml` (hypothesis, falsification, metrics)
and the pins of `config.toml`; nothing here changes, weakens or adds a success criterion. It supersedes
`PREREGISTRATION.md`, whose proposals are confirmed or corrected below (section 3). The harness is
`bins/ptr-bench/src/experiments/r007/` (subcommand `span-forget <seed>`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command, set at the freeze):
`cargo run --release --locked -p ptr-bench -- span-forget <seed>`

Scope: after a span is forgotten (revoked) and everything after it is recomputed, the state is bit-identical to a fresh
prefill of the prompt without the span, and the span leaves no probeable trace in the next-token logits. The mechanism
under test is lineage revocation plus parent-chained chunk keys, not a mid-sequence eviction primitive (the catalog has
none). Because a chunk key hashes its parent chain, serving a stale downstream row is nearly structural; the claim is a
regression guard for forget semantics, and the negative control (C6) is what shows the guard can fail. The code under test
is the CPU reference KV backend (`ReferenceKvReuseBackend`, with the WP1 probes `probe`, `ReferenceKvProbeMode` and
`ReferenceKvProbe`), driven only through its public API.

## 1. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is
written `>= N`, a "> 0" violation `<= 0`, a "!= x" violation `== x`). COMPLETED iff every seed meets every coverage rule
and no falsifier trips; FALSIFIED if a coverage rule (the run is vacuous) or a falsifier fails on any seed; INVALID if a
record is missing, does not describe the frozen run, or a counter is missing or mistyped, or the exit status disagrees
with the rules (INVALID says nothing about the hypothesis). The harness exits 0 when every rule holds, 1 when one fails,
2 when it cannot run a case: a refusal that is not the expected one, an insert refusal on the honest path, a baseline that
does not reproduce itself (warm on an empty catalog differing from cold), a baseline probe that does not reuse the aligned
prefix, a span that changes nothing, or a negative control that does not trip every falsifier. A reuse deviation on the
forget side is not exit 2 but a falsifier (section 3, item 7). No rule changes after a run.

Coverage (the run is vacuous if unmet): span_insertions, span_evictions, downstream_recomputations, canary_probes,
canary_span_tokens, downstream_rows_before_forget, prefix_reuse_tokens, post_forget_probes all >= 1; canary_probes >= 32
(the 32 distinct pinned canaries); selection_digest_mismatches, procedure_mismatches, model_pin_mismatches all 0; the
recorded `canary_identity_set_digest` equals the pinned
`65f9b8cd60e57cb38a8588c23d5bb494c7817fc10511353980f8be1c690efe19`. Falsifiers: bitwise_mismatch_count > 0,
canary_logit_shift != 0, stale_downstream_rows_reused > 0, recompute_tokens != tokens_after_span.

## 2. Definitions

- **Prompt**: five segments from the seeded corpus generator (`corpus_generator_seed` 20240607, byte tokenizer,
  vocabulary 256), in order: system (64 tokens, Raw), document head (1024, Raw), span (256, `Derived`), document tail
  (2816, Raw), question (64, Raw). 4224 tokens with the span, 3968 without. Weights are the seeded random tiny model
  (4 layers, hidden 64, 4 heads, 2 kv heads, head_dim 16, weight_seed 20240607; the digest of that configuration is pinned
  in `config.toml`). Every segment boundary is a multiple of the 16-token chunk.
- **Span**: a revocable `Derived` prompt segment (identity `span:canary-1`, generation 1) of 256 tokens at prompt positions
  1088 to 1343, so at offset 1024 of the 4096-token document (middle). Its tokens are the bytes of canaries 0 to 15
  (16 x 16 = 256), so `canary_span_tokens` = 256; canaries 16 to 31 are never in the span and act as controls.
- **Forget**: revoke the span's lineage identity with `invalidate_inputs` (every catalog entry that cites it is retired),
  then run the prompt without the span against the same backend and catalog. Not a mid-sequence page eviction.
- **Fresh prefill without the span (the baseline)**: a `Cold` probe of the 3968-token prompt on an empty backend B.
- **Bitwise comparison**: on `f32` bit patterns (`to_bits`), no tolerance; a NaN payload or a signed-zero difference is a
  mismatch. Two logits vectors are equal iff all 256 bit patterns are equal.
- **State digest**: the KV-state digest of `ptr-runtime` (`KvSession::kv_state_digest`): SHA-256 over every layer's K and V
  rows for all prompt positions. `bitwise_mismatch_count` counts, per seed, one per differing KV-state digest, one per
  differing logits entry (f32 bit patterns), one per differing decode digest or generated tokens, and one per non-finite
  logits entry on either side (identical NaN vectors are not a pass), for: the forget run against the cold baseline; each of
  the 32 probes (forget backend against baseline backend, warm against warm); and the two cold anchors (below) against both
  warm probes of the same request.
- **Cold anchors**: warm-vs-warm alone could hide a defect shared by both warm paths (fork or reuse of a long aligned
  prefix), so per seed the probe requests of canary 0 (in the span) and canary 16 (never in it) are also run `Cold` on the
  baseline backend (no lookup, no insert) and compared with the forget-backend and baseline-backend warm probes of the same
  request. Their bit differences count into `bitwise_mismatch_count` and their ulp distance into `canary_logit_shift`.
- **Canary probe** (owner decision 3): per probe, the full next-token logits vector (256 entries) after the probe suffix.
  The probe request is the prompt without the span plus the 16 bytes of canary i as a final Raw segment, run `Warm` on B and
  on F; each reuses the aligned prefix (3968 tokens) and recomputes only the tail (a baseline probe that does not reuse the aligned
  prefix is exit 2, a forget-side deviation is a falsifier, section 3 item 7). **`canary_logit_shift`** is printed as a JSON integer: the largest distance, in representable `f32`
  steps (ulps), between F's and B's logit over all 256 entries and all 32 probes. It is 0 iff the maximum absolute
  difference is exactly 0 (a signed-zero pair is 0 ulps and is counted by `bitwise_mismatch_count`; a NaN pair with
  different bits is 2^32). The `f64` maximum absolute difference is printed under `cases.C5.max_abs_f32`, which no rule
  reads. The integer form is needed because the rule grammar compares integers only; it keeps "exactly 0, no noise
  threshold".
- **Canary set**: canary_i = first 16 hex characters of sha256(`20240607:` + i), i in 0..31; the identity-set digest is the
  sha256 over the 32 strings joined by newlines. The harness computes it from the canaries read back from the requests it
  actually probed (and checks that the 16 embedded in the span are the first 16 of them); a probe counts only when both
  runtime calls returned.
- **Counts, measured from the runtime's outcome** (never asserted from a constant): `prefix_reuse_tokens` = `hit_tokens` of
  the forget run (1088 expected); `recompute_tokens` = its `prefilled_tokens` (2880 expected); `tokens_after_span` = the
  tokens after the span segment in the with-span request built (tail + question = 2880);
  `downstream_recomputations` = chunk rows the forget run inserted (`inserted_chunks`, 180 expected);
  `downstream_rows_before_forget` = rows the warm run inserted beyond the chunks ending at the span's end
  (`inserted_chunks` 264 minus 84 = 180); `span_insertions` = 1 iff the warm run inserted a row for every span-citing chunk
  (264 - 68 = 196 >= 16); `span_evictions` = 1 iff the revocation retired exactly the rows that cite the span (196);
  `stale_downstream_rows_reused` = 16-token rows the forget run was served beyond the prefix before the span
  (`(hit_tokens - 1088) / 16`; every row in that catalog is the with-span run's); `post_forget_probes` = probes made after
  the revocation retired its rows and the forget run recomputed.
- **Raised limits**: the harness raises `ReferenceKvLimits` (max_prompt_tokens = longest probe request, 4224 + 16; catalog
  `max_trie_nodes` 65536, `max_kv_page_bytes` 2^32, `max_metadata_bytes` 2^30); the crate default stays 256 (owner decision
  5). An insert refusal on the honest path is exit 2. The configuration digest is recomputed from the raised-limit backend's
  own stack configuration and must equal the pinned one, as must the KV chain id of the default backend; a difference is a
  harness failure (exit 2), not a result.

## 3. Head decisions of 2026-10-08 on the draft's open definitions (confirmed; no criterion changed)

1. Canary layout: the span carries canaries 0 to 15; 16 to 31 are controls; all 32 probes are compared.
2. Probe suffix: the 16 bytes of canary i appended after the question.
3. `bitwise_mismatch_count`: one per differing KV-state digest plus one per differing logits entry, over the forget run and
   all 32 probes.
4. `recompute_tokens == tokens_after_span` stays exact.
5. `downstream_recomputations` and `downstream_rows_before_forget` are counted in 16-token chunks after the span (180).
6. Refinement of 3 (review): two cold anchors per seed (canary 0 and canary 16) plus the 32 warm-vs-warm comparisons, as
   defined in section 2; decode digests and non-finite entries count into `bitwise_mismatch_count`.
7. Forget-side reuse deviations falsify instead of exiting 2: a hit on the empty catalog or a probe hit above the aligned
   prefix counts into `stale_downstream_rows_reused` (rows); prefilled tokens beyond what the expected hit leaves, and any
   disagreement of the outcome's books (`prompt_tokens` against the request, `prompt_tokens - hit_tokens` against
   `prefilled_tokens`), add to `recompute_tokens` (never cancelling it). Only baseline validity and harness errors are exit 2.

Where this harness differs from the draft's table, before any run: the probes run `Warm` on a baseline backend B whose
catalog was populated by a warm run of the same prompt (C1; B's warm run must equal its cold run bitwise, else exit 2),
instead of `Cold` with the suffix appended; this costs 16 tokens per probe instead of 3984 and compares like with like
(each side reuses its own entries of the prompt without the span). The negative control runs on a third backend.

## 4. Pins the config left open, with reasons

| value | pinned | reason / how verified |
|---|---|---|
| seeds / harness | 17, 29, 43, 71, 101 / `span-forget` | `experiment.toml`; `params.rs` |
| intermediate_size / rope_theta / rms_norm_eps / QK-norm / max_new_tokens | 128 / 10000.0 / 1e-6 / disabled / 8 | inherited from R005; reproduced by `layered_kv::pins` |
| corpus | sequence `i` of `corpus_generator_seed` 20240607, `i` in `0..8`, 2816 tokens each; a role takes a prefix of its sequence | `layered_kv::corpus::sequence` |
| corpus selection digest | `sha256:48de01d2f91315b6d124dce668e01f783e599a00677f92b7325cf1467d3308c1` (domain `ptr.r007.corpus-selection.v1`, 8 sequences) | `params.rs` pins it; a test reproduces it; written into the table at the freeze |
| per-seed role | the seed picks which corpus sequence fills the system, head, tail and question roles (`order(seed, 7, 8)`, first four) and the probe order of the 32 canaries (`order(seed, 107, 32)`); the corpus, weights and canaries are fixed by their pinned seeds | recorded as `corpus_roles` and `cases.C5.order` |
| model pins | tiny_model_config digest (from the backend's own stack configuration), tokenizer, shape, chain id, cold run reproduced by an independent stack | `layered_kv::pins::base_checks` |

## 5. Cases (per seed) and the metric map

Exactly these prefill-scale runs per seed (a 4224-token cold prefill takes about a minute in release):

| Case | What runs | Metrics produced |
|---|---|---|
| C1 baseline | backend B: `Cold` probe of the 3968-token prompt without the span; then `Warm` of the same prompt to populate B's catalog; the two must be bitwise equal | reference values; exit 2 if unequal |
| C2 warm with span | backend F: `Warm` probe of the 4224-token prompt on an empty catalog; a hit on the empty catalog counts as stale reuse; the span must change the state against C1 (else exit 2) | span_insertions, downstream_rows_before_forget, canary_span_tokens (via the canary identity) |
| C3 forget and rerun | revoke the span's lineage identity on F; `Warm` probe of the prompt without the span | span_evictions, prefix_reuse_tokens, recompute_tokens, tokens_after_span, downstream_recomputations, stale_downstream_rows_reused |
| C4 compare | the forget run against C1's cold run: KV-state digest and the 256 logits bit patterns | bitwise_mismatch_count (part) |
| C5 canary probes | 32 probes (seeded order), each the prompt without the span plus canary i, `Warm` on B and on F after the forget; all 256 entries, KV-state and decode digests, finiteness compared; plus 2 cold anchors (canary 0 and 16, `Cold` on B) compared with both warm probes | canary_probes, post_forget_probes, canary_logit_shift, bitwise_mismatch_count (rest), canary_identity_set_digest |
| C6 negative control | backend G under `set_faults(IndexUpdate::Suppressed)`: the same warm run with the span, a revocation that does not reach the index (retires 0), then a `StaleDonor` probe of the prompt without the span; its digest and logits must differ from C1's and all four falsifiers, judged on its counters, must trip | no manifest metric; recorded under `cases.C6`; exit 2 if it does not trip |
| C7 pins | corpus selection digest, the procedure parameters (the 23 names of `procedure_parameters_recorded_per_cell`, each observed from the requests or the backend and compared with the pinned value), model pins, canary digest | selection_digest_mismatches, procedure_mismatches, model_pin_mismatches, canary_identity_set_digest |

All 17 metrics of `experiment.toml` are produced, as JSON integers except `canary_identity_set_digest` (a digest string).

## 6. Weak metrics and what is not measured

- `model_pin_mismatches` and `procedure_mismatches` are weak: the base checkpoint, topology and the canary generator seed
  are constants here; the shape and the tokenizer are read from the backend, the segment sizes from the requests built.
- `canary_probes` counts probes whose two runtime calls returned with finite logits (per probe); `post_forget_probes`
  counts those made after the revocation retired exactly the span's rows and the forget run recomputed. Both are weak:
  they count completed work, not that a probe could have seen the span, and in the absence of an `Err` they are 32.
- `procedure_mismatches` records `canary_generator_seed` and `moe_topology` as constants (listed under
  `procedure.constants_never_counted`, never counted); `seeds` compares the manifest file with the list the harness was
  written for. A finite-comparison count (`internal.finite_comparisons`) is recorded, not a coverage rule (no rule is added).
- Measured values are printed next to shape-derived expectations (`expected.*`: 1088, 68, 2880, 180, 180); no rule reads
  them, the tests assert them.
- `span_insertions` and `span_evictions` are 0/1 per the single span; they read the rows the catalog accepted and retired.
- `stale_downstream_rows_reused` is structurally near-zero because chunk keys hash the parent; only the negative control
  shows it can be non-zero.
- Timings (`timing_ms`) are never read by a rule.

## 7. Stated limits (what this experiment does NOT show)

- Tiny seeded random model (4 layers, hidden 64): nothing is claimed about a trained model, a GPU kernel or any
  non-deterministic reduction order. Bitwise equality is a property of this CPU reference path on one machine profile
  (`hardware/default.toml`).
- CPU reference backend only; the paged or GPU backends are not tested.
- The forget is revocation plus parent-chained chunk keys, not a mid-sequence eviction primitive: it shows that revoking
  a lineage segment and re-running leaves no stale row reachable, not that a page-level eviction works.
- The probe is a logits comparison after a 16-byte suffix at one position per probe; it does not detect a trace that never
  reaches those logits, nor a side channel (timing, memory) outside the logits and the KV-state digest.
- Canaries 16 to 31 never appear in the span; no claim about their placement.
- The KV-state digest covers K and V rows for all prompt positions; hidden activations that are not cached are not digested.
- Regression-guard character: a pass is the expected outcome and protects the lineage and re-prefill path against
  regressions.
