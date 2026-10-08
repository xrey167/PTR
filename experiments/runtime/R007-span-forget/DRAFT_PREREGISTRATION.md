# R007 span-forget: draft preregistration

DRAFT. R007 is `planned`, not frozen. Nothing here is a result and no run exists. This text sharpens definitions the
manifest left open; it does not change, weaken or add a success criterion. Every criterion is the verbatim text of
`experiment.toml` (hypothesis, falsification, metrics) and the pins of `config.toml`. Every definition below that is not
an owner decision of 2026-10-08 is a proposal, marked "proposal", to be confirmed or corrected before the freeze; none
of them touches a criterion. At the freeze this file becomes the frozen preregistration text, as R012's
`PREREGISTRATION.md` did (R012 history: draft, pins, freeze).

Intended entrypoint (to be set at the freeze): `cargo run --release --locked -p ptr-bench -- span-forget <seed>`,
harness `bins/ptr-bench/src/experiments/r007/` (not written yet).

## 1. Claim and scope

Claim: after a span is forgotten (revoked) and everything after it is recomputed, the resulting state is bit-identical to
a fresh prefill of the prompt without the span, and the span leaves no probeable trace in the next-token logits. It is a
regression guard for forget semantics (revocation plus re-prefill after the span), not a deep claim: the chunk keys hash
the parent, so serving a stale downstream row is nearly structural. The negative control (section 6, case C6) exists so
that the guard is shown to be able to fail.

## 2. Decision rule (unchanged manifest text)

Five seeds (17, 29, 43, 71, 101), each must pass every criterion ("every seed must pass"). Coverage prerequisites
(span_insertions, span_evictions, downstream_recomputations, canary_probes, canary_span_tokens,
downstream_rows_before_forget, prefix_reuse_tokens, post_forget_probes all >= 1; canary_probes = 32 distinct) make an
unmet run vacuous and falsified. Falsified if bitwise_mismatch_count > 0, canary_logit_shift != 0 (no noise threshold),
stale_downstream_rows_reused > 0, recompute_tokens != tokens_after_span, or any pin or digest mismatch
(selection_digest_mismatches, procedure_mismatches, model_pin_mismatches, canary_identity_set_digest != the pinned
65f9b8cd60e57cb38a8588c23d5bb494c7817fc10511353980f8be1c690efe19). `criteria.toml` and `aggregate.py` (not yet written)
will state these mechanically, following R012; no rule changes after a run.

## 3. Owner decisions of 2026-10-08 (binding, verbatim)

1. Order: R007 span-forget first, then R006 positional-reuse-oracle (with its own decisions later).
2. R007 FORGET mechanism: the span is a revocable PromptSegment (Derived or Memory); forget = revoke its lineage segment,
   then run the prompt without it (prefix reused, everything after the span recomputed). README states the mechanism is
   revocation plus parent-chained chunk keys; a negative control with injected stale reuse (set_faults) shows the
   criterion is not vacuous.
3. Canary probe: per probe, the full next-token logits vector (all 256 entries) after the probe suffix;
   canary_logit_shift = max absolute difference over all 256 entries and all 32 probes; frozen before the preregistration.
4. R008 and M016 stay planned and are deferred to the GPU phase (pretrained Qwen2.5-0.5B); no re-pinning to the tiny model.
5. (head, per recommendation 4) Reference limits are raised in the R007 harness only via ReferenceKvLimits; the default
   stays 256.

## 4. Definitions

- **Prompt**: system part 64 tokens, document 4096 tokens, question 64 tokens (4224 tokens with the span), from the seeded
  corpus generator (`corpus_generator_seed` 20240607, byte tokenizer, vocabulary 256). Weights are the seeded random tiny
  model (layers 4, hidden 64, heads 4, kv_heads 2, head_dim 16, weight_seed 20240607; config digest pinned in
  `config.toml`).
- **Span**: 256 tokens starting at offset 1024 of the document, so at prompt positions 1088 to 1343, modelled as one
  revocable `PromptSegment` (Derived or Memory kind, owner decision 2) with its own lineage identity. The prompt is the
  ordered segments: system, document head (1024 tokens), span, document tail (4096 - 1024 - 256 = 2816 tokens), question.
  Every boundary is a multiple of the 16-token chunk (`KV_CHUNK_TOKENS`), so segment and chunk boundaries coincide.
- **Prompt without the span**: the same segments with the span omitted: 3968 tokens, positions renumbered; this is the
  manifest's baseline "fresh prefill of the prompt without the span".
- **Forget**: revoke the span's lineage segment (every catalog entry that cites it dies), then run the prompt without the
  span against the same catalog and backend. It is NOT a mid-sequence page eviction primitive: the catalog has none, and
  the preregistered text does not require one. What is tested is lineage revocation plus parent-chained chunk keys (a
  chunk's key hashes its parent chain, so a chunk after the span cannot match a chunk of the span-free prompt).
- **Bitwise comparison**: on `f32` bit patterns (`f32::to_bits`), never on values, no tolerance; a NaN payload or a
  signed zero difference is a mismatch. Two logits vectors are equal iff all 256 bit patterns are equal.
- **State digest (KV-state digest)**: SHA-256 over every layer's K rows and V rows for all prompt positions, as
  implemented in `ptr-runtime` (WP1 accessor). The exact domain string, row order and encoding are left to the freeze and
  will be copied here from the code; they are not chosen in this draft. `bitwise_mismatch_count` counts, per seed, the
  number of compared artefacts that differ: the state digest and the logits bit patterns of the forget run versus the
  cold baseline (proposal: one count per differing digest plus one per differing logits entry of the final-position
  vector; the criterion is only whether it is 0).
- **Canary probe** (owner decision 3): per probe, the full next-token logits vector (256 entries) after the probe suffix
  has been appended to the prompt. `canary_logit_shift` = max over the 32 probes and the 256 entries of the absolute
  difference between the post-forget logit and the logit of the same probe on the cold baseline. Required value: exactly
  0, no noise threshold.
- **Canary set**: 32 canaries, canary_i = first 16 hex characters of sha256(`20240607:` + i), i in 0..31; digest
  sha256 over the canaries joined by newlines must equal the pinned
  65f9b8cd60e57cb38a8588c23d5bb494c7817fc10511353980f8be1c690efe19; a recorded digest that differs rejects the run.
- **Probe suffix** (proposal): the 16 bytes of canary_i appended after the question. Layout of the canaries in the span
  (proposal): 32 canaries of 16 bytes (512 bytes) do not fit in a 256-token span, so the span carries the bytes of
  canaries 0 to 15 (16 x 16 = 256 tokens, `canary_span_tokens` = 256) and probes 16 to 31 are probes of canaries that were
  never in the span. All 32 probes are made and compared; the claim (no trace) is checked for the 16 present and the 16
  absent canaries alike. This layout is not fixed by the manifest or by an owner decision and must be confirmed at the
  freeze; the criteria are unaffected.
- **Reuse accounting**: `prefix_reuse_tokens` = the aligned prefix reused by the forget run = 64 + 1024 = 1088 tokens =
  68 chunks of 16. `tokens_after_span` = 4096 - 1024 - 256 + 64 = 2880 tokens (document tail plus question).
  `recompute_tokens` = tokens prefilled by the forget run = 3968 - 1088 = 2880, required to equal `tokens_after_span`
  exactly. `downstream_recomputations` = chunks recomputed after the span = 180 (proposal).
  `downstream_rows_before_forget` = catalog chunk rows after the span present from the warm run before the forget (proposal:
  180 chunks). `stale_downstream_rows_reused` = catalog rows of the warm run that lie after the span and are served to the
  forget run; required 0.
- **Raised limits**: the reference backend default `max_prompt_tokens` is 256. The harness raises it through
  `ReferenceKvLimits` to fit 4224 plus the probe suffix; the crate default stays 256 (owner decision 5). The tiny-model
  config digest is over the model shape and seed, not the limits, and must still match the pinned digest; this will be
  checked, not assumed (`max_position_embeddings` follows the limit, so a mismatch is a harness failure, not a result).

## 5. Cases (per seed) and the metric map

| Case | What runs | Metrics produced |
|---|---|---|
| C1 baseline, cold, without span | fresh prefill of the 3968-token prompt without the span on an empty catalog; record logits, state digest, then the 32 probes (suffix appended, cold) | reference values for bitwise_mismatch_count and canary_logit_shift |
| C2 warm, with span | prefill the 4224-token prompt with the span, populating the catalog | span_insertions (span segment admitted: 1), canary_span_tokens, downstream_rows_before_forget |
| C3 forget and rerun | revoke the span (span_evictions: 1), then run the prompt without it on the same catalog | prefix_reuse_tokens, recompute_tokens, tokens_after_span, downstream_recomputations, stale_downstream_rows_reused |
| C4 compare | forget-run versus C1: logits bit patterns and KV-state digest | bitwise_mismatch_count |
| C5 canary probes | 32 probes after the forget and the recompute (probe suffix on the post-forget state), compared with C1's | canary_probes (32), post_forget_probes, canary_logit_shift, canary_identity_set_digest |
| C6 negative control | `set_faults` injects stale reuse of the span-era downstream rows into a rerun; the digests must change and the criteria must trip | no manifest metric; recorded as a control result. If it does not trip, the run is vacuous and reported as such (proposal: exit 2) |
| C7 pins | corpus selection digest, procedure parameters (the names in `procedure_parameters_recorded_per_cell`), tiny-model config digest, tokenizer, topology, canary digest | selection_digest_mismatches, procedure_mismatches, model_pin_mismatches, canary_identity_set_digest |

All 17 metrics of `experiment.toml` are covered: bitwise_mismatch_count (C4), canary_logit_shift (C5),
recompute_tokens (C3), span_insertions (C2), span_evictions (C3), downstream_recomputations (C3), canary_probes (C5),
stale_downstream_rows_reused (C3), tokens_after_span (C3), canary_span_tokens (C2), downstream_rows_before_forget (C2),
prefix_reuse_tokens (C3), post_forget_probes (C5), selection_digest_mismatches (C7), procedure_mismatches (C7),
canary_identity_set_digest (C7), model_pin_mismatches (C7).

## 6. Stated limits

- Tiny seeded random model (4 layers, hidden 64): nothing is claimed about a trained model, a GPU kernel or any
  non-deterministic reduction order. Bitwise equality is a property of this CPU reference path, run on one machine
  profile (`hardware/default.toml`).
- CPU reference backend only; the paged or GPU backends are not tested.
- The forget is revocation plus parent-chained chunk keys, not a mid-sequence eviction primitive. The experiment shows
  that revoking a lineage segment and re-running leaves no stale row reachable; it does not show a page-level eviction
  mechanism works.
- Regression-guard character: because chunk keys hash the parent, stale reuse is nearly structurally impossible; a pass
  is the expected outcome and mainly protects the lineage and re-prefill path against regressions. The negative control
  is what shows the guard can fail.
- The probe is a logits comparison after a short probe suffix at one position per probe; it does not detect a trace
  that never reaches those logits, nor any side channel (timing, memory) outside the logits and KV-state digest.
- Canaries 16 to 31 never appear in the span under the proposed layout (section 4); no claim about their placement.
- The KV-state digest covers K and V rows for all prompt positions; hidden activations that are not cached are not
  digested.

## 7. Metrics that may not be producible honestly (to settle before the freeze)

- `bitwise_mismatch_count` and the state digest depend on the WP1 accessor existing in `ptr-runtime`; without it only
  the logits comparison is possible, which would be weaker than the manifest's "any state difference".
- `recompute_tokens == tokens_after_span` (2880) holds only if the backend reuses all 68 prefix chunks and does not hold
  back the last chunk or token. If the backend recomputes a final chunk anyway, the criterion trips honestly; the
  criterion is not adjusted to fit.
- `canary_span_tokens`, the canary layout and the probe suffix are proposals (section 4).

## Head decisions on the open definitions (2026-10-08; no criterion changed)

1. Canary layout: the span carries canaries 0 to 15 (16 canaries x 16 bytes = the 256 span tokens, `canary_span_tokens` = 256); canaries 16 to 31 were never in the span and act as controls; all 32 probes are compared between the forget run and the cold baseline.
2. Probe suffix: the 16 bytes of canary i appended after the question.
3. `bitwise_mismatch_count`: one per differing KV-state digest plus one per differing logits entry (f32 bit patterns), over the forget run and all 32 probes.
4. `recompute_tokens == tokens_after_span` stays exact; a backend that recomputes more trips the criterion honestly.
5. `downstream_recomputations` and `downstream_rows_before_forget` are counted in 16-token chunks after the span (180 chunks for the pinned shape); the tiny-model config digest must be unchanged by the raised limits, else the run is a harness failure.
