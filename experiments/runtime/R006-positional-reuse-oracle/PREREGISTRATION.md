# R006 positional-reuse-oracle: preregistration

DRAFT, not frozen. This is the pre-freeze text of the preregistration of R006; at the freeze it becomes the
`preregistration_text` file of `config.toml`'s `[preregistration]` table (as `PREREGISTRATION.md`) and its SHA-256 is frozen with
the table. R006 is `planned`: nothing here is a result and no run is evidence. The decisions are made by `criteria.toml` and
applied by `aggregate.py`; this file gives their reasons, the definitions, the case list, what the harness cannot produce and
the limits of the claim. Every criterion of `experiment.toml` (hypothesis, falsification, metrics) and every pin of `config.toml`
is taken verbatim; the only rules added are the five harness-added vacuity and bystander rules over internal counters listed in
section 2, which can only make a seed vacuous or falsified and never turn a failing seed into a pass.

The harness is `bins/ptr-bench/src/experiments/r006/` (subcommand `positional-reuse-oracle <seed>`). It runs the cases C1 to C11
below once per seed against the CPU reference KV backend of `ptr-runtime` with positional reuse opted in (the position-independent
secondary index of its prefix catalog and the positional session), through its public API only, and prints one JSON object.

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command, set at the freeze):
`cargo run --release --locked -p ptr-bench -- positional-reuse-oracle <seed>`

Scope: with pre-RoPE key storage, a prompt segment reused at a shifted position in a one-layer decoder is bit-identical to a
full prefill at the new position (the baseline), for every pinned shift (1, 7, 64, 513, 4096 tokens) and for non-page-aligned
insertions of 1 and 7 tokens; reuse goes through the catalog and its position-independent secondary index; a shift hit is
admitted only if the page's own source-segment lineage equals the request's segment exactly, every upstream input of the
page's original prefix is live, every segment of that original prefix and every recorded raw ancestor is present in the whole
request as a multiset; the protection triggers on every invalidation (revocation, supersession, expert fence) across the 52 cells
of the matrix, in-flight consumption included. It is the one-layer Exact positional oracle: multi-layer positional reuse is
Approximate and belongs to M016.

## 1. Owner decisions of 2026-10-08 (binding, verbatim)

These are also the keys `owner_decision_2026_10_08_*` of `config.toml`.

1. Granularity: rolling hash over sub-page spans in the secondary index, so non-aligned shifts (1- and 7-token insertions) and all preregistered shifts are reusable; a repack step assembles the reused PreRope rows at the new positions.
2. Matrix: reuse the R005/R010 matrix machinery (layered_kv shared helpers; same cell semantics normal, lag, restore, in-flight); R005's frozen files stay untouched.
3. Refusals: one typed error per refusal reason of the secondary index (e.g. original upstream dead, upstream segment omitted, raw-ancestor multiset differs, own lineage differs), so criteria count per reason and negative cases fail for the named reason.
4. Go: implement R006 fully after R007 (size L).
5. (2026-10-08) Staleness is measured structurally: a stale hit = a lookup that returned (or admitted) a dead run; an escaped output = tokens emitted or cache inserts after the event step. Stated in the preregistration. Negative controls: (a) a PostRope stack where the same shifted reuse must mismatch bitwise; (b) a liveness-bypass fault mode under which the reachability / dead-original admission counters must trip; (c) primary-only run gets 0 reused tokens.
6. Multiset presence of original-prefix segments is checked over the whole request (reordering allowed).
7. Reuse unit = a whole admitted segment, located at any offset by the rolling hash (W pinned in config); an edit inside a segment gives zero reuse and a typed refusal.
8. Adapter and expert cells run as declared dependencies with real liveness oracles (adapter capsule; real expert records incl. fence), labelled evidence = declared_dependency. KV snapshots carry no rows; restore re-attaches rows by idempotent re-insert with a rows-digest check.
9. (head) config.toml additions before the freeze are allowed (R006 is planned): rolling_window_tokens = 8, the shift construction (Raw filler of s tokens upstream), harness limits, evidence labels. Positional reuse is opt-in per backend and off by default so frozen R005/R007 outputs do not change.

## 2. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written
`>= N`, a "> 0" violation `<= 0`, a "!= x" violation `== x`; a grouped metric has one rule per member: 5 shifts, 2 insertion sizes,
52 matrix cells, 6 in-flight members). COMPLETED iff every seed meets every coverage rule (97) and no falsifier (69) trips;
FALSIFIED if a coverage rule (the run is vacuous) or a falsifier fails on any seed; INVALID if a record is missing, does not
describe the frozen run, a counter is missing or mistyped, or the exit status disagrees with the rules (INVALID says nothing
about the hypothesis). No rule changes after a run.

**The five harness-added rules.** Of the 97 + 69 rules, 161 are the manifest's (one per member of a grouped metric) and five read a
counter under `internal`, the harness's own bookkeeping, which is no metric of `experiment.toml`: coverage
`internal.live_control_failures <= 0`, `internal.source_kind_mismatch_cases >= 1`, `internal.source_identity_mismatch_cases >= 1`,
`internal.bystander_reuse_cases >= 1`, and the falsifier `internal.bystander_reuse_losses <= 0`. They are listed so that the freeze
covers them; each resolves to a member the harness always prints (at 0 at least).

**Exit status of the harness.** 0 when every rule holds, 1 when one fails, 2 when it cannot judge.

*Exit 2 is INVALID (a harness-integrity failure, evidence about nothing), and only these:*

- a refusal of the wrong type where another typed reason was expected (for example `OriginalUpstreamOmitted` in a case that names
  `OriginalUpstreamDead`), and a cancellation with a reason other than the one the in-flight cell asserts;
- a cold baseline that differs from the independent one-layer stack (so the baseline is no anchor);
- an outcome accounting that does not add up (prompt tokens != primary hit + repacked + prefilled), a pin still held after a run;
- a positional run without a `kv_report` (its taint cannot be read), a fixture that does not build, an origin run that was not a
  first run on an empty index;
- a negative control that runs and does not trip its falsifier (the criterion could not fail), a primary-only run that reused
  shifted content, an Approximate insert that was not refused as `NotExact`.

*A failed positive control is not exit 2.* A live control that does not reuse the whole segment (or is not bit-identical to a cold
run: that is counted in the Exact counters), an origin the index did not take (an insert refusal on the honest path, a capacity
refusal), a lag cell whose stale page was never returned, a snapshot without secondary runs, a restore control that did not get its
rows back, an over-refusing reorder control, an in-flight session with no admitted span, and a negative control whose premise fails
(nothing was reused, so there is nothing to bypass) are recorded: `internal.live_control_failures` counts them,
`cases.control_failures` names them, the case or cell is not counted (its case counter stays 0), its post-event assertions are
skipped, and the coverage rules fail, so the seed FALSIFIES (vacuous). A runtime whose reuse returns 0 everywhere therefore
falsifies; it is never INVALID.

## 3. Definitions

- **Model**: the seeded random tiny decoder of R005 with one layer (hidden 64, 4 heads, 2 kv heads, head_dim 16, vocabulary
  256, weight_seed 20240607); the digest of `tiny_model_config` is `sha256:a77e017c8ee3b56bb06a0b22512aae3a70a1b65d4ae283bf2d8bb02353029c2e` and equals the canonical
  JSON digest computed by `r006::pins` (checked by a test). Keys are stored `PreRope`; the rotation is applied at use, at the
  row's position in the consuming request. Tokenizer: the byte tokenizer. Checkpoint: `none: seeded random`; topology:
  `none: dense tiny decoder`.
- **Why one layer**: the stored K and V rows of a token at layer 0 are a pure function of the token (no attention over context
  yet), so they are the same bits wherever the token sits; a test of the harness (`r006::pins`) shows this on the independent
  one-layer stack and shows by a two-layer control that layer 1 rows depend on the context. This is also why staleness cannot
  be seen in any output (section 5).
- **Shift construction** (owner decision 9): origin run `[C, tail]` populates the index; the baseline and the shifted run are
  `[filler, C, tail]`, where `filler` is a `Raw` segment of exactly `s` tokens inserted upstream of `C` (`s` in
  1, 7, 64, 513, 4096). `C` is a segment of 48 tokens and `tail` one of 16 tokens, so `C` keeps its own lineage (kind, identity,
  generation, digest) while the ordered prefix lineage differs. Prompts reach 4160 tokens; the harness raises
  `ReferenceKvLimits.max_prompt_tokens` to 4352 (crate default stays 256) and uses `max_new_tokens` 8.
- **Non-aligned insertion**: origin `[A, C, tail]` with `A` a `Raw` segment of 24 tokens (so `C` starts at 24, not a multiple of the
  16-token page); the request inserts a `Raw` segment of 1 or of 7 tokens between `A` and `C` (`C` starts at 25 or 31, not
  aligned either). The comparison is the same as the shift's; the shift distance of `C` is 1 and 7, so a mismatch counts into
  `bitwise_mismatch_count.shift_1` and `.shift_7` (the same members and so the same rules as C1: a C2 mismatch trips the same
  falsifier as a C1 mismatch of that distance).
- **Secondary index and rolling hash** (owner decisions 1 and 7): the index key has no parent; inside a namespace, a polynomial
  fingerprint over stride-1 windows of 8 tokens (`rolling_window_tokens`) locates a run; the caller verifies the actual tokens,
  so a collision cannot cause a hit; a run shorter than the window is keyed once by the hash of its length and tokens. The reuse
  unit is a whole admitted segment found at any offset; an edit inside a segment changes its digest and gives zero reuse and a
  typed `SourceDigestMismatch` refusal. A span with no whole-segment match is recomputed, never partly reused; a spliced span never
  includes the last prompt token. Reuse of the shifted segment is required to be **whole**: the segment's 48 tokens repacked and
  nothing of the shift or the tail under-reused (`under_reused_tokens`); `secondary_index_reuse_cases_per_shift`,
  `nonaligned_insertion_cases` and the parity case count only then, so a reuse of one token is not parity.
- **Typed refusals** (owner decision 3), checked in this fixed order, the first failing one is the one reported:
  `SourceKindMismatch`, `SourceIdentityMismatch`, `SourceGenerationMismatch`, `SourceDigestMismatch` (own lineage differs),
  `OriginalUpstreamDead` (an upstream or own bound dependency of the original prefix is revoked, superseded or never committed;
  checked before omission, so a superseded `U` is dead, not omitted), `BindingDependencyDead` (base model, adapter, expert),
  `OriginalUpstreamOmitted`, `RawAncestorOmitted`. These names and this order are the code's and
  `config.toml` `refusal_reasons_in_check_order` (a test pins the order). A negative case fails for its named reason. The kind and
  identity refusals have their own coverage rules (`internal.source_kind_mismatch_cases`,
  `internal.source_identity_mismatch_cases`), so they are not left to the wrong-type check alone.
- **Liveness**: a segment or dependency is live iff its exact (identity, generation, digest) is committed at the ledger anchor and
  not revoked (the `Authority` oracle of `layered_kv::lab`, a real `PtrRuntime`).
- **Multiset scope** (owner decision 6): a segment of a run's original prefix is present iff the whole request holds at least as many
  segments with the same (kind, identity, generation, digest); the position in the request does not matter (reordering allowed).
- **Raw ancestors**: the direct `Raw` segments (kind, content digest) of the run's prefix, as a multiset over the whole request.
  A `Derived` segment carries no ancestor list, so ancestors reached only through a derivation hop are not covered (section 7).
- **Structural staleness** (owner decision 5): a **stale hit** is a lookup that returned or admitted a run with a dead dependency
  (revoked, superseded, never committed at the anchor, or fenced); an **escaped output** is tokens emitted, spans repacked or cache
  inserts made after the event step by a session whose dependency was invalidated. No output comparison is used for any of these.
- **Reachable, admitted**: `reachable` is the number of **candidate runs a lookup returned**, not a number of lookups: the spans it
  admitted, the segments whose only admitted run covered the last prompt token (the runtime never splices the last token, so these
  are counted by `last_token_skips`, not as a span), and the candidates it refused with a typed reason. `revoked_` and
  `superseded_upstream_page_reachable_count` count it for the post-event lookups with matching content, so a lost invalidation
  that the use-time check still refuses with a typed reason still counts. *Admitted* is the spliced spans plus the last-segment
  skips. In every **negative** case (C3 to C8 and the matrix's normal, lag and restore cells) the tail run is refused, so a correct
  runtime gives 0 for both; the admission counters (`shift_hits_with_dead_original_upstream`, `omitted_upstream_admissions`,
  `source_lineage_mismatch_admissions`, `reuse_after_raw_ancestor_removed`) count the skips too, so that a runtime that stops
  enforcing a check for the final segment is not invisible. In the **positive** cases (C1, C2, the reorder control) the spliced
  spans alone are the measure. Per path: a normal cell counts every candidate returned after the event (admitted or refused: after an
  applied invalidation nothing may be returned); a lag cell counts only the admissions (the dead page is reachable by design and is
  refused with its typed reason); a restore cell counts the rows attached or segments inserted by a re-insert under the post-event
  ledger plus whatever a lookup returns afterwards.
- **Bitwise comparison**: on `f32` bit patterns (`to_bits`), no tolerance, no magnitude interpreted: each shifted result is compared
  with a true `Cold` probe of the same request (no lookup, no insert, no secondary index) and with an independent one-layer pods
  stack (nothing of the runtime under test but the model code). `bitwise_mismatch_count.shift_<s>` is, per shifted comparison, the
  sum of seven terms: (1) 1 if the KV-state digest (`KvSession::kv_state_digest`) differs from the cold baseline's; (2) the number
  of last-prompt logits entries (256) whose bits differ from the cold baseline's, plus any length difference; (3) the number of
  generated tokens that differ from the baseline's, plus any length difference; (4) 1 if the decode logits digest (the hash of the
  logits bits of the decode steps) differs from the baseline's; (5) one per non-finite logits entry on either side (a non-finite
  vector is never a silent pass); (6) the number of generated tokens that differ from the independent stack's cold run; (7) 1 if the
  decode logits digest differs from the independent stack's. The rule is `<= 0`. The independent anchor covers the generated tokens
  and the decode logits digest, not the KV digest, and not the last-prompt logits entries. A cold baseline that differs from the
  independent stack is exit 2, not a count (the baseline would be no anchor).
- **Exact and untainted, and their scope**: `tainted_exact_results` counts positional runs whose `KvReuseReport` is not Exact or
  carries a taint; `non_bit_exact_exact_results` counts runs labelled Exact that differ from the cold baseline or the independent
  stack in any compared bit. Both count **every Exact-labelled positional run the harness compares with a cold run**, not only the
  shifted results: the shifted runs of C1 and C2, the live controls of C3, C4 and the matrix (so a restored row that is reused is
  compared bitwise too), the fresh prefills of C3 and C4 and the matrix's post-event runs, and the refused runs of C5 to C8 (which
  must equal a cold run). They exclude the runs of C10. The mismatches of a refused run also go to `negative_case_output_mismatches`
  (internal). `exact_shifted_parity_cases` counts the shifted cases that are Exact, untainted, bit-equal and whole-segment. A run
  without a `kv_report` is exit 2, not a tainted result.
- **Evidence labels** (owner decision 8, split by path). The label of a matrix cell says what it measures:
  - `session_lookup`: the normal, lag and restore cells of the semantic, memory and derived segments run on real positional
    sessions under the `Authority` oracle. The events reach the backend's catalog through the runtime's own event mapping
    (`apply_committed_event`); **the semantic supersession is applied by identity from the harness** (`supersede_inputs`), because a
    backend has no prepared semantic delta and the committed event carries none.
  - `session_lineage_guard`: the **in-flight** cells of the semantic, memory and derived segments. `shifted_reuse_inflight_race_cases.<semantic|memory>_*` counts
    races in which a positional session lost an input of its own lineage; the secondary-index guards are not independently
    exercised for segment kinds, because by the omission rule they are a subset of the lineage. The cancel reason (`InputNotLive`)
    is asserted; any other reason is exit 2.
  - `session_recorded_dependency`: every cell of the base model, a real recorded dependency of every positional session; its
    in-flight cells isolate `SecondaryDependencyNotLive` (asserted; any other reason is exit 2). The pin tombstone
    (`SecondaryRunTombstoned`) is the deciding reason in no session cell.
  - `declared_dependency`: the normal, lag and restore cells of the adapter and the expert: runs inserted into a standalone
    catalog's secondary index with the adapter capsule or the expert declared as an extra dependency, under real oracles (an adapter capsule on a ledger;
    real expert records including the fence epoch). The events reach that catalog through `KvPrefixCatalog::apply_committed_event`,
    the runtime's own mapping (a public additive wrapper of `invalidate_catalog_for_event`), not through a primitive the harness
    chooses.
  - `declared_pin_protocol`: the in-flight cells of the adapter and the expert: the index's pin protocol (`revalidate` before each
    consumption and the insert-time dependency check), not a session. Their two variants differ **only by a lag flag** (the index
    was or was not told of the event), not by a position before or after a splice.
- **Matrix paths** (`invalidation_matrix_path_meaning`): `normal` = ordinary lookup after the invalidation; `lag` = lagging
  invalidation or index loss (`IndexUpdate::Suppressed`; the liveness check alone must refuse); `restore` = an actual restore from a
  snapshot (snapshots carry no rows: the rows are re-attached by an idempotent re-insert with a rows-digest check; the mismatch
  error itself is not exercised in the run); `inflight` = an in-flight race, run in two variants (the event before the splice; after
  the splice and before decode).
- **Restart and fresh prefill**: after any cancellation the harness calls `restart_with`, which runs cold (no lookup, no insert),
  and requires a **fresh prefill**: no primary hit, nothing repacked, no splice, no lookup, the whole prompt prefilled and
  `cold_restart` set; its generated tokens, decode logits digest, event log and KV-state digest must equal a cold run's and its
  tokens and logits digest the independent stack's. A differing result counts into `fresh_prefill_output_mismatches`. A restart that is not fresh, or whose digests differ,
  leaves the cell out of the completed set, so the identity-set rule fails: a vacuous seed (FALSIFIED), not a dedicated falsifier. The same
  fresh-prefill check (and the same counter) applies to the post-event requests of C3, C4 and the matrix. In C3, `fresh_prefill_after_revocation` counts
  the post-event requests that ran as a fresh prefill and `fresh_prefills_completed_after_revocation` the cases all of whose
  requests did; the two differ in unit and are equal only when each case has one request.
- **Opt-in**: positional reuse is per backend and off by default, so the frozen R005 and R007 outputs do not change.
- **Reduced scale**: the smoke members are `[1, 7, 20, 33, 129]` at a reduced `max_prompt_tokens`; a reduced record is never evidence.
  The harness compares its procedure parameters with the pins of `config.toml` (`position_shift_tokens`, `harness_max_prompt_tokens`,
  segment, tail and origin-prefix lengths, the model), not with the shape the run was given, so a reduced run has
  `procedure_mismatches > 0`; `aggregate.py` independently rejects a `scale` other than `pinned` and a `procedure.shape_is_pinned`
  other than true. `rolling_window_tokens` is observed from an index built with the harness's own configuration (the window is not
  otherwise observable through the backend), so it is a constant on both sides.
- **Corpus selection**: sequence `i` of `corpus_generator_seed` 20240607 for `i` in 0..8, 4096 tokens each, under the domain
  `ptr.r006.corpus-selection.v1`; digest `sha256:3bcd8736b20516454cae2f5007f24201c92756c9a2643e7f126adab6144ab066` (pinned in `params.rs`, a test reproduces it, written into the
  table at the freeze). The seed picks which sequence fills which role (`order(seed, 60, 8)`).

## 4. Cases (per seed) and the metric map

| Case | What runs | Metrics produced |
|---|---|---|
| C1 shift parity | for s in 1, 7, 64, 513, 4096: the origin run populates the index; a baseline cold run at `[filler, C, tail]` through the probe API; the shifted warm run through the secondary index; logits bits, `kv_state_digest`, generated tokens and decode digest compared with the baseline and with the independent stack; the whole segment must be repacked | `bitwise_mismatch_count`, `reuse_latency_ms`, `reuse_attempts_per_shift`, `reused_tokens_per_shift`, `secondary_index_reuse_cases_per_shift`, `secondary_index_lookups`, `exact_shifted_parity_cases`, `tainted_exact_results`, `non_bit_exact_exact_results` |
| C2 non-aligned insertion | whole-segment `Raw` insertions of 1 and 7 tokens between unchanged segments with an unaligned origin start; parity as in C1 | `nonaligned_insertion_cases`, `bitwise_mismatch_count`, `secondary_index_lookups`, `exact_shifted_parity_cases` |
| C3 upstream revocation | seed a page built on `U` and an unrelated bystander, a live control (the content reused in full while `U` is live, bit-identical to cold), revoke `U`, post-event lookups with matching content, a fresh prefill, and the bystander must still be reused | `upstream_revocation_cases`, `old_upstream_pages_seeded_before_revocation`, `revoked_upstream_page_reachable_count`, `post_revocation_secondary_lookups_matching_content`, `fresh_prefill_after_revocation`, `fresh_prefills_completed_after_revocation`, `fresh_prefill_output_mismatches`, `invalidated_case_id_set_digest`, `fresh_prefill_completed_case_id_set_digest` |
| C4 upstream supersession | the same shape with `U` superseded (generation replaced or edited), both the stale and the new generation requested | `upstream_supersession_cases`, `superseded_upstream_page_reachable_count`, `post_supersession_secondary_lookups_matching_content`, `invalidated_case_id_set_digest`, `fresh_prefill_completed_case_id_set_digest` |
| C5 dead original upstream | lag (index loss), a lookup under a ledger that never committed `U`, and a restore into a runtime that never committed `U` (rows attached and admissions after it feed `shift_hits_with_dead_original_upstream`); the request cites `U`, which is dead; the refusal must be `OriginalUpstreamDead` | `dead_original_upstream_shift_attempts`, `shift_hits_with_dead_original_upstream` |
| C6 omitted upstream | a request omitting a segment of the page's original prefix, including the multiplicity case (fewer copies than the run holds); the refusal must be `OriginalUpstreamOmitted`; a reordering that keeps every segment must be admitted (control) | `omitted_upstream_segment_cases`, `omitted_upstream_admissions` |
| C7 own-lineage mismatch | all upstream live: same identity and tokens but another generation; one token changed (digest); another identity; another kind; each refusal is its named reason; an edit inside a segment gives zero reuse | `source_generation_mismatch_cases`, `source_digest_mismatch_cases`, `source_lineage_mismatch_admissions` (and the internal kind and identity counters) |
| C8 raw ancestor removed | a raw upstream segment removed, the rest unchanged; the refusal must be `RawAncestorOmitted` | `raw_ancestor_omission_cases`, `reuse_after_raw_ancestor_removed` |
| C9 matrix | the 52 cells (6 kinds x revocation, supersession x normal, lag, restore, in-flight, plus the expert fence x 4 paths), each recorded per cell; in-flight cells in two variants | `invalidation_matrix_cases`, `invalidation_matrix_stale_hits`, `shifted_reuse_inflight_race_cases`, `shifted_reuse_inflight_outputs`, `shifted_reuse_inflight_cache_inserts`, `invalidated_case_id_set_digest`, `fresh_prefill_completed_case_id_set_digest` |
| C10 non-vacuity controls | (a) a PostRope stack: the same shifted reuse must mismatch bitwise; (b) a liveness-bypass fault mode: the reachability (revoked and superseded), dead-original and in-flight output counters must trip, and an in-flight race whose event closure also performs a live run that inserts must trip the cache-insert counter; (b2) a really admitted report, and one whose only admission is the last segment's, must trip each admission counter; (c) a primary-only run on the shifted content gets 0 reused tokens; (d) the index partitions by production mode; (e) a four-layer backend refuses positional reuse. A control that runs and does not trip is exit 2 | no manifest metric (recorded under `cases.C10`) |
| C11 pins | corpus selection digest, the procedure parameters (the names of `procedure_parameters_recorded_per_cell`, each observed and compared with the pinned value), model pins | `model_pin_mismatches`, `procedure_mismatches`, `selection_digest_mismatches` |

Every metric of `experiment.toml` and the case(s) that feed it:

| metric | case |
|---|---|
| `bitwise_mismatch_count` | C1, C2 |
| `reuse_latency_ms` | C1 |
| `reuse_attempts_per_shift` | C1 |
| `secondary_index_lookups` | C1, C2 |
| `nonaligned_insertion_cases` | C2 |
| `upstream_revocation_cases` | C3 |
| `revoked_upstream_page_reachable_count` | C3 |
| `fresh_prefill_after_revocation` | C3 |
| `reused_tokens_per_shift` | C1 |
| `secondary_index_reuse_cases_per_shift` | C1 |
| `post_revocation_secondary_lookups_matching_content` | C3 |
| `old_upstream_pages_seeded_before_revocation` | C3 |
| `dead_original_upstream_shift_attempts` | C5 |
| `shift_hits_with_dead_original_upstream` | C5 |
| `omitted_upstream_segment_cases` | C6 |
| `omitted_upstream_admissions` | C6 |
| `upstream_supersession_cases` | C4 |
| `superseded_upstream_page_reachable_count` | C4 |
| `selection_digest_mismatches` | C11 |
| `procedure_mismatches` | C11 |
| `source_generation_mismatch_cases` | C7 |
| `source_digest_mismatch_cases` | C7 |
| `source_lineage_mismatch_admissions` | C7 |
| `fresh_prefills_completed_after_revocation` | C3 |
| `fresh_prefill_output_mismatches` | C3 |
| `post_supersession_secondary_lookups_matching_content` | C4 |
| `invalidated_case_id_set_digest` | C3, C4, C9 |
| `fresh_prefill_completed_case_id_set_digest` | C3, C4, C9 |
| `invalidation_matrix_cases` | C9 |
| `invalidation_matrix_stale_hits` | C9 |
| `shifted_reuse_inflight_race_cases` | C9 |
| `shifted_reuse_inflight_outputs` | C9 |
| `shifted_reuse_inflight_cache_inserts` | C9 |
| `exact_shifted_parity_cases` | C1, C2 |
| `tainted_exact_results` | C1 to C9 (every Exact positional run compared with a cold run) |
| `non_bit_exact_exact_results` | C1 to C9 (as above) |
| `raw_ancestor_omission_cases` | C8 |
| `reuse_after_raw_ancestor_removed` | C8 |
| `model_pin_mismatches` | C11 |

The map is held in code (`r006::metrics::CASE_MAP`) and a test checks that every metric has a case. Case-id sets: the invalidated case
ids are those of C3, C4 and the matrix's revoked or superseded runs; the fresh-prefill-completed ids are those of the runs that
completed by fresh prefill; both are digested by `id_set_digest` and compared as sets.

## 5. Why staleness is structural

In a one-layer model the layer-0 K and V rows of a token are a pure function of the token. A page built on a revoked upstream
segment `U` is bit-identical to a fresh one, because the rows of the reused segment do not depend on `U`. So no comparison of
logits, KV digests or generated tokens can tell a stale hit from a fresh one. All revocation, supersession, omission and
in-flight evidence is therefore a counter: a lookup that returned or admitted a dead run, and tokens emitted or cache inserts
after the event step. The controls of C10 are the proof that these counters can trip; without them a pass would be a statement
about a harness that cannot fail.

## 6. Metrics that are weak or cannot be produced honestly

- `reuse_latency_ms` is timing only (an object of per-shift milliseconds), read by no rule; splicing saves only the span's compute and
  snapshot hashing dominates, so there is no wall-clock or speedup claim.
- `fresh_prefill_after_revocation` is described in the manifest as "no layers >= 1": in one layer there is no surviving state to
  keep, so the counter records that a fresh prefill followed the revocation, not that upper layers were dropped.
- `raw_ancestor_omission_cases` and `reuse_after_raw_ancestor_removed` cover direct `Raw` segments of the prefix only. The
  manifest's "transitive lineage" cannot be measured through a `Derived` segment, which carries no ancestor list.
- The adapter and expert cells of `invalidation_matrix_cases` and `invalidation_matrix_stale_hits` are declared dependencies (section 3).
- `tainted_exact_results` and `non_bit_exact_exact_results`: no Approximate producer exists in R006, so `derived_state_carriers =
  {secondary_index_entry: ["mode"]}` is tested only as a namespace partition by mode, not as taint on approximate content.
- `model_pin_mismatches` is weak for the checkpoint and the topology (constants: `none: seeded random`, `none: dense tiny decoder`); the
  shape, tokenizer, configuration digest and chain are read from the backend. `procedure_mismatches` is weak for the constants and
  for `rolling_window_tokens` (section 3).
- **Coverage counters that hold by construction**: `reuse_attempts_per_shift`, `secondary_index_lookups`,
  `old_upstream_pages_seeded_before_revocation`, `post_revocation_secondary_lookups_matching_content`,
  `post_supersession_secondary_lookups_matching_content` and `invalidation_matrix_cases` are increments that happen whenever the
  harness completes the case; they cannot fail on their own. The informative coverage rules are `reused_tokens_per_shift`,
  `secondary_index_reuse_cases_per_shift` (the whole segment reused), the typed-refusal counts, `internal.live_control_failures`, the kind
  and identity refusal counters, `internal.bystander_reuse_cases` and the identity-set rule.
- **Criteria without a run-time control**: the omission, own-lineage (generation, digest, identity, kind), raw-ancestor and
  superseded-reachability families (the last only in the bypass control) cannot be broken by a fault in the runtime from the harness. They are guarded twice, by
  an admission counter (which also counts the last segment's skip) and by the coverage rule of the typed refusal (a check that is
  missing also zeroes it, so the seed goes vacuous); the wiring of the four admission counters is shown by feeding a really admitted
  report (C10 b2). Cache inserts after an in-flight event cannot move on a correct runtime (a cancelled session inserts nothing and the
  insert-time dependency check refuses a dead one): the measure is shown to move by a race whose event closure also performs a live
  run, which says the counter reads inserts, not that a defective session would insert. `tainted_exact_results` has no control (no
  Approximate producer).
- All staleness metrics are structural (section 5).

## 7. Stated limits (what this experiment does NOT show)

- One layer, tiny seeded random model, CPU reference path, one machine profile (`hardware/default.toml`): nothing about a trained
  model, a GPU kernel or a non-deterministic reduction order. Bit equality is a property of this path; a different build binds a
  different chain.
- Not the multi-layer case: layers above 0 depend on context and positional reuse there is Approximate (M016); taint and the
  effect boundary are not tested here.
- Staleness is structural, not numeric (section 5); the claim is only as strong as the controls of C10 and their reach (section 6).
- The in-flight cells of the segment kinds measure the request's own lineage guard, not the secondary-index guards (section 3).
- Adapter and expert dependencies are declared, with real liveness oracles, not model influence; an expert at layer j cannot
  affect layer-0 K/V under the conservative per-layer rule. The semantic supersession is applied by identity from the harness.
- Whole-segment reuse only: no sub-segment reuse claim; an edit inside a segment gives zero reuse.
- Raw ancestors are direct `Raw` prefix segments; derivation hops are not traced.
- The shift is built with a `Raw` filler, so the shifted segment's own lineage stays equal while the ordered prefix lineage differs;
  other ways of shifting (a reorder of whole segments) are covered by the multiset rule and C6 but not by every pinned shift.
- Single thread: the event lands between two steps of a session, never inside one; there is no concurrency test. There is no
  cross-salt isolation test. The restore rows-digest mismatch error is not exercised in the run.
- No wall-clock claim; `reuse_latency_ms` is a timing record.
- KV snapshots carry no rows; restore re-attaches rows by re-insert.

## 8. Not frozen

`config.toml` has no `[preregistration]` table, `experiment.toml` has no `preregistration_sha256`, and R006 is absent from
`experiments/preregistration.toml`, so `aggregate.py` refuses to aggregate. The freeze (a later step, not part of this draft) adds the
table, the digests of this file (renamed `PREREGISTRATION.md`) and `criteria.toml`, the entrypoint and the registry entry.
