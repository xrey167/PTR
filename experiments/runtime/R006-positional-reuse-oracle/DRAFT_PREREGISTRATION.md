# R006 positional-reuse-oracle: draft preregistration

DRAFT, not frozen. This is the pre-freeze text of the preregistration of R006; at the freeze it becomes the
`preregistration_text` file of `config.toml`'s `[preregistration]` table (as `PREREGISTRATION.md`) and its SHA-256 is frozen with
the table. R006 is `planned`: nothing here is a result and no run is evidence. The decisions are made by `criteria.toml` and
applied by `aggregate.py`; this file gives their reasons, the definitions, the case list, what the harness cannot produce and
the limits of the claim. Every criterion is the verbatim text of `experiment.toml` (hypothesis, falsification, metrics) and
the pins of `config.toml`; nothing here changes, weakens or adds a success criterion. Items marked "to confirm" are
proposals of the harness author for the head before the freeze.

The harness is `bins/ptr-bench/src/experiments/r006/` (subcommand `positional-reuse-oracle <seed>`). Today it is a skeleton: the
parameters, the metric list with its rules, the one-layer pins and the stubs of the cases exist, and running the subcommand
fails with a typed "not built yet" error (exit status 2) and prints no metric. The code under test (a position-independent
secondary index and a positional session in `ptr-runtime`) is not part of this tree yet.

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
52 matrix cells, 6 in-flight members). COMPLETED iff every seed meets every coverage rule (93) and no falsifier (68) trips;
FALSIFIED if a coverage rule (the run is vacuous) or a falsifier fails on any seed; INVALID if a record is missing, does not
describe the frozen run, a counter is missing or mistyped, or the exit status disagrees with the rules (INVALID says nothing
about the hypothesis). The harness exits 0 when every rule holds, 1 when one fails, 2 when it cannot run a case (a refusal
that is not the expected typed one, a fixture that does not build, a negative control that does not trip). No rule changes after
a run.

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
  `bitwise_mismatch_count.shift_1` and `.shift_7` (to confirm).
- **Secondary index and rolling hash** (owner decisions 1 and 7): the index key has no parent; inside a namespace, a polynomial
  fingerprint over stride-1 windows of 8 tokens (`rolling_window_tokens`) locates a run; the caller verifies the actual tokens,
  so a collision cannot cause a hit; a run shorter than the window is keyed once by the hash of its length and tokens. The reuse
  unit is a whole admitted segment found at any offset; an edit inside a segment changes its digest and gives zero reuse and a
  typed `SourceDigestMismatch` refusal. A span with no whole-segment match is recomputed, never partly reused; a spliced span never
  includes the last prompt token.
- **Typed refusals** (owner decision 3), checked in this fixed order, the first failing one is the one reported:
  `SourceKindMismatch`, `SourceIdentityMismatch`, `SourceGenerationMismatch`, `SourceDigestMismatch` (own lineage differs),
  `OriginalUpstreamDead` (an upstream or own bound dependency of the original prefix is revoked, superseded or never committed;
  checked before omission, so a superseded `U` is dead, not omitted), `BindingDependencyDead` (base model, adapter, expert),
  `OriginalUpstreamOmitted`, `RawAncestorOmitted`. Names are the plan's; the implementation may rename them before the freeze,
  in which case `config.toml` `refusal_reasons_in_check_order` follows (to confirm). A negative case fails for its named reason.
- **Liveness**: a segment or dependency is live iff its exact (identity, generation, digest) is committed at the ledger anchor and
  not revoked (the `Authority` oracle of `layered_kv::lab`, a real `PtrRuntime`).
- **Multiset scope** (owner decision 6): a segment of a run's original prefix is present iff the whole request holds at least as many
  segments with the same (kind, identity, generation, digest); the position in the request does not matter (reordering allowed).
- **Raw ancestors**: the direct `Raw` segments (kind, content digest) of the run's prefix, as a multiset over the whole request.
  A `Derived` segment carries no ancestor list, so ancestors reached only through a derivation hop are not covered (section 7).
- **Structural staleness** (owner decision 5): a **stale hit** is a lookup that returned or admitted a run with a dead dependency
  (revoked, superseded, never committed at the anchor, or fenced); an **escaped output** is tokens emitted or cache inserts
  made after the event step by a session whose dependency was invalidated. `revoked_upstream_page_reachable_count` and
  `superseded_upstream_page_reachable_count` count post-event lookups that returned the page, whether or not admitted. No
  output comparison is used for any of these.
- **Bitwise comparison**: on `f32` bit patterns (`to_bits`), no tolerance; for a shifted result against the baseline: generated
  tokens, logits bits and the KV-state digest (`KvSession::kv_state_digest`). `bitwise_mismatch_count.shift_<s>` counts one per
  differing KV-state digest, one per differing logits entry, one per differing generated token, and one per non-finite logits
  entry on either side.
- **Exact and untainted**: `tainted_exact_results` counts shifted results whose `KvReuseReport` is not Exact or carries a taint;
  `non_bit_exact_exact_results` counts shifted results labelled Exact that differ from the baseline in any compared bit.
  `exact_shifted_parity_cases` counts shifted cases that are Exact, untainted and bit-equal.
- **Evidence labels** (owner decision 8): the semantic, memory and derived matrix cells run on real sessions with the `Authority`
  oracle (`session_lookup`); the base model is a real recorded dependency of positional sessions (`session_recorded_dependency`;
  if the session work package does not deliver it, the cell is relabelled `declared_dependency` before the freeze, to confirm);
  the adapter and expert cells are declared dependencies with real liveness oracles (an adapter capsule; real expert
  records including the fence epoch), labelled `declared_dependency`.
- **Matrix paths** (`invalidation_matrix_path_meaning`): `normal` = ordinary lookup after the invalidation; `lag` = lagging
  invalidation or index loss (`IndexUpdate::Suppressed`; the liveness check alone must refuse); `restore` = an actual restore from a
  snapshot (snapshots carry no rows: the rows are re-attached by an idempotent re-insert with a rows-digest check, a mismatch
  is a typed error); `inflight` = an in-flight race, run in two variants (the event before the splice; after the splice and before
  decode), after which the session restarts cold with no lookup and no insert.
- **Opt-in**: positional reuse is per backend and off by default, so the frozen R005 and R007 outputs do not change.
- **Corpus selection**: sequence `i` of `corpus_generator_seed` 20240607 for `i` in 0..8, 4096 tokens each, under the domain
  `ptr.r006.corpus-selection.v1`; digest `sha256:3bcd8736b20516454cae2f5007f24201c92756c9a2643e7f126adab6144ab066` (pinned in `params.rs`, a test reproduces it, written into the
  table at the freeze). The seed picks which sequence fills which role (`order(seed, 60, 8)`).

## 4. Cases (per seed) and the metric map

| Case | What runs | Metrics produced |
|---|---|---|
| C1 shift parity | for s in 1, 7, 64, 513, 4096: the origin run populates the index; a baseline cold run at `[filler, C, tail]` through the probe API; the shifted warm run through the secondary index; logits bits, `kv_state_digest` and generated tokens compared | `bitwise_mismatch_count`, `reuse_latency_ms`, `reuse_attempts_per_shift`, `reused_tokens_per_shift`, `secondary_index_reuse_cases_per_shift`, `secondary_index_lookups`, `exact_shifted_parity_cases`, `tainted_exact_results`, `non_bit_exact_exact_results` |
| C2 non-aligned insertion | whole-segment `Raw` insertions of 1 and 7 tokens between unchanged segments with an unaligned origin start; parity as in C1 | `nonaligned_insertion_cases`, `bitwise_mismatch_count`, `secondary_index_lookups`, `exact_shifted_parity_cases` |
| C3 upstream revocation | seed a page built on `U`, revoke `U`, then post-event lookups with matching content, then a fresh prefill | `upstream_revocation_cases`, `old_upstream_pages_seeded_before_revocation`, `revoked_upstream_page_reachable_count`, `post_revocation_secondary_lookups_matching_content`, `fresh_prefill_after_revocation`, `fresh_prefills_completed_after_revocation`, `fresh_prefill_output_mismatches`, `invalidated_case_id_set_digest`, `fresh_prefill_completed_case_id_set_digest` |
| C4 upstream supersession | the same shape with `U` superseded (generation replaced or edited) | `upstream_supersession_cases`, `superseded_upstream_page_reachable_count`, `post_supersession_secondary_lookups_matching_content`, `invalidated_case_id_set_digest`, `fresh_prefill_completed_case_id_set_digest` |
| C5 dead original upstream | lag (index loss) and restore into a runtime that never committed `U`; the request cites `U`, which is dead; the refusal must be `OriginalUpstreamDead` | `dead_original_upstream_shift_attempts`, `shift_hits_with_dead_original_upstream` |
| C6 omitted upstream | a request omitting a segment of the page's original prefix, including the multiplicity case (fewer copies than the run holds); the refusal must be `OriginalUpstreamOmitted` | `omitted_upstream_segment_cases`, `omitted_upstream_admissions` |
| C7 own-lineage mismatch | all upstream live: same identity and tokens but another generation; one token changed (digest); refusals `SourceGenerationMismatch` and `SourceDigestMismatch`; an edit inside a segment gives zero reuse | `source_generation_mismatch_cases`, `source_digest_mismatch_cases`, `source_lineage_mismatch_admissions` |
| C8 raw ancestor removed | a raw upstream segment removed, the rest unchanged; the refusal must be `RawAncestorOmitted` | `raw_ancestor_omission_cases`, `reuse_after_raw_ancestor_removed` |
| C9 matrix | the 52 cells (6 kinds x revocation, supersession x normal, lag, restore, in-flight, plus the expert fence x 4 paths), each recorded per cell; in-flight cells in two variants | `invalidation_matrix_cases`, `invalidation_matrix_stale_hits`, `shifted_reuse_inflight_race_cases`, `shifted_reuse_inflight_outputs`, `shifted_reuse_inflight_cache_inserts`, `invalidated_case_id_set_digest`, `fresh_prefill_completed_case_id_set_digest` |
| C10 non-vacuity controls | (a) a PostRope stack: the same shifted reuse must mismatch bitwise; (b) a liveness-bypass fault mode: the reachability and dead-original admission counters must trip; (c) a primary-only run on the shifted content gets 0 reused tokens; a control that does not trip is exit 2 | no manifest metric (recorded under `cases.C10`) |
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
| `tainted_exact_results` | C1 |
| `non_bit_exact_exact_results` | C1 |
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
  shape, tokenizer, configuration digest and chain are read from the backend. `procedure_mismatches` is weak for the constants.
- All staleness metrics are structural (section 5).

## 7. Stated limits (what this experiment does NOT show)

- One layer, tiny seeded random model, CPU reference path, one machine profile (`hardware/default.toml`): nothing about a trained
  model, a GPU kernel or a non-deterministic reduction order. Bit equality is a property of this path; a different build binds a
  different chain.
- Not the multi-layer case: layers above 0 depend on context and positional reuse there is Approximate (M016); taint and the
  effect boundary are not tested here.
- Staleness is structural, not numeric (section 5); the claim is only as strong as the controls of C10.
- Adapter and expert dependencies are declared, with real liveness oracles, not model influence; an expert at layer j cannot
  affect layer-0 K/V under the conservative per-layer rule.
- Whole-segment reuse only: no sub-segment reuse claim; an edit inside a segment gives zero reuse.
- Raw ancestors are direct `Raw` prefix segments; derivation hops are not traced.
- The shift is built with a `Raw` filler, so the shifted segment's own lineage stays equal while the ordered prefix lineage differs;
  other ways of shifting (a reorder of whole segments) are covered by the multiset rule and C6 but not by every pinned shift.
- No wall-clock claim; `reuse_latency_ms` is a timing record.
- KV snapshots carry no rows; restore re-attaches rows by re-insert.

## 8. Not frozen

`config.toml` has no `[preregistration]` table, `experiment.toml` has no `preregistration_sha256`, and R006 is absent from
`experiments/preregistration.toml`, so `aggregate.py` refuses to aggregate. The freeze (a later step, not part of this draft) adds the
table, the digests of this file (renamed `PREREGISTRATION.md`) and `criteria.toml`, the entrypoint and the registry entry.
