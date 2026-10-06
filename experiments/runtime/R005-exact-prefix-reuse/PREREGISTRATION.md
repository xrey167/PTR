# R005 exact-prefix-reuse: draft preregistration (NOT frozen)

Status: **draft for owner review.** R005 stays `planned`. Nothing here is frozen, no record or result
exists, `experiments/preregistration.toml` and the experiment's `config.toml` / `experiment.toml` are
untouched. The harness is `bins/ptr-bench/src/experiments/r005/` (subcommand `exact-prefix-reuse <seed>`,
which is **not** yet the manifest's entrypoint). This file is the only document of the draft.

Scope: Exact half only. Approximate cases are R012; witness reuse is evidenced statically (owner decisions of 2026-10-06).

## 0. Headline of the dry run (read this first)

Dry run, seeds 1, 17, 29, 43, 71, 101 (release build, 4 vCPU Xeon 2.1 GHz): about **112 s per seed**, identical outcome on every seed.
No coverage prerequisite is unmet and no metric is missing. **Eight falsifier rows trip on every seed, all from one
runtime gap** (finding A below): a semantic input superseded by a committed edit is not seen by the use-time liveness check.
Every other non-unsupported falsifier passes (bitwise parity on 158 warm/cold comparisons, hit rate 0.4921875, no revoked fork,
no stale hit on revocation, no cross-principal/project/lineage/environment/position hit, no admitted forged or stale witness,
capacity bounds and saturation, regimes, transform change). A control with the semantic key held as a lifecycle target shows
zero stale hits, so the check works where an input has a lifecycle generation. Do not freeze until the owner decides A (section 5).

## 1. (a) Values the config leaves open, with proposal and reason

| value | proposal | reason / how verified |
|---|---|---|
| max_new_tokens | 8 | backend default (`ReferenceKvLimits::default`); regime admission is prompt + 8 |
| max_prompt_tokens | 256 | the main prompt length; max_position_embeddings 264 (not hashed in the chain) |
| intermediate_size | 128 | hard-coded in `kv_infer.rs::stack_config`; `pin::verify` rebuilds the stack from the proposal and reproduces the backend's chain id and a cold run's logits digest |
| rope_theta / rms_norm_eps | 10000.0 / 1e-6 | same; hidden_act silu, no sliding window, key encoding PreRope |
| QK-norm, main arms | disabled | the backend cannot enable it |
| QK-norm, transform-change arm | enabled, separate stack of the same tiny shape | needs a key-norm weight to change; evidenced at chain derivation plus catalog lookup (per-layer entries under chain(l)), not through backend inference |
| NTK factor / original max positions | 4.0 / 30 | 16-token request (24 with new tokens) and 17-token request (25) stay plain; a 24-token prompt (32) is below the boundary yet crosses it while decoding (positive control); a 36-token request (44) forces a transition |
| over-capacity workloads | 128-token unique prompts (8 nodes); 512 fill (exactly 4096 nodes / 64 MiB), 64 extra, 8 excess requests, max_sessions 520 | each workload binds one limit exactly; pinned by one open fork per filled prompt so demand exceeds capacity |
| principal/project names | r005-principal-a/b, r005-project-a/b | |
| per-seed role | the seed orders requests and places in-flight race points (seeded Fisher-Yates / SplitMix64); the corpus is fixed by `corpus_generator_seed` | config pins the corpus independent of the seed |
| CPU environment fields | arch `cpu:x86_64:<model name from /proc/cpuinfo>`; driver and CUDA runtime `not-applicable:cpu-reference`; kernel library `ptr-pods reference f32 kernels, sequential reduction order, no external library`; backend build identity `ptr-runtime@0.1.0\|cpu-f32-sequential-reference\|linux-x86_64` | build identity is verified, not asserted: it reproduces the backend's chain id |
| corpus selection digests | main `sha256:d06fd633832bb4bdb7b50e21473dcde2882ff234855aa9e32da6a97f67a7968d`, nodes_only `sha256:38b1b1da0c23c6a6fea31aa53b52bb92bce2e82df3674f928d87387fc2251b63`, bytes_only `sha256:4f5a40e2b33245b3a99deda30331571c563a986438ea17d4ece9c592856465f9` | `params.rs` pins them; a test reproduces them |

Also chosen, in the harness only: matrix request shape (two bound 16-token segments + raw tail), in-flight request with a 24-token tail, race point uniform over the steps.

## 2. (b) Proposed `[preregistration]` table and rules entry

Flat values only (int, str, bool, int-list, str-list, file; no floats, printable ASCII). Floats become strings. Frozen as part of config.toml
(whole file is bound), plus the digest of `criteria`.

```toml
[experiment.R005.required]        # for experiments/preregistration.toml (not edited)
seeds = "int-list"
harness = "str"
max_new_tokens = "int"
intermediate_size = "int"
rope_theta = "str"
rms_norm_eps = "str"
qk_norm_main = "str"
qk_norm_transform_arm = "str"
ntk_factor = "str"
ntk_original_max_positions = "int"
over_capacity_prompt_tokens = "int"
over_capacity_fill_prompts = "int"
over_capacity_extra_prompts = "int"
saturation_max_sessions = "int"
saturation_excess_requests = "int"
corpus_selection_digest_main = "str"
corpus_selection_digest_nodes_only = "str"
corpus_selection_digest_bytes_only = "str"
semantic_liveness_model = "str"
criteria = "file"
static_evidence = "str-list"
```

```toml
[preregistration]                  # in config.toml (not edited)
seeds = [17, 29, 43, 71, 101]
harness = "exact-prefix-reuse"
max_new_tokens = 8
intermediate_size = 128
rope_theta = "10000.0"
rms_norm_eps = "1e-6"
qk_norm_main = "disabled"
qk_norm_transform_arm = "enabled"
ntk_factor = "4.0"
ntk_original_max_positions = 30
over_capacity_prompt_tokens = 128
over_capacity_fill_prompts = 512
over_capacity_extra_prompts = 64
saturation_max_sessions = 520
saturation_excess_requests = 8
corpus_selection_digest_main = "sha256:d06fd633832bb4bdb7b50e21473dcde2882ff234855aa9e32da6a97f67a7968d"
corpus_selection_digest_nodes_only = "sha256:38b1b1da0c23c6a6fea31aa53b52bb92bce2e82df3674f928d87387fc2251b63"
corpus_selection_digest_bytes_only = "sha256:4f5a40e2b33245b3a99deda30331571c563a986438ea17d4ece9c592856465f9"
semantic_liveness_model = "committed-state"   # see finding A; owner decision
criteria = "experiments/runtime/R005-exact-prefix-reuse/criteria.toml"
criteria_sha256 = "<digest at freeze>"
static_evidence = ["cargo test -p ptr-pods --locked --doc KvForkWitness", "cargo test -p ptr-pods --locked witness_tests::second_consume_of_the_same_witness_state_is_refused"]
```

No baseline directory: the baseline is the cold twin (full prefill on the CPU reference decoder), run beside every warm request in the same process.
Note: `config.toml` also needs the harness source frozen; the gate treats Rust files as provenance, so no extra key is proposed.
A decision script (`aggregate.py`, every seed must pass every rule) is still to be written; it is not part of this work.

### criteria.toml (proposed; generated from the harness's own coverage()/falsifiers() lists, so the rules are what the harness judges)

```toml
# R005 decision criteria (DRAFT, not frozen). One string per rule: "<counter> <op> <value>".
# A counter named metric.member is a member of an object-valued metric of the harness's JSON.
# Counts are exact integers; hit_rate is the one float and is compared as printed (>= 0.4).
version = 1
experiment = "R005"
seed_aggregation = "every_seed_must_pass"
decision = "completed iff every seed meets every coverage rule and every falsifier rule and each unsupported item is evidenced by its static command; failed otherwise; no rule changes after a run"

# A rule below its minimum makes the seed vacuous (counts as falsified).
coverage = [
  "revoked_fork_attempts >= 1",
  "evictions_nodes_only_run >= 1",
  "evictions_bytes_only_run >= 1",
  "evictions_with_active_fork_pinned >= 1",
  "invalidation_suppressed_arms >= 1",
  "restore_arms >= 1",
  "restored_pre_revocation_entries >= 1",
  "stale_eligible_entries_before_event >= 1",
  "post_event_lookups_on_stale_eligible_entries == stale_eligible_entries_before_event",
  "post_event_lookups >= 1",
  "input_revocations_performed >= 1",
  "post_revocation_lookups_matching_tokens >= 1",
  "same_tokens_different_generation_cases >= 1",
  "revoked_pinned_page_attempts >= 1",
  "cross_principal_attempts >= 1",
  "cross_project_attempts >= 1",
  "discarded_overlay_entry_attempts >= 1",
  "same_tokens_distinct_lineage_entries >= 2",
  "active_fork_revocation_attempts >= 1",
  "forged_witness_fork_attempts >= 1",
  "cancelled_run_effect_attempts >= 1",
  "cancelled_forks >= 1",
  "saturation_demand_cases >= 1",
  "capacity_exhausted_refusals >= 1",
  "raw_prefix_reuse_cases >= 1",
  "supersession_active_fork_race_cases >= 1",
  "memory_supersession_active_fork_race_cases >= 1",
  "memory_revocation_active_fork_race_cases >= 1",
  "selection_digest_mismatches <= 0",
  "procedure_mismatches <= 0",
  "capacity_refused_requests_completed == capacity_exhausted_refusals",
  "supersession_cancelled_forks >= 1",
  "member_cancelled_forks.semantic_revocation >= 1",
  "member_race_cases.semantic_revocation >= 1",
  "member_cancelled_forks.semantic_supersession >= 1",
  "member_race_cases.semantic_supersession >= 1",
  "member_cancelled_forks.memory_revocation >= 1",
  "member_race_cases.memory_revocation >= 1",
  "member_cancelled_forks.memory_supersession >= 1",
  "member_race_cases.memory_supersession >= 1",
  "cancelled_fork_id_set_digest == restarted_fork_id_set_digest",
  "capacity_refused_request_id_set_digest == capacity_completed_request_id_set_digest",
  "base_model_inflight_race_cases.revocation >= 1",
  "base_model_inflight_race_cases.supersession >= 1",
  "invalidation_matrix_cases.semantic_input_normal_revocation >= 1",
  "invalidation_matrix_cases.semantic_input_normal_supersession >= 1",
  "invalidation_matrix_cases.semantic_input_lag_revocation >= 1",
  "invalidation_matrix_cases.semantic_input_lag_supersession >= 1",
  "invalidation_matrix_cases.semantic_input_restore_revocation >= 1",
  "invalidation_matrix_cases.semantic_input_restore_supersession >= 1",
  "invalidation_matrix_cases.semantic_input_inflight_revocation >= 1",
  "invalidation_matrix_cases.semantic_input_inflight_supersession >= 1",
  "invalidation_matrix_cases.memory_input_normal_revocation >= 1",
  "invalidation_matrix_cases.memory_input_normal_supersession >= 1",
  "invalidation_matrix_cases.memory_input_lag_revocation >= 1",
  "invalidation_matrix_cases.memory_input_lag_supersession >= 1",
  "invalidation_matrix_cases.memory_input_restore_revocation >= 1",
  "invalidation_matrix_cases.memory_input_restore_supersession >= 1",
  "invalidation_matrix_cases.memory_input_inflight_revocation >= 1",
  "invalidation_matrix_cases.memory_input_inflight_supersession >= 1",
  "invalidation_matrix_cases.derived_input_normal_revocation >= 1",
  "invalidation_matrix_cases.derived_input_normal_supersession >= 1",
  "invalidation_matrix_cases.derived_input_lag_revocation >= 1",
  "invalidation_matrix_cases.derived_input_lag_supersession >= 1",
  "invalidation_matrix_cases.derived_input_restore_revocation >= 1",
  "invalidation_matrix_cases.derived_input_restore_supersession >= 1",
  "invalidation_matrix_cases.derived_input_inflight_revocation >= 1",
  "invalidation_matrix_cases.derived_input_inflight_supersession >= 1",
  "invalidation_matrix_cases.base_model_normal_revocation >= 1",
  "invalidation_matrix_cases.base_model_normal_supersession >= 1",
  "invalidation_matrix_cases.base_model_lag_revocation >= 1",
  "invalidation_matrix_cases.base_model_lag_supersession >= 1",
  "invalidation_matrix_cases.base_model_restore_revocation >= 1",
  "invalidation_matrix_cases.base_model_restore_supersession >= 1",
  "invalidation_matrix_cases.base_model_inflight_revocation >= 1",
  "invalidation_matrix_cases.base_model_inflight_supersession >= 1",
  "invalidation_matrix_cases.adapter_generation_normal_revocation >= 1",
  "invalidation_matrix_cases.adapter_generation_normal_supersession >= 1",
  "invalidation_matrix_cases.adapter_generation_lag_revocation >= 1",
  "invalidation_matrix_cases.adapter_generation_lag_supersession >= 1",
  "invalidation_matrix_cases.adapter_generation_restore_revocation >= 1",
  "invalidation_matrix_cases.adapter_generation_restore_supersession >= 1",
  "invalidation_matrix_cases.adapter_generation_inflight_revocation >= 1",
  "invalidation_matrix_cases.adapter_generation_inflight_supersession >= 1",
  "invalidation_matrix_cases.expert_generation_normal_revocation >= 1",
  "invalidation_matrix_cases.expert_generation_normal_supersession >= 1",
  "invalidation_matrix_cases.expert_generation_normal_fence >= 1",
  "invalidation_matrix_cases.expert_generation_lag_revocation >= 1",
  "invalidation_matrix_cases.expert_generation_lag_supersession >= 1",
  "invalidation_matrix_cases.expert_generation_lag_fence >= 1",
  "invalidation_matrix_cases.expert_generation_restore_revocation >= 1",
  "invalidation_matrix_cases.expert_generation_restore_supersession >= 1",
  "invalidation_matrix_cases.expert_generation_restore_fence >= 1",
  "invalidation_matrix_cases.expert_generation_inflight_revocation >= 1",
  "invalidation_matrix_cases.expert_generation_inflight_supersession >= 1",
  "invalidation_matrix_cases.expert_generation_inflight_fence >= 1",
  "superseded_lookup_cases.semantic_normal >= 1",
  "superseded_lookup_cases.semantic_lag >= 1",
  "superseded_lookup_cases.semantic_restore >= 1",
  "superseded_lookup_cases.memory_normal >= 1",
  "superseded_lookup_cases.memory_lag >= 1",
  "superseded_lookup_cases.memory_restore >= 1",
  "current_layer_kv_transform_change_cases >= 1",
  "stale_witness_replay_cases >= 1",
  "witness_reuse_static_evidence_passing >= 2",
  "model_pin_mismatches <= 0",
  "fixed_regime_decode_crossing_cases >= 1",
  "effective_regime_transition_cases >= 1",
  "mismatched_position_cases >= 1",
  "cross_environment_restore_cases >= 1",
]

# A tripped rule falsifies R005 on that seed.
falsifiers = [
  "bitwise_mismatch_count <= 0",
  "revoked_fork_count <= 0",
  ">= 0.4",
  "peak_trie_nodes_by_workload.main <= 4096",
  "peak_kv_page_bytes_by_workload.main <= 134217728",
  "peak_metadata_bytes_by_workload.main <= 8388608",
  "peak_trie_nodes_by_workload.nodes_only <= 4096",
  "peak_kv_page_bytes_by_workload.nodes_only <= 268435456",
  "peak_metadata_bytes_by_workload.nodes_only <= 67108864",
  "peak_trie_nodes_by_workload.bytes_only <= 16384",
  "peak_kv_page_bytes_by_workload.bytes_only <= 67108864",
  "peak_metadata_bytes_by_workload.bytes_only <= 16777216",
  "active_fork_page_evictions <= 0",
  "stale_hits <= 0",
  "revoked_lineage_entry_hits <= 0",
  "revoked_pinned_page_served_after_revocation <= 0",
  "revoked_pinned_page_bytes_reclaimed_after_release >= revoked_pinned_page_bytes_total",
  "cross_principal_hits <= 0",
  "cross_project_hits <= 0",
  "discarded_overlay_entry_hits <= 0",
  "unreachable_revoked_page_bytes_unreclaimed <= 0",
  "cross_lineage_hits <= 0",
  "cancelled_forks_restarted_from_fresh_prefill >= cancelled_forks",
  "outputs_escaped_after_revocation <= 0",
  "forged_witness_forks_accepted <= 0",
  "cancelled_run_effects_admitted <= 0",
  "raw_identical_prefix_hits >= 1",
  "outputs_after_supersession_semantic <= 0",
  "outputs_after_supersession_memory <= 0",
  "capacity_fallback_output_mismatches <= 0",
  "capacity_refusal_cache_inserts <= 0",
  "supersession_forks_restarted_from_fresh_prefill >= supersession_cancelled_forks",
  "cache_inserts_after_supersession <= 0",
  "base_model_inflight_outputs_after_invalidation.revocation <= 0",
  "base_model_inflight_cache_inserts.revocation <= 0",
  "base_model_inflight_outputs_after_invalidation.supersession <= 0",
  "base_model_inflight_cache_inserts.supersession <= 0",
  "invalidation_matrix_stale_hits.semantic_input_normal_revocation <= 0",
  "invalidation_matrix_stale_hits.semantic_input_normal_supersession <= 0",
  "invalidation_matrix_stale_hits.semantic_input_lag_revocation <= 0",
  "invalidation_matrix_stale_hits.semantic_input_lag_supersession <= 0",
  "invalidation_matrix_stale_hits.semantic_input_restore_revocation <= 0",
  "invalidation_matrix_stale_hits.semantic_input_restore_supersession <= 0",
  "invalidation_matrix_stale_hits.semantic_input_inflight_revocation <= 0",
  "invalidation_matrix_stale_hits.semantic_input_inflight_supersession <= 0",
  "invalidation_matrix_stale_hits.memory_input_normal_revocation <= 0",
  "invalidation_matrix_stale_hits.memory_input_normal_supersession <= 0",
  "invalidation_matrix_stale_hits.memory_input_lag_revocation <= 0",
  "invalidation_matrix_stale_hits.memory_input_lag_supersession <= 0",
  "invalidation_matrix_stale_hits.memory_input_restore_revocation <= 0",
  "invalidation_matrix_stale_hits.memory_input_restore_supersession <= 0",
  "invalidation_matrix_stale_hits.memory_input_inflight_revocation <= 0",
  "invalidation_matrix_stale_hits.memory_input_inflight_supersession <= 0",
  "invalidation_matrix_stale_hits.derived_input_normal_revocation <= 0",
  "invalidation_matrix_stale_hits.derived_input_normal_supersession <= 0",
  "invalidation_matrix_stale_hits.derived_input_lag_revocation <= 0",
  "invalidation_matrix_stale_hits.derived_input_lag_supersession <= 0",
  "invalidation_matrix_stale_hits.derived_input_restore_revocation <= 0",
  "invalidation_matrix_stale_hits.derived_input_restore_supersession <= 0",
  "invalidation_matrix_stale_hits.derived_input_inflight_revocation <= 0",
  "invalidation_matrix_stale_hits.derived_input_inflight_supersession <= 0",
  "invalidation_matrix_stale_hits.base_model_normal_revocation <= 0",
  "invalidation_matrix_stale_hits.base_model_normal_supersession <= 0",
  "invalidation_matrix_stale_hits.base_model_lag_revocation <= 0",
  "invalidation_matrix_stale_hits.base_model_lag_supersession <= 0",
  "invalidation_matrix_stale_hits.base_model_restore_revocation <= 0",
  "invalidation_matrix_stale_hits.base_model_restore_supersession <= 0",
  "invalidation_matrix_stale_hits.base_model_inflight_revocation <= 0",
  "invalidation_matrix_stale_hits.base_model_inflight_supersession <= 0",
  "invalidation_matrix_stale_hits.adapter_generation_normal_revocation <= 0",
  "invalidation_matrix_stale_hits.adapter_generation_normal_supersession <= 0",
  "invalidation_matrix_stale_hits.adapter_generation_lag_revocation <= 0",
  "invalidation_matrix_stale_hits.adapter_generation_lag_supersession <= 0",
  "invalidation_matrix_stale_hits.adapter_generation_restore_revocation <= 0",
  "invalidation_matrix_stale_hits.adapter_generation_restore_supersession <= 0",
  "invalidation_matrix_stale_hits.adapter_generation_inflight_revocation <= 0",
  "invalidation_matrix_stale_hits.adapter_generation_inflight_supersession <= 0",
  "invalidation_matrix_stale_hits.expert_generation_normal_revocation <= 0",
  "invalidation_matrix_stale_hits.expert_generation_normal_supersession <= 0",
  "invalidation_matrix_stale_hits.expert_generation_normal_fence <= 0",
  "invalidation_matrix_stale_hits.expert_generation_lag_revocation <= 0",
  "invalidation_matrix_stale_hits.expert_generation_lag_supersession <= 0",
  "invalidation_matrix_stale_hits.expert_generation_lag_fence <= 0",
  "invalidation_matrix_stale_hits.expert_generation_restore_revocation <= 0",
  "invalidation_matrix_stale_hits.expert_generation_restore_supersession <= 0",
  "invalidation_matrix_stale_hits.expert_generation_restore_fence <= 0",
  "invalidation_matrix_stale_hits.expert_generation_inflight_revocation <= 0",
  "invalidation_matrix_stale_hits.expert_generation_inflight_supersession <= 0",
  "invalidation_matrix_stale_hits.expert_generation_inflight_fence <= 0",
  "superseded_stale_hits.semantic_normal <= 0",
  "superseded_stale_hits.semantic_lag <= 0",
  "superseded_stale_hits.semantic_restore <= 0",
  "superseded_stale_hits.memory_normal <= 0",
  "superseded_stale_hits.memory_lag <= 0",
  "superseded_stale_hits.memory_restore <= 0",
  "stale_exact_hits <= 0",
  "stale_or_reused_witness_forks_admitted <= 0",
  "fixed_regime_crossing_parity_failures <= 0",
  "exact_outputs_across_effective_regime_transition_without_recompute <= 0",
  "exact_hits_with_mismatched_positions <= 0",
  "exact_hits_across_environment <= 0",
]

# Items the harness cannot produce; each needs the named command to pass at the frozen commit.
static_evidence = [
  "cargo test -p ptr-pods --locked --doc KvForkWitness",
  "cargo test -p ptr-pods --locked witness_tests::second_consume_of_the_same_witness_state_is_refused",
]
```

(52 `invalidation_matrix_cases.*` coverage rules and 52 `invalidation_matrix_stale_hits.*` falsifier rules are the config's cells; 6 `superseded_*` rules per config pair.)

## 3. (c) Entrypoint

`entrypoint = "cargo run --release --locked -p ptr-bench -- exact-prefix-reuse <seed>"` (only `<seed>`, so no extra table key is a placeholder). stdout is exactly one JSON object; exit 0 clean, 1 unmet/tripped/gap, 2 harness cannot run a case. Required artifacts `run.json`, `metrics.json` (numeric top-level fields are the metrics; objects hold per-workload/per-cell values).

## 4. (d) Unsupported (null with reason in the JSON)

- `witness_reuse_static_evidence_passing`: needs the doctest and unit test to run; the harness only reports (`witness_reuse_static_evidence_present`) that both items are in the source. Aggregator must run the two `static_evidence` commands.
- `base_model_inflight_outputs_after_invalidation` (both types): no inference step carries a base-model dependency (the backend records only prompt segments). The in-flight base-model race is observed on a bare catalog (pin cancelled, insert refused: `base_model_inflight_cache_inserts` = 0, race cases 1 per type) but no output exists to count.
- Escaped outputs for the 7 catalog-layer in-flight cells (base model, adapter, expert x types incl. fence): same reason (`unsupported["invalidation_matrix.<cell>.escaped_outputs"]`). Their stale hit is a still-live pin or an accepted insert after the event. Base model, adapter and expert cells (28 of 52) are catalog-layer in every path; semantic, memory, derived (24) are session-layer. Expert liveness uses a scripted oracle (the runtime reads it from a materialized expert record); base model/adapter use the ledger.
- `revoked_pinned_page_bytes_reclaimed_after_release.backend_resident_bytes`: the backend exposes no resident-byte accounting; reclamation and all `*_kv_page_bytes` figures are the catalog's charged bytes.
- Modelling choices, not unsupported but owner-visible: a cancelled run has no outcome, so "effect of a cancelled run" = `finish`, one more token, and `authorize_action` on the revoked generation at the current revision (control: admitted before the revocation); memory and derived inputs are lifecycle targets (a committed capsule / evidence id string) because the runtime keeps no other generation for them; the restore arm recomputes the same content-addressed pages by running the tokens as a raw segment so only the liveness check can refuse a restored entry (control: a live restored entry with pages forks, 32 tokens); peaks are sampled after every prefill chunk.

## 5. Findings (runtime code not touched)

**A. Semantic supersession by a committed edit is invisible to the use-time check** (`crates/ptr-runtime/src/kv_prefix.rs:1800-1824`, `impl KvLivenessOracle for PtrRuntime`). For a semantic key with no lifecycle generation it answers "live while the key is in the snapshot", and the doc comment says the content digest is "not re-verified"; doc 36 (P2 row) says admission validates identity, generation **and `content_digest`**. Consequences in the harness: after a committed edit of the key, (i) with the catalog's invalidation lost (lag) or after a restore (pages present) a request citing the old content hits the old entry (2 stale hits per path); (ii) even with the invalidation applied (normal), a request citing the old generation is admitted again (`KvPrefixCatalog::insert_paged` asks the same oracle) and re-inserts 2 entries (`stale_lineage_entries_reinserted.semantic_input_normal_supersession` = 2), and the next lookup hits one (1 stale hit); the re-inserted lineage keeps 32768 B charged (`unreachable_revoked_page_bytes_unreclaimed`). Revocation of a semantic key (`Revoked`) and every memory/derived/base/adapter/expert cell are caught. Repro: `cargo test -p ptr-bench --locked -- --ignored a_committed_edit_of_a_semantic_input` (fails today, passes once fixed), or the dry run. Control: `semantic_lifecycle_control_stale_hits` = 0 (key held as a lifecycle target).
Decision for the owner before freezing: (1) fix the oracle to compare the dependency's content digest with the committed value (and refuse a generation below the live one), then re-dry-run, or (2) redefine "semantic generation" as lifecycle-backed (`semantic_liveness_model = "lifecycle"` would need a harness switch; not added, to avoid choosing after seeing results).

**B. Config arithmetic.** README/doc 36 say each workload's metadata limit is node limit x 1 KiB. That holds only for bytes_only (16384 x 1 KiB = 16 MiB); main (8 MiB) and nodes_only (64 MiB) are 2x and 16x 4096 x 1 KiB (doc 36 itself says 4096 nodes give 4 MiB). Observed peak metadata is 0.46 MB main and 2.3 MB at 4096 nodes, so no falsifier is near; fix the text or the limits before freezing.

**C. Observation (policy, not a violation).** `KvPrefixCatalog::admit` loops `evict_lru()` (`kv_prefix.rs:1004-1009`, `:1058`) whenever a bound is hit, and evicts unpinned *ancestor* entries whose removal frees no node or page (their descendants keep the node). Under saturation a one-node request therefore drains every unpinned shallow entry (4096 evictions per workload, `evictions_with_active_fork_pinned` = 8192 total) before `CapacityExhausted`. Pinned entries are never touched (`active_fork_page_evictions` = 0).

## 6. (e) Dry run

`cargo run --release --locked -p ptr-bench -- exact-prefix-reuse 1` (not saved under experiments/): wall **111 s** (112-114 s for the five declared seeds). Phases: main workload 21 s, nodes-only 45 s, bytes-only 44 s, matrix 1.3 s, isolated cases 0.4 s. Seed 1 is not a declared seed (stderr says so). Per-seed headline values (identical on all seeds): hit_rate 0.4921875 (8064 of 16384 tokens), 158 bitwise comparisons 0 mismatches, 512 evictions per over-capacity unique-prompt phase, 512 pinned forks and 8 refused-then-completed requests per workload, 8 cancelled forks all restarted fresh, 38 stale-eligible entries all looked up, 52/52 cells run. Tripped falsifiers (all finding A): `stale_hits` (4), `unreachable_revoked_page_bytes_unreclaimed` (32768), `invalidation_matrix_stale_hits` and `superseded_stale_hits` for semantic supersession normal/lag/restore (1/2/2).

Build and tests: `cargo test -p ptr-bench --locked` passes (the reduced-scale end-to-end smoke takes about 50 s of the 57 s in a debug build; mark it `#[ignore]` if too slow for CI). Reproduce a seed: run the entrypoint above and read `summary`, `coverage`, `falsifiers`, `unsupported`.
