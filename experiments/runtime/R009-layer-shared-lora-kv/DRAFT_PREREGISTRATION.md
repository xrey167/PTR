# R009 layer-shared-lora-kv: DRAFT preregistration

Status: DRAFT. R009 stays `planned`; nothing here is frozen and nothing is a result. This text, the
`criteria.toml` block and the `[preregistration]` table below are proposals for the freeze. The harness is
`bins/ptr-bench/src/experiments/{layered_kv,r009}/` (subcommand `layer-shared-lora-kv <seed>`).

Entrypoint (proposed value of `experiment.toml`'s `entrypoint`, and of the runner's command):
`cargo run --release --locked -p ptr-bench -- layer-shared-lora-kv <seed>`

Scope: KV-memory sharing only (`claim_scope = "kv_memory_only"`): no prefill-FLOPs claim. The code under test is
the CPU reference path: `ReferenceKvReuseBackend` and its bound `KvSession`s over the tiny seeded Qwen2 stack
with seeded LoRA/aLoRA adapters (`Qwen2Adapter`), per-layer catalog entries, layer-scoped fork and checked
adoption, and `PtrRuntime` as the ledger-anchored liveness oracle.

## 1. Decision rule

For each of the seeds 17, 29, 43, 71, 101 the harness prints one JSON object (stdout; exit 0 clean, 1 unmet
or tripped or a gap, 2 a case it cannot run, including a vacuous adapter: B == 0 or no change of the logits).
`aggregate.py` (to be written at the freeze from R005's, same grammar) reads the five records and applies
`criteria.toml` mechanically: COMPLETED iff every seed meets every coverage rule and every falsifier rule;
FALSIFIED if any rule fails on any seed; INVALID if a seed has no finished record, describes another run, or a
rule cannot be resolved (missing, null or mistyped counter). Every seed must pass; no tolerance, no majority.

## 2. Configurations run (the pinned set)

| configuration | adapter type | first layer f | predicted shared rows (of 4 x 64 = 256) |
|---|---|---|---|
| qkv_lora_f2 | qkv_lora | 2 (= L/2, the required interior one) | 2 x 64 = 128 (f/L = 1/2) |
| qkv_lora_f0 | qkv_lora | 0 | 0 (0/L) |
| q_only_lora_f2 | q_only_lora | 2 | 3 x 64 = 192 ((f+1)/L = 3/4) |
| q_only_lora_f0 | q_only_lora | 0 | 1 x 64 = 64 (1/4) |
| mlp_only_lora_f2 | mlp_only_lora | 2 | 192 (3/4) |
| alora_inv39 | alora | 0 (QKV at every layer) | 4 x 39 = 156 (all full pre-invocation pages plus the 7 pre-invocation rows of the boundary page, in every layer) |

`shared_row_fraction.<configuration>` is an object of integers (numerator, denominator, predicted numerator,
predicted denominator, cross difference); the falsifier is `cross_difference == 0` per configuration, and the
per-layer rows are checked too (`internal.per_layer_fraction_mismatches`).

## 3. Pins the config left open, with reasons

| value | pinned | reason / how verified |
|---|---|---|
| max_new_tokens | 8 | backend default (`ReferenceKvLimits::default`) |
| intermediate_size / rope_theta / rms_norm_eps | 128 / 10000.0 / 1e-6 | inherited from R005 (hard-coded in `kv_infer.rs::stack_config`); `layered_kv::pins` rebuilds the stack from them and reproduces the backend's chain id and cold run |
| QK-norm | disabled | the backend cannot enable it |
| catalog limits | `ReferenceKvLimits::default` | R009 pins none; recorded in `procedure.chosen` |
| LoRA rank / scale | 4 / 2.0 | owner decision 5 (`QWEN2_ADAPTER_PINNED_RANK/SCALE`, asserted equal by a test) |
| adapter seed | `weight_seed` (20240607), mixed with (adapter id, generation) | owner decision 5; B != 0 is asserted (`b_is_nonzero`) and the adapter must change the logits, else exit 2 |
| adapter attachment | every layer >= first adapted layer | owner decision 5 |
| aLoRA | QKV (q, k, v) at all layers, active from position 39 (page offset 7) | owner decision 5 / config |
| MLP adapter form | residual-space correction of the MLP output: y = mlp(x) + s B (A x) | design choice accepted by the head |
| partial-layer hits | compute-then-adopt in checked mode (every row compared bitwise before the shared allocation is swapped in); a full hit forks the assembled layer prefixes. Peak memory is transiently higher than a true partial computation | owner decision 2 |
| restart after invalidation | re-resolves to the live binding: supersession to the new generation, revocation to the base model; judged against a full prefill of that same binding | owner decision 3 |
| total_kv_rows | layers x prompt tokens (64 x 4 = 256); generated tokens' rows not counted | owner decision 5 |
| per-seed role | the seed orders the six configurations and places the in-flight race points (seeded SplitMix64); the corpus is fixed by `corpus_generator_seed` | owner decision 5 |
| corpus selection digest | `sha256:8093b0b8b9859c83a0a145a993e2158be39c31e2b80812a4f7da4ab0a6bb117e` (domain `ptr.r009.corpus-selection.v1`, 38 sequences) | `params.rs` pins it; a test reproduces it |
| matrix binding | a QKV-LoRA `lora-m` first attached at layer 2, generation 2 as successor | the 40 cells run on bound sessions of it |
| isolation adapters | `lora-b` (same weight bits as `lora-a`, another identity), `lora-g` generations 1 and 2 (own seeds) | cross-adapter and old-generation attempts |
| aLoRA request shape | base request = first 48 tokens of the prompt, adapter request = those 48 + 16 others | a base prefix shorter than the prompt can be forked and truncated at the invocation; see section 6 |
| model pins | tiny_model_config digest, tokenizer, shape, chain id, cold run, and a cold run under an adapter of each kind, all reproduced by an independent stack | `layered_kv::pins` |

## 4. Owner decisions of 2026-10-06 (binding, verbatim)

1. aLoRA boundary rows: IMPLEMENT physical row sharing (page segment = shared row-prefix Arc + owned tail). Prediction unchanged.
2. Partial-layer hits: compute-then-adopt (checked mode), stated in the preregistration.
3. Restart after revocation: re-resolve to the LIVE binding (supersession -> new generation; revocation -> base model for R009, router with the expert masked for R010); parity vs full prefill of that same binding.
4. Expert fence: implement the REAL authority path (ledger event -> Fenced in state -> catalog invalidation) (WP5).
5. Pins as proposed in section "Owner decisions" item 5; integer numerators/denominators for fraction equality.

Item 5 of the plan, the pins: LoRA rank 4, scale 2.0, A and B seeded from (weight_seed, adapter id, generation), B != 0; the adapter is attached at every layer >= f; aLoRA adapts QKV at all layers, active from token 39; R010 MoE in all 4 layers, expert intermediate 64, top-2 without renormalisation, lowest index wins ties, i = the most-routed expert of layer 1 on the probe prompt (record it and require it to be routed at least once); batch-variant control: capacity 1 token per expert per batch of 2 requests; intermediate 128, theta 10000 and eps 1e-6 inherited from R005; per-seed role: request order and race points; corpus selection digests; `total_kv_rows` counts prompt rows only (64 x 4); a new metric `post_invocation_rows_under_base_key`, max 0. Fraction equality with tolerance 0.0: record integer numerators and denominators and compare them as integers, never as floats.

## 5. Design choices accepted by the head

- A masked expert stays in the softmax denominator with selection score -inf (R010; recorded here for the shared substrate).
- The MlpOnly adapter is a residual-space correction of the MLP output.
- Partial hits are computed in full and then adopted (compute-then-adopt, checked mode).
- Restart re-resolves to the live binding (owner decision 3).

## 6. Modelling choices, and how each metric is measured

- `shared_kv_rows` is physical: rows of each page layer that are the same allocation in the adapter request's cache
  and the base request's cache (`KvSession::shared_leading_rows`), summed over layers and pages. It is checked against
  the backend's own report (`internal.report_physical_mismatches`) and against a bitwise comparison of the K and V rows
  of two independent prefills (`bitwise_mismatch_count`), and the layer from which K/V differ must be the predicted one
  (`internal.prediction_structure_mismatches`).
- `adapter_logits_bitwise_mismatch_count`: the adapter run (reuse path) against a cold run of the same binding.
- `base_pages_before_revoke`: the entries (one per layer and 16-token chunk) of the base run. `base_pages_dropped_on_revoke`: those
  entries minus what a base probe of the same prompt (+8 tokens, no new page boundary) still hits after the adapter's
  revocation. `adapter_entries_before_revocation`: entries the adapter run inserted (checked against the catalog count).
- `cross_adapter_*`: adapter B (A's weight bits, another identity) on A's prompt may share only the base layers (hits =
  shared rows above the prediction). `old_generation_*`: generation 2 of `lora-g` runs while generation 1's entries are
  still in the index (the supersession is committed to the ledger with the catalog update dropped), so it can miss them only
  through its chain key (`internal.generation_chain_isolation_rejections` stays 0); presenting generation 1 as current is then refused.
- In-flight: a session forks the adapter's entries, the adapter is revoked at a seeded step, the session is cancelled at
  its next step and restarted (re-resolved to the base model), compared with a cold run of the base binding.
- Matrix: every one of the 40 cells runs on bound sessions of the matrix binding; see the module documentation of
  `layered_kv/matrix.rs`. Segment kinds (semantic, memory, derived) get real post-event lookups; binding-carried kinds (base
  model, adapter) are refused at `begin_session_bound` before any lookup (see the unsupported list).
- Lag arm: the index update is dropped by the fault hook (`IndexUpdate::Suppressed`) while the ledger commits the event.
  Restore arm: a pre-event snapshot is restored into a fresh backend whose pages are recomputed by running the same tokens
  as a raw segment; a live control must fork (`internal.restore_control_failures`).
- Selection: `selection_digest_mismatches` compares the computed digest of all sequences the run draws with the pinned one.

## 7. Unsupported by the harness, and limits (null with a reason in the JSON)

- `base_model_inflight_restart`: no live base model exists to re-resolve to: the in-flight base-model cells check cancellation, silence and no insert, and do not restart (a restart would compute from the revoked base model)
- `binding_carried_post_event_lookups`: a request on a binding with a dead dependency (base model, adapter) is refused by begin_session_bound before any catalog lookup, so post-event lookups on stale-eligible entries (post_event_lookups*, restored/stale-eligible counters) are counted for the segment cells only; the use-time rejection of an entry that depends on such a dependency is exercised in flight (a session cancelled at its next step) and by the adapter-revocation probes in the configuration arms, which are begin refusals
- `prefill_flops_saving`: the claim is a KV-memory saving only (claim_scope kv_memory_only): partial hits are computed in full and adopted in checked mode, so no forward-compute saving exists to measure

Also, as measured limitations visible to the reader (not unsupported): with the whole prompt already cached under the base key an
aLoRA request shares nothing (`internal.alora_full_prompt_cached_shared_rows`), because a deeper base-keyed hit is refused for adoption
(`depth <= activation`) and a hit of the whole prompt cannot be forked; the pinned shape therefore uses a base request shorter than the prompt.

## 8. Proposed `[preregistration]` table of `config.toml` and `[experiment.R009.required]` of `preregistration.toml`

```toml
[preregistration]
seeds = [17, 29, 43, 71, 101]
harness = "layer-shared-lora-kv"
max_new_tokens = 8
intermediate_size = 128
rope_theta = "10000.0"
rms_norm_eps = "1e-6"
qk_norm = "disabled"
lora_rank = 4
lora_scale = "2.0"
adapter_seed = 20240607
adapter_layers = "every layer >= first adapted layer"
alora_first_layer = 0
alora_projections = "q, k, v"
mlp_adapter_form = "residual-space correction of the MLP output"
partial_hit_mode = "compute-then-adopt, checked"
restart_resolution = "re-resolve to the live binding: supersession to the new generation, revocation to the base model"
total_kv_rows_counts = "prompt rows of every layer"
per_seed_role = "request order and race points"
matrix_adapter_first_layer = 2
corpus_selection_digest = "sha256:8093b0b8b9859c83a0a145a993e2158be39c31e2b80812a4f7da4ab0a6bb117e"
preregistration_text = "experiments/runtime/R009-layer-shared-lora-kv/PREREGISTRATION.md"
criteria = "experiments/runtime/R009-layer-shared-lora-kv/criteria.toml"
# preregistration_text_sha256 and criteria_sha256 are written at the freeze.
```

```toml
[experiment.R009.required]
seeds = "int-list"
harness = "str"
max_new_tokens = "int"
intermediate_size = "int"
rope_theta = "str"
rms_norm_eps = "str"
qk_norm = "str"
lora_rank = "int"
lora_scale = "str"
adapter_seed = "int"
adapter_layers = "str"
alora_first_layer = "int"
alora_projections = "str"
mlp_adapter_form = "str"
partial_hit_mode = "str"
restart_resolution = "str"
total_kv_rows_counts = "str"
per_seed_role = "str"
matrix_adapter_first_layer = "int"
corpus_selection_digest = "str"
preregistration_text = "file"
criteria = "file"
```

Also proposed for `config.toml` `[pinned_thresholds]`: `max_post_invocation_rows_under_base_key = 0` (owner decision 5; the harness reports
`internal.post_invocation_rows_under_base_key` and judges it `<= 0`; it is not in the manifest's metric list yet, so the manifest
should name it before the freeze).

## 9. Proposed `criteria.toml`

```toml
# R009 decision criteria (DRAFT). They are what the harness itself judges (bins/ptr-bench,
# r009/metrics.rs coverage() and falsifiers()); a harness test pins this block to those lists, in order.
#
# Rule grammar (one string per rule, exactly three tokens separated by single spaces):
#     <left> <op> <right>
# <op> is one of  >=  <=  == .
# A token is an integer literal iff it matches -?[0-9]+ (a decimal point makes it a reference and so a
# parse error: no decimal literal exists, fractions are integer numerators and denominators).
# Any other token is a reference: a path of object keys from the root of the harness's single JSON result
# object, separated by '.' (no key holds a '.', no array is indexed): "stale_hits" is a top-level key,
# "invalidation_matrix_cases.expert_generation_normal_fence" a member of an object-valued key,
# "shared_row_fraction.qkv_lora_f2.cross_difference" and "internal.restore_control_failures" deeper ones.
# A reference that is missing, null, or of the wrong type makes the seed INVALID (it is not a failed rule).
# A reference compared with an integer literal or another reference must be a JSON integer (not a
# boolean, not a float); two references under == may both be JSON strings (digests). Exact comparisons.
# Fractions: shared_row_fraction.<config> is {numerator, denominator, predicted_numerator,
# predicted_denominator, cross_difference} with cross_difference = numerator*predicted_denominator -
# predicted_numerator*denominator; "equal to the prediction with tolerance 0.0" is cross_difference == 0.
# coverage: a rule that does not hold makes the seed vacuous (counts as falsified).
# falsifiers: a rule that does not hold falsifies R009 on that seed.
# Decision: every seed must hold every coverage rule and every falsifier rule; COMPLETED iff so,
# INVALID if a seed is missing, unresolvable, or its record does not describe the frozen run, FALSIFIED
# otherwise.
version = 1
experiment = "R009"
seed_aggregation = "every_seed_must_pass"

coverage = [
  "internal.config_runs.qkv_lora_f2 >= 1",
  "internal.config_runs.qkv_lora_f0 >= 1",
  "internal.config_runs.q_only_lora_f2 >= 1",
  "internal.config_runs.q_only_lora_f0 >= 1",
  "internal.config_runs.mlp_only_lora_f2 >= 1",
  "internal.config_runs.alora_inv39 >= 1",
  "alora_prefix_reuse_attempts >= 1",
  "alora_boundary_page_cases >= 1",
  "adapter_revocations_performed_by_type.qkv_lora >= 1",
  "adapter_reuse_attempts_by_type.qkv_lora >= 1",
  "adapter_revocations_performed_by_type.q_only_lora >= 1",
  "adapter_reuse_attempts_by_type.q_only_lora >= 1",
  "adapter_revocations_performed_by_type.mlp_only_lora >= 1",
  "adapter_reuse_attempts_by_type.mlp_only_lora >= 1",
  "adapter_revocations_performed_by_type.alora >= 1",
  "adapter_reuse_attempts_by_type.alora >= 1",
  "post_revocation_lookups_on_adapter_entries >= 1",
  "base_pages_before_revoke >= 1",
  "adapter_entries_before_revocation >= 1",
  "cross_adapter_attempts >= 1",
  "old_generation_as_new_attempts >= 1",
  "input_revocations_performed >= 1",
  "post_revocation_lookups_matching_tokens >= 1",
  "same_tokens_different_generation_cases >= 1",
  "invalidation_suppressed_arms >= 1",
  "restore_arms >= 1",
  "internal.restore_control_failures <= 0",
  "restored_pre_revocation_entries >= 1",
  "stale_eligible_entries_before_event >= 1",
  "post_event_lookups_on_stale_eligible_entries == stale_eligible_entries_before_event",
  "post_event_lookups >= 1",
  "inflight_adapter_revocation_race_cases.revocation >= 1",
  "inflight_input_race_cases.revocation >= 1",
  "inflight_adapter_revocation_race_cases.supersession >= 1",
  "inflight_input_race_cases.supersession >= 1",
  "selection_digest_mismatches <= 0",
  "procedure_mismatches <= 0",
  "inflight_adapter_requests_restarted_from_fresh_prefill >= inflight_adapter_requests_cancelled",
  "inflight_adapter_cancelled_request_id_set_digest == inflight_adapter_restarted_request_id_set_digest",
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
  "model_pin_mismatches <= 0",
]

falsifiers = [
  "bitwise_mismatch_count <= 0",
  "adapter_logits_bitwise_mismatch_count <= 0",
  "shared_row_fraction.qkv_lora_f2.cross_difference == 0",
  "shared_row_fraction.qkv_lora_f0.cross_difference == 0",
  "shared_row_fraction.q_only_lora_f2.cross_difference == 0",
  "shared_row_fraction.q_only_lora_f0.cross_difference == 0",
  "shared_row_fraction.mlp_only_lora_f2.cross_difference == 0",
  "shared_row_fraction.alora_inv39.cross_difference == 0",
  "internal.per_layer_fraction_mismatches <= 0",
  "internal.prediction_structure_mismatches <= 0",
  "internal.report_physical_mismatches <= 0",
  "internal.adoption_mismatches <= 0",
  "base_pages_dropped_on_revoke <= 0",
  "revoked_adapter_entry_hits <= 0",
  "internal.post_invocation_rows_under_base_key <= 0",
  "cross_adapter_hits <= 0",
  "old_generation_hits <= 0",
  "revoked_lineage_entry_hits <= 0",
  "stale_hits <= 0",
  "outputs_after_adapter_revocation <= 0",
  "outputs_after_input_invalidation <= 0",
  "adapter_restart_output_mismatches <= 0",
  "internal.restart_resolution_mismatches <= 0",
  "internal.matrix_selectivity_mismatches <= 0",
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
]
```
