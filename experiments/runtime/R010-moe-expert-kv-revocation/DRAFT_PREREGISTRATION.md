# R010 moe-expert-kv-revocation: DRAFT preregistration

Status: DRAFT. R010 stays `planned`; nothing here is frozen and nothing is a result. This text, the
`criteria.toml` block and the `[preregistration]` table below are proposals for the freeze. The harness is
`bins/ptr-bench/src/experiments/{layered_kv,r010}/` (subcommand `moe-expert-kv-revocation <seed>`).

Entrypoint (proposed value of `experiment.toml`'s `entrypoint`, and of the runner's command):
`cargo run --release --locked -p ptr-bench -- moe-expert-kv-revocation <seed>`

Scope: KV-memory property only (`claim_scope = "kv_memory_only"`). The code under test is the CPU reference path: the
bound `KvSession`s of `ReferenceKvReuseBackend` over the tiny seeded Qwen2 stack with a seeded tiny MoE (`Qwen2Moe`), per-layer
entries whose dependency sets carry the experts of the layers below, and `PtrRuntime` as the liveness oracle with **the real expert
admission and fence authority path** (birth, retire, reactivate: `admit_expert_proposal`; fence and readmission: `fence_expert`,
`readmit_expert`; the registry's materialised `expert:{project}/{pod}` record answers liveness; nothing is scripted).

## 1. Decision rule

Same as R009: five seeds, one JSON object each, `criteria.toml` applied mechanically, COMPLETED / FALSIFIED / INVALID, every seed must
pass. Exit 2 when the revoked expert is never routed to on the probe prompt, or masking it does not change the logits (a vacuous run).

## 2. Scenario

- Model: 4 layers, MoE in every layer, 4 experts, top-2, expert intermediate 64, dropless; j = 1, so layers 2 and 3 are above j.
- `i` = the most-routed expert of layer 1 on the probe prompt (`RoutingCounts::most_routed`), recomputed each run and compared with the pinned
  value (a difference is a `procedure_mismatches`); the run requires it routed at least once (`internal.revoked_expert_routed_tokens >= 1`).
- A binding tracks every live expert of every MoE layer (16 experts), so a layer l entry depends on every expert of layers < l; revoking `(1, i)`
  therefore ends exactly the entries of layers 2 and 3 (8 of 16 pages of the 64-token prompt; layers 0 and 1 are retained).
- Deterministic arm: warm (16 entries: 8 above j, 8 at or below), a reuse event (prompt + 8 tokens forks 64), the nondeterministic control,
  the committed `Retire` of `(1, i)` with the catalog tombstoning, then three probes: the masked router (restart's binding) on the extended prompt
  (pages at or below j reused, none above j served), a request on the revoked binding (refused), and a second masked request that is served the pages
  recomputed after the revocation. Physical sharing is read with `KvSession::shared_leading_rows` against the pre-revocation run.
- Nondeterministic control: the same MoE with capacity-limited routing (capacity 1 per expert per step, 2 co-batched requests) is never tagged Exact
  (an ExactOnly attempt is refused, an AllowApproximate run is tagged Approximate and moves nothing in the Exact catalog), and its outputs diverge with the
  co-batch order (the dropless control does not).
- Quarantine arm (fence): Exact KV populated, `fence_expert(AuthorityQuarantine)` under the revocation authority's grant, entries tombstoned, a matching
  request refused / not served from before the fence, `readmit_expert` under the gate authority with a passed bound report, the old-epoch binding still dead
  and the new-epoch binding computing its own pages; plus an `InputRevoked` fence (naming a committed revocation) that leaves the expert not live.
- Matrix: 44 cells (the expert dependency also has `fence`) on bound sessions of the MoE binding.

## 3. Pins the config left open, with reasons

| value | pinned | reason / how verified |
|---|---|---|
| max_new_tokens | 8 | backend default |
| prompt tokens / page tokens | 64 / 16 | R005 chunk; 4 pages per layer |
| intermediate_size / rope_theta / rms_norm_eps / QK-norm | 128 / 10000.0 / 1e-6 / disabled | inherited from R005; reproduced by `layered_kv::pins` |
| MoE seed | `weight_seed` (20240607) in the MoE's own domain (`ptr.qwen2.moe`) | dense weights unchanged |
| MoE layers / experts / top-k / expert intermediate | every layer / 4 / 2 / 64 | owner decision 5 |
| top-k normalisation / ties | none (selected probabilities as is) / lowest index wins | owner decision 5 |
| masked expert | stays in the softmax denominator with selection score -inf | design choice accepted by the head |
| revoked expert index i | 0 (the most-routed expert of layer 1 on the probe prompt) | recomputed and compared each run |
| batch-variant control | capacity 1 token per expert per step, batch of 2 requests | owner decision 5 |
| expert tracking | every live expert of every MoE layer | WP4 review (`UntrackedLiveExpert`) |
| fence epoch recorded by a binding | 0 in the revocation, supersession and matrix cells; the registry's state version of each expert in the quarantine arm | an epoch-0 dependency ignores the state version; see section 7 |
| partial-layer hits | compute-then-adopt, checked | owner decision 2 |
| restart after invalidation | re-resolves to the live binding: revoked or fenced expert masked, superseded expert to its successor generation | owner decision 3 |
| per-seed role | the seed orders the three arms and places the in-flight race points | owner decision 5 |
| corpus selection digest | `sha256:55a21badbe657d7bbee30769013d801c4f53136f6ea504e69cbefd3329652a05` (domain `ptr.r010.corpus-selection.v1`, 24 sequences) | `params.rs` pins it; a test reproduces it |
| model pins | tiny_model_config digest (with num_experts, top_k), tokenizer, shape, chain id, cold run, the MoE topology of the run's own MoE and a cold MoE run reproduced by an independent stack | `layered_kv::pins` |

## 4. Owner decisions of 2026-10-06 (binding, verbatim)

1. aLoRA boundary rows: IMPLEMENT physical row sharing (page segment = shared row-prefix Arc + owned tail). Prediction unchanged.
2. Partial-layer hits: compute-then-adopt (checked mode), stated in the preregistration.
3. Restart after revocation: re-resolve to the LIVE binding (supersession -> new generation; revocation -> base model for R009, router with the expert masked for R010); parity vs full prefill of that same binding.
4. Expert fence: implement the REAL authority path (ledger event -> Fenced in state -> catalog invalidation) (WP5).
5. Pins as proposed in section "Owner decisions" item 5; integer numerators/denominators for fraction equality.

Item 5 of the plan, the pins: LoRA rank 4, scale 2.0, A and B seeded from (weight_seed, adapter id, generation), B != 0; the adapter is attached at every layer >= f; aLoRA adapts QKV at all layers, active from token 39; R010 MoE in all 4 layers, expert intermediate 64, top-2 without renormalisation, lowest index wins ties, i = the most-routed expert of layer 1 on the probe prompt (record it and require it to be routed at least once); batch-variant control: capacity 1 token per expert per batch of 2 requests; intermediate 128, theta 10000 and eps 1e-6 inherited from R005; per-seed role: request order and race points; corpus selection digests; `total_kv_rows` counts prompt rows only (64 x 4); a new metric `post_invocation_rows_under_base_key`, max 0. Fraction equality with tolerance 0.0: record integer numerators and denominators and compare them as integers, never as floats.

## 5. Design choices accepted by the head

- A masked expert stays in the softmax denominator with selection score -inf.
- The MlpOnly adapter is a residual-space correction of the MLP output (R009).
- Partial hits are computed in full and then adopted (compute-then-adopt, checked mode).
- Restart re-resolves to the live binding (owner decision 3).

## 6. Modelling choices, and how each metric is measured

- A page is one layer's page of one 16-token chunk (one catalog entry): `pages_above_j_before_revocation` = 2 x 4 = 8, `pages_at_or_below_j_before_revocation` = 2 x 4 = 8.
- `stale_pages_served`: page layers above j of the masked-router probe sharing any row with the pre-revocation cache. `wrongly_dropped_pages`: pages at or below j minus those the probe
  shares as whole pages. `retained_pages_reused_after_revocation`: those whole pages. `recomputed_pages_above_j_served`: whole pages above j that the follow-up request shares with the probe's recomputed pages.
- `probed_page_set_digest` / `cached_page_set_digest_above_j`: digests of the sets of (layer, chunk) pages above j the probe looks up and the warm run cached; both derive from the request geometry and the
  run's own insert count (the backend exposes no entry listing); equality is identity-set equality, not count equality.
- `nondeterministic_divergence_observed`: two requests co-batched in both orders under capacity-limited routing differ in some last-token logit bit.
- `bitwise_mismatch_count` and `mismatching_runs_tagged_exact`: every reused or restarted run against a cold run of the same binding.
- Fence cells: the expert is fenced by the real `fence_expert` (AuthorityQuarantine); the harness then tombstones the entries with `invalidate_inputs` on the expert's identity because the backend's catalog is
  not the runtime's hooked catalog (the runtime's own hook is unit-tested in `crates/ptr-runtime/tests/expert_fence.rs`). A fenced expert is not live for the oracle, so a request citing it is refused at begin.
- Lag and restore arms as in R009 (section 6 there), including the live restore control.

## 7. Unsupported by the harness, and limits

- `base_model_inflight_restart`: no live base model exists to re-resolve to: the in-flight base-model cells check cancellation, silence and no insert, and do not restart (a restart would compute from the revoked base model)
- `binding_carried_post_event_lookups`: a request on a binding with a dead dependency (base model, expert) is refused by begin_session_bound before any catalog lookup, so post-event lookups on stale-eligible entries (post_event_lookups*, restored/stale-eligible counters) are counted for the segment cells only; the use-time rejection of an entry that depends on such a dependency is exercised in flight (a session cancelled at its next step) and by the post-revocation probes of the deterministic arm, which are begin refusals and masked-router lookups under other chains
- `fence_catalog_hook`: the backend's catalog is not the runtime's hooked catalog: a fence commits through the real authority path (PtrRuntime::fence_expert) and the harness then tombstones the entries by the expert's identity with invalidate_inputs; the runtime's own hook is unit-tested in crates/ptr-runtime/tests/expert_fence.rs

Also: a successor generation (`ExpertSuccessor`) carries no fence epoch, so a binding that recorded a non-zero epoch for the superseded expert cannot re-resolve to the successor (the epoch is checked against the
registry's state version, which the retire and reactivate moved); the supersession cells therefore use epoch-0 bindings.

## 8. Proposed `[preregistration]` table of `config.toml` and `[experiment.R010.required]` of `preregistration.toml`

```toml
[preregistration]
seeds = [17, 29, 43, 71, 101]
harness = "moe-expert-kv-revocation"
max_new_tokens = 8
prompt_tokens = 64
page_tokens = 16
intermediate_size = 128
rope_theta = "10000.0"
rms_norm_eps = "1e-6"
qk_norm = "disabled"
moe_seed = 20240607
moe_layers = "every decoder layer"
expert_intermediate_size = 64
top_k_normalisation = "none: the selected probabilities are used as is"
tie_break = "lowest expert index wins"
masked_expert = "stays in the softmax denominator with selection score -inf"
revoked_expert_index = 0
batch_variant_capacity = 1
cobatch_requests = 2
expert_tracking = "every live expert of every MoE layer"
partial_hit_mode = "compute-then-adopt, checked"
restart_resolution = "re-resolve to the live binding: revoked or fenced expert masked, superseded expert to its successor generation"
per_seed_role = "request order and race points"
corpus_selection_digest = "sha256:55a21badbe657d7bbee30769013d801c4f53136f6ea504e69cbefd3329652a05"
preregistration_text = "experiments/runtime/R010-moe-expert-kv-revocation/PREREGISTRATION.md"
criteria = "experiments/runtime/R010-moe-expert-kv-revocation/criteria.toml"
# preregistration_text_sha256 and criteria_sha256 are written at the freeze.
```

```toml
[experiment.R010.required]
seeds = "int-list"
harness = "str"
max_new_tokens = "int"
prompt_tokens = "int"
page_tokens = "int"
intermediate_size = "int"
rope_theta = "str"
rms_norm_eps = "str"
qk_norm = "str"
moe_seed = "int"
moe_layers = "str"
expert_intermediate_size = "int"
top_k_normalisation = "str"
tie_break = "str"
masked_expert = "str"
revoked_expert_index = "int"
batch_variant_capacity = "int"
cobatch_requests = "int"
expert_tracking = "str"
partial_hit_mode = "str"
restart_resolution = "str"
per_seed_role = "str"
corpus_selection_digest = "str"
preregistration_text = "file"
criteria = "file"
```

## 9. Proposed `criteria.toml`

```toml
# R010 decision criteria (DRAFT). They are what the harness itself judges (bins/ptr-bench,
# r010/metrics.rs coverage() and falsifiers()); a harness test pins this block to those lists, in order.
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
# falsifiers: a rule that does not hold falsifies R010 on that seed.
# Decision: every seed must hold every coverage rule and every falsifier rule; COMPLETED iff so,
# INVALID if a seed is missing, unresolvable, or its record does not describe the frozen run, FALSIFIED
# otherwise.
version = 1
experiment = "R010"
seed_aggregation = "every_seed_must_pass"

coverage = [
  "deterministic_revocation_attempts >= 1",
  "pages_above_j_before_revocation >= 1",
  "pages_at_or_below_j_before_revocation >= 1",
  "deterministic_reuse_events >= 1",
  "nondeterministic_reuse_attempts >= 1",
  "post_revocation_reuse_probes >= 1",
  "retained_pages_reused_after_revocation >= 1",
  "recomputed_pages_above_j_served >= 1",
  "probed_page_set_digest == cached_page_set_digest_above_j",
  "nondeterministic_divergence_observed >= 1",
  "internal.revoked_expert_routed_tokens >= 1",
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
  "expert_generation_revocations_performed >= 1",
  "pages_above_layer_cached_before_revocation >= 1",
  "matching_post_revocation_lookups >= 1",
  "inflight_expert_revocation_race_cases.revocation >= 1",
  "inflight_input_race_cases.revocation >= 1",
  "inflight_expert_revocation_race_cases.supersession >= 1",
  "inflight_input_race_cases.supersession >= 1",
  "selection_digest_mismatches <= 0",
  "procedure_mismatches <= 0",
  "inflight_race_cases_by_cell.revocation_expert >= 1",
  "inflight_race_cases_by_cell.revocation_semantic_input >= 1",
  "inflight_race_cases_by_cell.supersession_expert >= 1",
  "inflight_race_cases_by_cell.supersession_semantic_input >= 1",
  "invalidation_matrix_cases.base_model_normal_revocation >= 1",
  "invalidation_matrix_cases.base_model_normal_supersession >= 1",
  "invalidation_matrix_cases.base_model_lag_revocation >= 1",
  "invalidation_matrix_cases.base_model_lag_supersession >= 1",
  "invalidation_matrix_cases.base_model_restore_revocation >= 1",
  "invalidation_matrix_cases.base_model_restore_supersession >= 1",
  "invalidation_matrix_cases.base_model_inflight_revocation >= 1",
  "invalidation_matrix_cases.base_model_inflight_supersession >= 1",
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
  "expert_quarantine_kv_cases >= 1",
  "model_pin_mismatches <= 0",
]

falsifiers = [
  "stale_pages_served <= 0",
  "wrongly_dropped_pages <= 0",
  "nondeterministic_exact_tags <= 0",
  "mismatching_runs_tagged_exact <= 0",
  "bitwise_mismatch_count <= 0",
  "revoked_lineage_entry_hits <= 0",
  "stale_hits <= 0",
  "stale_expert_generation_hits <= 0",
  "outputs_after_expert_revocation <= 0",
  "outputs_after_input_invalidation <= 0",
  "inflight_outputs_after_invalidation.revocation_expert <= 0",
  "inflight_outputs_after_invalidation.revocation_semantic_input <= 0",
  "inflight_outputs_after_invalidation.supersession_expert <= 0",
  "inflight_outputs_after_invalidation.supersession_semantic_input <= 0",
  "invalidation_matrix_stale_hits.base_model_normal_revocation <= 0",
  "invalidation_matrix_stale_hits.base_model_normal_supersession <= 0",
  "invalidation_matrix_stale_hits.base_model_lag_revocation <= 0",
  "invalidation_matrix_stale_hits.base_model_lag_supersession <= 0",
  "invalidation_matrix_stale_hits.base_model_restore_revocation <= 0",
  "invalidation_matrix_stale_hits.base_model_restore_supersession <= 0",
  "invalidation_matrix_stale_hits.base_model_inflight_revocation <= 0",
  "invalidation_matrix_stale_hits.base_model_inflight_supersession <= 0",
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
  "kv_reuse_after_expert_fence <= 0",
  "internal.adoption_mismatches <= 0",
  "internal.restart_resolution_mismatches <= 0",
  "internal.matrix_selectivity_mismatches <= 0",
  "internal.batch_invariance_mismatches <= 0",
]
```
