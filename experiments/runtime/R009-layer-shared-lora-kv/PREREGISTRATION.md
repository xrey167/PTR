# R009 layer-shared-lora-kv: preregistration

Frozen text of the preregistration of R009. It is a `file` key of `config.toml`'s `[preregistration]` table, so its
SHA-256 is frozen with the table. Nothing in it is a result. The decisions are made by `criteria.toml` (frozen the same
way) and applied by `aggregate.py`; this file gives their reasons, the modelling choices and what the harness cannot
produce. The harness is
`bins/ptr-bench/src/experiments/{layered_kv,r009}/` (subcommand `layer-shared-lora-kv <seed>`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command):
`cargo run --release --locked -p ptr-bench -- layer-shared-lora-kv <seed>`

Scope: KV-memory sharing only (`claim_scope = "kv_memory_only"`): no prefill-FLOPs claim. The code under test is
the CPU reference path: `ReferenceKvReuseBackend` and its bound `KvSession`s over the tiny seeded Qwen2 stack
with seeded LoRA/aLoRA adapters (`Qwen2Adapter`), per-layer catalog entries, layer-scoped fork and checked
adoption, and `PtrRuntime` as the ledger-anchored liveness oracle.

## 1. Decision rule

For each of the seeds 17, 29, 43, 71, 101 the harness prints one JSON object (stdout; exit 0 clean, 1 unmet
or tripped or a gap, 2 a case it cannot run, including a vacuous adapter: B == 0 or no change of the logits).
`aggregate.py` (R005's, same grammar) reads the five records and applies
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
| aLoRA request shape | base request = first 48 tokens of the prompt, adapter request = those 48 + 16 others | a base prefix shorter than the prompt can be forked and truncated at the invocation; see section 7 |
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
  as a raw segment; a live control must fork or the harness stops (exit 2), so a restore cell's misses are never vacuous.
  Restored per-layer entries keep their layer scope (snapshot v3); a scope mismatch is tombstoned and counted
  (`internal.layered_scope_mismatches`, a falsifier).
- Events: every committed event of a cell (revocation, supersession, expert retire/reactivate/fence) is handed to
  `ReferenceKvReuseBackend::apply_committed_event`, the host's equivalent of the runtime's catalog hook, which also updates the
  backend's own liveness; only a semantic edit's delta (not readable from the event) is applied by identity.
- `internal.post_invocation_rows_under_base_key` (owner decision 5: max 0) is judged `<= 0`; it is not in the manifest's metric list, so it is reported under `internal`.
- Selection: `selection_digest_mismatches` compares the computed digest of all sequences the run draws with the pinned one.

## 7. Unsupported by the harness, and limits (null with a reason in the JSON)

- `base_model_inflight_restart`: no live base model exists to re-resolve to: the in-flight base-model cells check cancellation, silence and no insert, and do not restart (a restart would compute from the revoked base model)
- `binding_carried_post_event_lookups`: a request on a binding with a dead dependency (base model, adapter) is refused by begin_session_bound before any catalog lookup, so post-event lookups on stale-eligible entries (post_event_lookups*, restored/stale-eligible counters) are counted for the segment cells only; the use-time rejection of an entry that depends on such a dependency is exercised in flight (a session cancelled at its next step) and by the adapter-revocation probes in the configuration arms, which are begin refusals
- `prefill_flops_saving`: the claim is a KV-memory saving only (claim_scope kv_memory_only): partial hits are computed in full and adopted in checked mode, so no forward-compute saving exists to measure

Also, as measured limitations visible to the reader (not unsupported): with the whole prompt already cached under the base key an
aLoRA request shares nothing (`internal.alora_full_prompt_cached_shared_rows`), because a deeper base-keyed hit is refused for adoption
(`depth <= activation`) and a hit of the whole prompt cannot be forked; the pinned shape therefore uses a base request shorter than the prompt.

## 8. Files

`PREREGISTRATION.md` (this frozen text), `criteria.toml` (the rules, with their grammar; a harness test pins them to the rules the harness judges),
`aggregate.py` (applies them to the five seed records; verdict COMPLETED, FALSIFIED or INVALID; no static evidence), `config.toml` `[preregistration]` (the frozen table).
The harness is `bins/ptr-bench` (`layer-shared-lora-kv <seed>`, one JSON object on stdout).
