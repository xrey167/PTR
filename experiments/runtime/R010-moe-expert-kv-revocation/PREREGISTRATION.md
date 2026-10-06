# R010 moe-expert-kv-revocation: preregistration

Frozen text of the preregistration of R010. It is a `file` key of `config.toml`'s `[preregistration]` table, so its
SHA-256 is frozen with the table. Nothing in it is a result. The decisions are made by `criteria.toml` (frozen the same
way) and applied by `aggregate.py`; this file gives their reasons, the modelling choices and what the harness cannot
produce. The harness is
`bins/ptr-bench/src/experiments/{layered_kv,r010}/` (subcommand `moe-expert-kv-revocation <seed>`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command):
`cargo run --release --locked -p ptr-bench -- moe-expert-kv-revocation <seed>`

Scope: KV-memory property only (`claim_scope = "kv_memory_only"`). The code under test is the CPU reference path: the
bound `KvSession`s of `ReferenceKvReuseBackend` over the tiny seeded Qwen2 stack with a seeded tiny MoE (`Qwen2Moe`), per-layer
entries whose dependency sets carry the experts of the layers below, and `PtrRuntime` as the liveness oracle with **the real expert
admission and fence authority path** (birth, retire, reactivate: `admit_expert_proposal`; fence and readmission: `fence_expert`,
`readmit_expert`; the registry's materialised `expert:{project}/{pod}` record answers liveness; nothing is scripted).

## 1. Decision rule

Same as R009 (`aggregate.py` is R005's with this experiment's names): five seeds, one JSON object each, `criteria.toml` applied mechanically, COMPLETED / FALSIFIED / INVALID, every seed must
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
| fence epoch recorded by a binding | the registry's state version of each tracked expert at binding time (non-zero); the successor generation records 1 | a recorded epoch is judged against the state version, so a readmitted expert never revives earlier entries |
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
- Fence cells: the expert is fenced by the real `fence_expert` (AuthorityQuarantine) and the committed `ExpertFenceCommitted` event is handed to `apply_committed_event`. A fenced expert is not live for the oracle, so a request citing it is refused at begin.
- Lag and restore arms as in R009 (section 6 there), including the live restore control.

## 7. Unsupported by the harness, and limits

- `base_model_inflight_restart`: no live base model exists to re-resolve to: the in-flight base-model cells check cancellation, silence and no insert, and do not restart (a restart would compute from the revoked base model)
- `binding_carried_post_event_lookups`: a request on a binding with a dead dependency (base model, expert) is refused by begin_session_bound before any catalog lookup, so post-event lookups on stale-eligible entries (post_event_lookups*, restored/stale-eligible counters) are counted for the segment cells only; the use-time rejection of an entry that depends on such a dependency is exercised in flight (a session cancelled at its next step) and by the post-revocation probes of the deterministic arm, which are begin refusals and masked-router lookups under other chains
- `fence_catalog_hook`: the backend's catalog is not the runtime's hooked catalog: a fence commits through the real authority path (PtrRuntime::fence_expert) and the harness then tombstones the entries by the expert's identity with invalidate_inputs; the runtime's own hook is unit-tested in crates/ptr-runtime/tests/expert_fence.rs

Also: every binding records, per tracked expert, the fence epoch (state version) the registry holds at binding time, so any later transition of any
kind ends the dependency; the successor generation records its own epoch (a reactivated generation starts at state version 1, `ExpertSuccessor::fence_epoch`).

## 8. Files

`PREREGISTRATION.md` (this frozen text), `criteria.toml` (the rules, with their grammar; a harness test pins them to the rules the harness judges),
`aggregate.py` (applies them to the five seed records; verdict COMPLETED, FALSIFIED or INVALID; no static evidence), `config.toml` `[preregistration]` (the frozen table).
The harness is `bins/ptr-bench` (`moe-expert-kv-revocation <seed>`, one JSON object on stdout).
