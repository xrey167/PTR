# R005 exact-prefix-reuse: preregistration

Frozen text of the preregistration of R005 (Exact half only). It is a `file` key of `config.toml`'s
`[preregistration]` table, so its SHA-256 is frozen with the table. Nothing in it is a result.
The decisions are made by `criteria.toml` (frozen the same way) and applied by `aggregate.py`; this
file gives their reasons, the modelling choices and what the harness cannot produce.

Scope: Exact half only. Approximate cases are R012; witness reuse is evidenced statically (owner
decisions of 2026-10-06). The harness is `bins/ptr-bench/src/experiments/r005/` (subcommand
`exact-prefix-reuse <seed>`), the manifest's entrypoint.

## 1. Decision rule

For each of the seeds 17, 29, 43, 71, 101 the harness prints one JSON object (stdout; exit 0 clean, 1
unmet or tripped, 2 a case it cannot run). `aggregate.py` reads the five run records, re-runs nothing
of the experiment, and applies `criteria.toml` mechanically:

- **COMPLETED** iff every seed meets every coverage rule and every falsifier rule, and both
  `static_evidence` commands pass at the frozen commit.
- **FALSIFIED** if any coverage rule (the run is vacuous) or any falsifier rule fails on any seed, or a
  `static_evidence` command fails.
- **INVALID** if a seed has no finished record, a record does not describe the frozen run (another
  benchmark or seed, another preregistration, other code), or a rule cannot be read or resolved
  (missing, null or mistyped counter). INVALID is not a result about the hypothesis.

Every seed must pass; there is no majority and no tolerance. No rule changes after a run; a change
needs a successor experiment.

## 2. Pinned values the config left open, with reasons

| value | pinned | reason / how verified |
|---|---|---|
| max_new_tokens | 8 | backend default (`ReferenceKvLimits::default`); regime admission is prompt + 8 |
| max_prompt_tokens | 256 | the main prompt length; max_position_embeddings 264 (not hashed in the chain) |
| intermediate_size | 128 | hard-coded in `kv_infer.rs::stack_config`; `pin::verify` rebuilds the stack from the pinned value and reproduces the backend's chain id and a cold run's logits digest |
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

## 3. Semantic liveness model

`semantic_liveness_model = "committed-edit-revision"`: the rule implemented by the `PtrRuntime`
liveness oracle (`crates/ptr-runtime/src/kv_prefix.rs`) and the operationalisation, for semantic keys,
of "(identity, generation, content_digest) committed at the ledger anchor" (doc 36, P2 row).

- A Semantic dependency is live iff its key is committed and not revoked, and
  `last_edit(key) <= the cited generation <= the current revision`. A compacted restore floors the
  edit history at the snapshot revision.
- A committed revocation of a key kills every cited generation of that key until a later committed
  edit of it; the catalog tombstones all entries of the key.
- A Semantic dependency never consults lifecycle generations: a lifecycle target (a capsule) that
  has the name of a semantic key does not make the key live. The harness has a case for it
  (`semantic_lifecycle_identity_*`).
- Production paths build semantic segments with `PtrRuntime::semantic_prompt_segment`, which renders
  the committed value and stamps the segment with the revision of the key's last edit. The generic
  `Semantic` constructors are caller-asserted: they cite whatever generation the caller names. The
  matrix cells use them on purpose, to cite a stale generation, and a separate case
  (`semantic_runtime_segment_*`) uses the runtime-built segment (live and reused while unedited,
  refused at use after an edit with the catalog's invalidation lost, not buildable once revoked).
- Memory and derived inputs are lifecycle targets (a committed evidence or capsule id with a
  lifecycle generation), base model and adapter use the ledger, expert liveness uses a scripted
  oracle in the harness (the runtime reads it from a materialised expert record).

Documented open risk (not a falsifier, not covered): the rule compares generations, not content, and a
caller-asserted segment is trusted for its content. A caller that cites the *current* generation for
stale content, through a generic constructor, is not detected by the oracle.

## 4. Metadata limits, as pinned per workload

`max_metadata_bytes` is pinned per workload in `workload_limits` (the numbers are not changed by this
text): main 8 MiB for 4096 nodes (2 KiB a node), nodes-only 64 MiB for 4096 nodes (16 KiB a node),
bytes-only 16 MiB for 16384 nodes (1 KiB a node). A per-node rate of 1 KiB holds for bytes-only only.
`metadata_bytes_per_node = 1024` is the harness's per-node charge, not a limit divided by nodes.
One trie node holds 16384 B of KV pages in the pinned shape, so 4096 nodes are 64 MiB.

## 5. Observation, not a falsifier

`KvPrefixCatalog::admit` loops its eviction whenever a bound is hit, and the eviction policy may evict
unpinned ancestor entries whose removal frees no node or page (their descendants keep the node)
before it returns `CapacityExhausted`. Pinned entries are never evicted
(`active_fork_page_evictions` is a falsifier). The harness counts `evictions_with_active_fork_pinned`
as coverage only. No rule is made of this behaviour.

## 6. Unsupported by the harness (null with a reason in the JSON), and the static evidence

- `witness_reuse_static_evidence_passing`: needs the doctest and unit test to run; the harness only reports (`witness_reuse_static_evidence_present`) that both items are in the source. `aggregate.py` runs the two `static_evidence` commands of `criteria.toml` itself, in the checkout it has shown to hold the frozen run's code, and records their exit codes and the SHA-256 of their output in `run.json`; either failing is FALSIFIED.
- `base_model_inflight_outputs_after_invalidation` (both types): no inference step carries a base-model dependency (the backend records only prompt segments). The in-flight base-model race is observed on a bare catalog (pin cancelled, insert refused: `base_model_inflight_cache_inserts` = 0, race cases 1 per type) but no output exists to count.
- Escaped outputs for the 7 catalog-layer in-flight cells (base model, adapter, expert x types incl. fence): same reason (`unsupported["invalidation_matrix.<cell>.escaped_outputs"]`). Their stale hit is a still-live pin or an accepted insert after the event. Base model, adapter and expert cells (28 of 52) are catalog-layer in every path; semantic, memory, derived (24) are session-layer. Expert liveness uses a scripted oracle (the runtime reads it from a materialized expert record); base model/adapter use the ledger.
- `revoked_pinned_page_bytes_reclaimed_after_release.backend_resident_bytes`: the backend exposes no resident-byte accounting; reclamation and all `*_kv_page_bytes` figures are the catalog's charged bytes.
- Modelling choices (not unsupported, but visible to the reader): a cancelled run has no outcome, so "effect of a cancelled run" = `finish`, one more token, and `authorize_action` on the revoked generation at the current revision (control: admitted before the revocation); memory and derived inputs are lifecycle targets (a committed capsule / evidence id string) because the runtime keeps no other generation for them; the restore arm recomputes the same content-addressed pages by running the tokens as a raw segment so only the liveness check can refuse a restored entry (control: a live restored entry with pages forks, 32 tokens); peaks are sampled after every prefill chunk.

## 7. History before the freeze (not results)

A dry run of the harness before the freeze (seeds 1, 17, 29, 43, 71, 101; not recorded, not
committed as a record) showed that a semantic input superseded by a committed edit was invisible to the
use-time liveness check (`impl KvLivenessOracle for PtrRuntime` treated a semantic key without a
lifecycle generation as live while it was in the snapshot). That is a defect of the code under test,
not of the preregistration; it was fixed in commit `4d056718` (the `committed-edit-revision` rule of
section 3, with the revocation floor and the runtime-built segment added by a later review before the freeze) and the dry run is not part of the evidence. The criteria were not changed by it: the same
rules the harness already judged. Two draft text errors were also corrected before the freeze: the
metadata limit arithmetic (section 4) and the hit-rate rule, written `hit_rate >= 0.4`.

## 8. Procedure (the head)

See `README.md` ("Freeze and run procedure"). The aggregation is `python3
experiments/runtime/R005-exact-prefix-reuse/aggregate.py`.
