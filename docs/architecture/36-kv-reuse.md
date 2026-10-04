# 36 — KV-cache reuse: control plane in PTR, data plane in engines

Status: **design.** Nothing in this document is implemented beyond the existing
`ptr-pods` CPU reference code described in section A, which is not wired into any
inference path. Every claim about speed-up or hit rate is an **experiment** with a
falsification rule (section E), and all of them are planned, none has run.
Nothing here changes the authority order of ADR-0002: the ledger stays the only
authority, and a KV block is derived state under invariants #3, #5 and #17.

The one-sentence design: **PTR decides whether a cached KV block may be used;
an engine (or the CPU reference) computes and stores it.** Admission, keys,
invalidation and revocation live in `ptr-runtime`, `ptr-semdb` and `ptr-ledger`;
tensors, paging and kernels live behind a backend trait.

## A. Current state in code

All line numbers were opened and checked against the tree at the time of writing.

### No real inference path

- `ModelRequest` carries `raw_text: String`
  (`crates/ptr-model-api/src/request.rs:7`); there are no token ids.
- `InferenceBackend::infer` returns a synchronous `Vec<ModelEvent>`
  (`crates/ptr-model-api/src/backend.rs:9`); there is no streaming and no cache
  report.
- The only backends are `ReferenceEchoBackend` and `ScenarioBackend`
  (`crates/ptr-model-api/src/backend.rs:20` and `:44`), and `bins/ptrd` accepts
  exactly those two names (`bins/ptrd/src/main.rs:83-84`) and refuses others.
- No workspace crate tokenizes model input. `ptr-memory` has a
  `SemanticTokenizer` (`crates/ptr-memory/src/ingestion.rs:33`), but it turns a raw
  event into a semantic event, not a model vocabulary id sequence. The `tokenizers`
  crate appears only in `Cargo.lock` as a transitive dependency of `candle-core`
  (patched via `Cargo.toml:87`).

### Unwired CPU KV reference in `ptr-pods`

| Piece | Where | What it is |
|---|---|---|
| `KvPageSize` | `crates/ptr-pods/src/paged_kv.rs:12`, default 16 at `:15` | tokens per page |
| `KvPageBinding` | `paged_kv.rs:51` | model, adapter, generation, execution manifest, principal, device, dtype, page size |
| `publish_prefix` / `fork_prefix` | `paged_kv.rs:799` / `:845` | seal full pages, register a prefix, fork it copy-on-write into an empty cache |
| `KvContinuationExecutor` | `crates/ptr-pods/src/attention.rs:113` (`prefill` `:118`, `decode` `:125`) | continuation over a paged backend |
| `Qwen2PagedDecoder` | `crates/ptr-pods/src/qwen.rs:261` | one-layer Qwen2 decoder step |
| single-layer limit | `qwen.rs:297` (`layer_count != 1` is rejected) | |
| `apply_rope` | `qwen.rs:423`, applied to the key at `:328` before `append` | keys are stored **after** RoPE |

Around it in the runtime and memory crates:

- `ManagedKvRegistry` (`crates/ptr-runtime/src/tensor_kv.rs:99`) owns KV handles
  with placement and fencing.
- `StateBinding` (`crates/ptr-runtime/src/neural.rs:289`) binds opaque neural state
  to a journal anchor, revision and codebook; `NeuralStateCache::admit`
  (`neural.rs:972`) re-decides admission at every use and exposes no payload
  accessor. This is the model for how KV must be admitted (doc 27).
- `KvStateMetadata` (`crates/ptr-memory/src/kv.rs:26`) records
  `dependency_digests` (`kv.rs:34`) so semantic edits invalidate dependent state.
- `ptr-storage` has `TierObjectDomain::KvSnapshot`
  (`crates/ptr-storage/src/tier.rs:18`) and a content-addressed
  `TierObjectManifest` (`tier.rs:113`).
- `proto/podwire.proto` deliberately removed `session_id` (`:6-13`, number and name reserved).
  **A KV handle must never cross the wire**; only activations and policy do
  (doc 33). Today this is a **design rule, not an enforced guarantee**: removing
  `session_id` does not stop KV material from crossing PodWire, because
  `TypedPayload` is an arbitrary `type_id` plus bytes (`crates/ptr-protocol/src/lib.rs:16-19`)
  and the `PodCall` conversion checks only that the type id is non-empty
  (`crates/ptr-protocol/src/lib.rs:61-85`, `MissingTypeId` at `:73`). Enforcement is
  phase P5a.

### Five defects that block reuse

1. **The prefix key cannot be looked up before computing.** `publish_prefix`
   derives `prefix_digest` from the digests of the K/V tensor pages
   (`paged_kv.rs:814`, `digest_list(b"ptr-kv-prefix-v1\0", &page_digests)`). A new
   request does not have those pages yet, so it cannot ask "is my prefix cached?".
   Reuse needs keys derived from **tokens**: a chain `parent hash -> chunk hash`
   as in vLLM Automatic Prefix Caching.
2. **The prefix catalog is never pruned and revocation depends on the caller.**
   `PagedBackendState::prefixes` (`paged_kv.rs:333`) is only inserted into
   (`:826`) and read (`:867`). `fork_prefix` compares the generation of the
   **caller's own** binding (`:854-862`), not the currently admitted generation, so
   invariant #3 holds only if every caller passes a fresh binding. The fix (P2) is
   that the backend's `fork_prefix` takes a **runtime-minted admission witness**, as
   `NeuralStateCache::admit` does, instead of trusting a caller-supplied binding.
3. **Keys are stored after RoPE** (`qwen.rs:321-336`), so a cached key is bound to
   its absolute position. That forbids positional reuse (section B, T2).
4. **`ModelRequest` is a public-field struct built literally in five places** outside
   its definition: `crates/ptr-model-api/tests/smoke.rs:6`,
   `crates/ptr-model-api/tests/backend.rs:8`,
   `crates/ptr-protocol/tests/prost.rs:307`, `crates/ptr-runtime/src/lib.rs:1305`
   and `:1420`. Adding a field is a breaking change in all five; a **new trait**
   avoids it.
5. **Every page bundle holds all layers, and the adapter binds the whole cache.**
   `HostPageBundle.layers: Vec<KvLayerSnapshot>` (`paged_kv.rs:304`, field at `:309`),
   `KvTensorSchema.adapter` (`crates/ptr-pods/src/kv.rs:50`) and
   `KvPageBinding.adapter` (`paged_kv.rs:53`). Base-model KV and LoRA KV therefore
   cannot be shared per layer even where they are mathematically identical.

## B. Literature, tiered

Each entry was checked against the paper; corrections to common claims are stated
where they matter. Tier decides what PTR may do with the method.

### T1 — exact, adopt

| Method | Reference | What PTR takes, and corrections |
|---|---|---|
| PagedAttention / vLLM | arXiv 2309.06180, SOSP'23 | fixed blocks (16 tokens), copy-on-write sharing; matches the existing 16-token `KvPageSize` |
| RadixAttention / SGLang | arXiv 2312.07104 | radix tree over token prefixes with LRU; the shape of `KvPrefixCatalog` |
| vLLM Automatic Prefix Caching | vLLM documentation | block hash = parent hash + block tokens + extras (for example the LoRA id); `cache_salt` is mixed into the first block; the `sha256_cbor` hash is reproducible across versions, the default pickle-based `sha256` is **not** |
| Activated LoRA (aLoRA) | arXiv 2504.12397, NeurIPS 2025 | states before the invocation token are identical to the base model in all layers (Prop. 1), so the base prefix is shared. Its vLLM PR #19710 was **closed unmerged on 2026-02-09**; a vLLM follow-up paper is arXiv 2512.17910 |

### T2 — positional and edit reuse (exact only for one layer with pre-RoPE keys, otherwise Approximate)

| Method | Reference | Notes and corrections |
|---|---|---|
| PIE | arXiv 2407.03157 | position-integrated re-encoding for edits. The ">85%" figure refers to **KV-update latency only** and was evaluated on **code only** |
| "Suffix Cache Reuse (SCR)" | none | **not a named method.** The mechanism is PIE plus Leyline |
| Leyline | arXiv 2606.01065 | guarantee is "positional, not informational"; its FORGET mode is a real re-prefill |
| CacheBlend | arXiv 2405.16444, EuroSys'25 | selective recompute of about 15% of tokens; implemented in LMCache |
| EPIC | arXiv 2410.15332, ICML'25 | LegoLink recomputes the first k (k <= 32) tokens of each chunk. "Up to 8x" TTFT is **versus CacheBlend under synthetic multi-request load, not versus full prefill**; up to 3x versus CacheBlend for a single request |
| MEPIC | arXiv 2512.16822 | page-aligned chunks, pre-RoPE storage, RoPE inside a fused kernel. The gain is **HBM memory 2-5x**, latency only 9-11% |
| CacheSlide | FAST'26 | needs LoRA/CoPE training, **deferred** |

### T3 — lossy and offload: optional derived tier only, always Approximate

| Method | Reference | Notes |
|---|---|---|
| ShadowKV | arXiv 2410.21465 | pre-RoPE low-rank keys plus CPU values, nothing permanently evicted; the best fit for the tier system (phase P6) |
| TailorKV | arXiv 2505.19586 | per-layer choice between quantize and offload; needs calibration |
| CacheGen | arXiv 2310.07240, SIGCOMM'24 | lossy codec, transport only |
| MiniCache | arXiv 2405.14366 | cross-layer merge; needs a deep model |

**Rejected:** SnapKV (arXiv 2404.14469), Ada-KV (arXiv 2407.11550) and K-VEC
(arXiv 2606.29563; an unreviewed preprint without code). Query-dependent eviction
breaks prefix sharing, because what survives depends on the request, and Ada-KV's
variable-length heads conflict with fixed pages.

### T4 — learned hidden-state compaction: rejected as an authoritative path

| Method | Reference | Why |
|---|---|---|
| KV-streams | arXiv 2609.35750 | RL-trained. Retained KV deliberately carries deleted content, which is a **revocation leak**. Its reported speed-ups are mutually inconsistent across abstract, body and conclusion |
| MEMENTO | arXiv 2604.09852 | needs SFT; the paper reports about 2.5x lower peak KV and 1.75x throughput. It has an implicit KV channel; restarting from text only loses 15.3 percentage points on AIME24. **Only the text channel is adopted**, as a `ptr-memory` object with provenance |
| STILL | arXiv 2606.07878 (Baseten) | not called "Neural Compactor"; TTFT is worse (6.8 s versus 3.1 s) |
| DroidSpeak | arXiv 2411.02820 | approximate cross-model reuse |

### Key insight, and its limit

In a single-layer model the K/V of token i depends only on its embedding and its
position. With **pre-RoPE** storage, any positional reuse (shift, splice, drop a
span) is then exact. This is a **mechanism check on the reference code, not
evidence for deep models**, where the K/V of layer l depends on all earlier
layers' outputs, which depend on the whole prefix. Re-rotating an already rotated
key is not bit-identical in floating point, so keys are stored pre-RoPE and rotated
at use. The same limit applies to positional reuse: after an **insert of whole
`PromptSegment`s between unchanged segments, or a shift or reorder, in which every
recorded lineage segment is unchanged** (path (b) of P3), a
page reused out of position is exact for layer 0 only; layers `>= 1` may be reused
only Approximately, with selective recompute, tagged `KvReuseTaint`. A content edit
(including a deletion or redaction) or a revocation is path (a): the page is
unreachable for every layer, layer 0 included, and is freshly prefilled (section D).

## C. The Exact / Approximate rule

```rust
enum KvReuseMode { Exact, Approximate }
struct KvReusePolicy { allow_approximate: bool } // default: false
```

- **Exact** means the logits are bitwise equal to a full prefill of the same
  tokens under the same numerics. Only Exact reuse is admitted by default.
  The one **proven** Exact case is positional reuse with pre-RoPE keys in a
  **single-layer** model, and only once R006 has passed.
- **Approximate** reuse is admitted only when `allow_approximate` is set, and a run
  that used it may only produce **candidates**. Approximate are: positional
  re-encoding in multi-layer models, selective recompute, lossy tiers and
  non-dropless MoE.
- An effectful action reachable from a run with Approximate reuse fails with
  `RuntimeError::ApproximateStateAtEffectBoundary`.
- The mode is recorded in provenance of the result, **but provenance alone does not
  enforce the effect boundary** (next rule).

**Design rule: the mode must travel with the action.** Today the runtime copies a
bare `ActionIr` out of `ModelEvent::ActionReady` into `ResumableRun::action`
(`crates/ptr-runtime/src/lib.rs:1446-1473`, assignment at `:1472`; field at `:567`),
and `authorize_action` receives only that action (`:1558`), so a mode kept only in
result provenance is lost before authorization. Therefore:

- a `KvReuseTaint` (the mode, plus the reuse report digest) is carried with every
  model event and with `ResumableRun::action`;
- action admission and `authorize_action` check it and deny an Approximate-tainted
  action with `ApproximateStateAtEffectBoundary`;
- the taint is also attached to the result of every Pod the run requested
  (`ModelEvent::PodRequested`), and `promote_verified_pod_output` rejects or
  candidate-routes a tainted result, so an Approximate run cannot durably publish
  authoritative SemDB state either;
- taint propagates **transitively**: a candidate derived from an Approximate run
  stays tainted until it is recomputed Exact.

This is phase P2a below; until it exists the effect-boundary sentence above is a
design rule, not an enforced guarantee.

This applies invariant #5 (a cache cannot self-promote to truth), #7 (hard effect
boundary) and #11 (a learned or approximate signal cannot override a deterministic
verifier).

## D. LoRA, MoE and split Pods: the per-layer chain key

The K/V of layer l for tokens up to t depends on: the tokens, their positions, the
K/V projections and norms of layer l, and **all modules of layers below l**
(attention, FFN, router, experts). It does **not** depend on layer l's own FFN or
experts, which run after attention. So a cache key is a chain over layers:

```
chain(-1) = H( base-model identity + generation,
               embedding-table digest + generation, any embedding adapter,
               numerics config )

chain(l) = H( chain(l-1),                       // one formula for every l >= 0
              all weights of layer l-1 incl. router, experts, adapters, + their generations,
              every transformation applied to the residual stream before layer l's
                K/V projections (pre-attention norm weights, eps/config), + generations,
              K/V weights of layer l incl. adapter,
              numerics config: dtype, quantization, kernel set / batch-invariance,
                               TP/EP layout, inter-pod activation precision,
              positional-encoding config: rope_theta, rope_scaling policy,
                head_dim split (rotated vs pass-through dims), + their generation,
                included if l >= 1 (attention of layers < l rotated Q/K and so
                shaped the residual) or if KvKeyEncoding = PostRope (the stored
                key itself is rotated, so layer 0 too) )
```

For `l = 0` the term "all weights of layer l-1" is empty, so `chain(0)` binds
`chain(-1)` (the embedding table and any embedding adapter), layer 0's pre-attention
norm (weights, config, eps), layer 0's K/V projection with any layer-0 K/V adapter
generation and digest, the numerics config and, for `PostRope`, the RoPE
configuration. A QKV-LoRA attached at f = 0 therefore changes `chain(0)` and shares
no layers.

The **attention configuration** of every layer below l is also bound, because it
changes the residual that layer l projects K/V from: attention mask (causal,
bidirectional or custom), sliding-window size, attention sinks, GQA grouping
(`num_key_value_heads` per query heads), softmax scale or soft-cap, and QK-norm
weights and eps. The **base model is identified by identity and generation**, not by
a name alone.

General rule: the key covers **every transformation applied to the residual stream
before layer l's K/V projections, with its configuration and generation**. In the
Qwen2 reference path that includes `input_layernorm` and `rms_norm_eps`, which
`Qwen2PagedDecoder` applies before the K and V projections
(`crates/ptr-pods/src/qwen.rs:313-321`, weights field at `:115`); changing them
changes K and V of layer l even though no K/V weight changed.

**The positional encoding is part of the key, for every layer l >= 1 and, with
post-RoPE keys, for layer 0 too.** Two separate effects bind it:

- *Stored rotation (`PostRope`).* The stored key is already rotated, so the rotation
  configuration is baked into the cached bytes: `Qwen2PagedDecoder` passes
  `self.config.rope_theta` to `apply_rope` while it builds the key that is appended
  to the cache (`crates/ptr-pods/src/qwen.rs:328-334`; the config field is at `:22`,
  validated at `:45`). A `PostRope` chain key therefore binds `rope_theta`, the
  `rope_scaling` policy (any context-extension scheme, including its factors) **and its effective runtime state**, the
  `head_dim` split between rotated and pass-through dimensions, and the generation
  of that configuration at **every** layer, including layer 0.
- *Rotation upstream of the residual (both encodings).* With `PreRope` storage the
  RoPE configuration is bound at use for **layer 0's own K/V only**: layer 0 projects
  K/V from the embedding, before any attention has run, so its cached pre-RoPE
  tensors do not depend on the rotation. For every layer `l >= 1` the cached
  pre-RoPE K/V was projected from a residual that went through the attention of
  layers `< l`, and that attention rotated Q and K under the RoPE configuration in
  force. The residual, and so layer l's K/V, therefore depends on `rope_theta`,
  scaling and the rotated-dimension split even though the stored tensor carries no
  rotation. The configuration must enter the per-layer chain key at every `l >= 1`
  (equivalently: it enters above the first attention layer, at the step from
  `chain(0)` to `chain(1)`). Binding it only at use would classify layers `>= 1` as
  valid after a RoPE change and serve stale tensors as Exact.

**Effective scaling state, not only the static policy.** For context-length-dependent
schemes (dynamic NTK and similar) the effective rotation frequencies depend on the
sequence length. The chain key therefore includes the **effective scaling factors in
force** (or the sequence-length regime that selects them) for every affected layer,
and requests in different regimes never share entries: reuse across a regime boundary
is a **miss**. The alternative, forbidding reuse under dynamic scaling, is not
chosen; binding the effective state is preferred.

So a RoPE configuration change invalidates layers `>= 1` for `PreRope` storage and
all layers for `PostRope` storage. In a single-layer model (the reference path)
`PreRope` has no layer `>= 1`, which is why the one-layer positional-reuse check in
section B holds and does not extend to deep models.

Consequences:

| Change | Layers invalidated |
|---|---|
| pre-attention norm weights or eps at layer k | layers >= k |
| `rope_theta`, `rope_scaling` or `head_dim` split, `PostRope` keys | every layer |
| `rope_theta`, `rope_scaling` or `head_dim` split, `PreRope` keys | layers >= 1 (layer 0's own K/V is unaffected and is bound at use; layers >= 1 see the rotation through the residual) |
| QKV-LoRA attached at layer k | layers >= k |
| Q-only or MLP-only LoRA at layer k | layers >= k+1 |
| aLoRA | every layer before the invocation token is shared with the base |
| swap or revoke expert (j, i) | only layers > j |

### The token trie is independent of adapter identity

The prefix catalog has two levels, and adapter identity belongs only to the second:

1. **Token trie (radix over token ids).** `KvChunkKey { parent, chunk (16 tokens),
   salt }` with `salt = H(PrincipalId, ProjectId, base-model identity, tokenizer,
   numerics, schema)`. The salt contains **neither the adapter, nor the execution-manifest
   digest, nor any generation**: generations live on the entries and in `chain(0)`.
   Isolation is unchanged: two principals, or two projects, have different salts and
   diverge at the first node.
2. **Per-layer entries on each trie node**, keyed by
   `(chain(l), semantic_lineage_digest)`, where `chain(l)` is the per-layer chain key
   above and the lineage digest is the one defined below. One trie node can therefore
   hold **several entries for identical tokens with different lineages**, and
   admission compares the request's lineage with the entry's and admits only an exact
   match. A request can hit only an entry whose `chain(l)` equals the one computed for
   **its own** adapter binding. The adapter enters `chain(l)` only for layers at or after the first
   layer the adapter touches (`AdapterAttachment::first_layer`, plus one for a
   Q-only or MLP-only adapter, table below), and for an aLoRA adapter only for
   tokens after the activation point. Layers below that point have the same
   `chain(l)` for the base model and for every adapter, which is what lets them be
   shared.

**Why the manifest digest cannot be in the trie salt.** The execution-manifest digest
folds in the adapter binding: `calculate_digest` hashes the manifest generation
(`crates/ptr-pods/src/manifest.rs:229`) and then, in one loop over knowledge,
artifacts, reader **and adapter** bindings (`:230-236`), each binding's key,
generation and digest (`:237-239`). Any adapter key, generation or digest change
therefore changes the manifest digest. A salt that mixed in the manifest digest, or
the adapter value, would differ between a base request and a LoRA or aLoRA request,
so the two would diverge at the **first radix node**, before any per-layer chain
could discover that the lower-layer pages are identical. Layer sharing (P2b) and
R009 could not work as specified. Partitioning the token trie by the whole adapter
binding is therefore rejected.

**Adapter admission metadata lives on the per-layer entries**, not in the trie salt:
each entry records its **effective dependencies**: the base-model identity and
generation, the layer-range chain key, and, **only if the adapter enters `chain(l)`**,
the adapter key, generation and digest. Admission validates those effective
dependencies and that their generations are still live (not revoked), re-deciding at
every use as `NeuralStateCache::admit` does (section A). It does **not** require the
producing execution-manifest digest to equal the requesting one: that digest differs
between a base and an adapter request even for layers whose pages are identical, so
requiring equality would defeat the sharing. A base-keyed entry records no adapter
dependency, so revoking an adapter drops only the entries that name it and never a
base-keyed page.

**Entries also record their semantic input dependencies.** For **every `PromptSegment`
before and including the entry's own** (source `Raw`, `Semantic` or `Memory`, per the
P1 contract), not only the segments that supplied its own tokens, the entry records
the validated input identity, its generation and its `content_digest`: its
**transitive prefix lineage**. This matters because the K/V of every layer `>= 1`
depends on the whole prefix (each earlier segment shaped the residual), so a hidden
state can carry removed context. The lineage is **derived from the trie ancestors**: a
segment reference is stored only on the node where a segment begins, so the metadata
is O(n) in prompt length and not O(n^2) per-page copies; the digest over the ordered
list and the per-segment set for revocation lookup are computed from that chain. A recorded segment is **live** iff its exact (identity, generation, `content_digest`)
is the one **committed at the current ledger anchor** and is not revoked (invariants #3
and #4; `StateBinding` binds a journal anchor). Admission re-checks liveness at every use and does **not** rely only on subtree invalidation: `invalidate_inputs` may be delayed, missed or lost across a restore, so invalidation is an eviction optimisation and the use-time check is the guarantee. The catalog keeps a
reverse index from input key to trie nodes, and `invalidate_inputs` (proposed, not in
the tree) maps the semantic keys affected by a **committed** delta to entries through
the recorded identities. `PreparedDelta::affected()` (`crates/ptr-semdb/src/lib.rs:459`,
returning `&BTreeSet<String>`; the type is at `:444`) is the source of that set, but it
describes a delta that is only **prepared and unpublished**, so invalidation is wired
to the commit of the delta, never to the prepare step: an entry derived from an
overlay that is later discarded must never be admissible. A token-identical entry derived
from an edited or revoked input is therefore never admissible, even though its token
chain key matches. Revocation of **any** upstream segment makes the entry
**unreachable**, not merely tainted, and its bytes are **tombstoned and purged**
(a revocation **cancels every active fork whose lineage includes the revoked input**:
its run is cancelled and must restart from a fresh prefill, and no token, candidate or
result produced after the revocation may escape; the page is tombstoned immediately and
its bytes are reclaimed when the refcount reaches 0;
`KvSnapshot` tier objects are purged the same way, with a reclamation check): this keeps "revoked generations cannot become
usable after restart/cache/index lag" (`SECURITY.md:17-21`, the invariant list, whose
third item is that sentence) true for cached hidden state.

**Security property kept:** a request can hit only per-layer entries whose chain key
matches exactly the one derived from its own salt (principal and project), base model, adapter
binding, numerics and RoPE configuration, and only after every effective dependency
of that entry is validated as live at the current committed anchor. Sharing a trie node never lets a request read an entry computed
under a different adapter, generation, principal or project; it only lets requests with
different adapters reach the same node, where the chain keys then decide.

**Disaggregated expert pods are KV-stateless**: KV lives with attention
(MegaScale-Infer, arXiv 2504.02263; Step-3 attention-FFN disaggregation,
arXiv 2507.19427). Only **pipeline (layer) split** is specified for the wire: a
network split of experts is **not supported at L1** (doc 37, section 2) and has no
E007 coverage. **Pipeline split:** the KV of a layer range lives on the Pod hosting
it; only activations cross Pods, as a `TypedPayload`:

```
ActivationFrame { boundary, positions, upstream_chain, tensor }
```

`proto/podwire.proto` is unchanged, and revocation uses the existing
`Revoke { subject, generation }` (`proto/podwire.proto:22`). A `Revoke` message can be
dropped or delayed, so it is not the guarantee: each Pod **re-validates against a
runtime-issued fence epoch at use**, as `ManagedKvRegistry` fencing does
(`crates/ptr-runtime/src/tensor_kv.rs:99`), and a downstream Pod never serves KV
derived from an upstream chain whose stage generation has been revoked.

**Exact requires dropless routing and batch-invariant kernels.** Capacity limits
make routing depend on the batch (Soft MoE, arXiv 2308.00951, section 2.2), and
kernel reductions are batch-dependent unless made invariant (Thinking Machines,
"Defeating Nondeterminism in LLM Inference", Sep 2025; vLLM `VLLM_BATCH_INVARIANT`).
Without both, the MoE layers and every layer above them are **Approximate**.

Proposed types (none exists yet):

- `ptr-types`: `LayerIndex`, `LayerRange`, `ExpertId`.
- `ptr-pods`: `ModelComposition`, `AdapterAttachment { first_layer, kv_proj_adapted,
  activation: Always | AfterInvocation }`, `MoeTopology`, `RoutingDeterminism`.
- `KvLayerBinding { layers, effective_weights_digest, generations }`.

## E. Phases, experiments and falsification

All phases are **planned**. P0 is this document; every other phase has an
experiment that can fail.

| Phase | Content | Experiment and falsification |
|---|---|---|
| P0 | this document, experiment records | none |
| P1 | token-aware contract in `ptr-model-api`: `TokenId`, `Tokenizer` trait, `ByteTokenizer`, `PromptSegment` (with `content_digest` and source `Raw` / `Semantic` / `Memory`), `SegmentedModelRequest`, `KvReuseHint`, `KvReuseReport`, `ModelRun`, and a **new trait** `KvReusingBackend`. Existing backends are untouched. A real HF tokenizer comes later in a new crate `ptr-tokenizers` | contract tests only |
| P2 | exact prefix reuse on CPU. A `retire_prefix` trait method with default `Unsupported` (the candle path is not compiled in CI). `ptr-runtime/src/kv_prefix.rs`: `KvPrefixCatalog`, a radix tree over `KvChunkKey { parent, chunk (16 tokens), salt }` with `salt = H(PrincipalId, ProjectId, base-model identity, tokenizer, numerics, schema)`, a project-scoped security namespace (the runtime's `AdmissionPolicy` binds a peer to one principal and one grant set and carries no project: `crates/ptr-runtime/src/execution.rs:291-298`, so one principal can act in several projects), deliberately **without** the adapter, the execution-manifest digest or any generation (section D); per-layer entries on each node keyed by `chain(l)`. **Admission** is per layer at the deepest node and validates at **every use** each entry's effective dependencies (base model, chain key, adapter only if it enters the chain) **and its recorded semantic-input dependencies** (identity, generation and `content_digest` per `PromptSegment`; live means committed at the current ledger anchor and not revoked), not the producing manifest digest, and **not relying only on invalidation**, since `invalidate_inputs` may be delayed, missed or lost across a restore; then a walk back to the admitted ancestor. Backend `fork_prefix` takes a **runtime-minted admission witness** (like `NeuralStateCache::admit`) instead of trusting a caller binding. `invalidate_inputs` is wired to **committed** deltas, not to `PreparedDelta::affected()` at prepare time, so an entry from a discarded overlay is never admissible. A revocation **cancels every active fork whose lineage includes the revoked input**: the run is cancelled and must restart from a fresh prefill, no token, candidate or result produced after the revocation may escape, the page is tombstoned immediately and its bytes are reclaimed when the refcount reaches 0. Per-layer entries are keyed by `(chain(l), semantic_lineage_digest)`, so one node can hold several entries for identical tokens with different lineages and admission admits only an exact lineage match. A **bounded catalog**: capacity limits on trie nodes and metadata and on KV page bytes (pinned in the R005 config), LRU (or equivalent) eviction that reclaims trie metadata and unreferenced pages, refcounts so a page held by an active fork or run is never evicted; eviction never changes correctness, a miss recomputes (this fixes the "never pruned" defect of section A). Lineage metadata is derived from trie ancestors (O(n)) and bounded by `max_metadata_bytes` (preregistered: a per-node budget of 1 KiB, so 4096 nodes give 4 MiB; the page bytes per node are stated for the pinned shape; the nodes-only and bytes-only over-capacity workloads run independently); tombstoned revoked pages and `KvSnapshot` tier objects are purged with a reclamation check. `ReferenceKvReuseBackend` | **R005 exact-prefix-reuse**. **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** `cross_principal_attempts` below 1; `cross_project_attempts` below 1 (same principal, same prefix, different project); `input_revocations_performed` below 1; post-revocation lookups with matching tokens below 1 on entries whose lineage includes the revoked segment; same_tokens_distinct_lineage_entries below 2; `active_fork_revocation_attempts` below 1; attempts to read a revoked page pinned by an active fork below 1; attempts to read an entry from a discarded overlay below 1; `evictions` below 1 under the unique-prompt workload larger than capacity; `forged_witness_fork_attempts` below 1; attempts to admit an effect from a cancelled run below 1; the lag arm (`invalidation_suppressed_arms` below 1, the revocation committed but `invalidate_inputs` fault-injected to be dropped), the restore arm (`restore_arms` below 1: catalog restored from a pre-revocation snapshot or tier object, or runtime restart), or post-event lookups below 1 in either arm; fewer seeds than pinned. **Falsified if any of:** a bitwise difference against full prefill; `forged_witness_forks_accepted` above 0 (a fork with a forged admission witness); `cancelled_run_effects_admitted` above 0 (an effect of a cancelled run admitted; invariant #7 and `authorize_action`); any successful fork of a revoked generation; a hit rate below 40% at 50% shared prefix (workload pinned in the manifest: request count, prompt length a multiple of 16, shared length, cold-start handling, concurrency, so the target is achievable); catalog nodes, page bytes or `peak_metadata_bytes` above the bound under that unique-prompt workload; eviction of a page referenced by an active fork; `cross_principal_hits` above 0; `cross_project_hits` above 0; `revoked_lineage_entry_hits` above 0; `cross_lineage_hits` above 0; a revoked pinned page served after the revocation; bytes of a revoked pinned page not all reclaimed after release; `discarded_overlay_entry_hits` above 0; `outputs_escaped_after_revocation` above 0; any stale hit in the lag or restore arm. Every seed must meet each threshold (preregistered) |
| P2a | `KvReuseTaint` carried with every model event and with `ResumableRun::action`, checked by action admission and `authorize_action`, propagated transitively to derived candidates (section C). The taint also covers **Pod results**, not only actions: on an Approximate run a `ModelEvent::PodRequested` (handled at `crates/ptr-runtime/src/lib.rs:1337` and `:1438`) makes the Pod result carry the `KvReuseTaint`, and `promote_verified_pod_output` (defined at `crates/ptr-runtime/src/semantic.rs:58`, called from `lib.rs:1376` and `:1515`) must reject tainted output or route it to a candidate; it never publishes it to SemDB as authoritative state. A persisted Approximate candidate **stays tainted when reloaded after a restart** | **R005**, additional clauses. **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** `fresh_exact_recomputation_admitted` below 1; `tainted_candidate_reload_attempts` below 1; approximate_pod_result_promotion_attempts below 1; taint checked over fewer than 2 hops. The Approximate producer is a pinned stub backend returning `KvReuseMode::Approximate`, and attempt counters count only attempts made after persistence and before recomputation. **Falsified if any of:** an action admitted from a run whose reuse mode was Approximate; a SemDB promotion from an Approximate run; an admission along the path Approximate result, persisted candidate, later otherwise-Exact run before a fresh Exact recomputation; `taint_lost` above 0 after a reload; an Approximate Pod result promoted (accepted above 0) |
| P2b | LoRA layer sharing and aLoRA. Sharing is of **KV memory only**: a forward-compute saving would need a layer-boundary residual cache, which is out of scope, so no `prefill_flops` claim is made. The sharing oracle is derived per adapter type from `kv_proj_adapted` and the exact affected layer set (consequences table): a K/V-changing adapter (QKV-LoRA) first attached at layer f shares layers `< f`, fraction `f/L`; a Q-only or MLP-only adapter at f leaves layer f's own K/V unchanged and shares layers `<= f`, fraction `(f+1)/L`; an aLoRA shares every layer for the rows before the invocation point. **Invocation inside a page:** the boundary page is split by a per-page **invocation offset** kept as sub-page dependency metadata (the alternative, forcing a page boundary at the invocation point with the partial page closed, is not chosen): rows before the offset are keyed by the base chain, rows from the offset on by the adapter-dependent chain, and a page containing any post-invocation row is **never admitted under a base-only key**. The aLoRA sharing oracle counts all full pre-invocation pages plus the pre-invocation rows of the boundary page, for every layer. Base and adapter requests share one token trie and discover shared lower-layer pages through per-layer chain keys; adapter dependencies are recorded on, and admitted per, the per-layer entries (section D); admission never compares whole manifest digests | **R009 layer-shared-lora-kv**. **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** `base_pages_before_revoke` below 1; `adapter_entries_before_revocation` below 1; adapter A's entry requested by adapter B (and an old generation against the new one) attempted fewer than 1 time; any of QKV-LoRA and Q-only at f = L/2 and at f = 0, MLP-only, or the aLoRA case (invocation position and prompt length pinned) not run; with N layers, `input_revocations_performed` below 1 or post-revocation lookups with matching tokens below 1; the lag arm (`invalidation_suppressed_arms` below 1, the revocation committed but `invalidate_inputs` fault-injected to be dropped), the restore arm (`restore_arms` below 1: catalog restored from a pre-revocation snapshot or tier object, or runtime restart), or post-event lookups below 1 in either arm. **Falsified if any of:** shared pages differ bitwise from the base pages; the measured sharing deviates from the fraction predicted per adapter type; revoking the adapter drops any base-keyed page; a hit after revocation lands on an entry whose chain key depends on the revoked adapter generation; the aLoRA prefix is not bit-identical; the adapter request's logits are not bit-identical to a full adapter prefill; cross_adapter_hits_on_mismatched_chain above 0; `revoked_lineage_entry_hits` above 0; any stale hit in the lag or restore arm |
| P2c | MoE expert generations. Sharing is of **KV memory only** (no `prefill_flops` claim, as in P2b). Invalidation is **conservative per layer**: revoking expert (j, i) makes every page above j unreachable even where it would still be valid, and only pages at or below j must survive | **R010 moe-expert-kv-revocation**. **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** `nondeterministic_divergence_observed` below 1; with N layers, `input_revocations_performed` below 1 or post-revocation lookups with matching tokens below 1; the lag arm (`invalidation_suppressed_arms` below 1, the revocation committed but `invalidate_inputs` fault-injected to be dropped), the restore arm (`restore_arms` below 1: catalog restored from a pre-revocation snapshot or tier object, or runtime restart), or post-event lookups below 1 in either arm. **Falsified if any of:** a page above j that was created before the revocation is served after it; a page at or below j is dropped because of the revocation (pages recomputed after the revocation are allowed); `mismatching_runs_tagged_exact` above 0 (a nondeterministic-routing run tagged Exact); a bitwise mismatch of deterministic (dropless, batch-invariant) MoE reuse against a full prefill of the same prompt under the same expert generations; `revoked_lineage_entry_hits` above 0; any stale hit in the lag or restore arm |
| P3 | pre-RoPE keys: `KvKeyEncoding { PostRope, PreRope }` with a versioned digest; N-layer `Qwen2PagedStack`. `KvChunkKey` includes the parent-prefix hash, so after an edit before a cached chunk every later key changes even if the contents do not. P3 therefore adds a **position-independent secondary index**: key `H(salt, chunk token ids, chain-relevant config)` **without the parent**, mapping to `PreRope` pages plus their original absolute positions; after an edit, lookup realigns by content. The index is built at **sub-page granularity** (rolling hashes over small token spans, 1 token up to a page, or content-defined boundaries) so a non-page-aligned insert can still be matched; a shifted suffix is **repacked from old pre-RoPE K/V rows by span** (rows are per token): old **layer 0** rows are copied into new pages at the new alignment **only if the page's lineage is live**, and a span with no content match is **recomputed, never partially reused**. Every page records its **transitive prefix lineage** (section D) and admission revalidates liveness at use, where a segment is live iff its exact (identity, generation, `content_digest`) is committed at the current ledger anchor and not revoked. **Path (a), revocation or content edit:** if any lineage segment is revoked, or its content was edited (including a deletion or redaction, which supersedes the old generation under invariant #3), the page is unreachable for **every layer, layer 0 included**, and is freshly prefilled (`SECURITY.md:17-21`). **Path (b), insertion of whole new `PromptSegment`s between unchanged segments, shift or reorder, in which every recorded segment is unchanged:** (an insertion **inside** a segment changes that segment's `content_digest`, which is path (a) with zero reuse, so P3 helps only reorder, shift or insertion of whole segments) approximate upper-layer reuse with selective recompute (CacheBlend/EPIC-style, k tokens) is permitted; the result is tagged Approximate and carries `KvReuseTaint` (section C), so it never reaches an effect path. No removed context can leak on path (b), because it requires every lineage segment to be unchanged and live. The index yields Exact reuse only for the single-layer case R006 proves; multi-layer positional reuse stays Approximate (section C) | **R006 positional-reuse-oracle** (one layer) and **M016 approximate-chunk-reuse**, both exercising lookup through the catalog and its secondary index, not pages handed to the backend. **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** `reused_tokens_per_shift` equal to 0; secondary_index_hits_per_shift below 1; non-page-aligned insertions of whole segments of 1 token and of 7 tokens not run; the in-segment-insertion case (zero reuse expected) not run; post-revocation secondary lookups with matching content below 1; `canary_probes` below 1 in the canary-edit case; the lag arm (`invalidation_suppressed_arms` below 1, the revocation committed but `invalidate_inputs` fault-injected to be dropped), the restore arm (`restore_arms` below 1: catalog restored from a pre-revocation snapshot or tier object, or runtime restart), or post-event lookups below 1 in either arm. **R006 is falsified if any of:** a bitwise difference against a `PreRope` full prefill at the new position. **M016** has a single selection rule: `k_sel` is the smallest k in [4, 7, 10, 13] whose top-1 agreement on a disjoint calibration split is at least 99% at every shift (if none exists, M016 is falsified), and it is then evaluated on a held-out split, where `k_sel` needs agreement of at least 99% **and** a median speedup of at least 2x at every shift. Agreement is counted only at positions whose K/V used approximate rows. The recomputed tokens are pinned (preregistered: the CacheBlend-style highest-deviation tokens, in every layer `>= 1`). The workload is pinned so that the target is feasible: a 512-token prompt in 64-token chunks, 7 of 8 chunks reused (448 reused tokens, 64-token suffix). The FLOP cost model includes the deviation-selection pass over the reused tokens, and the analytic bound in the config requires each candidate k to reach 2x in the ideal model; a k that cannot is dropped and reported. Bitwise experiments (R006, R009, R010, R005, E007) use seeded random weights with a pinned seed and shape; the agreement experiments (R008, M016) use the pinned pretrained `Qwen/Qwen2.5-0.5B` at a resolved Hugging Face revision sha, with the checkpoint's layer count and vocabulary; `prefill_speedup` is the median after a pinned warm-up and includes lookup, secondary-index, repack and recompute time; the 1-token and 7-token insertions count toward both criteria, and the 2x target is measured on path (b) only (live_lineage_insert_shift_cases means insertion of whole segments, or shift, with every recorded segment unchanged). **M016 is falsified if any of:** any reuse in the in-segment-insertion case; no k meeting the calibration rule; held-out top-1 agreement below 99% at `k_sel` at any shift; median `prefill_speedup` below 2x at `k_sel` at any shift; `revoked_page_layers_reused` above 0; `canary_logit_shift` above 0 in the case where an edit removes a canary; `reused_results_tagged_approximate` not equal to `reused_results`, or `approximate_tag_violations` above 0; any stale hit in the lag or restore arm. The safety checks do not count toward the speed metric |
| P4 | span eviction without hidden carry-over (Leyline FORGET) | **R007 span-forget** (bitwise only; there is no percentile path). **Falsified if any of:** the state differs from a fresh prefill without the span; a canary token remains probeable; `canary_logit_shift` (the absolute difference of the canary token's logit, maximised over probes) is above 0; `stale_downstream_rows_reused` above 0; `recompute_tokens` not equal to the number of tokens after the span |
| P5a | **pipeline (layer) split** only; a network split of experts is not supported at L1 (doc 37, section 2) and is outside P5a and E007. Includes an explicit **wire-type allowlist**: `TypedPayload` is arbitrary bytes plus a caller-controlled `type_id` (section A), so a deny-list of KV type ids can be evaded by mislabelling. PodWire admission for split-pod operations therefore accepts **only** the activation-frame and policy schemas valid for that operation, decodes them (typed, schema-validated, including tensor shape tokens x width x dtype and the length of `positions`), and rejects any other `type_id`, in `ptr-protocol` conversion and in PodWire admission, with a test; the shape check is kept only for wrong-shape payloads. The real invariant is that the **receiving Pod has no KV-install path from PodWire**: an activation frame is only ever fed to compute. `wire_kv_page_bytes` is the byte count of any payload that does not decode as an allowlisted schema **or** matches the digest of a KV page held by either Pod. Until it lands, "no KV handle on the wire" is a design rule, not an enforced guarantee (section A). It also settles the size of an `ActivationFrame`: PodWire bounds one payload at `MAX_BODY_BYTES = 4 MiB` (`crates/ptr-podwire/src/frame.rs:65`, enforced on the request body at `:503-508`). The reference path supports **F32 only** (`crates/ptr-pods/src/kv.rs:244`), so the boundary dtype is pinned to F32 and one frame of 2048 tokens x 2048 width x 4 B is **16 MiB**, which does not fit (the model width 2048 and the vocabulary size are pinned, preregistered). P5a therefore specifies **bounded chunking of activation frames**: each frame has a stable `frame_id`, and each chunk is its own `TypedPayload` below the body bound carrying `{ frame_id, chunk_index, chunk_total, chunk_digest, frame_digest, upstream_chain }`, where `chunk_digest` binds `frame_id`, `chunk_index`, `chunk_total`, `frame_digest`, `upstream_chain` and the chunk bytes and **every chunk also carries a whole-frame digest**; the chunks of one frame travel as **distinct PodWire requests with distinct request ids**, because a request id that a session already completed with different bytes is refused with `PodWireError::DuplicateRequest` (`crates/ptr-podwire/src/endpoint.rs:242`, raised at `:1094`, `:1152` and `:1191`), so one outer request id must never be reused across chunks; `upstream_chain` is repeated on **every** chunk so a stage can refuse a chunk stored under a mismatched chain without waiting for the rest; an incomplete frame carries a **frame-level deadline**: it expires after a bounded time or when its session or request is cancelled, its buffered chunks are released, and a typed error is returned; the number of incomplete frames per peer is limited; the receiver reassembles keyed by `(session, frame_id)` into a **bounded queue** (invariant #8) with a limit on chunks in flight, and rejects a frame that is incomplete, duplicated, reordered beyond the window, mixes chunks of two frame versions under one `frame_id`, or whose per-chunk or whole-frame digests do not match. E007 pins (preregistered) 5 chunks of at most 3.5 MiB for the 16 MiB frame, `reorder_window = 4`, `frame_deadline_ms = 5000`, `max_incomplete_frames_per_peer = 8` and `max_chunks_in_flight = 32`. The alternative is an explicit, tested **supported-shape limit** (tokens x width x dtype below one body) that fails closed with a typed error; the document does not claim to support larger shapes under that alternative | **E007 split-pod-kv**. **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** `non_allowlisted_type_attempts` below 1 (each with a typed rejection, separate from the mislabelled-KV attempts); fewer than 1 adversarial mislabelled KV payload; any of chain_mismatch_attempts, duplicate_chunk_attempts, out_of_window_attempts, bad_chunk_digest_attempts, mixed_version_frame_attempts, incomplete_frames_over_peer_limit, queue_overflow_attempts below 1; `splice_attempts` below 1 (individually valid chunks from different frames); attempts with an activation-shaped smuggled KV page, consumed as activations and never installed, below 1; `reassemblies_after_abandon` below 1; `interleaved_chunk_switches` below 1 (two frames sharing an `upstream_chain` reassembled concurrently); `abandoned_chunks_sent` below 1, or buffered bytes not above baseline before the abandonment; no frame above one body (the large F32 frame of 16 MiB), unless the shape-limit alternative is used and its typed refusal is shown; stale_upstream_entries_before_revocation below 1 (a downstream KV entry under the old upstream generation must exist before the revocation); stale-upstream attempts below 1 (the `Revoke` message dropped or delayed, then a request sent). **Falsified if any of:** any payload with a non-allowlisted `type_id` admitted (`non_allowlisted_payloads_admitted` above 0); `wire_kv_handle_count`, `wire_kv_page_bytes` or any other `wire_kv_*` check above 0; kv_pages_installed_from_wire above 0; any attempt not rejected with a typed error; frames_with_consistent_whole_frame_digest not equal to `frames_sent` (all chunks of a frame must carry the same whole-frame digest); a spliced frame accepted; `abandoned_chunks_released` not equal to `abandoned_chunks_sent`, or buffered bytes not returned to baseline; the large F32 frame not bit-identical to the single-pod run (output means the final logits bytes plus the per-layer KV bytes on each pod; the single-pod baseline applies the identical boundary cast, a no-op in F32); `stale_upstream_kv_served` above 0 |
| P5 | engine adapter crate `ptr-model-engine` (`vllm-http`, `sglang-http`); an engine cannot do per-layer live-generation admission, so PTR defines an **engine-specific cache salt** containing the **project and the security principal** (`ProjectId` and `PrincipalId`, the same project-scoped namespace as the P2 salt) and every admitted generation and digest: base-model generation and artifact digest, adapter key, generation and digest, expert generations for MoE, and the numerics config. A revoked or replaced binding then can never reproduce an old block hash. Because the principal is in the salt, one principal cannot observe through timing whether another principal's prefix is cached; this is a claim E006 tests (below). **Every semantic or memory input revocation requires an engine-acknowledged salt rotation before further requests are served**, done by **salt-epoch rotation**, not a global flush (the engine is not required to evict). The semantic state is global (`crates/ptr-semdb/src/lib.rs:471-475`: one `SemanticHost` holds one `SemanticState`), so PTR keeps an explicit **principal-to-input index** and tracks, **per semantic generation, the set of principals whose cached blocks depend on it**, and a revocation rotates the salt epoch of **every dependent principal**; only a principal with no dependency on the revoked generation keeps its epoch and its hit rate; requests arriving during the rotation window are queued or rejected. The committed delta (not `PreparedDelta::affected()` at prepare time) drives the rotation. A single request-level engine salt cannot carry page-specific lineage, so the lineage digest is **not** put in the salt; per-block salts at segment boundaries would be required to do better and are out of scope unless an engine supports them. The salt must cover **every input that section D's per-layer chain key binds**, or state a fallback: (1) base-model weights, **normalization weights and eps/config**, embedding table, attention configuration, router and K/V weights: covered by the base-model generation and artifact digest, which must fold in those weights and configs; (2) adapter and expert generations: in the salt, as above; (3) **tokenizer** identity and version: in the salt, since the engine hashes token ids, not text; (4) **numerics** (dtype, quantization, kernel set and batch-invariance, TP/EP layout, inter-pod activation precision): in the salt; (5) **positional config**: `rope_theta`, scaling policy, rotated-dimension split, **its generation**, and the **effective RoPE scaling regime** (the scaling factors in force, or the sequence-length regime that selects them, as in section D), so that requests in different dynamic-RoPE regimes never share engine blocks; (6) `KvKeyEncoding`: in the salt. **Fallback:** if the engine cannot key a block per request regime (or cannot expose any item above), the model runs with engine prefix caching **disabled** and E006 is reported **not applicable for it, not passed**. Trade-off: generation-salted engine caches cannot share base pages across adapters; P2b's sharing is a property of the P2 reference backend, not of an engine, unless the engine supports per-layer keys. Router cost term `KvReuseEstimate` | **E006 engine-prefix-reuse**, run on a pinned vLLM version (preregistered) with `VLLM_BATCH_INVARIANT=1`, on a GPU profile in `hardware/` that supports vLLM's batch-invariant kernels (if no profile there does, E006 is inconclusive by design until suitable hardware exists), with the model pinned (OLMoE-1B-7B-0125 at revision 9b0c1aa87e34a20052389dce1f0cf01da783f654), a pinned test LoRA (rank, targets, seed), pinned expert generations and a pinned baseline workload for principal B; a missing batch invariance makes the run **inconclusive, not a pass**, and a model for which caching must be disabled is reported **not applicable, not passed**. **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** no regime-crossing probe; a salt-input transition with stale probes missing for any of model, adapter, expert, RoPE, tokenizer, numerics and `KvKeyEncoding`; `stale_semantic_probe_requests` below 8; no unsalted or no-rotation control that shows the stale hit; requests attempted during the flush window below 1; `cross_principal_attempts` below 1; `cross_project_attempts` below 1 (same principal, same prefix, different project); dependent-principal attempts below 1 (principal B used the revoked input and B's epoch is rotated); n_timing_samples_per_arm below 1000, measured separately from the hit-rate workload; the lag arm (`invalidation_suppressed_arms` below 1, the revocation committed but `invalidate_inputs` fault-injected to be dropped), the restore arm (`restore_arms` below 1: catalog restored from a pre-revocation snapshot or tier object, or runtime restart), or post-event lookups below 1 in either arm. **Falsified if any of:** the TTFT gain is below 1.5x at 50% shared prefix (workload pinned: request count, prompt length, shared length, concurrency, warm-up, and whether the first request is in the TTFT median); a hit on a block created under a revoked or replaced generation; a block shared across RoPE regimes; any bitwise difference between cache-enabled engine output and a cache-disabled full prefill under the identical binding and numerics; a stale hit, detected from the engine-reported cached tokens per request or from canary output divergence against a full prefill under the new binding; stale_semantic_hits_dependent_principals above 0; a request during the flush window that is neither queued nor rejected; the hit rate of a principal with no dependency on the revoked generation changed by the rotation; `cross_principal_hits` above 0; `cross_project_hits` above 0; a TOST equivalence test on median TTFT that fails to show equivalence within +/-5% of the baseline median at alpha 0.05 (bound preregistered), for the cross-principal arm and for the cross-project arm; any stale hit in the lag or restore arm |
| P6 | lossy tier `TierObjectDomain::KvSnapshotLossy` | **R008 lossy-kv-tier** (model, context length and positions pinned for top-1 agreement; `lossy_kv_bytes` counts all tiers plus index and metadata). **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** attempts with `allow_approximate` false below 1; revoked_lossy_objects_present_before_event below 1 (an object whose dependency was revoked must exist before the event); revocation attempts below 1; restart attempts below 1 (revocation and restart are counted separately). **Falsified if any of:** memory reduction below 2x at top-1 agreement of at least 98%; lossy_hits_with_allow_approximate_false above 0; a `KvSnapshotLossy` object whose dependency was revoked served after the revocation (hits above 0); the same object served after a restart (hits above 0); a still-valid object may survive a restart |

## F. Risks

- All P0-P4 evidence is **CPU reference math**. It is mechanism evidence, not
  evidence about production engines or deep models.
- **One-layer over-claiming:** exact positional reuse (section B) holds for one
  layer only; any text that generalizes it is wrong.
- The candle path is not compiled in CI, so GPU behaviour is unchecked.
- **Splitting pages by layer is the largest breaking change** (defect 5): the
  default range is `[0, L)` and digests are versioned.
- Invalidation above layer j is **conservative**: it may drop blocks that would
  still be valid.
- Cross-machine Exact needs reproducible arithmetic (numerics config in the chain
  key; batch-invariant kernels).
- **Metadata and tombstones are bounded.** Lineage metadata is O(n) (segment
  references at segment starts), and R005 pins `max_metadata_bytes` (preregistered)
  and reports `peak_metadata_bytes`. Unreachable revoked page bytes and `KvSnapshot`
  tier objects are tombstoned and purged, with a reclamation check.
- **Engine-side eviction is not controllable** by PTR: a miss is always possible,
  so the engine path can only claim "never a stale hit", not a hit rate.

## G. Control plane and data plane

```mermaid
flowchart LR
    subgraph CP["Control plane (PTR)"]
        L["ptr-ledger\ngenerations, revocation"]
        S["ptr-semdb\ncommitted delta (affected set)\ninvalidate_inputs"]
        A["ptr-runtime admission\nStateBinding, KvReusePolicy"]
        C["KvPrefixCatalog\nradix over KvChunkKey"]
        L -->|"live generations"| A
        S -->|"invalidated inputs"| C
        A -->|"salt from admitted binding"| C
    end
    subgraph DP["Data plane (tensors)"]
        R["ptr-pods reference\npaged KV, CPU"]
        E["engine\nvllm / sglang"]
    end
    C -->|"KvReuseHint (chunk chain)"| R
    C -->|"KvReuseHint (chunk chain)"| E
    R -->|"KvReuseReport (hit tokens, mode)"| A
    E -->|"KvReuseReport (hit tokens, mode)"| A
```

Per-layer chain key (section D):

```mermaid
flowchart LR
    E["embedding table + adapter\nbase model identity"] --> K0["chain(0)"]
    T["tokens, positions"] --> K0
    AC["attention config of layers < l\nmask, window, sinks, GQA, scale, QK-norm"] --> KL
    K0 -->|"weights of layer 0\nincl. router, experts, adapters"| K1["chain(1)"]
    K1 -->|"weights of layer 1"| K2["chain(2)"]
    K2 -->|"..."| KL["chain(l)"]
    N["numerics config\ndtype, kernels, TP/EP"] --> K0
    PN["pre-attention norm of layer l\nweights, eps, generation"] --> KL
    RP["rope_theta, rope_scaling,\nhead_dim split, generation"] -->|"PostRope: layer 0 too"| K0
    RP -->|"every l >= 1 (both encodings)"| K1
    KV["K/V weights of layer l\nincl. adapter"] --> KL
```

A changed weight, adapter or expert generation at layer j changes `chain(l)` for
every l > j (and for l = j when it touches the pre-attention norm or K/V), and
changes nothing below. A changed RoPE configuration is the exception: with
`PostRope` keys it enters at `chain(0)` and so changes every layer; with `PreRope`
keys it enters at `chain(1)` and so changes every layer `>= 1`, leaving only layer
0's own K/V reusable (its use-time rotation is checked at admission).
