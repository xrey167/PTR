//! Per-layer KV chain keys computed from the actual weights and configuration of a
//! [`Qwen2PagedStack`] (doc 36 section D, "the per-layer chain key").
//!
//! The K/V of layer `l` depend on the tokens and on everything that produced the residual
//! stream feeding layer `l`'s K/V projections. The chain therefore is
//!
//! ```text
//! chain(-1) = H(domain, base model + generation, embedding-table digest, build identity,
//!               tokenizer, key encoding, attention shape)
//! chain(l)  = H(domain, l, chain(l-1),
//!               [l >= 1] digest of ALL of layer l-1 (attention incl. Q/K/V/O and QK-norm,
//!                         both norms, MLP, eps, attention configuration),
//!               layer l input norm weights + eps,
//!               layer l K/V projection weights and biases,
//!               layer l key-norm weights + eps, or "disabled",
//!               [l >= 1 or PostRope] RoPE: theta, scaling policy, regime base bits)
//! ```
//!
//! Rules and where doc 36 states them:
//!
//! * Layer `l`'s own FFN, output projection and query norm run after its K/V are stored, so
//!   they enter only `chain(l+1)` through the "all of layer l-1" term ("sweep rule").
//! * The key norm is a post-projection K transformation of layer `l` and so enters
//!   `chain(l)` ("every post-projection K/V transformation of layer l as stored"). The
//!   query norm does not touch stored K/V and reaches later layers through the residual.
//! * RoPE: with `PostRope` the stored key is rotated, so the configuration enters every
//!   `chain(l)` including layer 0. With `PreRope` layer 0's K/V are projected from the
//!   embedding before any attention, so the configuration is bound at use there and enters
//!   `chain(l)` for `l >= 1` only, because the attention of layers `< l` rotated Q/K and
//!   shaped the residual (consequences table: "PreRope keys: layers >= 1").
//! * Regime: the effective RoPE scaling regime is fixed at admission from the maximum
//!   sequence length (doc 36) and enters the same `[l >= 1 or PostRope]` term as the exact
//!   bits of the effective rotary base ([`Qwen2RopeRegime`]); the scaling policy (factor,
//!   original maximum positions) is hashed beside it. Requests in different regimes never
//!   share a chain, so reuse across a regime boundary is a miss. All lengths `<= omp`
//!   give the base `theta` and so one regime.
//! * Floats are hashed as exact little-endian bit patterns (bitwise reuse: `-0.0` and
//!   `0.0` differ). The stack refuses non-finite weights at construction, so no NaN
//!   payload can alias.
//! * `max_position_embeddings` is deliberately not hashed: it bounds capacity but changes
//!   no stored K/V byte.
//!
//! The catalog id of a whole stack is the last chain, [`KvLayerChains::final_id`].
//!
//! Adapters ([`derive_adapted_kv_layer_chains`]) follow the same placement rules, and a
//! stack without adapter and without MoE hashes exactly the bytes it always did:
//!
//! * a K/V delta (QKV, aLoRA) of layer `l` enters `chain(l)` (adapter identity, generation,
//!   kind, rank, scale, layer and the exact A/B bits of the K and V pairs), so
//!   `chain(l' < first_layer)` equals the base chain;
//! * Q and MLP deltas never touch a stored K/V row of their own layer, so they enter only
//!   `chain(l + 1)`, through the "all of layer l" term, which also carries every pair of the
//!   layer (QOnly and MLP-only: `chain(l <= first_layer)` equals the base chain);
//! * an aLoRA's adapted chain additionally binds the invocation position (inside the kind),
//!   and pages wholly before the invocation use the base chain
//!   ([`KvAdaptedChains::effective_layer`]);
//! * a MoE layer `j` enters the "all of layer j" term with its router, every expert's weights,
//!   generation and live flag, the top-k and the routing mode, so it reaches `chain(l)` for
//!   `l > j` only. Expert (j, i) never changes `chain(l <= j)`.

use crate::{
    KvKeyEncoding, Qwen2Activation, Qwen2Adapter, Qwen2AdapterKind, Qwen2AdapterLayer,
    Qwen2LinearWeights, Qwen2LoraPair, Qwen2MoeLayer, Qwen2PagedStack, Qwen2QkNorm,
    Qwen2RopeRegime, Qwen2RopeScaling, Qwen2RoutingMode, Qwen2StackLayerWeights,
};
use sha2::{Digest as _, Sha256};
use std::ops::Range;

const DOMAIN: &[u8] = b"ptr.kv.chain.v2\0";

/// What identifies the producer outside the weights.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvChainIdentity {
    /// Base model name; identity plus [`Self::generation`], never the name alone.
    pub base_model: String,
    pub generation: u64,
    /// Backend build identity: engine version, numerics, producer environment.
    pub build: String,
    /// Tokenizer name and version.
    pub tokenizer: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvChainError {
    /// One identity field is blank.
    BlankIdentity(&'static str),
    /// The stack scales RoPE by sequence length: use
    /// [`derive_kv_layer_chains_in_regime`] with the regime fixed at admission.
    RopeRegimeRequired,
    /// The regime is not the one the stack's configuration yields for its admitted length.
    RopeRegimeMismatch,
}

impl std::fmt::Display for KvChainError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for KvChainError {}

/// `chain(-1)` and `chain(0..L)` of one stack.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvLayerChains {
    root: [u8; 32],
    layers: Vec<[u8; 32]>,
}

impl KvLayerChains {
    /// `chain(-1)`.
    pub fn root(&self) -> [u8; 32] {
        self.root
    }

    /// `chain(layer)`, `None` past the last layer.
    pub fn layer(&self, layer: usize) -> Option<[u8; 32]> {
        self.layers.get(layer).copied()
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// `chain(L-1)`, the id a whole-stack cache entry is stored under.
    pub fn final_id(&self) -> [u8; 32] {
        self.layers.last().copied().unwrap_or(self.root)
    }
}

/// Derive the chain of every layer from the stack's weights and configuration, for a stack
/// without length-dependent RoPE scaling. A dynamic-NTK stack needs the regime fixed at
/// admission and is refused with [`KvChainError::RopeRegimeRequired`].
pub fn derive_kv_layer_chains(
    stack: &Qwen2PagedStack,
    identity: &KvChainIdentity,
) -> Result<KvLayerChains, KvChainError> {
    match stack.config().layer.rope_scaling {
        Qwen2RopeScaling::None => {
            let regime = stack.config().layer.static_rope_regime();
            derive_kv_layer_chains_in_regime(stack, identity, &regime)
        }
        _ => Err(KvChainError::RopeRegimeRequired),
    }
}

/// Derive the chain of every layer under the effective RoPE regime fixed at admission.
pub fn derive_kv_layer_chains_in_regime(
    stack: &Qwen2PagedStack,
    identity: &KvChainIdentity,
    regime: &Qwen2RopeRegime,
) -> Result<KvLayerChains, KvChainError> {
    derive_chains(stack, identity, regime, None)
}

/// The base chains of a stack and the chains of the same stack under an adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvAdaptedChains {
    base: KvLayerChains,
    adapted: KvLayerChains,
    /// First position at which the adapter acts (0 unless aLoRA).
    activation_position: usize,
}

impl KvAdaptedChains {
    pub fn base(&self) -> &KvLayerChains {
        &self.base
    }

    pub fn adapted(&self) -> &KvLayerChains {
        &self.adapted
    }

    pub fn activation_position(&self) -> usize {
        self.activation_position
    }

    /// The chain under which rows of `layer` at `positions` are stored. A range wholly
    /// before the aLoRA invocation position uses the base chain; a range that contains any
    /// position at or after it (including one that straddles it) uses the adapted chain,
    /// which binds the invocation. For QKV, Q-only and MLP-only adapters the adapted chain is
    /// returned throughout (it equals the base chain below the adapted layers). `None` for
    /// a layer out of range or an empty range.
    pub fn effective_layer(&self, layer: usize, positions: Range<usize>) -> Option<[u8; 32]> {
        if positions.is_empty() {
            return None;
        }
        if positions.end <= self.activation_position {
            self.base.layer(layer)
        } else {
            self.adapted.layer(layer)
        }
    }
}

/// Derive base and adapted chains for a stack without length-dependent RoPE scaling.
pub fn derive_adapted_kv_layer_chains(
    stack: &Qwen2PagedStack,
    identity: &KvChainIdentity,
    adapter: &Qwen2Adapter,
) -> Result<KvAdaptedChains, KvChainError> {
    match stack.config().layer.rope_scaling {
        Qwen2RopeScaling::None => {
            let regime = stack.config().layer.static_rope_regime();
            derive_adapted_kv_layer_chains_in_regime(stack, identity, adapter, &regime)
        }
        _ => Err(KvChainError::RopeRegimeRequired),
    }
}

/// Derive base and adapted chains under the effective RoPE regime fixed at admission.
pub fn derive_adapted_kv_layer_chains_in_regime(
    stack: &Qwen2PagedStack,
    identity: &KvChainIdentity,
    adapter: &Qwen2Adapter,
    regime: &Qwen2RopeRegime,
) -> Result<KvAdaptedChains, KvChainError> {
    Ok(KvAdaptedChains {
        base: derive_chains(stack, identity, regime, None)?,
        adapted: derive_chains(stack, identity, regime, Some(adapter))?,
        activation_position: adapter.activation_position(),
    })
}

fn derive_chains(
    stack: &Qwen2PagedStack,
    identity: &KvChainIdentity,
    regime: &Qwen2RopeRegime,
    adapter: Option<&Qwen2Adapter>,
) -> Result<KvLayerChains, KvChainError> {
    if !stack.config().layer.accepts_regime(regime) {
        return Err(KvChainError::RopeRegimeMismatch);
    }
    for (name, value) in [
        ("base_model", &identity.base_model),
        ("build", &identity.build),
        ("tokenizer", &identity.tokenizer),
    ] {
        if value.trim().is_empty() {
            return Err(KvChainError::BlankIdentity(name));
        }
    }
    let config = stack.config();
    let weights = stack.weights();
    let post_rope = config.key_encoding == KvKeyEncoding::PostRope;

    let root: [u8; 32] = {
        let mut hasher = start("root");
        text(&mut hasher, &identity.base_model);
        hasher.update(identity.generation.to_le_bytes());
        hasher.update(embedding_digest(&weights.embedding));
        text(&mut hasher, &identity.build);
        text(&mut hasher, &identity.tokenizer);
        text(
            &mut hasher,
            match config.key_encoding {
                KvKeyEncoding::PreRope => "pre-rope",
                KvKeyEncoding::PostRope => "post-rope",
            },
        );
        for value in [
            config.vocab_size,
            config.layer.hidden_size,
            config.layer.num_attention_heads,
            config.layer.num_key_value_heads,
        ] {
            hasher.update((value as u64).to_le_bytes());
        }
        hasher.finalize().into()
    };

    let mut layers = Vec::with_capacity(weights.layers.len());
    let mut previous = root;
    for (index, layer) in weights.layers.iter().enumerate() {
        let mut hasher = start("layer");
        hasher.update((index as u64).to_le_bytes());
        hasher.update(previous);
        if index >= 1 {
            hasher.update(full_layer_digest(
                stack,
                index - 1,
                &weights.layers[index - 1],
            ));
            if let Some((adapter, layer)) = adapter.and_then(|a| Some((a, a.layer(index - 1)?))) {
                adapter_term(&mut hasher, adapter, index - 1, layer, AdapterPart::All);
            }
        }
        // Everything applied to the residual before the K/V projections.
        floats(&mut hasher, &layer.decoder.input_layernorm);
        hasher.update(config.layer.rms_norm_eps.to_bits().to_le_bytes());
        linear(&mut hasher, &layer.decoder.key);
        linear(&mut hasher, &layer.decoder.value);
        match (config.qk_norm, &layer.key_norm) {
            (Qwen2QkNorm::Enabled, Some(norm)) => {
                text(&mut hasher, "key-norm");
                floats(&mut hasher, norm);
                hasher.update(config.layer.rms_norm_eps.to_bits().to_le_bytes());
            }
            _ => text(&mut hasher, "key-norm:disabled"),
        }
        if let Some((adapter, layer)) = adapter.and_then(|a| Some((a, a.layer(index)?))) {
            if adapter.adapts_kv() {
                adapter_term(&mut hasher, adapter, index, layer, AdapterPart::KeyValue);
            }
        }
        if index >= 1 || post_rope {
            rope(&mut hasher, stack, regime);
        } else {
            text(&mut hasher, "rope:bound-at-use");
        }
        previous = hasher.finalize().into();
        layers.push(previous);
    }
    Ok(KvLayerChains { root, layers })
}

fn start(role: &str) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    text(&mut hasher, role);
    hasher
}

fn text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn floats(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

fn linear(hasher: &mut Sha256, weights: &Qwen2LinearWeights) {
    hasher.update((weights.output_size as u64).to_le_bytes());
    hasher.update((weights.input_size as u64).to_le_bytes());
    floats(hasher, &weights.weight);
    match &weights.bias {
        Some(bias) => {
            hasher.update([1u8]);
            floats(hasher, bias);
        }
        None => hasher.update([0u8]),
    }
}

fn rope(hasher: &mut Sha256, stack: &Qwen2PagedStack, regime: &Qwen2RopeRegime) {
    let layer = &stack.config().layer;
    text(hasher, "rope");
    hasher.update(layer.rope_theta.to_bits().to_le_bytes());
    match &layer.rope_scaling {
        Qwen2RopeScaling::None => text(hasher, "scaling:none"),
        Qwen2RopeScaling::DynamicNtk {
            factor,
            original_max_position_embeddings,
        } => {
            text(hasher, "scaling:dynamic-ntk");
            hasher.update(factor.to_bits().to_le_bytes());
            hasher.update((*original_max_position_embeddings as u64).to_le_bytes());
        }
        Qwen2RopeScaling::Unsupported(name) => {
            text(hasher, "scaling:unsupported");
            text(hasher, name);
        }
    }
    // The effective regime fixed at admission (doc 36): exact bits of the effective base.
    text(hasher, "regime:effective-base");
    hasher.update(regime.effective_base_bits().to_le_bytes());
    hasher.update(((layer.hidden_size / layer.num_attention_heads) as u64).to_le_bytes());
}

fn embedding_digest(embedding: &[f32]) -> [u8; 32] {
    let mut hasher = start("embedding");
    floats(&mut hasher, embedding);
    hasher.finalize().into()
}

/// Canonical digest of ALL of one layer: both norms, Q/K/V/O, QK-norms, MLP and the
/// attention and numerics configuration that shapes its output.
fn full_layer_digest(
    stack: &Qwen2PagedStack,
    index: usize,
    layer: &Qwen2StackLayerWeights,
) -> [u8; 32] {
    let config = &stack.config().layer;
    let decoder = &layer.decoder;
    let mut hasher = start("full-layer");
    floats(&mut hasher, &decoder.input_layernorm);
    for part in [
        &decoder.query,
        &decoder.key,
        &decoder.value,
        &decoder.output,
    ] {
        linear(&mut hasher, part);
    }
    floats(&mut hasher, &decoder.post_attention_layernorm);
    for part in [&decoder.gate, &decoder.up, &decoder.down] {
        linear(&mut hasher, part);
    }
    for norm in [&layer.query_norm, &layer.key_norm] {
        match norm {
            Some(values) => {
                hasher.update([1u8]);
                floats(&mut hasher, values);
            }
            None => hasher.update([0u8]),
        }
    }
    hasher.update(config.rms_norm_eps.to_bits().to_le_bytes());
    hasher.update((config.intermediate_size as u64).to_le_bytes());
    match config.sliding_window {
        Some(window) => {
            hasher.update([1u8]);
            hasher.update((window as u64).to_le_bytes());
        }
        None => hasher.update([0u8]),
    }
    match &config.hidden_act {
        Qwen2Activation::Silu => text(&mut hasher, "silu"),
        Qwen2Activation::Unsupported(name) => {
            text(&mut hasher, "unsupported");
            text(&mut hasher, name);
        }
    }
    if let Some((moe, moe_layer)) = stack.moe().and_then(|moe| Some((moe, moe.layer(index)?))) {
        moe_term(&mut hasher, moe.top_k(), moe.mode(), moe_layer);
    }
    hasher.finalize().into()
}

/// Router, every expert (weights, generation, live flag), top-k and routing mode of one MoE
/// layer. Appended only for MoE layers, so dense layers hash as before.
fn moe_term(hasher: &mut Sha256, top_k: usize, mode: Qwen2RoutingMode, layer: &Qwen2MoeLayer) {
    text(hasher, "moe");
    hasher.update((top_k as u64).to_le_bytes());
    match mode {
        Qwen2RoutingMode::Dropless => text(hasher, "routing:dropless"),
        Qwen2RoutingMode::BatchVariantCapacity { capacity } => {
            text(hasher, "routing:batch-variant-capacity");
            hasher.update((capacity as u64).to_le_bytes());
        }
    }
    linear(hasher, &layer.router);
    hasher.update((layer.experts.len() as u64).to_le_bytes());
    for expert in &layer.experts {
        hasher.update(expert.generation.to_le_bytes());
        hasher.update([u8::from(expert.live)]);
        linear(hasher, &expert.gate);
        linear(hasher, &expert.up);
        linear(hasher, &expert.down);
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum AdapterPart {
    /// The K and V pairs only (what a stored K/V row of the layer depends on).
    KeyValue,
    /// Every pair of the layer (what the layer's output depends on).
    All,
}

/// Adapter identity, generation, kind (with the aLoRA invocation position), rank, scale,
/// layer index and the exact bits of the selected pairs of one layer.
fn adapter_term(
    hasher: &mut Sha256,
    adapter: &Qwen2Adapter,
    index: usize,
    layer: &Qwen2AdapterLayer,
    part: AdapterPart,
) {
    text(
        hasher,
        match part {
            AdapterPart::KeyValue => "adapter:kv",
            AdapterPart::All => "adapter:layer",
        },
    );
    text(hasher, adapter.identity());
    hasher.update(adapter.generation().to_le_bytes());
    match adapter.kind() {
        Qwen2AdapterKind::Qkv => text(hasher, "kind:qkv"),
        Qwen2AdapterKind::QOnly => text(hasher, "kind:q-only"),
        Qwen2AdapterKind::MlpOnly => text(hasher, "kind:mlp-only"),
        Qwen2AdapterKind::Activated { invocation } => {
            text(hasher, "kind:activated");
            hasher.update((invocation as u64).to_le_bytes());
        }
    }
    hasher.update((adapter.rank() as u64).to_le_bytes());
    hasher.update(adapter.scale().to_bits().to_le_bytes());
    hasher.update((index as u64).to_le_bytes());
    let pairs: [(&str, &Option<Qwen2LoraPair>, bool); 4] = [
        ("q", &layer.q, part == AdapterPart::All),
        ("k", &layer.k, true),
        ("v", &layer.v, true),
        ("mlp", &layer.mlp, part == AdapterPart::All),
    ];
    for (name, pair, selected) in pairs {
        if !selected {
            continue;
        }
        text(hasher, name);
        match pair {
            Some(pair) => {
                hasher.update([1u8]);
                floats(hasher, &pair.a);
                floats(hasher, &pair.b);
            }
            None => hasher.update([0u8]),
        }
    }
}
