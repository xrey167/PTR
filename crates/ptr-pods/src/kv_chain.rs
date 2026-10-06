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
//!               [l >= 1 or PostRope] RoPE: theta, scaling policy, regime placeholder)
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
//!   sequence length (doc 36). The stack supports no scaling, so the regime term is the
//!   constant placeholder `none`; it must become the real regime when scaling lands.
//! * Floats are hashed as exact little-endian bit patterns (bitwise reuse: `-0.0` and
//!   `0.0` differ). The stack refuses non-finite weights at construction, so no NaN
//!   payload can alias.
//! * `max_position_embeddings` is deliberately not hashed: it bounds capacity but changes
//!   no stored K/V byte.
//!
//! The catalog id of a whole stack is the last chain, [`KvLayerChains::final_id`].

use crate::{
    KvKeyEncoding, Qwen2Activation, Qwen2LinearWeights, Qwen2PagedStack, Qwen2QkNorm,
    Qwen2RopeScaling, Qwen2StackLayerWeights,
};
use sha2::{Digest as _, Sha256};

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

/// Derive the chain of every layer from the stack's weights and configuration.
pub fn derive_kv_layer_chains(
    stack: &Qwen2PagedStack,
    identity: &KvChainIdentity,
) -> Result<KvLayerChains, KvChainError> {
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
            hasher.update(full_layer_digest(stack, &weights.layers[index - 1]));
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
        if index >= 1 || post_rope {
            rope(&mut hasher, stack);
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

fn rope(hasher: &mut Sha256, stack: &Qwen2PagedStack) {
    let layer = &stack.config().layer;
    text(hasher, "rope");
    hasher.update(layer.rope_theta.to_bits().to_le_bytes());
    match &layer.rope_scaling {
        Qwen2RopeScaling::None => text(hasher, "scaling:none"),
        Qwen2RopeScaling::Unsupported(name) => {
            text(hasher, "scaling:unsupported");
            text(hasher, name);
        }
    }
    // Placeholder for the effective regime fixed at admission (doc 36); no scaling exists.
    text(hasher, "regime:none");
    hasher.update(((layer.hidden_size / layer.num_attention_heads) as u64).to_le_bytes());
}

fn embedding_digest(embedding: &[f32]) -> [u8; 32] {
    let mut hasher = start("embedding");
    floats(&mut hasher, embedding);
    hasher.finalize().into()
}

/// Canonical digest of ALL of one layer: both norms, Q/K/V/O, QK-norms, MLP and the
/// attention and numerics configuration that shapes its output.
fn full_layer_digest(stack: &Qwen2PagedStack, layer: &Qwen2StackLayerWeights) -> [u8; 32] {
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
    hasher.finalize().into()
}
