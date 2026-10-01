//! Saving and loading A0 weights with the codebook they were trained under.
//!
//! Burn records parameters and nothing else — its constant fields are recorded as
//! empty — so a bare record cannot say which assignment its codes belong to. The
//! record therefore travels inside a [`CheckpointHeader`], and a build whose
//! codebook has moved refuses to load it rather than reading the weights under a
//! taxonomy they never saw.
use crate::{PtrA0, PtrA0Config, RouterMode, TypedAttentionMode};
use burn::{
    prelude::*,
    store::{ModuleRecord, RecordError},
    tensor::Bytes,
};
use ptr_types::{CheckpointError, CheckpointHeader, CodeFamily, TableSize};
use sha2::{Digest, Sha256};

/// Stable name recorded in the header. Not the crate version: what matters to a
/// reader is which model's parameter layout this is.
pub const MODEL: &str = "ptr-a0";

/// The families A0 embeds, and therefore the ones a checkpoint must agree about.
///
/// `ReasoningOperator` is included although it is not an embedding: it sizes the
/// router's output, so a checkpoint written under a different operator table would
/// produce logits that mean different things.
pub const EMBEDDED_FAMILIES: [CodeFamily; 3] = [
    CodeFamily::SemanticRole,
    CodeFamily::EpistemicState,
    CodeFamily::ReasoningOperator,
];

/// Why a checkpoint could not be written or read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckpointIoError {
    /// The header is malformed, or its identity disagrees with this build.
    Header(CheckpointError),
    /// The parameters did not fit this architecture, or the record is unreadable.
    Record(RecordError),
    /// The artifact belongs to another model, whose parameter layout is not this
    /// one's. Refused rather than attempted: burn would report missing tensors,
    /// but "this is not that model" is the more useful answer.
    WrongModel { found: String },
    /// The model-owned architecture binding disagrees with the requested config.
    ArchitectureMismatch,
    /// Format-2 artifacts have no architecture binding and require an explicit,
    /// exact legacy opt-in.
    LegacyArchitectureUnbound,
}

impl From<CheckpointError> for CheckpointIoError {
    /// Carry a header refusal through unchanged.
    fn from(error: CheckpointError) -> Self {
        Self::Header(error)
    }
}

impl From<RecordError> for CheckpointIoError {
    /// Carry a record failure through unchanged.
    fn from(error: RecordError) -> Self {
        Self::Record(error)
    }
}

impl core::fmt::Display for CheckpointIoError {
    /// Render a stable refusal code.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Header(error) => error.fmt(f),
            Self::Record(_) => f.write_str("PTR_A0_CKPT_RECORD"),
            Self::WrongModel { .. } => f.write_str("PTR_A0_CKPT_WRONG_MODEL"),
            Self::ArchitectureMismatch => f.write_str("PTR_A0_CKPT_ARCHITECTURE"),
            Self::LegacyArchitectureUnbound => f.write_str("PTR_A0_CKPT_LEGACY_UNBOUND"),
        }
    }
}

impl std::error::Error for CheckpointIoError {}

/// The identity header for a trained model.
///
/// The two embedding table sizes are read from the weights themselves rather than
/// from the config that built them, so the header cannot claim a width the tensors
/// do not have. The router's width is the count `init` used; a forward pass
/// asserts it in `tests/forward.rs`.
///
/// The slot-encoding version cannot be read from the weights, because the encoding
/// produces the model's *input* and no parameters. That is exactly why the header
/// records it: for every other identity in here a wrong value would eventually show
/// up as a shape, and for this one nothing would.
pub fn header(model: &PtrA0) -> CheckpointHeader {
    let book = model.codebook();
    CheckpointHeader::new(
        MODEL,
        &book,
        model.encoding(),
        &[
            TableSize {
                family: CodeFamily::SemanticRole,
                rows: model.slot_type_rows() as u16,
            },
            TableSize {
                family: CodeFamily::EpistemicState,
                rows: model.epistemic_rows() as u16,
            },
            TableSize {
                family: CodeFamily::ReasoningOperator,
                rows: model.operator_count() as u16,
            },
        ],
    )
    .with_architecture(architecture_digest(
        model.vocabulary_size(),
        model.provenance_rows(),
        model.slot_type_rows(),
        model.epistemic_rows(),
        model.operator_count(),
        model.latent_steps(),
        model.typed_attention_mode(),
        model.typed_attention_rank(),
        model.typed_attention_limit(),
        model.typed_query(),
        model.latent_nonlinearity(),
        model.router_mode(),
        model.router_logit_scale(),
    ))
}

pub(crate) fn config_architecture_digest(config: &PtrA0Config) -> Vec<u8> {
    architecture_digest(
        config.vocab_size,
        config.provenance_bucket_count,
        config.slot_type_count(),
        config.epistemic_count(),
        config.operator_count(),
        config.latent_steps,
        config.typed_attention,
        config.typed_attention_rank,
        config.typed_attention_limit,
        config.typed_query,
        config.latent_nonlinearity,
        config.router_mode,
        config.router_logit_scale,
    )
}

#[allow(clippy::too_many_arguments)]
fn architecture_digest(
    vocab_size: usize,
    provenance_buckets: usize,
    slot_types: usize,
    epistemic_states: usize,
    operators: usize,
    latent_steps: usize,
    attention: TypedAttentionMode,
    attention_rank: usize,
    attention_limit: f32,
    typed_query: bool,
    latent_nonlinearity: bool,
    router: RouterMode,
    router_scale: f32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64);
    bytes.extend_from_slice(b"ptr-a0-architecture-v1\0");
    for value in [
        vocab_size,
        provenance_buckets,
        slot_types,
        epistemic_states,
        operators,
        latent_steps,
        attention_rank,
    ] {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    bytes.push(attention as u8);
    bytes.extend_from_slice(&attention_limit.to_bits().to_le_bytes());
    bytes.push(u8::from(typed_query));
    bytes.push(u8::from(latent_nonlinearity));
    bytes.push(router as u8);
    bytes.extend_from_slice(&router_scale.to_bits().to_le_bytes());
    Sha256::digest(bytes).to_vec()
}

fn exact_legacy_config(config: &PtrA0Config) -> bool {
    let current = config_architecture_digest(config);
    config.legacy_v2_architecture.as_deref() == Some(current.as_slice())
        && config.typed_attention == TypedAttentionMode::LegacyScalarV1
        && config.router_mode == RouterMode::LegacyMeanLinear
        && config.typed_query
        && config.latent_nonlinearity
}

/// Serialize a model as a checkpoint: identity header, then its parameters.
pub fn save(model: &PtrA0) -> Result<Vec<u8>, CheckpointIoError> {
    let record = model.clone().into_record().into_bytes()?;
    Ok(header(model).write(&record))
}

/// Read a checkpoint into a model built from `config`.
///
/// Order matters. The identity is checked first, because if the assignment has
/// moved there is no point in loading tensors that would then mean the wrong
/// thing; the model-owned architecture digest is checked second, before the
/// record is decoded or any weight is accepted.
///
/// `frozen_router` remains a training-time optimizer choice and is not applied
/// on this path. Every forward-graph choice is bound by format 3. An unbound
/// format-2 artifact requires explicit opt-in to the exact legacy graph.
///
/// # Errors
///
/// Returns a header error for malformed or incompatible identity metadata,
/// `WrongModel` for another model's checkpoint, `ArchitectureMismatch` for a
/// different graph, `LegacyArchitectureUnbound` for an unapproved format-2
/// artifact, or a record error when the payload cannot be decoded.
pub fn load(
    bytes: &[u8],
    config: &PtrA0Config,
    device: &Device,
) -> Result<PtrA0, CheckpointIoError> {
    let (header, payload) = CheckpointHeader::read(bytes)?;
    if header.model != MODEL {
        return Err(CheckpointIoError::WrongModel {
            found: header.model,
        });
    }
    header.verify(&config.codebook(), config.encoding(), &EMBEDDED_FAMILIES)?;
    if header.architecture.is_empty() {
        if header.format() != ptr_types::FORMAT_V2 {
            return Err(CheckpointIoError::ArchitectureMismatch);
        }
        if !exact_legacy_config(config) {
            return Err(CheckpointIoError::LegacyArchitectureUnbound);
        }
    } else if header.architecture != config_architecture_digest(config) {
        return Err(CheckpointIoError::ArchitectureMismatch);
    }
    let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(payload.to_vec()))?;
    Ok(config.init_lazy(device).try_load_record(record)?)
}
