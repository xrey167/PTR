//! Saving and loading A0 weights with the codebook they were trained under.
//!
//! Burn records parameters and nothing else — its constant fields are recorded as
//! empty — so a bare record cannot say which assignment its codes belong to. The
//! record therefore travels inside a [`CheckpointHeader`], and a build whose
//! codebook has moved refuses to load it rather than reading the weights under a
//! taxonomy they never saw.
use crate::{PtrA0, PtrA0Config};
use burn::{
    prelude::*,
    store::{ModuleRecord, RecordError},
    tensor::Bytes,
};
use ptr_types::{CheckpointError, CheckpointHeader, CodeFamily, TableSize};

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
/// thing; the architecture is checked second, by the record itself refusing a
/// shape that does not fit.
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
    let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(payload.to_vec()))?;
    Ok(config.init(device).try_load_record(record)?)
}
