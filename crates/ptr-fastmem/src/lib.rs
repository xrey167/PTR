//! Fast-weight working memory: a gated delta-rule associative memory that is a
//! derived, revocable projection of lifecycle-managed semantic inputs.
//!
//! The memory holds `heads` matrices updated by the Kimi Delta Attention form of
//! the delta rule (`S <- (I - beta k k^T) Diag(alpha) S + beta k v^T`) and read by
//! `o = S^T q`. It is never authority: a readout is decoded — against
//! identifier codes of the facts actually written — into ordinary search
//! candidates, or into `Unknown` when no fact leads clearly. Every write names
//! the semantic input, generation and input digest it came from; a read is
//! admitted only while every one of them is still admissible, and revoking an
//! input removes its writes and refolds the matrices from the last earlier
//! checkpoint. Because the fold is deterministic `f32` arithmetic, the refolded
//! state is bit-identical to one that never saw the revoked writes. That is
//! exact revocation, not erasure: copies held by storage are the storage's to
//! erase.

mod codec;
mod config;
mod decode;
mod error;
mod memory;
mod projection;
mod state;
mod write;

pub use codec::{decode_state, encode_state, STATE_MAGIC};
pub use config::{
    check_config, FastMemoryConfig, MAX_CHECKPOINT_INTERVAL, MAX_HEADS, MAX_HEAD_DIM,
    MAX_STATE_CELLS, MAX_WRITES,
};
pub use decode::{decode_readout, DecodePolicy, FactCode, Recall, FASTMEM_BACKEND};
pub use error::FastMemoryError;
pub use memory::{binding_digest_of, FastMemory, RevocationReport, WriteReceipt};
pub use projection::{IdentifierCodebook, ProjectionSpec, SeededProjection};
pub use state::{FastWeightState, Readout};
pub use write::{validate_write, Decay, MemoryWrite, Query, SourceRef, WriteRequest, WriteSeq};
