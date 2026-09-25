//! Adapter lineage for continual specialisation: identity, gated lifecycle,
//! subspace interference, forgetting-driven replay and consolidation.
//!
//! An adapter is a sealed, content-addressed checkpoint bound to one exact base
//! model and one domain; agents differ by context and memory, not by weights.
//! The in-memory [`Lineage`] is a working model, not authority: until adapter
//! promotion and revocation are committed ledger events and serving is
//! admitted through checkpoint binding, which adapter serves must not be
//! decided by any registry row. It serves only after a gate measured on
//! held-out and public suites, and revoking a training input names every
//! adapter whose weights depend on it. Interference with earlier adapters is measured as
//! principal-angle overlap between update subspaces, layer by layer; when a
//! lineage grows too deep or too entangled, the next step is a consolidation
//! (a TIES merge) rather than another link. Replay samples are scheduled by a
//! forgetting model fitted from probe losses on a training clock, and sampled
//! stratified and without replacement.

mod error;
mod forgetting;
mod lineage;
mod merge;
mod replay;
mod subspace;

pub use error::LineageError;
pub use forgetting::{AccuracyMatrix, ForgettingGate, GateReport, GateViolation, PublicSuite};
pub use lineage::{
    AdapterId, AdapterRecord, AdapterStatus, BaseModel, ConsolidationPolicy, Lineage, Origin,
};
pub use merge::ties_merge;
pub use replay::{
    retrievability, MemoryState, ModelTime, NewSample, ReplayParams, ReplayPool, ReplaySample,
    Split,
};
pub use subspace::{
    activation_interference, chance_overlap, column_basis, measure_interference, principal_cosines,
    subspace_overlap, Basis, InterferenceReport, LayerInterference, LayerUpdate, Matrix,
};
