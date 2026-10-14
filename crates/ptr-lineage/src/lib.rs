//! Adapter lineage for continual specialisation: identity, gated lifecycle,
//! subspace interference, forgetting-driven replay and consolidation.
//!
//! An adapter is a sealed, content-addressed checkpoint bound to one exact base
//! model and one domain; agents differ by context and memory, not by weights.
//! The in-memory [`Lineage`] is a working model, not authority: which adapter
//! serves is decided by the committed `AdapterPromoted` and `AdapterRevoked`
//! ledger events that `ptr-runtime` writes from a [`PromotionProposal`] (this
//! crate does not depend on the ledger), never by a registry row. It serves only after a gate measured on
//! held-out and public suites, and revoking a training input names every
//! adapter whose weights depend on it through the data manifests registered
//! with the lineage, which it records as declared and cannot check against
//! the weights. Interference with earlier adapters is measured as
//! principal-angle overlap between update subspaces, layer by layer; when a
//! lineage grows too deep or too entangled, the next step is a consolidation
//! (a TIES merge) rather than another link. Replay samples are scheduled by a
//! forgetting model fitted from probe losses on a training clock, and sampled
//! stratified and without replacement. Experts of a layer can be ranked for
//! pruning by the calibration-free AIMER score of their weights; that only
//! orders them, and a prune still has to pass the gate.

mod aimer;
mod error;
mod forgetting;
mod lineage;
mod merge;
mod promotion;
mod replay;
mod scale;
mod subspace;

pub use aimer::{aimer_score, rank_experts_for_pruning, ExpertWeights};
pub use error::LineageError;
pub use forgetting::{
    AccuracyMatrix, BoundGateReport, ForgettingGate, GateReport, GateSubject, GateViolation,
    PromotionSuite, PublicSuite, MAX_BOUND_GATE_REPORT_BYTES,
};
pub use lineage::{
    AdapterId, AdapterRecord, AdapterStatus, BaseModel, ConsolidationPolicy, Lineage, Origin,
};
pub use merge::ties_merge;
pub use promotion::{promotion_operation_id, PromotionProposal, SplitLock};
pub use replay::{
    retrievability, MemoryState, ModelTime, NewSample, ReplayParams, ReplayPool, ReplaySample,
    Split,
};
pub use subspace::{
    activation_interference, chance_overlap, column_basis, measure_interference, principal_cosines,
    subspace_overlap, Basis, InterferenceReport, LayerInterference, LayerUpdate, Matrix,
};
