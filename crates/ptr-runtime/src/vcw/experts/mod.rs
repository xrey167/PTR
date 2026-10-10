//! Expert Pods for the verifier-closed loop: real text generators and seeded programmatic
//! baselines, both behind [`RolePods`](super::RolePods).
//!
//! ```text
//!  VcwLoop::run ----ProposeCall / RepairCall----> RolePods adapter
//!                                                    |
//!            +---------------------------------------+---------------------------+
//!            | LlmExpertPods                                    SimulatedExperts |
//!            |  prompt (system per Pod, task, one finding code)  seeded error rate|
//!            |  -> ptr_pods::TextGenerator (bounded, greedy)      per finding code|
//!            |  -> extract answer -> TextPayload                  repair success  |
//!            +------------------------------------------------------ per code ----+
//!                                |                                        |
//!                     LlmExpertPod (DynPod, manifest)          SimulatedPod (DynPod, manifest)
//!                                  \__________ registered in the PodRegistry ___/
//! ```
//!
//! # What these adapters do and do not decide
//!
//! * The loop decides which Pod is asked, from the eligibility view at route time. The adapters
//!   re-check at **call time**: [`Guarded`] binds an adapter to a [`PodEligibility`] (the view the
//!   loop itself uses) and a call whose Pod is no longer at the generation the loop routed to,
//!   or is unknown or revoked, is declined before the Pod's generator runs (and, for a
//!   model-backed Pod, once more after the shared generator's lock is taken, since waiting for
//!   the lock can be long). The decline is recorded as
//!   [`CallOutcome::IneligibleAtCallTime`]. The adapters implement no [`RolePods`](super::RolePods)
//!   themselves, so an adapter cannot be handed to the loop without that check.
//! * A repair Pod sees the previous candidate and **one** finding code with a short, fixed
//!   description. It never sees the verifier, its report or its score. Whether the edit is
//!   good is the verifier's verdict, through `classify`, as for any Pod.
//! * A proposer sees the task and, after round 0, only an attempt number. A greedy model given
//!   the same message repeats its text, so the attempt number is a perturbation that may make a
//!   regenerated proposal differ; it does not guarantee that it does, and it carries no finding.
//! * Nothing here accepts, admits or commits. The text a model produced is only a candidate.
//! * The model backend stays behind the [`ptr_pods::TextGenerator`] trait; this module names no
//!   backend type. A generator error is not a "no answer": it is recorded in the call log as
//!   [`CallOutcome::Failed`] with a typed [`ExpertErrorKind`].
//!
//! The programmatic experts ([`SimulatedExperts`]) are the deterministic baseline for tests and
//! for experiments on routing: every draw is keyed by the seed, the request, the Pod, the role,
//! the finding code and the occurrence index of that tuple within the request, so a run is
//! replayable bit for bit and paired arms get identical draws for identical tuples, whatever
//! else happens in between.

mod guard;
mod llm;
mod manifest;
mod simulated;
mod task;

use std::fmt;
use std::sync::Arc;

use ptr_pods::ModelIdentity;
use ptr_router::PodEligibility;
use ptr_types::{Generation, PodId};

pub use guard::Guarded;
pub use llm::{ExpertError, LlmExpertPod, LlmExpertPods, SharedGenerator};
pub use manifest::{expert_manifest, expert_manifest_for_model, ExpertRole};
pub use simulated::{
    DefectDoc, DefectDocVerifier, DefectPayload, SimulatedExperts, SimulatedPod, SimulationConfig,
    SimulationError,
};
pub use task::{
    ExpertTask, TaskKind, TextPayload, ARITHMETIC_ANSWER_TYPE, JSON_OBJECT_ANSWER_TYPE,
};

/// Most calls an adapter remembers; older calls are dropped.
pub const MAX_LOGGED_CALLS: usize = 4096;

/// Why a generator call failed, without the backend's message or types.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ExpertErrorKind {
    /// The request was outside the generator's bounds.
    InvalidRequest,
    /// The backend failed (a non-finite logit, a prompt over the limit, ...).
    Backend,
    /// The shared generator's lock was poisoned by a panic in an earlier call.
    LockPoisoned,
}

/// What became of a call to an expert Pod.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum CallOutcome {
    /// The Pod produced a candidate.
    Produced,
    /// The Pod ran and gave no answer (a configured decline).
    Declined,
    /// The Pod was not eligible at the generation routed to when the call was made; its
    /// generator was not invoked.
    IneligibleAtCallTime,
    /// The Pod's generator failed; nothing was produced. This is a fault, not a skill result.
    Failed(ExpertErrorKind),
}

impl fmt::Display for CallOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Produced => f.write_str("produced"),
            Self::Declined => f.write_str("declined"),
            Self::IneligibleAtCallTime => f.write_str("ineligible at call time"),
            Self::Failed(kind) => write!(f, "failed: {kind:?}"),
        }
    }
}

/// A Pod id was added twice to one adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RosterError {
    /// The id is already held, as proposer or repairer.
    DuplicatePod(PodId),
}

impl fmt::Display for RosterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicatePod(pod) => write!(f, "Pod {} is already in the roster", pod.0),
        }
    }
}

impl std::error::Error for RosterError {}

/// One call an adapter answered (or refused), for audit and tests.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ExpertCall {
    pub role: ExpertRole,
    pub pod: PodId,
    pub generation: Generation,
    pub round: u32,
    /// The finding code of a repair call.
    pub code: Option<String>,
    /// Whether the Pod produced a candidate (`outcome == Produced`).
    pub produced: bool,
    /// What became of the call; a decline, an ineligible Pod and a generator error differ.
    pub outcome: CallOutcome,
    /// Tokens generated (zero for a programmatic Pod).
    pub new_tokens: u32,
    /// The candidate text the Pod produced, cut to [`MAX_LOGGED_ANSWER_BYTES`] (empty for a
    /// programmatic Pod, whose candidate is a structured value).
    pub answer: String,
    /// The digests of the model files the Pod runs, when its generator reports them.
    pub model: Option<Arc<ModelIdentity>>,
}

/// What an adapter knows about one call, to build its [`ExpertCall`].
pub(crate) struct CallRecord<'a> {
    pub role: ExpertRole,
    pub pod: &'a ptr_router::PodRef,
    pub round: u32,
    pub code: Option<&'a str>,
    pub outcome: CallOutcome,
    pub new_tokens: u32,
    pub answer: String,
    pub model: Option<Arc<ModelIdentity>>,
}

impl CallRecord<'_> {
    pub(crate) fn into_call(self) -> ExpertCall {
        ExpertCall {
            role: self.role,
            pod: self.pod.pod.clone(),
            generation: self.pod.generation,
            round: self.round,
            code: self.code.map(str::to_owned),
            produced: self.outcome == CallOutcome::Produced,
            outcome: self.outcome,
            new_tokens: self.new_tokens,
            answer: self.answer,
            model: self.model,
        }
    }
}

/// Whether `standing` still lists `pod` at the generation the call was routed to.
pub(crate) fn still_eligible(standing: &dyn PodEligibility, pod: &ptr_router::PodRef) -> bool {
    standing.current_generation(&pod.pod) == Some(pod.generation)
}

/// Longest candidate text kept in an [`ExpertCall`].
pub const MAX_LOGGED_ANSWER_BYTES: usize = 256;

/// `text` cut at a character boundary to at most [`MAX_LOGGED_ANSWER_BYTES`] bytes.
fn clipped(text: &str) -> String {
    let mut end = text.len().min(MAX_LOGGED_ANSWER_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}
