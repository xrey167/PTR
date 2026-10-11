//! The verifier-closed workspace loop (VCW-MoE), composed from its parts as a new runtime path.
//!
//! This module composes five merged components and adds no authority to any of them:
//!
//! * the bounded, verifier-gated [`ptr_pods::Workspace`] (package A),
//! * the finding router with propensity logging, [`ptr_router::FindingRouter`] (package B),
//! * the self-model and the attention schema (package C), observing and, at most, scoring inside
//!   the admissible set,
//! * the L3 structure policy and expert admission, [`crate::expert`] (package E),
//! * the critique, exhaustion and accept semantics of `ptr-feedback`'s repair loop.
//!
//! The existing run paths (`run_resumable_with_pods`, `run_model_with_pods`, `SteppedRun`) and
//! their signatures are unchanged; this is a separate entry point, [`VcwLoop::run`].
//!
//! # One request
//!
//! ```text
//!  budget (calls, rounds)
//!        |
//!  route a proposer  --> Pod (host RolePods) --> candidate x
//!        ^                                           |
//!        |                       verify once  (host Verifier<P>, one call, charged)
//!        |                                           |
//!        |                         classify(report, required)  <- the only Accept
//!        |                                           |
//!        |        workspace.submit + run_cycle(cached report, Pod registry view as authors)
//!        |        snapshot broadcast; self-model predicts then learns; schema observes/settles
//!        |                                           |
//!   Accept ---> VerifiedAnswer (provisional) --> admit_verified_answer --> admit_pod_output
//!   Repair(codes) -> router -> one repair Pod per code (named code only) -> next candidate
//!   Fallback      -> FallbackPolicy: Regenerate (fresh proposer) | Escalate (host target)
//!   budget spent  -> Exhausted        (never the best failed candidate)
//! ```
//!
//! # Learning and structure
//!
//! A [`VcwLoop`] can be given a learner ([`VcwLoop::with_learner`]) that is told the verifier's
//! verdict on every routed Pod. The loop only reports; routing changes when a host promotes a
//! snapshot through `ptr_router::PolicyBook`, which nothing in this module can reach. The
//! host-side driver that completes the structure policy's requests, gates them and admits them
//! lives in [`crate::vcw_host`], outside this module on purpose: this module has no handle on the
//! runtime's admission, grants or gates.
//!
//! Across requests, per window: the tally of routed/cleared/unroutable becomes a
//! `StructureObservation`; [`VcwLoop::structure_step`] returns the policy's *requests* to the
//! host. The host trains, binds a gate report to the completed proposal and calls
//! `PtrRuntime::admit_expert_proposal`; [`VcwLoop::sync_roster`] then enrols the committed expert
//! for its codes, and the next request's eligibility view lists it.
//!
//! # Closure constraints
//!
//! * **C1/C2 no path accepts without `classify`.** `Accept` is built only by
//!   `ptr_router::classify` (a sealed proof) and the loop additionally cross-checks it with
//!   `ptr_feedback::Candidate::passes`; scores, hints, bids and workspace membership are not
//!   arguments. A forged `Pass` (hard finding), a `Pass` below the required level, `Unknown`
//!   and `Disputed` are never accepted. The accepted author must also still be eligible.
//! * **C3 bounded.** Budgets are typed ([`RequestBudget`]) and metered; the workspace,
//!   router counters, self-model tables, trace, windows and outstanding requests are bounded.
//! * **C4 provisional.** A [`VerifiedAnswer`] is not authoritative; the workspace has no ledger
//!   dependency. Commit is the existing admission path.
//! * **C5 propensities.** Every sampled routing record is in the trace with its support.
//! * **C7 Exhausted.** [`VcwOutcome`] has no variant carrying a rejected candidate.
//! * **C8 Unknown/Disputed** are never repair targets (`classify`).
//! * **C9 structure only through admission.** Nothing in this module calls
//!   `admit_expert_proposal`, `fence_expert`, `readmit_expert` or `commit`; the one runtime
//!   mutation is `admit_pod_output` in [`admit_verified_answer`], which first re-checks that every
//!   contributing Pod is live now and that the request is for the accepted answer and project.
//! * **C10 replay.** The trace digest is a function of the seeds, the Pods' answers and the
//!   verifier's reports.
//!
//! # The workspace does not decide acceptance
//!
//! Acceptance is the verifier's verdict through `classify`, plus the liveness of every Pod whose
//! output is in the candidate. What the workspace does with the candidate (admitted, outbid,
//! duplicate, slot cap) is integration and broadcast, not a gate: a refusal for a reason other
//! than authority neither blocks acceptance nor is hidden, it is in the trace
//! ([`WorkspaceOutcome`]) and in [`VerifiedAnswer::workspace`]. An author the registry no longer
//! lists is refused by the workspace *and* fails the liveness check, so it cannot be accepted.
//!
//! # Escalation
//!
//! Every escalation, whichever fallback decided it (a policy action on a report, no eligible
//! proposer, a declined repair under an escalating `FailWithoutHardFinding`), is an attempt
//! that calls [`RolePods::escalate`]. Only an escalation nobody answers ends the request as
//! `Escalated`.
//!
//! # Guards, not proofs
//!
//! The source-scan test is a guard against casual regressions (substring matching can be
//! bypassed by an alias); the signature test and the types are what bound the module.
//!
//! # Not built
//!
//! A second, stronger *verifier* for an escalated candidate (it would be a second verdict on one
//! candidate); escalation goes to a Pod or a human through [`RolePods::escalate`]. The first
//! proposal is drawn through a regenerate route of an `Unknown` report because the router has no
//! proposer-class API; the class label of that record is therefore `Regenerate(Unknown)` at
//! round 0.

mod admit;
mod budget;
mod closed_loop;
mod eligibility;
pub mod experts;
mod l3;
#[cfg(test)]
mod test_support;
mod trace;

use std::fmt;

pub use admit::{admit_verified_answer, VcwAdmitError};
pub use budget::{
    BudgetError, BudgetUse, ExhaustionReason, RequestBudget, RoundBudget, VerifierCallBudget,
    MAX_ROUNDS, MAX_VERIFIER_CALLS,
};
pub use closed_loop::{
    EscalateCall, HintMode, ProposeCall, RepairCall, RolePods, Roster, VcwConfig, VcwCounters,
    VcwLoop, VcwOutcome, VcwParts, VcwPayload, VcwRequest, VcwRun, VerifiedAnswer,
};
pub(crate) use eligibility::state_is_routable;
pub use eligibility::{AuthorStanding, RuntimeStanding};
pub use l3::{RosterSync, StructureHost};
pub use trace::{
    CandidateTrace, ClassSummary, OutcomeTag, ReportSummary, SelfStepTrace, StepDecision,
    StepOrigin, TraceStep, VcwTrace, WorkspaceOutcome, MAX_TRACE_CODES, MAX_TRACE_CODE_BYTES,
};

use ptr_pods::{WorkspaceConfigError, WorkspaceError};
use ptr_router::{FindingRouteError, RegistryError};
use ptr_types::PodId;

use crate::expert::StructureObservationError;

/// Why the loop itself failed. A request that finds no answer is not an error: it ends
/// `Exhausted` or `Escalated`.
#[derive(Debug)]
#[non_exhaustive]
pub enum VcwError {
    Config(&'static str),
    WorkspaceConfig(WorkspaceConfigError),
    Workspace(WorkspaceError),
    BadVerifierIdentity,
    Registry(RegistryError),
    Route(FindingRouteError),
    Structure(StructureObservationError),
    /// No observation window has closed yet.
    NoClosedWindow,
    /// The requests would not fit the outstanding table; nothing was recorded.
    OutstandingFull,
    /// The operation was not handed out by this loop (or is already resolved).
    UnknownOperation,
    /// An expert the request created is not committed in a routable state.
    NotCommitted {
        pod: PodId,
    },
    /// `classify` and `Candidate::passes` disagree. This cannot happen unless one of the two
    /// restatements of the accept predicate was changed; the request fails closed.
    GateDisagreement,
    /// An invariant of the loop's own state broke.
    Internal(&'static str),
}

impl fmt::Display for VcwError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(why) => write!(f, "invalid configuration: {why}"),
            Self::WorkspaceConfig(error) => write!(f, "workspace configuration: {error}"),
            Self::Workspace(error) => write!(f, "workspace: {error}"),
            Self::BadVerifierIdentity => f.write_str("verifier identity is blank or too long"),
            Self::Registry(error) => write!(f, "finding registry: {error}"),
            Self::Route(error) => write!(f, "routing: {error}"),
            Self::Structure(error) => write!(f, "structure observation: {error:?}"),
            Self::NoClosedWindow => f.write_str("no observation window has closed"),
            Self::OutstandingFull => f.write_str("too many unresolved structural requests"),
            Self::UnknownOperation => f.write_str("operation was not handed out by this loop"),
            Self::NotCommitted { pod } => write!(f, "expert {pod} is not committed and routable"),
            Self::GateDisagreement => f.write_str("the accept predicate restatements disagree"),
            Self::Internal(why) => write!(f, "internal: {why}"),
        }
    }
}

impl std::error::Error for VcwError {}
