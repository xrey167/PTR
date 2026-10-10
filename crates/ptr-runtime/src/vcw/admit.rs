//! From a verified answer into the runtime's normal output admission.
//!
//! A [`VerifiedAnswer`] is provisional (invariant 4). It becomes runtime state only through
//! [`PtrRuntime::admit_pod_output`], the path every Pod output takes: scope and lease checks,
//! manifest and artifact binding, dependency and KV-taint checks, a verifier run on the typed
//! payload, and the admission record. [`admit_verified_answer`] adds no authority of its own; it
//! is a narrowing of that path. Before forwarding it checks that
//!
//! * the request is *for the answer the loop accepted*: same request id, same project (the
//!   project of the request's scope), same Pod and generation, same payload type and bytes;
//! * every Pod whose output is in the payload is **still live now** (the eligibility view is
//!   rebuilt here from the runtime, because a revocation, fence or supersession between `run`
//!   and the admission would otherwise be missed: `admit_pod_output` itself does not look at the
//!   Pod's lifecycle);
//!
//! and it consumes the answer, so one acceptance is one admission attempt.
//!
//! The `verifier` argument is the **runtime's own re-verification** of the typed payload inside
//! `admit_pod_output`. It is chosen by the caller and is independent of the verifier the loop
//! used; a permissive one weakens only this second check, never the loop's acceptance.

use std::fmt;

use ptr_events::RuntimeEvent;
use ptr_pods::PodRegistry;
use ptr_protocol::TypedPayload;
use ptr_router::PodEligibility;
use ptr_types::{Digest, Generation, PodId, RequestId};
use ptr_verifier::Verifier;

use super::{RuntimeStanding, VerifiedAnswer};
use crate::{PodOutputAdmission, PodOutputAdmissionRequest, PtrRuntime, RuntimeError};

/// Why an answer was not forwarded, or what the runtime said.
#[derive(Debug)]
#[non_exhaustive]
pub enum VcwAdmitError {
    /// The admission request is for another request id than the answer.
    RequestMismatch,
    /// The admission request's scope belongs to another project than the answer's.
    ProjectMismatch,
    /// The admission request names another Pod than the answer's author.
    PodMismatch,
    /// The output was produced at another generation than the author's.
    GenerationMismatch,
    /// The output's payload is not the payload the loop accepted.
    PayloadMismatch,
    /// A Pod whose output is in the payload is no longer live at its generation (revoked,
    /// superseded, fenced, retired or unknown to the lifecycle authority).
    AuthorNoLongerEligible { pod: PodId, generation: Generation },
    /// The runtime's own admission refused it (nothing was appended).
    Runtime(RuntimeError),
}

impl fmt::Display for VcwAdmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestMismatch => f.write_str("admission request is for another request"),
            Self::ProjectMismatch => f.write_str("admission scope belongs to another project"),
            Self::PodMismatch => f.write_str("admission request names another Pod"),
            Self::GenerationMismatch => f.write_str("output generation differs from the author's"),
            Self::PayloadMismatch => f.write_str("output payload is not the accepted payload"),
            Self::AuthorNoLongerEligible { pod, generation } => {
                write!(
                    f,
                    "Pod {pod} at generation {} is no longer live",
                    generation.0
                )
            }
            Self::Runtime(error) => write!(f, "runtime admission refused: {error:?}"),
        }
    }
}

impl std::error::Error for VcwAdmitError {}

/// Forward `request` to [`PtrRuntime::admit_pod_output`] if and only if it carries exactly the
/// payload of `answer`, from the Pod and generation that authored it, in the answer's project,
/// and every contributing Pod is still live. `pods` and `base_pods` rebuild the eligibility view
/// ([`RuntimeStanding`]) at this moment; `base_pods` are the non-expert Pods and their declared
/// generations.
///
/// # Errors
/// [`VcwAdmitError`]: a mismatch or a Pod that is no longer live (nothing reached the runtime's
/// admission) or the runtime's refusal.
pub fn admit_verified_answer<V>(
    runtime: &mut PtrRuntime,
    pods: &PodRegistry,
    base_pods: &[(PodId, Generation)],
    answer: VerifiedAnswer<TypedPayload>,
    request: PodOutputAdmissionRequest,
    verifier: &V,
) -> Result<PodOutputAdmission, VcwAdmitError>
where
    V: Verifier<TypedPayload> + ?Sized,
{
    if &request.request_id != answer.request_id() {
        return Err(VcwAdmitError::RequestMismatch);
    }
    if request.pod_id != answer.author().pod {
        return Err(VcwAdmitError::PodMismatch);
    }
    if request.output.generation != answer.author().generation {
        return Err(VcwAdmitError::GenerationMismatch);
    }
    let accepted = answer.payload();
    if request.output.payload.type_id != accepted.type_id
        || request.output.payload.bytes != accepted.bytes
    {
        return Err(VcwAdmitError::PayloadMismatch);
    }
    let scope_project = runtime
        .scopes()
        .get(&request.scope_id)
        .map_err(|error| VcwAdmitError::Runtime(RuntimeError::Scope(error)))?
        .project
        .clone();
    if &scope_project != answer.project() {
        return Err(VcwAdmitError::ProjectMismatch);
    }
    // Liveness now, from the runtime's own state, for every Pod whose output is in the payload.
    {
        let mut standing = RuntimeStanding::new(runtime, pods, answer.project());
        for (pod, generation) in base_pods {
            standing = standing.with_base_pod(pod.clone(), *generation);
        }
        for author in answer.authors() {
            if standing.current_generation(&author.pod) != Some(author.generation) {
                return Err(VcwAdmitError::AuthorNoLongerEligible {
                    pod: author.pod.clone(),
                    generation: author.generation,
                });
            }
        }
    }
    runtime
        .admit_pod_output(request, verifier)
        .map_err(VcwAdmitError::Runtime)
}

impl PtrRuntime {
    /// Record, as a [`RuntimeEvent::VcwFinished`], how a verifier-closed request ended and the
    /// digest of its trace. Observability only: nothing is committed and no state changes; an
    /// accepted answer reaches the ledger through [`admit_verified_answer`], and an exhausted or
    /// escalated request leaves no committed answer at all.
    pub fn record_vcw_finished(
        &mut self,
        request: &RequestId,
        outcome: &str,
        trace_digest: Digest,
    ) {
        self.emit(RuntimeEvent::VcwFinished {
            request: request.clone(),
            outcome: outcome.to_owned(),
            trace_digest,
        });
    }

    /// Record, as a [`RuntimeEvent::VcwPolicyChanged`], that a host promoted or rolled back a
    /// routing policy. Observability of a host action; nothing is committed and no state
    /// changes here (the change itself is `ptr_router::PolicyBook::promote` or `rollback`,
    /// which call this through the [`ptr_router::PolicySink`] impl below, so a change cannot be
    /// made without the record).
    ///
    /// This is not an audit record: the event is in memory, lost on restart, and replay does
    /// not check that a promotion had a gate record. Until a ledger event exists, do not
    /// promote on this record alone in production.
    pub fn record_vcw_policy_event(&mut self, event: &ptr_router::PolicyEvent) {
        self.emit(RuntimeEvent::VcwPolicyChanged {
            scope: event.scope.clone(),
            sequence: event.sequence,
            action: match event.action {
                ptr_router::PolicyAction::Promoted => "promoted",
                ptr_router::PolicyAction::RolledBack => "rolled_back",
            }
            .to_owned(),
            from: event.from,
            to: event.to,
            evidence: event.evidence,
        });
    }

    /// Record, as a [`RuntimeEvent::VcwShadowEstimate`], an off-policy estimate of a candidate
    /// policy a loop learned in shadow mode. Observability only.
    pub fn record_vcw_shadow_estimate(
        &mut self,
        scope: &str,
        candidate: Digest,
        records: u64,
        difference: f64,
        lower_bound: f64,
    ) {
        self.emit(RuntimeEvent::VcwShadowEstimate {
            scope: scope.to_owned(),
            candidate,
            records,
            difference,
            lower_bound,
        });
    }
}

/// The runtime is the sink a host passes to `PolicyBook::promote` and `rollback`: every change
/// of routing policy is emitted as a [`RuntimeEvent::VcwPolicyChanged`] before the call returns.
impl ptr_router::PolicySink for PtrRuntime {
    fn record(&mut self, event: &ptr_router::PolicyEvent) {
        self.record_vcw_policy_event(event);
    }
}
