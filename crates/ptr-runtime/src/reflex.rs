//! Reflex path: failover to an explicit default pod, and honest counters.
//!
//! Every Pod request goes through `dispatch_pod`. The primary path (the router, or
//! the plain deterministic selector) either yields a pod that answers, or it misses: no
//! candidate, an abstention, no learned representation, or a pod that fails when invoked. After a
//! miss the runtime consults the [`DefaultPodPlan`] the host configured for the request's project
//! and capability. A plan is host configuration, not history: it is not a ledger record, a
//! runtime rebuilt from its log has none, and with no plan a miss fails the request exactly as it
//! did before this module existed.
//!
//! # The default pod is a joint, not a shortcut
//!
//! The failover pod is held to everything a routed pod is held to, and checked again at every
//! use, not once at configuration:
//!
//! - it is a registry candidate of the request's project, capability and input type (so the
//!   project boundary and input typing hold);
//! - its generation is the one the lifecycle authority holds live for it, and not revoked
//!   (invariant 3). The plan names a [`PodRevisionAddress`]; a revocation or a newer live
//!   generation committed after the plan was set stops the failover with the very next request;
//! - the registry holds exactly the implementation that revision names (the same namespace and
//!   generation, `PodRegistry::register_at`): a registry whose default is another generation, or
//!   was registered without one, is not the code the lifecycle check was about;
//! - its manifest declares no effect beyond pure or read (invariant 7);
//! - its output goes through [`PtrRuntime::promote_verified_pod_output`], the same verifier path
//!   as any pod output, so a deterministic contradiction is never promoted (invariants 7 and 11).
//!   A learned score chose nothing here: the reflex only substitutes a pod, the verifier
//!   commits.
//!
//! A failover is not a sampled decision, so it logs no `PodRouted` propensity.
//!
//! # Counters
//!
//! The [`ReflexCounters`] are in-memory telemetry. They are not authoritative, nothing in the
//! runtime reads them to decide anything, and a restarted runtime starts at zero. A miss is
//! counted when it happens and settles as a failover (the default pod's output was promoted) or
//! as an uncovered miss with its reason; a path that ends without settling leaves the miss
//! open, which can only make the counters look worse, never better. Every request is timed,
//! failures included.

use crate::routing::{PodSelection, Unselected};
use crate::{PtrRuntime, RuntimeError};
use ptr_events::RuntimeEvent;
use ptr_model_api::KvReuseTaint;
use ptr_pods::{DynPod, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    DefaultPodPlan, LatencySample, MissReason, ReflexCounters, ReflexOutcome, UncoveredReason,
};
use ptr_types::{
    CapabilityId, Effect, PodId, PodRevisionAddress, ProjectId, RequestId, Revision, TypeId,
};
use ptr_verifier::Verifier;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

/// Most fallback plans a runtime holds (invariant 8). A plan for a new `(project, capability)` is
/// refused beyond it; replacing an existing plan always fits.
pub const MAX_FALLBACK_PLANS: usize = 1024;

/// A request to set the fallback plan for `capability` in `project`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SetFallbackPlan {
    pub project: ProjectId,
    pub capability: CapabilityId,
    pub plan: DefaultPodPlan,
    /// The exact revision of the plan's default pod, whose generation is checked against the
    /// lifecycle authority at every failover. Required when the plan names a default pod, and
    /// refused when it names none.
    pub default_revision: Option<PodRevisionAddress>,
}

/// Why a fallback plan was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FallbackConfigError {
    /// The plan names a default pod but no revision to check its generation against.
    MissingDefaultRevision { pod: PodId },
    /// A revision was given for a plan that names no default pod.
    UnexpectedDefaultRevision,
    /// The revision is of another pod than the plan's default.
    PodMismatch { plan: PodId, revision: PodId },
    /// The revision belongs to another project than the plan.
    ProjectMismatch {
        expected: ProjectId,
        actual: ProjectId,
    },
    /// The revision is malformed.
    InvalidRevision,
    /// The runtime holds [`MAX_FALLBACK_PLANS`] plans already.
    TooManyPlans { limit: usize },
}

impl fmt::Display for FallbackConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDefaultRevision { pod } => {
                write!(
                    f,
                    "fallback plan names default pod {pod:?} without a revision"
                )
            }
            Self::UnexpectedDefaultRevision => {
                f.write_str("fallback plan names no default pod but a revision was given")
            }
            Self::PodMismatch { plan, revision } => write!(
                f,
                "fallback plan default {plan:?} does not match revision of pod {revision:?}"
            ),
            Self::ProjectMismatch { expected, actual } => write!(
                f,
                "fallback revision belongs to project {actual:?}, plan is for {expected:?}"
            ),
            Self::InvalidRevision => f.write_str("fallback default revision is malformed"),
            Self::TooManyPlans { limit } => write!(f, "at most {limit} fallback plans"),
        }
    }
}

impl std::error::Error for FallbackConfigError {}

#[derive(Clone, Debug)]
struct Binding {
    plan: DefaultPodPlan,
    revision: Option<PodRevisionAddress>,
}

/// The fallback plans and counters of a runtime. Neither is history.
#[derive(Clone, Debug, Default)]
pub(crate) struct ReflexState {
    plans: BTreeMap<(ProjectId, CapabilityId), Binding>,
    counters: ReflexCounters,
}

/// One Pod request as the reflex path sees it.
#[derive(Clone, Copy)]
pub(crate) struct PodRequest<'a> {
    pub(crate) pods: &'a PodRegistry,
    pub(crate) project: &'a ProjectId,
    pub(crate) request_id: &'a RequestId,
    pub(crate) capability: &'a CapabilityId,
    pub(crate) input_type: &'a TypeId,
    pub(crate) payload: &'a [u8],
}

/// A Pod whose output was promoted.
pub(crate) struct DispatchedPod {
    pub(crate) pod: Arc<dyn DynPod>,
    pub(crate) output: TypedPayload,
    pub(crate) revision: Revision,
}

fn effect_allowed(pod: &dyn DynPod) -> bool {
    pod.manifest()
        .effects
        .iter()
        .all(|effect| matches!(effect, Effect::Pure | Effect::Read))
}

impl PtrRuntime {
    /// Sets (or replaces) the fallback plan for `request.capability` in `request.project`. Host
    /// configuration: not a ledger record, not replayed.
    ///
    /// # Errors
    /// [`RuntimeError::FallbackConfig`] when the plan and its revision disagree, the revision is
    /// malformed, or the runtime holds [`MAX_FALLBACK_PLANS`] plans already.
    pub fn set_fallback_plan(&mut self, request: SetFallbackPlan) -> Result<(), RuntimeError> {
        let SetFallbackPlan {
            project,
            capability,
            plan,
            default_revision,
        } = request;
        let refuse = |error| Err(RuntimeError::FallbackConfig(error));
        match (plan.default_pod(), default_revision.as_ref()) {
            (None, None) => {}
            (None, Some(_)) => return refuse(FallbackConfigError::UnexpectedDefaultRevision),
            (Some(pod), None) => {
                return refuse(FallbackConfigError::MissingDefaultRevision { pod: pod.clone() })
            }
            (Some(pod), Some(revision)) => {
                if revision.validate().is_err() {
                    return refuse(FallbackConfigError::InvalidRevision);
                }
                if revision.address.project != project {
                    return refuse(FallbackConfigError::ProjectMismatch {
                        expected: project,
                        actual: revision.address.project.clone(),
                    });
                }
                if revision.address.pod_id != *pod {
                    return refuse(FallbackConfigError::PodMismatch {
                        plan: pod.clone(),
                        revision: revision.address.pod_id.clone(),
                    });
                }
            }
        }
        let key = (project, capability);
        if !self.reflex.plans.contains_key(&key) && self.reflex.plans.len() >= MAX_FALLBACK_PLANS {
            return refuse(FallbackConfigError::TooManyPlans {
                limit: MAX_FALLBACK_PLANS,
            });
        }
        self.reflex.plans.insert(
            key,
            Binding {
                plan,
                revision: default_revision,
            },
        );
        Ok(())
    }

    /// Removes the plan for `capability` in `project`; whether there was one.
    pub fn clear_fallback_plan(&mut self, project: &ProjectId, capability: &CapabilityId) -> bool {
        self.reflex
            .plans
            .remove(&(project.clone(), capability.clone()))
            .is_some()
    }

    /// The plan for `capability` in `project`, if one is set.
    pub fn fallback_plan(
        &self,
        project: &ProjectId,
        capability: &CapabilityId,
    ) -> Option<&DefaultPodPlan> {
        self.reflex
            .plans
            .get(&(project.clone(), capability.clone()))
            .map(|binding| &binding.plan)
    }

    /// A snapshot of the reflex counters. Telemetry: not authoritative, not persisted, and never
    /// read by the runtime to decide anything.
    pub fn reflex_counters(&self) -> ReflexCounters {
        self.reflex.counters.clone()
    }

    /// Routes, invokes and promotes the Pod for one Pod request, failing over to the plan's
    /// default pod on a miss. `input_taint` is the taint of the request's input, recorded
    /// before the first pod is invoked (`None` when the caller recorded it already).
    pub(crate) fn dispatch_pod<V>(
        &mut self,
        selection: &PodSelection<'_>,
        request: &PodRequest<'_>,
        input_taint: Option<KvReuseTaint>,
        verifier: &V,
    ) -> Result<DispatchedPod, RuntimeError>
    where
        V: Verifier<TypedPayload> + ?Sized,
    {
        let started = Instant::now();
        let result = self.dispatch_pod_untimed(selection, request, input_taint, verifier);
        // Failures are timed like successes: a path that fails more must not look faster.
        self.reflex.counters.record_latency(LatencySample {
            micros: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
            served: result.is_ok(),
        });
        result
    }

    fn dispatch_pod_untimed<V>(
        &mut self,
        selection: &PodSelection<'_>,
        request: &PodRequest<'_>,
        input_taint: Option<KvReuseTaint>,
        verifier: &V,
    ) -> Result<DispatchedPod, RuntimeError>
    where
        V: Verifier<TypedPayload> + ?Sized,
    {
        match self.classify_pod_selection(selection, request) {
            Ok(pod) => self.dispatch_primary(pod, request, input_taint, verifier),
            Err(Unselected {
                error,
                miss: Some(reason),
            }) => {
                self.record_miss(reason);
                self.fail_over(
                    request,
                    FailedPrimary {
                        pod: None,
                        attempts: 0,
                        taint_recorded: false,
                        error,
                    },
                    reason,
                    input_taint,
                    verifier,
                )
            }
            // A router failure that is not a miss is never papered over by a fallback.
            Err(Unselected { error, miss: None }) => Err(error),
        }
    }

    fn dispatch_primary<V>(
        &mut self,
        pod: Arc<dyn DynPod>,
        request: &PodRequest<'_>,
        input_taint: Option<KvReuseTaint>,
        verifier: &V,
    ) -> Result<DispatchedPod, RuntimeError>
    where
        V: Verifier<TypedPayload> + ?Sized,
    {
        if !effect_allowed(pod.as_ref()) {
            return Err(RuntimeError::PodEffectRequiresActionBoundary {
                pod: pod.manifest().id.to_string(),
            });
        }
        // Approximate input: the Request subject is recorded before any Pod output of this
        // request is promoted, so promotion (which checks the request subject) refuses it.
        if let Some(taint) = input_taint {
            self.record_input_request_taint(request.request_id, taint)?;
        }
        self.emit(RuntimeEvent::PodInvoked(pod.manifest().id.to_string()));
        let invoked = pod.invoke(TypedPayload {
            type_id: request.input_type.clone(),
            bytes: request.payload.to_vec(),
        });
        let output = match invoked {
            Ok(output) => output,
            Err(message) => {
                // Only a Pure or Read pod gets here (an effectful one was refused above, before
                // any invocation), so an error cannot hide an effect that may already have
                // happened (invariant 7): failing over to a second answer is safe.
                self.record_miss(MissReason::DispatchError);
                return self.fail_over(
                    request,
                    FailedPrimary {
                        pod: Some(pod.manifest().id.clone()),
                        attempts: 1,
                        taint_recorded: true,
                        error: RuntimeError::Pod(message),
                    },
                    MissReason::DispatchError,
                    input_taint,
                    verifier,
                );
            }
        };
        match self.promote_verified_pod_output(
            request.request_id,
            pod.manifest(),
            &output,
            verifier,
        ) {
            Ok(revision) => {
                let id = pod.manifest().id.clone();
                self.emit(RuntimeEvent::ReflexHit {
                    pod: id.to_string(),
                });
                self.reflex.counters.record(&ReflexOutcome::Hit { pod: id });
                Ok(DispatchedPod {
                    pod,
                    output,
                    revision,
                })
            }
            Err(error) => {
                // The pod answered and the answer was not accepted: neither a hit nor a miss.
                self.reflex.counters.record_rejected();
                Err(error)
            }
        }
    }

    fn record_miss(&mut self, reason: MissReason) {
        self.emit(RuntimeEvent::ReflexMiss {
            reason: reason.code().to_owned(),
        });
        self.reflex.counters.record(&ReflexOutcome::Miss { reason });
    }

    fn fail_over<V>(
        &mut self,
        request: &PodRequest<'_>,
        failed: FailedPrimary,
        miss: MissReason,
        input_taint: Option<KvReuseTaint>,
        verifier: &V,
    ) -> Result<DispatchedPod, RuntimeError>
    where
        V: Verifier<TypedPayload> + ?Sized,
    {
        let FailedPrimary {
            pod: failed_pod,
            attempts,
            taint_recorded,
            error,
        } = failed;
        let key = (request.project.clone(), request.capability.clone());
        let Some(binding) = self.reflex.plans.get(&key).cloned() else {
            self.reflex
                .counters
                .record_uncovered(UncoveredReason::NoPlan);
            return Err(error);
        };
        let (Some(default), Some(revision)) =
            (binding.plan.default_pod().cloned(), binding.revision)
        else {
            self.reflex
                .counters
                .record_uncovered(UncoveredReason::NoDefault);
            return Err(error);
        };
        // From here on the plan wants a failover and the request fails with a refusal that names
        // the miss and the reason, never a pod.
        let refuse = |this: &mut Self, reason: UncoveredReason| {
            this.reflex.counters.record_uncovered(reason);
            Err(RuntimeError::FailoverRefused { miss, reason })
        };
        if attempts >= binding.plan.max_attempts() {
            return refuse(self, UncoveredReason::AttemptsExhausted);
        }
        if failed_pod.as_ref() == Some(&default) {
            return refuse(self, UncoveredReason::DefaultIsFailedPod);
        }
        // Checked at every use: candidate of this project, capability and input type ...
        if !request
            .pods
            .candidates(request.project, request.capability, request.input_type)
            .contains(&default)
        {
            return refuse(self, UncoveredReason::NotACandidate);
        }
        // ... a generation that is live and not revoked ...
        if !self.pod_head_is_current(&revision) {
            return refuse(self, UncoveredReason::GenerationNotLive);
        }
        let Some(pod) = request.pods.get(request.project, &default) else {
            return refuse(self, UncoveredReason::NotACandidate);
        };
        // ... the registry holds the implementation the checked revision names, not another
        // namespace or generation of the same pod id ...
        if request.pods.namespace(request.project, &default) != Some(&revision.address.namespace)
            || request.pods.generation(request.project, &default) != Some(revision.generation)
        {
            return refuse(self, UncoveredReason::StaleFallback);
        }
        // ... and no effect beyond pure or read.
        if !effect_allowed(pod.as_ref()) {
            return refuse(self, UncoveredReason::EffectNotAllowed);
        }
        if !taint_recorded {
            if let Some(taint) = input_taint {
                if self
                    .record_input_request_taint(request.request_id, taint)
                    .is_err()
                {
                    return refuse(self, UncoveredReason::DefaultRejected);
                }
            }
        }
        self.emit(RuntimeEvent::PodInvoked(default.to_string()));
        let output = match pod.invoke(TypedPayload {
            type_id: request.input_type.clone(),
            bytes: request.payload.to_vec(),
        }) {
            Ok(output) => output,
            // The primary miss is kept and the default's own error is not propagated: it may
            // name the default pod, and the refusal names none.
            Err(_) => return refuse(self, UncoveredReason::DefaultFailed),
        };
        // The same verifier path as any pod output: a contradiction is never promoted.
        match self.promote_verified_pod_output(
            request.request_id,
            pod.manifest(),
            &output,
            verifier,
        ) {
            Ok(revision) => {
                self.emit(RuntimeEvent::ReflexFailover {
                    to: default.to_string(),
                });
                self.reflex
                    .counters
                    .record(&ReflexOutcome::Failover { to: default });
                Ok(DispatchedPod {
                    pod,
                    output,
                    revision,
                })
            }
            // A verifier contradiction (or a taint or type refusal) of the default's output: not
            // promoted, the miss stays uncovered, and the refusal names no pod.
            Err(_) => refuse(self, UncoveredReason::DefaultRejected),
        }
    }
}

/// The primary path that missed, as the failover needs it.
struct FailedPrimary {
    /// The pod that was invoked and failed, if one was.
    pod: Option<PodId>,
    /// Dispatch attempts the primary path used.
    attempts: u8,
    /// Whether the request's input taint was recorded already.
    taint_recorded: bool,
    /// What the request fails with when no failover happens.
    error: RuntimeError,
}
