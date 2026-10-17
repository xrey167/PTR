//! Steppable resumable runs and cooperative cancellation on a revision change.
//!
//! [`PtrRuntime::run_resumable_with_pods`] holds `&mut PtrRuntime` for the whole run, so nothing
//! else can write to the runtime while it is in progress. A [`SteppedRun`] is the same loop cut
//! into rounds: [`PtrRuntime::begin_resumable_run`] ingests the request and asks the backend,
//! each [`PtrRuntime::step_resumable_run`] performs one round (classify what the backend asked
//! for, run one Pod, promote its verified output, resume the backend) and returns control, so a
//! host can interleave other work, including writes to the semantic store, between rounds.
//!
//! # Cancellation (Invariant 2, coarse and cooperative)
//!
//! A run is bound to the semantic revision it last wrote or read. The run's own writes (the
//! ingested request and each promoted Pod output) rebase that bound; any other change of the
//! revision between two steps makes the run stale. With [`CancelPolicy::StaleRevision`] (the
//! default, from `semantic.cancel_stale_revisions`) the next step ends a stale run as
//! [`RunStep::Cancelled`] before it does any more work; with [`CancelPolicy::Off`] the run goes
//! on with its next snapshot, exactly as the blocking entry points always did.
//!
//! The check sits at the top of a step and nowhere else. It is a saving of work, never a safety
//! device: the hard effect boundary ([`PtrRuntime::authorize_action`],
//! [`PtrRuntime::prepare_execution`], [`PtrRuntime::execute_prepared`]) re-checks the current
//! revision and generation of every action whatever the policy, and nothing here is consulted
//! after a run has produced its action or after an effect was attempted. A revision that moves
//! after the run finished, or a generation that is revoked, is the boundary's to refuse.
//!
//! # Scope: no shipped host steps runs
//!
//! `ptr-server` and `ptrd` call the blocking entry points, which begin their run with
//! [`CancelPolicy::Off`] and step it to its end while holding `&mut PtrRuntime`; nothing else can
//! write meanwhile. So `semantic.cancel_stale_revisions` has no effect on any production path
//! today, whatever its value: only a host that calls [`PtrRuntime::step_resumable_run`] itself
//! can see it.
//!
//! The check is a statement about one run. It makes no claim about several runs that share a
//! runtime: each run's promoted output is a write the others did not make, so interleaved runs
//! cancel one another, and a host that keeps beginning new runs can keep every older one from
//! finishing (livelock). Generation staleness is deliberately not a trigger: a lifecycle record
//! moves no semantic revision and the boundary refuses the action.

use crate::reflex::{DispatchedPod, PodRequest};
use crate::routing::PodSelection;
use crate::{PodSampler, PtrRuntime, ResumableRun, RuntimeError};
use ptr_events::RuntimeEvent;
use ptr_model_api::{
    ModelEvent, ModelObservation, ModelRequest, ModelResumeRequest, ResumableInferenceBackend,
};
use ptr_pods::PodRegistry;
use ptr_protocol::TypedPayload;
use ptr_types::{ProjectId, RequestId, Revision};
use ptr_verifier::Verifier;

/// Whether a stepped run ends itself when the semantic revision moved under it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelPolicy {
    /// Never cancel: a step always continues on the current snapshot (the behaviour of the
    /// blocking entry points).
    Off,
    /// Cancel, before the next round does any work, a run whose revision moved for a reason
    /// other than its own writes.
    StaleRevision,
}

impl CancelPolicy {
    /// The policy `semantic.cancel_stale_revisions` selects.
    pub fn from_setting(cancel_stale_revisions: bool) -> Self {
        if cancel_stale_revisions {
            Self::StaleRevision
        } else {
            Self::Off
        }
    }
}

/// What one step needs from the host. The run itself owns no borrowed state, so the host can
/// hand different borrows to different steps.
pub struct StepResources<'a, B: ?Sized, V: ?Sized> {
    pub backend: &'a B,
    pub pods: &'a PodRegistry,
    /// `Some`: every Pod is chosen through this trusted checked router (and each choice emits
    /// [`RuntimeEvent::PodRouted`]); `None`: the plain, unsampled selection.
    pub router: Option<&'a dyn PodSampler>,
    pub verifier: &'a V,
}

/// A resumable run in progress. Created by [`PtrRuntime::begin_resumable_run`], advanced by
/// [`PtrRuntime::step_resumable_run`] until a step returns [`RunStep::Finished`] or
/// [`RunStep::Cancelled`] or fails; a run that has ended refuses further steps.
#[derive(Debug)]
pub struct SteppedRun {
    request_id: RequestId,
    project: ProjectId,
    request: ModelRequest,
    pending: Vec<ModelEvent>,
    run: ResumableRun,
    round: usize,
    max_rounds: usize,
    bound: Revision,
    policy: CancelPolicy,
    ended: bool,
}

impl SteppedRun {
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    /// The revision the run is bound to: the one its own latest write produced.
    pub fn bound_revision(&self) -> Revision {
        self.bound
    }

    /// The number of rounds that completed (Pod runs promoted and the backend resumed).
    pub fn rounds_completed(&self) -> usize {
        self.round
    }

    pub fn policy(&self) -> CancelPolicy {
        self.policy
    }

    pub fn has_ended(&self) -> bool {
        self.ended
    }

    /// What the run holds so far: the model events and observations of the rounds that
    /// completed. The events the backend produced for the next round are not in it until that
    /// round runs.
    pub fn partial(&self) -> &ResumableRun {
        &self.run
    }
}

/// How one step ended.
#[derive(Debug, PartialEq)]
pub enum RunStep {
    /// A round completed and the run is waiting for the next step.
    Pending,
    /// The run is over, as the blocking entry points return it.
    Finished(ResumableRun),
    /// The run was cancelled before it did any more work.
    Cancelled(CancelledRun),
}

/// A run ended by a revision change. It carries what the run committed before it was cancelled
/// (the observations stay in the store) and never an action: the events the backend had
/// produced for the cancelled round are discarded unread. The cancelled step itself writes no
/// semantic state and appends nothing for Exact input; for input whose KV-reuse taint is
/// Approximate it records that taint (as any returned run does), which appends to the ledger.
#[derive(Debug, PartialEq)]
pub struct CancelledRun {
    pub request_id: RequestId,
    /// The revision the run was bound to.
    pub bound: Revision,
    /// The revision the store had when the run was cancelled.
    pub current: Revision,
    pub run: ResumableRun,
}

impl CancelledRun {
    /// The cancellation as an error, for a caller that wants `?`.
    pub fn into_error(self) -> RuntimeError {
        RuntimeError::RunCancelled {
            bound: self.bound,
            current: self.current,
        }
    }
}

impl PtrRuntime {
    /// Starts a resumable run: ingests the request text, asks the backend and returns the run
    /// to step. The cancellation policy is the one `semantic.cancel_stale_revisions` selects.
    pub fn begin_resumable_run<B>(
        &mut self,
        request_id: RequestId,
        project: &ProjectId,
        raw_text: impl Into<String>,
        backend: &B,
        max_rounds: usize,
    ) -> Result<SteppedRun, RuntimeError>
    where
        B: ResumableInferenceBackend + ?Sized,
    {
        let policy = CancelPolicy::from_setting(self.config.semantic.cancel_stale_revisions);
        self.begin_with_policy(request_id, project, raw_text, backend, max_rounds, policy)
    }

    pub(crate) fn begin_with_policy<B>(
        &mut self,
        request_id: RequestId,
        project: &ProjectId,
        raw_text: impl Into<String>,
        backend: &B,
        max_rounds: usize,
        policy: CancelPolicy,
    ) -> Result<SteppedRun, RuntimeError>
    where
        B: ResumableInferenceBackend + ?Sized,
    {
        let raw_text = raw_text.into();
        // Same input taint as `run_model_run_resumable`, computed before the backend runs.
        let input_taint = self.input_kv_taint(&request_id, &raw_text);
        let bound = self.ingest_text(request_id.clone(), raw_text.clone())?;
        let request = ModelRequest {
            request_id: request_id.clone(),
            semantic_context: self.snapshot().semantic_context(),
            raw_text,
        };
        let pending = backend
            .infer(&request)
            .map_err(|error| RuntimeError::Model(error.message().to_owned()))?;
        Ok(SteppedRun {
            request_id,
            project: project.clone(),
            request,
            pending,
            run: ResumableRun {
                kv_taint: input_taint,
                ..ResumableRun::default()
            },
            round: 0,
            max_rounds,
            bound,
            policy,
            ended: false,
        })
    }

    /// Performs one round of `run`. An error ends the run like it ended the blocking loop.
    pub fn step_resumable_run<B, V>(
        &mut self,
        run: &mut SteppedRun,
        resources: &StepResources<'_, B, V>,
    ) -> Result<RunStep, RuntimeError>
    where
        B: ResumableInferenceBackend + ?Sized,
        V: Verifier<TypedPayload> + ?Sized,
    {
        if run.ended {
            return Err(RuntimeError::SteppedRunEnded {
                request: run.request_id.to_string(),
            });
        }
        let outcome = self.step_inner(run, resources);
        if !matches!(outcome, Ok(RunStep::Pending)) {
            run.ended = true;
        }
        outcome
    }

    fn step_inner<B, V>(
        &mut self,
        run: &mut SteppedRun,
        resources: &StepResources<'_, B, V>,
    ) -> Result<RunStep, RuntimeError>
    where
        B: ResumableInferenceBackend + ?Sized,
        V: Verifier<TypedPayload> + ?Sized,
    {
        let max_rounds = run.max_rounds;
        let round = run.round;
        if round > max_rounds {
            return Err(RuntimeError::ModelResumeLimit { max_rounds });
        }

        // The cancellation point: before the round touches the pending events, a Pod or the
        // backend. It is the only one; nothing after the effect boundary is ever recalled.
        if run.policy == CancelPolicy::StaleRevision {
            let current = self.revision();
            if current != run.bound {
                return self.cancel_run(run, current);
            }
        }

        let pending = std::mem::take(&mut run.pending);
        let finished = pending
            .iter()
            .any(|event| matches!(event, ModelEvent::Finished));
        let pod_requests = pending
            .iter()
            .filter_map(|event| match event {
                ModelEvent::PodRequested {
                    capability,
                    input_type,
                    payload,
                } => Some((capability.clone(), input_type.clone(), payload.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        let actions = pending
            .iter()
            .filter_map(|event| match event {
                ModelEvent::ActionReady(action) => Some(action.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        run.run.model_events.extend(pending);

        let control_count = pod_requests.len() + actions.len();
        if control_count > 1 {
            if !pod_requests.is_empty() && !actions.is_empty() {
                return Err(RuntimeError::MixedControlEvents);
            }
            return Err(RuntimeError::MultipleControlEvents {
                count: control_count,
            });
        }

        if finished && control_count == 0 {
            // Approximate input: request, generated text and actions are recorded
            // taint-first before the run is returned. Exact input appends nothing.
            self.record_input_taint(&run.request_id, run.run.kv_taint, &run.run.model_events)?;
            self.emit(RuntimeEvent::RequestFinished(run.request_id.clone()));
            return Ok(RunStep::Finished(std::mem::take(&mut run.run)));
        }

        if let Some(action) = actions.into_iter().next() {
            run.run.action = Some(action);
            self.record_input_taint(&run.request_id, run.run.kv_taint, &run.run.model_events)?;
            return Ok(RunStep::Finished(std::mem::take(&mut run.run)));
        }

        let Some((capability, input_type, payload)) = pod_requests.into_iter().next() else {
            return Err(RuntimeError::ModelNoProgress);
        };

        if round == max_rounds {
            return Err(RuntimeError::ModelResumeLimit { max_rounds });
        }

        let selection = match resources.router {
            Some(router) => PodSelection::Sampled(router),
            None => PodSelection::Plain,
        };
        // A Pod in another project is unavailable in exactly the same words as a Pod that does
        // not exist. Saying which it was would answer, across the boundary, whether that Pod
        // exists.
        let DispatchedPod {
            pod,
            output,
            revision,
        } = self.dispatch_pod(
            &selection,
            &PodRequest {
                pods: resources.pods,
                project: &run.project,
                request_id: &run.request_id,
                capability: &capability,
                input_type: &input_type,
                payload: &payload,
            },
            Some(run.run.kv_taint),
            resources.verifier,
        )?;
        // The promotion is the run's own write: rebase, so only a write by somebody else makes
        // the run stale. Under `&mut self` the revision it returns is the current one; a write
        // inside a step by anybody else would be absorbed here, so it fails loudly in debug.
        debug_assert_eq!(
            revision,
            self.revision(),
            "a write other than the run's own inside a step"
        );
        run.bound = revision;
        // A promoted output is clean (promotion refuses a tainted request or content), but
        // the action it feeds carries the join of everything it consumed.
        run.run.kv_taint = run
            .run
            .kv_taint
            .join(self.text_kv_taint(&output.bytes))
            .join(self.request_kv_taint(&run.request_id));

        let observation = ModelObservation {
            revision,
            source: pod.manifest().id.to_string(),
            type_id: output.type_id.clone(),
            payload: output.bytes.clone(),
        };
        run.run.observations.push(output);

        run.request.semantic_context = self.snapshot().semantic_context();
        run.pending = resources
            .backend
            .resume(&ModelResumeRequest {
                request_id: run.request_id.clone(),
                semantic_context: run.request.semantic_context.clone(),
                round: round as u32 + 1,
                observation,
            })
            .map_err(|error| RuntimeError::Model(error.message().to_owned()))?;
        run.round = round + 1;
        Ok(RunStep::Pending)
    }

    /// Ends `run` as cancelled: the unread events are dropped, the run keeps no action, the
    /// taint of what it did produce is recorded as for any returned run (an append for
    /// Approximate input, nothing for Exact), and the host is told by a runtime event.
    fn cancel_run(
        &mut self,
        run: &mut SteppedRun,
        current: Revision,
    ) -> Result<RunStep, RuntimeError> {
        run.pending.clear();
        let mut partial = std::mem::take(&mut run.run);
        partial.action = None;
        self.record_input_taint(&run.request_id, partial.kv_taint, &partial.model_events)?;
        self.emit(RuntimeEvent::RunCancelled {
            request: run.request_id.clone(),
            bound: run.bound,
            current,
        });
        Ok(RunStep::Cancelled(CancelledRun {
            request_id: run.request_id.clone(),
            bound: run.bound,
            current,
            run: partial,
        }))
    }

    /// The blocking loop: begins a run with cancellation off and steps it to its end. Nothing
    /// else can write while this holds `&mut self`, so a cancellation could not fire; `Off`
    /// makes that a property of the code, not of the schedule.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_resumable_blocking<B, V>(
        &mut self,
        request_id: RequestId,
        project: &ProjectId,
        raw_text: impl Into<String>,
        backend: &B,
        pods: &PodRegistry,
        router: Option<&dyn PodSampler>,
        verifier: &V,
        max_rounds: usize,
    ) -> Result<ResumableRun, RuntimeError>
    where
        B: ResumableInferenceBackend + ?Sized,
        V: Verifier<TypedPayload> + ?Sized,
    {
        let mut run = self.begin_with_policy(
            request_id,
            project,
            raw_text,
            backend,
            max_rounds,
            CancelPolicy::Off,
        )?;
        let resources = StepResources {
            backend,
            pods,
            router,
            verifier,
        };
        loop {
            match self.step_resumable_run(&mut run, &resources)? {
                RunStep::Pending => {}
                RunStep::Finished(finished) => return Ok(finished),
                RunStep::Cancelled(cancelled) => return Err(cancelled.into_error()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The blocking entry points are checked against a verbatim copy of the loop they replaced
    //! (`legacy_run`, the body of `run_resumable_inner` before it became a wrapper), over
    //! scripted backends that reach every exit of that loop. Results, errors, the runtime event
    //! sequence, the revision and the committed ledger must be identical.

    use super::*;
    use crate::{PodSampler, ResumableRun};
    use ptr_config::PtrConfig;
    use ptr_model_api::{InferenceBackend, ModelError};
    use ptr_pods::{DynPod, PodManifest};
    use ptr_types::{
        ActionIr, CapabilityId, Effect, Generation, PodId, Probability, TypeId, VerificationLevel,
    };
    use ptr_verifier::{VerificationReport, VerificationStatus};
    use std::sync::Arc;

    #[allow(clippy::too_many_arguments)]
    fn legacy_run<B, V>(
        runtime: &mut PtrRuntime,
        request_id: RequestId,
        project: &ProjectId,
        raw_text: impl Into<String>,
        backend: &B,
        pods: &PodRegistry,
        selection: PodSelection<'_>,
        verifier: &V,
        max_rounds: usize,
    ) -> Result<ResumableRun, RuntimeError>
    where
        B: ResumableInferenceBackend + ?Sized,
        V: Verifier<TypedPayload> + ?Sized,
    {
        let raw_text = raw_text.into();
        let input_taint = runtime.input_kv_taint(&request_id, &raw_text);
        let _revision = runtime.ingest_text(request_id.clone(), raw_text.clone())?;
        let mut request = ModelRequest {
            request_id: request_id.clone(),
            semantic_context: runtime.snapshot().semantic_context(),
            raw_text,
        };
        let mut pending = backend
            .infer(&request)
            .map_err(|error| RuntimeError::Model(error.message().to_owned()))?;
        let mut run = ResumableRun {
            kv_taint: input_taint,
            ..ResumableRun::default()
        };
        for round in 0..=max_rounds {
            let finished = pending
                .iter()
                .any(|event| matches!(event, ModelEvent::Finished));
            let pod_requests = pending
                .iter()
                .filter_map(|event| match event {
                    ModelEvent::PodRequested {
                        capability,
                        input_type,
                        payload,
                    } => Some((capability.clone(), input_type.clone(), payload.clone())),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let actions = pending
                .iter()
                .filter_map(|event| match event {
                    ModelEvent::ActionReady(action) => Some(action.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            run.model_events.extend(pending);
            let control_count = pod_requests.len() + actions.len();
            if control_count > 1 {
                if !pod_requests.is_empty() && !actions.is_empty() {
                    return Err(RuntimeError::MixedControlEvents);
                }
                return Err(RuntimeError::MultipleControlEvents {
                    count: control_count,
                });
            }
            if finished && control_count == 0 {
                runtime.record_input_taint(&request_id, run.kv_taint, &run.model_events)?;
                runtime.emit(RuntimeEvent::RequestFinished(request_id));
                return Ok(run);
            }
            if let Some(action) = actions.into_iter().next() {
                run.action = Some(action);
                runtime.record_input_taint(&request_id, run.kv_taint, &run.model_events)?;
                return Ok(run);
            }
            let Some((capability, input_type, payload)) = pod_requests.into_iter().next() else {
                return Err(RuntimeError::ModelNoProgress);
            };
            if round == max_rounds {
                return Err(RuntimeError::ModelResumeLimit { max_rounds });
            }
            let pod = runtime.select_pod(
                &selection,
                pods,
                project,
                &request_id,
                &capability,
                &input_type,
                &payload,
            )?;
            if pod
                .manifest()
                .effects
                .iter()
                .any(|effect| !matches!(effect, Effect::Pure | Effect::Read))
            {
                return Err(RuntimeError::PodEffectRequiresActionBoundary {
                    pod: pod.manifest().id.to_string(),
                });
            }
            runtime.record_input_request_taint(&request_id, run.kv_taint)?;
            runtime.emit(RuntimeEvent::PodInvoked(pod.manifest().id.to_string()));
            let output = pod
                .invoke(TypedPayload {
                    type_id: input_type,
                    bytes: payload,
                })
                .map_err(RuntimeError::Pod)?;
            let revision = runtime.promote_verified_pod_output(
                &request_id,
                pod.manifest(),
                &output,
                verifier,
            )?;
            run.kv_taint = run
                .kv_taint
                .join(runtime.text_kv_taint(&output.bytes))
                .join(runtime.request_kv_taint(&request_id));
            let observation = ModelObservation {
                revision,
                source: pod.manifest().id.to_string(),
                type_id: output.type_id.clone(),
                payload: output.bytes.clone(),
            };
            run.observations.push(output);
            request.semantic_context = runtime.snapshot().semantic_context();
            pending = backend
                .resume(&ModelResumeRequest {
                    request_id: request_id.clone(),
                    semantic_context: request.semantic_context.clone(),
                    round: round as u32 + 1,
                    observation,
                })
                .map_err(|error| RuntimeError::Model(error.message().to_owned()))?;
        }
        Err(RuntimeError::ModelResumeLimit { max_rounds })
    }

    /// Replays a script: round 0 answers `infer`, round n answers the n-th `resume`. A
    /// script entry of `None` is a backend error.
    struct Scripted(Vec<Option<Vec<ModelEvent>>>);

    impl InferenceBackend for Scripted {
        fn name(&self) -> &'static str {
            "scripted"
        }

        fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
            self.0[0].clone().ok_or_else(|| ModelError::new("infer"))
        }
    }

    impl ResumableInferenceBackend for Scripted {
        fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
            self.0
                .get(request.round as usize)
                .cloned()
                .unwrap_or(Some(vec![ModelEvent::Finished]))
                .ok_or_else(|| ModelError::new("resume"))
        }
    }

    struct Echo {
        manifest: PodManifest,
        fail: bool,
    }

    impl DynPod for Echo {
        fn manifest(&self) -> &PodManifest {
            &self.manifest
        }

        fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
            if self.fail {
                Err("pod failed".into())
            } else {
                Ok(input)
            }
        }
    }

    struct Verdict(VerificationStatus);

    impl Verifier<TypedPayload> for Verdict {
        fn verify(&self, _: &TypedPayload) -> VerificationReport {
            VerificationReport {
                status: self.0,
                level: VerificationLevel::Deterministic,
                score: Probability::new(1.0).unwrap(),
                findings: vec![],
            }
        }
    }

    fn pods(effect: Effect, fail: bool) -> PodRegistry {
        let mut registry = PodRegistry::default();
        registry.register(Arc::new(Echo {
            manifest: PodManifest {
                project: ProjectId::from("p"),
                id: PodId::from("echo"),
                capabilities: vec![CapabilityId::from("Echo<Text>")],
                accepts: vec![TypeId::from("Text")],
                produces: vec![TypeId::from("Text")],
                effects: vec![effect],
                protocol_version: 1,
            },
            fail,
        }));
        registry
    }

    fn request(n: u8) -> Vec<ModelEvent> {
        vec![ModelEvent::PodRequested {
            capability: CapabilityId::from("Echo<Text>"),
            input_type: TypeId::from("Text"),
            payload: vec![n],
        }]
    }

    fn action() -> ModelEvent {
        ModelEvent::ActionReady(ActionIr {
            operation: "write".into(),
            target: "t".into(),
            capability: CapabilityId::from("c"),
            effect: Effect::Mutation,
            input_type: TypeId::from("Bytes"),
            generation: Generation(1),
            revision: ptr_types::Revision(1),
            payload: b"x".to_vec(),
        })
    }

    fn token(text: &str) -> ModelEvent {
        ModelEvent::Token(text.into())
    }

    fn scripts() -> Vec<(&'static str, Vec<Option<Vec<ModelEvent>>>)> {
        vec![
            (
                "finish at once",
                vec![Some(vec![token("a"), ModelEvent::Finished])],
            ),
            (
                "action at once",
                vec![Some(vec![action(), ModelEvent::Finished])],
            ),
            (
                "action without finish",
                vec![Some(vec![token("t"), action()])],
            ),
            (
                "two pod rounds then an action",
                vec![Some(request(1)), Some(request(2)), Some(vec![action()])],
            ),
            (
                "two pod rounds then finish",
                vec![
                    Some(request(1)),
                    Some(request(2)),
                    Some(vec![ModelEvent::Finished]),
                ],
            ),
            ("no progress", vec![Some(vec![token("only text")])]),
            ("empty", vec![Some(vec![])]),
            (
                "two pod requests",
                vec![Some([request(1), request(2)].concat())],
            ),
            (
                "mixed control",
                vec![Some([request(1), vec![action()]].concat())],
            ),
            ("two actions", vec![Some(vec![action(), action()])]),
            ("backend error at infer", vec![None]),
            ("backend error at resume", vec![Some(request(1)), None]),
            (
                "pod request and finish together",
                vec![Some([request(1), vec![ModelEvent::Finished]].concat())],
            ),
            (
                "round budget",
                vec![
                    Some(request(1)),
                    Some(request(2)),
                    Some(request(3)),
                    Some(request(4)),
                    Some(request(5)),
                ],
            ),
        ]
    }

    #[derive(Debug, PartialEq)]
    struct Observed {
        outcome: Result<ResumableRun, RuntimeError>,
        revision: ptr_types::Revision,
        events: Vec<String>,
        ledger: usize,
    }

    fn observe(runtime: &PtrRuntime, outcome: Result<ResumableRun, RuntimeError>) -> Observed {
        Observed {
            outcome,
            revision: runtime.revision(),
            // The loop this test copies predates the reflex counters, so their telemetry events
            // are left out and the rest is compared in order (renumbered, because the reflex
            // events took sequence numbers).
            events: runtime
                .events()
                .iter()
                .filter(|envelope| {
                    !matches!(
                        envelope.event,
                        RuntimeEvent::ReflexHit { .. }
                            | RuntimeEvent::ReflexMiss { .. }
                            | RuntimeEvent::ReflexFailover { .. }
                    )
                })
                .enumerate()
                .map(|(position, envelope)| format!("{position} {:?}", envelope.event))
                .collect(),
            ledger: runtime.committed_events().len(),
        }
    }

    #[test]
    fn the_blocking_entry_points_equal_the_loop_they_replaced_on_every_exit() {
        let project = ProjectId::from("p");
        let mut compared = 0;
        for (name, script) in scripts() {
            for max_rounds in [0usize, 1, 2, 8] {
                for (label, registry, verdict) in [
                    ("ok", pods(Effect::Pure, false), VerificationStatus::Pass),
                    ("no pod", PodRegistry::default(), VerificationStatus::Pass),
                    (
                        "pod error",
                        pods(Effect::Pure, true),
                        VerificationStatus::Pass,
                    ),
                    (
                        "effect pod",
                        pods(Effect::Mutation, false),
                        VerificationStatus::Pass,
                    ),
                    (
                        "verifier fail",
                        pods(Effect::Pure, false),
                        VerificationStatus::Fail,
                    ),
                ] {
                    let backend = Scripted(script.clone());
                    let verifier = Verdict(verdict);

                    let mut old = PtrRuntime::new(PtrConfig::default()).unwrap();
                    let legacy = legacy_run(
                        &mut old,
                        RequestId::from("r"),
                        &project,
                        "start",
                        &backend,
                        &registry,
                        PodSelection::Plain,
                        &verifier,
                        max_rounds,
                    );
                    let legacy = observe(&old, legacy);

                    // cancellation off, and on (nothing else can write during a blocking run)
                    for cancel in [false, true] {
                        let mut config = PtrConfig::default();
                        config.semantic.cancel_stale_revisions = cancel;
                        let mut new = PtrRuntime::new(config).unwrap();
                        let actual = new.run_resumable_with_pods(
                            RequestId::from("r"),
                            &project,
                            "start",
                            &backend,
                            &registry,
                            &verifier,
                            max_rounds,
                        );
                        assert_eq!(
                            observe(&new, actual),
                            legacy,
                            "{name} / {label} / max_rounds={max_rounds} / cancel={cancel}"
                        );
                        compared += 1;
                    }
                }
            }
        }
        assert!(compared >= 14 * 4 * 5 * 2);
    }

    #[test]
    fn the_routed_entry_point_equals_the_loop_it_replaced() {
        use ptr_router::RoutedChoice;
        // The sampled path with a router that finds no candidate: both loops must refuse in the
        // same words (the successful sampled path is covered by tests/routed_pods.rs, which runs
        // through the same wrapper).
        struct First;
        impl PodSampler for First {
            fn route(
                &self,
                _: &ptr_router::RouteRequest<'_>,
            ) -> Result<Option<RoutedChoice>, ptr_router::RoutedError> {
                Ok(None)
            }
            fn key_id(&self) -> [u8; 8] {
                [7; 8]
            }
        }
        let project = ProjectId::from("p");
        let backend = Scripted(vec![Some(request(1)), Some(vec![ModelEvent::Finished])]);
        let registry = pods(Effect::Pure, false);

        let mut old = PtrRuntime::new(PtrConfig::default()).unwrap();
        let legacy = legacy_run(
            &mut old,
            RequestId::from("r"),
            &project,
            "start",
            &backend,
            &registry,
            PodSelection::Sampled(&First),
            &Verdict(VerificationStatus::Pass),
            4,
        );
        let legacy = observe(&old, legacy);
        assert!(matches!(
            legacy.outcome,
            Err(RuntimeError::PodUnavailable { .. })
        ));

        let mut new = PtrRuntime::new(PtrConfig::default()).unwrap();
        let actual = new.run_resumable_with_pods_using_router(
            RequestId::from("r"),
            &project,
            "start",
            &backend,
            &registry,
            &First,
            &Verdict(VerificationStatus::Pass),
            4,
        );
        assert_eq!(observe(&new, actual), legacy);
    }
}
