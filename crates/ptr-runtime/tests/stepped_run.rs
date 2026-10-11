//! Steppable resumable runs and the cooperative cancellation that sits in front of the effect
//! boundary. Public API only.
//!
//! What is shown here: the blocking entry points are the stepped loop with cancellation off (the
//! same results and the same event sequence); a run's own writes never cancel it; a write by
//! anybody else cancels it at its next step, before any more work and without an action; a fresh
//! run is never cancelled; and the hard effect boundary refuses a stale action whatever the
//! cancellation setting, so the cancellation is a saving of work and never a safety check.

mod common;
#[path = "common/semantic.rs"]
mod semantic_common;

use common::{grant, scope, ExecutorMode, Probe, TTL};
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_events::RuntimeEvent;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_runtime::execution::{ExecutionError, RequiredVerification};
use ptr_runtime::{CancelPolicy, PtrRuntime, RunStep, RuntimeError, StepResources, SteppedRun};
use ptr_security::AuthorizationDenial;
use ptr_semdb::SemanticDelta;
use ptr_types::{
    CapabilityId, CapsuleId, Effect, Generation, PodId, Probability, ProjectId, RequestId,
    Revision, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use semantic_common::{granted, host_write_now};
use std::sync::Arc;

const PROJECT: &str = "p";
const TARGET: &str = "capsule:a";

/// `rounds` Pod requests, then an action stamped with the revision of the context the backend
/// was last shown (what a model that re-reads its snapshot does).
struct Chain {
    rounds: u32,
}

fn pod_request(round: u32) -> Vec<ModelEvent> {
    vec![ModelEvent::PodRequested {
        capability: CapabilityId::from("Echo<Text>"),
        input_type: TypeId::from("Text"),
        payload: format!("round-{round}").into_bytes(),
    }]
}

fn action_at(revision: Revision) -> Vec<ModelEvent> {
    vec![
        ModelEvent::ActionReady(ActionIr {
            operation: "write".into(),
            target: TARGET.into(),
            capability: CapabilityId::from("file.write"),
            effect: Effect::Mutation,
            input_type: TypeId::from("Bytes"),
            generation: Generation(1),
            revision,
            payload: b"verified payload".to_vec(),
        }),
        ModelEvent::Finished,
    ]
}

impl InferenceBackend for Chain {
    fn name(&self) -> &'static str {
        "chain"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(if self.rounds == 0 {
            action_at(request.semantic_context.revision)
        } else {
            pod_request(0)
        })
    }
}

impl ResumableInferenceBackend for Chain {
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(if request.round >= self.rounds {
            action_at(request.revision())
        } else {
            pod_request(request.round)
        })
    }
}

struct Echo {
    manifest: PodManifest,
}

impl DynPod for Echo {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        Ok(input)
    }
}

struct Pass;

impl Verifier<TypedPayload> for Pass {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

fn registry() -> PodRegistry {
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(Echo {
        manifest: PodManifest {
            project: ProjectId::from(PROJECT),
            id: PodId::from("echo"),
            capabilities: vec![CapabilityId::from("Echo<Text>")],
            accepts: vec![TypeId::from("Text")],
            produces: vec![TypeId::from("Text")],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        },
    }));
    registry
}

fn runtime_with(config: PtrConfig) -> PtrRuntime {
    let mut runtime = PtrRuntime::new(config).unwrap();
    granted(&mut runtime);
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from(PROJECT),
            capsule: CapsuleId::from(TARGET),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("file.write"));
    runtime.permissions_mut().allow_mutation = true;
    runtime
}

fn runtime() -> PtrRuntime {
    runtime_with(PtrConfig::default())
}

fn config(cancel_stale_revisions: bool) -> PtrConfig {
    let mut config = PtrConfig::default();
    config.semantic.cancel_stale_revisions = cancel_stale_revisions;
    config
}

fn begin(runtime: &mut PtrRuntime, id: &str, rounds: u32) -> SteppedRun {
    runtime
        .begin_resumable_run(
            RequestId::from(id),
            &ProjectId::from(PROJECT),
            format!("start {id}"),
            &Chain { rounds },
            16,
        )
        .unwrap()
}

fn step(
    runtime: &mut PtrRuntime,
    run: &mut SteppedRun,
    rounds: u32,
) -> Result<RunStep, RuntimeError> {
    let pods = registry();
    runtime.step_resumable_run(
        run,
        &StepResources {
            backend: &Chain { rounds },
            pods: &pods,
            router: None,
            verifier: &Pass,
        },
    )
}

/// A write by somebody other than the run: a fresh key, so the revision always moves.
fn foreign_write(runtime: &mut PtrRuntime, name: &str) -> Revision {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(format!("foreign:{name}"), "x".into());
    host_write_now(runtime, delta).unwrap().revision
}

fn cancelled_events(runtime: &PtrRuntime) -> Vec<(RequestId, Revision, Revision)> {
    runtime
        .events()
        .iter()
        .filter_map(|envelope| match &envelope.event {
            RuntimeEvent::RunCancelled {
                request,
                bound,
                current,
            } => Some((request.clone(), *bound, *current)),
            _ => None,
        })
        .collect()
}

fn drive_to_end(runtime: &mut PtrRuntime, run: &mut SteppedRun, rounds: u32) -> RunStep {
    loop {
        match step(runtime, run, rounds).unwrap() {
            RunStep::Pending => {}
            other => return other,
        }
    }
}

fn effect_attempts(runtime: &PtrRuntime) -> usize {
    runtime
        .committed_events()
        .iter()
        .filter(|committed| matches!(committed.event, LedgerEvent::EffectAttempted { .. }))
        .count()
}

fn event_kinds(runtime: &PtrRuntime) -> Vec<String> {
    runtime
        .events()
        .iter()
        .map(|envelope| format!("{:?}", envelope.event))
        .collect()
}

// ---- the blocking entry points are the stepped loop with cancellation off -------------------

#[test]
fn the_blocking_entry_point_and_the_stepped_loop_give_the_same_run_and_the_same_events() {
    for rounds in [0, 1, 3] {
        let mut blocking = runtime_with(config(false));
        let expected = blocking
            .run_resumable_with_pods(
                RequestId::from("r"),
                &ProjectId::from(PROJECT),
                "start r",
                &Chain { rounds },
                &registry(),
                &Pass,
                16,
            )
            .unwrap();

        for cancel in [false, true] {
            let mut stepped = runtime_with(config(cancel));
            let mut run = begin(&mut stepped, "r", rounds);
            assert_eq!(
                run.policy(),
                if cancel {
                    CancelPolicy::StaleRevision
                } else {
                    CancelPolicy::Off
                }
            );
            let RunStep::Finished(actual) = drive_to_end(&mut stepped, &mut run, rounds) else {
                panic!("a run nobody else writes to finishes");
            };
            assert_eq!(actual, expected, "rounds={rounds} cancel={cancel}");
            assert_eq!(stepped.revision(), blocking.revision());
            assert_eq!(event_kinds(&stepped), event_kinds(&blocking));
            assert_eq!(
                stepped.committed_events().len(),
                blocking.committed_events().len()
            );
            assert!(cancelled_events(&stepped).is_empty());
        }
    }
}

#[test]
fn the_blocking_entry_point_never_cancels_and_keeps_its_limits() {
    let mut runtime = runtime();
    let run = runtime
        .run_resumable_with_pods(
            RequestId::from("r"),
            &ProjectId::from(PROJECT),
            "start r",
            &Chain { rounds: 3 },
            &registry(),
            &Pass,
            16,
        )
        .unwrap();
    assert_eq!(run.observations.len(), 3);
    assert!(run.action.is_some());
    assert!(cancelled_events(&runtime).is_empty());

    // The resume budget still stops a run that asks for more rounds than it was given.
    let mut limited = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert_eq!(
        limited.run_resumable_with_pods(
            RequestId::from("r"),
            &ProjectId::from(PROJECT),
            "start r",
            &Chain { rounds: 5 },
            &registry(),
            &Pass,
            2,
        ),
        Err(RuntimeError::ModelResumeLimit { max_rounds: 2 })
    );
}

// ---- own writes never cancel ------------------------------------------------------------------

#[test]
fn a_runs_own_writes_never_cancel_it_and_rebase_its_bound() {
    let mut runtime = runtime();
    let mut run = begin(&mut runtime, "r", 4);
    let after_ingest = runtime.revision();
    assert_eq!(run.bound_revision(), after_ingest);
    let mut bounds = vec![run.bound_revision()];
    loop {
        match step(&mut runtime, &mut run, 4).unwrap() {
            RunStep::Pending => {
                assert_eq!(run.bound_revision(), runtime.revision());
                bounds.push(run.bound_revision());
            }
            RunStep::Finished(finished) => {
                let action = finished.action.expect("the chain ends in an action");
                assert_eq!(action.revision, runtime.revision(), "not stale");
                break;
            }
            RunStep::Cancelled(cancelled) => panic!("own writes cancelled the run: {cancelled:?}"),
        }
    }
    assert!(
        bounds.windows(2).all(|pair| pair[0] < pair[1]),
        "{bounds:?}"
    );
    assert!(cancelled_events(&runtime).is_empty());
    assert!(run.has_ended());
}

#[test]
fn a_no_op_own_write_does_not_move_the_bound_and_does_not_cancel() {
    // The Pod output of every round is the same bytes: the second promotion changes nothing.
    struct Same;
    impl InferenceBackend for Same {
        fn name(&self) -> &'static str {
            "same"
        }
        fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
            Ok(vec![ModelEvent::PodRequested {
                capability: CapabilityId::from("Echo<Text>"),
                input_type: TypeId::from("Text"),
                payload: b"same".to_vec(),
            }])
        }
    }
    impl ResumableInferenceBackend for Same {
        fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
            Ok(if request.round >= 2 {
                action_at(request.revision())
            } else {
                vec![ModelEvent::PodRequested {
                    capability: CapabilityId::from("Echo<Text>"),
                    input_type: TypeId::from("Text"),
                    payload: b"same".to_vec(),
                }]
            })
        }
    }
    let mut runtime = runtime();
    let mut run = runtime
        .begin_resumable_run(
            RequestId::from("r"),
            &ProjectId::from(PROJECT),
            "start",
            &Same,
            16,
        )
        .unwrap();
    let pods = registry();
    let resources = StepResources {
        backend: &Same,
        pods: &pods,
        router: None,
        verifier: &Pass,
    };
    let mut steps = 0;
    loop {
        match runtime.step_resumable_run(&mut run, &resources).unwrap() {
            RunStep::Pending => steps += 1,
            RunStep::Finished(_) => break,
            RunStep::Cancelled(cancelled) => panic!("{cancelled:?}"),
        }
    }
    assert_eq!(steps, 2);
}

// ---- a foreign write cancels at the next step -------------------------------------------------

#[test]
fn a_foreign_write_cancels_the_run_at_its_next_step_before_any_more_work() {
    let mut runtime = runtime();
    let mut run = begin(&mut runtime, "r", 4);
    assert_eq!(step(&mut runtime, &mut run, 4).unwrap(), RunStep::Pending);
    let bound = run.bound_revision();
    let invoked_before = pod_invocations(&runtime);

    let foreign = foreign_write(&mut runtime, "a");
    assert!(foreign > bound);
    let events_before = runtime.committed_events().len();

    let RunStep::Cancelled(cancelled) = step(&mut runtime, &mut run, 4).unwrap() else {
        panic!("the foreign write must cancel the run");
    };
    assert_eq!(cancelled.bound, bound);
    assert_eq!(cancelled.current, foreign);
    assert_eq!(cancelled.request_id, RequestId::from("r"));
    // Nothing was done on the cancelled step: no Pod, no write, no record.
    assert_eq!(pod_invocations(&runtime), invoked_before);
    assert_eq!(runtime.revision(), foreign);
    assert_eq!(runtime.committed_events().len(), events_before);
    // It keeps what it had committed, and never an action.
    assert_eq!(cancelled.run.observations.len(), 1);
    assert_eq!(cancelled.run.action, None);
    assert!(!cancelled
        .run
        .model_events
        .iter()
        .any(|event| matches!(event, ModelEvent::Finished | ModelEvent::ActionReady(_))));
    assert_eq!(
        cancelled_events(&runtime),
        vec![(RequestId::from("r"), bound, foreign)]
    );
    assert!(!runtime
        .events()
        .iter()
        .any(|envelope| matches!(&envelope.event, RuntimeEvent::RequestFinished(_))));
    // The run is over.
    assert!(run.has_ended());
    assert_eq!(
        step(&mut runtime, &mut run, 4),
        Err(RuntimeError::SteppedRunEnded {
            request: "r".into()
        })
    );
    assert_eq!(
        RunStep::Cancelled(cancelled).into_cancelled_error(),
        RuntimeError::RunCancelled {
            bound,
            current: foreign
        }
    );
}

fn pod_invocations(runtime: &PtrRuntime) -> usize {
    runtime
        .events()
        .iter()
        .filter(|envelope| matches!(envelope.event, RuntimeEvent::PodInvoked(_)))
        .count()
}

trait IntoCancelledError {
    fn into_cancelled_error(self) -> RuntimeError;
}

impl IntoCancelledError for RunStep {
    fn into_cancelled_error(self) -> RuntimeError {
        match self {
            RunStep::Cancelled(cancelled) => cancelled.into_error(),
            other => panic!("not cancelled: {other:?}"),
        }
    }
}

#[test]
fn a_write_between_begin_and_the_first_step_cancels_with_no_work_done() {
    let mut runtime = runtime();
    let mut run = begin(&mut runtime, "r", 3);
    let foreign = foreign_write(&mut runtime, "early");
    let RunStep::Cancelled(cancelled) = step(&mut runtime, &mut run, 3).unwrap() else {
        panic!("cancelled")
    };
    assert_eq!(cancelled.current, foreign);
    assert!(cancelled.run.observations.is_empty());
    assert!(cancelled.run.model_events.is_empty());
    assert_eq!(pod_invocations(&runtime), 0);
}

#[test]
fn a_write_after_the_last_round_cancels_before_the_action_is_handed_out() {
    let mut runtime = runtime();
    let mut run = begin(&mut runtime, "r", 2);
    assert_eq!(step(&mut runtime, &mut run, 2).unwrap(), RunStep::Pending);
    assert_eq!(step(&mut runtime, &mut run, 2).unwrap(), RunStep::Pending);
    // The backend has produced the action; the run has not returned it yet.
    assert_eq!(run.rounds_completed(), 2);
    foreign_write(&mut runtime, "late");
    let RunStep::Cancelled(cancelled) = step(&mut runtime, &mut run, 2).unwrap() else {
        panic!("the action of a stale run must not be handed out")
    };
    assert_eq!(cancelled.run.action, None);
    assert_eq!(cancelled.run.observations.len(), 2);
}

#[test]
fn every_foreign_write_position_cancels_at_the_very_next_step() {
    let rounds = 4u32;
    for position in 0..=rounds {
        let mut runtime = runtime();
        let mut run = begin(&mut runtime, "r", rounds);
        for _ in 0..position {
            assert_eq!(
                step(&mut runtime, &mut run, rounds).unwrap(),
                RunStep::Pending
            );
        }
        foreign_write(&mut runtime, "p");
        let RunStep::Cancelled(cancelled) = step(&mut runtime, &mut run, rounds).unwrap() else {
            panic!("position {position}: not cancelled")
        };
        assert_eq!(cancelled.run.observations.len(), position as usize);
        assert_eq!(cancelled.run.action, None);
    }
}

#[test]
fn a_fresh_run_after_a_cancelled_one_is_not_cancelled() {
    let mut runtime = runtime();
    let mut first = begin(&mut runtime, "first", 3);
    step(&mut runtime, &mut first, 3).unwrap();
    foreign_write(&mut runtime, "x");
    assert!(matches!(
        step(&mut runtime, &mut first, 3).unwrap(),
        RunStep::Cancelled(_)
    ));
    let mut second = begin(&mut runtime, "second", 3);
    assert!(matches!(
        drive_to_end(&mut runtime, &mut second, 3),
        RunStep::Finished(_)
    ));
    assert_eq!(cancelled_events(&runtime).len(), 1);
}

#[test]
fn a_run_begun_after_foreign_writes_is_fresh() {
    let mut runtime = runtime();
    for n in 0..3 {
        foreign_write(&mut runtime, &format!("before-{n}"));
    }
    let mut run = begin(&mut runtime, "r", 3);
    assert!(matches!(
        drive_to_end(&mut runtime, &mut run, 3),
        RunStep::Finished(_)
    ));
    assert!(cancelled_events(&runtime).is_empty());
}

#[test]
fn runs_that_share_a_runtime_cancel_each_other_because_the_check_is_revision_granular() {
    // The stated cost of the coarse check: another run's promoted output is a write the first
    // run did not make, so the older run is cancelled and the younger finishes.
    let mut runtime = runtime();
    let mut older = begin(&mut runtime, "older", 2);
    let mut younger = begin(&mut runtime, "younger", 2);
    assert!(matches!(
        step(&mut runtime, &mut older, 2).unwrap(),
        RunStep::Cancelled(_)
    ));
    assert!(matches!(
        drive_to_end(&mut runtime, &mut younger, 2),
        RunStep::Finished(_)
    ));
}

// ---- cancellation off -------------------------------------------------------------------------

#[test]
fn with_cancellation_off_a_foreign_write_does_not_stop_the_run() {
    let mut runtime = runtime_with(config(false));
    let mut run = begin(&mut runtime, "r", 3);
    assert_eq!(run.policy(), CancelPolicy::Off);
    step(&mut runtime, &mut run, 3).unwrap();
    let foreign = foreign_write(&mut runtime, "a");
    let RunStep::Finished(finished) = drive_to_end(&mut runtime, &mut run, 3) else {
        panic!("cancellation is off")
    };
    assert_eq!(finished.observations.len(), 3);
    // It carried on with the newer snapshot: its action is stamped at or after the write.
    assert!(finished.action.unwrap().revision >= foreign);
    assert!(cancelled_events(&runtime).is_empty());
}

// ---- the hard effect boundary does not depend on the cancellation -----------------------------

fn execution_session(
    runtime: &mut PtrRuntime,
    action: &ActionIr,
    probe: &Probe,
) -> ptr_runtime::execution::ExecutionSession {
    runtime
        .register_execution_session(
            "operator",
            vec![grant(
                scope(action),
                probe,
                RequiredVerification::Deterministic,
                ExecutorMode::Success,
            )],
            TTL,
        )
        .unwrap()
}

#[test]
fn the_boundary_refuses_a_stale_action_whether_cancellation_is_on_or_off() {
    for cancel in [true, false] {
        let mut runtime = runtime_with(config(cancel));
        // The run finishes (its last step returns the action), then somebody writes.
        let mut run = begin(&mut runtime, "r", 2);
        let RunStep::Finished(finished) = drive_to_end(&mut runtime, &mut run, 2) else {
            panic!("finished")
        };
        let action = finished.action.unwrap();
        assert_eq!(runtime.authorize_action(&action), Ok(()));
        foreign_write(&mut runtime, "after-finish");

        // The run is over: cancellation has nothing left to do, and the boundary refuses.
        assert!(cancelled_events(&runtime).is_empty());
        let current = runtime.revision();
        assert_eq!(
            runtime.authorize_action(&action),
            Err(RuntimeError::StaleRevision {
                action: action.revision,
                current
            }),
            "cancel={cancel}"
        );
        let probe = Probe::default();
        let session = execution_session(&mut runtime, &action, &probe);
        let refused = runtime
            .prepare_execution(&session, &ProjectId::from(PROJECT), &action, TTL)
            .unwrap_err();
        assert!(
            matches!(
                refused,
                ExecutionError::AuthorizationDenied(AuthorizationDenial::StaleRevision { .. })
            ),
            "{refused:?}"
        );
        assert_eq!(probe.executions(), 0);
        assert_eq!(probe.verifications(), 0);
        assert_eq!(effect_attempts(&runtime), 0, "no attempt was recorded");
    }
}

#[test]
fn with_cancellation_off_a_stale_final_action_reaches_the_boundary_and_is_refused() {
    // The write lands after the backend produced the action and before the last step returns
    // it: with cancellation off the run hands out an action stamped before the write.
    let mut runtime = runtime_with(config(false));
    let mut run = begin(&mut runtime, "r", 2);
    step(&mut runtime, &mut run, 2).unwrap();
    step(&mut runtime, &mut run, 2).unwrap();
    foreign_write(&mut runtime, "just-before-action");
    let RunStep::Finished(finished) = step(&mut runtime, &mut run, 2).unwrap() else {
        panic!("off: the run finishes")
    };
    let action = finished.action.unwrap();
    assert!(action.revision < runtime.revision());
    assert!(matches!(
        runtime.authorize_action(&action),
        Err(RuntimeError::StaleRevision { .. })
    ));
    let probe = Probe::default();
    let session = execution_session(&mut runtime, &action, &probe);
    assert!(runtime
        .prepare_execution(&session, &ProjectId::from(PROJECT), &action, TTL)
        .is_err());
    assert_eq!(probe.executions(), 0);
    assert_eq!(effect_attempts(&runtime), 0);
}

#[test]
fn a_permit_prepared_before_a_write_is_refused_at_execution_by_the_boundary() {
    // Cancellation has no say after the run: every commit retires the permits prepared before
    // it, so a permit prepared on the current revision is refused when anything is written
    // before it executes.
    let mut runtime = runtime();
    let mut run = begin(&mut runtime, "r", 1);
    let RunStep::Finished(finished) = drive_to_end(&mut runtime, &mut run, 1) else {
        panic!("finished")
    };
    let action = finished.action.unwrap();
    let probe = Probe::default();
    let session = execution_session(&mut runtime, &action, &probe);
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from(PROJECT), &action, TTL)
        .unwrap();
    foreign_write(&mut runtime, "between-prepare-and-execute");
    assert_eq!(
        runtime.execute_prepared(&session, permit),
        Err(ExecutionError::StalePermit)
    );
    assert_eq!(probe.executions(), 0);
    assert_eq!(effect_attempts(&runtime), 0);
}

#[test]
fn a_fresh_action_passes_the_boundary_with_cancellation_on_and_off() {
    for cancel in [true, false] {
        let mut runtime = runtime_with(config(cancel));
        let mut run = begin(&mut runtime, "r", 2);
        let RunStep::Finished(finished) = drive_to_end(&mut runtime, &mut run, 2) else {
            panic!("finished")
        };
        let action = finished.action.unwrap();
        let probe = Probe::default();
        let session = execution_session(&mut runtime, &action, &probe);
        let permit = runtime
            .prepare_execution(&session, &ProjectId::from(PROJECT), &action, TTL)
            .unwrap();
        assert_eq!(
            runtime.execute_prepared(&session, permit).unwrap(),
            b"executed"
        );
        assert_eq!(probe.executions(), 1);
        assert_eq!(effect_attempts(&runtime), 1);
    }
}

#[test]
fn a_revoked_generation_does_not_move_the_revision_so_the_boundary_alone_refuses_it() {
    for cancel in [true, false] {
        let mut runtime = runtime_with(config(cancel));
        let mut run = begin(&mut runtime, "r", 2);
        step(&mut runtime, &mut run, 2).unwrap();
        let before = runtime.revision();
        runtime
            .commit(LedgerEvent::CapsuleSuperseded {
                capsule: CapsuleId::from(TARGET),
                old: Generation(1),
                new: Generation(2),
            })
            .unwrap();
        assert_eq!(
            runtime.revision(),
            before,
            "a lifecycle record is no revision"
        );
        let RunStep::Finished(finished) = drive_to_end(&mut runtime, &mut run, 2) else {
            panic!("the revision did not move: nothing to cancel")
        };
        let action = finished.action.unwrap();
        assert!(matches!(
            runtime.authorize_action(&action),
            Err(RuntimeError::StaleGeneration { .. })
        ));
        let probe = Probe::default();
        let session = execution_session(&mut runtime, &action, &probe);
        assert!(runtime
            .prepare_execution(&session, &ProjectId::from(PROJECT), &action, TTL)
            .is_err());
        assert_eq!(probe.executions(), 0);
        assert_eq!(effect_attempts(&runtime), 0);
    }
}

// ---- the lifecycle of a run -------------------------------------------------------------------

#[test]
fn an_error_ends_the_run_like_it_ended_the_blocking_loop() {
    let mut runtime = runtime();
    let mut run = begin(&mut runtime, "r", 2);
    let empty = PodRegistry::default();
    let outcome = runtime.step_resumable_run(
        &mut run,
        &StepResources {
            backend: &Chain { rounds: 2 },
            pods: &empty,
            router: None,
            verifier: &Pass,
        },
    );
    assert!(matches!(outcome, Err(RuntimeError::PodUnavailable { .. })));
    assert!(run.has_ended());
    assert!(matches!(
        step(&mut runtime, &mut run, 2),
        Err(RuntimeError::SteppedRunEnded { .. })
    ));
}

#[test]
fn the_resume_budget_ends_a_stepped_run() {
    let mut runtime = runtime();
    let mut run = runtime
        .begin_resumable_run(
            RequestId::from("r"),
            &ProjectId::from(PROJECT),
            "start",
            &Chain { rounds: 9 },
            1,
        )
        .unwrap();
    let pods = registry();
    let resources = StepResources {
        backend: &Chain { rounds: 9 },
        pods: &pods,
        router: None,
        verifier: &Pass,
    };
    assert_eq!(
        runtime.step_resumable_run(&mut run, &resources).unwrap(),
        RunStep::Pending
    );
    assert_eq!(
        runtime.step_resumable_run(&mut run, &resources),
        Err(RuntimeError::ModelResumeLimit { max_rounds: 1 })
    );
}

#[test]
fn the_setting_selects_the_policy() {
    for (setting, policy) in [
        (true, CancelPolicy::StaleRevision),
        (false, CancelPolicy::Off),
    ] {
        assert_eq!(CancelPolicy::from_setting(setting), policy);
        let mut runtime = runtime_with(config(setting));
        assert_eq!(begin(&mut runtime, "r", 1).policy(), policy);
    }
    assert!(PtrConfig::default().semantic.cancel_stale_revisions);
}
