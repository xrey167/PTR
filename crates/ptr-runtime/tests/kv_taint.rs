mod common;

use common::{fixture, session, Probe};
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    KvReuseMode, KvReusePolicy, KvReuseReport, KvReuseTaint, ModelEvent, ModelRun,
};
use ptr_runtime::execution::ExecutionError;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{
    CapabilityId, CapsuleId, Effect, Generation, ProjectId, RequestId, Revision, TypeId,
};
use std::time::Duration;

fn action(revision: Revision, payload: &[u8]) -> ActionIr {
    ActionIr {
        operation: "write".into(),
        target: "artifact:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision,
        payload: payload.to_vec(),
    }
}

fn ready_runtime(payload: &[u8]) -> (PtrRuntime, ActionIr) {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let revision = runtime.ingest_text(RequestId::from("r1"), "hello").unwrap();
    let action = action(revision, payload);
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from(action.target.as_str()),
            generation: action.generation,
        })
        .unwrap();
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
    (runtime, action)
}

fn run_with(action: &ActionIr, mode: KvReuseMode) -> ModelRun {
    let policy = match mode {
        KvReuseMode::Exact => KvReusePolicy::ExactOnly,
        KvReuseMode::Approximate => KvReusePolicy::AllowApproximate,
    };
    ModelRun {
        events: vec![ModelEvent::ActionReady(action.clone())],
        kv_report: Some(KvReuseReport::new(16, mode, 64, policy).unwrap()),
    }
}

fn effect_attempts(runtime: &PtrRuntime) -> usize {
    runtime
        .committed_events()
        .iter()
        .filter(|committed| matches!(committed.event, LedgerEvent::EffectAttempted { .. }))
        .count()
}

#[test]
fn approximate_action_is_denied_and_nothing_is_attempted() {
    let (mut runtime, action) = ready_runtime(b"x");
    let before = runtime.committed_events().len();
    let run = runtime
        .ingest_model_run(&run_with(&action, KvReuseMode::Approximate))
        .unwrap();
    assert!(run.kv_taint.is_approximate());
    assert_eq!(run.action.as_ref(), Some(&action));
    let denied = runtime.authorize_action(&action);
    assert!(matches!(
        denied,
        Err(RuntimeError::ApproximateStateAtEffectBoundary { .. })
    ));
    assert_eq!(effect_attempts(&runtime), 0);
    // The only record the ingest appended is its taint record, written before it returned.
    assert_eq!(runtime.committed_events().len(), before + 1);
    assert!(matches!(
        runtime.committed_events().last().unwrap().event,
        LedgerEvent::KvTaintCommitted { .. }
    ));
    // The explicit-taint entry point refuses too, even for a never-ingested action.
    let (runtime2, action2) = ready_runtime(b"y");
    assert!(runtime2.authorize_action(&action2).is_ok());
    assert!(matches!(
        runtime2.authorize_action_with_taint(&action2, run.kv_taint),
        Err(RuntimeError::ApproximateStateAtEffectBoundary { .. })
    ));
}

#[test]
fn exact_and_no_reuse_paths_are_unchanged() {
    let (mut runtime, action) = ready_runtime(b"x");
    let exact = runtime
        .ingest_model_run(&run_with(&action, KvReuseMode::Exact))
        .unwrap();
    assert_eq!(exact.kv_taint, KvReuseTaint::Exact);
    assert!(runtime.authorize_action(&action).is_ok());
    assert!(runtime
        .authorize_action_with_taint(&action, exact.kv_taint)
        .is_ok());

    let none = runtime
        .ingest_model_run(&ModelRun {
            events: vec![ModelEvent::ActionReady(action.clone())],
            kv_report: None,
        })
        .unwrap();
    assert_eq!(none.kv_taint, KvReuseTaint::Exact);
    assert!(runtime.authorize_action(&action).is_ok());
    assert_eq!(runtime.action_kv_taint(&action), KvReuseTaint::Exact);
}

#[test]
fn taint_is_not_cleared_by_a_later_exact_ingest_of_the_same_action() {
    let (mut runtime, action) = ready_runtime(b"x");
    runtime
        .ingest_model_run(&run_with(&action, KvReuseMode::Approximate))
        .unwrap();
    runtime
        .ingest_model_run(&run_with(&action, KvReuseMode::Exact))
        .unwrap();
    assert!(runtime.action_kv_taint(&action).is_approximate());
    assert!(matches!(
        runtime.authorize_action(&action),
        Err(RuntimeError::ApproximateStateAtEffectBoundary { .. })
    ));
}

#[test]
fn other_actions_are_unaffected_by_a_tainted_one() {
    let (mut runtime, dirty) = ready_runtime(b"x");
    let clean = ActionIr {
        payload: b"other".to_vec(),
        ..dirty.clone()
    };
    runtime
        .ingest_model_run(&run_with(&dirty, KvReuseMode::Approximate))
        .unwrap();
    assert!(runtime.authorize_action(&clean).is_ok());
}

#[test]
fn malformed_runs_are_refused_before_recording() {
    let (mut runtime, action) = ready_runtime(b"x");
    let two = ModelRun {
        events: vec![
            ModelEvent::ActionReady(action.clone()),
            ModelEvent::ActionReady(action.clone()),
        ],
        kv_report: Some(
            KvReuseReport::new(
                1,
                KvReuseMode::Approximate,
                4,
                KvReusePolicy::AllowApproximate,
            )
            .unwrap(),
        ),
    };
    assert_eq!(
        runtime.ingest_model_run(&two),
        Err(RuntimeError::MultipleControlEvents { count: 2 })
    );
    assert_eq!(runtime.action_kv_taint(&action), KvReuseTaint::Exact);
}

#[test]
fn tainted_registry_is_bounded_and_fails_closed() {
    let (mut runtime, base) = ready_runtime(b"0");
    for i in 0..ptr_runtime::kv_taint::MAX_LIVE_TAINT_SUBJECTS {
        let a = ActionIr {
            payload: i.to_le_bytes().to_vec(),
            ..base.clone()
        };
        runtime
            .ingest_model_run(&run_with(&a, KvReuseMode::Approximate))
            .unwrap();
    }
    let extra = ActionIr {
        payload: b"one-too-many".to_vec(),
        ..base
    };
    assert!(matches!(
        runtime.ingest_model_run(&run_with(&extra, KvReuseMode::Approximate)),
        Err(RuntimeError::KvTaintRegistryFull { .. })
    ));
}

fn is_approximate_denial(error: &ExecutionError) -> bool {
    matches!(
        error,
        ExecutionError::Audit(inner)
            if matches!(**inner, RuntimeError::ApproximateStateAtEffectBoundary { .. })
    )
}

const PREPARE_TTL: Duration = Duration::from_secs(60);

#[test]
fn prepare_execution_once_refuses_a_tainted_action_without_authorize_action() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    runtime
        .ingest_model_run(&run_with(&action, KvReuseMode::Approximate))
        .unwrap();
    let committed = runtime.committed_events().len();

    let denied = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, PREPARE_TTL, "k1")
        .unwrap_err();
    assert!(is_approximate_denial(&denied), "{denied:?}");
    let denied = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, PREPARE_TTL)
        .unwrap_err();
    assert!(is_approximate_denial(&denied), "{denied:?}");

    assert_eq!(effect_attempts(&runtime), 0);
    assert_eq!(runtime.committed_events().len(), committed);
    assert_eq!(probe.executions(), 0);
}

#[test]
fn admit_effect_refuses_an_action_tainted_after_its_permit_was_prepared() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, PREPARE_TTL, "k2")
        .unwrap();
    runtime
        .ingest_model_run(&run_with(&action, KvReuseMode::Approximate))
        .unwrap();
    let committed = runtime.committed_events().len();

    // The taint record is a ledger commit, so it also stales the outstanding permit: the old
    // permit is refused (as StalePermit, before the taint check is reached) and nothing runs.
    let denied = runtime.execute_prepared(&session, permit).unwrap_err();
    assert_eq!(denied, ExecutionError::StalePermit, "{denied:?}");
    assert_eq!(effect_attempts(&runtime), 0);
    assert_eq!(runtime.committed_events().len(), committed);
    assert_eq!(probe.executions(), 0);

    // A fresh permit for the tainted action cannot be obtained: the taint check refuses it.
    let denied = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, PREPARE_TTL, "k2b")
        .unwrap_err();
    assert!(is_approximate_denial(&denied), "{denied:?}");
    assert_eq!(effect_attempts(&runtime), 0);
    assert_eq!(probe.executions(), 0);
}

#[test]
fn untainted_and_exact_actions_still_execute_on_the_execution_path() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    runtime
        .ingest_model_run(&run_with(&action, KvReuseMode::Exact))
        .unwrap();
    let permit = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, PREPARE_TTL, "k3")
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, permit),
        Ok(b"executed".to_vec())
    );
    assert_eq!(effect_attempts(&runtime), 1);
    assert_eq!(probe.executions(), 1);
}

#[test]
fn restamped_tainted_action_is_still_refused_on_both_paths() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    runtime
        .ingest_model_run(&run_with(&action, KvReuseMode::Approximate))
        .unwrap();
    // An unrelated commit moves the revision; the stale action is re-stamped to it.
    let new_revision = runtime
        .ingest_text(RequestId::from("later"), "unrelated")
        .unwrap();
    let restamped = ActionIr {
        revision: new_revision,
        generation: Generation(1),
        ..action.clone()
    };
    assert_ne!(restamped.revision, action.revision);
    assert!(runtime.action_kv_taint(&restamped).is_approximate());
    assert!(matches!(
        runtime.authorize_action(&restamped),
        Err(RuntimeError::ApproximateStateAtEffectBoundary { .. })
    ));
    let committed = runtime.committed_events().len();
    // A different idempotency key does not shed it either.
    for key in ["key-a", "key-b"] {
        let denied = runtime
            .prepare_execution_once(
                &session,
                &ProjectId::from("p"),
                &restamped,
                PREPARE_TTL,
                key,
            )
            .unwrap_err();
        assert!(is_approximate_denial(&denied), "{denied:?}");
    }
    assert_eq!(effect_attempts(&runtime), 0);
    assert_eq!(runtime.committed_events().len(), committed);

    // A different payload is a different action and is not tainted.
    let other = ActionIr {
        payload: b"verified payload but different".to_vec(),
        ..restamped
    };
    assert_eq!(runtime.action_kv_taint(&other), KvReuseTaint::Exact);
}
