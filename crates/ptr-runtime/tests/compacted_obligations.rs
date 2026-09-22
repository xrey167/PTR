//! What a compacted snapshot has to carry across its own floor.
//!
//! Compaction exists to let a log's floor rise. The execution layer's two
//! durable facts — the fence, and which at-most-once keys have been spent — are
//! *derived by replaying committed records*, which is exactly what a raised floor
//! makes impossible. So a snapshot that describes only semantic and lifecycle
//! state describes a runtime that has quietly forgotten what it promised.
//!
//! The failure this file pins is not abstract. Before the execution section
//! existed, compacting after an at-most-once effect and restoring produced a
//! runtime that ran the same effect a second time under the same key — the exact
//! retention obligation `28-durable-execution-audit.md` names, discharged by
//! nothing.

mod common;

use common::*;
use ptr_config::PtrConfig;
use ptr_runtime::compacted::CompactedError;
use ptr_runtime::PtrRuntime;
use ptr_types::ProjectId;

/// Permissions are this process's, not committed history's, so a restored
/// runtime starts with none. Re-granting them keeps each test's question about
/// the obligations rather than about authorization.
fn regrant(runtime: &mut PtrRuntime, action: &ptr_core::action_head::ActionIr) {
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
}

fn round_trip(runtime: &PtrRuntime) -> PtrRuntime {
    let snapshot = runtime
        .export_compacted_snapshot()
        .expect("a settled runtime exports");
    PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .expect("its own snapshot restores")
}

#[test]
fn an_at_most_once_key_survives_the_floor_rising_past_its_attempt() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let opened = session(&mut runtime, &action, &probe, "alice");
    let first = runtime
        .prepare_execution_once(&opened, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    runtime.execute_prepared(&opened, first).unwrap();
    assert_eq!(probe.executions(), 1, "the effect applied exactly once");

    let mut restored = round_trip(&runtime);
    regrant(&mut restored, &action);

    // A fresh probe, so any execution here is a *second* application of an effect
    // the world has already seen.
    let after = Probe::default();
    let reopened = session(&mut restored, &action, &after, "alice");
    let retry = restored
        .prepare_execution_once(&reopened, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    let answer = restored.execute_prepared(&reopened, retry).unwrap();

    assert_eq!(
        after.executions(),
        0,
        "the key was spent before the floor rose; compaction must not refund it"
    );
    assert_eq!(
        answer, b"executed",
        "a retry under a spent key is answered from history, not refused"
    );
}

/// The control.
///
/// The test above would pass just as well against a restored runtime that
/// refused *everything* — a snapshot that broke execution entirely would look
/// identical from the outside. So the same restored runtime must still execute a
/// key it has never seen.
#[test]
fn a_key_the_snapshot_never_carried_still_executes_after_restore() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let opened = session(&mut runtime, &action, &probe, "alice");
    let spent = runtime
        .prepare_execution_once(&opened, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    runtime.execute_prepared(&opened, spent).unwrap();

    let mut restored = round_trip(&runtime);
    regrant(&mut restored, &action);

    let after = Probe::default();
    let reopened = session(&mut restored, &action, &after, "alice");
    let fresh = restored
        .prepare_execution_once(&reopened, &ProjectId::from("p"), &action, TTL, "invoice-8")
        .unwrap();
    restored.execute_prepared(&reopened, fresh).unwrap();

    assert_eq!(
        after.executions(),
        1,
        "an unspent key must still execute; otherwise the test above proves nothing"
    );
}

#[test]
fn a_snapshot_in_the_format_that_carried_no_obligations_is_refused_by_version() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let opened = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution_once(&opened, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    runtime.execute_prepared(&opened, permit).unwrap();

    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let mut bytes = snapshot.bytes().to_vec();
    bytes[..8].copy_from_slice(b"PTRCS001");

    // Refused rather than read as a snapshot with no obligations. Accepting the
    // older layout would restore a runtime that silently forgot every spent key,
    // which is the failure this format change exists to end — so the magic moved
    // rather than only the reserved field.
    let error = PtrRuntime::restore_compacted(PtrConfig::default(), &bytes, snapshot.anchor(), &[])
        .err()
        .map(|error| format!("{error:?}"))
        .expect("an older layout must not be read as an empty obligation set");
    assert!(
        error.contains(CompactedError::UnsupportedVersion.code())
            || error.contains("UnsupportedVersion"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn a_corrupted_execution_section_is_refused_rather_than_partially_restored() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let opened = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution_once(&opened, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    runtime.execute_prepared(&opened, permit).unwrap();

    let snapshot = runtime.export_compacted_snapshot().unwrap();
    // Drop one byte from the tail of the body. The digest no longer matches, which
    // is the point: a section that decodes to "fewer obligations" must never be a
    // reachable outcome of damage.
    let mut bytes = snapshot.bytes().to_vec();
    let end = bytes.len() - 32;
    bytes.remove(end - 1);

    assert!(
        PtrRuntime::restore_compacted(PtrConfig::default(), &bytes, snapshot.anchor(), &[])
            .is_err(),
        "a damaged snapshot must be refused, not partially restored"
    );
}

/// A retained response is a payload, not a collection.
///
/// The section first encoded it through the writer's `count`, which bounds a
/// collection's *cardinality* at 65,536. The runtime accepts a response up to
/// `MAX_RETAINED_RESPONSE` (1 MiB) and retains it, so any legal response over
/// 64 KiB made `export_compacted_snapshot` fail — not once, but for the rest of
/// that runtime's life, because the offending entry stays in `settled`. A runtime
/// that can never export can never let its floor rise again.
#[test]
fn a_retained_response_larger_than_the_item_bound_still_round_trips() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let opened = session_with(
        &mut runtime,
        &action,
        &probe,
        "alice",
        ExecutorMode::Sized(100_000),
    );
    let permit = runtime
        .prepare_execution_once(&opened, &ProjectId::from("p"), &action, TTL, "invoice-9")
        .unwrap();
    let answer = runtime.execute_prepared(&opened, permit).unwrap();
    assert_eq!(
        answer.len(),
        100_000,
        "the effect returned a large response"
    );

    let restored = round_trip(&runtime);
    let mut restored = restored;
    regrant(&mut restored, &action);

    let after = Probe::default();
    let reopened = session(&mut restored, &action, &after, "alice");
    let retry = restored
        .prepare_execution_once(&reopened, &ProjectId::from("p"), &action, TTL, "invoice-9")
        .unwrap();
    let replayed = restored.execute_prepared(&reopened, retry).unwrap();

    assert_eq!(
        after.executions(),
        0,
        "the key was spent before the floor rose"
    );
    assert_eq!(
        replayed.len(),
        100_000,
        "and the retained response survives the round trip intact"
    );
}
