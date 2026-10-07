//! The approximate-KV taint registry is durable: it survives a reopen of the ledger and a
//! compacted-snapshot restore, is written before the run is returned, never written for an
//! Exact run, and fails closed on a forged Clear or a malformed restored entry.

use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::integrity;
use ptr_ledger::{FileLedger, KvTaintChange, KvTaintSubjectKind, LedgerEvent};
use ptr_model_api::{
    ByteTokenizer, KvReuseMode, KvReusePolicy, KvReuseReport, ModelEvent, ModelRun, PromptSegment,
    SegmentedModelRequest, Tokenizer as _,
};
use ptr_runtime::compacted::CompactedAnchor;
use ptr_runtime::kv_taint::{KvTaintError, KvTaintRequest};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{
    CapabilityId, CapsuleId, Effect, Generation, ProjectId, RequestId, Revision, SemanticContext,
    TypeId,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ptr-kv-taint-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn log(&self) -> PathBuf {
        self.0.join("ledger")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn action(payload: &[u8]) -> ActionIr {
    ActionIr {
        operation: "write".into(),
        target: "artifact:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision: Revision(0),
        payload: payload.to_vec(),
    }
}

/// Grant what `authorize_action` needs, so a refusal can only come from the taint.
fn arm(runtime: &mut PtrRuntime, action: &ActionIr) {
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

fn taint_records(runtime: &PtrRuntime) -> Vec<LedgerEvent> {
    runtime
        .committed_events()
        .iter()
        .filter(|committed| matches!(committed.event, LedgerEvent::KvTaintCommitted { .. }))
        .map(|committed| committed.event.clone())
        .collect()
}

fn is_denied(result: Result<(), RuntimeError>) -> bool {
    matches!(
        result,
        Err(RuntimeError::ApproximateStateAtEffectBoundary { .. })
    )
}

fn request(id: &str) -> SegmentedModelRequest {
    let tokenizer = ByteTokenizer::new();
    let segment = PromptSegment::raw_from_text(&tokenizer, "hello taint").unwrap();
    SegmentedModelRequest::new(
        RequestId::from(id),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        tokenizer.identity().clone(),
        vec![segment],
    )
    .unwrap()
}

#[test]
fn an_approximate_action_is_still_refused_after_reopening_the_durable_ledger() {
    let temp = Temp::new();
    let tainted = action(b"approximate");
    let clean = action(b"clean");
    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
        arm(&mut runtime, &tainted);
        runtime
            .ingest_model_run(&run_with(&tainted, KvReuseMode::Approximate))
            .unwrap();
        assert!(is_denied(runtime.authorize_action(&tainted)));
    }
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    runtime
        .permissions_mut()
        .capabilities
        .insert(tainted.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
    assert!(runtime.action_kv_taint(&tainted).is_approximate());
    assert!(is_denied(runtime.authorize_action(&tainted)));
    // A different payload is a different action and stays authorized.
    assert!(runtime.authorize_action(&clean).is_ok());
    // An Exact ingest after the restart still does not clear it.
    runtime
        .ingest_model_run(&run_with(&tainted, KvReuseMode::Exact))
        .unwrap();
    assert!(is_denied(runtime.authorize_action(&tainted)));
}

#[test]
fn an_approximate_action_is_still_refused_after_a_compacted_snapshot_restore() {
    let tainted = action(b"approximate");
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    arm(&mut runtime, &tainted);
    runtime
        .ingest_model_run_for(
            &KvTaintRequest {
                request: &request("r-compact"),
                producer: "stub-approximate-producer/1",
                generation_parameters: b"max_new_tokens=8",
            },
            &run_with(&tainted, KvReuseMode::Approximate),
        )
        .unwrap();
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let mut restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    restored
        .permissions_mut()
        .capabilities
        .insert(tainted.capability.clone());
    restored.permissions_mut().allow_mutation = true;
    assert!(restored.action_kv_taint(&tainted).is_approximate());
    assert!(is_denied(restored.authorize_action(&tainted)));
    assert!(restored.authorize_action(&action(b"clean")).is_ok());
}

#[test]
fn a_taint_record_above_the_snapshot_floor_is_applied_on_restore() {
    let tainted = action(b"approximate");
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    arm(&mut runtime, &tainted);
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let floor = runtime.committed_events().len();
    runtime
        .ingest_model_run(&run_with(&tainted, KvReuseMode::Approximate))
        .unwrap();
    let above = runtime.committed_events()[floor..].to_vec();
    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &above,
    )
    .unwrap();
    assert!(restored.action_kv_taint(&tainted).is_approximate());
}

fn forged_log(temp: &Temp, events: Vec<LedgerEvent>) {
    let mut log = FileLedger::open(temp.log()).unwrap();
    for event in events {
        log.append_durable(event).unwrap();
    }
}

fn taint_event(digest: [u8; 32], change: KvTaintChange) -> LedgerEvent {
    LedgerEvent::KvTaintCommitted {
        subjects: vec![(KvTaintSubjectKind::Content, digest)],
        change,
    }
}

fn record(recipe: [u8; 32]) -> KvTaintChange {
    KvTaintChange::Record {
        report_digest: [3; 32],
        recipe_digest: recipe,
    }
}

fn clear(recipe: [u8; 32]) -> KvTaintChange {
    KvTaintChange::Clear {
        recipe_digest: recipe,
        recompute_digest: [4; 32],
    }
}

#[test]
fn a_forged_clear_of_a_subject_that_is_not_live_makes_reopen_fail() {
    let temp = Temp::new();
    forged_log(&temp, vec![taint_event([1; 32], clear([2; 32]))]);
    assert_eq!(
        PtrRuntime::open_durable(PtrConfig::default(), temp.log()).err(),
        Some(RuntimeError::KvTaint(KvTaintError::ClearSubjectNotLive))
    );
}

#[test]
fn a_forged_clear_with_another_recipe_makes_reopen_fail() {
    let temp = Temp::new();
    forged_log(
        &temp,
        vec![
            taint_event([1; 32], record([2; 32])),
            taint_event([1; 32], clear([9; 32])),
        ],
    );
    assert_eq!(
        PtrRuntime::open_durable(PtrConfig::default(), temp.log()).err(),
        Some(RuntimeError::KvTaint(KvTaintError::ClearRecipeMismatch))
    );
}

#[test]
fn a_forged_duplicate_record_of_a_live_subject_makes_reopen_fail() {
    // The compacted projection overwrites `kv-taint:<kind>:<hex>` on every Record while the
    // registry keeps the first live entry, so a second Record of a live subject with another
    // recipe would make a compacted restore differ from live and replayed state. It is refused.
    let temp = Temp::new();
    forged_log(
        &temp,
        vec![
            taint_event([1; 32], record([2; 32])),
            taint_event([1; 32], record([5; 32])),
        ],
    );
    assert_eq!(
        PtrRuntime::open_durable(PtrConfig::default(), temp.log()).err(),
        Some(RuntimeError::KvTaint(
            KvTaintError::RecordSubjectAlreadyLive
        ))
    );
    let ledger = FileLedger::open(temp.log()).unwrap();
    assert_eq!(
        PtrRuntime::replay(PtrConfig::default(), ledger.events()).err(),
        Some(RuntimeError::KvTaint(
            KvTaintError::RecordSubjectAlreadyLive
        ))
    );
}

#[test]
fn a_record_after_a_clear_of_the_same_subject_is_not_a_duplicate() {
    let temp = Temp::new();
    forged_log(
        &temp,
        vec![
            taint_event([1; 32], record([2; 32])),
            taint_event([1; 32], clear([2; 32])),
            taint_event([1; 32], record([5; 32])),
        ],
    );
    assert!(PtrRuntime::open_durable(PtrConfig::default(), temp.log()).is_ok());
}

#[test]
fn a_forged_record_is_applied_and_a_matching_clear_is_honoured() {
    // A Record of a subject that is not live, written straight into the log, applies: taint
    // only ever refuses more, so replay has no reason to reject it. A Clear is different (it lifts a refusal), so it
    // must match a live Record with the same recipe, as here.
    let temp = Temp::new();
    let tainted = action(b"forged");
    let digest = {
        // The Action subject digest is internal; take it from a runtime that ingested it.
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime
            .ingest_model_run(&run_with(&tainted, KvReuseMode::Approximate))
            .unwrap();
        match &taint_records(&runtime)[0] {
            LedgerEvent::KvTaintCommitted { subjects, .. } => subjects[0].1,
            _ => unreachable!(),
        }
    };
    forged_log(
        &temp,
        vec![LedgerEvent::KvTaintCommitted {
            subjects: vec![(KvTaintSubjectKind::Action, digest)],
            change: record([2; 32]),
        }],
    );
    let runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    assert!(runtime.action_kv_taint(&tainted).is_approximate());
    drop(runtime);
    forged_log(
        &temp,
        vec![LedgerEvent::KvTaintCommitted {
            subjects: vec![(KvTaintSubjectKind::Action, digest)],
            change: clear([2; 32]),
        }],
    );
    let runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    assert!(!runtime.action_kv_taint(&tainted).is_approximate());
}

#[test]
fn an_exact_run_appends_nothing_and_an_approximate_run_appends_before_it_returns() {
    let a = action(b"x");
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let before = runtime.committed_events().len();
    runtime
        .ingest_model_run(&run_with(&a, KvReuseMode::Exact))
        .unwrap();
    runtime
        .ingest_model_run(&ModelRun {
            events: vec![ModelEvent::ActionReady(a.clone())],
            kv_report: None,
        })
        .unwrap();
    runtime
        .ingest_model_run_for(
            &KvTaintRequest {
                request: &request("exact"),
                producer: "p",
                generation_parameters: b"",
            },
            &run_with(&a, KvReuseMode::Exact),
        )
        .unwrap();
    assert_eq!(runtime.committed_events().len(), before);

    // The record is already committed when the call returns: Action and Request subjects.
    let resumable = runtime
        .ingest_model_run_for(
            &KvTaintRequest {
                request: &request("approx"),
                producer: "p",
                generation_parameters: b"",
            },
            &run_with(&a, KvReuseMode::Approximate),
        )
        .unwrap();
    assert!(resumable.kv_taint.is_approximate());
    let records = taint_records(&runtime);
    assert_eq!(records.len(), 1);
    let LedgerEvent::KvTaintCommitted { subjects, change } = &records[0] else {
        unreachable!()
    };
    let kinds: Vec<_> = subjects.iter().map(|(kind, _)| *kind).collect();
    assert_eq!(
        kinds,
        vec![KvTaintSubjectKind::Action, KvTaintSubjectKind::Request]
    );
    assert!(matches!(change, KvTaintChange::Record { .. }));
    assert_eq!(runtime.committed_events().len(), before + 1);
    // The same ingest again is idempotent.
    runtime
        .ingest_model_run_for(
            &KvTaintRequest {
                request: &request("approx"),
                producer: "p",
                generation_parameters: b"",
            },
            &run_with(&a, KvReuseMode::Approximate),
        )
        .unwrap();
    assert_eq!(runtime.committed_events().len(), before + 1);
}

#[test]
fn the_recipe_depends_on_request_producer_and_parameters() {
    let a = action(b"x");
    let recipe_of = |req: &SegmentedModelRequest, producer: &str, params: &[u8]| {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime
            .ingest_model_run_for(
                &KvTaintRequest {
                    request: req,
                    producer,
                    generation_parameters: params,
                },
                &run_with(&a, KvReuseMode::Approximate),
            )
            .unwrap();
        match &taint_records(&runtime)[0] {
            LedgerEvent::KvTaintCommitted {
                change: KvTaintChange::Record { recipe_digest, .. },
                ..
            } => *recipe_digest,
            _ => unreachable!(),
        }
    };
    let base = recipe_of(&request("r"), "p", b"g");
    assert_eq!(base, recipe_of(&request("r"), "p", b"g"));
    assert_ne!(base, recipe_of(&request("r"), "q", b"g"));
    assert_ne!(base, recipe_of(&request("r"), "p", b"h"));
    // The request id is not part of the recipe (the lineage digest excludes it): a clean
    // recomputation of the same content under a new request must reproduce it to clear.
    assert_eq!(base, recipe_of(&request("s"), "p", b"g"));
    // Different content is a different recipe.
    let tokenizer = ByteTokenizer::new();
    let other = SegmentedModelRequest::new(
        RequestId::from("r"),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        tokenizer.identity().clone(),
        vec![PromptSegment::raw_from_text(&tokenizer, "other text").unwrap()],
    )
    .unwrap();
    assert_ne!(base, recipe_of(&other, "p", b"g"));
}

/// Overwrite the bytes `offset` after the first `pattern` in a sealed snapshot with `to` and
/// seal it again, so only the restore's own checks can refuse it.
fn resealed_with(
    snapshot: &ptr_runtime::compacted::CompactedSnapshot,
    pattern: &[u8],
    offset: usize,
    to: &[u8],
) -> (Vec<u8>, CompactedAnchor) {
    let mut bytes = snapshot.bytes().to_vec();
    let at = bytes
        .windows(pattern.len())
        .position(|window| window == pattern)
        .expect("pattern present in the snapshot")
        + offset;
    bytes[at..at + to.len()].copy_from_slice(to);
    let end = bytes.len() - 32;
    let digest = integrity::sha256(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest);
    let mut trusted = snapshot.anchor();
    trusted.digest = digest;
    (bytes, trusted)
}

#[test]
fn a_malformed_taint_entry_in_a_restored_state_fails_closed() {
    let tainted = action(b"approximate");
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .ingest_model_run(&run_with(&tainted, KvReuseMode::Approximate))
        .unwrap();
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    // Unchanged, resealed: restores (the harness itself is sound).
    let (bytes, trusted) = resealed_with(&snapshot, b"approx:", 0, b"approx:");
    assert!(PtrRuntime::restore_compacted(PtrConfig::default(), &bytes, trusted, &[]).is_ok());
    for (what, pattern, offset, to) in [
        ("unknown value form", &b"approx:"[..], 6, &b";"[..]),
        (
            "unknown subject kind",
            &b"kv-taint:action:"[..],
            15,
            &b"N"[..],
        ),
        ("non-hex digit in the report", &b"approx:"[..], 7, &b"G"[..]),
        (
            "uppercase subject digest",
            &b"kv-taint:action:"[..],
            16,
            &b"F"[..],
        ),
    ] {
        let (bytes, trusted) = resealed_with(&snapshot, pattern, offset, to);
        let result =
            PtrRuntime::restore_compacted(PtrConfig::default(), &bytes, trusted, &[]).err();
        assert!(
            matches!(
                result,
                Some(RuntimeError::KvTaint(
                    KvTaintError::MalformedStateEntry { .. }
                ))
            ),
            "{what}: {result:?}"
        );
    }
}
