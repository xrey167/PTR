#[path = "common/semantic.rs"]
mod semantic_common;
use ptr_config::PtrConfig;
use ptr_ledger::{
    integrity, Attestation, CommittedEvent, FileLedger, LedgerEvent, MergeAuthorityRecord,
    MergeRecord, SemanticOrigin,
};
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_runtime::{
    persistence::SnapshotAnchor,
    semantic::{pod_output_key, request_raw_key},
    PtrRuntime, RuntimeError,
};
use ptr_semdb::{SemanticDelta, SemanticError, SemanticSnapshot, SemanticValue};
use ptr_types::{
    CommitIndex, Generation, PodId, Probability, ProjectId, RequestId, Revision, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use semantic_common::{granted, host_write, host_write_now};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ptr-semantic-{}-{}-{}",
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
fn delta(key: &str, value: &str) -> SemanticDelta {
    let mut d = SemanticDelta::default();
    d.upserts.insert(key.into(), value.into());
    d
}
fn entries(s: &SemanticSnapshot) -> Vec<(String, SemanticValue, Vec<String>)> {
    s.keys()
        .map(|k| {
            (
                k.to_owned(),
                s.value(k).unwrap().clone(),
                s.inputs(k).map(str::to_owned).collect(),
            )
        })
        .collect()
}
fn chain() -> SemanticDelta {
    let mut d = delta("source", "original");
    d.upserts.insert("derived".into(), "derived text".into());
    d.upserts.insert("plan".into(), "derived plan".into());
    d.upserts.insert("unrelated".into(), "keep".into());
    d.dependencies
        .insert("derived".into(), ["source".into()].into());
    d.dependencies
        .insert("plan".into(), ["derived".into()].into());
    d
}

#[test]
fn reopen_restores_exact_contents_dependency_state_and_revision() {
    let tmp = Temp::new();
    let mut r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut r);
    r.ingest_text("r1".into(), "München\n東京\0").unwrap();
    host_write_now(&mut r, chain()).unwrap();
    r.commit(LedgerEvent::CapsuleCommitted {
        project: "p".into(),
        capsule: "a".into(),
        generation: Generation(2),
    })
    .unwrap();
    let before = r.snapshot();
    let committed = r.committed_events().to_vec();
    assert_eq!(before.revision, Revision(2));
    drop(r);
    let mut restored = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut restored);
    assert_eq!(entries(&before), entries(&restored.snapshot()));
    assert_eq!(before.revision, restored.revision());
    assert_eq!(restored.committed_events(), committed);
    assert_eq!(restored.live_generation("a"), Some(Generation(2)));
    assert!(!restored.snapshot_is_current(&before));
    assert!(restored.snapshot_is_current(&restored.snapshot()));
    let changed = host_write_now(&mut restored, delta("source", "new")).unwrap();
    assert_eq!(
        changed.affected,
        ["source".into(), "derived".into(), "plan".into()].into()
    );
    assert_eq!(restored.snapshot().get("derived"), None);
    assert_eq!(restored.snapshot().get("unrelated"), Some("keep"));
    let after_edit = restored.snapshot();
    drop(restored);
    let mut restored = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut restored);
    assert_eq!(entries(&after_edit), entries(&restored.snapshot()));
    let mut remove = SemanticDelta::default();
    remove.removals.insert("source".into());
    host_write_now(&mut restored, remove).unwrap();
    drop(restored);
    let restored = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    assert_eq!(restored.revision(), Revision(4));
    assert_eq!(restored.snapshot().get("source"), None);
    assert_eq!(
        restored.snapshot().inputs("derived").collect::<Vec<_>>(),
        vec!["source"]
    );
}

#[test]
fn decoded_in_memory_replay_and_file_replay_are_equivalent() {
    let tmp = Temp::new();
    let mut r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut r);
    for i in 0..50 {
        let key = format!("input:{}", i % 7);
        let mut d = delta(&key, &i.to_string());
        if i % 4 == 0 {
            d.upserts.clear();
            d.removals.insert(key);
        }
        host_write_now(&mut r, d).unwrap();
    }
    let expected = r.snapshot();
    let history = r.committed_events().to_vec();
    drop(r);
    let memory = PtrRuntime::replay(PtrConfig::default(), &history).unwrap();
    let file = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    assert_eq!(entries(&expected), entries(&memory.snapshot()));
    assert_eq!(entries(&expected), entries(&file.snapshot()));
    assert_eq!(memory.revision(), expected.revision);
    assert_eq!(file.revision(), expected.revision);
    assert_eq!(
        file.materialized_state().values.get("semdb:revision"),
        Some(&expected.revision.0.to_string())
    );
}

#[test]
fn invalid_and_stale_updates_never_enter_history_or_publish_partial_state() {
    let tmp = Temp::new();
    let mut r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut r);
    host_write(&mut r, Revision(0), chain()).unwrap();
    drop(r);
    let bytes = std::fs::read(tmp.log()).unwrap();
    let mut r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut r);
    let original = r.snapshot();
    assert!(matches!(
        host_write(&mut r, Revision(0), delta("source", "wrong base")),
        Err(RuntimeError::Semantic(
            SemanticError::RevisionMismatch { .. }
        ))
    ));
    let mut conflict = delta("source", "must not publish");
    conflict.removals.insert("source".into());
    assert!(host_write_now(&mut r, conflict).is_err());
    let mut missing = delta("derived", "missing input");
    missing
        .dependencies
        .insert("derived".into(), ["absent".into()].into());
    assert!(host_write_now(&mut r, missing).is_err());
    assert_eq!(r.committed_events().len(), 1);
    assert!(r.snapshot_is_current(&original));
    assert_eq!(entries(&r.snapshot()), entries(&original));
    drop(r);
    assert_eq!(bytes, std::fs::read(tmp.log()).unwrap());
}

#[test]
fn generic_commit_and_replay_reject_invalid_schema_base_result_revision_and_noop_records() {
    let encoded = delta("source", "one").encode().unwrap();
    let cases = [
        (Revision(1), Revision(2), encoded.clone()),
        (Revision(0), Revision(9), encoded.clone()),
        (Revision(0), Revision(1), b"unknown-schema".to_vec()),
        (
            Revision(0),
            Revision(1),
            SemanticDelta::default().encode().unwrap(),
        ),
    ];
    for (base_revision, revision, encoded_delta) in cases {
        let event = LedgerEvent::SemanticDeltaCommitted {
            base_revision,
            revision,
            encoded_delta,
            origin: ptr_ledger::SemanticOrigin::Legacy,
        };
        let mut r = PtrRuntime::new(PtrConfig::default()).unwrap();
        // No caller commits a semantic record; replay still judges one.
        assert_eq!(
            r.commit(event.clone()).err(),
            Some(RuntimeError::SemanticRecordOutsideSemanticPath)
        );
        assert!(r.committed_events().is_empty());
        assert_eq!(r.revision(), Revision(0));
        let history = [CommittedEvent {
            index: CommitIndex(1),
            event: event.clone(),
        }];
        assert!(matches!(
            PtrRuntime::replay(PtrConfig::default(), &history),
            Err(RuntimeError::Semantic(_))
        ));
        // Complete invalid records fail closed rather than becoming crash tails.
        let tmp = Temp::new();
        let mut log = FileLedger::open(tmp.log()).unwrap();
        log.append_durable(event).unwrap();
        drop(log);
        let before = std::fs::read(tmp.log()).unwrap();
        assert!(matches!(
            PtrRuntime::open_durable(PtrConfig::default(), tmp.log()),
            Err(RuntimeError::Semantic(_))
        ));
        assert_eq!(before, std::fs::read(tmp.log()).unwrap());
    }
}

/// A recovery snapshot holding exactly `history`, sealed as the runtime seals one.
fn recovery_snapshot(history: &[CommittedEvent]) -> (Vec<u8>, SnapshotAnchor) {
    let log = integrity::encode_log(history).unwrap();
    let anchor = integrity::decode_log(&log).unwrap().anchor();
    let mut bytes = b"PTRSN001".to_vec();
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&anchor.index.0.to_le_bytes());
    bytes.extend_from_slice(&(log.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&anchor.digest);
    bytes.extend_from_slice(&log);
    let digest = integrity::sha256(&bytes);
    bytes.extend_from_slice(&digest);
    let trusted = SnapshotAnchor {
        revision: Revision(0),
        log: anchor,
        digest,
    };
    (bytes, trusted)
}

#[test]
fn a_merge_record_is_refused_on_every_rebuild_path_by_a_build_that_does_not_merge() {
    let encoded = delta("source", "one").encode().unwrap();
    let record = |origin| LedgerEvent::SemanticDeltaCommitted {
        base_revision: Revision(0),
        revision: Revision(1),
        encoded_delta: encoded.clone(),
        origin,
    };
    // The same record without an origin replays, so each refusal below is
    // the origin's alone.
    let legacy = [CommittedEvent {
        index: CommitIndex(1),
        event: record(SemanticOrigin::Legacy),
    }];
    assert!(PtrRuntime::replay(PtrConfig::default(), &legacy).is_ok());

    let event = record(SemanticOrigin::Merge(MergeRecord {
        branch: "b1".into(),
        author: "agent-1".into(),
        seal: [1; 32],
        plan: [2; 32],
        dependencies: [3; 32],
        rebased: Default::default(),
        verification: Attestation {
            required: VerificationLevel::Deterministic,
            level: VerificationLevel::Deterministic,
            verifiers: vec![semantic_common::ACCEPT_ALL.into()],
            findings: Vec::new(),
        },
        authority: MergeAuthorityRecord::Reviewed {
            reviewer: "reviewer-1".into(),
        },
    }));
    // No caller commits a semantic record of any origin.
    let mut r = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert_eq!(
        r.commit(event.clone()).err(),
        Some(RuntimeError::SemanticRecordOutsideSemanticPath)
    );
    assert!(r.committed_events().is_empty());

    let history = [CommittedEvent {
        index: CommitIndex(1),
        event: event.clone(),
    }];
    let tmp = Temp::new();
    let anchor = {
        let mut log = FileLedger::open(tmp.log()).unwrap();
        log.append_durable(event.clone()).unwrap();
        log.anchor().unwrap()
    };
    let before = std::fs::read(tmp.log()).unwrap();
    let (snapshot, trusted) = recovery_snapshot(&history);
    let empty = PtrRuntime::new(PtrConfig::default())
        .unwrap()
        .export_compacted_snapshot()
        .unwrap();
    let rebuilt = [
        (
            "replay",
            PtrRuntime::replay(PtrConfig::default(), &history).err(),
        ),
        (
            "open_durable",
            PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).err(),
        ),
        (
            "open_durable_at",
            PtrRuntime::open_durable_at(PtrConfig::default(), tmp.log(), anchor).err(),
        ),
        (
            "restore_recovery_snapshot",
            PtrRuntime::restore_recovery_snapshot(PtrConfig::default(), &snapshot, trusted).err(),
        ),
        (
            "restore_compacted",
            PtrRuntime::restore_compacted(
                PtrConfig::default(),
                empty.bytes(),
                empty.anchor(),
                &history,
            )
            .err(),
        ),
    ];
    for (path, outcome) in rebuilt {
        assert_eq!(
            outcome,
            Some(RuntimeError::InvalidSemanticOrigin {
                index: Some(CommitIndex(1)),
                reason: "merge records need a build that merges",
            }),
            "{path}"
        );
    }
    // Refusing to open never rewrites the log.
    assert_eq!(before, std::fs::read(tmp.log()).unwrap());
}

#[test]
fn noop_ingestion_has_no_extra_record_and_survives_reopen() {
    let tmp = Temp::new();
    let mut r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut r);
    assert_eq!(r.ingest_text("same".into(), "text").unwrap(), Revision(1));
    for _ in 0..3 {
        assert_eq!(r.ingest_text("same".into(), "text").unwrap(), Revision(1));
    }
    let empty = host_write_now(&mut r, SemanticDelta::default()).unwrap();
    assert_eq!(empty.commit_index, None);
    assert!(empty.affected.is_empty());
    assert_eq!(r.committed_events().len(), 1);
    drop(r);
    let r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    assert_eq!(r.revision(), Revision(1));
    assert_eq!(
        r.snapshot().get(&request_raw_key(&"same".into())),
        Some("text")
    );
}

#[test]
fn all_incomplete_semantic_tail_prefixes_recover_only_prior_complete_transactions() {
    let tmp = Temp::new();
    let mut r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut r);
    host_write(&mut r, Revision(0), chain()).unwrap();
    drop(r);
    let prefix = std::fs::read(tmp.log()).unwrap();
    let mut r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut r);
    host_write_now(&mut r, delta("source", "new value")).unwrap();
    drop(r);
    let complete = std::fs::read(tmp.log()).unwrap();
    let tail = &complete[prefix.len()..];
    for end in 0..tail.len() {
        // Synthetic unacknowledged partial tail, not a hardware power-loss proof.
        let path = tmp.0.join(format!("tail-{end}"));
        std::fs::write(&path, [&prefix[..], &tail[..end]].concat()).unwrap();
        let anchor = ptr_ledger::integrity::decode_log(&prefix).unwrap().anchor();
        if end > 0 {
            assert!(PtrRuntime::open_durable(PtrConfig::default(), &path).is_err());
            assert_eq!(
                std::fs::read(&path).unwrap(),
                [&prefix[..], &tail[..end]].concat()
            );
        }
        drop(FileLedger::recover_unacknowledged_tail(&path, anchor).unwrap());
        let r = PtrRuntime::open_durable_at(PtrConfig::default(), &path, anchor).unwrap();
        assert_eq!(r.revision(), Revision(1), "cut {end}");
        assert_eq!(r.snapshot().get("source"), Some("original"));
        assert_eq!(r.snapshot().get("plan"), Some("derived plan"));
        drop(r);
        assert_eq!(std::fs::read(path).unwrap(), prefix);
    }
    let r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    assert_eq!(r.revision(), Revision(2));
    assert_eq!(r.snapshot().get("plan"), None);
}

#[test]
fn semantic_append_then_exit_child() {
    let Some(path) = std::env::var_os("PTR_SEMANTIC_EXIT_LOG") else {
        return;
    };
    let mut ledger = FileLedger::open(path).unwrap();
    ledger
        .append_durable(LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(1),
            revision: Revision(2),
            encoded_delta: delta("source", "child committed").encode().unwrap(),
            // After the parent's attested writes a record without an origin
            // would be refused on replay, so the child writes what a host
            // write under the fixture grant records.
            origin: SemanticOrigin::Host {
                principal: semantic_common::operator().0,
                verification: Attestation {
                    required: VerificationLevel::Deterministic,
                    level: VerificationLevel::Deterministic,
                    verifiers: vec![semantic_common::ACCEPT_ALL.into()],
                    findings: Vec::new(),
                },
            },
        })
        .unwrap();
    std::process::exit(23);
}

#[test]
fn process_exit_after_durable_append_before_materialization_is_replayed() {
    let tmp = Temp::new();
    let mut r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut r);
    host_write(&mut r, Revision(0), chain()).unwrap();
    drop(r);
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "semantic_append_then_exit_child", "--nocapture"])
        .env("PTR_SEMANTIC_EXIT_LOG", tmp.log())
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(23),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    assert_eq!(r.revision(), Revision(2));
    assert_eq!(r.snapshot().get("source"), Some("child committed"));
    assert_eq!(r.snapshot().get("derived"), None);
    assert_eq!(r.snapshot().get("unrelated"), Some("keep"));
}

struct Want(Vec<u8>);
impl InferenceBackend for Want {
    fn name(&self) -> &'static str {
        "semantic-durability-fixture"
    }
    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::PodRequested {
            capability: "echo".into(),
            input_type: "bytes".into(),
            payload: self.0.clone(),
        }])
    }
}
impl ResumableInferenceBackend for Want {
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        assert_eq!(request.observation.payload, self.0);
        assert_eq!(request.observation.type_id.0, "bytes");
        assert_eq!(request.observation.source, "echo");
        assert_eq!(request.observation.revision, request.revision);
        Ok(vec![ModelEvent::Finished])
    }
}
struct Echo(PodManifest);
impl DynPod for Echo {
    fn manifest(&self) -> &PodManifest {
        &self.0
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
#[test]
fn same_type_changed_pod_bytes_advance_revision_and_resume_from_durable_observation() {
    let tmp = Temp::new();
    let mut pods = PodRegistry::default();
    pods.register(Arc::new(Echo(PodManifest {
        project: ProjectId::from("p"),
        id: "echo".into(),
        capabilities: vec!["echo".into()],
        accepts: vec!["bytes".into()],
        produces: vec!["bytes".into()],
        effects: vec![ptr_types::Effect::Pure],
        protocol_version: 1,
    })));
    let request = RequestId::from("r:separator");
    let pod = PodId::from("echo");
    let mut r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut r);
    for (bytes, expected) in [
        (vec![0, 255, 1], 2),
        (vec![0, 255, 2], 3),
        (vec![0, 255, 2], 3),
    ] {
        r.run_resumable_with_pods(
            request.clone(),
            &ProjectId::from("p"),
            "unchanged request",
            &Want(bytes.clone()),
            &pods,
            &Pass,
            1,
        )
        .unwrap();
        assert_eq!(r.revision(), Revision(expected));
        assert_eq!(
            r.snapshot()
                .payload(&pod_output_key(&request, &pod))
                .unwrap()
                .bytes,
            bytes
        );
    }
    assert_eq!(r.committed_events().len(), 3);
    drop(r);
    let mut r = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    granted(&mut r);
    let snapshot = r.snapshot();
    let stored = snapshot.payload(&pod_output_key(&request, &pod)).unwrap();
    assert_eq!(stored.bytes, [0, 255, 2]);
    assert_eq!(stored.source, "echo");
    assert_eq!(stored.type_id.0, "bytes");
    r.ingest_text(request.clone(), "changed request").unwrap();
    assert!(r
        .snapshot()
        .payload(&pod_output_key(&request, &pod))
        .is_none());
    assert!(!r.snapshot_is_current(&snapshot));
}

#[test]
fn observation_keys_cannot_alias_through_embedded_delimiters() {
    assert_ne!(
        pod_output_key(&"a:b".into(), &"c".into()),
        pod_output_key(&"a".into(), &"b:c".into())
    );
    assert_ne!(
        pod_output_key(&"a".into(), &"b".into()),
        request_raw_key(&"a:pod:b".into())
    );
}
