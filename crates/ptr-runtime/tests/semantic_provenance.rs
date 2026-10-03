//! What replay accepts as a semantic record's origin, on every path that
//! rebuilds a runtime from history: a record without an origin only before
//! the first attributed one (R1), ingress records only in the exact shapes
//! ingress writes (R2), host writes and merges only with an attestation that
//! holds and no ingress key written or evicted (R3), and merges only with the
//! plan digest of their own delta, no dependency entry, and once per branch
//! (R4). The runtime never writes a record its own replay refuses.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ptr_branch::{
    counter_value, merge_plan_digest, AutoThreshold, Branch, BranchId, BranchOp, PolicyRecord,
    TriagePolicy,
};
use ptr_config::PtrConfig;
use ptr_ledger::{
    integrity, Attestation, CommittedEvent, FileLedger, LedgerEvent, MergeAuthorityRecord,
    MergeRecord, SemanticOrigin,
};
use ptr_model_api::{InferenceBackend, ModelError, ModelEvent, ModelRequest};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::persistence::SnapshotAnchor;
use ptr_runtime::semantic::{pod_output_key, request_raw_key};
use ptr_runtime::{MergeAuthority, PtrRuntime, RuntimeError, SemanticGrant};
use ptr_semdb::{SemanticDelta, SemanticPayload, SemanticValue};
use ptr_types::{
    CapsuleId, CommitIndex, Generation, PodId, PrincipalId, Probability, ProjectId, RequestId,
    Revision, VerificationLevel,
};
use ptr_verifier::{VerificationReport, Verifier};

#[path = "common/semantic.rs"]
mod semantic_common;
use semantic_common::{finding, pass, AcceptAll, FnVerifier, ACCEPT_ALL};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ptr-provenance-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn log(&self) -> PathBuf {
        self.0.join("log")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn delta(key: &str, value: &str) -> SemanticDelta {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.into(), value.into());
    delta
}

/// A semantic record from revision `base` to `base + 1`.
fn semantic(base: u64, delta: &SemanticDelta, origin: SemanticOrigin) -> LedgerEvent {
    LedgerEvent::SemanticDeltaCommitted {
        base_revision: Revision(base),
        revision: Revision(base + 1),
        encoded_delta: delta.encode().unwrap(),
        origin,
    }
}

/// `events` committed at indices 1, 2, …
fn history(events: Vec<LedgerEvent>) -> Vec<CommittedEvent> {
    events
        .into_iter()
        .enumerate()
        .map(|(offset, event)| CommittedEvent {
            index: CommitIndex(offset as u64 + 1),
            event,
        })
        .collect()
}

fn attestation() -> Attestation {
    Attestation {
        required: VerificationLevel::Deterministic,
        level: VerificationLevel::Deterministic,
        verifiers: vec![ACCEPT_ALL.into()],
        findings: Vec::new(),
    }
}

fn host(verification: Attestation) -> SemanticOrigin {
    SemanticOrigin::Host {
        principal: "test-operator".into(),
        verification,
    }
}

/// A recovery snapshot holding exactly `history`, sealed as the runtime seals
/// one, at the revision its last semantic record claims.
fn recovery_snapshot(history: &[CommittedEvent]) -> (Vec<u8>, SnapshotAnchor) {
    let revision = history
        .iter()
        .rev()
        .find_map(|committed| match &committed.event {
            LedgerEvent::SemanticDeltaCommitted { revision, .. } => Some(*revision),
            _ => None,
        })
        .unwrap_or(Revision(0));
    let log = integrity::encode_log(history).unwrap();
    let anchor = integrity::decode_log(&log).unwrap().anchor();
    let mut bytes = b"PTRSN001".to_vec();
    bytes.extend_from_slice(&revision.0.to_le_bytes());
    bytes.extend_from_slice(&anchor.index.0.to_le_bytes());
    bytes.extend_from_slice(&(log.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&anchor.digest);
    bytes.extend_from_slice(&log);
    let digest = integrity::sha256(&bytes);
    bytes.extend_from_slice(&digest);
    let trusted = SnapshotAnchor {
        revision,
        log: anchor,
        digest,
    };
    (bytes, trusted)
}

/// What a rebuilt runtime holds that the runtime which wrote its history
/// must hold too: its revision, its materialized state, and every semantic
/// key with its value and its inputs.
type Held = (
    Revision,
    BTreeMap<String, String>,
    Vec<(String, Option<SemanticValue>, Vec<String>)>,
);

fn held(runtime: &PtrRuntime) -> Held {
    let snapshot = runtime.snapshot();
    (
        runtime.revision(),
        runtime.materialized_state().values.clone(),
        snapshot
            .keys()
            .map(|key| {
                (
                    key.to_owned(),
                    snapshot.value(key).cloned(),
                    snapshot.inputs(key).map(str::to_owned).collect(),
                )
            })
            .collect(),
    )
}

/// What every path that rebuilds a runtime from `history` answers: replay,
/// `open_durable` and `open_durable_at` over a log holding it,
/// `restore_recovery_snapshot` over a snapshot of it, and `restore_compacted`
/// from a compacted snapshot of its first `floor` records with the rest above
/// the floor; for a path that opened, what the runtime holds.
fn rebuilt_held(
    history: &[CommittedEvent],
    floor: usize,
) -> Vec<(&'static str, Result<Held, RuntimeError>)> {
    let temp = Temp::new();
    let anchor = {
        let mut log = FileLedger::open(temp.log()).unwrap();
        for committed in history {
            assert_eq!(
                log.append_durable(committed.event.clone()).unwrap(),
                committed.index
            );
        }
        log.anchor().unwrap()
    };
    let before = std::fs::read(temp.log()).unwrap();
    let (snapshot, trusted) = recovery_snapshot(history);
    let compacted = PtrRuntime::replay(PtrConfig::default(), &history[..floor])
        .unwrap()
        .export_compacted_snapshot()
        .unwrap();
    // Each runtime is dropped as soon as what it holds is read, so no log is
    // held open when it is read back below.
    let outcomes = vec![
        (
            "replay",
            PtrRuntime::replay(PtrConfig::default(), history).map(|runtime| held(&runtime)),
        ),
        (
            "open_durable",
            PtrRuntime::open_durable(PtrConfig::default(), temp.log())
                .map(|runtime| held(&runtime)),
        ),
        (
            "open_durable_at",
            PtrRuntime::open_durable_at(PtrConfig::default(), temp.log(), anchor)
                .map(|runtime| held(&runtime)),
        ),
        (
            "restore_recovery_snapshot",
            PtrRuntime::restore_recovery_snapshot(PtrConfig::default(), &snapshot, trusted)
                .map(|runtime| held(&runtime)),
        ),
        (
            "restore_compacted",
            PtrRuntime::restore_compacted(
                PtrConfig::default(),
                compacted.bytes(),
                compacted.anchor(),
                &history[floor..],
            )
            .map(|runtime| held(&runtime)),
        ),
    ];
    // Opening, or refusing to, never rewrites the log.
    assert_eq!(before, std::fs::read(temp.log()).unwrap());
    outcomes
}

/// [`rebuilt_held`]'s refusals: `None` for a path that opened.
fn rebuilt(history: &[CommittedEvent], floor: usize) -> Vec<(&'static str, Option<RuntimeError>)> {
    rebuilt_held(history, floor)
        .into_iter()
        .map(|(path, outcome)| (path, outcome.err()))
        .collect()
}

fn assert_every_path(history: &[CommittedEvent], floor: usize, expected: Option<RuntimeError>) {
    for (path, outcome) in rebuilt(history, floor) {
        assert_eq!(outcome, expected, "{path}");
    }
}

#[test]
fn a_legacy_record_after_an_attested_one_is_refused_on_replay_open_and_after_compaction() {
    let legacy_then_attested = history(vec![
        semantic(0, &delta("a", "1"), SemanticOrigin::Legacy),
        semantic(1, &delta("b", "2"), host(attestation())),
    ]);
    // A legacy prefix followed by attributed records opens everywhere.
    assert_every_path(&legacy_then_attested, 1, None);

    let after = history(vec![
        semantic(0, &delta("a", "1"), SemanticOrigin::Legacy),
        semantic(1, &delta("b", "2"), host(attestation())),
        semantic(2, &delta("c", "3"), SemanticOrigin::Legacy),
    ]);
    let refused = Some(RuntimeError::LegacySemanticRecord {
        index: CommitIndex(3),
    });
    // Whether the attested record is replayed or lies below a compacted
    // floor, whose materialized marker records it, the legacy record after it
    // is refused.
    for floor in [0, 1, 2] {
        assert_every_path(&after, floor, refused.clone());
    }
}

#[test]
fn a_log_of_only_legacy_records_still_replays() {
    let legacy = history(vec![
        semantic(0, &delta("a", "1"), SemanticOrigin::Legacy),
        LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        },
        semantic(1, &delta("b", "2"), SemanticOrigin::Legacy),
        semantic(2, &delta("a", "3"), SemanticOrigin::Legacy),
    ]);
    for floor in 0..=legacy.len() {
        assert_every_path(&legacy, floor, None);
    }
    let replayed = PtrRuntime::replay(PtrConfig::default(), &legacy).unwrap();
    assert_eq!(replayed.snapshot().get("a"), Some("3"));
    assert!(!replayed
        .materialized_state()
        .values
        .contains_key(ptr_state::ATTESTED_MARKER));
}

#[test]
fn an_ingress_record_of_another_shape_is_refused() {
    let request = RequestId::from("r1");
    let raw = request_raw_key(&request);
    let output = pod_output_key(&request, &PodId::from("echo"));
    let payload = |source: &str| {
        SemanticValue::Payload(SemanticPayload {
            type_id: "bytes".into(),
            source: source.into(),
            bytes: vec![1, 2],
        })
    };
    let request_origin = SemanticOrigin::Request {
        request: "r1".into(),
    };
    let pod_origin = SemanticOrigin::PodOutput {
        request: "r1".into(),
        pod: "echo".into(),
        level: VerificationLevel::Deterministic,
    };
    let pod_shape = |value: SemanticValue, inputs: &[&str]| {
        let mut delta = SemanticDelta::default();
        delta.upserts.insert(output.clone(), value);
        if !inputs.is_empty() {
            delta.dependencies.insert(
                output.clone(),
                inputs.iter().map(|input| (*input).to_owned()).collect(),
            );
        }
        delta
    };
    // The shapes ingress writes replay.
    let valid = history(vec![
        semantic(0, &delta(&raw, "text"), request_origin.clone()),
        semantic(1, &pod_shape(payload("echo"), &[&raw]), pod_origin.clone()),
    ]);
    assert_every_path(&valid, 1, None);

    let mut extra = delta(&raw, "text");
    extra.upserts.insert("other".into(), "x".into());
    let mut with_payload = SemanticDelta::default();
    with_payload.upserts.insert(raw.clone(), payload("echo"));
    let mut with_dependency = delta(&raw, "text");
    with_dependency
        .dependencies
        .insert(raw.clone(), ["other".to_owned()].into());
    let mut with_removal = delta(&raw, "text");
    with_removal.removals.insert("other".into());
    let request_shapes = [
        extra,
        with_payload,
        with_dependency,
        with_removal,
        delta(&request_raw_key(&RequestId::from("r2")), "text"),
    ];
    for shape in request_shapes {
        assert_every_path(
            &history(vec![semantic(0, &shape, request_origin.clone())]),
            0,
            Some(RuntimeError::InvalidSemanticOrigin {
                index: Some(CommitIndex(1)),
                reason: "a request record writes exactly its own raw text",
            }),
        );
    }
    let mut extra_input = pod_shape(payload("echo"), &[&raw]);
    extra_input
        .dependencies
        .get_mut(&output)
        .unwrap()
        .insert("other".into());
    let mut pod_removal = pod_shape(payload("echo"), &[&raw]);
    pod_removal.removals.insert("other".into());
    let mut pod_extra = pod_shape(payload("echo"), &[&raw]);
    pod_extra.upserts.insert("other".into(), "x".into());
    let mut pod_second_dependency = pod_shape(payload("echo"), &[&raw]);
    pod_second_dependency
        .dependencies
        .insert("other".into(), [raw.clone()].into());
    let pod_shapes = [
        pod_shape(payload("other-pod"), &[&raw]),
        pod_shape(payload("echo"), &[]),
        extra_input,
        pod_shape(SemanticValue::Text("text".into()), &[&raw]),
        pod_removal,
        pod_extra,
        pod_second_dependency,
    ];
    for shape in pod_shapes {
        assert_every_path(
            &history(vec![
                semantic(0, &delta(&raw, "text"), request_origin.clone()),
                semantic(1, &shape, pod_origin.clone()),
            ]),
            1,
            Some(RuntimeError::InvalidSemanticOrigin {
                index: Some(CommitIndex(2)),
                reason: "a Pod output record writes exactly the Pod's output for its request",
            }),
        );
    }
}

#[test]
fn an_attested_record_touching_an_ingress_key_is_refused() {
    let raw = request_raw_key(&RequestId::from("r1"));
    let output = pod_output_key(&RequestId::from("r1"), &PodId::from("echo"));
    let mut removal = delta("a", "1");
    removal.removals.insert(output.clone());
    let mut derived = delta("a", "1");
    derived
        .dependencies
        .insert(raw.clone(), ["a".to_owned()].into());
    for touching in [delta(&raw, "forged"), removal, derived] {
        assert_every_path(
            &history(vec![semantic(0, &touching, host(attestation()))]),
            0,
            Some(RuntimeError::InvalidSemanticOrigin {
                index: Some(CommitIndex(1)),
                reason: "a host write writes, removes or derives an ingress key",
            }),
        );
    }
    // An ingress key as a dependency input is not touching it.
    let mut input = delta("summary", "short");
    input
        .dependencies
        .insert("summary".into(), [raw.clone()].into());
    assert_every_path(
        &history(vec![
            semantic(
                0,
                &delta(&raw, "text"),
                SemanticOrigin::Request {
                    request: "r1".into(),
                },
            ),
            semantic(1, &input, host(attestation())),
        ]),
        1,
        None,
    );
    // A principal that is not provenance text.
    for principal in ["", " padded"] {
        assert_every_path(
            &history(vec![semantic(
                0,
                &delta("a", "1"),
                SemanticOrigin::Host {
                    principal: principal.into(),
                    verification: attestation(),
                },
            )]),
            0,
            Some(RuntimeError::InvalidSemanticOrigin {
                index: Some(CommitIndex(1)),
                reason: "a host write's principal is not valid provenance text",
            }),
        );
    }
}

#[test]
fn an_attestation_below_its_required_level_is_refused() {
    let with = |required, level| Attestation {
        required,
        level,
        ..attestation()
    };
    let refused = |reason| {
        Some(RuntimeError::InvalidSemanticOrigin {
            index: Some(CommitIndex(1)),
            reason,
        })
    };
    let below = "an attestation's weakest level does not meet its requirement";
    for (verification, expected) in [
        (
            with(
                VerificationLevel::FullSemantic,
                VerificationLevel::SampleVerified,
            ),
            refused(below),
        ),
        (
            with(
                VerificationLevel::Deterministic,
                VerificationLevel::FullSemantic,
            ),
            refused(below),
        ),
        (
            with(
                VerificationLevel::FullSemantic,
                VerificationLevel::Deterministic,
            ),
            None,
        ),
        (
            with(
                VerificationLevel::FullSemantic,
                VerificationLevel::FullSemantic,
            ),
            None,
        ),
    ] {
        assert_every_path(
            &history(vec![semantic(0, &delta("a", "1"), host(verification))]),
            0,
            expected,
        );
    }
    // Verifier names and findings a grant could not have produced.
    let names = "an attestation names an invalid or repeated verifier";
    let order = "an attestation's findings are not sorted and distinct";
    let unknown = "an attestation finding is not a recorded verifier's valid code";
    let cases = [
        (vec!["a/b"], vec![], names),
        (vec!["same", "same"], vec![], names),
        (vec![" padded"], vec![], names),
        (vec!["v"], vec!["v/b", "v/a"], order),
        (vec!["v"], vec!["v/a", "v/a"], order),
        (vec!["v"], vec!["w/a"], unknown),
        (vec!["v"], vec!["v/"], unknown),
        (vec!["v"], vec!["no-slash"], unknown),
    ];
    for (verifiers, findings, reason) in cases {
        let verification = Attestation {
            verifiers: verifiers.into_iter().map(str::to_owned).collect(),
            findings: findings.into_iter().map(str::to_owned).collect(),
            ..attestation()
        };
        assert_every_path(
            &history(vec![semantic(0, &delta("a", "1"), host(verification))]),
            0,
            refused(reason),
        );
    }
}

#[test]
fn an_attestation_naming_no_verifier_or_more_than_a_record_holds_is_refused() {
    // The ledger refuses to write such a record, so only a history held in
    // memory carries one: replay, and the records a compacted restore replays
    // above its floor.
    let named = |count: usize| (0..count).map(|n| format!("v{n:02}")).collect::<Vec<_>>();
    let noted = |count: usize| {
        (0..count)
            .map(|n| format!("v00/f{n:02}"))
            .collect::<Vec<_>>()
    };
    let verifiers = "an attestation names no verifier, or more than a grant installs";
    let findings = "an attestation carries more soft findings than a record holds";
    let empty = PtrRuntime::new(PtrConfig::default())
        .unwrap()
        .export_compacted_snapshot()
        .unwrap();
    for (names, codes, reason) in [
        (named(0), vec![], verifiers),
        (
            named(ptr_runtime::merge::MAX_SEMANTIC_VERIFIERS + 1),
            vec![],
            verifiers,
        ),
        (
            named(1),
            noted(ptr_runtime::merge::MAX_ATTESTED_FINDINGS + 1),
            findings,
        ),
    ] {
        let event = semantic(
            0,
            &delta("a", "1"),
            host(Attestation {
                verifiers: names,
                findings: codes,
                ..attestation()
            }),
        );
        assert!(ptr_ledger::check_encodable(&event).is_err());
        let history = history(vec![event]);
        let refused = Some(RuntimeError::InvalidSemanticOrigin {
            index: Some(CommitIndex(1)),
            reason,
        });
        assert_eq!(
            PtrRuntime::replay(PtrConfig::default(), &history).err(),
            refused
        );
        assert_eq!(
            PtrRuntime::restore_compacted(
                PtrConfig::default(),
                empty.bytes(),
                empty.anchor(),
                &history,
            )
            .err(),
            refused
        );
    }
    // Exactly the bounds replay on every path.
    assert_every_path(
        &history(vec![semantic(
            0,
            &delta("a", "1"),
            host(Attestation {
                verifiers: named(ptr_runtime::merge::MAX_SEMANTIC_VERIFIERS),
                findings: noted(ptr_runtime::merge::MAX_ATTESTED_FINDINGS),
                ..attestation()
            }),
        )]),
        0,
        None,
    );
}

#[test]
fn a_host_write_that_would_evict_an_ingress_key_is_refused() {
    // Before origins existed any delta could be committed, so a history may
    // hold a Pod's output derived from a key a host write changes. Changing
    // that key would evict the output, a key only ingress writes.
    let request = RequestId::from("r1");
    let output = pod_output_key(&request, &PodId::from("echo"));
    let mut legacy = delta("price", "1");
    legacy.upserts.insert(
        output.clone(),
        SemanticValue::Payload(SemanticPayload {
            type_id: "bytes".into(),
            source: "echo".into(),
            bytes: vec![1],
        }),
    );
    legacy
        .dependencies
        .insert(output.clone(), ["price".to_owned()].into());
    let prefix = semantic(0, &legacy, SemanticOrigin::Legacy);

    // The writer refuses it before any verifier is asked.
    let mut runtime =
        PtrRuntime::replay(PtrConfig::default(), &history(vec![prefix.clone()])).unwrap();
    let asked = Arc::new(AtomicU64::new(0));
    let counted = asked.clone();
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(FnVerifier::new(ACCEPT_ALL, move |_| {
                    counted.fetch_add(1, Ordering::SeqCst);
                    pass(VerificationLevel::Deterministic)
                }))
                .allow_host_writes(),
        )
        .unwrap();
    let events = runtime.committed_events().len();
    assert_eq!(
        runtime.apply_verified_semantic_delta(
            Revision(1),
            delta("price", "2"),
            &PrincipalId::from("test-operator"),
        ),
        Err(RuntimeError::ReservedSemanticNamespace {
            key: output.clone()
        })
    );
    assert_eq!(asked.load(Ordering::SeqCst), 0);
    assert_eq!(runtime.committed_events().len(), events);
    assert!(runtime.snapshot().value(&output).is_some());
    // A write that does not reach the output is admitted.
    runtime
        .apply_verified_semantic_delta(
            Revision(1),
            delta("other", "2"),
            &PrincipalId::from("test-operator"),
        )
        .unwrap();

    // Replay refuses a host record that evicts one, with the output below
    // the floor and above it.
    for floor in [0, 1] {
        assert_every_path(
            &history(vec![
                prefix.clone(),
                semantic(1, &delta("price", "2"), host(attestation())),
            ]),
            floor,
            Some(RuntimeError::InvalidSemanticOrigin {
                index: Some(CommitIndex(2)),
                reason: "a host write or merge evicts an ingress key",
            }),
        );
    }
}

/// The dependency digest every merge record here carries.
const DEPENDENCIES: [u8; 32] = [3; 32];

/// A merge of `branch` from revision `base`, writing `delta` and naming
/// `rebased`, whose plan digest is the one its parts give.
fn merge(branch: &str, base: u64, delta: &SemanticDelta, rebased: &[&str]) -> MergeRecord {
    let rebased: BTreeSet<String> = rebased.iter().map(|key| (*key).to_owned()).collect();
    MergeRecord {
        branch: branch.into(),
        author: "agent-1".into(),
        seal: [1; 32],
        plan: merge_plan_digest(
            branch,
            Revision(base),
            &delta.encode().unwrap(),
            &DEPENDENCIES,
            &rebased,
        ),
        dependencies: DEPENDENCIES,
        rebased,
        verification: attestation(),
        authority: MergeAuthorityRecord::Reviewed {
            reviewer: "reviewer-1".into(),
        },
    }
}

fn refused_at(index: u64, reason: &'static str) -> Option<RuntimeError> {
    Some(RuntimeError::InvalidSemanticOrigin {
        index: Some(CommitIndex(index)),
        reason,
    })
}

#[test]
fn a_merge_record_whose_plan_digest_does_not_match_its_delta_or_rebased_keys_is_refused() {
    let digest = "a merge's plan digest is not the digest of its branch, base, delta, \
                  dependencies and rebased keys";
    let write = delta("a", "1");
    // The record as a merge writes it replays on every path.
    assert_every_path(
        &history(vec![semantic(
            0,
            &write,
            SemanticOrigin::Merge(merge("b1", 0, &write, &["a"])),
        )]),
        0,
        None,
    );
    // A digest of any other plan does not stand for this record's delta.
    let other = |plan| {
        SemanticOrigin::Merge(MergeRecord {
            plan,
            ..merge("b1", 0, &write, &["a"])
        })
    };
    let encoded = write.encode().unwrap();
    let rebased = BTreeSet::from(["a".to_owned()]);
    for plan in [
        // of another delta,
        merge("b1", 0, &delta("a", "2"), &["a"]).plan,
        // with no rebased key where the record names one,
        merge("b1", 0, &write, &[]).plan,
        // of another branch,
        merge("b2", 0, &write, &["a"]).plan,
        // against another revision,
        merge_plan_digest("b1", Revision(1), &encoded, &DEPENDENCIES, &rebased),
        // with other dependencies,
        merge_plan_digest("b1", Revision(0), &encoded, &[4; 32], &rebased),
        // or none at all.
        [0; 32],
    ] {
        assert_every_path(
            &history(vec![semantic(0, &write, other(plan))]),
            0,
            refused_at(1, digest),
        );
    }
    // Rebased keys the record names but the digest does not cover, and keys
    // it names but its delta removes or only ingress writes.
    let widened = MergeRecord {
        rebased: BTreeSet::from(["a".to_owned(), "z".to_owned()]),
        ..merge("b1", 0, &write, &["a"])
    };
    assert_every_path(
        &history(vec![semantic(0, &write, SemanticOrigin::Merge(widened))]),
        0,
        refused_at(1, digest),
    );
    let mut removal = delta("a", "1");
    removal.removals.insert("z".into());
    let seeded = |origin| {
        history(vec![
            semantic(0, &delta("z", "0"), host(attestation())),
            semantic(1, &removal, origin),
        ])
    };
    assert_every_path(
        &seeded(SemanticOrigin::Merge(merge("b1", 1, &removal, &["z"]))),
        1,
        refused_at(2, "a merge's rebased key is removed or reserved to ingress"),
    );
    let raw = request_raw_key(&RequestId::from("r1"));
    assert_every_path(
        &history(vec![semantic(
            0,
            &write,
            SemanticOrigin::Merge(merge("b1", 0, &write, &[raw.as_str()])),
        )]),
        0,
        refused_at(1, "a merge's rebased key is removed or reserved to ingress"),
    );
}

#[test]
fn a_merge_record_breaking_a_provenance_or_attestation_rule_is_refused() {
    let write = delta("a", "1");
    let record = || merge("b1", 0, &write, &[]);
    let long = "a".repeat(ptr_runtime::merge::MAX_PROVENANCE_TEXT + 1);
    let provenance = "a merge's branch or author is not valid provenance text";
    let mut cases: Vec<(MergeRecord, &'static str)> = Vec::new();
    for text in ["", " b1", "b1\n", long.as_str()] {
        // The digest is recomputed for the branch, so only the text is wrong.
        cases.push((merge(text, 0, &write, &[]), provenance));
        cases.push((
            MergeRecord {
                author: text.into(),
                ..record()
            },
            provenance,
        ));
        cases.push((
            MergeRecord {
                authority: MergeAuthorityRecord::Reviewed {
                    reviewer: text.into(),
                },
                ..record()
            },
            "a merge's reviewer is not valid provenance text",
        ));
        cases.push((
            MergeRecord {
                authority: MergeAuthorityRecord::Triage {
                    policy_version: text.into(),
                    score_bits: 0.9_f32.to_bits(),
                },
                ..record()
            },
            "a merge's policy version is not valid provenance text",
        ));
    }
    for score in [f32::NAN, f32::INFINITY, -0.5, 1.5] {
        cases.push((
            MergeRecord {
                authority: MergeAuthorityRecord::Triage {
                    policy_version: "policy-v1".into(),
                    score_bits: score.to_bits(),
                },
                ..record()
            },
            "a merge's triage score is not a probability",
        ));
    }
    cases.push((
        MergeRecord {
            verification: Attestation {
                level: VerificationLevel::SampleVerified,
                ..attestation()
            },
            ..record()
        },
        "an attestation's weakest level does not meet its requirement",
    ));
    let raw = request_raw_key(&RequestId::from("r1"));
    let ingress = delta(&raw, "forged");
    cases.push((
        merge("b1", 0, &ingress, &[]),
        "a merge writes, removes or derives an ingress key",
    ));
    for (record, reason) in cases {
        let written = if record.plan == merge("b1", 0, &ingress, &[]).plan {
            &ingress
        } else {
            &write
        };
        assert_every_path(
            &history(vec![semantic(0, written, SemanticOrigin::Merge(record))]),
            0,
            refused_at(1, reason),
        );
    }
    // Scores at both ends of the range, and a triage authority, replay.
    for score in [0.0_f32, 1.0] {
        let triaged = MergeRecord {
            authority: MergeAuthorityRecord::Triage {
                policy_version: "policy-v1".into(),
                score_bits: score.to_bits(),
            },
            ..record()
        };
        assert_every_path(
            &history(vec![semantic(0, &write, SemanticOrigin::Merge(triaged))]),
            0,
            None,
        );
    }
}

#[test]
fn a_merge_record_with_dependency_entries_is_refused() {
    // Certification never rewires a key: a merge's delta carries no
    // dependency entry, even one a host write could carry.
    let mut derived = delta("a", "2");
    derived
        .dependencies
        .insert("a".into(), ["b".to_owned()].into());
    let seeded = |origin| {
        history(vec![
            semantic(0, &delta("b", "1"), host(attestation())),
            semantic(1, &derived, origin),
        ])
    };
    assert_every_path(&seeded(host(attestation())), 1, None);
    assert_every_path(
        &seeded(SemanticOrigin::Merge(merge("b1", 1, &derived, &[]))),
        1,
        refused_at(2, "a merge carries a dependency entry"),
    );
}

#[test]
fn a_second_merge_record_of_one_branch_is_refused_also_across_compaction() {
    let first = delta("a", "1");
    let second = delta("a", "2");
    let merges = |branch: &str| {
        history(vec![
            semantic(
                0,
                &first,
                SemanticOrigin::Merge(merge("b1", 0, &first, &[])),
            ),
            semantic(
                1,
                &second,
                SemanticOrigin::Merge(merge(branch, 1, &second, &[])),
            ),
        ])
    };
    // With the first merge replayed above the floor, and below it, carried
    // by the compacted snapshot's materialized state.
    for floor in [0, 1] {
        assert_every_path(
            &merges("b1"),
            floor,
            refused_at(2, "a merge of a branch that is already merged"),
        );
        // Another branch merges, and so does an id that only looks like it
        // under another length.
        for other in ["b2", "b1:", "1:b1", "b"] {
            assert_every_path(&merges(other), floor, None);
        }
    }
}

struct Asks(Vec<u8>);
impl InferenceBackend for Asks {
    fn name(&self) -> &'static str {
        "provenance-fixture"
    }
    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::PodRequested {
            capability: "echo".into(),
            input_type: "bytes".into(),
            payload: self.0.clone(),
        }])
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

struct Passes;
impl Verifier<TypedPayload> for Passes {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        pass(VerificationLevel::SampleVerified)
    }
}

/// A small deterministic generator, so each seed is one reproducible history.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

#[test]
fn the_runtime_never_writes_a_record_its_replay_refuses() {
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
    let keys = ["a", "b", "c", "d"];
    let mut ingress_refusals = 0;
    // What the histories held, so the property is known to cover each kind
    // of record: requests, Pod outputs, host writes with and without soft
    // findings, derived keys, and clean and rebased merges under triage and
    // review.
    let (mut requests, mut outputs, mut plain, mut noted, mut derived) = (0, 0, 0, 0, 0);
    let (mut clean, mut rebased, mut triaged, mut reviewed) = (0, 0, 0, 0);
    for seed in 0..200u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        // Two verifiers: one always deterministic, one whose level and soft
        // findings vary with the change, so records carry varied attestations.
        runtime
            .install_semantic_grant(
                SemanticGrant::new(RequiredVerification::FullSemantic)
                    .with_verifier(AcceptAll)
                    .with_merge_policy(
                        PolicyRecord::manual(
                            "policy-v1",
                            TriagePolicy::new(AutoThreshold::AtLeast(0.8), 0.25).unwrap(),
                        )
                        .unwrap(),
                        seed,
                    )
                    .with_reviewer(PrincipalId::from("reviewer-1"))
                    .with_verifier(FnVerifier::new("varied", |change| {
                        let next = change.next_revision().0;
                        let mut report = if next % 2 == 0 {
                            pass(VerificationLevel::FullSemantic)
                        } else {
                            pass(VerificationLevel::Deterministic)
                        };
                        if next % 3 == 0 {
                            report.findings.push(finding("style", false));
                        }
                        if next % 5 == 0 {
                            report.findings.push(finding("alpha", false));
                        }
                        report
                    }))
                    .allow_host_writes(),
            )
            .unwrap();
        let mut generation = 0;
        for _ in 0..20 {
            let request = RequestId(format!("r{}", rng.below(3)));
            match rng.below(8) {
                0 => {
                    let text = format!("text {}", rng.below(4));
                    runtime.ingest_text(request, text).unwrap();
                }
                1 => {
                    let key = keys[rng.below(4) as usize];
                    let mut write = delta(key, &format!("v{}", rng.below(5)));
                    if rng.below(3) == 0 {
                        // A derived key, from another key or a request's text.
                        let input = if rng.below(2) == 0 {
                            keys[rng.below(4) as usize].to_owned()
                        } else {
                            request_raw_key(&request)
                        };
                        if input != key {
                            write.dependencies.insert(key.into(), [input].into());
                        }
                    }
                    let revision = runtime.revision();
                    // A dependency on an absent input is refused before
                    // anything is written, as it should be.
                    let _ = runtime.apply_verified_semantic_delta(
                        revision,
                        write,
                        &"test-operator".into(),
                    );
                }
                2 => {
                    let mut removal = SemanticDelta::default();
                    removal.removals.insert(keys[rng.below(4) as usize].into());
                    let revision = runtime.revision();
                    let _ = runtime.apply_verified_semantic_delta(
                        revision,
                        removal,
                        &"test-operator".into(),
                    );
                }
                3 => {
                    generation += 1;
                    runtime
                        .commit(LedgerEvent::CapsuleCommitted {
                            project: ProjectId::from("p"),
                            capsule: CapsuleId::from("capsule:a"),
                            generation: Generation(generation),
                        })
                        .unwrap();
                }
                4 => {
                    let payload = vec![rng.below(256) as u8; 1 + rng.below(3) as usize];
                    runtime
                        .run_model_with_pods(
                            request,
                            &ProjectId::from("p"),
                            format!("asks {}", rng.below(3)),
                            &Asks(payload),
                            &pods,
                            &Passes,
                        )
                        .unwrap();
                }
                6 | 7 => {
                    // A branch over the current state, merged under triage or
                    // on review of its preview. Refusals and holds write
                    // nothing; a merge that commits is checked below like
                    // every other record.
                    let key = keys[rng.below(4) as usize];
                    let counter = format!("count:{key}");
                    let id = BranchId(format!("b{}", rng.below(12)));
                    let mut work = Branch::open(id, "agent-1".into(), runtime.snapshot());
                    let staged = if rng.below(2) == 0 {
                        work.read(key)
                            .and_then(|_| work.put(key, format!("m{}", rng.below(5)).into()))
                    } else {
                        work.stage_commutative(BranchOp::Add {
                            key: counter.clone(),
                            amount: 1 + rng.below(3) as i64,
                        })
                    };
                    let Ok(sealed) = staged.and_then(|_| work.seal()) else {
                        continue;
                    };
                    if rng.below(2) == 0 {
                        // Another writer moves the counter, so the addition
                        // rebases.
                        let revision = runtime.revision();
                        let _ = runtime.apply_verified_semantic_delta(
                            revision,
                            {
                                let mut write = SemanticDelta::default();
                                write
                                    .upserts
                                    .insert(counter, counter_value(rng.below(9) as i64));
                                write
                            },
                            &"test-operator".into(),
                        );
                    }
                    let authority = match (rng.below(2), runtime.preview_merge(&sealed)) {
                        (0, Ok(preview)) => MergeAuthority::Reviewed {
                            plan_digest: preview.plan_digest,
                            reviewer: PrincipalId::from("reviewer-1"),
                        },
                        _ => MergeAuthority::Triage {
                            score: Probability::new(0.5 + rng.below(50) as f32 / 100.0).unwrap(),
                        },
                    };
                    let events = runtime.committed_events().len();
                    match runtime.merge_branch(&sealed, authority) {
                        Ok(ptr_runtime::MergeOutcome::Committed(_)) => {
                            assert_eq!(runtime.committed_events().len(), events + 1)
                        }
                        _ => assert_eq!(runtime.committed_events().len(), events),
                    }
                }
                _ => {
                    // A host write aimed at ingress is refused and writes
                    // nothing.
                    let revision = runtime.revision();
                    let events = runtime.committed_events().len();
                    assert!(matches!(
                        runtime.apply_verified_semantic_delta(
                            revision,
                            delta(&request_raw_key(&request), "forged"),
                            &"test-operator".into(),
                        ),
                        Err(RuntimeError::ReservedSemanticNamespace { .. })
                    ));
                    assert_eq!(runtime.committed_events().len(), events);
                    ingress_refusals += 1;
                }
            }
        }
        let written = runtime.committed_events().to_vec();
        // Every semantic record the runtime wrote carries an origin.
        for committed in &written {
            if let LedgerEvent::SemanticDeltaCommitted {
                origin,
                encoded_delta,
                ..
            } = &committed.event
            {
                match origin {
                    SemanticOrigin::Legacy => panic!("seed {seed}: a record without an origin"),
                    SemanticOrigin::Request { .. } => requests += 1,
                    SemanticOrigin::PodOutput { .. } => outputs += 1,
                    SemanticOrigin::PodCandidate { .. } => outputs += 1,
                    SemanticOrigin::Host { verification, .. } => {
                        if verification.findings.is_empty() {
                            plain += 1;
                        } else {
                            noted += 1;
                        }
                        if !SemanticDelta::decode(encoded_delta)
                            .unwrap()
                            .dependencies
                            .is_empty()
                        {
                            derived += 1;
                        }
                    }
                    SemanticOrigin::Merge(merge) => {
                        if merge.rebased.is_empty() {
                            clean += 1;
                        } else {
                            rebased += 1;
                        }
                        match merge.authority {
                            MergeAuthorityRecord::Triage { .. } => triaged += 1,
                            MergeAuthorityRecord::Reviewed { .. } => reviewed += 1,
                        }
                    }
                }
            }
        }
        // And every path that rebuilds from it accepts it, reaching the
        // writer's state: revision, materialized state, and every value with
        // its inputs.
        let expected = held(&runtime);
        for (path, outcome) in rebuilt_held(&written, written.len() / 2) {
            match outcome {
                Ok(rebuilt) => assert!(rebuilt == expected, "seed {seed}, {path}"),
                Err(error) => panic!("seed {seed}, {path}: {error:?}"),
            }
        }
    }
    assert!(ingress_refusals > 0);
    for (kind, count) in [
        ("request", requests),
        ("Pod output", outputs),
        ("host write without findings", plain),
        ("host write with findings", noted),
        ("derived key", derived),
        ("clean merge", clean),
        ("rebased merge", rebased),
        ("triaged merge", triaged),
        ("reviewed merge", reviewed),
    ] {
        assert!(count > 0, "no {kind} in any history");
    }
}
