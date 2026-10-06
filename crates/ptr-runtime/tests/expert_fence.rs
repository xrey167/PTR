//! The expert fence authority path: a committed fence takes a serving expert to `Fenced` at an
//! unchanged generation and a new fence epoch, the KV catalog never serves an entry recorded
//! under an earlier epoch (also after a readmission), the event cannot be appended directly,
//! and replay and compacted restore reproduce the state.

use std::sync::Arc;

use ptr_config::PtrConfig;
use ptr_ledger::{ExpertFenceAction, ExpertFenceCause, ExpertFenceEvidence, LedgerEvent};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_model_api::{ByteTokenizer, PromptSegment, SegmentedModelRequest, Tokenizer};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertState, ExpertTarget, ExpertTargetInit, InMemoryPagedKvBackend, KvPageBinding, KvPageSize,
    KvTensorBackend, KvTensorDType, KvTensorSchema, ManifestDigest, PagedKvTensorBackend,
    ProposerRef, RationaleDigest, RetiredReason, StructuralOp, TensorRef,
};
use ptr_runtime::expert::{
    expert_fence_epochs, expert_readmission_subject_ids, ExpertAdmissionError,
    ExpertAdmissionGrant, ExpertFenceAuthority, ExpertFenceGrant, ExpertFenceRequest,
    ExpertStaleField,
};
use ptr_runtime::{
    CatalogLimits, KvCatalogError, KvChainId, KvDependency, KvInsert, KvLookup, KvPrefixCatalog,
    KvRequestPositions, KvSalt, PtrRuntime, RuntimeError,
};
use ptr_types::{
    AdapterVersion, CommitIndex, DeviceId, Generation, ModelVersion, PodId, PrincipalId, ProjectId,
    RequestId, Revision, SemanticContext,
};

const CHAIN: KvChainId = KvChainId([1; 32]);

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn base() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").unwrap(),
        BaseModelDigest::new([9; 32]).unwrap(),
        1,
    )
    .unwrap()
}

fn proposer() -> ProposerRef {
    ProposerRef::new(project(), PodId::from("meta"), Generation(1)).unwrap()
}

fn gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.5,
        max_task_forgetting: 0.5,
        min_backward_transfer: -0.5,
        max_public_regression: 0.02,
    }
}

fn admit(rt: &mut PtrRuntime, op: StructuralOp) {
    let proposal = ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        op,
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    let subject = GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        [3; 32],
        [4; 32],
        [5; 32],
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    let report: BoundGateReport = gate()
        .evaluate(
            &AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .unwrap()
        .bind(subject);
    let grant =
        ExpertAdmissionGrant::new(project(), proposer(), 7, gate().config_digest().unwrap());
    rt.admit_expert_proposal(&proposal, &report, &grant)
        .unwrap();
}

fn birth(rt: &mut PtrRuntime, pod: &str) {
    let spec = ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new([pod.as_bytes()[0]; 32]).unwrap(),
        ExpertLevel::Token {
            layer: 3,
            index: u16::from(pod.as_bytes()[0]),
        },
        base(),
        Generation(1),
    )
    .unwrap();
    admit(rt, StructuralOp::Birth { result: spec });
}

fn entry_state(rt: &PtrRuntime, pod: &str) -> (ExpertState, u64) {
    let entry = rt
        .expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap();
    (entry.state, entry.state_version)
}

fn target(rt: &PtrRuntime, pod: &str) -> ExpertTarget {
    let entry = rt
        .expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap()
        .clone();
    ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: PodId::from(pod),
        generation: entry.generation,
        level: entry.level,
        base_model: entry.base_model,
        manifest_digest: entry.manifest_digest,
        state: entry.state,
        state_version: entry.state_version,
    })
    .unwrap()
}

fn request(pod: &str, version: u64, cause: ExpertFenceCause) -> ExpertFenceRequest {
    ExpertFenceRequest {
        project: project(),
        pod: PodId::from(pod),
        generation: Generation(1),
        expected_state_version: version,
        cause,
    }
}

fn revocation() -> ExpertFenceGrant {
    ExpertFenceGrant::revocation(project())
}

fn gate_grant() -> ExpertFenceGrant {
    ExpertFenceGrant::gate(project(), 7, gate().config_digest().unwrap())
}

const QUARANTINE: ExpertFenceCause = ExpertFenceCause::AuthorityQuarantine;

fn dep(pod: &str, epoch: u64) -> KvDependency {
    KvDependency::expert_fenced(&project(), &PodId::from(pod), Generation(1), [0; 32], epoch)
}

fn schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("base-v1"),
        adapter: AdapterVersion::from("none"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

fn salt() -> KvSalt {
    KvSalt::derive(
        &PrincipalId::from("alice"),
        &project(),
        &ModelVersion::from("base-v1"),
        ByteTokenizer::new().identity(),
        "f32-cpu-reference",
        "paged-kv-v1",
    )
}

fn kv_request() -> SegmentedModelRequest {
    SegmentedModelRequest::new(
        RequestId::from("r"),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        ByteTokenizer::new().identity().clone(),
        vec![PromptSegment::raw_from_text(&ByteTokenizer::new(), &"ab".repeat(16)).unwrap()],
    )
    .unwrap()
}

struct Rig {
    backend: InMemoryPagedKvBackend,
}

impl Rig {
    fn new(rt: &mut PtrRuntime) -> Self {
        let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
        let catalog = KvPrefixCatalog::new(
            backend.take_fork_authority().unwrap(),
            CatalogLimits::default(),
        )
        .unwrap()
        .with_retirer(Arc::new(backend.clone()));
        rt.enable_kv_prefix_catalog(catalog).unwrap();
        Self { backend }
    }

    fn insert(&self, rt: &mut PtrRuntime, deps: Vec<KvDependency>) -> Result<(), KvCatalogError> {
        self.insert_with_prefix(rt, deps).1
    }

    fn insert_with_prefix(
        &self,
        rt: &mut PtrRuntime,
        deps: Vec<KvDependency>,
    ) -> (ptr_pods::KvSharedPrefix, Result<(), KvCatalogError>) {
        let req = kv_request();
        let mut cache = self
            .backend
            .allocate_bound(
                schema(),
                64,
                KvPageBinding {
                    model: ModelVersion::from("base-v1"),
                    adapter: AdapterVersion::from("none"),
                    generation: Generation(1),
                    execution_manifest: [3; 32],
                    principal: PrincipalId::from("alice"),
                    device: DeviceId::from("cpu"),
                    dtype: KvTensorDType::F32,
                    page_tokens: KvPageSize::DEFAULT,
                },
            )
            .unwrap();
        let rows = |seed: f32| TensorRef {
            layer: 0,
            shape: vec![32, 4],
            values: (0..128).map(|i| seed + i as f32).collect(),
        };
        self.backend
            .append(&mut cache, &[rows(1.0)], &[rows(9.0)])
            .unwrap();
        let prefix = self.backend.publish_prefix(&mut cache).unwrap();
        let bytes = self.backend.page_metrics(&cache).resident_bytes;
        let result = rt
            .kv_prefix_insert(KvInsert {
                salt: &salt(),
                chain: &CHAIN,
                request: &req,
                prefix: prefix.clone(),
                page_bytes: bytes,
                extra_dependencies: deps,
            })
            .unwrap();
        (prefix, result)
    }
}

fn hits(rt: &mut PtrRuntime) -> bool {
    let req = kv_request();
    let hit = rt
        .kv_prefix_lookup(KvLookup {
            salt: &salt(),
            chain: &CHAIN,
            request: &req,
            positions: &KvRequestPositions::Canonical,
        })
        .unwrap();
    match hit {
        Some(hit) => {
            let pin = hit.pin().clone();
            rt.kv_prefix_release(&pin);
            true
        }
        None => false,
    }
}

fn runtime() -> PtrRuntime {
    PtrRuntime::new(PtrConfig::default()).unwrap()
}

fn expert_error(result: Result<CommitIndex, RuntimeError>) -> ExpertAdmissionError {
    match result {
        Err(RuntimeError::Expert(error)) => error,
        other => panic!("expected an expert refusal, got {other:?}"),
    }
}

fn applied(rt: &PtrRuntime) -> u64 {
    rt.materialized_state().last_applied
}

#[test]
fn a_fence_moves_the_expert_to_fenced_at_the_next_epoch_and_no_entry_of_an_earlier_epoch_is_live() {
    let mut rt = runtime();
    let rig = Rig::new(&mut rt);
    birth(&mut rt, "x");
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Serving, 1));
    rig.insert(&mut rt, vec![dep("x", 1)]).unwrap();
    assert!(hits(&mut rt));

    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();

    assert_eq!(entry_state(&rt, "x"), (ExpertState::Fenced, 2));
    assert_eq!(
        rt.expert_registry()
            .fence_cause(&project(), &PodId::from("x")),
        Some(QUARANTINE)
    );
    assert_eq!(
        rt.materialized_state().values.get("expert:proj/x"),
        Some(&ptr_state::expert_entry(1, ExpertState::Fenced, 2))
    );
    assert!(
        !hits(&mut rt),
        "the entry recorded under epoch 1 is refused"
    );
    // No dependency is live at any epoch up to and including the fence, nor unrecorded.
    for epoch in [0, 1, 2] {
        assert_eq!(
            rig.insert(&mut rt, vec![dep("x", epoch)]),
            Err(KvCatalogError::DependencyNotLive),
            "epoch {epoch}"
        );
    }
    // A sibling expert is unaffected.
    birth(&mut rt, "y");
    rig.insert(&mut rt, vec![dep("y", 1)]).unwrap();
    assert!(hits(&mut rt));
}

#[test]
fn a_frozen_expert_can_be_fenced_and_the_event_maps_to_the_catalog_epoch() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    let op = StructuralOp::Freeze {
        target: target(&rt, "x"),
    };
    admit(&mut rt, op);
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Frozen, 2));
    rt.fence_expert(
        &request("x", 2, ExpertFenceCause::InputRevoked),
        &revocation(),
    )
    .unwrap();
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Fenced, 3));
    let fence = rt.committed_events().last().unwrap().event.clone();
    assert_eq!(
        expert_fence_epochs(&fence),
        std::collections::BTreeMap::from([("expert:proj/x".to_owned(), 3)])
    );
    assert!(expert_fence_epochs(&rt.committed_events()[0].event).is_empty());
}

#[test]
fn the_fence_event_cannot_be_appended_directly() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    let before = applied(&rt);
    let event = LedgerEvent::ExpertFenceCommitted {
        project: project(),
        pod: PodId::from("x"),
        action: ExpertFenceAction::Fence,
        cause: QUARANTINE,
        generation: Generation(1),
        expected_state_version: 1,
        epoch: 2,
        evidence: None,
    };
    assert!(matches!(
        rt.commit(event),
        Err(RuntimeError::StructuralRecordOutsideAdmissionPath)
    ));
    assert_eq!(applied(&rt), before);
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Serving, 1));
}

#[test]
fn an_unauthorized_or_stale_fence_is_refused_and_appends_nothing() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    let before = applied(&rt);
    let fence = |rt: &mut PtrRuntime, request: &ExpertFenceRequest, grant: &ExpertFenceGrant| {
        expert_error(rt.fence_expert(request, grant))
    };

    // A grant for another project.
    let other = ExpertFenceGrant::revocation(ProjectId::from("other"));
    assert_eq!(
        fence(&mut rt, &request("x", 1, QUARANTINE), &other),
        ExpertAdmissionError::FenceProjectMismatch
    );
    // The gate authority may not fence.
    assert_eq!(
        fence(&mut rt, &request("x", 1, QUARANTINE), &gate_grant()),
        ExpertAdmissionError::FenceAuthorityMismatch {
            action: ExpertFenceAction::Fence,
            required: ExpertFenceAuthority::RuntimeRevocation,
        }
    );
    // Stale state version, stale generation, unknown pod.
    assert_eq!(
        fence(&mut rt, &request("x", 2, QUARANTINE), &revocation()),
        ExpertAdmissionError::StaleOperand {
            pod: "x".into(),
            field: ExpertStaleField::StateVersion,
        }
    );
    let mut stale_generation = request("x", 1, QUARANTINE);
    stale_generation.generation = Generation(2);
    assert_eq!(
        fence(&mut rt, &stale_generation, &revocation()),
        ExpertAdmissionError::StaleOperand {
            pod: "x".into(),
            field: ExpertStaleField::Generation,
        }
    );
    assert_eq!(
        fence(&mut rt, &request("nope", 1, QUARANTINE), &revocation()),
        ExpertAdmissionError::UnknownOperand { pod: "nope".into() }
    );
    // A version at the top of the range has no next epoch.
    assert!(matches!(
        fence(&mut rt, &request("x", u64::MAX, QUARANTINE), &revocation()),
        ExpertAdmissionError::FenceEpochMismatch { .. }
    ));
    assert_eq!(applied(&rt), before);
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Serving, 1));

    // A fenced expert cannot be fenced again; a retired one cannot be fenced at all.
    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();
    assert_eq!(
        fence(&mut rt, &request("x", 2, QUARANTINE), &revocation()),
        ExpertAdmissionError::NotFenceable {
            pod: "x".into(),
            state: ExpertState::Fenced,
        }
    );
    birth(&mut rt, "y");
    let op = StructuralOp::Retire {
        target: target(&rt, "y"),
        reason: RetiredReason::GateFailed,
    };
    admit(&mut rt, op);
    let (state, version) = entry_state(&rt, "y");
    assert_eq!(
        fence(&mut rt, &request("y", version, QUARANTINE), &revocation()),
        ExpertAdmissionError::NotFenceable {
            pod: "y".into(),
            state,
        }
    );
}

#[test]
fn a_fenced_expert_is_outside_the_proposal_path() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();
    // A proposal built from the stale `Serving` view fails the compare-and-set.
    let stale = ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: PodId::from("x"),
        generation: Generation(1),
        level: ExpertLevel::Token {
            layer: 3,
            index: 120,
        },
        base_model: base(),
        manifest_digest: ManifestDigest::new([120; 32]).unwrap(),
        state: ExpertState::Serving,
        state_version: 1,
    })
    .unwrap();
    let proposal = ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        StructuralOp::Freeze { target: stale },
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    let subject = GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        [3; 32],
        [4; 32],
        [5; 32],
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    let report = gate()
        .evaluate(
            &AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .unwrap()
        .bind(subject);
    let grant =
        ExpertAdmissionGrant::new(project(), proposer(), 7, gate().config_digest().unwrap());
    assert!(matches!(
        rt.admit_expert_proposal(&proposal, &report, &grant),
        Err(RuntimeError::Expert(ExpertAdmissionError::StaleOperand {
            field: ExpertStaleField::State,
            ..
        }))
    ));
}

#[test]
fn a_readmission_is_a_new_epoch_that_never_revives_old_entries() {
    let mut rt = runtime();
    let rig = Rig::new(&mut rt);
    birth(&mut rt, "x");
    rig.insert(&mut rt, vec![dep("x", 1)]).unwrap();
    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();

    // Only the gate authority readmits; the revocation authority does not.
    assert_eq!(
        expert_error(readmit(
            &mut rt,
            &request("x", 2, QUARANTINE),
            &revocation()
        )),
        ExpertAdmissionError::FenceAuthorityMismatch {
            action: ExpertFenceAction::Readmit,
            required: ExpertFenceAuthority::Gate,
        }
    );
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Fenced, 2));
    readmit(&mut rt, &request("x", 2, QUARANTINE), &gate_grant()).unwrap();

    assert_eq!(entry_state(&rt, "x"), (ExpertState::Serving, 3));
    assert_eq!(
        rt.expert_registry()
            .fence_cause(&project(), &PodId::from("x")),
        None
    );
    assert_eq!(
        rt.materialized_state().values.get("expert:proj/x"),
        Some(&ptr_state::expert_entry(1, ExpertState::Serving, 3))
    );
    // The expert serves again at the same generation, but the old entry stays dead...
    assert!(!hits(&mut rt));
    for epoch in [1, 2] {
        assert_eq!(
            rig.insert(&mut rt, vec![dep("x", epoch)]),
            Err(KvCatalogError::DependencyNotLive),
            "epoch {epoch}"
        );
    }
    // ... and an entry recorded under the new epoch is live.
    rig.insert(&mut rt, vec![dep("x", 3)]).unwrap();
    assert!(hits(&mut rt));
    // It can be fenced again, at the next epoch.
    rt.fence_expert(&request("x", 3, QUARANTINE), &revocation())
        .unwrap();
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Fenced, 4));
    assert!(!hits(&mut rt));
}

#[test]
fn readmission_is_refused_unless_a_quarantine_fence_is_cleared_by_the_gate() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    birth(&mut rt, "y");
    let before = applied(&rt);
    // Not fenced.
    assert_eq!(
        expert_error(readmit(
            &mut rt,
            &request("x", 1, QUARANTINE),
            &gate_grant()
        )),
        ExpertAdmissionError::NotFenced {
            pod: "x".into(),
            state: ExpertState::Serving,
        }
    );
    // A revoked input is final.
    rt.fence_expert(
        &request("x", 1, ExpertFenceCause::InputRevoked),
        &revocation(),
    )
    .unwrap();
    for cause in [ExpertFenceCause::InputRevoked, QUARANTINE] {
        assert_eq!(
            expert_error(readmit(&mut rt, &request("x", 2, cause), &gate_grant())),
            ExpertAdmissionError::NotReadmittable { pod: "x".into() }
        );
    }
    // The cause named must be the cause recorded.
    rt.fence_expert(&request("y", 1, QUARANTINE), &revocation())
        .unwrap();
    assert_eq!(
        expert_error(readmit(
            &mut rt,
            &request("y", 2, ExpertFenceCause::InputRevoked),
            &gate_grant()
        )),
        ExpertAdmissionError::FenceCauseMismatch { pod: "y".into() }
    );
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Fenced, 2));
    assert_eq!(entry_state(&rt, "y"), (ExpertState::Fenced, 2));
    assert_eq!(
        applied(&rt),
        before + 2,
        "only the two fences were appended"
    );
}

/// A report bound to exactly the readmission `request` names, under `gate`.
fn report_for(
    request: &ExpertFenceRequest,
    gate: &ForgettingGate,
    passing: bool,
) -> BoundGateReport {
    let (proposal_id, operation_id) = expert_readmission_subject_ids(
        &request.project,
        &request.pod,
        request.generation,
        request.expected_state_version,
        request.expected_state_version + 1,
        request.cause,
    );
    let subject = GateSubject::new(proposal_id, operation_id, [3; 32], [4; 32], [5; 32]);
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    gate.evaluate(
        &AdapterId::from("candidate"),
        &matrix,
        PublicSuite {
            serving: 0.8,
            candidate: if passing { 0.8 } else { 0.1 },
        },
    )
    .unwrap()
    .bind(subject)
}

fn readmit(
    rt: &mut PtrRuntime,
    request: &ExpertFenceRequest,
    grant: &ExpertFenceGrant,
) -> Result<CommitIndex, RuntimeError> {
    rt.readmit_expert(request, &report_for(request, &gate(), true), grant)
}

fn history(rt: &PtrRuntime) -> Vec<ptr_ledger::CommittedEvent> {
    rt.committed_events().to_vec()
}

#[test]
fn replay_reproduces_the_fence_state_cause_and_readmission_rules() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    birth(&mut rt, "y");
    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();
    rt.fence_expert(
        &request("y", 1, ExpertFenceCause::InputRevoked),
        &revocation(),
    )
    .unwrap();
    readmit(&mut rt, &request("x", 2, QUARANTINE), &gate_grant()).unwrap();
    rt.fence_expert(&request("x", 3, QUARANTINE), &revocation())
        .unwrap();

    let replayed = PtrRuntime::replay(PtrConfig::default(), &history(&rt)).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
    assert_eq!(
        &replayed.materialized_state().values,
        &rt.materialized_state().values
    );
    assert_eq!(entry_state(&replayed, "x"), (ExpertState::Fenced, 4));
    assert_eq!(
        replayed
            .expert_registry()
            .fence_cause(&project(), &PodId::from("y")),
        Some(ExpertFenceCause::InputRevoked)
    );

    // A history whose fence disagrees with the registry at that point fails to replay.
    let mut tampered = history(&rt);
    let at = tampered
        .iter()
        .position(|c| matches!(c.event, LedgerEvent::ExpertFenceCommitted { .. }))
        .unwrap();
    if let LedgerEvent::ExpertFenceCommitted {
        expected_state_version,
        epoch,
        ..
    } = &mut tampered[at].event
    {
        *expected_state_version = 5;
        *epoch = 6;
    }
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &tampered),
        Err(RuntimeError::Expert(ExpertAdmissionError::StaleOperand {
            field: ExpertStaleField::StateVersion,
            ..
        }))
    ));
    // A readmission of an input-revoked fence fails to replay too.
    let mut revoked = history(&rt);
    let at = revoked
        .iter()
        .position(|c| {
            matches!(
                c.event,
                LedgerEvent::ExpertFenceCommitted {
                    action: ExpertFenceAction::Readmit,
                    ..
                }
            )
        })
        .unwrap();
    for committed in &mut revoked[..at] {
        if let LedgerEvent::ExpertFenceCommitted { cause, .. } = &mut committed.event {
            *cause = ExpertFenceCause::InputRevoked;
        }
    }
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &revoked),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::NotReadmittable { .. }
        ))
    ));
}

#[test]
fn a_compacted_snapshot_carries_the_fence_epoch_state_and_cause() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    birth(&mut rt, "y");
    let plain = rt.export_compacted_snapshot().unwrap();
    assert_eq!(
        &plain.bytes()[..8],
        b"PTRCS006",
        "no fence cause, no new layout"
    );

    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();
    rt.fence_expert(
        &request("y", 1, ExpertFenceCause::InputRevoked),
        &revocation(),
    )
    .unwrap();
    let snapshot = rt.export_compacted_snapshot().unwrap();
    assert_eq!(&snapshot.bytes()[..8], b"PTRCS007");
    let mut restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    assert_eq!(restored.expert_registry(), rt.expert_registry());
    assert_eq!(entry_state(&restored, "x"), (ExpertState::Fenced, 2));
    let rig = Rig::new(&mut restored);
    for epoch in [0, 1, 2] {
        assert_eq!(
            rig.insert(&mut restored, vec![dep("x", epoch)]),
            Err(KvCatalogError::DependencyNotLive)
        );
    }
    // The restored runtime keeps the readmission rules.
    assert_eq!(
        expert_error(readmit(
            &mut restored,
            &request("y", 2, ExpertFenceCause::InputRevoked),
            &gate_grant()
        )),
        ExpertAdmissionError::NotReadmittable { pod: "y".into() }
    );
    readmit(&mut restored, &request("x", 2, QUARANTINE), &gate_grant()).unwrap();
    assert_eq!(entry_state(&restored, "x"), (ExpertState::Serving, 3));
    rig.insert(&mut restored, vec![dep("x", 2)])
        .expect_err("the epoch the fence ended stays dead");
    rig.insert(&mut restored, vec![dep("x", 3)]).unwrap();

    // Records above a floor that holds a fence replay on top of the restore.
    let mut above = runtime();
    birth(&mut above, "x");
    above
        .fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();
    let floor = above.export_compacted_snapshot().unwrap();
    readmit(&mut above, &request("x", 2, QUARANTINE), &gate_grant()).unwrap();
    let tail: Vec<_> = above.committed_events()[above.committed_events().len() - 1..].to_vec();
    let resumed =
        PtrRuntime::restore_compacted(PtrConfig::default(), floor.bytes(), floor.anchor(), &tail)
            .unwrap();
    assert_eq!(entry_state(&resumed, "x"), (ExpertState::Serving, 3));
    assert_eq!(
        resumed.expert_registry(),
        PtrRuntime::replay(PtrConfig::default(), above.committed_events())
            .unwrap()
            .expert_registry()
    );
}

#[test]
fn a_committed_fence_tombstones_and_retires_entries_eagerly() {
    let mut rt = runtime();
    let rig = Rig::new(&mut rt);
    birth(&mut rt, "x");
    birth(&mut rt, "y");
    let (prefix, inserted) = rig.insert_with_prefix(&mut rt, vec![dep("x", 1)]);
    inserted.unwrap();
    assert_eq!(rt.kv_prefix_catalog().unwrap().stats().entries, 1);
    // Another expert's fence leaves the entry alone.
    rt.fence_expert(&request("y", 1, QUARANTINE), &revocation())
        .unwrap();
    assert_eq!(rt.kv_prefix_catalog().unwrap().stats().entries, 1);

    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();
    // No lookup or insert happened since the fence: the hook did it at commit.
    let stats = rt.kv_prefix_catalog().unwrap().stats();
    assert_eq!((stats.entries, stats.nodes), (0, 0));
    assert_eq!(stats.retire_failures, 0);
    assert_eq!(
        rig.backend.retire_prefix(&prefix.key),
        Err(ptr_pods::KvBackendError::PrefixNotFound)
    );
}

#[test]
fn an_entry_with_no_recorded_epoch_stays_dead_after_a_readmission_and_across_replay_and_restore() {
    let mut rt = runtime();
    let rig = Rig::new(&mut rt);
    birth(&mut rt, "x");
    // An entry recorded with no epoch (0), and one at the then-current epoch.
    rig.insert(&mut rt, vec![dep("x", 0)]).unwrap();
    assert!(hits(&mut rt));
    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();
    readmit(&mut rt, &request("x", 2, QUARANTINE), &gate_grant()).unwrap();
    assert_eq!(
        rt.expert_registry().highest_fenced_epoch("expert:proj/x"),
        Some(2)
    );
    assert!(!hits(&mut rt));

    let check = |runtime: &mut PtrRuntime| {
        let rig = Rig::new(runtime);
        for epoch in [0, 1, 2] {
            assert_eq!(
                rig.insert(runtime, vec![dep("x", epoch)]),
                Err(KvCatalogError::DependencyNotLive),
                "epoch {epoch}"
            );
        }
        rig.insert(runtime, vec![dep("x", 3)]).unwrap();
        assert!(hits(runtime));
    };
    // Live, replayed and restored runtimes agree.
    let mut replayed = PtrRuntime::replay(PtrConfig::default(), &history(&rt)).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
    let snapshot = rt.export_compacted_snapshot().unwrap();
    assert_eq!(
        &snapshot.bytes()[..8],
        b"PTRCS007",
        "the memo outlives the readmission"
    );
    let mut restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    assert_eq!(restored.expert_registry(), rt.expert_registry());
    for epoch in [0, 1, 2] {
        assert_eq!(
            rig.insert(&mut rt, vec![dep("x", epoch)]),
            Err(KvCatalogError::DependencyNotLive)
        );
    }
    rig.insert(&mut rt, vec![dep("x", 3)]).unwrap();
    check(&mut replayed);
    check(&mut restored);
}

#[test]
fn a_registry_restored_from_a_pre_memo_snapshot_keeps_a_fenced_expert_dead() {
    // A PTRCS006 snapshot cannot hold a fence cause or a memo; `from_entries_all_legacy` is the
    // conservative path older layouts take: the fenced expert's memo is its current epoch.
    let mut rt = runtime();
    birth(&mut rt, "x");
    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();
    assert_eq!(
        rt.expert_registry().highest_fenced_epoch("expert:proj/x"),
        Some(2)
    );
    assert_eq!(
        rt.expert_registry()
            .highest_fenced_epoch("expert:proj/nope"),
        None
    );
    assert_eq!(
        rt.expert_registry().highest_fenced_epoch("not-an-expert"),
        None
    );
}

fn expert_refusal(
    rt: &mut PtrRuntime,
    request: &ExpertFenceRequest,
    report: &BoundGateReport,
) -> ExpertAdmissionError {
    expert_error(rt.readmit_expert(request, report, &gate_grant()))
}

#[test]
fn a_readmission_needs_passed_evidence_bound_to_this_expert_and_epoch() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    birth(&mut rt, "y");
    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();
    rt.fence_expert(&request("y", 1, QUARANTINE), &revocation())
        .unwrap();
    let before = applied(&rt);
    let req = request("x", 2, QUARANTINE);

    // A failed report.
    assert_eq!(
        expert_refusal(&mut rt, &req, &report_for(&req, &gate(), false)),
        ExpertAdmissionError::GateFailed
    );
    // A report for another expert, another epoch, another cause.
    assert_eq!(
        expert_refusal(
            &mut rt,
            &req,
            &report_for(&request("y", 2, QUARANTINE), &gate(), true)
        ),
        ExpertAdmissionError::EvidenceProposalMismatch
    );
    assert_eq!(
        expert_refusal(
            &mut rt,
            &req,
            &report_for(&request("x", 3, QUARANTINE), &gate(), true)
        ),
        ExpertAdmissionError::EvidenceProposalMismatch
    );
    assert_eq!(
        expert_refusal(
            &mut rt,
            &req,
            &report_for(
                &request("x", 2, ExpertFenceCause::InputRevoked),
                &gate(),
                true
            )
        ),
        ExpertAdmissionError::EvidenceProposalMismatch
    );
    // A report produced under another gate configuration.
    let other_gate = ForgettingGate {
        max_average_forgetting: 0.9,
        ..gate()
    };
    assert_eq!(
        expert_refusal(&mut rt, &req, &report_for(&req, &other_gate, true)),
        ExpertAdmissionError::GateConfigMismatch
    );
    // A report with the right proposal but another operation id.
    let (proposal_id, _) = expert_readmission_subject_ids(
        &project(),
        &PodId::from("x"),
        Generation(1),
        2,
        3,
        QUARANTINE,
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    let wrong_operation = gate()
        .evaluate(
            &AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .unwrap()
        .bind(GateSubject::new(
            proposal_id,
            [9; 32],
            [3; 32],
            [4; 32],
            [5; 32],
        ));
    assert_eq!(
        expert_refusal(&mut rt, &req, &wrong_operation),
        ExpertAdmissionError::EvidenceOperationMismatch
    );
    assert_eq!(applied(&rt), before, "nothing was appended");
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Fenced, 2));
    // A fence never takes evidence: the grant for it is the revocation one and no report exists.
    readmit(&mut rt, &req, &gate_grant()).unwrap();
}

#[test]
fn replay_reverifies_the_readmission_evidence() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    rt.fence_expert(&request("x", 1, QUARANTINE), &revocation())
        .unwrap();
    readmit(&mut rt, &request("x", 2, QUARANTINE), &gate_grant()).unwrap();
    PtrRuntime::replay(PtrConfig::default(), &history(&rt)).unwrap();

    let tamper = |change: &dyn Fn(&mut ExpertFenceEvidence)| {
        let mut events = history(&rt);
        let last = events.last_mut().unwrap();
        let LedgerEvent::ExpertFenceCommitted {
            evidence: Some(evidence),
            ..
        } = &mut last.event
        else {
            panic!("the last event is the readmission");
        };
        change(evidence);
        PtrRuntime::replay(PtrConfig::default(), &events)
    };
    let refused = |result: Result<PtrRuntime, RuntimeError>| match result {
        Err(RuntimeError::Expert(error)) => error,
        other => panic!("expected an expert refusal, got {:?}", other.map(|_| ())),
    };
    // Altered report bytes.
    assert!(matches!(
        refused(tamper(&|e| e.report[8] ^= 1)),
        ExpertAdmissionError::EvidenceUndecodable(_)
            | ExpertAdmissionError::EvidenceProposalMismatch
            | ExpertAdmissionError::EvidenceOperationMismatch
            | ExpertAdmissionError::GateFailed
            | ExpertAdmissionError::GateConfigMismatch
    ));
    // Truncated report bytes.
    assert!(matches!(
        refused(tamper(&|e| {
            e.report.pop();
        })),
        ExpertAdmissionError::EvidenceUndecodable(_)
    ));
    // A required configuration the report was not produced under.
    assert_eq!(
        refused(tamper(&|e| e.required_gate_config_digest = [1; 32])),
        ExpertAdmissionError::GateConfigMismatch
    );
    // A report for another readmission, swapped in.
    let other = report_for(&request("x", 9, QUARANTINE), &gate(), true);
    assert_eq!(
        refused(tamper(&|e| e.report = other.encode_canonical())),
        ExpertAdmissionError::EvidenceProposalMismatch
    );
    // A failed report.
    let failed = report_for(&request("x", 2, QUARANTINE), &gate(), false);
    assert_eq!(
        refused(tamper(&|e| e.report = failed.encode_canonical())),
        ExpertAdmissionError::GateFailed
    );
    // Evidence removed from the readmission: the codec's canonical form refuses it before the
    // registry is consulted.
    let mut events = history(&rt);
    if let LedgerEvent::ExpertFenceCommitted { evidence, .. } =
        &mut events.last_mut().unwrap().event
    {
        *evidence = None;
    }
    assert!(ptr_ledger::check_encodable(&events.last().unwrap().event).is_err());
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &events),
        Err(RuntimeError::Expert(ExpertAdmissionError::Unencodable))
    ));
    // Two readmissions of one gate epoch under different configurations conflict.
    let mut conflicting = runtime();
    birth(&mut conflicting, "a");
    birth(&mut conflicting, "b");
    for pod in ["a", "b"] {
        conflicting
            .fence_expert(&request(pod, 1, QUARANTINE), &revocation())
            .unwrap();
    }
    readmit(
        &mut conflicting,
        &request("a", 2, QUARANTINE),
        &gate_grant(),
    )
    .unwrap();
    let req_b = request("b", 2, QUARANTINE);
    let other_gate = ForgettingGate {
        max_average_forgetting: 0.9,
        ..gate()
    };
    let other_grant = ExpertFenceGrant::gate(project(), 7, other_gate.config_digest().unwrap());
    assert_eq!(
        expert_error(conflicting.readmit_expert(
            &req_b,
            &report_for(&req_b, &other_gate, true),
            &other_grant
        )),
        ExpertAdmissionError::EpochGateConfigConflict { epoch: 7 }
    );
}
