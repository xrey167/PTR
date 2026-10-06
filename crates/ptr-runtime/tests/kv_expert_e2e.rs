//! KV prefix catalog against the real expert admission path: an entry that depends on an
//! expert is accepted while the materialized registry shows it serving, and is invalidated
//! and refused once an admitted structural record changes it, including after a compacted
//! restore.

use std::sync::Arc;

use ptr_config::PtrConfig;
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_model_api::{ByteTokenizer, PromptSegment, SegmentedModelRequest, Tokenizer};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertState, ExpertTarget, ExpertTargetInit, InMemoryPagedKvBackend, KvBackendError,
    KvPageBinding, KvPageSize, KvTensorBackend, KvTensorDType, KvTensorSchema, ManifestDigest,
    PagedKvTensorBackend, ProposerRef, RationaleDigest, RetiredReason, StructuralOp, TensorRef,
};
use ptr_runtime::expert::ExpertAdmissionGrant;
use ptr_runtime::{
    CatalogLimits, KvCatalogError, KvChainId, KvDependency, KvInsert, KvLookup, KvPrefixCatalog,
    KvRequestPositions, KvSalt, PtrRuntime,
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

fn grant() -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(project(), proposer(), 7, gate().config_digest().unwrap())
}

fn level_of(pod: &str) -> ExpertLevel {
    ExpertLevel::Token {
        layer: 3,
        index: u16::from(pod.as_bytes()[0]),
    }
}

fn spec(pod: &str) -> ExpertPodSpec {
    let mut digest = [1u8; 32];
    digest[0] = pod.as_bytes()[0];
    ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new(digest).unwrap(),
        level_of(pod),
        base(),
        Generation(1),
    )
    .unwrap()
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
    rt.admit_expert_proposal(&proposal, &report, &grant())
        .unwrap();
}

/// The registry's current view of `pod` as an operand target.
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

fn birth(rt: &mut PtrRuntime, pod: &str) {
    admit(rt, StructuralOp::Birth { result: spec(pod) });
}

fn dep(pod: &str) -> KvDependency {
    KvDependency::expert(&project(), &PodId::from(pod), Generation(1), [0; 32])
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

fn request() -> SegmentedModelRequest {
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
    fn new() -> Self {
        Self {
            backend: InMemoryPagedKvBackend::new(KvPageSize::DEFAULT),
        }
    }

    /// Install a fresh catalog on `rt` that retires on this rig's backend.
    fn enable(&self, rt: &mut PtrRuntime) {
        let catalog = KvPrefixCatalog::new(
            self.backend.take_fork_authority().unwrap(),
            CatalogLimits::default(),
        )
        .unwrap()
        .with_retirer(Arc::new(self.backend.clone()));
        rt.enable_kv_prefix_catalog(catalog).unwrap();
    }

    /// Prefill and publish the request's 32 tokens, then offer them with `deps`.
    fn insert(
        &self,
        rt: &mut PtrRuntime,
        deps: Vec<KvDependency>,
    ) -> (ptr_pods::KvSharedPrefix, Result<(), KvCatalogError>) {
        let req = request();
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
    let req = request();
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

#[test]
fn an_entry_depending_on_a_born_expert_hits_and_a_retire_invalidates_and_retires_it() {
    let mut rt = runtime();
    let rig = Rig::new();
    rig.enable(&mut rt);
    birth(&mut rt, "x");
    assert_eq!(
        rt.materialized_state().values.get("expert:proj/x"),
        Some(&ptr_state::expert_entry(1, ExpertState::Serving, 1))
    );

    let (prefix, inserted) = rig.insert(&mut rt, vec![dep("x")]);
    inserted.unwrap();
    assert!(hits(&mut rt));

    let op = StructuralOp::Retire {
        target: target(&rt, "x"),
        reason: RetiredReason::GateFailed,
    };
    admit(&mut rt, op);

    assert!(!hits(&mut rt), "the committed retire invalidated the entry");
    let stats = rt.kv_prefix_catalog().unwrap().stats();
    assert_eq!((stats.entries, stats.nodes), (0, 0));
    assert_eq!(stats.retire_failures, 0);
    assert_eq!(
        rig.backend.retire_prefix(&prefix.key),
        Err(KvBackendError::PrefixNotFound),
        "the hook already retired the prefix on the backend"
    );
    // A fresh offer that still names the retired expert is refused.
    let (_, again) = rig.insert(&mut rt, vec![dep("x")]);
    assert_eq!(again, Err(KvCatalogError::DependencyNotLive));
}

#[test]
fn a_prune_invalidates_every_operand() {
    let mut rt = runtime();
    let rig = Rig::new();
    rig.enable(&mut rt);
    birth(&mut rt, "x");
    birth(&mut rt, "y");
    rig.insert(&mut rt, vec![dep("y")]).1.unwrap();
    assert!(hits(&mut rt));
    let op = StructuralOp::Prune {
        targets: vec![target(&rt, "x"), target(&rt, "y")],
    };
    admit(&mut rt, op);
    assert!(!hits(&mut rt));
    assert_eq!(rt.kv_prefix_catalog().unwrap().stats().entries, 0);
}

#[test]
fn freeze_writes_the_state_name_the_oracle_accepts_and_invalidates_older_entries() {
    let mut rt = runtime();
    let rig = Rig::new();
    rig.enable(&mut rt);
    birth(&mut rt, "x");
    rig.insert(&mut rt, vec![dep("x")]).1.unwrap();
    assert!(hits(&mut rt));

    let op = StructuralOp::Freeze {
        target: target(&rt, "x"),
    };
    admit(&mut rt, op);
    assert_eq!(
        rt.materialized_state().values.get("expert:proj/x"),
        Some(&ptr_state::expert_entry(1, ExpertState::Frozen, 2))
    );
    // Any structural record naming the expert invalidates entries computed before it.
    assert!(!hits(&mut rt));
    // A frozen expert still serves, so a fresh entry is accepted and hits.
    rig.insert(&mut rt, vec![dep("x")]).1.unwrap();
    assert!(hits(&mut rt));
}

#[test]
fn the_expert_check_survives_a_compacted_restore() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    birth(&mut rt, "y");
    let op = StructuralOp::Retire {
        target: target(&rt, "x"),
        reason: RetiredReason::GateFailed,
    };
    admit(&mut rt, op);

    let snapshot = rt.export_compacted_snapshot().unwrap();
    assert_eq!(&snapshot.bytes()[..8], b"PTRCS006");
    let mut restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();

    let rig = Rig::new();
    rig.enable(&mut restored);
    assert_eq!(restored.kv_prefix_catalog().unwrap().stats().entries, 0);
    let (_, retired) = rig.insert(&mut restored, vec![dep("x")]);
    assert_eq!(retired, Err(KvCatalogError::DependencyNotLive));
    let (_, serving) = rig.insert(&mut restored, vec![dep("y")]);
    assert_eq!(serving, Ok(()));
    assert!(hits(&mut restored));
}
