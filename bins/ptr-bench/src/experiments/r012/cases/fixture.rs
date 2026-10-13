//! What the cases share: a durable runtime that can be reopened and compacted, the execution
//! session the effect boundary is driven through, and the Pod, SemDB and merge fixtures.

use super::super::stub;
use crate::experiments::layered_kv::lab::{fail, Res};
use ptr_branch::{AutoThreshold, PolicyRecord, TriagePolicy};
use ptr_config::PtrConfig;
use ptr_ledger::{KvTaintChange, LedgerEvent};
use ptr_pods::{
    ExecutionManifest, LineageBinding, PodManifest, PodOutput, PodOutputKind, TypedPayload,
};
use ptr_runtime::execution::{
    ActionExecutor, ActionScope, ExecutionGrant, ExecutionSession, RequiredVerification,
    VerifiedDispatch,
};
use ptr_runtime::{
    ExecutionScope, FencingToken, PlacementEpoch, PodOutputAdmissionRequest, PtrRuntime,
    ScopeState, SemanticChange, SemanticGrant,
};
use ptr_types::{
    ActionIr, CapabilityId, CapsuleId, Effect, Generation, PodId, PrincipalId, Probability,
    ProjectId, ProvenanceRef, RequestId, Revision, ScopeId, ScopeLeaseBinding, SessionId,
    Timestamp, TypeId, VerificationLevel,
};
use ptr_verifier::{NamedVerifier, VerificationReport, VerificationStatus, Verifier};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub const PROJECT: &str = "r012-project";

/// A temporary directory removed on drop, the home of one case's `FileLedger`.
pub struct Temp(PathBuf);

impl Temp {
    pub fn new() -> Res<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ptr-r012-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos()),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).map_err(fail("a temporary directory"))?;
        Ok(Self(dir))
    }

    pub fn log(&self) -> PathBuf {
        self.0.join("ledger")
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A runtime over a durable `FileLedger` in a temporary directory, with the permissions the
/// action boundary needs, that a case can restart (`reopen`) and compact (`compact_restore`).
pub struct Rt {
    pub runtime: PtrRuntime,
    dir: Temp,
    /// Taint records committed before the last compaction: a restored runtime's committed
    /// events start above the snapshot floor, so they are counted here.
    floor_records: usize,
    floor_clears: usize,
}

fn arm(runtime: &mut PtrRuntime) {
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from(stub::CAPABILITY));
    runtime.permissions_mut().allow_mutation = true;
}

fn count_taint(runtime: &PtrRuntime, clears: bool) -> usize {
    runtime
        .committed_events()
        .iter()
        .filter(|committed| match &committed.event {
            LedgerEvent::KvTaintCommitted { change, .. } => {
                matches!(change, KvTaintChange::Clear { .. }) == clears
            }
            _ => false,
        })
        .count()
}

impl Rt {
    /// A fresh durable runtime with the action target committed (generation 1) and the
    /// capability granted.
    pub fn fresh() -> Res<Self> {
        let dir = Temp::new()?;
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), dir.log())
            .map_err(fail("opening a durable ledger"))?;
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from(PROJECT),
                capsule: CapsuleId::from(stub::TARGET),
                generation: Generation(1),
            })
            .map_err(fail("committing the action target"))?;
        arm(&mut runtime);
        Ok(Self {
            runtime,
            dir,
            floor_records: 0,
            floor_clears: 0,
        })
    }

    /// Drop the runtime and open the same ledger again (a restart). Permissions belong to the
    /// process, so they are granted again.
    pub fn reopen(&mut self) -> Res<()> {
        let placeholder =
            PtrRuntime::new(PtrConfig::default()).map_err(fail("a placeholder runtime"))?;
        drop(std::mem::replace(&mut self.runtime, placeholder));
        self.runtime = PtrRuntime::open_durable(PtrConfig::default(), self.dir.log())
            .map_err(fail("reopening the durable ledger"))?;
        arm(&mut self.runtime);
        Ok(())
    }

    /// Export a compacted snapshot and continue on the runtime restored from it (an in-memory
    /// runtime: the case ends its durable phase here).
    pub fn compact_restore(&mut self) -> Res<()> {
        let records = self.taint_records();
        let clears = self.clear_records();
        let snapshot = self
            .runtime
            .export_compacted_snapshot()
            .map_err(fail("exporting a compacted snapshot"))?;
        let mut restored = PtrRuntime::restore_compacted(
            PtrConfig::default(),
            snapshot.bytes(),
            snapshot.anchor(),
            &[],
        )
        .map_err(fail("restoring a compacted snapshot"))?;
        arm(&mut restored);
        self.runtime = restored;
        self.floor_records = records;
        self.floor_clears = clears;
        Ok(())
    }

    /// `KvTaintCommitted` Record events committed so far (including before a compaction).
    pub fn taint_records(&self) -> usize {
        self.floor_records + count_taint(&self.runtime, false)
    }

    /// `KvTaintCommitted` Clear events committed so far.
    pub fn clear_records(&self) -> usize {
        self.floor_clears + count_taint(&self.runtime, true)
    }

    /// Whether any taint record is persisted in the ledger (the append has returned).
    pub fn persisted(&self) -> bool {
        self.taint_records() > 0
    }

    pub fn events(&self) -> usize {
        self.runtime.committed_events().len()
    }

    /// `EffectAttempted` events in the committed tail.
    pub fn effect_attempts(&self) -> usize {
        self.runtime
            .committed_events()
            .iter()
            .filter(|committed| matches!(committed.event, LedgerEvent::EffectAttempted { .. }))
            .count()
    }
}

/// `action` as the runtime would stamp it now: the taint identity ignores the revision, but
/// the hard boundary wants the current one.
pub fn stamped(runtime: &PtrRuntime, action: &ActionIr) -> ActionIr {
    ActionIr {
        revision: runtime.revision(),
        ..action.clone()
    }
}

// ---- the execution session -------------------------------------------------------------------

/// Counts what the executor ran.
#[derive(Clone, Default)]
pub struct Probe(Arc<AtomicUsize>);

impl Probe {
    pub fn executions(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

struct PassAction;

impl Verifier<ActionIr> for PassAction {
    fn verify(&self, _: &ActionIr) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).expect("a bounded score"),
            findings: Vec::new(),
        }
    }
}

struct CountingExecutor(Probe);

impl ActionExecutor for CountingExecutor {
    fn execute(&self, _: VerifiedDispatch<'_>) -> Result<Vec<u8>, String> {
        (self.0).0.fetch_add(1, Ordering::SeqCst);
        Ok(b"executed".to_vec())
    }
}

pub const TTL: Duration = Duration::from_secs(60);

/// A session whose single grant covers the stub's action target, operation and capability.
pub fn exec_session(runtime: &mut PtrRuntime, probe: &Probe) -> Res<ExecutionSession> {
    let scope = ActionScope {
        project: ProjectId::from(PROJECT),
        target: stub::TARGET.to_string(),
        operation: stub::OPERATION.to_string(),
        capability: CapabilityId::from(stub::CAPABILITY),
        input_type: TypeId::from(stub::INPUT_TYPE),
        effect: Effect::Mutation,
    };
    runtime
        .register_execution_session(
            "r012-operator",
            vec![ExecutionGrant::new(
                scope,
                RequiredVerification::Deterministic,
                PassAction,
                CountingExecutor(probe.clone()),
            )],
            TTL,
        )
        .map_err(fail("registering an execution session"))
}

// ---- Pod, SemDB and merge fixtures ------------------------------------------------------------

/// Passing verifier of Pod payloads that counts how often it was asked.
#[derive(Clone, Default)]
pub struct CountingPodVerifier(Arc<AtomicUsize>);

impl CountingPodVerifier {
    pub fn calls(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

fn pass_report() -> VerificationReport {
    VerificationReport {
        status: VerificationStatus::Pass,
        level: VerificationLevel::Deterministic,
        score: Probability::new(1.0).expect("a bounded score"),
        findings: Vec::new(),
    }
}

impl Verifier<TypedPayload> for CountingPodVerifier {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        self.0.fetch_add(1, Ordering::SeqCst);
        pass_report()
    }
}

/// Passes every semantic change of the grant (merge and host write).
#[derive(Clone)]
pub struct PassSemantic;

impl<'a> Verifier<SemanticChange<'a>> for PassSemantic {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        pass_report()
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for PassSemantic {
    fn name(&self) -> &'static str {
        "r012-pass-semantic"
    }
}

pub const REVIEWER: &str = "r012-reviewer";

/// The semantic grant of the cases: host writes allowed, a manual merge policy with a 0.8
/// auto-propose threshold and one reviewer.
pub fn install_grant(runtime: &mut PtrRuntime) -> Res<()> {
    let policy = PolicyRecord::manual(
        "r012-merge-policy",
        TriagePolicy::new(AutoThreshold::AtLeast(0.8), 0.0).map_err(fail("the triage policy"))?,
    )
    .map_err(fail("the policy record"))?;
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(PassSemantic)
                .allow_host_writes()
                .with_merge_policy(policy, 7)
                .with_reviewer(PrincipalId::from(REVIEWER)),
        )
        .map_err(fail("the semantic grant"))
}

pub const SESSION: &str = "r012-session";
pub const SCOPE: &str = "r012-scope";
pub const POD_PROJECT: &str = "r012-pod-project";

/// The Pod scope `admit_pod_output` checks.
pub fn create_pod_scope(runtime: &mut PtrRuntime) -> Res<()> {
    runtime
        .create_scope(
            ExecutionScope {
                id: ScopeId::from(SCOPE),
                parent: None,
                session: SessionId::from(SESSION),
                project: ProjectId::from(POD_PROJECT),
                created_at: Timestamp(1),
                deadline: None,
                state: ScopeState::Created,
                cancellation_requested: false,
            },
            ScopeLeaseBinding::default(),
        )
        .map(|_| ())
        .map_err(fail("the Pod scope"))
}

/// The binding of the manifest to a snapshot key (a tainted stored candidate, for the
/// dependency case) or to the fixed fact every Pod output cites.
pub fn manifest(extra_knowledge: Vec<LineageBinding>) -> Res<ExecutionManifest> {
    let mut knowledge = vec![LineageBinding {
        key: "knowledge:fact".into(),
        generation: Generation(1),
        digest: [3; 32],
    }];
    knowledge.extend(extra_knowledge);
    ExecutionManifest::build(
        Generation(1),
        knowledge,
        vec![LineageBinding {
            key: "artifact:pod".into(),
            generation: Generation(1),
            digest: [2; 32],
        }],
        vec!["raw:test".into()],
        None,
        Revision(1),
        [4; 32],
        PrincipalId::from("r012-principal"),
        Revision(1),
    )
    .map_err(fail("the execution manifest"))
}

pub fn pod_manifest() -> PodManifest {
    PodManifest {
        project: ProjectId::from(POD_PROJECT),
        id: PodId::from("r012-pod"),
        capabilities: vec![],
        accepts: vec![TypeId::from("text")],
        produces: vec![TypeId::from("text")],
        effects: vec![Effect::Pure],
        protocol_version: 1,
    }
}

pub fn text_payload(bytes: &[u8]) -> TypedPayload {
    TypedPayload {
        type_id: TypeId::from("text"),
        bytes: bytes.to_vec(),
    }
}

/// An admission request of a `kind` output with `payload` for `request_id`, from Pod `pod`,
/// whose caller says Exact (the runtime must find the taint itself).
pub fn pod_request(
    request_id: &str,
    pod: &str,
    kind: PodOutputKind,
    payload: &[u8],
    manifest: ExecutionManifest,
) -> PodOutputAdmissionRequest {
    let manifest_digest = manifest.manifest_digest;
    PodOutputAdmissionRequest {
        request_id: RequestId::from(request_id),
        manifest_digest,
        manifest,
        pod_id: PodId::from(pod),
        output: PodOutput {
            kind,
            payload: text_payload(payload),
            generation: Generation(1),
            manifest_digest,
            artifact_digest: [2; 32],
            provenance: vec![ProvenanceRef {
                source: ptr_types::EvidenceId::from("r012-evidence"),
                note: None,
            }],
            dependencies: vec![[3; 32]],
            revision: Revision(1),
            verified: matches!(
                kind,
                PodOutputKind::StateDelta | PodOutputKind::VerifiedResult
            ),
        },
        expected_type: TypeId::from("text"),
        session_id: SessionId::from(SESSION),
        scope_id: ScopeId::from(SCOPE),
        placement_epoch: PlacementEpoch(1),
        fencing_token: FencingToken(1),
        kv_taint: ptr_model_api::KvReuseTaint::Exact,
    }
}
