//! What the cases share: a `PtrRuntime` over a durable `FileLedger` in a temporary directory
//! that a case can restart (`reopen`) and compact (`compact_restore`), the host authority a
//! grant rests on, and the builders of proposals, reports, grants and requests the public
//! expert API takes.
//!
//! Everything here goes through the public API of `ptr_runtime`; the harness holds no runtime
//! state of its own beyond what it needs to build an operand (the registry entry it reads).

use crate::experiments::layered_kv::lab::{fail, Res};
use ed25519_dalek::SigningKey;
use ptr_config::PtrConfig;
use ptr_ledger::{
    CommittedEvent, ExpertAuthorityClass, ExpertAuthorityKind, ExpertFenceCause,
    ExpertFenceRevocation, LedgerEvent,
};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertInputRef, ExpertInputs, ExpertLevel,
    ExpertPodSpec, ExpertProposal, ExpertProposalError, ExpertState, ExpertTarget,
    ExpertTargetInit, ManifestDigest, ProposerRef, RationaleDigest, RetiredReason, StructuralOp,
};
use ptr_runtime::execution::AdmissionPolicy;
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_artifact_digest_with_inputs,
    expert_evidence_base_model_digest, expert_readmission_subject_ids, ExpertAdmissionGrant,
    ExpertAuthorityChange, ExpertAuthorityOutcome, ExpertFenceGrant, ExpertFenceRequest,
    ExpertGateRequirement, ExpertGrantBinding, ExpertGrantKind, InputInvalidation,
    InputInvalidationOutcome, RouteRefusal,
};
use ptr_runtime::PtrRuntime;
use ptr_runtime::RuntimeError;
use ptr_types::{
    AuthenticationLevel, CommitIndex, Generation, IdentityContext, NodeId, PodId, ProjectId,
    SessionId, Timestamp,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// The project of every case but the fan-out (which uses a longer one, see `Rt::fresh_in`).
pub const PROJECT: &str = "r011-project";
pub const KEY_ID: &str = "r011-policy-key";
pub const PEER: &str = "r011-host-peer";
pub const SESSION: &str = "r011-session";
pub const SUBJECT: &str = "r011-host";
/// The host clock reading grants are minted and committed at.
pub const NOW: Timestamp = Timestamp(1_000);
pub const ISSUED_AT: u64 = 100;
pub const EXPIRES_AT: u64 = 10_000;
/// The evaluation manifest every host in this harness expects.
pub const EVAL_MANIFEST: [u8; 32] = [5; 32];
/// Another evaluation's manifest.
pub const OTHER_EVAL_MANIFEST: [u8; 32] = [6; 32];
/// The digest of the base model every expert is on.
pub const BASE_DIGEST: [u8; 32] = [9; 32];
/// The gate epoch of the pre-regime host (before an authority change gives a kind a standing).
pub const GATE_EPOCH: u64 = 7;
/// The layer of every token-level expert the core cases birth.
pub const LAYER: u16 = 3;

/// A temporary directory removed on drop, the home of one case's `FileLedger`.
pub struct Temp(PathBuf);

impl Temp {
    pub fn new() -> Res<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ptr-r011-{}-{}-{}",
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

    /// A file next to the ledger (the replacement blob of a supersession).
    pub fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---- gate, reports and evidence ----------------------------------------------------------------

/// The gate configuration of version `version`: each version has its own configuration digest.
pub fn gate_v(version: u8) -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.5 - 0.02 * f64::from(version),
        max_task_forgetting: 0.5,
        min_backward_transfer: -0.5,
        max_public_regression: 0.02,
    }
}

pub fn gate() -> ForgettingGate {
    gate_v(0)
}

/// A looser gate than [`gate`]: it admits more forgetting.
pub fn weaker_gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.9,
        max_task_forgetting: 0.9,
        ..gate()
    }
}

pub fn config_digest(gate: &ForgettingGate) -> Res<[u8; 32]> {
    gate.config_digest()
        .map_err(fail("the gate configuration digest"))
}

pub fn evaluator_id(version: u8) -> [u8; 32] {
    [0xE0 + version; 32]
}

fn matrix() -> AccuracyMatrix {
    AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).expect("a fixed accuracy matrix")
}

/// What the evidence of a report is made of: the gate it ran under, whether it passed, and the
/// subject fields a negative case replaces.
#[derive(Clone, Debug)]
pub struct ReportSpec {
    pub gate: ForgettingGate,
    pub passing: bool,
    pub artifact: Option<[u8; 32]>,
    pub base_model: Option<[u8; 32]>,
    pub eval_manifest: Option<[u8; 32]>,
    pub ids: Option<([u8; 32], [u8; 32])>,
}

impl ReportSpec {
    pub fn correct() -> Self {
        Self::under(gate())
    }

    pub fn under(gate: ForgettingGate) -> Self {
        Self {
            gate,
            passing: true,
            artifact: None,
            base_model: None,
            eval_manifest: None,
            ids: None,
        }
    }

    pub fn failing() -> Self {
        Self {
            passing: false,
            ..Self::correct()
        }
    }

    pub fn wrong_artifact() -> Self {
        Self {
            artifact: Some([0xA1; 32]),
            ..Self::correct()
        }
    }

    pub fn wrong_base_model() -> Self {
        Self {
            base_model: Some([0xB2; 32]),
            ..Self::correct()
        }
    }

    pub fn other_eval_manifest() -> Self {
        Self {
            eval_manifest: Some(OTHER_EVAL_MANIFEST),
            ..Self::correct()
        }
    }

    /// The gate configuration of another evaluation (a different version).
    pub fn other_gate() -> Self {
        Self::under(gate_v(5))
    }

    pub fn weaker_gate() -> Self {
        Self::under(weaker_gate())
    }

    pub fn evaluate(&self, subject: GateSubject) -> Res<BoundGateReport> {
        let public = PublicSuite {
            serving: 0.8,
            candidate: if self.passing { 0.8 } else { 0.1 },
        };
        let subject = GateSubject::new(
            self.ids.map_or(subject.proposal_id, |ids| ids.0),
            self.ids.map_or(subject.operation_id, |ids| ids.1),
            self.artifact.unwrap_or(subject.artifact_digest),
            self.base_model.unwrap_or(subject.base_model_digest),
            self.eval_manifest.unwrap_or(subject.eval_manifest_digest),
        );
        Ok(self
            .gate
            .evaluate(&AdapterId::from("candidate"), &matrix(), public)
            .map_err(fail("the gate evaluation"))?
            .bind(subject))
    }
}

/// The report for `proposal` built from `spec`. With a dependency block the artifact digest is
/// the one that attests the block.
pub fn report_for(
    proposal: &ExpertProposal,
    inputs: Option<&ExpertInputs>,
    spec: &ReportSpec,
) -> Res<BoundGateReport> {
    let artifact = match inputs {
        Some(inputs) => expert_evidence_artifact_digest_with_inputs(proposal.op(), inputs),
        None => expert_evidence_artifact_digest(proposal.op()),
    };
    let base_model = expert_evidence_base_model_digest(proposal.op())
        .ok_or_else(|| "the proposal names no expert".to_string())?;
    spec.evaluate(GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        artifact,
        base_model,
        EVAL_MANIFEST,
    ))
}

/// The report that justifies the readmission `request` names for an expert whose manifest
/// digest is `manifest`.
pub fn readmission_report(
    request: &ExpertFenceRequest,
    manifest: [u8; 32],
    spec: &ReportSpec,
) -> Res<BoundGateReport> {
    let (proposal_id, operation_id) = expert_readmission_subject_ids(
        &request.project,
        &request.pod,
        request.generation,
        request.expected_state_version,
        request.expected_state_version + 1,
        request.cause,
    );
    spec.evaluate(GateSubject::new(
        proposal_id,
        operation_id,
        manifest,
        BASE_DIGEST,
        EVAL_MANIFEST,
    ))
}

/// What a grant claims about the authority its evidence comes from: the gate epoch, the gate
/// configuration version and the evaluator version (`None`: names none).
#[derive(Clone, Copy, Debug)]
pub struct Setup {
    pub epoch: u64,
    pub version: u8,
    pub evaluator: Option<u8>,
}

impl Setup {
    /// The pre-regime host: no kind has a standing, so nothing is enforced.
    pub const PRE: Setup = Setup {
        epoch: GATE_EPOCH,
        version: 0,
        evaluator: None,
    };

    pub fn requirement(&self) -> Res<ExpertGateRequirement> {
        Ok(ExpertGateRequirement {
            gate_epoch: self.epoch,
            config_digest: config_digest(&gate_v(self.version))?,
            eval_manifest_digest: EVAL_MANIFEST,
            require_inputs: false,
        })
    }

    pub fn gate(&self) -> ForgettingGate {
        gate_v(self.version)
    }
}

// ---- operands ------------------------------------------------------------------------------

pub fn base() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").expect("a base model id"),
        BaseModelDigest::new(BASE_DIGEST).expect("a base model digest"),
        1,
    )
    .expect("a base model ref")
}

/// The identity of the base model as an input (`base-model:{id}`).
pub fn base_model_input() -> ExpertInputRef {
    ExpertInputRef::from_base_model(&base())
}

pub fn input(identity: &str, generation: u64, digest: [u8; 32]) -> ExpertInputRef {
    ExpertInputRef::new(identity, Generation(generation), digest).expect("a valid input reference")
}

pub fn inputs(
    training: &[(&str, u64, [u8; 32])],
    evaluation: &[(&str, u64, [u8; 32])],
) -> ExpertInputs {
    ExpertInputs::new(
        training.iter().map(|i| input(i.0, i.1, i.2)).collect(),
        evaluation.iter().map(|i| input(i.0, i.1, i.2)).collect(),
    )
    .expect("a valid dependency block")
}

/// The token-level slot of a pod: its first byte.
pub fn level_of(pod: &str) -> ExpertLevel {
    ExpertLevel::Token {
        layer: LAYER,
        index: u16::from(pod.as_bytes()[0]),
    }
}

/// A token-level expert of generation `generation` whose manifest digest follows pod and
/// generation.
pub fn spec_at(
    project: &ProjectId,
    pod: &str,
    generation: u64,
    level: ExpertLevel,
) -> ExpertPodSpec {
    let mut digest = [pod.as_bytes()[0]; 32];
    digest[1] = generation as u8;
    ExpertPodSpec::new(
        project.clone(),
        PodId::from(pod),
        ManifestDigest::new(digest).expect("a non-zero manifest digest"),
        level,
        base(),
        Generation(generation),
    )
    .expect("an expert pod spec")
}

pub fn proposer_of(project: &ProjectId) -> ProposerRef {
    ProposerRef::new(project.clone(), PodId::from("meta"), Generation(1)).expect("a proposer")
}

/// The target of `pod` exactly as the registry holds it.
pub fn init_of(rt: &PtrRuntime, project: &ProjectId, pod: &str) -> Res<ExpertTargetInit> {
    let entry = rt
        .expert_registry()
        .get(project, &PodId::from(pod))
        .ok_or_else(|| format!("expert {pod} is not in the registry"))?
        .clone();
    Ok(ExpertTargetInit {
        project: project.clone(),
        pod: PodId::from(pod),
        generation: entry.generation,
        level: entry.level,
        base_model: entry.base_model,
        manifest_digest: entry.manifest_digest,
        state: entry.state,
        state_version: entry.state_version,
    })
}

/// Which of the seven operations a scenario exercises (Reactivate in its two forms).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scenario {
    Birth,
    Merge,
    Split,
    Freeze,
    ReactivateFrozen,
    ReactivateRetired,
    Prune,
    Retire,
}

pub const SCENARIOS: [Scenario; 8] = [
    Scenario::Birth,
    Scenario::Merge,
    Scenario::Split,
    Scenario::Freeze,
    Scenario::ReactivateFrozen,
    Scenario::ReactivateRetired,
    Scenario::Prune,
    Scenario::Retire,
];

impl Scenario {
    /// The event type the draft names for this operation.
    pub fn event_type(self) -> &'static str {
        match self {
            Self::Birth => "ExpertPromoted",
            Self::Merge => "Merge",
            Self::Split => "Split",
            Self::Freeze => "Freeze",
            Self::ReactivateFrozen | Self::ReactivateRetired => "Reactivate",
            Self::Prune => "Prune",
            Self::Retire => "ExpertRetired",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Birth => "birth",
            Self::Merge => "merge",
            Self::Split => "split",
            Self::Freeze => "freeze",
            Self::ReactivateFrozen => "reactivate_frozen",
            Self::ReactivateRetired => "reactivate_retired",
            Self::Prune => "prune",
            Self::Retire => "retire",
        }
    }

    /// The operand pods the scenario operates on.
    pub fn operands(self) -> &'static [&'static str] {
        match self {
            Self::Birth => &[],
            Self::Merge | Self::Prune => &["a", "b"],
            _ => &["a"],
        }
    }
}

/// An observed-operand tweak: the pod name and the init the registry holds, mapped to the
/// init the proposal claims.
pub type Tweak<'a> = &'a dyn Fn(&str, ExpertTargetInit) -> ExpertTargetInit;

pub fn exact(_: &str, init: ExpertTargetInit) -> ExpertTargetInit {
    init
}

// ---- the runtime -------------------------------------------------------------------------------

/// The identity of the host every grant rests on.
pub fn identity_for(session: &str, issued_at: u64, expires_at: u64) -> IdentityContext {
    let mut identity = IdentityContext {
        subject: SUBJECT.into(),
        issuer: "r011".into(),
        groups: vec!["operators".into()],
        authentication_level: AuthenticationLevel::MultiFactor,
        session_id: SessionId(session.into()),
        issued_at: Timestamp(issued_at),
        expires_at: Timestamp(expires_at),
        identity_digest: [0; 32],
    };
    identity.identity_digest = identity.canonical_digest();
    identity
}

pub fn identity() -> IdentityContext {
    identity_for(SESSION, ISSUED_AT, EXPIRES_AT)
}

pub fn peer() -> NodeId {
    NodeId::from(PEER)
}

/// The signing key the policy trust store holds under [`KEY_ID`].
fn policy_key() -> ed25519_dalek::VerifyingKey {
    SigningKey::from_bytes(&[7; 32]).verifying_key()
}

/// Install the policy key, admit the identity's principal and the transport peer: what the host
/// does before it mints a grant. A runtime that was restarted holds none of it.
pub fn prime(runtime: &mut PtrRuntime) -> Res<()> {
    runtime.install_policy_key(KEY_ID, policy_key());
    runtime
        .admit_identity_context(&identity(), NOW)
        .map_err(fail("admitting the host identity"))?;
    let mut policy = AdmissionPolicy::new();
    policy
        .admit(
            peer(),
            SUBJECT,
            std::time::Duration::from_secs(60),
            Vec::new,
        )
        .map_err(fail("admitting the host peer"))?;
    runtime.install_admission_policy(policy);
    Ok(())
}

/// A runtime over a durable `FileLedger` in a temporary directory.
pub struct Rt {
    pub runtime: PtrRuntime,
    dir: Temp,
    pub project: ProjectId,
    compacted: bool,
}

impl Rt {
    /// A fresh durable runtime, primed with the host authority.
    pub fn fresh() -> Res<Self> {
        Self::fresh_in(PROJECT)
    }

    /// Like [`Self::fresh`], for the experts of `project`.
    pub fn fresh_in(project: &str) -> Res<Self> {
        let dir = Temp::new()?;
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), dir.log())
            .map_err(fail("opening a durable ledger"))?;
        prime(&mut runtime)?;
        Ok(Self {
            runtime,
            dir,
            project: ProjectId::from(project),
            compacted: false,
        })
    }

    pub fn proposer(&self) -> ProposerRef {
        proposer_of(&self.project)
    }

    pub fn scratch(&self, name: &str) -> PathBuf {
        self.dir.file(name)
    }

    /// Drop the runtime and open the same ledger again (a crash after an append, or a restart).
    /// The host authority belongs to the process, so it is installed again.
    pub fn reopen(&mut self) -> Res<()> {
        if self.compacted {
            return Err("reopen after a compaction restore".to_string());
        }
        let placeholder =
            PtrRuntime::new(PtrConfig::default()).map_err(fail("a placeholder runtime"))?;
        drop(std::mem::replace(&mut self.runtime, placeholder));
        self.runtime = PtrRuntime::open_durable(PtrConfig::default(), self.dir.log())
            .map_err(fail("reopening the durable ledger"))?;
        prime(&mut self.runtime)
    }

    /// Export a compacted snapshot and continue on the runtime restored from it (an in-memory
    /// runtime: the case ends its durable phase here).
    pub fn compact_restore(&mut self) -> Res<()> {
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
        prime(&mut restored)?;
        self.runtime = restored;
        self.compacted = true;
        Ok(())
    }

    /// A runtime rebuilt from the whole committed history by `replay` (no durable ledger).
    pub fn replayed(&self) -> Res<PtrRuntime> {
        PtrRuntime::replay(PtrConfig::default(), self.runtime.committed_events())
            .map_err(fail("replaying the committed history"))
    }

    /// Replay the history `events` (a tampered or truncated copy).
    pub fn replay_of(events: &[CommittedEvent]) -> Result<PtrRuntime, RuntimeError> {
        PtrRuntime::replay(PtrConfig::default(), events)
    }

    pub fn events(&self) -> Vec<CommittedEvent> {
        self.runtime.committed_events().to_vec()
    }

    pub fn event_count(&self) -> usize {
        self.runtime.committed_events().len()
    }

    // ---- authority ----

    /// A binding at the generation the runtime stands at now for `kind`.
    pub fn binding(&self, kind: ExpertGrantKind) -> ExpertGrantBinding {
        ExpertGrantBinding::new(
            identity(),
            peer(),
            NOW,
            KEY_ID,
            self.runtime.expert_grant_generation(kind),
        )
    }

    pub fn admission_grant(&self, setup: Setup) -> Res<ExpertAdmissionGrant> {
        let grant = ExpertAdmissionGrant::new(
            self.project.clone(),
            self.proposer(),
            setup.requirement()?,
            self.binding(ExpertGrantKind::Admission),
        );
        Ok(match setup.evaluator {
            Some(version) => grant.with_evaluator(evaluator_id(version)),
            None => grant,
        })
    }

    pub fn fence_grant(&self) -> ExpertFenceGrant {
        ExpertFenceGrant::revocation(self.project.clone(), self.binding(ExpertGrantKind::Fence))
    }

    pub fn global_grant(&self) -> ExpertFenceGrant {
        ExpertFenceGrant::global_revocation(self.binding(ExpertGrantKind::GlobalRevocation))
    }

    pub fn gate_grant(&self, setup: Setup) -> Res<ExpertFenceGrant> {
        let grant = ExpertFenceGrant::gate(
            self.project.clone(),
            setup.requirement()?,
            self.binding(ExpertGrantKind::Gate),
        );
        Ok(match setup.evaluator {
            Some(version) => grant.with_evaluator(evaluator_id(version)),
            None => grant,
        })
    }

    // ---- operands ----

    pub fn spec(&self, pod: &str) -> ExpertPodSpec {
        spec_at(&self.project, pod, 1, level_of(pod))
    }

    /// A segment-level expert (no routing slot, so many may be born).
    pub fn segment_spec(&self, pod: &str, generation: u64) -> ExpertPodSpec {
        spec_at(&self.project, pod, generation, ExpertLevel::Segment)
    }

    pub fn init(&self, pod: &str) -> Res<ExpertTargetInit> {
        init_of(&self.runtime, &self.project, pod)
    }

    pub fn target_with(&self, pod: &str, tweak: Tweak<'_>) -> Res<ExpertTarget> {
        ExpertTarget::new(tweak(pod, self.init(pod)?)).map_err(fail("an expert target"))
    }

    pub fn target(&self, pod: &str) -> Res<ExpertTarget> {
        self.target_with(pod, &exact)
    }

    pub fn state(&self, pod: &str) -> Option<(ExpertState, u64)> {
        self.runtime
            .expert_registry()
            .get(&self.project, &PodId::from(pod))
            .map(|entry| (entry.state, entry.state_version))
    }

    pub fn manifest(&self, pod: &str) -> Res<[u8; 32]> {
        Ok(*self
            .runtime
            .expert_registry()
            .get(&self.project, &PodId::from(pod))
            .ok_or_else(|| format!("expert {pod} is not in the registry"))?
            .manifest_digest
            .as_bytes())
    }

    /// The proposal for `op`, or the typed refusal of the proposal constructor.
    pub fn try_proposal(&self, op: StructuralOp) -> Result<ExpertProposal, ExpertProposalError> {
        ExpertProposal::new(
            self.project.clone(),
            self.proposer(),
            CommitIndex(1),
            op,
            RationaleDigest::new([7; 32]).expect("a non-zero rationale digest"),
        )
    }

    pub fn proposal(&self, op: StructuralOp) -> Res<ExpertProposal> {
        self.try_proposal(op).map_err(fail("an expert proposal"))
    }

    // ---- admission ----

    /// Admit `op` with a correct report under the current pre-regime grant.
    pub fn admit(&mut self, op: StructuralOp) -> Result<CommitIndex, RuntimeError> {
        self.admit_as(op, None, &ReportSpec::correct(), Setup::PRE)
    }

    /// Admit `op` (with the dependency block `inputs`, when given) on a report built from
    /// `spec`, under a grant minted now for `setup`.
    pub fn admit_as(
        &mut self,
        op: StructuralOp,
        inputs: Option<ExpertInputs>,
        spec: &ReportSpec,
        setup: Setup,
    ) -> Result<CommitIndex, RuntimeError> {
        let proposal = self.proposal(op).map_err(RuntimeError::InvalidConfig)?;
        let report =
            report_for(&proposal, inputs.as_ref(), spec).map_err(RuntimeError::InvalidConfig)?;
        let grant = self
            .admission_grant(setup)
            .map_err(RuntimeError::InvalidConfig)?;
        self.commit_proposal(&proposal, &report, &grant, inputs)
    }

    /// The admission call itself, for a proposal, a report and a grant the caller built.
    pub fn commit_proposal(
        &mut self,
        proposal: &ExpertProposal,
        report: &BoundGateReport,
        grant: &ExpertAdmissionGrant,
        inputs: Option<ExpertInputs>,
    ) -> Result<CommitIndex, RuntimeError> {
        match inputs {
            Some(inputs) => self
                .runtime
                .admit_expert_proposal_with_inputs(proposal, report, grant, inputs),
            None => self.runtime.admit_expert_proposal(proposal, report, grant),
        }
    }

    /// A setup step: admit `op` or fail the harness.
    pub fn must(&mut self, what: &str, op: StructuralOp) -> Res<()> {
        self.admit(op)
            .map(|_| ())
            .map_err(|error| format!("{what}: {error:?}"))
    }

    pub fn must_with(
        &mut self,
        what: &str,
        op: StructuralOp,
        inputs: Option<ExpertInputs>,
        setup: Setup,
    ) -> Res<()> {
        self.admit_as(op, inputs, &ReportSpec::correct_under(setup), setup)
            .map(|_| ())
            .map_err(|error| format!("{what}: {error:?}"))
    }

    pub fn birth(&mut self, pod: &str) -> Res<()> {
        let op = StructuralOp::Birth {
            result: self.spec(pod),
        };
        self.must(&format!("birth {pod}"), op)
    }

    /// Birth the token-level expert `pod` resting on `inputs`.
    pub fn birth_with(&mut self, pod: &str, inputs: ExpertInputs) -> Res<()> {
        let op = StructuralOp::Birth {
            result: self.spec(pod),
        };
        self.must_with(&format!("birth {pod}"), op, Some(inputs), Setup::PRE)
    }

    pub fn birth_segment(&mut self, pod: &str, inputs: ExpertInputs) -> Res<()> {
        let op = StructuralOp::Birth {
            result: self.segment_spec(pod, 1),
        };
        self.must_with(&format!("birth {pod}"), op, Some(inputs), Setup::PRE)
    }

    pub fn freeze(&mut self, pod: &str) -> Res<()> {
        let op = StructuralOp::Freeze {
            target: self.target(pod)?,
        };
        self.must(&format!("freeze {pod}"), op)
    }

    pub fn unfreeze(&mut self, pod: &str) -> Res<()> {
        let op = StructuralOp::Reactivate {
            target: self.target(pod)?,
            result: None,
        };
        self.must(&format!("unfreeze {pod}"), op)
    }

    pub fn retire(&mut self, pod: &str, reason: RetiredReason) -> Res<()> {
        let op = StructuralOp::Retire {
            target: self.target(pod)?,
            reason,
        };
        self.must(&format!("retire {pod}"), op)
    }

    /// Reactivate the retired `pod` as its next generation (same pod, same slot).
    pub fn reactivate_retired(&mut self, pod: &str, inputs: Option<ExpertInputs>) -> Res<()> {
        let target = self.target(pod)?;
        let next = spec_at(
            &self.project,
            pod,
            target.generation().0 + 1,
            target.level(),
        );
        let op = StructuralOp::Reactivate {
            target,
            result: Some(next),
        };
        self.must_with(&format!("reactivate {pod}"), op, inputs, Setup::PRE)
    }

    pub fn merge(
        &mut self,
        sources: &[&str],
        result: ExpertPodSpec,
        inputs: Option<ExpertInputs>,
    ) -> Res<()> {
        let sources = sources
            .iter()
            .map(|pod| self.target(pod))
            .collect::<Res<Vec<_>>>()?;
        let op = StructuralOp::Merge { sources, result };
        self.must_with("merge", op, inputs, Setup::PRE)
    }

    pub fn split(
        &mut self,
        source: &str,
        results: Vec<ExpertPodSpec>,
        inputs: Option<ExpertInputs>,
    ) -> Res<()> {
        let op = StructuralOp::Split {
            source: self.target(source)?,
            results,
        };
        self.must_with("split", op, inputs, Setup::PRE)
    }

    pub fn prune(&mut self, pods: &[&str]) -> Res<()> {
        let targets = pods
            .iter()
            .map(|pod| self.target(pod))
            .collect::<Res<Vec<_>>>()?;
        self.must("prune", StructuralOp::Prune { targets })
    }

    /// Seed the registry for `scenario` (the operands it needs).
    pub fn seed(&mut self, scenario: Scenario) -> Res<()> {
        match scenario {
            Scenario::Birth => {}
            Scenario::Merge | Scenario::Prune => {
                self.birth("a")?;
                self.birth("b")?;
            }
            Scenario::Split | Scenario::Freeze | Scenario::Retire => self.birth("a")?,
            Scenario::ReactivateFrozen => {
                self.birth("a")?;
                self.freeze("a")?;
            }
            Scenario::ReactivateRetired => {
                self.birth("a")?;
                self.retire("a", RetiredReason::GateFailed)?;
            }
        }
        Ok(())
    }

    /// The operation of `scenario` over the registry, every operand passed through `tweak`.
    pub fn operation(&self, scenario: Scenario, tweak: Tweak<'_>) -> Res<StructuralOp> {
        Ok(match scenario {
            Scenario::Birth => StructuralOp::Birth {
                result: self.spec("n"),
            },
            Scenario::Merge => StructuralOp::Merge {
                sources: vec![self.target_with("a", tweak)?, self.target_with("b", tweak)?],
                result: self.spec("m"),
            },
            Scenario::Split => StructuralOp::Split {
                source: self.target_with("a", tweak)?,
                results: vec![self.spec("x"), self.spec("y")],
            },
            Scenario::Freeze => StructuralOp::Freeze {
                target: self.target_with("a", tweak)?,
            },
            Scenario::ReactivateFrozen => StructuralOp::Reactivate {
                target: self.target_with("a", tweak)?,
                result: None,
            },
            Scenario::ReactivateRetired => {
                let target = self.target_with("a", tweak)?;
                let next = spec_at(
                    &self.project,
                    "a",
                    target.generation().0 + 1,
                    target.level(),
                );
                StructuralOp::Reactivate {
                    target,
                    result: Some(next),
                }
            }
            Scenario::Prune => StructuralOp::Prune {
                targets: vec![self.target_with("a", tweak)?, self.target_with("b", tweak)?],
            },
            Scenario::Retire => StructuralOp::Retire {
                target: self.target_with("a", tweak)?,
                reason: RetiredReason::Superseded,
            },
        })
    }

    // ---- fences ----

    pub fn fence_request(&self, pod: &str, cause: ExpertFenceCause) -> Res<ExpertFenceRequest> {
        let (_, version) = self
            .state(pod)
            .ok_or_else(|| format!("expert {pod} is not in the registry"))?;
        let generation = self.init(pod)?.generation;
        Ok(ExpertFenceRequest {
            project: self.project.clone(),
            pod: PodId::from(pod),
            generation,
            expected_state_version: version,
            cause,
            revocation: None,
        })
    }

    /// The request that fences `pod` for the committed revocation of `(subject, generation)`.
    pub fn revoked_request(
        &self,
        pod: &str,
        subject: &str,
        generation: u64,
    ) -> Res<ExpertFenceRequest> {
        Ok(ExpertFenceRequest {
            revocation: Some(ExpertFenceRevocation {
                subject: subject.to_string(),
                generation: Generation(generation),
            }),
            ..self.fence_request(pod, ExpertFenceCause::InputRevoked)?
        })
    }

    pub fn revoke(&mut self, subject: &str, generation: u64) -> Res<()> {
        self.runtime
            .commit(LedgerEvent::Revoked {
                subject: subject.to_string(),
                generation: Generation(generation),
            })
            .map(|_| ())
            .map_err(fail("committing a revocation"))
    }

    pub fn fence_quarantine(&mut self, pod: &str) -> Res<()> {
        let request = self.fence_request(pod, ExpertFenceCause::AuthorityQuarantine)?;
        let grant = self.fence_grant();
        self.runtime
            .fence_expert(&request, &grant)
            .map(|_| ())
            .map_err(fail("fencing an expert"))
    }

    /// Fence `pod` for the committed revocation of `(subject, generation)`.
    pub fn fence_revoked(&mut self, pod: &str, subject: &str, generation: u64) -> Res<()> {
        let request = self.revoked_request(pod, subject, generation)?;
        let grant = self.fence_grant();
        self.runtime
            .fence_expert(&request, &grant)
            .map(|_| ())
            .map_err(fail("fencing an expert"))
    }

    /// Readmit the quarantined `pod` through the gate path under `setup`.
    pub fn readmit(&mut self, pod: &str, setup: Setup) -> Result<CommitIndex, RuntimeError> {
        let request = self
            .fence_request(pod, ExpertFenceCause::AuthorityQuarantine)
            .map_err(RuntimeError::InvalidConfig)?;
        let manifest = self.manifest(pod).map_err(RuntimeError::InvalidConfig)?;
        let report = readmission_report(&request, manifest, &ReportSpec::under(setup.gate()))
            .map_err(RuntimeError::InvalidConfig)?;
        let grant = self
            .gate_grant(setup)
            .map_err(RuntimeError::InvalidConfig)?;
        self.runtime.readmit_expert(&request, &report, &grant)
    }

    // ---- invalidation and authority ----

    pub fn invalidate(
        &mut self,
        request: InputInvalidation,
    ) -> Result<InputInvalidationOutcome, RuntimeError> {
        let grant = self.global_grant();
        self.runtime.invalidate_input(&grant, request)
    }

    pub fn change_authority(
        &mut self,
        kind: ExpertAuthorityKind,
        epoch: u64,
        class: ExpertAuthorityClass,
    ) -> Result<ExpertAuthorityOutcome, RuntimeError> {
        let id = match kind {
            ExpertAuthorityKind::GateConfig => {
                config_digest(&gate_v(epoch as u8)).map_err(RuntimeError::InvalidConfig)?
            }
            ExpertAuthorityKind::Evaluator => evaluator_id(epoch as u8),
        };
        let grant = self.global_grant();
        self.runtime
            .change_expert_authority(&grant, ExpertAuthorityChange::new(kind, id, epoch, class))
    }

    // ---- routes ----

    pub fn route(&self, pod: &str, generation: u64) -> Result<(), RouteRefusal> {
        self.runtime.expert_route_admission(
            &self.project,
            &PodId::from(pod),
            Generation(generation),
        )
    }

    /// Whether `pod` at `generation` is routable now. For positive controls only, where `Ok` is
    /// the expected answer; a probe of an expert that must be refused goes through
    /// `probe_routes` / `classify_route`, which check the kind of the refusal.
    pub fn routable(&self, pod: &str, generation: u64) -> bool {
        self.route(pod, generation).is_ok()
    }
}

impl ReportSpec {
    /// The report an honest evaluator produces for a grant of `setup`.
    pub fn correct_under(setup: Setup) -> Self {
        Self::under(setup.gate())
    }
}

/// The inputs the dependency cases rest experts on.
pub const TRAIN: &str = "dataset:train";
pub const EVAL: &str = "dataset:eval";
pub const OTHER: &str = "dataset:other";
pub const ADAPTER: &str = "adapter:lora-1";
pub const D1: [u8; 32] = [0x11; 32];
pub const D2: [u8; 32] = [0x22; 32];
pub const D3: [u8; 32] = [0x33; 32];
/// The digest of the replacement artifact a supersession names (evidence only).
pub const REPLACEMENT: [u8; 32] = [0x77; 32];
