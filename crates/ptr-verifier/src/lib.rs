//! Verifier contract, aggregation and deterministic verifiers.
//!
//! Besides the [`Verifier`] contract, the [`VerifierFabric`] and the [`ActionContractVerifier`],
//! the crate carries four deterministic task verifiers whose findings drive the finding router of
//! the verifier-closed workspace loop: [`ArithmeticVerifier`], [`JsonSchemaVerifier`],
//! [`CodeTestVerifier`] and [`FactVerifier`]. Their finding codes, which of them are hard and the
//! repair role each maps to are one table, in [`codes`].
//!
//! Further deterministic verifiers: [`RevisionVerifier`] (a claimed revision/generation against
//! the current snapshot binding), [`PromotionVerifier`] (a request to promote a claim to
//! verified/authoritative must cite committed, non-derived, independent evidence) and the
//! prototype [`ProcessCommandVerifier`] (an allowlisted child process under a wall-clock
//! timeout; no namespaces or seccomp). [`bound`] adds [`BoundVerificationReport`], which carries
//! the snapshot a verdict was judged against next to the unchanged [`VerificationReport`].

pub mod arithmetic;
mod bigint;
pub mod bound;
pub mod code_test;
pub mod codes;
pub mod fact;
pub mod json_schema;
pub mod process;
pub mod promotion;
pub mod revision;

pub use arithmetic::ArithmeticVerifier;
pub use bound::{
    AtSnapshot, BoundVerificationReport, BoundVerifier, BoundVerifierFabric, NamedBoundVerifier,
    SnapshotBinding,
};
pub use code_test::{CodeTask, CodeTest, CodeTestVerifier, TestVisibility};
pub use codes::{code_spec, codes_for_role, repair_role_for, CodeSpec, RepairRole, CODE_TABLE};
pub use fact::{fact_key, BoundReport, FactClaim, FactVerifier};
pub use json_schema::JsonSchemaVerifier;
pub use process::{
    CommandRequest, ProcessCommandVerifier, ProcessConfig, ProcessConfigError, ProcessExit,
    ProcessLimits, ProcessRejection, ProcessRun, MAX_STUCK_CAPTURE_THREADS,
};
pub use promotion::{
    Evidence, EvidenceOrigin, PromotionClaim, PromotionPolicy, PromotionPolicyError,
    PromotionTarget, PromotionVerifier,
};
pub use revision::{RevisionClaim, RevisionVerifier};

use ptr_types::{ActionIr, CapabilityId, Effect, Probability, TypeId, VerificationLevel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerificationStatus {
    Pass,
    Fail,
    Disputed,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub code: String,
    pub message: String,
    pub hard: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerificationReport {
    pub status: VerificationStatus,
    pub level: VerificationLevel,
    pub score: Probability,
    pub findings: Vec<Finding>,
}

pub trait Verifier<T> {
    fn verify(&self, candidate: &T) -> VerificationReport;
}

pub trait NamedVerifier<T>: Verifier<T> + Send + Sync {
    fn name(&self) -> &'static str;
}

/// Runs every verifier pushed on a candidate and combines their reports into
/// one. The combined report carries:
///
/// - the worst status, except that any hard finding forces `Fail`: a
///   deterministic contradiction dominates whatever the other verifiers
///   (learned or statistical ones included) said;
/// - the lowest score and every finding;
/// - as its level the WEAKEST level any report reached. A fabric is only as
///   deep as its shallowest member, so a weaker verifier must not let the
///   fabric satisfy a stronger requirement: a fabric of one deterministic and
///   one latent-agreement verifier is `LatentAgreement`, never `Deterministic`.
///
/// With no verifier the report is `Unknown` at `Unverified` with score 0. The
/// fabric implements [`Verifier`], so it can be installed wherever a single
/// verifier is expected.
pub struct VerifierFabric<T> {
    verifiers: Vec<Box<dyn NamedVerifier<T>>>,
}

// Not derived: a derive would require `T: Default`, and a fabric holds no `T`, so it could not
// be built for a type such as `ActionIr`.
impl<T> Default for VerifierFabric<T> {
    fn default() -> Self {
        Self {
            verifiers: Vec::new(),
        }
    }
}

impl<T> VerifierFabric<T> {
    pub fn push<V>(&mut self, verifier: V)
    where
        V: NamedVerifier<T> + 'static,
    {
        self.verifiers.push(Box::new(verifier));
    }

    pub fn is_empty(&self) -> bool {
        self.verifiers.is_empty()
    }
}

impl<T> Verifier<T> for VerifierFabric<T> {
    fn verify(&self, candidate: &T) -> VerificationReport {
        if self.verifiers.is_empty() {
            return VerificationReport {
                status: VerificationStatus::Unknown,
                level: VerificationLevel::Unverified,
                score: Probability::new(0.0).expect("zero probability"),
                findings: vec![],
            };
        }

        let mut reports = self
            .verifiers
            .iter()
            .map(|verifier| verifier.verify(candidate))
            .collect::<Vec<_>>();

        let hard = reports
            .iter()
            .any(|report| report.findings.iter().any(|finding| finding.hard));
        let status = if hard {
            VerificationStatus::Fail
        } else {
            reports
                .iter()
                .map(|report| report.status)
                .max_by_key(|status| status_rank(*status))
                .expect("non-empty verifier reports")
        };

        let level = reports
            .iter()
            .map(|report| report.level)
            .min_by_key(|level| level_rank(*level))
            .expect("non-empty verifier reports");

        let min_score = reports
            .iter()
            .map(|report| report.score.get())
            .fold(1.0_f32, f32::min);

        let findings = reports
            .drain(..)
            .flat_map(|report| report.findings)
            .collect();

        VerificationReport {
            status,
            level,
            score: Probability::new(min_score).expect("score remains bounded"),
            findings,
        }
    }
}

/// Whether an action payload must be valid UTF-8 text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayloadEncoding {
    Utf8,
    Opaque,
}

/// The exact contract an action must meet, field by field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionContract {
    pub target: String,
    pub operation: String,
    pub capability: CapabilityId,
    pub input_type: TypeId,
    pub effect: Effect,
    pub max_payload_bytes: usize,
    pub payload_encoding: PayloadEncoding,
}

/// Deterministic verifier for an [`ActionIr`]: target, operation, capability,
/// input type, effect, payload size and payload encoding must all match its
/// [`ActionContract`]. Every violation is one hard finding with a stable code
/// (`target`, `operation`, `capability`, `input-type`, `effect`,
/// `payload-size`, `payload-utf8`); a violated contract is `Fail` at
/// `Deterministic` level with score 0, a met one `Pass` with score 1.
///
/// Revision and generation are not checked here: they are the runtime's
/// authorization step, which compares them with the live state.
#[derive(Clone, Debug)]
pub struct ActionContractVerifier {
    contract: ActionContract,
}

impl ActionContractVerifier {
    pub fn new(contract: ActionContract) -> Self {
        Self { contract }
    }

    pub fn contract(&self) -> &ActionContract {
        &self.contract
    }
}

fn violation(code: &str, message: String) -> Finding {
    Finding {
        code: code.into(),
        message,
        hard: true,
    }
}

impl Verifier<ActionIr> for ActionContractVerifier {
    fn verify(&self, action: &ActionIr) -> VerificationReport {
        let contract = &self.contract;
        let mut findings = Vec::new();
        if action.target != contract.target {
            findings.push(violation(
                "target",
                "action target differs from contract".into(),
            ));
        }
        if action.operation != contract.operation {
            findings.push(violation(
                "operation",
                "action operation differs from contract".into(),
            ));
        }
        if action.capability != contract.capability {
            findings.push(violation(
                "capability",
                "action capability differs from contract".into(),
            ));
        }
        if action.input_type != contract.input_type {
            findings.push(violation(
                "input-type",
                "action input type differs from contract".into(),
            ));
        }
        if action.effect != contract.effect {
            findings.push(violation(
                "effect",
                "action effect differs from contract".into(),
            ));
        }
        if action.payload.len() > contract.max_payload_bytes {
            findings.push(violation(
                "payload-size",
                format!(
                    "payload is {} bytes, contract allows {}",
                    action.payload.len(),
                    contract.max_payload_bytes
                ),
            ));
        }
        if contract.payload_encoding == PayloadEncoding::Utf8
            && std::str::from_utf8(&action.payload).is_err()
        {
            findings.push(violation("payload-utf8", "payload is not UTF-8".into()));
        }
        let passed = findings.is_empty();
        VerificationReport {
            status: if passed {
                VerificationStatus::Pass
            } else {
                VerificationStatus::Fail
            },
            level: VerificationLevel::Deterministic,
            score: Probability::new(if passed { 1.0 } else { 0.0 }).expect("bounded score"),
            findings,
        }
    }
}

impl NamedVerifier<ActionIr> for ActionContractVerifier {
    fn name(&self) -> &'static str {
        "action-contract"
    }
}

fn status_rank(status: VerificationStatus) -> u8 {
    match status {
        VerificationStatus::Pass => 0,
        VerificationStatus::Unknown => 1,
        VerificationStatus::Disputed => 2,
        VerificationStatus::Fail => 3,
    }
}

fn level_rank(level: VerificationLevel) -> u8 {
    match level {
        VerificationLevel::Unverified => 0,
        VerificationLevel::LatentAgreement => 1,
        VerificationLevel::SampleVerified => 2,
        VerificationLevel::FullSemantic => 3,
        VerificationLevel::Deterministic => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::{Generation, Revision};

    fn contract() -> ActionContract {
        ActionContract {
            target: "note".into(),
            operation: "create".into(),
            capability: CapabilityId::from("note.create"),
            input_type: TypeId::from("note.v1"),
            effect: Effect::Mutation,
            max_payload_bytes: 8,
            payload_encoding: PayloadEncoding::Utf8,
        }
    }

    fn action() -> ActionIr {
        ActionIr {
            operation: "create".into(),
            target: "note".into(),
            capability: CapabilityId::from("note.create"),
            effect: Effect::Mutation,
            input_type: TypeId::from("note.v1"),
            generation: Generation(1),
            revision: Revision(0),
            payload: b"hello".to_vec(),
        }
    }

    fn codes(report: &VerificationReport) -> Vec<&str> {
        report.findings.iter().map(|f| f.code.as_str()).collect()
    }

    #[test]
    fn matching_action_passes_deterministically() {
        let report = ActionContractVerifier::new(contract()).verify(&action());
        assert_eq!(report.status, VerificationStatus::Pass);
        assert_eq!(report.level, VerificationLevel::Deterministic);
        assert_eq!(report.score.get(), 1.0);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn each_violation_has_its_own_hard_finding_code() {
        let verifier = ActionContractVerifier::new(contract());
        type Mutation = fn(&mut ActionIr);
        let cases: [(&str, Mutation); 7] = [
            ("target", |a| a.target = "other".into()),
            ("operation", |a| a.operation = "delete".into()),
            ("capability", |a| a.capability = CapabilityId::from("x")),
            ("input-type", |a| a.input_type = TypeId::from("x")),
            ("effect", |a| a.effect = Effect::Read),
            ("payload-size", |a| a.payload = vec![b'a'; 9]),
            ("payload-utf8", |a| a.payload = vec![0xff, 0xfe]),
        ];
        for (code, mutate) in cases {
            let mut candidate = action();
            mutate(&mut candidate);
            let report = verifier.verify(&candidate);
            assert_eq!(report.status, VerificationStatus::Fail, "{code}");
            assert_eq!(report.level, VerificationLevel::Deterministic, "{code}");
            assert_eq!(report.score.get(), 0.0, "{code}");
            assert_eq!(codes(&report), [code]);
            assert!(report.findings[0].hard);
        }
    }

    #[test]
    fn several_violations_report_several_findings() {
        let mut candidate = action();
        candidate.target = "other".into();
        candidate.payload = vec![0xff; 9];
        let report = ActionContractVerifier::new(contract()).verify(&candidate);
        assert_eq!(codes(&report), ["target", "payload-size", "payload-utf8"]);
    }

    #[test]
    fn opaque_payload_may_be_non_utf8() {
        let mut opaque = contract();
        opaque.payload_encoding = PayloadEncoding::Opaque;
        let mut candidate = action();
        candidate.payload = vec![0xff, 0xfe];
        let report = ActionContractVerifier::new(opaque).verify(&candidate);
        assert_eq!(report.status, VerificationStatus::Pass);
    }

    #[test]
    fn action_contract_verifier_is_named() {
        assert_eq!(
            NamedVerifier::<ActionIr>::name(&ActionContractVerifier::new(contract())),
            "action-contract"
        );
    }

    struct Fixed(VerificationStatus, VerificationLevel, bool);
    impl Verifier<u32> for Fixed {
        fn verify(&self, _: &u32) -> VerificationReport {
            VerificationReport {
                status: self.0,
                level: self.1,
                score: Probability::new(0.5).unwrap(),
                findings: if self.2 {
                    vec![violation("hard", "hard".into())]
                } else {
                    vec![]
                },
            }
        }
    }
    impl NamedVerifier<u32> for Fixed {
        fn name(&self) -> &'static str {
            "fixed"
        }
    }

    #[test]
    fn fabric_level_is_the_weakest_level() {
        let mut fabric = VerifierFabric::default();
        fabric.push(Fixed(
            VerificationStatus::Pass,
            VerificationLevel::Deterministic,
            false,
        ));
        fabric.push(Fixed(
            VerificationStatus::Pass,
            VerificationLevel::LatentAgreement,
            false,
        ));
        let report = fabric.verify(&0);
        assert_eq!(report.status, VerificationStatus::Pass);
        assert_eq!(report.level, VerificationLevel::LatentAgreement);
    }

    #[test]
    fn hard_finding_forces_fail_over_a_passing_status() {
        let mut fabric = VerifierFabric::default();
        fabric.push(Fixed(
            VerificationStatus::Pass,
            VerificationLevel::Deterministic,
            true,
        ));
        fabric.push(Fixed(
            VerificationStatus::Pass,
            VerificationLevel::SampleVerified,
            false,
        ));
        let report = fabric.verify(&0);
        assert_eq!(report.status, VerificationStatus::Fail);
        assert_eq!(report.level, VerificationLevel::SampleVerified);
    }

    #[test]
    fn worst_status_wins_without_hard_findings() {
        let mut fabric = VerifierFabric::default();
        fabric.push(Fixed(
            VerificationStatus::Unknown,
            VerificationLevel::FullSemantic,
            false,
        ));
        fabric.push(Fixed(
            VerificationStatus::Disputed,
            VerificationLevel::FullSemantic,
            false,
        ));
        assert_eq!(fabric.verify(&0).status, VerificationStatus::Disputed);
    }

    #[test]
    fn a_fabric_can_be_built_for_a_type_without_a_default() {
        let mut fabric = VerifierFabric::<ActionIr>::default();
        assert!(fabric.is_empty());
        fabric.push(ActionContractVerifier::new(contract()));
        assert!(!fabric.is_empty());
        assert_eq!(fabric.verify(&action()).status, VerificationStatus::Pass);
    }

    #[test]
    fn empty_fabric_is_unknown_and_unverified() {
        let fabric = VerifierFabric::<u32>::default();
        let report = fabric.verify(&0);
        assert!(fabric.is_empty());
        assert_eq!(report.status, VerificationStatus::Unknown);
        assert_eq!(report.level, VerificationLevel::Unverified);
        assert_eq!(report.score.get(), 0.0);
    }
}
