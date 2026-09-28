//! The host policy every semantic write that is not ingress is admitted
//! under, what its verifiers judge, and the one way an agent branch is
//! merged.
//!
//! A semantic record reaches the ledger in one of three ways. Ingress writes
//! the raw text of a request ([`PtrRuntime::ingest_text`]) or a Pod's
//! verified output, in shapes the runtime fixes and replay checks again. A
//! host write ([`PtrRuntime::apply_verified_semantic_delta`]) is admitted
//! only by every verifier of the [`SemanticGrant`] the host installed, and
//! its record names the principal, the verifiers, the level they reached and
//! their soft findings. A merge ([`PtrRuntime::merge_branch`]) certifies a
//! sealed agent branch against the runtime's own state, is admitted by the
//! same verifiers, commits only under the grant's merge policy or the
//! approval of a reviewer the grant lists, and its record names the branch,
//! its author, the sealed branch, the plan, the verifiers and the authority,
//! at most once per branch id.
//!
//! A grant is installed once and is not history: nothing of it is journaled
//! but what a record it admits names. Replay re-checks everything that is
//! deterministic about a record and never runs a verifier again.
//!
//! [`PtrRuntime::ingest_text`]: crate::PtrRuntime::ingest_text
//! [`PtrRuntime::apply_verified_semantic_delta`]: crate::PtrRuntime::apply_verified_semantic_delta
//! [`PtrRuntime::merge_branch`]: crate::PtrRuntime::merge_branch
use std::collections::BTreeSet;
use std::fmt;

use ptr_branch::{
    calibration_draw, certify, BranchId, Certification, CertificationKind, MergePlan, PolicyRecord,
    SealedBranch, TriageDecision, TriageOutcome, TriageOutcomeParts,
};
use ptr_events::RuntimeEvent;
use ptr_ledger::{Attestation, LedgerEvent, MergeAuthorityRecord, MergeRecord, SemanticOrigin};
use ptr_semdb::{PreparedDelta, PreparedView, SemanticDelta};
use ptr_types::{CommitIndex, PrincipalId, Probability, Revision, VerificationLevel};
use ptr_verifier::{NamedVerifier, VerificationReport, VerificationStatus};

use crate::execution::{self, RequiredVerification};
use crate::semantic::{evicted_ingress_key, written_ingress_key, SemanticCommit};
use crate::{PtrRuntime, RuntimeError};

/// Most verifiers a grant installs, and so most a record names.
pub const MAX_SEMANTIC_VERIFIERS: usize = ptr_ledger::MAX_ATTESTATION_VERIFIERS;
/// Longest verifier name, in bytes.
pub const MAX_VERIFIER_NAME: usize = 64;
/// Longest finding code a verifier may report, in bytes.
pub const MAX_FINDING_CODE: usize = 64;
/// Longest provenance text a record carries, in bytes: a host write's
/// principal, a merged branch's id and author, a reviewer and a merge
/// policy's version.
pub const MAX_PROVENANCE_TEXT: usize = 256;
/// Most soft findings one record carries, over all its verifiers.
pub const MAX_ATTESTED_FINDINGS: usize = ptr_ledger::MAX_ATTESTATION_FINDINGS;

/// What the host allows semantic writes that are not ingress to do, and which
/// verifiers must admit each one. Installed once
/// ([`PtrRuntime::install_semantic_grant`](crate::PtrRuntime::install_semantic_grant))
/// and never journaled.
///
/// Every verifier judges every change, and the change is admitted only if
/// every one passes with no hard finding and the weakest level any of them
/// reports meets the requirement. Host writes are off until
/// [`SemanticGrant::allow_host_writes`] turns them on. A merge the verifiers
/// admit commits under the merge policy
/// ([`SemanticGrant::with_merge_policy`], [`MergeAuthority::Triage`]) or on
/// the approval of a listed reviewer ([`SemanticGrant::with_reviewer`],
/// [`MergeAuthority::Reviewed`]); with neither, no merge commits.
pub struct SemanticGrant {
    required: RequiredVerification,
    verifiers: Vec<Box<dyn for<'a> NamedVerifier<SemanticChange<'a>>>>,
    host_writes: bool,
    /// The merge policy and the seed of its calibration draws, which is
    /// never recorded, returned or printed.
    merge_policy: Option<(PolicyRecord, u64)>,
    reviewers: BTreeSet<PrincipalId>,
}

impl SemanticGrant {
    /// A grant requiring `required` of every verifier, with no verifier yet
    /// and host writes off. It cannot be installed until it has a verifier.
    pub fn new(required: RequiredVerification) -> Self {
        Self {
            required,
            verifiers: Vec::new(),
            host_writes: false,
            merge_policy: None,
            reviewers: BTreeSet::new(),
        }
    }

    /// Add a verifier; every installed verifier judges every change, in the
    /// order they were added.
    pub fn with_verifier<V>(mut self, verifier: V) -> Self
    where
        V: for<'a> NamedVerifier<SemanticChange<'a>> + 'static,
    {
        self.verifiers.push(Box::new(verifier));
        self
    }

    /// Let the host write semantic deltas of its own through
    /// [`PtrRuntime::apply_verified_semantic_delta`](crate::PtrRuntime::apply_verified_semantic_delta).
    pub fn allow_host_writes(mut self) -> Self {
        self.host_writes = true;
        self
    }

    /// Let the runtime triage merges under `policy`
    /// ([`MergeAuthority::Triage`]): a merge the verifiers admit commits only
    /// when the policy auto-proposes it. Whether a branch falls in the
    /// policy's calibration slice is drawn from its id and
    /// `calibration_seed` ([`ptr_branch::calibration_draw`]); the seed is
    /// never recorded, returned or printed, so no one who can read the
    /// runtime can tell in advance which branches a person will see. A
    /// second call replaces the first.
    pub fn with_merge_policy(mut self, policy: PolicyRecord, calibration_seed: u64) -> Self {
        self.merge_policy = Some((policy, calibration_seed));
        self
    }

    /// List a reviewer whose approval of a plan lets a merge the verifiers
    /// admit commit ([`MergeAuthority::Reviewed`]). No approval lets a merge
    /// the verifiers did not admit commit.
    pub fn with_reviewer(mut self, reviewer: PrincipalId) -> Self {
        self.reviewers.insert(reviewer);
        self
    }

    pub(crate) fn info(&self) -> SemanticGrantInfo {
        SemanticGrantInfo {
            required: self.required,
            verifiers: self.names().map(str::to_owned).collect(),
            host_writes: self.host_writes,
            policy_version: self
                .merge_policy
                .as_ref()
                .map(|(policy, _)| policy.version().to_owned()),
            reviewers: self.reviewers.clone(),
        }
    }

    pub(crate) fn host_writes(&self) -> bool {
        self.host_writes
    }

    fn merge_policy(&self) -> Option<(&PolicyRecord, u64)> {
        self.merge_policy
            .as_ref()
            .map(|(policy, seed)| (policy, *seed))
    }

    fn lists_reviewer(&self, reviewer: &PrincipalId) -> bool {
        self.reviewers.contains(reviewer)
    }

    fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.verifiers.iter().map(|verifier| verifier.name())
    }

    /// Whether the grant can be installed: 1 to [`MAX_SEMANTIC_VERIFIERS`]
    /// verifiers whose names are distinct identifiers of at most
    /// [`MAX_VERIFIER_NAME`] bytes without a `/`, which separates a name from
    /// a finding code in a record, and a merge policy version and reviewers
    /// that are identifiers of at most [`MAX_PROVENANCE_TEXT`] bytes, since a
    /// merge record names them.
    pub(crate) fn check(&self) -> Result<(), RuntimeError> {
        if self.verifiers.is_empty() {
            return Err(RuntimeError::InvalidSemanticGrant {
                reason: "a grant needs at least one verifier",
            });
        }
        if self.verifiers.len() > MAX_SEMANTIC_VERIFIERS {
            return Err(RuntimeError::InvalidSemanticGrant {
                reason: "a grant has more verifiers than a record can name",
            });
        }
        let mut seen = BTreeSet::new();
        for name in self.names() {
            if !valid_verifier_name(name) {
                return Err(RuntimeError::InvalidSemanticGrant {
                    reason: "a verifier name is not an identifier of at most 64 bytes without '/'",
                });
            }
            if !seen.insert(name) {
                return Err(RuntimeError::InvalidSemanticGrant {
                    reason: "two verifiers share a name",
                });
            }
        }
        if let Some((policy, _)) = &self.merge_policy {
            if !valid_provenance(policy.version()) {
                return Err(RuntimeError::InvalidSemanticGrant {
                    reason: "a merge policy version is not an identifier of at most 256 bytes",
                });
            }
        }
        if self
            .reviewers
            .iter()
            .any(|reviewer| !valid_provenance(&reviewer.0))
        {
            return Err(RuntimeError::InvalidSemanticGrant {
                reason: "a reviewer is not an identifier of at most 256 bytes",
            });
        }
        Ok(())
    }

    /// Let every verifier judge `change`, in grant order, and combine their
    /// reports, with each verifier's result as far as they got.
    ///
    /// The verdict is [`RuntimeError::InvalidVerificationReport`] naming the
    /// verifier whose report has a finding code that is not an identifier of
    /// at most [`MAX_FINDING_CODE`] bytes, or whose soft findings take the
    /// record past [`MAX_ATTESTED_FINDINGS`]. Such a report fails closed
    /// rather than be recorded as something else; it is that verifier's
    /// result, not passed, and no verifier after it judges.
    pub(crate) fn judge(&self, change: &SemanticChange<'_>) -> Judgement {
        let mut results = Vec::with_capacity(self.verifiers.len());
        let mut reports = Vec::with_capacity(self.verifiers.len());
        let mut soft = BTreeSet::new();
        for verifier in &self.verifiers {
            let name = verifier.name();
            let report = verifier.verify(change);
            let mut fits = true;
            for finding in &report.findings {
                if !valid_finding_code(&finding.code) {
                    fits = false;
                    break;
                }
                if !finding.hard {
                    soft.insert(format!("{name}/{}", finding.code));
                    if soft.len() > MAX_ATTESTED_FINDINGS {
                        fits = false;
                        break;
                    }
                }
            }
            if !fits {
                results.push((name, false));
                return Judgement {
                    results,
                    verdict: Err(RuntimeError::InvalidVerificationReport {
                        verifier: name.to_owned(),
                    }),
                };
            }
            let passed = report.status == VerificationStatus::Pass
                && self.required.accepts(report.level)
                && !report.findings.iter().any(|finding| finding.hard);
            results.push((name, passed));
            reports.push((name, report));
        }
        let status = reports
            .iter()
            .map(|(_, report)| report.status)
            .max_by_key(|status| status_rank(*status))
            .expect("an installed grant has a verifier");
        let level = reports
            .iter()
            .map(|(_, report)| report.level)
            .min_by_key(|level| level_rank(*level))
            .expect("an installed grant has a verifier");
        let score = reports
            .iter()
            .map(|(_, report)| report.score)
            .min_by(|left, right| left.get().total_cmp(&right.get()))
            .expect("an installed grant has a verifier");
        Judgement {
            results,
            verdict: Ok(SemanticVerdict {
                required: self.required,
                status,
                level,
                score,
                reports,
            }),
        }
    }
}

/// What a grant's verifiers made of one change.
pub(crate) struct Judgement {
    /// Each verifier that judged, in grant order, and whether it passed:
    /// `Pass` at a level the grant accepts with no hard finding. A verifier
    /// whose report failed closed is the last, and did not pass.
    pub(crate) results: Vec<(&'static str, bool)>,
    /// The combined verdict, or the refusal of a report that failed closed.
    pub(crate) verdict: Result<SemanticVerdict, RuntimeError>,
}

/// Prints the merge policy by its version and never the calibration seed.
impl fmt::Debug for SemanticGrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemanticGrant")
            .field("required", &self.required)
            .field("verifiers", &self.names().collect::<Vec<_>>())
            .field("host_writes", &self.host_writes)
            .field(
                "merge_policy",
                &self
                    .merge_policy
                    .as_ref()
                    .map(|(policy, _)| policy.version()),
            )
            .field("reviewers", &self.reviewers)
            .finish()
    }
}

/// What a host may read back about the installed grant: never the
/// calibration seed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticGrantInfo {
    pub required: RequiredVerification,
    /// The verifiers' names, in grant order.
    pub verifiers: Vec<String>,
    pub host_writes: bool,
    /// The merge policy's version, if the grant has one.
    pub policy_version: Option<String>,
    pub reviewers: BTreeSet<PrincipalId>,
}

/// What a semantic verifier judges: one prepared change, before it is
/// appended. Only the runtime builds one.
///
/// It shows the published state the change applies to
/// ([`SemanticChange::before`]), the exact state it would publish, derived
/// keys it evicts included ([`SemanticChange::after`]), exactly the delta that
/// would be recorded ([`SemanticChange::delta`]), the keys it invalidates
/// ([`SemanticChange::affected`]) and who asks for it
/// ([`SemanticChange::origin`]). A domain invariant is checked on `after`.
/// It never shows a merge's score, approval or triage: verification comes
/// first, and nothing after it can outvote it.
///
/// Its fields are private, so no change can be written out, whole or as an
/// update of one a verifier was given:
///
/// ```compile_fail
/// fn rebase<'a>(change: ptr_runtime::SemanticChange<'a>) -> ptr_runtime::SemanticChange<'a> {
///     ptr_runtime::SemanticChange { base: ptr_types::Revision(0), ..change }
/// }
/// ```
///
/// while the same function returning the change it was given compiles:
///
/// ```
/// fn rebase<'a>(change: ptr_runtime::SemanticChange<'a>) -> ptr_runtime::SemanticChange<'a> {
///     change
/// }
/// ```
pub struct SemanticChange<'a> {
    pub(crate) base: Revision,
    pub(crate) next: Revision,
    pub(crate) before: PreparedView<'a>,
    pub(crate) after: PreparedView<'a>,
    pub(crate) delta: &'a SemanticDelta,
    pub(crate) affected: &'a BTreeSet<String>,
    pub(crate) origin: ChangeOrigin<'a>,
}

impl<'a> SemanticChange<'a> {
    /// The revision the change applies to.
    pub fn base_revision(&self) -> Revision {
        self.base
    }

    /// The revision it would publish; the base revision for a no-op.
    pub fn next_revision(&self) -> Revision {
        self.next
    }

    /// The published state it applies to, with values and dependency sets.
    pub fn before(&self) -> PreparedView<'a> {
        self.before
    }

    /// The exact state it would publish, with values and dependency sets.
    pub fn after(&self) -> PreparedView<'a> {
        self.after
    }

    /// Exactly the delta the record would carry.
    pub fn delta(&self) -> &'a SemanticDelta {
        self.delta
    }

    /// The keys the change invalidates: those it writes and those derived
    /// from them.
    pub fn affected(&self) -> &'a BTreeSet<String> {
        self.affected
    }

    /// Who asks for the change.
    pub fn origin(&self) -> ChangeOrigin<'a> {
        self.origin
    }
}

/// Who asks for a change a grant judges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChangeOrigin<'a> {
    /// The host, naming the principal it writes for.
    Host { principal: &'a PrincipalId },
    /// A merge of `branch`, certified against the runtime's state as `plan`:
    /// the sealed branch's reads, scans, relied generations, touched digests,
    /// operations and author, and the plan's revision, delta, rebased keys
    /// and dependency digest.
    Merge {
        branch: &'a SealedBranch,
        plan: &'a MergePlan,
        kind: CertificationKind,
    },
}

/// The combined judgement of every verifier of a grant.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticVerdict {
    pub required: RequiredVerification,
    /// The worst status any verifier reported (`Fail` over `Disputed` over
    /// `Unknown` over `Pass`).
    pub status: VerificationStatus,
    /// The weakest level any verifier reported.
    pub level: VerificationLevel,
    /// The lowest score any verifier reported.
    pub score: Probability,
    /// Each verifier's report, in grant order.
    pub reports: Vec<(&'static str, VerificationReport)>,
}

impl SemanticVerdict {
    /// Every verifier passed, the weakest level meets the requirement, and no
    /// report has a hard finding.
    pub fn admitted(&self) -> bool {
        self.status == VerificationStatus::Pass
            && self.required.accepts(self.level)
            && self.hard_findings().is_empty()
    }

    /// Every hard finding, as `<verifier>/<code>`, in grant order.
    pub fn hard_findings(&self) -> Vec<String> {
        self.findings(true)
    }

    /// The combined report: the worst status, weakest level and lowest score,
    /// with every finding's code prefixed by its verifier's name.
    pub fn report(&self) -> VerificationReport {
        VerificationReport {
            status: self.status,
            level: self.level,
            score: self.score,
            findings: self
                .reports
                .iter()
                .flat_map(|(name, report)| {
                    report
                        .findings
                        .iter()
                        .map(move |finding| ptr_verifier::Finding {
                            code: format!("{name}/{}", finding.code),
                            ..finding.clone()
                        })
                })
                .collect(),
        }
    }

    fn findings(&self, hard: bool) -> Vec<String> {
        self.reports
            .iter()
            .flat_map(|(name, report)| {
                report
                    .findings
                    .iter()
                    .filter(move |finding| finding.hard == hard)
                    .map(move |finding| format!("{name}/{}", finding.code))
            })
            .collect()
    }

    /// What an admitted change's record names: the requirement, the weakest
    /// level, the verifiers in grant order and the soft findings, sorted and
    /// distinct.
    pub(crate) fn attestation(&self) -> Attestation {
        let soft: BTreeSet<String> = self.findings(false).into_iter().collect();
        Attestation {
            required: required_level(self.required),
            level: self.level,
            verifiers: self
                .reports
                .iter()
                .map(|(name, _)| (*name).to_owned())
                .collect(),
            findings: soft.into_iter().collect(),
        }
    }

    pub(crate) fn refusal(&self) -> SemanticRefusal {
        SemanticRefusal {
            status: self.status,
            level: self.level,
            hard_findings: self.hard_findings(),
        }
    }
}

/// Why a grant's verifiers did not admit a change: the combined status and
/// weakest level, and each hard finding's code as `<verifier>/<code>`, never
/// its message. Nothing was appended.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticRefusal {
    pub status: VerificationStatus,
    pub level: VerificationLevel,
    pub hard_findings: Vec<String>,
}

impl SemanticRefusal {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        "PTR_RUNTIME_SEMANTIC_VERIFICATION_REJECTED"
    }
}

impl fmt::Display for SemanticRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:?} at {:?}, hard findings {:?}",
            self.code(),
            self.status,
            self.level,
            self.hard_findings
        )
    }
}

impl std::error::Error for SemanticRefusal {}

/// Who lets a merge the grant's verifiers admit commit.
#[derive(Clone, Debug, PartialEq)]
pub enum MergeAuthority {
    /// The runtime triages the branch under the grant's merge policy and
    /// commits it only if the policy auto-proposes it. `score` is the host's
    /// calibrated score for the branch; the record carries it.
    Triage { score: Probability },
    /// A reviewer the grant lists approved exactly the plan whose digest is
    /// `plan_digest`, as [`PtrRuntime::preview_merge`] or a [`MergeHold`]
    /// showed it. Any change to the plan since voids the approval.
    Reviewed {
        plan_digest: [u8; 32],
        reviewer: PrincipalId,
    },
}

/// A branch certified against the runtime's current state and judged by the
/// grant's verifiers: what a reviewer approves and triage decides on.
#[derive(Clone, Debug, PartialEq)]
pub struct MergePreview {
    pub certification: Certification,
    /// The sealed branch's digest (`SealedBranch::seal_digest`).
    pub seal_digest: [u8; 32],
    /// The plan's digest (`MergePlan::digest`): what a reviewer approves.
    pub plan_digest: [u8; 32],
    pub verdict: SemanticVerdict,
    /// The keys the merge would invalidate.
    pub affected: BTreeSet<String>,
    /// The revision the branch was certified against.
    pub revision: Revision,
}

/// What [`PtrRuntime::merge_branch`] did with a branch it certified.
#[derive(Clone, Debug, PartialEq)]
pub enum MergeOutcome {
    /// The merge was appended.
    Committed(MergeReceipt),
    /// The verifiers admitted it, but it changes nothing: nothing was
    /// appended and the branch id is not marked merged, so a later merge of
    /// it that changes something is its first.
    NoChange(MergePreview),
    /// It was not committed, and nothing was appended.
    Held(MergeHold),
}

/// A committed merge, as its record names it.
#[derive(Clone, Debug, PartialEq)]
pub struct MergeReceipt {
    pub branch: BranchId,
    pub seal_digest: [u8; 32],
    pub plan_digest: [u8; 32],
    pub kind: CertificationKind,
    /// The keys certification rebased onto the runtime's values.
    pub rebased: BTreeSet<String>,
    /// Its commit index is always `Some`.
    pub commit: SemanticCommit,
    pub verdict: SemanticVerdict,
    /// The authority, as recorded.
    pub authority: MergeAuthorityRecord,
    /// The triage, under [`MergeAuthority::Triage`], to be logged with the
    /// policy version.
    pub triage: Option<TriageOutcome>,
    pub policy_version: Option<String>,
}

/// A merge the runtime did not commit, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct MergeHold {
    pub preview: MergePreview,
    pub reason: HoldReason,
    /// Under [`MergeAuthority::Triage`], the triage, to be logged with the
    /// policy version: discarded when a verifier failed the change,
    /// escalated otherwise.
    pub triage: Option<TriageOutcome>,
    pub policy_version: Option<String>,
}

/// Why a merge was held.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HoldReason {
    /// The grant's verifiers did not admit the change. No authority
    /// overrides that.
    VerificationRejected,
    /// The verifiers admitted it, but the merge policy did not auto-propose
    /// it: a person decides, by approving the preview's plan digest.
    Escalated,
}

/// A merge authority the grant allows, with what it needs from the grant.
enum Authorized<'g> {
    Triage {
        policy: &'g PolicyRecord,
        seed: u64,
        score: Probability,
    },
    Reviewed {
        plan_digest: [u8; 32],
        reviewer: PrincipalId,
    },
}

/// A branch certified here, before its verifiers judge it.
struct CertifiedMerge {
    certification: Certification,
    seal_digest: [u8; 32],
    plan_digest: [u8; 32],
    revision: Revision,
}

/// A certified branch the grant's verifiers judged: each verifier's result,
/// and the merge prepared to be appended or the refusal of a report that
/// failed closed.
struct JudgedMerge {
    results: Vec<(&'static str, bool)>,
    merge: Result<PreparedMerge, RuntimeError>,
}

/// A judged merge, prepared to be appended.
struct PreparedMerge {
    preview: MergePreview,
    encoded: Vec<u8>,
    prepared: PreparedDelta,
}

/// The level a requirement is recorded as.
pub(crate) fn required_level(required: RequiredVerification) -> VerificationLevel {
    match required {
        RequiredVerification::FullSemantic => VerificationLevel::FullSemantic,
        RequiredVerification::Deterministic => VerificationLevel::Deterministic,
    }
}

/// The requirement a recorded level stands for; `None` for a level no
/// requirement is recorded as.
pub(crate) fn requirement_of(level: VerificationLevel) -> Option<RequiredVerification> {
    match level {
        VerificationLevel::FullSemantic => Some(RequiredVerification::FullSemantic),
        VerificationLevel::Deterministic => Some(RequiredVerification::Deterministic),
        VerificationLevel::Unverified
        | VerificationLevel::LatentAgreement
        | VerificationLevel::SampleVerified => None,
    }
}

/// A verifier name: an identifier of at most [`MAX_VERIFIER_NAME`] bytes
/// without a `/`.
pub(crate) fn valid_verifier_name(name: &str) -> bool {
    execution::valid_identifier(name) && name.len() <= MAX_VERIFIER_NAME && !name.contains('/')
}

/// A finding code: an identifier of at most [`MAX_FINDING_CODE`] bytes.
pub(crate) fn valid_finding_code(code: &str) -> bool {
    execution::valid_identifier(code) && code.len() <= MAX_FINDING_CODE
}

/// Provenance text a record carries, such as a host write's principal: an
/// identifier of at most [`MAX_PROVENANCE_TEXT`] bytes.
pub(crate) fn valid_provenance(text: &str) -> bool {
    execution::valid_identifier(text) && text.len() <= MAX_PROVENANCE_TEXT
}

/// `VerificationLevel` has no order of its own; this is the one the weakest
/// level is taken by.
fn level_rank(level: VerificationLevel) -> u8 {
    match level {
        VerificationLevel::Unverified => 0,
        VerificationLevel::LatentAgreement => 1,
        VerificationLevel::SampleVerified => 2,
        VerificationLevel::FullSemantic => 3,
        VerificationLevel::Deterministic => 4,
    }
}

/// The order the worst status is taken by.
fn status_rank(status: VerificationStatus) -> u8 {
    match status {
        VerificationStatus::Pass => 0,
        VerificationStatus::Unknown => 1,
        VerificationStatus::Disputed => 2,
        VerificationStatus::Fail => 3,
    }
}

impl PtrRuntime {
    /// Install the host's policy for semantic writes that are not ingress.
    /// Host only, like registering an execution session: nothing a caller of
    /// the runtime's other paths holds can install one.
    ///
    /// # Errors
    /// [`RuntimeError::SemanticGrantInstalled`] if a grant is installed; it
    /// stays, since replacing it would change what later writes are admitted
    /// by without a record of the change. [`RuntimeError::InvalidSemanticGrant`]
    /// for a grant with no verifier, more than [`MAX_SEMANTIC_VERIFIERS`], or
    /// a verifier name that is not a distinct identifier of at most
    /// [`MAX_VERIFIER_NAME`] bytes without a `/`.
    pub fn install_semantic_grant(&mut self, grant: SemanticGrant) -> Result<(), RuntimeError> {
        if self.semantic_grant.is_some() {
            return Err(RuntimeError::SemanticGrantInstalled);
        }
        grant.check()?;
        self.semantic_grant = Some(grant);
        Ok(())
    }

    /// The installed grant's requirement, verifier names, whether it allows
    /// host writes, its merge policy's version and its reviewers; `None`
    /// before one is installed. Never the calibration seed.
    pub fn semantic_grant(&self) -> Option<SemanticGrantInfo> {
        self.semantic_grant.as_ref().map(SemanticGrant::info)
    }

    /// Where `branch` was merged, or `None` if no merge of it is committed.
    pub fn merged_at(&self, branch: &BranchId) -> Option<CommitIndex> {
        self.state
            .values
            .get(&ptr_state::merged_branch_key(&branch.0))
            .and_then(|entry| ptr_state::parse_merged_branch_entry(entry))
            .map(|(index, _)| CommitIndex(index))
    }

    /// Certify `sealed` against the runtime's current state and let the
    /// grant's verifiers judge the plan, without committing anything: what a
    /// reviewer approves and what triage decides on. It returns the verdict
    /// whether or not the verifiers admit the plan, since a person deciding
    /// needs a failing report too.
    ///
    /// [`Self::merge_branch`] never takes a preview's word: it certifies and
    /// judges again.
    ///
    /// # Errors
    /// What [`Self::merge_branch`] refuses in steps 2 and 4 to 9, apart from
    /// a changed plan (step 7), which needs an approval to compare with.
    pub fn preview_merge(&self, sealed: &SealedBranch) -> Result<MergePreview, RuntimeError> {
        let grant = self
            .semantic_grant
            .as_ref()
            .ok_or(RuntimeError::NoSemanticGrant)?;
        let certified = self.certify_merge(sealed)?;
        self.judge_merge(grant, sealed, certified)?
            .merge
            .map(|merge| merge.preview)
    }

    /// Merge an agent branch: certify it against the runtime's own state,
    /// let every verifier of the grant judge the exact state it would
    /// publish, and append it, as one record naming the branch, its author,
    /// the sealed branch, the plan, the verifiers and `authority`, only if
    /// they admit it and the authority allows it. This is the only way a
    /// branch reaches semantic state, and a branch id is merged at most once.
    ///
    /// In order, it refuses:
    /// 1. a fenced runtime ([`RuntimeError::ExecutionFenced`]);
    /// 2. a runtime with no grant ([`RuntimeError::NoSemanticGrant`]);
    /// 3. [`MergeAuthority::Triage`] under a grant with no merge policy
    ///    ([`RuntimeError::NoMergePolicy`]), and [`MergeAuthority::Reviewed`]
    ///    by a reviewer the grant does not list
    ///    ([`RuntimeError::UnknownReviewer`]);
    /// 4. a branch id or author that is not an identifier of at most
    ///    [`MAX_PROVENANCE_TEXT`] bytes
    ///    ([`RuntimeError::InvalidProvenanceText`]);
    /// 5. a branch already merged ([`RuntimeError::BranchAlreadyMerged`]);
    /// 6. a branch that does not certify against the current state, a
    ///    lifecycle generation it relied on that is no longer live included
    ///    ([`RuntimeError::Certification`]; `ptr_branch::certify` checks what
    ///    sealing guarantees again first);
    /// 7. under [`MergeAuthority::Reviewed`], a plan whose digest is not the
    ///    one approved ([`RuntimeError::MergePlanChanged`]);
    /// 8. a plan that writes, removes or derives an ingress key
    ///    ([`RuntimeError::ReservedSemanticNamespace`]);
    /// 9. a plan that does not encode or prepare, one that would evict an
    ///    ingress key a record written before origins existed derived from a
    ///    key it writes ([`RuntimeError::ReservedSemanticNamespace`]), and a
    ///    verifier report that does not fit a record
    ///    ([`RuntimeError::InvalidVerificationReport`]).
    ///
    /// Then it decides:
    /// 10. a change the verifiers do not admit is held
    ///     ([`HoldReason::VerificationRejected`]), whatever the authority;
    /// 11. under [`MergeAuthority::Triage`], one the policy does not
    ///     auto-propose is held ([`HoldReason::Escalated`]) for a person to
    ///     approve by its plan digest;
    /// 12. one that changes nothing is returned as
    ///     [`MergeOutcome::NoChange`], and the branch is not marked merged;
    /// 13. a record too large for the ledger to frame is refused
    ///     ([`RuntimeError::MergeRecordTooLarge`]);
    /// 14. the record is checked by the rules replay applies to it, then
    ///     appended, and its branch is marked merged.
    ///
    /// Each verifier's result is emitted as a [`RuntimeEvent::VerifierResult`]
    /// once the verifiers have judged, a verifier whose report failed closed
    /// as not passed. Every refusal and hold comes before
    /// anything is appended and leaves the runtime unfenced; neither is
    /// recorded. After an append whose outcome is unknown, which fences the
    /// runtime, merging the branch again once the runtime is reopened is
    /// refused if the first append committed and proceeds if it did not.
    ///
    /// # Errors
    /// The refusals above. Ledger append and state-application errors
    /// propagate; an error after append begins leaves execution fenced
    /// because the commit is uncertain.
    pub fn merge_branch(
        &mut self,
        sealed: &SealedBranch,
        authority: MergeAuthority,
    ) -> Result<MergeOutcome, RuntimeError> {
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        let grant = self
            .semantic_grant
            .as_ref()
            .ok_or(RuntimeError::NoSemanticGrant)?;
        let authorized = match authority {
            MergeAuthority::Triage { score } => {
                let (policy, seed) = grant.merge_policy().ok_or(RuntimeError::NoMergePolicy)?;
                Authorized::Triage {
                    policy,
                    seed,
                    score,
                }
            }
            MergeAuthority::Reviewed {
                plan_digest,
                reviewer,
            } => {
                if !grant.lists_reviewer(&reviewer) {
                    return Err(RuntimeError::UnknownReviewer {
                        reviewer: reviewer.0,
                    });
                }
                Authorized::Reviewed {
                    plan_digest,
                    reviewer,
                }
            }
        };
        let certified = self.certify_merge(sealed)?;
        if let Authorized::Reviewed { plan_digest, .. } = &authorized {
            if *plan_digest != certified.plan_digest {
                return Err(RuntimeError::MergePlanChanged {
                    approved: *plan_digest,
                    current: certified.plan_digest,
                });
            }
        }
        let judged = self.judge_merge(grant, sealed, certified)?;
        let (authority, triage) = match authorized {
            Authorized::Triage {
                policy,
                seed,
                score,
            } => (
                MergeAuthorityRecord::Triage {
                    policy_version: policy.version().to_owned(),
                    score_bits: score.get().to_bits(),
                },
                judged.merge.as_ref().ok().map(|merge| {
                    triage_merge(policy, seed, sealed.id(), &merge.preview.verdict, score)
                }),
            ),
            Authorized::Reviewed { reviewer, .. } => (
                MergeAuthorityRecord::Reviewed {
                    reviewer: reviewer.0,
                },
                None,
            ),
        };
        let policy_version = match &authority {
            MergeAuthorityRecord::Triage { policy_version, .. } => Some(policy_version.clone()),
            MergeAuthorityRecord::Reviewed { .. } => None,
        };
        self.emit_verifier_results(&judged.results);
        let PreparedMerge {
            preview,
            encoded,
            prepared,
        } = judged.merge?;
        let triage = triage.transpose()?;
        if !preview.verdict.admitted() {
            return Ok(MergeOutcome::Held(MergeHold {
                preview,
                reason: HoldReason::VerificationRejected,
                triage,
                policy_version,
            }));
        }
        if triage
            .as_ref()
            .is_some_and(|triage| triage.decision() != TriageDecision::AutoPropose)
        {
            return Ok(MergeOutcome::Held(MergeHold {
                preview,
                reason: HoldReason::Escalated,
                triage,
                policy_version,
            }));
        }
        let base = preview.revision;
        let revision = prepared.revision();
        if revision == base {
            return Ok(MergeOutcome::NoChange(preview));
        }
        let plan = preview.certification.plan();
        let origin = SemanticOrigin::Merge(MergeRecord {
            branch: sealed.id().0.clone(),
            author: sealed.author().0.clone(),
            seal: preview.seal_digest,
            plan: preview.plan_digest,
            dependencies: *plan.dependencies(),
            rebased: plan.rebased().clone(),
            verification: preview.verdict.attestation(),
            authority: authority.clone(),
        });
        // What replay will check, checked before the record exists.
        self.validate_semantic_origin(None, base, &encoded, prepared.delta(), &origin)?;
        let event = LedgerEvent::SemanticDeltaCommitted {
            base_revision: base,
            revision,
            encoded_delta: encoded,
            origin,
        };
        // Past the rules replay applies, a record fails to encode only for its
        // size or its number of rebased keys.
        ptr_ledger::check_encodable(&event).map_err(|_| RuntimeError::MergeRecordTooLarge)?;
        let affected = prepared.affected().clone();
        let index = self.append_prepared(event, Some(prepared))?;
        Ok(MergeOutcome::Committed(MergeReceipt {
            branch: sealed.id().clone(),
            seal_digest: preview.seal_digest,
            plan_digest: preview.plan_digest,
            kind: preview.certification.kind(),
            rebased: preview.certification.plan().rebased().clone(),
            commit: SemanticCommit {
                revision,
                affected,
                commit_index: Some(index),
            },
            verdict: preview.verdict,
            authority,
            triage,
            policy_version,
        }))
    }

    /// Steps 4 to 6 of [`Self::merge_branch`], and the plan's digest.
    fn certify_merge(&self, sealed: &SealedBranch) -> Result<CertifiedMerge, RuntimeError> {
        if !valid_provenance(&sealed.id().0) {
            return Err(RuntimeError::InvalidProvenanceText { field: "branch" });
        }
        if !valid_provenance(&sealed.author().0) {
            return Err(RuntimeError::InvalidProvenanceText { field: "author" });
        }
        if let Some(at) = self.merged_at(sealed.id()) {
            return Err(RuntimeError::BranchAlreadyMerged {
                branch: sealed.id().0.clone(),
                at,
            });
        }
        let seal_digest = sealed.seal_digest().map_err(RuntimeError::Certification)?;
        let certification = certify(sealed, &self.semdb.snapshot(), |target, generation| {
            self.generation_validity(target, generation)
        })
        .map_err(RuntimeError::Certification)?;
        let plan_digest = certification
            .plan()
            .digest()
            .map_err(RuntimeError::Certification)?;
        Ok(CertifiedMerge {
            revision: certification.plan().expected(),
            certification,
            seal_digest,
            plan_digest,
        })
    }

    /// Steps 8 and 9 of [`Self::merge_branch`].
    fn judge_merge(
        &self,
        grant: &SemanticGrant,
        sealed: &SealedBranch,
        certified: CertifiedMerge,
    ) -> Result<JudgedMerge, RuntimeError> {
        let plan = certified.certification.plan();
        if let Some(key) = written_ingress_key(plan.delta()) {
            return Err(RuntimeError::ReservedSemanticNamespace {
                key: key.to_owned(),
            });
        }
        let encoded = plan.delta().encode().map_err(RuntimeError::Semantic)?;
        let prepared = self
            .semdb
            .prepare_delta(plan.delta().clone())
            .map_err(RuntimeError::Semantic)?;
        if let Some(key) = evicted_ingress_key(&prepared) {
            return Err(RuntimeError::ReservedSemanticNamespace {
                key: key.to_owned(),
            });
        }
        let judgement = grant.judge(&SemanticChange {
            base: certified.revision,
            next: prepared.revision(),
            before: self
                .semdb
                .base_view(&prepared)
                .map_err(RuntimeError::Semantic)?,
            after: prepared.view(),
            delta: prepared.delta(),
            affected: prepared.affected(),
            origin: ChangeOrigin::Merge {
                branch: sealed,
                plan,
                kind: certified.certification.kind(),
            },
        });
        let affected = prepared.affected().clone();
        Ok(JudgedMerge {
            results: judgement.results,
            merge: judgement.verdict.map(|verdict| PreparedMerge {
                preview: MergePreview {
                    certification: certified.certification,
                    seal_digest: certified.seal_digest,
                    plan_digest: certified.plan_digest,
                    verdict,
                    affected,
                    revision: certified.revision,
                },
                encoded,
                prepared,
            }),
        })
    }

    /// Emit each verifier's result, as the grant's verifiers judged one
    /// change.
    pub(crate) fn emit_verifier_results(&mut self, results: &[(&'static str, bool)]) {
        for (name, passed) in results {
            self.emit(RuntimeEvent::VerifierResult {
                verifier: (*name).to_owned(),
                passed: *passed,
            });
        }
    }
}

/// The merge policy's triage of a judged merge. The policy decides only on a
/// change the verifiers admitted; one they did not is discarded when a
/// verifier failed it and escalated otherwise, as the policy would treat a
/// report verification decided, whatever it would make of the report on its
/// own terms (a pass at a level the policy accepts but the grant does not,
/// say).
fn triage_merge(
    policy: &PolicyRecord,
    seed: u64,
    branch: &BranchId,
    verdict: &SemanticVerdict,
    score: Probability,
) -> Result<TriageOutcome, RuntimeError> {
    let triage = if verdict.admitted() {
        policy
            .policy()
            .triage(&verdict.report(), score, calibration_draw(branch, seed))
    } else {
        TriageOutcome::from_parts(TriageOutcomeParts {
            decision: if verdict.status == VerificationStatus::Fail {
                TriageDecision::Discard
            } else {
                TriageDecision::Escalate
            },
            eligible: false,
            calibration_slice: false,
            score: score.get(),
            auto_propensity: 0.0,
        })
    };
    triage.map_err(|error| RuntimeError::InvalidMergePolicy { code: error.code() })
}
