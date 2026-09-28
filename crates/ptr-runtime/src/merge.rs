//! The host policy every semantic write that is not ingress is admitted
//! under, and what its verifiers judge.
//!
//! A semantic record reaches the ledger in this build in one of two ways.
//! Ingress writes the raw text of a request ([`PtrRuntime::ingest_text`]) or
//! a Pod's verified output, in shapes the runtime fixes and replay checks
//! again. A host write ([`PtrRuntime::apply_verified_semantic_delta`]) is
//! admitted only by every verifier of the [`SemanticGrant`] the host
//! installed, and its record names the principal, the verifiers, the level
//! they reached and their soft findings.
//!
//! A grant is installed once and is not history: nothing of it is journaled
//! but what a record it admits names. Replay re-checks everything that is
//! deterministic about a record and never runs a verifier again.
//!
//! [`PtrRuntime::ingest_text`]: crate::PtrRuntime::ingest_text
//! [`PtrRuntime::apply_verified_semantic_delta`]: crate::PtrRuntime::apply_verified_semantic_delta
use std::collections::BTreeSet;
use std::fmt;

use ptr_ledger::Attestation;
use ptr_semdb::{PreparedView, SemanticDelta};
use ptr_types::{PrincipalId, Probability, Revision, VerificationLevel};
use ptr_verifier::{NamedVerifier, VerificationReport, VerificationStatus};

use crate::execution::{self, RequiredVerification};
use crate::{PtrRuntime, RuntimeError};

/// Most verifiers a grant installs, and so most a record names.
pub const MAX_SEMANTIC_VERIFIERS: usize = ptr_ledger::MAX_ATTESTATION_VERIFIERS;
/// Longest verifier name, in bytes.
pub const MAX_VERIFIER_NAME: usize = 64;
/// Longest finding code a verifier may report, in bytes.
pub const MAX_FINDING_CODE: usize = 64;
/// Longest principal a host write may name, in bytes.
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
/// [`SemanticGrant::allow_host_writes`] turns them on.
pub struct SemanticGrant {
    required: RequiredVerification,
    verifiers: Vec<Box<dyn for<'a> NamedVerifier<SemanticChange<'a>>>>,
    host_writes: bool,
}

impl SemanticGrant {
    /// A grant requiring `required` of every verifier, with no verifier yet
    /// and host writes off. It cannot be installed until it has a verifier.
    pub fn new(required: RequiredVerification) -> Self {
        Self {
            required,
            verifiers: Vec::new(),
            host_writes: false,
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

    pub(crate) fn info(&self) -> SemanticGrantInfo {
        SemanticGrantInfo {
            required: self.required,
            verifiers: self.names().map(str::to_owned).collect(),
            host_writes: self.host_writes,
        }
    }

    pub(crate) fn host_writes(&self) -> bool {
        self.host_writes
    }

    fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.verifiers.iter().map(|verifier| verifier.name())
    }

    /// Whether the grant can be installed: 1 to [`MAX_SEMANTIC_VERIFIERS`]
    /// verifiers whose names are distinct identifiers of at most
    /// [`MAX_VERIFIER_NAME`] bytes without a `/`, which separates a name from
    /// a finding code in a record.
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

impl fmt::Debug for SemanticGrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemanticGrant")
            .field("required", &self.required)
            .field("verifiers", &self.names().collect::<Vec<_>>())
            .field("host_writes", &self.host_writes)
            .finish()
    }
}

/// What a host may read back about the installed grant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticGrantInfo {
    pub required: RequiredVerification,
    /// The verifiers' names, in grant order.
    pub verifiers: Vec<String>,
    pub host_writes: bool,
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

    /// The installed grant's requirement, verifier names and whether it
    /// allows host writes; `None` before one is installed.
    pub fn semantic_grant(&self) -> Option<SemanticGrantInfo> {
        self.semantic_grant.as_ref().map(SemanticGrant::info)
    }
}
