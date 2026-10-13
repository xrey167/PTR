//! Evidence promotion check.
//!
//! A claim asks to be promoted to [`PromotionTarget::Verified`] or
//! [`PromotionTarget::Authoritative`]. [`PromotionVerifier`] decides, deterministically and from
//! the claim alone, whether the evidence it cites is allowed to carry that promotion:
//!
//! * evidence must be cited at all (`promotion.no_evidence`);
//! * evidence derived only from search, an index or a learned model never counts: derived data
//!   cannot self-promote to truth (invariant 5), so a claim whose evidence is all derived is
//!   `promotion.derived_only`;
//! * evidence without a commit index is not authoritative state (invariant 4):
//!   `promotion.uncommitted`, and it does not count either;
//! * the counted evidence (committed and not derived) must come from at least
//!   [`PromotionPolicy::min_independent_sources`] distinct sources
//!   (`promotion.insufficient_sources`); the `source` label is the caller's identity for the
//!   upstream origin and the caller must canonicalise it, because two labels for one origin
//!   count as two sources;
//! * promotion to `Authoritative` needs at least one counted piece of evidence from the ledger,
//!   the authoritative history (`promotion.no_ledger_evidence`);
//! * a deterministic finding that contradicts the claim (a hard [`Finding`] the caller attaches)
//!   forces `Fail` (`promotion.contradicted`) whatever else holds: a learned positive score has
//!   no field here and cannot override it (invariant 11).
//!
//! An ill-formed claim (`promotion.malformed`) or more evidence than the policy allows
//! (`promotion.oversize`) is refused before anything is judged. A claim that satisfies every
//! rule is `Pass` at `Deterministic` level with score 1; every violated rule is one hard
//! finding and the report is `Fail` with score 0. See [`crate::codes`].

use std::collections::BTreeSet;
use std::fmt;

use ptr_types::{CommitIndex, Probability, VerificationLevel};

use crate::codes::finding;
use crate::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

/// Hard upper bound of [`PromotionPolicy::max_evidence`].
pub const MAX_EVIDENCE_LIMIT: usize = 256;
/// Longest claim text, evidence id or source label, in bytes.
pub const MAX_FIELD_BYTES: usize = 256;
/// Most hard-finding codes named in one `promotion.contradicted` message.
const MAX_NAMED_CONTRADICTIONS: usize = 8;
/// Most evidence ids named in one `promotion.uncommitted` message.
const MAX_NAMED_EVIDENCE: usize = 8;

/// Where a piece of evidence comes from.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvidenceOrigin {
    /// A committed ledger event.
    Ledger,
    /// Preserved raw input (invariant 1).
    Raw,
    /// The verdict of a deterministic verifier.
    Deterministic,
    /// A search hit. Derived: never evidence by itself.
    Search,
    /// An index or cache entry. Derived: never evidence by itself.
    Index,
    /// A learned model's output. Derived: never evidence by itself.
    Learned,
}

impl EvidenceOrigin {
    /// Search, index and learned data are derived and cannot promote a claim (invariant 5).
    pub const fn is_derived(self) -> bool {
        match self {
            Self::Ledger | Self::Raw | Self::Deterministic => false,
            Self::Search | Self::Index | Self::Learned => true,
        }
    }
}

/// One cited piece of evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Evidence {
    pub id: String,
    pub origin: EvidenceOrigin,
    /// The independent source behind the evidence (a document, a feed, an observer). Distinct
    /// labels count as distinct sources.
    pub source: String,
    /// The ledger commit that made the evidence authoritative; `None` if it is not committed.
    pub commit: Option<CommitIndex>,
}

/// The state a claim asks to be promoted to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromotionTarget {
    Verified,
    Authoritative,
}

impl PromotionTarget {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Authoritative => "authoritative",
        }
    }
}

/// A request to promote one claim.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionClaim {
    /// The claim's text or identifier; only used to refuse an empty or oversize claim.
    pub claim: String,
    pub target: PromotionTarget,
    pub evidence: Vec<Evidence>,
    /// Findings of deterministic verifiers about this claim. Any hard finding contradicts it.
    pub deterministic_findings: Vec<Finding>,
}

/// Why a [`PromotionPolicy`] is not usable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromotionPolicyError {
    /// A minimum of zero sources would let a claim without evidence be promoted.
    ZeroMinSources,
    ZeroMaxEvidence,
    /// `max_evidence` is above [`MAX_EVIDENCE_LIMIT`].
    MaxEvidenceAboveLimit,
    /// The minimum number of sources exceeds the evidence the policy accepts, so no claim could
    /// pass.
    MinSourcesAboveMaxEvidence,
}

impl fmt::Display for PromotionPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ZeroMinSources => "minimum independent sources must be at least 1",
            Self::ZeroMaxEvidence => "maximum evidence must be at least 1",
            Self::MaxEvidenceAboveLimit => "maximum evidence is above the hard limit",
            Self::MinSourcesAboveMaxEvidence => "minimum sources exceed the evidence bound",
        })
    }
}

impl std::error::Error for PromotionPolicyError {}

/// How much independent evidence a promotion needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromotionPolicy {
    min_independent_sources: usize,
    max_evidence: usize,
}

impl Default for PromotionPolicy {
    /// One independent source, at most 64 pieces of evidence.
    fn default() -> Self {
        Self {
            min_independent_sources: 1,
            max_evidence: 64,
        }
    }
}

impl PromotionPolicy {
    /// A policy needing `min_independent_sources` distinct sources, with the default evidence
    /// bound (raised to the minimum if that is larger).
    pub fn new(min_independent_sources: usize) -> Result<Self, PromotionPolicyError> {
        if min_independent_sources == 0 {
            return Err(PromotionPolicyError::ZeroMinSources);
        }
        let default = Self::default();
        Self {
            min_independent_sources,
            max_evidence: default.max_evidence.max(min_independent_sources),
        }
        .validated()
    }

    /// The same policy accepting at most `max_evidence` pieces of evidence per claim.
    pub fn with_max_evidence(self, max_evidence: usize) -> Result<Self, PromotionPolicyError> {
        Self {
            max_evidence,
            ..self
        }
        .validated()
    }

    fn validated(self) -> Result<Self, PromotionPolicyError> {
        if self.min_independent_sources == 0 {
            Err(PromotionPolicyError::ZeroMinSources)
        } else if self.max_evidence == 0 {
            Err(PromotionPolicyError::ZeroMaxEvidence)
        } else if self.max_evidence > MAX_EVIDENCE_LIMIT {
            Err(PromotionPolicyError::MaxEvidenceAboveLimit)
        } else if self.min_independent_sources > self.max_evidence {
            Err(PromotionPolicyError::MinSourcesAboveMaxEvidence)
        } else {
            Ok(self)
        }
    }

    pub fn min_independent_sources(&self) -> usize {
        self.min_independent_sources
    }

    pub fn max_evidence(&self) -> usize {
        self.max_evidence
    }
}

/// Decides whether a [`PromotionClaim`]'s evidence can carry its promotion. See the module
/// documentation.
#[derive(Clone, Copy, Debug, Default)]
pub struct PromotionVerifier {
    policy: PromotionPolicy,
}

impl PromotionVerifier {
    pub fn new(policy: PromotionPolicy) -> Self {
        Self { policy }
    }

    pub fn policy(&self) -> PromotionPolicy {
        self.policy
    }
}

fn well_formed(field: &str) -> bool {
    !field.trim().is_empty()
        && field.len() <= MAX_FIELD_BYTES
        && !field.chars().any(char::is_control)
}

fn report(findings: Vec<Finding>) -> VerificationReport {
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

impl Verifier<PromotionClaim> for PromotionVerifier {
    fn verify(&self, claim: &PromotionClaim) -> VerificationReport {
        let policy = self.policy;
        let target = claim.target.as_str();
        if claim.evidence.len() > policy.max_evidence {
            return report(vec![finding(
                "promotion.oversize",
                format!(
                    "{} pieces of evidence cited, the policy accepts at most {}",
                    claim.evidence.len(),
                    policy.max_evidence
                ),
            )]);
        }
        let mut malformed = Vec::new();
        if !well_formed(&claim.claim) {
            malformed.push(finding(
                "promotion.malformed",
                format!(
                    "claim text must be 1 to {MAX_FIELD_BYTES} bytes without control characters"
                ),
            ));
        }
        for (index, item) in claim.evidence.iter().enumerate() {
            if !well_formed(&item.id) || !well_formed(&item.source) {
                malformed.push(finding(
                    "promotion.malformed",
                    format!(
                        "evidence {index}: id and source must be 1 to {MAX_FIELD_BYTES} bytes \
                         without control characters"
                    ),
                ));
            }
        }
        if !malformed.is_empty() {
            return report(malformed);
        }

        let mut findings = Vec::new();
        let contradictions: Vec<&str> = claim
            .deterministic_findings
            .iter()
            .filter(|f| f.hard)
            .map(|f| f.code.as_str())
            .collect();
        if !contradictions.is_empty() {
            let named: Vec<&str> = contradictions
                .iter()
                .copied()
                .take(MAX_NAMED_CONTRADICTIONS)
                .collect();
            findings.push(finding(
                "promotion.contradicted",
                format!(
                    "{} deterministic finding(s) contradict promotion to {target}: {}",
                    contradictions.len(),
                    named.join(", ")
                ),
            ));
        }
        if claim.evidence.is_empty() {
            findings.push(finding(
                "promotion.no_evidence",
                format!("promotion to {target} cites no evidence"),
            ));
            return report(findings);
        }

        let primary: Vec<&Evidence> = claim
            .evidence
            .iter()
            .filter(|item| !item.origin.is_derived())
            .collect();
        if primary.is_empty() {
            findings.push(finding(
                "promotion.derived_only",
                format!(
                    "all {} piece(s) of evidence are derived (search, index or learned); \
                     derived data cannot promote a claim to {target}",
                    claim.evidence.len()
                ),
            ));
            return report(findings);
        }

        let uncommitted: Vec<&str> = primary
            .iter()
            .filter(|item| item.commit.is_none())
            .map(|item| item.id.as_str())
            .collect();
        if !uncommitted.is_empty() {
            let named: Vec<&str> = uncommitted
                .iter()
                .copied()
                .take(MAX_NAMED_EVIDENCE)
                .collect();
            findings.push(finding(
                "promotion.uncommitted",
                format!(
                    "{} piece(s) of evidence are not committed and do not count: {}",
                    uncommitted.len(),
                    named.join(", ")
                ),
            ));
        }
        let counted: Vec<&Evidence> = primary
            .iter()
            .copied()
            .filter(|item| item.commit.is_some())
            .collect();
        let sources: BTreeSet<&str> = counted.iter().map(|item| item.source.as_str()).collect();
        if sources.len() < policy.min_independent_sources {
            findings.push(finding(
                "promotion.insufficient_sources",
                format!(
                    "{} independent source(s) among committed evidence, the policy needs {}",
                    sources.len(),
                    policy.min_independent_sources
                ),
            ));
        }
        if claim.target == PromotionTarget::Authoritative
            && !counted
                .iter()
                .any(|item| item.origin == EvidenceOrigin::Ledger)
        {
            findings.push(finding(
                "promotion.no_ledger_evidence",
                "promotion to authoritative needs committed evidence from the ledger".into(),
            ));
        }
        report(findings)
    }
}

impl NamedVerifier<PromotionClaim> for PromotionVerifier {
    fn name(&self) -> &'static str {
        "promotion"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(id: &str, origin: EvidenceOrigin, source: &str, committed: bool) -> Evidence {
        Evidence {
            id: id.into(),
            origin,
            source: source.into(),
            commit: committed.then_some(CommitIndex(7)),
        }
    }

    fn claim(target: PromotionTarget, evidence: Vec<Evidence>) -> PromotionClaim {
        PromotionClaim {
            claim: "paris is the capital of france".into(),
            target,
            evidence,
            deterministic_findings: Vec::new(),
        }
    }

    fn codes(report: &VerificationReport) -> Vec<&str> {
        report.findings.iter().map(|f| f.code.as_str()).collect()
    }

    fn verifier(min: usize) -> PromotionVerifier {
        PromotionVerifier::new(PromotionPolicy::new(min).unwrap())
    }

    #[test]
    fn committed_primary_evidence_passes() {
        let c = claim(
            PromotionTarget::Authoritative,
            vec![evidence("e1", EvidenceOrigin::Ledger, "ledger", true)],
        );
        let report = verifier(1).verify(&c);
        assert_eq!(report.status, VerificationStatus::Pass);
        assert_eq!(report.level, VerificationLevel::Deterministic);
        assert_eq!(report.score.get(), 1.0);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn a_claim_without_evidence_is_refused() {
        let report = verifier(1).verify(&claim(PromotionTarget::Verified, vec![]));
        assert_eq!(report.status, VerificationStatus::Fail);
        assert_eq!(codes(&report), ["promotion.no_evidence"]);
        assert!(report.findings[0].hard);
        assert_eq!(report.score.get(), 0.0);
    }

    #[test]
    fn evidence_derived_only_from_search_index_or_learned_sources_is_rejected() {
        for origin in [
            EvidenceOrigin::Search,
            EvidenceOrigin::Index,
            EvidenceOrigin::Learned,
        ] {
            assert!(origin.is_derived());
            let c = claim(
                PromotionTarget::Verified,
                vec![
                    evidence("a", origin, "s1", true),
                    evidence("b", origin, "s2", true),
                ],
            );
            let report = verifier(1).verify(&c);
            assert_eq!(report.status, VerificationStatus::Fail, "{origin:?}");
            assert_eq!(codes(&report), ["promotion.derived_only"], "{origin:?}");
        }
        for origin in [
            EvidenceOrigin::Ledger,
            EvidenceOrigin::Raw,
            EvidenceOrigin::Deterministic,
        ] {
            assert!(!origin.is_derived());
        }
    }

    #[test]
    fn derived_evidence_never_counts_next_to_primary_evidence() {
        let c = claim(
            PromotionTarget::Verified,
            vec![
                evidence("a", EvidenceOrigin::Raw, "doc", true),
                evidence("b", EvidenceOrigin::Search, "engine", true),
            ],
        );
        assert_eq!(verifier(1).verify(&c).status, VerificationStatus::Pass);
        let report = verifier(2).verify(&c);
        assert_eq!(codes(&report), ["promotion.insufficient_sources"]);
        assert!(report.findings[0].message.contains("1 independent source"));
    }

    #[test]
    fn uncommitted_evidence_is_rejected_and_does_not_count() {
        let c = claim(
            PromotionTarget::Verified,
            vec![evidence("draft", EvidenceOrigin::Raw, "doc", false)],
        );
        let report = verifier(1).verify(&c);
        assert_eq!(
            codes(&report),
            ["promotion.uncommitted", "promotion.insufficient_sources"]
        );
        assert!(report.findings[0].message.contains("draft"));
        // One committed piece next to an uncommitted one: the uncommitted one is still named.
        let c = claim(
            PromotionTarget::Verified,
            vec![
                evidence("draft", EvidenceOrigin::Raw, "doc", false),
                evidence("ok", EvidenceOrigin::Raw, "doc2", true),
            ],
        );
        let report = verifier(1).verify(&c);
        assert_eq!(report.status, VerificationStatus::Fail);
        assert_eq!(codes(&report), ["promotion.uncommitted"]);
    }

    #[test]
    fn the_minimum_independent_source_count_is_configurable() {
        let one_source_twice = claim(
            PromotionTarget::Verified,
            vec![
                evidence("a", EvidenceOrigin::Raw, "doc", true),
                evidence("b", EvidenceOrigin::Deterministic, "doc", true),
            ],
        );
        assert_eq!(
            verifier(1).verify(&one_source_twice).status,
            VerificationStatus::Pass
        );
        let report = verifier(2).verify(&one_source_twice);
        assert_eq!(codes(&report), ["promotion.insufficient_sources"]);
        assert!(report.findings[0].message.contains("needs 2"));
        let two_sources = claim(
            PromotionTarget::Verified,
            vec![
                evidence("a", EvidenceOrigin::Raw, "doc", true),
                evidence("b", EvidenceOrigin::Deterministic, "other", true),
            ],
        );
        assert_eq!(
            verifier(2).verify(&two_sources).status,
            VerificationStatus::Pass
        );
    }

    #[test]
    fn authoritative_promotion_needs_ledger_evidence_verified_does_not() {
        let c = |target| {
            claim(
                target,
                vec![evidence(
                    "a",
                    EvidenceOrigin::Deterministic,
                    "verifier",
                    true,
                )],
            )
        };
        assert_eq!(
            verifier(1).verify(&c(PromotionTarget::Verified)).status,
            VerificationStatus::Pass
        );
        let report = verifier(1).verify(&c(PromotionTarget::Authoritative));
        assert_eq!(codes(&report), ["promotion.no_ledger_evidence"]);
    }

    #[test]
    fn a_hard_contradicting_finding_forces_fail_whatever_else_holds() {
        let mut c = claim(
            PromotionTarget::Authoritative,
            vec![evidence("e1", EvidenceOrigin::Ledger, "ledger", true)],
        );
        c.deterministic_findings = vec![
            Finding {
                code: "fact.contradicted".into(),
                message: "snapshot holds another value".into(),
                hard: true,
            },
            Finding {
                code: "fact.unknown".into(),
                message: "soft".into(),
                hard: false,
            },
        ];
        let report = verifier(1).verify(&c);
        assert_eq!(report.status, VerificationStatus::Fail);
        assert_eq!(report.score.get(), 0.0);
        assert_eq!(codes(&report), ["promotion.contradicted"]);
        assert!(report.findings[0].message.contains("fact.contradicted"));
        assert!(!report.findings[0].message.contains("fact.unknown"));
        // Soft findings alone do not contradict.
        c.deterministic_findings.remove(0);
        assert_eq!(verifier(1).verify(&c).status, VerificationStatus::Pass);
    }

    #[test]
    fn malformed_and_oversize_claims_are_refused_before_judging() {
        let good = evidence("e1", EvidenceOrigin::Ledger, "ledger", true);
        let mut c = claim(PromotionTarget::Verified, vec![good.clone()]);
        c.claim = "  ".into();
        assert_eq!(codes(&verifier(1).verify(&c)), ["promotion.malformed"]);
        c.claim = "x".repeat(MAX_FIELD_BYTES + 1);
        assert_eq!(codes(&verifier(1).verify(&c)), ["promotion.malformed"]);
        for bad in [
            Evidence {
                id: String::new(),
                ..good.clone()
            },
            Evidence {
                source: String::new(),
                ..good.clone()
            },
            Evidence {
                id: "a\nb".into(),
                ..good.clone()
            },
            Evidence {
                source: "s".repeat(MAX_FIELD_BYTES + 1),
                ..good.clone()
            },
        ] {
            let report = verifier(1).verify(&claim(PromotionTarget::Verified, vec![bad]));
            assert_eq!(report.status, VerificationStatus::Fail);
            assert_eq!(codes(&report), ["promotion.malformed"]);
        }
        let policy = PromotionPolicy::new(1)
            .unwrap()
            .with_max_evidence(2)
            .unwrap();
        let many = claim(
            PromotionTarget::Verified,
            vec![good.clone(), good.clone(), good],
        );
        let report = PromotionVerifier::new(policy).verify(&many);
        assert_eq!(codes(&report), ["promotion.oversize"]);
        assert!(report.findings.iter().all(|f| f.hard));
    }

    #[test]
    fn policy_construction_has_typed_errors() {
        assert_eq!(
            PromotionPolicy::new(0),
            Err(PromotionPolicyError::ZeroMinSources)
        );
        let policy = PromotionPolicy::new(3).unwrap();
        assert_eq!(policy.min_independent_sources(), 3);
        assert_eq!(policy.max_evidence(), 64);
        assert_eq!(
            policy.with_max_evidence(0),
            Err(PromotionPolicyError::ZeroMaxEvidence)
        );
        assert_eq!(
            policy.with_max_evidence(MAX_EVIDENCE_LIMIT + 1),
            Err(PromotionPolicyError::MaxEvidenceAboveLimit)
        );
        assert_eq!(
            policy.with_max_evidence(2),
            Err(PromotionPolicyError::MinSourcesAboveMaxEvidence)
        );
        assert_eq!(
            PromotionPolicy::new(MAX_EVIDENCE_LIMIT)
                .unwrap()
                .max_evidence(),
            MAX_EVIDENCE_LIMIT
        );
        assert_eq!(
            PromotionPolicy::new(MAX_EVIDENCE_LIMIT + 1),
            Err(PromotionPolicyError::MaxEvidenceAboveLimit)
        );
        assert_eq!(PromotionPolicy::default().min_independent_sources(), 1);
        for error in [
            PromotionPolicyError::ZeroMinSources,
            PromotionPolicyError::ZeroMaxEvidence,
            PromotionPolicyError::MaxEvidenceAboveLimit,
            PromotionPolicyError::MinSourcesAboveMaxEvidence,
        ] {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn promotion_verifier_is_named_and_deterministic() {
        let v = PromotionVerifier::default();
        assert_eq!(NamedVerifier::<PromotionClaim>::name(&v), "promotion");
        assert_eq!(v.policy(), PromotionPolicy::default());
        let c = claim(PromotionTarget::Verified, vec![]);
        assert_eq!(v.verify(&c), v.verify(&c));
    }
}
