use ptr_types::{CandidateId, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus};

/// A value together with the verifier's report on it, if it has been verified.
#[derive(Clone, Debug)]
pub struct Candidate<T> {
    pub id: CandidateId,
    pub value: T,
    pub verification: Option<VerificationReport>,
}

impl<T> Candidate<T> {
    /// Whether the verifier fabric passed this candidate at `required` depth or deeper. The rule
    /// mirrors `TriagePolicy::triage` in `ptr-branch`: the candidate must carry a report, the
    /// report's status must be `Pass`, no finding may be hard (a `Pass` that carries a hard
    /// finding contradicts itself and is refused, whatever its score), and the report's level
    /// must be at least `required`. A score never decides.
    pub fn passes(&self, required: VerificationLevel) -> bool {
        self.verification
            .as_ref()
            .is_some_and(|report| report_passes(report, required))
    }

    /// The critique of this candidate's report, or `None` when it has not been verified.
    pub fn critique(&self) -> Option<Critique> {
        self.verification
            .as_ref()
            .map(|report| Critique::from_report(&self.id, report))
    }
}

fn report_passes(report: &VerificationReport, required: VerificationLevel) -> bool {
    if report.status != VerificationStatus::Pass {
        return false;
    }
    if report.findings.iter().any(|finding| finding.hard) {
        return false;
    }
    level_rank(report.level) >= level_rank(required)
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

/// What a verifier report says is wrong with a candidate: the codes of its hard findings, and
/// whether the report is one a repair may act on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Critique {
    pub candidate: CandidateId,
    pub issues: Vec<String>,
    pub repairable: bool,
}

impl Critique {
    /// The critique of `report`. `issues` are the codes of its hard findings, each once, in
    /// report order. The critique is repairable only when the report is a `Fail` that names at
    /// least one hard finding: an `Unknown` or `Disputed` report names nothing to repair, a
    /// `Fail` without a hard finding names nothing either, and a `Pass` that carries a hard
    /// finding contradicts itself, so none of its findings is trusted as a repair target. Such a
    /// candidate is rewritten, not repaired.
    pub fn from_report(candidate: &CandidateId, report: &VerificationReport) -> Self {
        let mut issues: Vec<String> = Vec::new();
        for finding in report.findings.iter().filter(|finding| finding.hard) {
            if !issues.contains(&finding.code) {
                issues.push(finding.code.clone());
            }
        }
        let repairable = report.status == VerificationStatus::Fail && !issues.is_empty();
        Self {
            candidate: candidate.clone(),
            issues,
            repairable,
        }
    }
}

/// A set of candidates for one task.
#[derive(Clone, Debug)]
pub struct Population<T> {
    pub candidates: Vec<Candidate<T>>,
}

// Not derived: a derive would require `T: Default`, which an empty population has no use for.
impl<T> Default for Population<T> {
    fn default() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }
}

impl<T> Population<T> {
    pub fn push(&mut self, candidate: Candidate<T>) {
        self.candidates.push(candidate);
    }

    /// The highest-scoring candidate that [`Candidate::passes`] at `required`, if any. A failed
    /// candidate cannot win however high its score, and neither can a `Pass` that carries a hard
    /// finding or one below `required`.
    pub fn best_verified(&self, required: VerificationLevel) -> Option<&Candidate<T>> {
        self.candidates
            .iter()
            .filter(|candidate| candidate.passes(required))
            .max_by(|a, b| {
                let left = a
                    .verification
                    .as_ref()
                    .expect("a passing candidate has a report");
                let right = b
                    .verification
                    .as_ref()
                    .expect("a passing candidate has a report");
                left.score.get().total_cmp(&right.score.get())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::Probability;
    use ptr_verifier::Finding;

    fn finding(code: &str, hard: bool) -> Finding {
        Finding {
            code: code.into(),
            message: String::new(),
            hard,
        }
    }

    fn report(
        status: VerificationStatus,
        level: VerificationLevel,
        findings: Vec<Finding>,
    ) -> VerificationReport {
        VerificationReport {
            status,
            level,
            score: Probability::new(1.0).unwrap(),
            findings,
        }
    }

    fn candidate(report: Option<VerificationReport>) -> Candidate<u8> {
        Candidate {
            id: CandidateId::from("c"),
            value: 0,
            verification: report,
        }
    }

    #[test]
    fn only_a_clean_pass_at_the_required_level_passes() {
        use VerificationLevel as L;
        use VerificationStatus as S;
        let pass = |level| candidate(Some(report(S::Pass, level, vec![])));
        assert!(pass(L::Deterministic).passes(L::Deterministic));
        assert!(pass(L::Deterministic).passes(L::LatentAgreement));
        assert!(!pass(L::FullSemantic).passes(L::Deterministic));
        assert!(!pass(L::LatentAgreement).passes(L::Deterministic));
        assert!(pass(L::Unverified).passes(L::Unverified));
        for status in [S::Fail, S::Disputed, S::Unknown] {
            let c = candidate(Some(report(status, L::Deterministic, vec![])));
            assert!(!c.passes(L::Unverified), "{status:?}");
        }
        assert!(!candidate(None).passes(L::Unverified));
    }

    #[test]
    fn a_pass_with_a_hard_finding_is_refused_and_a_soft_finding_is_not() {
        use VerificationLevel::Deterministic;
        let forged = candidate(Some(report(
            VerificationStatus::Pass,
            Deterministic,
            vec![finding("target", true)],
        )));
        assert!(!forged.passes(VerificationLevel::Unverified));
        let soft = candidate(Some(report(
            VerificationStatus::Pass,
            Deterministic,
            vec![finding("style", false)],
        )));
        assert!(soft.passes(Deterministic));
    }

    #[test]
    fn a_population_can_be_built_for_a_type_without_a_default() {
        struct NoDefault;
        let mut population: Population<NoDefault> = Population::default();
        assert!(population.candidates.is_empty());
        population.push(Candidate {
            id: CandidateId::from("c"),
            value: NoDefault,
            verification: None,
        });
        assert_eq!(population.candidates.len(), 1);
    }

    #[test]
    fn the_level_scale_is_ordered_weakest_to_deepest() {
        use VerificationLevel as L;
        let order = [
            L::Unverified,
            L::LatentAgreement,
            L::SampleVerified,
            L::FullSemantic,
            L::Deterministic,
        ];
        for (i, held) in order.iter().enumerate() {
            let c = candidate(Some(report(VerificationStatus::Pass, *held, vec![])));
            for (j, required) in order.iter().enumerate() {
                assert_eq!(c.passes(*required), i >= j, "{held:?} vs {required:?}");
            }
        }
    }

    #[test]
    fn a_critique_names_each_hard_code_once_and_only_a_fail_is_repairable() {
        use VerificationStatus as S;
        let id = CandidateId::from("c");
        let findings = vec![
            finding("target", true),
            finding("style", false),
            finding("target", true),
            finding("effect", true),
        ];
        let fail = Critique::from_report(
            &id,
            &report(S::Fail, VerificationLevel::Deterministic, findings.clone()),
        );
        assert_eq!(fail.issues, ["target", "effect"]);
        assert!(fail.repairable);
        for status in [S::Unknown, S::Disputed, S::Pass] {
            let c = Critique::from_report(
                &id,
                &report(status, VerificationLevel::Deterministic, findings.clone()),
            );
            assert!(!c.repairable, "{status:?} names nothing to repair");
        }
        let soft_fail = Critique::from_report(
            &id,
            &report(
                S::Fail,
                VerificationLevel::Deterministic,
                vec![finding("style", false)],
            ),
        );
        assert!(soft_fail.issues.is_empty() && !soft_fail.repairable);
        assert!(candidate(None).critique().is_none());
    }
}
