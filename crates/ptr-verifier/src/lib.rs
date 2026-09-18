use ptr_types::{Probability, VerificationLevel};

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

#[derive(Default)]
pub struct VerifierFabric<T> {
    verifiers: Vec<Box<dyn NamedVerifier<T>>>,
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

    pub fn verify(&self, candidate: &T) -> VerificationReport {
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

        let status = reports
            .iter()
            .map(|report| report.status)
            .max_by_key(|status| status_rank(*status))
            .expect("non-empty verifier reports");

        let level = reports
            .iter()
            .map(|report| report.level)
            .max_by_key(|level| level_rank(*level))
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
