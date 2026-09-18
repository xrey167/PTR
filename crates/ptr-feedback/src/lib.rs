use ptr_types::CandidateId;
use ptr_verifier::{VerificationReport, VerificationStatus};

#[derive(Clone, Debug)]
pub struct Candidate<T> {
    pub id: CandidateId,
    pub value: T,
    pub verification: Option<VerificationReport>,
}

#[derive(Clone, Debug)]
pub struct Critique {
    pub candidate: CandidateId,
    pub issues: Vec<String>,
    pub repairable: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Population<T> {
    pub candidates: Vec<Candidate<T>>,
}

impl<T> Population<T> {
    pub fn push(&mut self, candidate: Candidate<T>) {
        self.candidates.push(candidate);
    }

    pub fn best_verified(&self) -> Option<&Candidate<T>> {
        self.candidates
            .iter()
            .filter(|candidate| {
                candidate
                    .verification
                    .as_ref()
                    .is_some_and(|report| report.status == VerificationStatus::Pass)
            })
            .max_by(|a, b| {
                let left = a.verification.as_ref().expect("filtered verified candidate");
                let right = b.verification.as_ref().expect("filtered verified candidate");
                left.score.get().total_cmp(&right.score.get())
            })
    }
}
