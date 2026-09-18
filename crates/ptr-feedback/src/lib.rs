use ptr_types::CandidateId;
use ptr_verifier::VerificationReport;

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
            .filter(|c| c.verification.is_some())
            .max_by(|a, b| {
                a.verification
                    .as_ref()
                    .unwrap()
                    .score
                    .get()
                    .total_cmp(&b.verification.as_ref().unwrap().score.get())
            })
    }
}
