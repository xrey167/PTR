use ptr_types::{Probability, VerificationLevel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerificationStatus { Pass, Fail, Disputed, Unknown }

#[derive(Clone, Debug, PartialEq)]
pub struct Finding { pub code: String, pub message: String, pub hard: bool }

#[derive(Clone, Debug, PartialEq)]
pub struct VerificationReport {
    pub status: VerificationStatus,
    pub level: VerificationLevel,
    pub score: Probability,
    pub findings: Vec<Finding>,
}

pub trait Verifier<T> { fn verify(&self, candidate: &T) -> VerificationReport; }
