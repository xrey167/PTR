use ptr_types::Probability;

#[derive(Clone, Debug, PartialEq)]
pub struct CoreVerification {
    pub confidence: Probability,
    pub contradiction_score: Probability,
    pub needs_external_verification: bool,
}
