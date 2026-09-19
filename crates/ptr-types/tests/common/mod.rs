use ptr_types::{ConfidenceEstimate, ConfidenceTarget, Probability};

pub fn estimate(target: ConfidenceTarget, probability: f32) -> ConfidenceEstimate {
    ConfidenceEstimate::new(
        target,
        Probability::new(probability).expect("bounded test fixture"),
    )
}
