use ptr_types::Probability;

pub fn normalize(raw: &[f32]) -> Vec<Probability> {
    let sum: f32 = raw.iter().copied().filter(|x| x.is_finite() && *x > 0.0).sum();
    if sum <= 0.0 { return vec![Probability::default(); raw.len()]; }
    raw.iter().map(|x| Probability::new((x.max(0.0) / sum).clamp(0.0, 1.0)).unwrap()).collect()
}
