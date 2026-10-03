use burn::{module::Module, prelude::Device};
use ptr_burn_a0::{PlainTransformerConfig, PtrA0Config};

const D_MODEL: usize = 48;
const OPERATORS: usize = 11;
const SEQUENCE: u64 = 72;

fn matched_flops() -> u64 {
    let slots = 6_u64;
    // Four raw-side projections, four slot-side projections, two latent
    // refinement passes, four SxL attention products, and the router.
    4 * SEQUENCE * 2 * D_MODEL as u64 * D_MODEL as u64
        + 6 * slots * 2 * D_MODEL as u64 * D_MODEL as u64
        + 4 * 2 * SEQUENCE * slots * D_MODEL as u64
        + 2 * slots * D_MODEL as u64 * OPERATORS as u64
}

#[test]
fn plain_baseline_matches_a0_parameter_budget() {
    let device = Device::flex();
    let plain = PlainTransformerConfig::new(216, D_MODEL, OPERATORS).init(&device);
    let a0 = PtrA0Config::new(216, D_MODEL)
        .with_provenance_buckets(8)
        .with_latent_steps(2)
        .init(&device);
    let plain_params = plain.num_params();
    let a0_params = a0.num_params();
    let relative = (plain_params as f64 / a0_params as f64 - 1.0).abs();
    assert!(
        relative <= 0.01,
        "plain parameters {plain_params} vs A0 {a0_params} differ by {relative:.4}"
    );
    assert_eq!(plain_params, 33_323);
    assert_eq!(a0_params, 33_324);
}

#[test]
fn plain_baseline_matches_a0_compute_budget() {
    assert_eq!(matched_flops(), 1_665_216);
    let a0_flops = 1_665_792_u64;
    let relative = (matched_flops() as f64 / a0_flops as f64 - 1.0).abs();
    assert!(relative <= 0.01, "FLOPs differ by {relative:.6}");
}
