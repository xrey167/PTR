//! The checkpoint `ptr-runtime`'s tests bind still loads into this model.
//!
//! `crates/ptr-runtime/tests/fixtures/ptr-a0-v1.ckpt` was written by
//! `examples/write_checkpoint.rs` and is committed, because the two workspaces do
//! not build together. The runtime reads only its header, so nothing on that side
//! notices when this crate's record stops accepting it. The ablation switches
//! added constant fields to `PtrA0`, and constants are recorded as empty, so the
//! record, and with it this fixture, must be unchanged; this test is what shows
//! it.

use burn::prelude::*;
use ptr_burn_a0::{load, PtrA0Config};

#[test]
fn the_committed_runtime_fixture_still_loads() {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../crates/ptr-runtime/tests/fixtures/ptr-a0-v1.ckpt"
    ))
    .expect("the fixture is committed");
    // The configuration examples/write_checkpoint.rs writes it with.
    let config = PtrA0Config::new(8, 4).with_provenance_buckets(4);
    let device = Device::flex();
    let model = load(&bytes, &config, &device).expect("the fixture loads");
    assert_eq!(model.operator_count(), 11);
}
