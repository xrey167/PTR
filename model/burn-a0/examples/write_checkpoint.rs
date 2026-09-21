//! Write the checkpoint fixture the production runtime's tests bind.
//!
//! The point of a committed fixture is that `ptr-runtime` is tested against an
//! artifact this model actually produced, rather than against bytes its own test
//! assembled: the two workspaces do not build together, so nothing else would
//! catch them disagreeing about the format.
//!
//! Regenerate with, from the repository root:
//!
//! ```text
//! cargo +stable run --manifest-path model/burn-a0/Cargo.toml \
//!     --example write_checkpoint --locked -- crates/ptr-runtime/tests/fixtures/ptr-a0-v1.ckpt
//! ```
//!
//! The weights are a fixed seed's initialisation and carry no training. They are
//! deliberately tiny — the consumer never reads them.
use burn::prelude::*;
use ptr_burn_a0::{save, PtrA0Config};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: write_checkpoint <path>");
    let device = Device::flex();
    device.seed(1);
    let model = PtrA0Config::new(8, 4)
        .with_provenance_buckets(4)
        .init(&device);
    let bytes = save(&model).expect("a fresh model serializes");
    std::fs::write(&path, &bytes).expect("the fixture is writable");
    println!("{} bytes -> {path}", bytes.len());
}
