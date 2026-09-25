//! The default forward pass, pinned before the ablation switches were added.
//!
//! The values below were produced at commit c9a13f7 (after `init` began drawing
//! every parameter in declaration order, before any of the `typed_query`,
//! `latent_nonlinearity` or `frozen_router` switches existed) for one fixed seed
//! and input with `latent_steps = 2`. Every switch must default to that behaviour,
//! so a model built without touching a switch must still produce them. Identical
//! bits were produced by rustc 1.95.0 and 1.98.1 on the study host
//! (`hardware/a0-cpu-4core.toml`).
//!
//! The comparison allows 1e-5, not zero: the gemm kernel dispatches on CPU
//! features (AVX-512 on the study host), and CI runners may take another path
//! whose last bits differ. A switch that leaked into the default path changes
//! these logits by far more than that. Bit-for-bit reproduction on the recorded
//! host is the study's own determinism gate (G2), not this test's job.
//!
//! This is the only test in the file: tests in one binary run on parallel
//! threads, and another test initializing a model would draw from the same
//! global generator between this test's seed and its init.

use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{admission_bias, CodeGrid, PtrA0Config, PtrSlotMetadata, SlotValues};
use ptr_types::{
    Codebook, EpistemicState, SemanticRole, SlotEncoding, TypeId, Validity, ValidityMask,
};

/// Router logits, [2, 11] row-major, as f32 bit patterns.
const GOLDEN_LOGITS: [u32; 22] = [
    0x3fad58fc, 0xbed1fe39, 0xbe10dc94, 0x401b6c2e, 0x3f2f18cc, 0x3f4a4ba0, 0xc0028bde, 0x3f010069,
    0xc0072b94, 0xbe8ea72e, 0xbf734d28, 0x3ec7d453, 0xbdcc5638, 0x3e748ef5, 0x3e71cfb1, 0xbf254b04,
    0x3d5b5aa3, 0xbe3a8149, 0xbd15351b, 0x3f0dc2df, 0x3e28f54d, 0xbeabf92b,
];
/// Sum of every element of the returned slot states, and of the raw states.
const GOLDEN_SLOT_SUM: u32 = 0x426d01e3;
const GOLDEN_RAW_SUM: u32 = 0x41e3dfd1;
const TOLERANCE: f32 = 1.0e-5;

#[test]
fn the_default_forward_pass_is_unchanged() {
    let device = Device::flex();
    device.seed(20260925);
    let model = PtrA0Config::new(64, 16)
        .with_provenance_buckets(8)
        .with_latent_steps(2)
        .init(&device);

    let roles: [&[SemanticRole]; 2] = [
        &[
            SemanticRole::Goal,
            SemanticRole::Evidence,
            SemanticRole::Claim,
        ],
        &[
            SemanticRole::Action,
            SemanticRole::Constraint,
            SemanticRole::Relation,
        ],
    ];
    let states: [&[EpistemicState]; 2] = [
        &[
            EpistemicState::Observed,
            EpistemicState::Hypothesis,
            EpistemicState::Verified,
        ],
        &[
            EpistemicState::Unknown,
            EpistemicState::Inferred,
            EpistemicState::Assumed,
        ],
    ];
    let vectors: Vec<Vec<_>> = (0..2)
        .map(|row| {
            (0..3)
                .map(|slot| {
                    SlotEncoding::V1
                        .encode(
                            &TypeId::from("Entity"),
                            format!("entity-{row}-{slot}").as_bytes(),
                            16,
                        )
                        .expect("a small payload")
                })
                .collect()
        })
        .collect();
    let rows: Vec<&[_]> = vectors.iter().map(Vec::as_slice).collect();
    let values = SlotValues::new(&rows, &device).expect("a rectangular batch");
    let metadata = PtrSlotMetadata {
        epistemic: CodeGrid::new(&Codebook::V1, &states, &device).expect("v1 states"),
        provenance_ids: Tensor::<2, Int>::from_data([[0, 1, 2], [3, 4, 5]], &device),
        confidence: Tensor::<2>::from_data([[0.95, 0.35, 0.6], [0.15, 0.8, 0.5]], &device),
        admission: admission_bias(
            &[
                ValidityMask::from_validities(&[Validity::Live, Validity::Revoked, Validity::Live]),
                ValidityMask::from_validities(&[Validity::Live; 3]),
            ],
            &device,
        ),
    };
    let tokens = Tensor::<2, Int>::from_data([[3, 17, 42, 5, 9], [60, 1, 1, 33, 8]], &device);
    let roles = CodeGrid::new(&Codebook::V1, &roles, &device).expect("v1 roles");

    let output = model.forward(tokens, &roles, &values, metadata);

    let logits: Vec<f32> = output
        .router_logits
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32 logits");
    assert_eq!(logits.len(), GOLDEN_LOGITS.len());
    for (index, (got, bits)) in logits.iter().zip(GOLDEN_LOGITS).enumerate() {
        let want = f32::from_bits(bits);
        assert!(
            (got - want).abs() <= TOLERANCE,
            "logit {index}: got {got} ({:#010x}), pinned {want} ({bits:#010x})",
            got.to_bits()
        );
    }
    for (name, got, bits) in [
        ("slot", output.slots.sum().into_scalar(), GOLDEN_SLOT_SUM),
        ("raw", output.raw.sum().into_scalar(), GOLDEN_RAW_SUM),
    ] {
        let got: f32 = got;
        let want = f32::from_bits(bits);
        // A sum over many elements carries their rounding, so its bound is relative.
        assert!(
            (got - want).abs() <= TOLERANCE * want.abs().max(1.0),
            "{name} sum: got {got}, pinned {want}"
        );
    }
}
