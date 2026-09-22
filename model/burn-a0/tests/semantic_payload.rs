//! A committed payload reaches what the model produces.
//!
//! This is the second half of the round trip issue #23's G1 asks for. The first half
//! — a committed semantic value becoming a slot vector — is in
//! `crates/ptr-semdb/tests/slot_vectors.rs`. The two cannot be one test: this
//! workspace depends on `ptr-types` and nothing else of PTR's, and the two workspaces
//! do not build together. `ptr-types` is the seam, which is why the encoding lives
//! there and why both sides can name the same one.
//!
//! What was wrong before this existed: `forward`'s `slot_values` argument was used —
//! `slots = slot_values + typed_metadata` — and every call site in the repository
//! passed `Tensor::<3>::zeros`. The channel was wired in and never carried a value,
//! so nothing the model produced could depend on what a slot *was*, only on its role,
//! its epistemic state, its provenance bucket and its confidence.
//!
//! The tests below assert both directions, because one alone proves nothing:
//! different payloads must reach the output, and identical payloads must not change
//! it. The first fails if the model ignores slot values; the second fails if the test
//! is measuring nondeterminism rather than the payload.
use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{
    admission_bias, CodeGrid, PtrA0, PtrA0Config, PtrSlotMetadata, SlotValueError, SlotValues,
};
use ptr_types::{
    Codebook, EncodingVersion, EpistemicState, SemanticRole, SlotEncoding, TypeId, Validity,
    ValidityMask,
};

const SLOTS: usize = 2;
const D_MODEL: usize = 8;

fn model(device: &Device) -> PtrA0 {
    device.seed(17);
    PtrA0Config::new(32, D_MODEL)
        .with_provenance_buckets(8)
        .with_latent_steps(1)
        .init(device)
}

fn metadata(device: &Device) -> PtrSlotMetadata {
    PtrSlotMetadata {
        epistemic: CodeGrid::new(
            &Codebook::V1,
            &[&[EpistemicState::Observed, EpistemicState::Assumed][..]],
            device,
        )
        .expect("every epistemic state is assigned in v1"),
        provenance_ids: Tensor::<2, Int>::zeros([1, SLOTS], device),
        confidence: Tensor::<2>::ones([1, SLOTS], device),
        admission: admission_bias(
            &[ValidityMask::from_validities(&[Validity::Live; SLOTS])],
            device,
        ),
    }
}

fn slot_types(device: &Device) -> CodeGrid<SemanticRole> {
    CodeGrid::new(
        &Codebook::V1,
        &[&[SemanticRole::Goal, SemanticRole::Claim][..]],
        device,
    )
    .expect("every role is assigned in v1")
}

fn encode(payload: &[u8]) -> ptr_types::SlotVector {
    SlotEncoding::V1
        .encode(&TypeId::from("Document"), payload, D_MODEL)
        .expect("a small payload at this width")
}

fn values(first: &[u8], second: &[u8], device: &Device) -> SlotValues {
    SlotValues::new(&[&[encode(first), encode(second)]], device).expect("one row of two slots")
}

/// The router's logits for one payload pair, with every other input fixed.
fn router(model: &PtrA0, first: &[u8], second: &[u8], device: &Device) -> Vec<f32> {
    model
        .forward(
            Tensor::<2, Int>::from_data([[1, 2]], device),
            &slot_types(device),
            &values(first, second, device),
            metadata(device),
        )
        .router_logits
        .into_data()
        .try_into_vec::<f32>()
        .expect("logits are f32")
}

#[test]
fn a_payload_reaches_the_router_and_an_unchanged_one_does_not_move_it() {
    let device = Device::flex();
    let model = model(&device);

    let baseline = router(&model, b"a goal", b"a claim", &device);

    // Same everything: the tokens, the roles, the epistemic states, the provenance,
    // the confidence, the admission, the weights and the second slot. Only the first
    // slot's committed payload differs.
    let changed = router(&model, b"a different goal", b"a claim", &device);
    assert_ne!(
        baseline, changed,
        "a committed payload must reach what the model produces"
    );

    // The other direction. Without this the assertion above would also hold for a
    // model whose output wandered between calls.
    let repeated = router(&model, b"a goal", b"a claim", &device);
    assert_eq!(
        baseline, repeated,
        "the same payloads must produce the same logits"
    );
}

#[test]
fn a_payload_in_any_slot_reaches_the_router() {
    // Not only the first. A construction that read one slot's value and dropped the
    // rest would pass the test above.
    let device = Device::flex();
    let model = model(&device);
    let baseline = router(&model, b"a goal", b"a claim", &device);
    assert_ne!(
        baseline,
        router(&model, b"a goal", b"a different claim", &device),
        "the second slot's payload must reach the router too"
    );
}

#[test]
fn a_one_byte_difference_is_enough() {
    // The encoding commits to identity rather than similarity, so two payloads that
    // differ by one byte are as different as any others. That property is worth
    // asserting here as well as in `ptr-types`: it is the reason a model can tell two
    // near-identical claims apart at all.
    let device = Device::flex();
    let model = model(&device);
    assert_ne!(
        router(&model, b"claim", b"a claim", &device),
        router(&model, b"claiM", b"a claim", &device)
    );
}

#[test]
fn constant_slot_values_cannot_distinguish_anything_which_is_what_zeros_were() {
    // The state of the world before this existed, stated as a test rather than as a
    // claim in a comment. When every slot carries the same value — as it did when
    // every call site passed zeros — the model's output is the same whatever the
    // payloads *would* have been, because the payloads are not in it.
    //
    // So this passes now and would also have passed before, which is the point: it
    // marks the baseline the test above is an improvement on.
    let device = Device::flex();
    let model = model(&device);
    let constant = router(&model, b"same", b"same", &device);
    let also_constant = router(&model, b"same", b"same", &device);
    assert_eq!(constant, also_constant);
    // And a different constant is a different input, so the channel is not inert —
    // it is that *identical* values carry no distinction, not that values are ignored.
    assert_ne!(constant, router(&model, b"other", b"other", &device));
}

#[test]
fn a_model_and_its_inputs_name_the_same_encoding() {
    // What this test can reach from outside: that the version a model records is the
    // one its inputs carry. The *guard* — `forward` refusing a mismatch — cannot be
    // exercised from here, because a `SlotValues` from another definition is not
    // constructible outside the crate. That is the point, and it is why the guard's
    // own test lives beside it in `src/lib.rs` where the field is reachable, exactly
    // as the codebook guards' tests do.
    let device = Device::flex();
    assert_eq!(model(&device).encoding(), SlotEncoding::V1);
    assert_eq!(SlotEncoding::V1.version(), EncodingVersion::V1);
    assert_eq!(
        values(b"a goal", b"a claim", &device).encoding(),
        EncodingVersion::V1
    );
    assert_eq!(SlotEncoding::at(EncodingVersion(2)), None, "one definition");
}

#[test]
fn a_ragged_or_mixed_batch_is_refused_rather_than_padded() {
    let device = Device::flex();
    let long = [encode(b"a"), encode(b"b")];
    let short = [encode(b"c")];
    assert_eq!(
        SlotValues::new(&[&long, &short], &device).expect_err("rows disagree"),
        SlotValueError::Ragged {
            row: 1,
            expected: 2,
            found: 1
        }
    );
    assert_eq!(
        SlotValues::new(&[], &device).expect_err("no rows"),
        SlotValueError::Empty
    );
    assert_eq!(
        SlotValues::new(&[&[]], &device).expect_err("no slots"),
        SlotValueError::Empty
    );

    // Widths are part of it: a batch mixing two widths cannot be one tensor.
    let narrow = SlotEncoding::V1
        .encode(&TypeId::from("Document"), b"a", D_MODEL - 1)
        .unwrap();
    assert_eq!(
        SlotValues::new(&[&[encode(b"a"), narrow]], &device).expect_err("mixed widths"),
        SlotValueError::MixedWidths {
            expected: D_MODEL,
            found: D_MODEL - 1
        }
    );

    // The control: a rectangular batch of one width is accepted, so the refusals
    // above are about the shape rather than about the constructor rejecting
    // everything.
    assert_eq!(
        SlotValues::new(&[&long, &long], &device).unwrap().dims(),
        [2, 2, D_MODEL]
    );
}

// A width that is not `d_model` is refused by name, which cannot be asserted from
// out here: the tensor addition that follows panics on a mismatched last dimension by
// itself, so a test that only checked *that* it panics would pass with the guard
// deleted. It is `slot_values_at_another_width_are_refused_by_name` in `src/lib.rs`,
// where `should_panic` can name the message.
