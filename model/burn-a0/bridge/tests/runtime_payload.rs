//! A committed payload reaches what A0 produces, through the runtime.
//!
//! The two halves of this round trip were tested apart before: committed value →
//! slot vector in `crates/ptr-semdb/tests/slot_vectors.rs`, and slot vector →
//! router logits in `model/burn-a0/tests/semantic_payload.rs`. Nothing ran them
//! together, because nothing could. Here the payload is committed to a runtime, the
//! bridge reads it through the runtime's snapshot and the admission through its
//! lifecycle state, and the observable is the router's logits, compared bitwise.
//!
//! Every positive assertion has its negative beside it, on the same observable:
//! a changed payload in an admitted slot moves the logits, an identical one does
//! not; the same change in an excluded slot moves nothing, and that slot is shown
//! to be responsive by admitting it.
mod common;

use burn::prelude::*;
use common::{forward, logit_bits, model, row, slot, world, D_MODEL};
use ptr_a0_bridge::{
    admission_from, forward_from_runtime, slot_values_from, BridgeError, Lifecycle,
};
use ptr_types::{Generation, SemanticRole, SlotEncoding, TypeId};

#[test]
fn a_changed_payload_in_an_admitted_slot_changes_the_output_and_an_identical_one_does_not() {
    let device = Device::flex();
    let model = model(&device);

    let alpha = forward(&world(b"alpha", b"x", true), &model, &device);
    let omega = forward(&world(b"omega", b"x", true), &model, &device);
    let alpha_again = forward(&world(b"alpha", b"x", true), &model, &device);

    // The slot under test is admitted, so this compares what it was meant to.
    assert!(alpha.masks[0].admits(1), "the evidence slot is live");

    assert_ne!(
        logit_bits(&alpha),
        logit_bits(&omega),
        "a different committed payload in an admitted slot must reach the logits"
    );
    // The other direction, from a separately built runtime: without it the
    // assertion above would also hold for output that wandered between calls.
    assert_eq!(
        logit_bits(&alpha),
        logit_bits(&alpha_again),
        "identical committed payloads must produce bit-identical logits"
    );
}

#[test]
fn the_same_difference_in_an_excluded_slot_changes_the_output_by_exactly_nothing() {
    let device = Device::flex();
    let model = model(&device);

    // `withdrawn` sits under a revoked generation: the runtime excludes it.
    let alpha = forward(&world(b"x", b"alpha", true), &model, &device);
    let omega = forward(&world(b"x", b"omega", true), &model, &device);
    assert!(!alpha.masks[0].admits(2), "the withdrawn slot is revoked");
    assert_eq!(
        logit_bits(&alpha),
        logit_bits(&omega),
        "a payload in an excluded slot must not reach the logits at all"
    );

    // The control that makes the line above evidence: the very same slot, with the
    // very same difference, *does* move the logits once its generation is live. So
    // the observable responds to this slot, and only the mask kept it out.
    let live_alpha = forward(&world(b"x", b"alpha", false), &model, &device);
    let live_omega = forward(&world(b"x", b"omega", false), &model, &device);
    assert!(live_alpha.masks[0].admits(2), "unrevoked, the slot is live");
    assert_ne!(
        logit_bits(&live_alpha),
        logit_bits(&live_omega),
        "admitted, the same slot's payload must reach the logits"
    );
}

#[test]
fn a_payload_survives_the_trip_to_slot_values_bit_for_bit() {
    let device = Device::flex();
    let runtime = world(b"the evidence bytes", b"withdrawn bytes", true);
    let snapshot = runtime.snapshot();

    let values = slot_values_from(
        &snapshot,
        &[&["goal", "evidence", "withdrawn"]],
        SlotEncoding::V1,
        D_MODEL,
        &device,
    )
    .expect("every key is committed");
    assert_eq!(values.dims(), [1, 3, D_MODEL]);
    assert_eq!(values.encoding(), SlotEncoding::V1.version());

    // Decode: read the tensor back and compare with the kernel's definition applied
    // directly to the committed type and bytes — not with the snapshot's own
    // encoder, which is the code under test. A text value encodes under the type
    // `Text`; a payload under its own type, with its source not participating.
    let decoded: Vec<u32> = values
        .values()
        .into_data()
        .try_into_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect();
    let mut expected = Vec::new();
    for (type_id, bytes) in [
        ("Text", &b"ship the gate"[..]),
        ("Document", &b"the evidence bytes"[..]),
        ("Document", &b"withdrawn bytes"[..]),
    ] {
        let vector = SlotEncoding::V1
            .encode(&TypeId::from(type_id), bytes, D_MODEL)
            .unwrap();
        expected.extend(vector.values().iter().map(|value| value.to_bits()));
    }
    assert_eq!(
        decoded, expected,
        "what the model receives is exactly the committed payload's encoding"
    );
}

#[test]
fn a_key_nothing_committed_is_refused_rather_than_fed_as_zeros() {
    let device = Device::flex();
    let model = model(&device);
    let runtime = world(b"alpha", b"x", true);
    let row = [
        slot("goal", "capsule:a", SemanticRole::Goal),
        slot("never committed", "capsule:a", SemanticRole::Claim),
    ];
    let error = forward_from_runtime(
        &runtime,
        &model,
        burn::tensor::Tensor::<2, burn::tensor::Int>::from_data([[1, 2]], &device),
        &[&row],
        &device,
    )
    .err()
    .expect("an uncommitted key has no payload");
    assert!(
        matches!(
            error,
            BridgeError::SlotVector {
                row: 0,
                slot: 1,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(error.code(), "PTR_SEMDB_NOT_COMMITTED");
}

#[test]
fn only_a_live_generation_is_admitted() {
    let mut runtime = world(b"alpha", b"x", true);
    runtime
        .commit(ptr_ledger::LedgerEvent::CapsuleSuperseded {
            capsule: "capsule:a".into(),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    let mask = admission_from(
        &runtime,
        &[
            Lifecycle {
                target: "capsule:a",
                generation: Generation(2),
            },
            Lifecycle {
                target: "capsule:a",
                generation: Generation(1),
            },
            Lifecycle {
                target: "capsule:b",
                generation: Generation(1),
            },
            Lifecycle {
                target: "capsule:a",
                generation: Generation(9),
            },
            Lifecycle {
                target: "capsule:unknown",
                generation: Generation(1),
            },
        ],
    );
    assert_eq!(
        (0..5).map(|slot| mask.admits(slot)).collect::<Vec<_>>(),
        [true, false, false, false, false],
        "live admits; superseded, revoked, ahead-of-authority and unknown do not"
    );
}

#[test]
fn the_read_set_names_the_admitted_slots_and_nothing_else() {
    let device = Device::flex();
    let model = model(&device);
    let runtime = world(b"alpha", b"x", true);
    let forward = forward(&runtime, &model, &device);
    assert_eq!(
        forward.read.keys().collect::<Vec<_>>(),
        ["evidence", "goal"],
        "the excluded slot's key reached nothing, so it is not something the model read"
    );
    assert_eq!(forward.read.targets().collect::<Vec<_>>(), ["capsule:a"]);
    assert_eq!(forward.read.position(), runtime.journal_anchor().unwrap());
    assert_eq!(forward.revision, runtime.revision());
    assert_eq!(row().len(), forward.masks[0].len());
}
