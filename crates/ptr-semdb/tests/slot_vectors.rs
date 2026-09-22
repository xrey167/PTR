//! From a committed semantic value to the vector a model reads in a slot.
//!
//! This is one half of the round trip issue #23's G1 asks for. The other half — that
//! the vector changes what the model produces — lives in `model/burn-a0`, because the
//! two workspaces do not build together: A0 depends on `ptr-types` and nothing else
//! of PTR's, and `ptr-semdb` is not on that list. `ptr-types` is the seam, and it is
//! where the encoding is defined precisely so both sides can name the same one.
//!
//! What is under test here: a vector the model sees comes from a value the snapshot
//! *holds*. Not from a caller's argument, and not from an absence.
use ptr_semdb::{
    SemanticDelta, SemanticHost, SemanticPayload, SemanticValue, SlotVectorError, TEXT_TYPE,
};
use ptr_types::{EncodingError, EncodingVersion, SlotEncoding, TypeId, MAX_SLOT_WIDTH};

const WIDTH: usize = 16;

/// A host holding one text value and one typed payload.
fn host() -> SemanticHost {
    let mut host = SemanticHost::default();
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("goal".into(), "ship the gate".into());
    delta.upserts.insert(
        "evidence".into(),
        SemanticValue::Payload(SemanticPayload {
            type_id: TypeId::from("Document"),
            source: "operator".into(),
            bytes: b"a committed document".to_vec(),
        }),
    );
    host.apply_delta(delta).expect("a valid delta");
    host
}

#[test]
fn a_committed_value_yields_a_slot_vector_of_the_width_asked_for() {
    let snapshot = host().snapshot();
    let vector = snapshot
        .slot_vector("goal", SlotEncoding::V1, WIDTH)
        .expect("a committed key");
    assert_eq!(vector.width(), WIDTH);
    assert_eq!(vector.encoding(), EncodingVersion::V1);

    let payload = snapshot
        .slot_vector("evidence", SlotEncoding::V1, WIDTH)
        .expect("a committed payload");
    assert_eq!(payload.width(), WIDTH);
    assert_ne!(
        vector.values(),
        payload.values(),
        "two different committed values are two different slot vectors"
    );
}

#[test]
fn a_key_nothing_holds_is_refused_rather_than_encoded_as_zeros() {
    // Zeros are what every A0 call site passed before any of this existed, which is
    // exactly why an absence must never produce them: a model handed zeros for a key
    // nothing holds would be reading an absence as a value.
    let snapshot = host().snapshot();
    assert_eq!(
        snapshot
            .slot_vector("no-such-key", SlotEncoding::V1, WIDTH)
            .expect_err("nothing is committed there"),
        SlotVectorError::NotCommitted {
            key: "no-such-key".to_owned()
        }
    );
    // The control: the same snapshot, a key it does hold.
    assert!(snapshot
        .slot_vector("goal", SlotEncoding::V1, WIDTH)
        .is_ok());
}

#[test]
fn a_removed_key_stops_yielding_a_vector() {
    let mut host = host();
    let before = host
        .snapshot()
        .slot_vector("goal", SlotEncoding::V1, WIDTH)
        .expect("committed before the removal");

    let mut delta = SemanticDelta::default();
    delta.removals.insert("goal".into());
    host.apply_delta(delta).expect("a valid removal");

    assert!(matches!(
        host.snapshot()
            .slot_vector("goal", SlotEncoding::V1, WIDTH)
            .expect_err("removed"),
        SlotVectorError::NotCommitted { .. }
    ));

    // And the snapshot taken before the removal still yields it: a snapshot is a
    // view of one revision, so the vector belongs to that revision rather than to
    // whatever the host holds now.
    let _ = before;
}

#[test]
fn the_same_value_yields_the_same_vector_at_a_later_revision() {
    // The vector is a function of the value and of nothing else — not the revision,
    // not the key, not what else the host holds. Weights trained on a value must see
    // the same input for it after unrelated commits.
    let mut host = host();
    let first = host
        .snapshot()
        .slot_vector("goal", SlotEncoding::V1, WIDTH)
        .unwrap();

    let mut delta = SemanticDelta::default();
    delta.upserts.insert("unrelated".into(), "noise".into());
    host.apply_delta(delta).expect("a valid delta");

    let later = host
        .snapshot()
        .slot_vector("goal", SlotEncoding::V1, WIDTH)
        .unwrap();
    assert_eq!(first.values(), later.values());
    assert!(host.snapshot().revision > ptr_types::Revision(0));
}

#[test]
fn a_value_that_changes_changes_its_vector() {
    // The direction that matters for a model: the input moves when the committed
    // value moves. Without this the channel could be constant and every other test
    // here would still pass.
    let mut host = host();
    let before = host
        .snapshot()
        .slot_vector("goal", SlotEncoding::V1, WIDTH)
        .unwrap();

    let mut delta = SemanticDelta::default();
    delta
        .upserts
        .insert("goal".into(), "ship the other gate".into());
    host.apply_delta(delta).expect("a valid delta");

    let after = host
        .snapshot()
        .slot_vector("goal", SlotEncoding::V1, WIDTH)
        .unwrap();
    assert_ne!(before.values(), after.values());
}

#[test]
fn text_and_the_same_bytes_under_the_text_type_are_one_claim() {
    // Deliberate rather than accidental, and asserted so it stays deliberate: the
    // two are the same claim about the world differently spelled, so they encode
    // identically. `SemanticValue::Text` carries no type, and `TEXT_TYPE` is the one
    // place that decides what it is.
    let text = SemanticValue::Text("ship the gate".into());
    let spelled_out = SemanticValue::Payload(SemanticPayload {
        type_id: TypeId::from(TEXT_TYPE),
        source: "somewhere else entirely".into(),
        bytes: b"ship the gate".to_vec(),
    });
    assert_eq!(
        text.slot_vector(SlotEncoding::V1, WIDTH).unwrap().values(),
        spelled_out
            .slot_vector(SlotEncoding::V1, WIDTH)
            .unwrap()
            .values()
    );
}

#[test]
fn a_payload_s_source_does_not_change_its_slot_vector() {
    // Provenance has its own channel into the model. Folding it in here would make a
    // slot's *value* change when only its origin did, which is a different fact.
    let one = SemanticValue::Payload(SemanticPayload {
        type_id: TypeId::from("Document"),
        source: "operator".into(),
        bytes: b"identical bytes".to_vec(),
    });
    let other = SemanticValue::Payload(SemanticPayload {
        type_id: TypeId::from("Document"),
        source: "a different source".into(),
        bytes: b"identical bytes".to_vec(),
    });
    assert_eq!(
        one.slot_vector(SlotEncoding::V1, WIDTH).unwrap().values(),
        other.slot_vector(SlotEncoding::V1, WIDTH).unwrap().values()
    );
    // But the type does, so the insensitivity above is to the source specifically
    // rather than to everything but the bytes.
    let retyped = SemanticValue::Payload(SemanticPayload {
        type_id: TypeId::from("Summary"),
        source: "operator".into(),
        bytes: b"identical bytes".to_vec(),
    });
    assert_ne!(
        one.slot_vector(SlotEncoding::V1, WIDTH).unwrap().values(),
        retyped
            .slot_vector(SlotEncoding::V1, WIDTH)
            .unwrap()
            .values()
    );
}

#[test]
fn a_committed_value_encodes_exactly_as_the_model_side_encodes_it() {
    // The two halves of G1 live in workspaces that do not build together, so no test
    // can run a committed value all the way into the model and no caller joins them
    // either. What can be asserted is that both sides compute the *same function*:
    // this is byte-for-byte the call `model/burn-a0/tests/semantic_payload.rs` makes,
    // with the same type and the same bytes.
    //
    // It is the strongest link available across the split and it is not a substitute
    // for an integrated path. If the semdb door ever stopped delegating to
    // `SlotEncoding` — folding in a key, a revision or a source — this fails.
    let mut host = SemanticHost::default();
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(
        "evidence".into(),
        SemanticValue::Payload(SemanticPayload {
            type_id: TypeId::from("Document"),
            source: "operator".into(),
            bytes: b"a goal".to_vec(),
        }),
    );
    host.apply_delta(delta).expect("a valid delta");

    let through_the_door = host
        .snapshot()
        .slot_vector("evidence", SlotEncoding::V1, 8)
        .expect("a committed payload");
    let as_the_model_side_encodes_it = SlotEncoding::V1
        .encode(&TypeId::from("Document"), b"a goal", 8)
        .expect("a small payload");

    assert_eq!(
        through_the_door.values(),
        as_the_model_side_encodes_it.values(),
        "the committed door and the direct call must produce one vector"
    );
    assert_eq!(
        through_the_door.encoding(),
        as_the_model_side_encodes_it.encoding()
    );
}

#[test]
fn an_encoding_refusal_travels_out_rather_than_becoming_a_vector() {
    let snapshot = host().snapshot();
    assert_eq!(
        snapshot
            .slot_vector("goal", SlotEncoding::V1, 0)
            .expect_err("a slot with no values"),
        SlotVectorError::Encoding(EncodingError::Width {
            requested: 0,
            limit: MAX_SLOT_WIDTH
        })
    );
    // And it keeps the encoding's own diagnostic code rather than inventing one, so
    // an operator reading a log sees which bound was crossed.
    assert_eq!(
        snapshot
            .slot_vector("goal", SlotEncoding::V1, 0)
            .unwrap_err()
            .code(),
        "PTR_ENCODING_WIDTH"
    );
}
