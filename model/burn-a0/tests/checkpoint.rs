//! A checkpoint is weights plus the assignment they were trained under.
//!
//! The positive test compares outputs bit for bit, with a control showing that a
//! freshly initialised model does *not* match: without that control, a load that
//! silently did nothing would pass.

use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{
    admission_bias, header, load, save, CheckpointIoError, CodeGrid, PtrA0, PtrA0Config,
    PtrSlotMetadata, SlotValues, EMBEDDED_FAMILIES, MODEL, PROVENANCE_BUCKET_COUNT,
    PROVENANCE_EXCEPTION,
};
use ptr_types::{
    CheckpointError, CheckpointHeader, CodeFamily, Codebook, CodebookVersion, EncodingVersion,
    EpistemicState, SemanticRole, SlotEncoding, TableSize, TypeId, Validity, ValidityMask,
};

const D_MODEL: usize = 12;
const VOCAB: usize = 16;

fn config() -> PtrA0Config {
    PtrA0Config::new(VOCAB, D_MODEL)
        .with_provenance_buckets(8)
        .with_latent_steps(1)
}

fn metadata(device: &Device) -> PtrSlotMetadata {
    PtrSlotMetadata {
        epistemic: CodeGrid::new(
            &Codebook::V1,
            &[&[EpistemicState::Observed, EpistemicState::Assumed][..]],
            device,
        )
        .expect("assigned in v1"),
        provenance_ids: Tensor::<2, Int>::zeros([1, 2], device),
        confidence: Tensor::<2>::ones([1, 2], device),
        admission: admission_bias(
            &[ValidityMask::from_validities(&[Validity::Live; 2])],
            device,
        ),
    }
}

/// Two committed payloads, fixed so the logits below are comparable across models.
fn slot_values(device: &Device) -> SlotValues {
    let encoding = SlotEncoding::V1;
    let goal = encoding
        .encode(&TypeId::from("Text"), b"a goal", D_MODEL)
        .expect("a small payload");
    let evidence = encoding
        .encode(&TypeId::from("Document"), b"some evidence", D_MODEL)
        .expect("a small payload");
    SlotValues::new(&[&[goal, evidence]], device).expect("one row of slots")
}

/// Router logits for a fixed input, as a comparable vector.
fn logits(model: &PtrA0, device: &Device) -> Vec<f32> {
    let slot_types = CodeGrid::new(
        &Codebook::V1,
        &[&[SemanticRole::Goal, SemanticRole::Evidence][..]],
        device,
    )
    .expect("assigned in v1");
    model
        .forward(
            Tensor::<2, Int>::from_data([[1, 2]], device),
            &slot_types,
            &slot_values(device),
            metadata(device),
        )
        .router_logits
        .into_data()
        .try_into_vec::<f32>()
        .unwrap()
}

/// The parameters of a model, as a checkpoint's payload would hold them.
fn payload(model: &PtrA0) -> Vec<u8> {
    model
        .clone()
        .into_record()
        .into_bytes()
        .expect("a record serializes")
        .to_vec()
}

#[test]
fn a_checkpoint_restores_the_exact_model_it_was_taken_from() {
    let device = Device::flex();
    device.seed(5);
    let trained = config().init(&device);
    let expected = logits(&trained, &device);

    let bytes = save(&trained).expect("a model serializes");

    // The control: a differently seeded model of the same shape must not already
    // agree, or the comparison below would prove nothing.
    device.seed(6);
    let fresh = config().init(&device);
    assert_ne!(
        logits(&fresh, &device),
        expected,
        "the control model must differ before loading"
    );

    let loaded = load(&bytes, &config(), &device).expect("its own checkpoint loads");
    assert_eq!(
        logits(&loaded, &device),
        expected,
        "a loaded checkpoint must reproduce the model exactly"
    );
}

#[test]
fn the_header_records_the_tables_the_weights_actually_have() {
    let device = Device::flex();
    let model = config().init(&device);
    let written = header(&model);

    assert_eq!(written.model, MODEL);
    assert_eq!(written.codebook, CodebookVersion::V1);
    assert_eq!(written.codebook_bytes, Codebook::V1.canonical_bytes());
    assert_eq!(
        written.table(CodeFamily::SemanticRole),
        Some(model.slot_type_rows() as u16)
    );
    assert_eq!(
        written.table(CodeFamily::EpistemicState),
        Some(model.epistemic_rows() as u16)
    );
    assert_eq!(
        written.table(CodeFamily::ReasoningOperator),
        Some(model.operator_count() as u16)
    );
    written
        .verify(&Codebook::V1, SlotEncoding::V1, &EMBEDDED_FAMILIES)
        .expect("this build wrote it");
}

#[test]
fn a_checkpoint_whose_assignment_moved_is_refused_before_any_weight_is_read() {
    let device = Device::flex();
    let model = config().init(&device);

    let mut moved = header(&model);
    moved.codebook_bytes[0] ^= 0x20;
    let bytes = moved.write(&payload(&model));

    assert_eq!(
        load(&bytes, &config(), &device).expect_err("a moved assignment is refused"),
        CheckpointIoError::Header(CheckpointError::CodebookMoved {
            version: CodebookVersion::V1
        })
    );
    // The same payload under the intact header still loads, so the refusal is
    // about the assignment and not about the weights.
    load(&header(&model).write(&payload(&model)), &config(), &device)
        .expect("the intact header loads");
}

#[test]
fn a_table_that_disagrees_with_the_codebook_is_refused() {
    // What the old configurable widths produced: weights built for eight slot
    // types while the kernel defines nine. Recorded honestly, it is now refused.
    let device = Device::flex();
    let model = config().init(&device);
    let book = Codebook::V1;
    let claimed = CheckpointHeader::new(
        MODEL,
        &book,
        SlotEncoding::V1,
        &[
            TableSize {
                family: CodeFamily::SemanticRole,
                rows: 8,
            },
            TableSize {
                family: CodeFamily::EpistemicState,
                rows: 6,
            },
            TableSize {
                family: CodeFamily::ReasoningOperator,
                rows: 11,
            },
        ],
    );
    assert_eq!(
        load(&claimed.write(&payload(&model)), &config(), &device).expect_err("eight is not nine"),
        CheckpointIoError::Header(CheckpointError::TableSize {
            family: CodeFamily::SemanticRole,
            stored: 8,
            required: 9,
        })
    );
}

#[test]
fn a_checkpoint_from_another_slot_encoding_is_refused_before_any_weight_is_read() {
    // The one identity in the header that no tensor could ever reveal. A codebook
    // that moved eventually shows up as a table width; a slot encoding that moved
    // shows up as nothing at all, because it produces the model's *input* and no
    // parameters. So this refusal has to come from the recorded version or from
    // nowhere.
    let device = Device::flex();
    let model = config().init(&device);
    let record = save(&model).expect("a fresh model serializes");

    let (mut moved, payload) = CheckpointHeader::read(&record).expect("we wrote it");
    assert_eq!(moved.encoding, EncodingVersion::V1, "written as v1");
    moved.encoding = EncodingVersion(2);
    let rewritten = moved.write(payload);

    assert_eq!(
        load(&rewritten, &config(), &device).expect_err("another encoding"),
        CheckpointIoError::Header(CheckpointError::EncodingMoved {
            stored: EncodingVersion(2),
            required: EncodingVersion::V1,
        })
    );

    // The control: the same bytes with the encoding left alone load. So the refusal
    // is about the encoding and not about the round trip through read and write.
    let untouched = CheckpointHeader::read(&record)
        .map(|(header, payload)| header.write(payload))
        .expect("we wrote it");
    assert!(load(&untouched, &config(), &device).is_ok());
}

#[test]
fn a_checkpoint_missing_a_required_family_is_refused() {
    let device = Device::flex();
    let model = config().init(&device);
    let partial = CheckpointHeader::new(
        MODEL,
        &Codebook::V1,
        SlotEncoding::V1,
        &[TableSize {
            family: CodeFamily::SemanticRole,
            rows: 9,
        }],
    );
    assert_eq!(
        load(&partial.write(&payload(&model)), &config(), &device)
            .expect_err("the router's family is not recorded"),
        CheckpointIoError::Header(CheckpointError::MissingTable {
            family: CodeFamily::EpistemicState
        })
    );
}

#[test]
fn another_model_s_checkpoint_is_refused_by_name() {
    let device = Device::flex();
    let model = config().init(&device);
    let foreign = CheckpointHeader::new(
        "ptr-ar",
        &Codebook::V1,
        SlotEncoding::V1,
        &[
            TableSize {
                family: CodeFamily::SemanticRole,
                rows: 9,
            },
            TableSize {
                family: CodeFamily::EpistemicState,
                rows: 6,
            },
            TableSize {
                family: CodeFamily::ReasoningOperator,
                rows: 11,
            },
        ],
    );
    assert_eq!(
        load(&foreign.write(&payload(&model)), &config(), &device)
            .expect_err("a different model's layout"),
        CheckpointIoError::WrongModel {
            found: "ptr-ar".to_owned()
        }
    );
}

#[test]
fn a_checkpoint_for_a_different_architecture_is_refused_rather_than_partly_loaded() {
    let device = Device::flex();
    let bytes = save(&config().init(&device)).expect("serializes");
    let wider = PtrA0Config::new(VOCAB, D_MODEL + 4)
        .with_provenance_buckets(8)
        .with_latent_steps(1);
    let error = load(&bytes, &wider, &device).expect_err("the shapes do not fit");
    assert!(
        matches!(error, CheckpointIoError::Record(_)),
        "expected a record refusal, got {error:?}"
    );
    // The identity was fine, so this refusal came from the parameters — and the
    // same bytes still load into the architecture they were written for.
    load(&bytes, &config(), &device).expect("the original architecture loads");
}

#[test]
fn every_prefix_of_a_checkpoint_is_refused() {
    let device = Device::flex();
    let bytes = save(&config().init(&device)).expect("serializes");
    for length in [0, 1, 7, 8, 9, 16, 64, bytes.len() / 2, bytes.len() - 1] {
        let error = load(&bytes[..length], &config(), &device)
            .err()
            .unwrap_or_else(|| panic!("a prefix of {length} bytes must not load"));
        assert!(
            matches!(error, CheckpointIoError::Header(_)),
            "prefix of {length} bytes gave {error:?}"
        );
    }
    load(&bytes, &config(), &device).expect("the whole checkpoint still loads");
}

/// The provenance width is a recorded exception, and the record is the only copy.
///
/// Every other cardinality an A0 table is sized by is the codebook's. This one is
/// not a family and never will be — it has no members to assign codes to — so what
/// keeps a dataset builder and a model from choosing two widths is that both read
/// the same recorded number.
#[test]
fn the_default_provenance_width_is_the_one_the_artifact_publishes() {
    // Deliberately read from the *file*, not from `exception_width`. A0's default
    // is resolved from the kernel at compile time, so comparing the two in Rust
    // compares a constant with itself and could not fail whatever either said.
    // What can fail - and is the thing this exception exists to prevent - is a
    // model built at one width while the artifact that a dataset builder reads
    // publishes another.
    let artifact = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../datasets/generated/codebook.json"
    ))
    .expect("datasets/generated/codebook.json is missing; run scripts/generate_codebook.py");

    let built = PtrA0Config::new(VOCAB, D_MODEL).provenance_bucket_count;
    assert!(
        artifact.contains(&format!("\"name\": \"{PROVENANCE_EXCEPTION}\"")),
        "the artifact records no {PROVENANCE_EXCEPTION} exception"
    );
    assert!(
        artifact.contains(&format!("\"width\": {built}")),
        "a model defaults to {built} provenance buckets and the artifact publishes \
         another width; run scripts/generate_codebook.py"
    );
    assert_eq!(built, PROVENANCE_BUCKET_COUNT);
}

/// A checkpoint written at one provenance width is refused by a model at another.
///
/// This is the property that matters, and it is what makes the width *checked*
/// rather than merely recorded: codes bucketed into 8 index a 4-row table just as
/// well as a 4-bucket dataset's, and the model would read a provenance it was
/// never given. Note what refuses it — the embedding's own shape, through burn's
/// record validation, which reports a shape and not a field name. That is a real
/// check reporting the wrong thing, which is why it is asserted here rather than
/// assumed from the header: the header records identity, and this is architecture.
#[test]
fn a_checkpoint_is_refused_by_a_model_with_another_provenance_width() {
    let device = Device::default();
    let written = PtrA0Config::new(VOCAB, D_MODEL)
        .with_provenance_buckets(8)
        .with_latent_steps(1);
    let narrower = PtrA0Config::new(VOCAB, D_MODEL)
        .with_provenance_buckets(4)
        .with_latent_steps(1);

    let bytes = save(&written.init(&device)).expect("a model saves");

    // The control first: the identical width loads, so the refusal below is about
    // the width and not about this pair of configs or this artifact.
    load(&bytes, &written, &device).expect("the same width loads");

    match load(&bytes, &narrower, &device) {
        Err(CheckpointIoError::Record(_)) => {}
        Err(other) => panic!("refused, but not by the record: {other:?}"),
        Ok(_) => panic!("an 8-bucket checkpoint loaded into a 4-bucket model"),
    }
}

/// And the identity header does not carry the width, which is the honest limit.
///
/// `CheckpointHeader` records the codebook and one `TableSize` per [`CodeFamily`].
/// The provenance width is not a family, so there is no slot for it and the header
/// verifies without ever looking at it. The refusal above comes one step later,
/// from the record. Asserted rather than left implied, because "the header checks
/// it" is exactly what a reader would assume.
#[test]
fn the_header_verifies_without_consulting_the_provenance_width() {
    let device = Device::default();
    let wide = PtrA0Config::new(VOCAB, D_MODEL).with_provenance_buckets(8);
    let narrow = PtrA0Config::new(VOCAB, D_MODEL).with_provenance_buckets(4);

    let written = header(&wide.init(&device));
    written
        .verify(&narrow.codebook(), narrow.encoding(), &EMBEDDED_FAMILIES)
        .expect("the header agrees: the width it differs by is not a family");
}
