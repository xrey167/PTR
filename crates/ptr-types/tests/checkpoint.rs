//! A checkpoint must not load under an assignment it was not trained under.
//!
//! The decoder is written against bytes rather than against the encoder: the tests
//! build malformed headers directly, because a decoder that only ever sees its own
//! writer's output has not been tested on the inputs that matter.

use ptr_types::{
    CheckpointError, CheckpointHeader, CodeFamily, Codebook, CodebookVersion, EncodingVersion,
    SlotEncoding, TableSize, FORMAT_V2,
};

const EMBEDDED: [CodeFamily; 2] = [CodeFamily::SemanticRole, CodeFamily::EpistemicState];

/// The header this build would write for a model embedding roles and states.
fn header(book: &Codebook) -> CheckpointHeader {
    CheckpointHeader::new(
        "ptr-a0",
        book,
        SlotEncoding::V1,
        &[
            TableSize {
                family: CodeFamily::SemanticRole,
                rows: book.cardinality(CodeFamily::SemanticRole),
            },
            TableSize {
                family: CodeFamily::EpistemicState,
                rows: book.cardinality(CodeFamily::EpistemicState),
            },
        ],
    )
}

/// Encode a header from parts, so a test can write bytes this build never would.
#[allow(clippy::too_many_arguments)]
fn encode(
    magic: &[u8],
    format: u16,
    model: &str,
    version: u32,
    assignment: &[u8],
    encoding: u32,
    tables: &[(&str, u16)],
    declared_payload: u64,
    payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(magic);
    out.extend_from_slice(&format.to_le_bytes());
    out.extend_from_slice(&(model.len() as u16).to_le_bytes());
    out.extend_from_slice(model.as_bytes());
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(assignment.len() as u32).to_le_bytes());
    out.extend_from_slice(assignment);
    out.extend_from_slice(&encoding.to_le_bytes());
    out.extend_from_slice(&(tables.len() as u16).to_le_bytes());
    for (name, rows) in tables {
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&rows.to_le_bytes());
    }
    out.extend_from_slice(&declared_payload.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// A well-formed encoding of this build's header, for tests that damage one part.
fn valid(payload: &[u8]) -> Vec<u8> {
    let book = Codebook::V1;
    encode(
        b"PTRCKPT\x00",
        FORMAT_V2,
        "ptr-a0",
        book.version().0,
        &book.canonical_bytes(),
        EncodingVersion::V1.0,
        &[
            ("semantic_role", book.cardinality(CodeFamily::SemanticRole)),
            (
                "epistemic_state",
                book.cardinality(CodeFamily::EpistemicState),
            ),
        ],
        payload.len() as u64,
        payload,
    )
}

#[test]
fn a_header_round_trips_with_its_payload() {
    let book = Codebook::V1;
    let weights = vec![9_u8; 64];
    let bytes = header(&book).write(&weights);
    let (read, payload) = CheckpointHeader::read(&bytes).expect("its own output reads back");
    assert_eq!(read, header(&book));
    assert_eq!(payload, weights.as_slice());
    assert_eq!(read.model, "ptr-a0");
    assert_eq!(read.codebook, CodebookVersion::V1);
    assert_eq!(read.table(CodeFamily::SemanticRole), Some(9));
    assert_eq!(read.table(CodeFamily::EpistemicState), Some(6));
    assert_eq!(read.table(CodeFamily::Validity), None);
    read.verify(&book, SlotEncoding::V1, &EMBEDDED)
        .expect("written by this build");
}

#[test]
fn an_empty_payload_is_a_valid_checkpoint_and_stays_empty() {
    let book = Codebook::V1;
    let bytes = header(&book).write(&[]);
    let (_, payload) = CheckpointHeader::read(&bytes).expect("an empty payload is well formed");
    assert!(payload.is_empty());
}

#[test]
fn the_assignment_is_stored_verbatim_so_a_reader_needs_no_hash() {
    let book = Codebook::V1;
    let bytes = header(&book).write(&[1, 2, 3]);
    let (read, _) = CheckpointHeader::read(&bytes).expect("well formed");
    assert_eq!(read.codebook_bytes, book.canonical_bytes());
}

#[test]
fn a_moved_assignment_is_refused_at_the_very_same_version() {
    // The failure the version alone cannot catch: tables edited in place. Every
    // code in the artifact now denotes something this build cannot reproduce, and
    // the version number says nothing is wrong.
    let book = Codebook::V1;
    let mut moved = header(&book);
    let last = moved.codebook_bytes.len() - 1;
    moved.codebook_bytes[last] ^= 0x01;
    assert_eq!(
        moved.verify(&book, SlotEncoding::V1, &EMBEDDED),
        Err(CheckpointError::CodebookMoved {
            version: CodebookVersion::V1
        })
    );
    // And with the assignment intact it passes, so the test above is not passing
    // for some unrelated reason.
    header(&book)
        .verify(&book, SlotEncoding::V1, &EMBEDDED)
        .expect("the intact header verifies");
}

#[test]
fn a_foreign_codebook_version_is_refused_rather_than_interpreted() {
    let book = Codebook::V1;
    let mut foreign = header(&book);
    foreign.codebook = CodebookVersion(7);
    assert_eq!(
        foreign.verify(&book, SlotEncoding::V1, &EMBEDDED),
        Err(CheckpointError::UnknownCodebookVersion {
            version: CodebookVersion(7)
        })
    );
}

#[test]
fn a_table_that_is_not_its_family_s_cardinality_is_refused_in_both_directions() {
    let book = Codebook::V1;
    for rows in [8_u16, 10] {
        let mut wrong = header(&book);
        wrong.tables[0] = TableSize {
            family: CodeFamily::SemanticRole,
            rows,
        };
        assert_eq!(
            wrong.verify(&book, SlotEncoding::V1, &EMBEDDED),
            Err(CheckpointError::TableSize {
                family: CodeFamily::SemanticRole,
                stored: rows,
                required: 9,
            }),
            "a table of {rows} rows is not nine semantic roles"
        );
    }
}

#[test]
fn a_required_family_that_was_never_recorded_is_refused() {
    let book = Codebook::V1;
    let partial = CheckpointHeader::new(
        "ptr-a0",
        &book,
        SlotEncoding::V1,
        &[TableSize {
            family: CodeFamily::SemanticRole,
            rows: 9,
        }],
    );
    assert_eq!(
        partial.verify(&book, SlotEncoding::V1, &EMBEDDED),
        Err(CheckpointError::MissingTable {
            family: CodeFamily::EpistemicState
        })
    );
}

#[test]
fn a_family_the_reader_does_not_require_is_left_alone() {
    // A model that embeds fewer families than the kernel defines is not wrong, so
    // verification asks only about the families the reader names.
    let book = Codebook::V1;
    header(&book)
        .verify(&book, SlotEncoding::V1, &[CodeFamily::SemanticRole])
        .expect("epistemic state is recorded but not asked about");
    header(&book)
        .verify(&book, SlotEncoding::V1, &[])
        .expect("naming no family still checks the assignment");
}

#[test]
fn foreign_bytes_are_not_a_checkpoint() {
    for bytes in [
        &b"PTRCODEBOOK\x00"[..],
        &b"\x00\x00\x00\x00\x00\x00\x00\x00extra"[..],
        &[0_u8; 8][..],
    ] {
        assert_eq!(
            CheckpointHeader::read(bytes).expect_err("not a checkpoint"),
            CheckpointError::NotACheckpoint
        );
    }
}

#[test]
fn an_unknown_format_is_refused_rather_than_parsed_anyway() {
    // Three rather than two: two *was* the unknown format until the slot-encoding
    // field made it this build's own. A test that names a version as "unknown" has to
    // move when that version arrives, or it silently starts asserting that the
    // current layout is refused — and passes for a while because it is comparing a
    // refusal it no longer gets to a value it no longer means.
    let mut bytes = valid(&[1, 2, 3]);
    bytes[8..10].copy_from_slice(&3_u16.to_le_bytes());
    assert_eq!(
        CheckpointHeader::read(&bytes).expect_err("format 3 is unknown here"),
        CheckpointError::UnknownFormat { format: 3 }
    );
}

#[test]
fn every_prefix_of_a_valid_checkpoint_is_refused_and_names_a_field() {
    // A truncated artifact must never decode. Checking every prefix rather than
    // one chosen length is the difference between testing the property and
    // testing an example.
    let bytes = valid(&[7, 7, 7, 7]);
    for length in 0..bytes.len() {
        let error =
            CheckpointHeader::read(&bytes[..length]).expect_err("a prefix is not a checkpoint");
        let acceptable = matches!(
            error,
            CheckpointError::NotACheckpoint
                | CheckpointError::Truncated { .. }
                | CheckpointError::PayloadLength { .. }
        );
        assert!(acceptable, "prefix of {length} bytes gave {error:?}");
    }
    CheckpointHeader::read(&bytes).expect("the whole thing still reads");
}

#[test]
fn trailing_bytes_are_refused_rather_than_ignored() {
    let mut bytes = valid(&[4, 5]);
    bytes.extend_from_slice(&[6, 7, 8]);
    assert_eq!(
        CheckpointHeader::read(&bytes).expect_err("trailing data means a layout disagreement"),
        CheckpointError::TrailingBytes { extra: 3 }
    );
}

#[test]
fn a_header_from_the_previous_layout_is_refused_rather_than_read() {
    // Format 1 had no slot-encoding field. Its bytes still begin plausibly — magic,
    // model, codebook, assignment — and reading on would take the *table count* as an
    // encoding version and then run off the end or, worse, parse. A layout is not
    // forward-compatible because its first fields happen to line up.
    let book = Codebook::V1;
    let mut previous = Vec::new();
    previous.extend_from_slice(b"PTRCKPT\x00");
    previous.extend_from_slice(&1_u16.to_le_bytes());
    previous.extend_from_slice(&("ptr-a0".len() as u16).to_le_bytes());
    previous.extend_from_slice(b"ptr-a0");
    previous.extend_from_slice(&book.version().0.to_le_bytes());
    let assignment = book.canonical_bytes();
    previous.extend_from_slice(&(assignment.len() as u32).to_le_bytes());
    previous.extend_from_slice(&assignment);
    previous.extend_from_slice(&1_u16.to_le_bytes());
    previous.extend_from_slice(&("semantic_role".len() as u16).to_le_bytes());
    previous.extend_from_slice(b"semantic_role");
    previous.extend_from_slice(&9_u16.to_le_bytes());
    previous.extend_from_slice(&0_u64.to_le_bytes());

    assert_eq!(
        CheckpointHeader::read(&previous).expect_err("the previous layout"),
        CheckpointError::UnknownFormat { format: 1 }
    );
}

#[test]
fn a_header_recording_another_slot_encoding_is_refused() {
    // The one identity in here that no tensor could reveal: the encoding produces the
    // model's input and no parameters, so nothing about the weights would show that
    // they were trained on vectors from a different definition.
    let book = Codebook::V1;
    let mut other = header(&book);
    other.encoding = EncodingVersion(2);
    assert_eq!(
        other.verify(&book, SlotEncoding::V1, &EMBEDDED),
        Err(CheckpointError::EncodingMoved {
            stored: EncodingVersion(2),
            required: EncodingVersion::V1,
        })
    );
    // It survives a round trip, so the refusal is about the value rather than about
    // the field being dropped on the way through.
    let (read, _) = CheckpointHeader::read(&other.write(&[])).expect("well formed");
    assert_eq!(read.encoding, EncodingVersion(2));

    // The control: the same header with this build's encoding verifies.
    header(&book)
        .verify(&book, SlotEncoding::V1, &EMBEDDED)
        .expect("this build's encoding");
}

#[test]
fn a_payload_shorter_than_declared_is_refused_with_both_lengths() {
    let book = Codebook::V1;
    let bytes = encode(
        b"PTRCKPT\x00",
        FORMAT_V2,
        "ptr-a0",
        book.version().0,
        &book.canonical_bytes(),
        EncodingVersion::V1.0,
        &[("semantic_role", 9)],
        64,
        &[1, 2, 3],
    );
    assert_eq!(
        CheckpointHeader::read(&bytes).expect_err("three bytes are not sixty-four"),
        CheckpointError::PayloadLength {
            declared: 64,
            found: 3
        }
    );
}

#[test]
fn an_unknown_family_name_is_refused_and_carries_the_name() {
    let book = Codebook::V1;
    let bytes = encode(
        b"PTRCKPT\x00",
        FORMAT_V2,
        "ptr-a0",
        book.version().0,
        &book.canonical_bytes(),
        EncodingVersion::V1.0,
        &[("semantic_role", 9), ("vibe", 3)],
        0,
        &[],
    );
    assert_eq!(
        CheckpointHeader::read(&bytes).expect_err("this build defines no such family"),
        CheckpointError::UnknownFamily {
            name: "vibe".to_owned()
        }
    );
}

#[test]
fn a_family_recorded_twice_is_refused_because_its_size_is_ambiguous() {
    let book = Codebook::V1;
    let bytes = encode(
        b"PTRCKPT\x00",
        FORMAT_V2,
        "ptr-a0",
        book.version().0,
        &book.canonical_bytes(),
        EncodingVersion::V1.0,
        &[("semantic_role", 9), ("semantic_role", 8)],
        0,
        &[],
    );
    assert_eq!(
        CheckpointHeader::read(&bytes).expect_err("two sizes for one family"),
        CheckpointError::DuplicateFamily {
            family: CodeFamily::SemanticRole
        }
    );
}

#[test]
fn a_field_that_is_not_utf8_is_refused() {
    let book = Codebook::V1;
    let mut bytes = encode(
        b"PTRCKPT\x00",
        FORMAT_V2,
        "model",
        book.version().0,
        &book.canonical_bytes(),
        EncodingVersion::V1.0,
        &[("semantic_role", 9)],
        0,
        &[],
    );
    // The model name starts after the eight magic bytes, the format and its own
    // length prefix.
    bytes[12] = 0xff;
    assert_eq!(
        CheckpointHeader::read(&bytes).expect_err("0xff starts no UTF-8 sequence"),
        CheckpointError::NotUtf8 { field: "model" }
    );
}

#[test]
fn a_family_name_maps_back_to_the_same_family_and_nothing_else_does() {
    for family in CodeFamily::ALL {
        assert_eq!(CodeFamily::from_name(family.name()), Some(family));
    }
    for name in [
        "",
        "SemanticRole",
        "semantic role",
        "semantic_roles",
        "validity ",
    ] {
        assert_eq!(
            CodeFamily::from_name(name),
            None,
            "{name:?} is not a family"
        );
    }
}

/// Not a test: a reminder the compiler delivers.
///
/// The array in `each_refusal_carries_its_own_diagnostic_code` is written by hand,
/// and a variant left out of it is invisible — every code the array does list stays
/// distinct, so the test passes while the omitted refusal's code and rendering go
/// unchecked. That is how `EncodingMoved` was missed. This match is exhaustive, so a
/// refusal a later build adds stops this file compiling; the fix is to name it in the
/// array above as well as here.
fn _every_refusal_is_accounted_for(error: &CheckpointError) {
    match error {
        CheckpointError::NotACheckpoint
        | CheckpointError::UnknownFormat { .. }
        | CheckpointError::Truncated { .. }
        | CheckpointError::NotUtf8 { .. }
        | CheckpointError::TrailingBytes { .. }
        | CheckpointError::PayloadLength { .. }
        | CheckpointError::UnknownFamily { .. }
        | CheckpointError::DuplicateFamily { .. }
        | CheckpointError::UnknownCodebookVersion { .. }
        | CheckpointError::CodebookMoved { .. }
        | CheckpointError::TableSize { .. }
        | CheckpointError::MissingTable { .. }
        | CheckpointError::EncodingMoved { .. } => {}
    }
}

#[test]
fn each_refusal_carries_its_own_diagnostic_code() {
    let errors = [
        CheckpointError::NotACheckpoint,
        CheckpointError::UnknownFormat { format: 2 },
        CheckpointError::Truncated { field: "magic" },
        CheckpointError::NotUtf8 { field: "model" },
        CheckpointError::TrailingBytes { extra: 1 },
        CheckpointError::PayloadLength {
            declared: 1,
            found: 0,
        },
        CheckpointError::UnknownFamily {
            name: "vibe".to_owned(),
        },
        CheckpointError::DuplicateFamily {
            family: CodeFamily::Validity,
        },
        CheckpointError::UnknownCodebookVersion {
            version: CodebookVersion(2),
        },
        CheckpointError::CodebookMoved {
            version: CodebookVersion::V1,
        },
        CheckpointError::TableSize {
            family: CodeFamily::Validity,
            stored: 1,
            required: 4,
        },
        CheckpointError::MissingTable {
            family: CodeFamily::Validity,
        },
        CheckpointError::EncodingMoved {
            stored: EncodingVersion(2),
            required: EncodingVersion::V1,
        },
    ];
    let mut codes: Vec<&str> = errors.iter().map(CheckpointError::code).collect();
    let total = codes.len();
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(codes.len(), total, "two refusals share a diagnostic code");
    for error in &errors {
        assert_eq!(error.to_string(), error.code());
    }
}
