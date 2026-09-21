//! Slot identity is the codebook's, not the caller's.
//!
//! The model used to take the size of every typed table as an argument, and the
//! numbers callers passed disagreed with the kernel: eight slot types against nine
//! semantic roles, eight epistemic states against six, four operators against
//! eleven. Two failures follow from that, and neither one is visible at runtime. A
//! table longer than its family trains rows that mean nothing. A table shorter than
//! its family cannot represent the members past its end — and since the ids were
//! bare integers, an index past the end was equally unremarkable.
//!
//! So the tables are sized from the codebook and the ids are built from members.
//! These tests assert both, and that the refusals are refusals.

use burn::prelude::*;
use ptr_burn_a0::{CodeGrid, CodeGridError, PtrA0Config};
use ptr_types::{CodeFamily, Codebook, EpistemicState, ReasoningOperator, SemanticRole};

#[test]
fn every_typed_table_is_exactly_its_family_s_cardinality() {
    let book = Codebook::V1;
    let config = PtrA0Config::new(32, 8);

    assert_eq!(
        config.slot_type_count(),
        usize::from(book.cardinality_of::<SemanticRole>())
    );
    assert_eq!(
        config.epistemic_count(),
        usize::from(book.cardinality_of::<EpistemicState>())
    );
    assert_eq!(
        config.operator_count(),
        usize::from(book.cardinality_of::<ReasoningOperator>())
    );

    // The numbers themselves, so that a codebook change has to be looked at
    // rather than absorbed, and so the record shows what the old arguments got
    // wrong: 8, 8 and 4.
    assert_eq!(config.slot_type_count(), 9);
    assert_eq!(config.epistemic_count(), 6);
    assert_eq!(config.operator_count(), 11);
}

#[test]
fn every_member_of_an_embedded_family_has_a_row_in_its_table() {
    // The property behind the cardinalities: no member of a family the model
    // embeds falls outside the table it is looked up in. A shorter table would
    // put the last members out of range; a longer one would carry rows that
    // denote nothing.
    let book = Codebook::V1;
    let config = PtrA0Config::new(32, 8);

    let roles = book
        .assignment_of::<SemanticRole>()
        .expect("v1 assigns semantic roles");
    assert_eq!(roles.len(), config.slot_type_count());
    for (code, _) in &roles {
        assert!(usize::from(code.index()) < config.slot_type_count());
    }

    let states = book
        .assignment_of::<EpistemicState>()
        .expect("v1 assigns epistemic states");
    assert_eq!(states.len(), config.epistemic_count());
    for (code, _) in &states {
        assert!(usize::from(code.index()) < config.epistemic_count());
    }

    let operators = book
        .assignment_of::<ReasoningOperator>()
        .expect("v1 assigns reasoning operators");
    assert_eq!(operators.len(), config.operator_count());
}

#[test]
fn a_code_is_the_codebook_s_index_and_carries_its_family_and_version() {
    let device = Device::flex();
    let book = Codebook::V1;
    let rows: [&[SemanticRole]; 2] = [
        &[SemanticRole::Action, SemanticRole::Goal],
        &[SemanticRole::Procedure, SemanticRole::Relation],
    ];
    let grid = CodeGrid::new(&book, &rows, &device).expect("every role is assigned in v1");

    assert_eq!(grid.family(), CodeFamily::SemanticRole);
    assert_eq!(grid.codebook(), book.version());
    assert_eq!(grid.dims(), [2, 2]);

    let ids: Vec<i32> = grid.ids().into_data().try_into_vec::<i32>().unwrap();
    let expected: Vec<i32> = rows
        .iter()
        .flat_map(|row| row.iter())
        .map(|role| i32::from(book.code_of(*role).expect("assigned").index()))
        .collect();
    assert_eq!(ids, expected);
    // Not the declaration order of the enum: `Action` is last in the v1 table.
    assert_eq!(ids[0], 8);
}

#[test]
fn a_ragged_batch_is_refused_and_says_where() {
    let device = Device::flex();
    let rows: [&[SemanticRole]; 2] = [
        &[SemanticRole::Goal, SemanticRole::Claim],
        &[SemanticRole::Goal],
    ];
    let error = CodeGrid::new(&Codebook::V1, &rows, &device).expect_err("ragged rows are refused");
    assert_eq!(
        error,
        CodeGridError::Ragged {
            row: 1,
            expected: 2,
            found: 1,
        }
    );
    assert_eq!(error.to_string(), "PTR_A0_RAGGED_CODE_GRID");
}

#[test]
fn an_empty_batch_is_refused_rather_than_shaped() {
    let device = Device::flex();
    let none: [&[SemanticRole]; 0] = [];
    assert_eq!(
        CodeGrid::new(&Codebook::V1, &none, &device).expect_err("no rows is refused"),
        CodeGridError::Empty
    );
    let empty_row: [&[SemanticRole]; 1] = [&[]];
    assert_eq!(
        CodeGrid::new(&Codebook::V1, &empty_row, &device).expect_err("no slots is refused"),
        CodeGridError::Empty
    );
}

#[test]
fn the_model_reports_the_codebook_its_tables_were_sized_by() {
    let device = Device::flex();
    let model = PtrA0Config::new(32, 8).init(&device);
    assert_eq!(model.codebook(), Codebook::V1);
    assert_eq!(
        model.operator_count(),
        usize::from(Codebook::V1.cardinality_of::<ReasoningOperator>())
    );
}
