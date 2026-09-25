mod common;

use common::{config, write_about};
use ptr_fastmem::{decode_state, encode_state, FastMemory, FastMemoryError, WriteSeq};

#[test]
fn a_restored_journal_folds_to_the_same_state_as_the_live_memory() {
    let mut live = FastMemory::new(config(3)).unwrap();
    for (salt, source) in ["a", "b", "c", "d", "e"].iter().enumerate() {
        live.write(write_about(source, salt as u32)).unwrap();
    }
    live.revoke(|source| source.key == "c");

    let journal: Vec<_> = live
        .writes()
        .iter()
        .map(|write| {
            let salt = ["a", "b", "c", "d", "e"]
                .iter()
                .position(|s| *s == write.source().key)
                .unwrap() as u32;
            (write.seq(), write_about(&write.source().key, salt))
        })
        .collect();
    // The journal has a gap where "c" was: sequence numbers 1, 2, 4, 5.
    assert_eq!(
        journal.iter().map(|(seq, _)| seq.0).collect::<Vec<_>>(),
        vec![1, 2, 4, 5]
    );
    let restored = FastMemory::restore(config(3), journal).unwrap();
    assert_eq!(restored.state().cells(), live.state().cells());
}

#[test]
fn a_journal_that_goes_backwards_is_refused() {
    let journal = vec![
        (WriteSeq(2), write_about("a", 0)),
        (WriteSeq(2), write_about("b", 1)),
    ];
    assert_eq!(
        FastMemory::restore(config(2), journal).unwrap_err(),
        FastMemoryError::OutOfOrderWrite {
            expected: 3,
            actual: 2
        }
    );
}

#[test]
fn the_incremental_state_is_the_fold_of_its_journal() {
    let mut memory = FastMemory::new(config(2)).unwrap();
    for salt in 0..7 {
        memory
            .write(write_about(&format!("s{salt}"), salt))
            .unwrap();
    }
    assert_eq!(memory.refold_from_journal(), *memory.state());
}

#[test]
fn an_encoded_checkpoint_restores_bit_for_bit() {
    let mut memory = FastMemory::new(config(2)).unwrap();
    for salt in 0..3 {
        memory
            .write(write_about(&format!("s{salt}"), salt))
            .unwrap();
    }
    let decoded = decode_state(&encode_state(memory.state())).unwrap();
    assert_eq!(&decoded, memory.state());
    assert_eq!(decoded.applied(), WriteSeq(3));
}

#[test]
fn a_full_journal_refuses_further_writes() {
    let mut small = config(2);
    small.max_writes = 2;
    let mut memory = FastMemory::new(small).unwrap();
    memory.write(write_about("a", 0)).unwrap();
    memory.write(write_about("b", 1)).unwrap();
    assert_eq!(
        memory.write(write_about("c", 2)).unwrap_err(),
        FastMemoryError::JournalFull { limit: 2 }
    );
    memory.revoke(|source| source.key == "a");
    assert!(memory.write(write_about("c", 2)).is_ok());
}

#[test]
fn an_invalid_write_preserves_state_binding_and_the_next_sequence() {
    let mut memory = FastMemory::new(config(1)).unwrap();
    memory.write(write_about("a", 0)).unwrap();
    let state = memory.state().clone();
    let binding = memory.binding_digest();
    let writes = memory.writes().to_vec();
    let mut invalid = write_about("b", 1);
    invalid.value.pop();
    assert_eq!(
        memory.write(invalid),
        Err(FastMemoryError::DimensionMismatch {
            field: "value",
            expected: config(1).value_len(),
            actual: config(1).value_len() - 1,
        })
    );
    assert_eq!(memory.state(), &state);
    assert_eq!(memory.binding_digest(), binding);
    assert_eq!(memory.writes(), writes);
    assert_eq!(memory.write(write_about("b", 1)).unwrap().seq, WriteSeq(2));
}

#[test]
fn restoring_a_gapped_journal_continues_after_its_largest_sequence() {
    let mut memory = FastMemory::restore(
        config(2),
        [
            (WriteSeq(3), write_about("a", 0)),
            (WriteSeq(7), write_about("b", 1)),
        ],
    )
    .unwrap();
    assert_eq!(memory.state().applied(), WriteSeq(7));
    assert_eq!(memory.write(write_about("c", 2)).unwrap().seq, WriteSeq(8));
    assert_eq!(memory.state(), &memory.refold_from_journal());
}

#[test]
fn restoring_refuses_zero_sequence_and_journals_over_capacity() {
    assert_eq!(
        FastMemory::restore(config(1), [(WriteSeq(0), write_about("a", 0))]).unwrap_err(),
        FastMemoryError::OutOfOrderWrite {
            expected: 1,
            actual: 0
        }
    );
    let mut small = config(1);
    small.max_writes = 1;
    assert_eq!(
        FastMemory::restore(
            small,
            [
                (WriteSeq(1), write_about("a", 0)),
                (WriteSeq(9), write_about("b", 1)),
            ]
        )
        .unwrap_err(),
        FastMemoryError::JournalFull { limit: 1 }
    );
}
