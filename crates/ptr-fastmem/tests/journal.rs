mod common;

use common::{codebook, config, write_about};
use ptr_fastmem::{
    decode_state, encode_state, Decay, FastMemory, FastMemoryConfig, FastMemoryError,
    FastWeightState, Query, SourceRef, WriteRequest, WriteSeq, MAX_VALUE_MAGNITUDE,
};
use ptr_types::Generation;

#[test]
fn a_restored_journal_folds_to_the_same_state_as_the_live_memory() {
    let mut live = FastMemory::new(config(3), codebook(&config(3))).unwrap();
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
    let restored = FastMemory::restore(config(3), codebook(&config(3)), journal).unwrap();
    assert_eq!(restored.state().cells(), live.state().cells());
}

#[test]
fn a_journal_that_goes_backwards_is_refused() {
    let journal = vec![
        (WriteSeq(2), write_about("a", 0)),
        (WriteSeq(2), write_about("b", 1)),
    ];
    assert_eq!(
        FastMemory::restore(config(2), codebook(&config(2)), journal).unwrap_err(),
        FastMemoryError::OutOfOrderWrite {
            expected: 3,
            actual: 2
        }
    );
}

#[test]
fn the_incremental_state_is_the_fold_of_its_journal() {
    let mut memory = FastMemory::new(config(2), codebook(&config(2))).unwrap();
    for salt in 0..7 {
        memory
            .write(write_about(&format!("s{salt}"), salt))
            .unwrap();
    }
    assert_eq!(memory.refold_from_journal(), *memory.state());
}

#[test]
fn an_encoded_checkpoint_restores_bit_for_bit() {
    let mut memory = FastMemory::new(config(2), codebook(&config(2))).unwrap();
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
    let mut memory = FastMemory::new(small, codebook(&small)).unwrap();
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
    let mut memory = FastMemory::new(config(1), codebook(&config(1))).unwrap();
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
        codebook(&config(2)),
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
        FastMemory::restore(
            config(1),
            codebook(&config(1)),
            [(WriteSeq(0), write_about("a", 0))]
        )
        .unwrap_err(),
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
            codebook(&small),
            [
                (WriteSeq(1), write_about("a", 0)),
                (WriteSeq(9), write_about("b", 1)),
            ]
        )
        .unwrap_err(),
        FastMemoryError::JournalFull { limit: 1 }
    );
}

#[test]
fn restoring_a_journal_whose_last_sequence_has_no_successor_is_refused() {
    // Gaps are allowed, so one entry can jump to the end of the range; its
    // successor used to overflow (a panic, or a wrap to the reserved zero).
    for journal in [
        vec![(WriteSeq(u64::MAX), write_about("a", 0))],
        vec![
            (WriteSeq(1), write_about("a", 0)),
            (WriteSeq(u64::MAX), write_about("b", 1)),
        ],
    ] {
        assert_eq!(
            FastMemory::restore(config(1), codebook(&config(1)), journal).unwrap_err(),
            FastMemoryError::SequenceExhausted { seq: u64::MAX }
        );
    }
    let last = FastMemory::restore(
        config(1),
        codebook(&config(1)),
        [(WriteSeq(u64::MAX - 1), write_about("a", 0))],
    )
    .unwrap();
    assert_eq!(last.state().applied(), WriteSeq(u64::MAX - 1));
}

#[test]
fn a_write_that_would_take_the_last_sequence_number_is_refused_and_leaves_the_memory_unchanged() {
    let mut memory = FastMemory::restore(
        config(1),
        codebook(&config(1)),
        [(WriteSeq(u64::MAX - 1), write_about("a", 0))],
    )
    .unwrap();
    let state = memory.state().clone();
    let binding = memory.binding_digest();
    let writes = memory.writes().to_vec();
    assert_eq!(
        memory.write(write_about("b", 1)),
        Err(FastMemoryError::SequenceExhausted { seq: u64::MAX })
    );
    assert_eq!(memory.state(), &state);
    assert_eq!(memory.binding_digest(), binding);
    assert_eq!(memory.writes(), writes);
    // Revocation never hands a sequence number out again.
    memory.revoke(|_| true);
    assert_eq!(
        memory.write(write_about("b", 1)),
        Err(FastMemoryError::SequenceExhausted { seq: u64::MAX })
    );
}

#[test]
fn a_write_that_could_overflow_the_fold_is_refused_and_leaves_the_memory_unchanged() {
    // f32::MAX and then -f32::MAX under one unit key: the second write's error
    // `target - current` is -inf. Both writes used to be folded and journaled,
    // leaving infinite cells that no checkpoint could decode.
    let mut memory = FastMemory::new(config(1), codebook(&config(1))).unwrap();
    memory.write(write_about("a", 0)).unwrap();
    let state = memory.state().clone();
    let binding = memory.binding_digest();
    let writes = memory.writes().to_vec();
    for extreme in [f32::MAX, -f32::MAX] {
        let mut request = write_about("b", 1);
        request.value.fill(extreme);
        assert_eq!(
            memory.write(request),
            Err(FastMemoryError::ValueOutOfRange {
                index: 0,
                value: extreme
            })
        );
    }
    assert_eq!(memory.state(), &state);
    assert_eq!(memory.binding_digest(), binding);
    assert_eq!(memory.writes(), writes);
    assert_eq!(memory.write(write_about("b", 1)).unwrap().seq, WriteSeq(2));
    assert_eq!(memory.state(), &memory.refold_from_journal());
    assert_eq!(
        decode_state(&encode_state(memory.state())).unwrap(),
        *memory.state()
    );
}

#[test]
fn restoring_a_journal_whose_fold_could_overflow_is_refused() {
    let mut up = write_about("a", 0);
    up.value.fill(f32::MAX);
    let mut down = write_about("a", 0);
    down.value.fill(-f32::MAX);
    assert_eq!(
        FastMemory::restore(
            config(1),
            codebook(&config(1)),
            [(WriteSeq(1), up), (WriteSeq(2), down)]
        )
        .unwrap_err(),
        FastMemoryError::ValueOutOfRange {
            index: 0,
            value: f32::MAX
        }
    );
}

#[test]
fn writes_at_the_value_bound_fold_to_finite_decodable_state_in_every_refold() {
    // The overflowing pattern above at the largest admitted magnitude: signs
    // alternate at full strength under three shared keys.
    let requests: Vec<_> = (0..64_u32)
        .map(|index| {
            let mut request = write_about(&format!("s{}", index % 8), index % 3);
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            request.value.fill(sign * MAX_VALUE_MAGNITUDE);
            request.beta = 1.0;
            request.decay = Decay::None;
            request
        })
        .collect();
    let finite = |state: &FastWeightState| state.cells().iter().all(|cell| cell.is_finite());
    let mut memory = FastMemory::new(config(4), codebook(&config(4))).unwrap();
    for request in &requests {
        assert!(memory.write(request.clone()).unwrap().surprise.is_finite());
    }
    assert!(finite(memory.state()));

    memory.revoke(|source| source.key == "s3");
    let never = FastMemory::restore(
        config(4),
        codebook(&config(4)),
        requests
            .iter()
            .enumerate()
            .filter(|(_, request)| request.source.key != "s3")
            .map(|(index, request)| (WriteSeq(index as u64 + 1), request.clone())),
    )
    .unwrap();
    assert!(finite(memory.state()));
    assert_eq!(memory.state(), never.state());
    assert_eq!(memory.state(), &memory.refold_from_journal());
    assert_eq!(
        decode_state(&encode_state(memory.state())).unwrap(),
        *memory.state()
    );
}

#[test]
fn keys_whose_squares_underflow_fold_to_a_finite_decodable_state_in_every_refold() {
    // Every entry is finite and nonzero, but every square is below f32's
    // smallest subnormal. The head used to be divided by a norm computed from
    // those squares, stored with norm 2.4 instead of 1, and 58 writes at full
    // strength drove the state to -inf; a restore of 60 of them was all NaN.
    let config = FastMemoryConfig {
        heads: 1,
        key_dim: 10,
        value_dim: 1,
        checkpoint_interval: 1000,
        max_writes: 65_536,
    };
    let mut key = vec![2.6e-23_f32; 10];
    key[0] = 4.5e-23;
    let request = WriteRequest {
        source: SourceRef {
            key: "a".into(),
            generation: Generation(1),
            input_digest: [1; 32],
        },
        key: key.clone(),
        value: vec![1.0],
        beta: 1.0,
        decay: Decay::None,
    };
    let mut memory = FastMemory::new(config, codebook(&config)).unwrap();
    for _ in 0..60 {
        assert!(memory.write(request.clone()).unwrap().surprise.is_finite());
    }
    let norm = |values: &[f32]| values.iter().map(|x| x * x).sum::<f32>().sqrt();
    let stored = norm(memory.writes()[0].key());
    assert!((stored - 1.0).abs() < 1e-6, "{stored}");
    assert!(memory.state().cells().iter().all(|cell| cell.is_finite()));
    let query = Query::new(&config, key).unwrap();
    assert!((norm(query.key()) - 1.0).abs() < 1e-6);
    let readout = memory.read_admitted(&query, |_| true).unwrap();
    assert!((readout.values[0] - 1.0).abs() < 1e-5, "{readout:?}");

    let restored = FastMemory::restore(
        config,
        codebook(&config),
        (1..=60).map(|seq| (WriteSeq(seq), request.clone())),
    )
    .unwrap();
    assert_eq!(restored.state(), memory.state());
    assert_eq!(memory.state(), &memory.refold_from_journal());
    assert_eq!(
        decode_state(&encode_state(memory.state())).unwrap(),
        *memory.state()
    );
}
