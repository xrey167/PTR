mod common;

use common::{config, write_about};
use ptr_fastmem::{FastMemory, Query, SourceRef, WriteSeq};
use ptr_types::Generation;

const SOURCES: [&str; 9] = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];

fn memory_with(interval: u32, sources: &[&str]) -> FastMemory {
    let mut memory = FastMemory::new(config(interval)).unwrap();
    for source in sources {
        // The salt is the source's position in SOURCES, so a source writes the
        // same request whichever memory it is written into.
        let salt = SOURCES.iter().position(|s| s == source).unwrap() as u32;
        memory.write(write_about(source, salt)).unwrap();
    }
    memory
}

#[test]
fn revoking_a_source_leaves_exactly_the_state_that_never_saw_it() {
    for interval in [1, 2, 3, 4, 100] {
        let mut revoked = memory_with(interval, &SOURCES);
        let report = revoked.revoke(|source| source.key == "d");
        assert_eq!(report.removed, 1);

        let never: Vec<&str> = SOURCES.iter().copied().filter(|s| *s != "d").collect();
        let clean = memory_with(interval, &never);

        // Bit-identical cells, not approximately equal ones.
        assert_eq!(
            revoked.state().cells(),
            clean.state().cells(),
            "interval {interval}"
        );
    }
}

#[test]
fn the_refold_restarts_from_the_last_checkpoint_before_the_revoked_write() {
    let mut memory = memory_with(3, &SOURCES);
    // Checkpoints exist after writes 3, 6 and 9; "e" is write 5.
    let report = memory.revoke(|source| source.key == "e");
    assert_eq!(report.restarted_from, WriteSeq(3));
    assert_eq!(report.replayed, 5);
}

#[test]
fn a_revoked_source_leaves_no_write_and_no_dependency_behind() {
    let mut memory = memory_with(2, &SOURCES);
    memory.revoke(|source| source.key == "a" || source.key == "h");
    assert!(memory
        .writes()
        .iter()
        .all(|write| write.source().key != "a" && write.source().key != "h"));
    assert!(!memory.sources().contains(&SourceRef {
        key: "a".into(),
        generation: Generation(1),
        input_digest: [0; 32],
    }));
    assert_eq!(memory.sources().len(), SOURCES.len() - 2);
}

#[test]
fn revoking_nothing_changes_nothing() {
    let mut memory = memory_with(4, &SOURCES);
    let before = memory.state().clone();
    let report = memory.revoke(|source| source.key == "absent");
    assert_eq!(report.removed, 0);
    assert_eq!(memory.state(), &before);
}

#[test]
fn writes_after_a_revocation_continue_the_sequence_and_stay_exact() {
    let mut memory = memory_with(2, &SOURCES[..5]);
    memory.revoke(|source| source.key == "b");
    let next = memory.write(write_about("f", 5)).unwrap();
    assert_eq!(next.seq, WriteSeq(6));

    let clean = memory_with(2, &["a", "c", "d", "e", "f"]);
    assert_eq!(memory.state().cells(), clean.state().cells());
    assert_eq!(memory.refold_from_journal().cells(), memory.state().cells());
}

#[test]
fn a_revoked_fact_is_no_longer_recalled() {
    let mut memory = memory_with(4, &SOURCES);
    let probe = write_about("c", 2);
    let query = Query::new(memory.config(), probe.key.clone()).unwrap();
    let everything = |_: &SourceRef| true;
    let before = memory.read_admitted(&query, everything).unwrap();
    memory.revoke(|source| source.key == "c");
    let after = memory.read_admitted(&query, everything).unwrap();
    assert_ne!(before.values, after.values);
    let clean = memory_with(4, &["a", "b", "d", "e", "f", "g", "h", "i"]);
    assert_eq!(
        after.values,
        clean.read_admitted(&query, everything).unwrap().values
    );
}

#[test]
fn a_read_is_denied_while_the_state_still_depends_on_a_revoked_input() {
    let mut memory = memory_with(4, &SOURCES);
    let query = Query::new(memory.config(), write_about("a", 0).key).unwrap();
    // The lifecycle authority has revoked "e", but the refold has not run yet.
    let admissible = |source: &SourceRef| source.key != "e";
    assert_eq!(
        memory.read_admitted(&query, admissible).unwrap_err(),
        ptr_fastmem::FastMemoryError::Denied { sources: 1 }
    );
    memory.revoke(|source| source.key == "e");
    assert!(memory.read_admitted(&query, admissible).is_ok());
}

#[test]
fn the_binding_digest_names_the_exact_set_of_folded_writes() {
    let mut memory = memory_with(4, &SOURCES);
    let full = memory.binding_digest();
    assert_eq!(full, memory_with(4, &SOURCES).binding_digest());
    memory.revoke(|source| source.key == "b");
    assert_ne!(memory.binding_digest(), full);
}

#[test]
fn revocation_distinguishes_generations_and_digests_of_the_same_source_key() {
    let first = write_about("shared", 0);
    let mut second = write_about("shared", 1);
    second.source.generation = Generation(2);
    let mut third = write_about("shared", 2);
    third.source.generation = Generation(2);
    let mut memory = FastMemory::new(config(1)).unwrap();
    for request in [first.clone(), second.clone(), third.clone()] {
        memory.write(request).unwrap();
    }
    let report = memory.revoke(|source| source == &second.source);
    assert_eq!(report.removed, 1);
    let clean =
        FastMemory::restore(config(1), [(WriteSeq(1), first), (WriteSeq(3), third)]).unwrap();
    assert_eq!(memory.state(), clean.state());
    assert_eq!(memory.binding_digest(), clean.binding_digest());
    assert_eq!(memory.sources(), clean.sources());
}

#[test]
fn revoking_every_write_resets_cells_and_dependencies_but_keeps_sequence_monotone() {
    let mut memory = memory_with(2, &SOURCES[..4]);
    let report = memory.revoke(|_| true);
    assert_eq!(report.removed, 4);
    assert_eq!(report.replayed, 0);
    assert_eq!(report.restarted_from, WriteSeq(0));
    let empty = FastMemory::new(config(2)).unwrap();
    assert_eq!(memory.state(), empty.state());
    assert_eq!(memory.binding_digest(), empty.binding_digest());
    assert!(memory.sources().is_empty());
    assert!(memory.writes().is_empty());
    assert_eq!(memory.write(write_about("e", 4)).unwrap().seq, WriteSeq(5));
    assert_eq!(memory.state(), &memory.refold_from_journal());
}
