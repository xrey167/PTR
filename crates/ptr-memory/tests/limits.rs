mod common;

use common::TempJournal;
use ptr_memory::{
    DurableMemoryStore, MemoryLimits, MemoryStoreError, MemoryStoreOptions, MAX_BODY_BYTES,
    MAX_ID_BYTES, MAX_LINKS_PER_RECORD, MAX_RECORDS_PER_CLASS, MAX_REVOCATION_DEPTH,
    MAX_REVOCATION_VISITED,
};

fn quiet(limits: MemoryLimits) -> MemoryStoreOptions {
    MemoryStoreOptions {
        limits,
        compact_after_records: None,
        ..MemoryStoreOptions::default()
    }
}

#[test]
fn limits_above_their_hard_ceilings_are_rejected_before_anything_is_created() {
    let file = TempJournal::new("ms-ceilings");
    let base = MemoryLimits::default();
    let cases: [(&str, usize, MemoryLimits); 7] = [
        (
            "max_records_per_class",
            MAX_RECORDS_PER_CLASS,
            MemoryLimits {
                max_records_per_class: usize::MAX,
                ..base
            },
        ),
        (
            "max_id_bytes",
            MAX_ID_BYTES,
            MemoryLimits {
                max_id_bytes: usize::MAX,
                ..base
            },
        ),
        (
            "max_body_bytes",
            MAX_BODY_BYTES,
            MemoryLimits {
                max_body_bytes: usize::MAX,
                ..base
            },
        ),
        (
            "max_supports",
            MAX_LINKS_PER_RECORD,
            MemoryLimits {
                max_supports: usize::MAX,
                ..base
            },
        ),
        (
            "max_derived_from",
            MAX_LINKS_PER_RECORD,
            MemoryLimits {
                max_derived_from: usize::MAX,
                ..base
            },
        ),
        (
            "max_revocation_depth",
            MAX_REVOCATION_DEPTH,
            MemoryLimits {
                max_revocation_depth: usize::MAX,
                ..base
            },
        ),
        (
            "max_revocation_visited",
            MAX_REVOCATION_VISITED,
            MemoryLimits {
                max_revocation_visited: usize::MAX,
                ..base
            },
        ),
    ];
    for (field, max, limits) in cases {
        assert!(matches!(
            DurableMemoryStore::open(&file.0, quiet(limits)),
            Err(MemoryStoreError::InvalidLimits { field: f, max: m }) if f == field && m == max
        ));
        assert!(!file.0.exists());
    }
}

#[test]
fn the_ceilings_themselves_are_accepted() {
    let file = TempJournal::new("ms-at-ceilings");
    let limits = MemoryLimits {
        max_records_per_class: MAX_RECORDS_PER_CLASS,
        max_id_bytes: MAX_ID_BYTES,
        max_body_bytes: MAX_BODY_BYTES,
        max_supports: MAX_LINKS_PER_RECORD,
        max_derived_from: MAX_LINKS_PER_RECORD,
        max_revocation_depth: MAX_REVOCATION_DEPTH,
        max_revocation_visited: MAX_REVOCATION_VISITED,
    };
    DurableMemoryStore::open(&file.0, quiet(limits)).unwrap();
}
