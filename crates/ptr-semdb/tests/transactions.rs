use ptr_semdb::{SemanticDelta, SemanticError, SemanticHost, SemanticPayload, MAX_DELTA_BYTES};
use ptr_types::{Revision, TypeId};
use std::collections::BTreeSet;

fn delta(key: &str, value: &str) -> SemanticDelta {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.into(), value.into());
    delta
}
fn chain() -> SemanticDelta {
    let mut d = delta("source", "one");
    d.upserts.insert("derived".into(), "two".into());
    d.upserts.insert("plan".into(), "three".into());
    d.upserts.insert("unrelated".into(), "retained".into());
    d.dependencies.insert("derived".into(), ["source".into()].into());
    d.dependencies.insert("plan".into(), ["derived".into()].into());
    d
}

#[test]
fn local_change_invalidates_only_dependency_closure() {
    let mut host = SemanticHost::default();
    let mut setup = SemanticDelta::default();
    setup.dependencies.insert("claim:x".into(), ["input:b".into()].into());
    setup.dependencies.insert("plan:p".into(), ["claim:x".into()].into());
    setup.dependencies.insert("unrelated".into(), ["input:a".into()].into());
    host.apply_delta(setup).unwrap();
    let (_, affected) = host.apply_delta(delta("input:b", "new")).unwrap();
    assert_eq!(affected, ["input:b".into(), "claim:x".into(), "plan:p".into()].into());
}

#[test]
fn prepared_change_is_unpublished_and_cannot_cross_hosts_or_revisions() {
    let mut host = SemanticHost::default();
    let view = host.snapshot();
    let prepared = host.prepare_delta(chain()).unwrap();
    assert_eq!(host.revision(), Revision(0));
    assert!(host.snapshot().value("source").is_none());
    assert!(!host.is_stale(&view));
    let mut other = SemanticHost::default();
    assert!(matches!(other.apply_prepared(prepared), Err(SemanticError::StalePreparation)));
    let prepared = host.prepare_delta(chain()).unwrap();
    host.apply_delta(delta("another", "intervening")).unwrap();
    assert!(matches!(host.apply_prepared(prepared), Err(SemanticError::StalePreparation)));
    assert!(host.is_stale(&view));
    assert!(host.snapshot().value("source").is_none());
}

#[test]
fn payload_bytes_type_and_source_are_all_revision_significant() {
    let mut host = SemanticHost::default();
    let mut d = SemanticDelta::default();
    let mut p = SemanticPayload { type_id: TypeId::from("bytes"), source: "pod:a".into(), bytes: (0..=255).collect() };
    for expected in 1..=4 {
        match expected { 2 => p.bytes[0] = 17, 3 => p.source = "pod:b".into(), 4 => p.type_id = "other".into(), _ => {} }
        d.upserts.insert("output".into(), p.clone().into());
        let encoded = d.encode().unwrap();
        assert_eq!(SemanticDelta::decode(&encoded).unwrap(), d);
        assert_eq!(host.apply_delta(d.clone()).unwrap().0, Revision(expected));
        assert_eq!(host.snapshot().payload("output"), Some(&p));
        assert_eq!(host.apply_delta(d.clone()).unwrap().0, Revision(expected));
    }
}

#[test]
fn edits_and_removals_evict_transitive_derivations_but_not_unrelated_values() {
    for remove in [false, true] {
        let mut host = SemanticHost::default();
        host.apply_delta(chain()).unwrap();
        let before = host.snapshot();
        let mut update = delta("source", "changed");
        if remove { update.upserts.clear(); update.removals.insert("source".into()); }
        let (revision, affected) = host.apply_delta(update).unwrap();
        assert_eq!(revision, Revision(2));
        assert_eq!(affected, ["source".into(), "derived".into(), "plan".into()].into());
        assert_eq!(host.snapshot().get("unrelated"), Some("retained"));
        assert_eq!(host.snapshot().get("derived"), None);
        assert_eq!(host.snapshot().get("plan"), None);
        assert_eq!(before.get("derived"), Some("two")); // immutable history, not live admission
        assert!(host.is_stale(&before));
        if remove {
            assert!(matches!(host.apply_delta(delta("derived", "old")), Err(SemanticError::MissingDependency { .. })));
        }
    }
}

#[test]
fn explicit_recomputation_and_dependency_replacement_have_deterministic_semantics() {
    let mut host = SemanticHost::default();
    host.apply_delta(chain()).unwrap();
    let mut update = delta("source", "new source");
    update.upserts.insert("derived".into(), "recomputed".into());
    host.apply_delta(update).unwrap();
    assert_eq!(host.snapshot().get("derived"), Some("recomputed"));
    assert_eq!(host.snapshot().get("plan"), None);
    let mut replacement = SemanticDelta::default();
    replacement.dependencies.insert("derived".into(), ["unrelated".into()].into());
    host.apply_delta(replacement).unwrap();
    assert_eq!(host.revision(), Revision(3));
    assert_eq!(host.snapshot().get("derived"), None);
    assert_eq!(host.snapshot().inputs("derived").collect::<Vec<_>>(), vec!["unrelated"]);
    host.apply_delta(delta("derived", "from new input")).unwrap();
    host.apply_delta(delta("source", "no longer used")).unwrap();
    assert_eq!(host.snapshot().get("derived"), Some("from new input"));
}

#[test]
fn missing_inputs_cycles_and_conflicting_operations_are_atomic_rejections() {
    let mut host = SemanticHost::default();
    host.apply_delta(chain()).unwrap();
    let mut missing = delta("new", "never publish");
    missing.dependencies.insert("new".into(), ["missing".into()].into());
    let mut cycle = SemanticDelta::default();
    cycle.dependencies.insert("source".into(), ["plan".into()].into());
    let mut conflict = delta("source", "replacement");
    conflict.removals.insert("source".into());
    for bad in [missing, cycle, conflict, delta("", "invalid key")] {
        let view = host.snapshot();
        assert!(host.apply_delta(bad).is_err());
        assert!(!host.is_stale(&view));
        assert_eq!(host.revision(), Revision(1));
        assert_eq!(host.snapshot().get("source"), Some("one"));
        assert_eq!(host.snapshot().get("derived"), Some("two"));
        assert_eq!(host.snapshot().get("new"), None);
    }
}

#[test]
fn snapshots_cannot_be_made_current_by_matching_or_overwriting_revision_numbers() {
    let mut host = SemanticHost::default();
    let other = SemanticHost::default();
    let mut old = host.snapshot();
    assert!(host.is_stale(&other.snapshot()));
    host.apply_delta(delta("a", "new")).unwrap();
    old.revision = host.revision();
    assert!(host.is_stale(&old));
    assert!(!host.is_stale(&host.snapshot()));
}

#[test]
fn codec_has_a_golden_empty_encoding_and_rejects_all_truncated_prefixes() {
    let empty = SemanticDelta::default().encode().unwrap();
    assert_eq!(empty, [b"PTRSD001".as_slice(), &[0; 12]].concat());
    let bytes = chain().encode().unwrap();
    for end in 0..bytes.len() { assert!(SemanticDelta::decode(&bytes[..end]).is_err(), "prefix {end}"); }
    let mut trailing = bytes.clone(); trailing.push(0);
    assert!(SemanticDelta::decode(&trailing).is_err());
    let mut wrong_version = bytes; wrong_version[7] = b'2';
    assert!(SemanticDelta::decode(&wrong_version).is_err());
}

#[test]
fn codec_rejects_duplicate_keys_noncanonical_order_and_hostile_counts() {
    fn entry(key: u8) -> Vec<u8> { vec![1,0,0,0,key,0,1,0,0,0,b'x'] }
    for keys in [[b'a',b'a'], [b'b',b'a']] {
        let bytes = [b"PTRSD001".to_vec(), 2u32.to_le_bytes().to_vec(), entry(keys[0]), entry(keys[1]), vec![0;8]].concat();
        assert!(SemanticDelta::decode(&bytes).is_err());
    }
    let bytes = [b"PTRSD001".to_vec(), u32::MAX.to_le_bytes().to_vec()].concat();
    assert!(SemanticDelta::decode(&bytes).is_err());
    let huge = delta("source", &"x".repeat(MAX_DELTA_BYTES));
    assert_eq!(huge.encode(), Err(SemanticError::LimitExceeded));
}

#[test]
fn seeded_transaction_streams_replay_identical_values_dependencies_and_revisions() {
    for seed in [17_u64, 29, 43, 71, 101] {
        let mut writer = SemanticHost::default();
        let mut replay = SemanticHost::default();
        let mut n = seed;
        for step in 0..100 {
            n = n.wrapping_mul(6364136223846793005).wrapping_add(1);
            let key = format!("source:{}", n % 8);
            let mut d = delta(&key, &format!("{seed}:{step}"));
            if n % 5 == 0 { d.upserts.clear(); d.removals.insert(key); }
            let encoded = d.encode().unwrap();
            assert_eq!(writer.apply_delta(d).unwrap(), replay.apply_delta(SemanticDelta::decode(&encoded).unwrap()).unwrap());
            let a = writer.snapshot(); let b = replay.snapshot();
            let keys: BTreeSet<_> = a.keys().chain(b.keys()).collect();
            for key in keys { assert_eq!(a.value(key), b.value(key)); assert_eq!(a.inputs(key).collect::<Vec<_>>(), b.inputs(key).collect::<Vec<_>>()); }
        }
    }
}
