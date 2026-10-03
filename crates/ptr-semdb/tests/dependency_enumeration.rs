use ptr_semdb::{SemanticDelta, SemanticError, SemanticHost};

fn graph() -> SemanticDelta {
    SemanticDelta {
        upserts: [
            ("source".into(), "input".into()),
            ("z".into(), "cached".into()),
        ]
        .into(),
        dependencies: [
            ("z".into(), ["source".into()].into()),
            ("a".into(), ["z".into()].into()),
        ]
        .into(),
        ..SemanticDelta::default()
    }
}

#[test]
fn derived_keys_enumerate_declarations_even_without_values_in_key_order() {
    let mut host = SemanticHost::default();
    assert_eq!(host.snapshot().derived_keys().count(), 0);
    host.apply_delta(graph()).unwrap();
    let snapshot = host.snapshot();
    assert_eq!(snapshot.derived_keys().collect::<Vec<_>>(), ["a", "z"]);
    assert_eq!(snapshot.keys().collect::<Vec<_>>(), ["source", "z"]);
    assert_eq!(snapshot.inputs("a").collect::<Vec<_>>(), ["z"]);
}

#[test]
fn removing_an_evicted_derivation_drops_only_its_declaration_and_preserves_old_snapshot() {
    let mut host = SemanticHost::default();
    host.apply_delta(graph()).unwrap();
    host.apply_delta(SemanticDelta {
        removals: ["source".into()].into(),
        ..SemanticDelta::default()
    })
    .unwrap();
    let before = host.snapshot();
    assert_eq!(before.keys().count(), 0);
    assert_eq!(before.derived_keys().collect::<Vec<_>>(), ["a", "z"]);
    host.apply_delta(SemanticDelta {
        removals: ["z".into()].into(),
        ..SemanticDelta::default()
    })
    .unwrap();
    let after = host.snapshot();
    assert_eq!(after.derived_keys().collect::<Vec<_>>(), ["a"]);
    assert_eq!(after.inputs("a").collect::<Vec<_>>(), ["z"]);
    assert_eq!(after.inputs("z").count(), 0);
    assert_eq!(before.derived_keys().collect::<Vec<_>>(), ["a", "z"]);
    assert_eq!(before.inputs("z").collect::<Vec<_>>(), ["source"]);
}

#[test]
fn clearing_inputs_removes_the_key_from_dependency_enumeration() {
    let mut host = SemanticHost::default();
    host.apply_delta(graph()).unwrap();
    host.apply_delta(SemanticDelta {
        dependencies: [("z".into(), Default::default())].into(),
        upserts: [("z".into(), "independent".into())].into(),
        ..SemanticDelta::default()
    })
    .unwrap();
    let snapshot = host.snapshot();
    assert_eq!(snapshot.derived_keys().collect::<Vec<_>>(), ["a"]);
    assert_eq!(snapshot.get("z"), Some("independent"));
    assert_eq!(snapshot.inputs("z").count(), 0);
}

#[test]
fn rejected_dependency_cycle_does_not_leak_into_snapshot_enumeration() {
    let mut host = SemanticHost::default();
    host.apply_delta(graph()).unwrap();
    assert_eq!(
        host.apply_delta(SemanticDelta {
            dependencies: [("source".into(), ["a".into()].into())].into(),
            ..SemanticDelta::default()
        }),
        Err(SemanticError::CyclicDependency)
    );
    let snapshot = host.snapshot();
    assert_eq!(snapshot.derived_keys().collect::<Vec<_>>(), ["a", "z"]);
    assert_eq!(snapshot.inputs("source").count(), 0);
}
