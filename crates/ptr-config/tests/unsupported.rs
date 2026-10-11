use ptr_config::{CliOverrides, FieldDisposition, PtrConfig, FIELD_DISPOSITIONS};
use std::collections::BTreeSet;

fn flatten(prefix: &str, value: &toml::Value, out: &mut Vec<String>) {
    match value {
        toml::Value::Table(table) => {
            for (key, inner) in table {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten(&path, inner, out);
            }
        }
        _ => out.push(prefix.to_owned()),
    }
}

fn set_path(root: &mut toml::Value, path: &str, value: toml::Value) {
    let mut node = root;
    let parts: Vec<&str> = path.split('.').collect();
    for part in &parts[..parts.len() - 1] {
        node = node.get_mut(*part).unwrap();
    }
    node.as_table_mut()
        .unwrap()
        .insert(parts[parts.len() - 1].to_owned(), value);
}

fn flipped(value: &toml::Value) -> toml::Value {
    match value {
        toml::Value::Boolean(b) => toml::Value::Boolean(!b),
        toml::Value::Integer(i) => toml::Value::Integer(i + 1),
        other => panic!("unexpected leaf type: {other:?}"),
    }
}

#[test]
fn every_config_field_has_a_consumer_or_a_refusal_rule() {
    let default = toml::Value::try_from(PtrConfig::default()).unwrap();
    let mut leaves = Vec::new();
    flatten("", &default, &mut leaves);
    let actual: BTreeSet<_> = leaves.into_iter().collect();
    let declared: BTreeSet<_> = FIELD_DISPOSITIONS
        .iter()
        .map(|(p, _)| (*p).to_owned())
        .collect();
    assert_eq!(
        actual, declared,
        "FIELD_DISPOSITIONS is out of sync with PtrConfig"
    );
}

#[test]
fn every_refused_field_accepts_default_and_refuses_a_changed_value() {
    let default = toml::Value::try_from(PtrConfig::default()).unwrap();
    let defaults_ok: PtrConfig = default.clone().try_into().unwrap();
    defaults_ok.validate().unwrap();
    for (path, disposition) in FIELD_DISPOSITIONS {
        if *disposition != FieldDisposition::RefusedUnlessDefault {
            continue;
        }
        let mut root = default.clone();
        let original = path
            .split('.')
            .fold(&default, |n, p| n.get(p).unwrap())
            .clone();
        set_path(&mut root, path, flipped(&original));
        let changed: PtrConfig = root.try_into().unwrap();
        let error = changed.validate().expect_err(path);
        assert!(error.contains(path), "{path}: {error}");
        let typed = changed.check_supported_settings().unwrap_err();
        assert_eq!(typed.field, *path);
    }
}

#[test]
fn retain_raw_evidence_false_is_refused_for_the_invariant() {
    let error = PtrConfig::from_toml_str("[semantic]\nretain_raw_evidence = false\n")
        .unwrap()
        .validate()
        .unwrap_err();
    assert!(error.contains("semantic.retain_raw_evidence") && error.contains("raw evidence"));
}

#[test]
fn unsupported_values_from_a_file_are_refused_at_load() {
    let dir = std::env::temp_dir().join(format!("ptr-config-unsupported-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("c.toml");
    std::fs::write(&path, "[observability]\nasync_backtrace = true\n").unwrap();
    let error = PtrConfig::from_path(&path).unwrap_err();
    assert!(error.contains("observability.async_backtrace") && error.contains("not supported yet"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unsupported_values_from_the_environment_are_refused() {
    for (var, value, field) in [
        (
            "PTR_RETAIN_RAW_EVIDENCE",
            "false",
            "semantic.retain_raw_evidence",
        ),
        (
            "PTR_STRUCTURED_TRACING",
            "false",
            "observability.structured_tracing",
        ),
        (
            "PTR_ASYNC_BACKTRACE",
            "true",
            "observability.async_backtrace",
        ),
        (
            "PTR_ALLOW_UNLOCKED_COMPONENTS",
            "false",
            "research.allow_unlocked_components",
        ),
        (
            "PTR_REQUIRE_EXPERIMENT_MANIFEST",
            "false",
            "research.require_experiment_manifest",
        ),
        (
            "PTR_MAX_PARALLEL_CANDIDATES",
            "8",
            "runtime.max_parallel_candidates",
        ),
    ] {
        let mut config = PtrConfig::default();
        let error = config.apply_env([(var, value)]).unwrap_err();
        assert!(error.contains(field), "{var}: {error}");
    }
    let mut config = PtrConfig::default();
    config
        .apply_env([
            ("PTR_RETAIN_RAW_EVIDENCE", "true"),
            ("PTR_MAX_PARALLEL_CANDIDATES", "4"),
        ])
        .unwrap();
}

#[test]
fn unsupported_cli_override_is_refused() {
    let overrides = CliOverrides::parse(["--max-parallel-candidates=9"]).unwrap();
    let mut config = PtrConfig::default();
    let error = overrides.apply_to(&mut config).unwrap_err();
    assert!(error.contains("runtime.max_parallel_candidates"));
}

#[test]
fn stale_revision_cancellation_can_be_switched_off_by_file_and_environment() {
    let from_file =
        PtrConfig::from_toml_str("[semantic]\ncancel_stale_revisions = false\n").unwrap();
    from_file.validate().unwrap();
    assert!(!from_file.semantic.cancel_stale_revisions);

    let mut from_env = PtrConfig::default();
    assert!(from_env.semantic.cancel_stale_revisions, "on by default");
    from_env
        .apply_env([("PTR_CANCEL_STALE_REVISIONS", "false")])
        .unwrap();
    from_env.validate().unwrap();
    assert!(!from_env.semantic.cancel_stale_revisions);

    // It is a consumed setting now: it is no longer in the refused-unless-default group.
    assert!(FIELD_DISPOSITIONS.iter().any(|(path, disposition)| {
        *path == "semantic.cancel_stale_revisions"
            && matches!(disposition, FieldDisposition::Consumed(_))
    }));
}
