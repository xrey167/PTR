use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_pg::{
    check_catalog, event_payload, event_subject, event_topic, lifecycle_change, plan_migrations,
    Capabilities, Identifier, LifecycleChange, Migration, PgError, SchemaClass, SchemaSet,
    MINIMUM_SERVER, WORK_MIGRATIONS,
};
use ptr_types::{CapsuleId, CommitIndex, Generation};

#[test]
fn identifiers_and_schema_prefixes_accept_their_exact_length_boundaries() {
    let identifier = "a".repeat(40);
    assert_eq!(Identifier::new(&identifier).unwrap().as_str(), identifier);
    let prefix = "p".repeat(29);
    let schemas = SchemaSet::with_prefix(&prefix).unwrap();
    assert_eq!(schemas.projection.as_str(), format!("{prefix}_projection"));
    assert_eq!(schemas.projection.as_str().len(), 40);
    let too_long = "a".repeat(41);
    assert_eq!(
        Identifier::new(&too_long),
        Err(PgError::InvalidIdentifier { value: too_long })
    );
    for invalid in ["a\0b", "a\nb", "é", "_prefix"] {
        assert_eq!(
            Identifier::new(invalid),
            Err(PgError::InvalidIdentifier {
                value: invalid.into()
            })
        );
    }
}

#[test]
fn supported_capabilities_require_both_server_and_iterative_vector_scan_boundaries() {
    let oldest = Capabilities::from_catalog(MINIMUM_SERVER, [("vector", "0.8")]);
    assert_eq!(oldest.check_supported(), Ok(()));
    assert!(oldest.iterative_scans());
    let old = Capabilities::from_catalog(MINIMUM_SERVER - 1, [("vector", "0.8.0")]);
    assert_eq!(
        old.check_supported(),
        Err(PgError::UnsupportedServer {
            found: MINIMUM_SERVER - 1,
            minimum: MINIMUM_SERVER,
        })
    );
    for version in ["0.7.99", "invalid", "0", "0.8.x"] {
        let caps = Capabilities::from_catalog(MINIMUM_SERVER, [("vector", version)]);
        assert_eq!(
            caps.check_supported(),
            Err(PgError::MissingExtension { name: "vector" })
        );
        assert!(!caps.iterative_scans());
    }
    let newer = Capabilities::from_catalog(MINIMUM_SERVER, [("vector", "1.0.0")]);
    assert_eq!(newer.check_supported(), Ok(()));
    assert!(newer.iterative_scans());
}

#[test]
fn partially_applied_migrations_resume_in_order_and_completed_catalogs_are_noops() {
    for applied_count in 0..=WORK_MIGRATIONS.len() {
        let applied: Vec<_> = WORK_MIGRATIONS[..applied_count]
            .iter()
            .map(|m| (m.version, m.checksum()))
            .collect();
        let pending = plan_migrations(SchemaClass::Work, WORK_MIGRATIONS, &applied).unwrap();
        let expected: Vec<_> = WORK_MIGRATIONS[applied_count..].iter().collect();
        assert_eq!(pending, expected, "applied {applied_count}");
    }
}

#[test]
fn catalogs_refuse_duplicate_versions_wrong_schema_and_missing_names() {
    let first = WORK_MIGRATIONS[0];
    assert_eq!(
        check_catalog(SchemaClass::Work, &[first, first]),
        Err(PgError::InvalidCatalog {
            message: "versions must be 1..=n without gaps, in order",
        })
    );
    assert_eq!(
        check_catalog(
            SchemaClass::Work,
            &[Migration {
                sql: "CREATE TABLE {{projection}}.x (id int);",
                ..first
            }]
        ),
        Err(PgError::InvalidCatalog {
            message: "every migration must address its own schema through its placeholder"
        })
    );
    assert_eq!(
        check_catalog(SchemaClass::Work, &[Migration { name: "", ..first }]),
        Err(PgError::InvalidCatalog {
            message: "every migration needs a name"
        })
    );
}

#[test]
fn migration_identity_binds_version_and_sql_before_schema_substitution() {
    let migration = Migration {
        version: 1,
        name: "links",
        sql: "{{work}}.x REFERENCES {{projection}}.y;\r\n{{derived}}.z\r\n",
    };
    let other_version = Migration {
        version: 2,
        ..migration
    };
    let other_sql = Migration {
        sql: "{{work}}.different",
        ..migration
    };
    assert_ne!(migration.checksum(), other_version.checksum());
    assert_ne!(migration.checksum(), other_sql.checksum());
    for prefix in ["one", "two"] {
        let rendered = migration.render(&SchemaSet::with_prefix(prefix).unwrap());
        assert_eq!(
            rendered,
            format!("{prefix}_work.x REFERENCES {prefix}_projection.y;\n{prefix}_derived.z\n")
        );
    }
}

#[test]
fn superseding_keeps_the_capsule_subject_without_overwriting_its_project() {
    let committed = CommittedEvent {
        index: CommitIndex(7),
        event: LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("a:quoted\""),
            old: Generation(1),
            new: Generation(2),
        },
    };
    assert_eq!(
        lifecycle_change(&committed),
        LifecycleChange::SetLive {
            target: "a:quoted\"".into(),
            generation: Generation(2),
            project: None,
        }
    );
    assert_eq!(event_subject(&committed), "a:quoted\"");
    assert_eq!(event_topic(&committed.event), "capsule.superseded");
    assert_eq!(
        event_payload(&committed),
        serde_json::json!({
            "commit_index": 7, "entries": { "capsule:a:quoted\":generation": "2" },
        })
    );
}

#[test]
fn verifier_attestations_publish_results_without_changing_lifecycle_admission() {
    for passed in [false, true] {
        let committed = CommittedEvent {
            index: CommitIndex(9),
            event: LedgerEvent::VerifierAttested {
                subject: "fact:a".into(),
                passed,
            },
        };
        assert_eq!(lifecycle_change(&committed), LifecycleChange::None);
        assert_eq!(event_subject(&committed), "fact:a");
        assert_eq!(event_topic(&committed.event), "verifier.attested");
        assert_eq!(
            event_payload(&committed),
            serde_json::json!({
                "commit_index": 9, "entries": { "verifier:fact:a": if passed { "pass" } else { "fail" } },
            })
        );
    }
}
