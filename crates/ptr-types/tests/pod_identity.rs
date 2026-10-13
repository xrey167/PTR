use ptr_types::{ClusterKey, PodIdentityDescriptor, PodIdentityError, POD_IDENTITY_SCHEMA_VERSION};

fn descriptor() -> PodIdentityDescriptor {
    PodIdentityDescriptor {
        schema_version: POD_IDENTITY_SCHEMA_VERSION,
        domain: "supply".into(),
        kind: "Retrieval".into(),
        predicate: Some("lead_time".into()),
        semantic_role: "reader".into(),
        aliases: ["lead time".to_owned(), "delivery time".to_owned()].into(),
    }
}

#[test]
fn identity_and_cluster_key_are_deterministic() {
    assert_eq!(descriptor().identity(), descriptor().identity());
    assert_eq!(descriptor().cluster_key(), descriptor().cluster_key());
}

#[test]
fn alias_change_changes_identity_but_not_cluster_key() {
    let base = descriptor();
    let mut changed = base.clone();
    changed.aliases.insert("shipping delay".into());
    assert_ne!(base.identity(), changed.identity());
    assert_eq!(base.cluster_key(), changed.cluster_key());
}

#[test]
fn role_domain_kind_and_predicate_changes_change_both() {
    let base = descriptor();
    let mut role = base.clone();
    role.semantic_role = "writer".into();
    let mut domain = base.clone();
    domain.domain = "finance".into();
    let mut kind = base.clone();
    kind.kind = "Memory".into();
    let mut predicate = base.clone();
    predicate.predicate = None;
    for changed in [role, domain, kind, predicate] {
        assert_ne!(base.identity(), changed.identity());
        assert_ne!(base.cluster_key(), changed.cluster_key());
    }
}

#[test]
fn field_boundaries_cannot_be_shifted_between_fields() {
    let mut left = descriptor();
    left.domain = "ab".into();
    left.kind = "c".into();
    let mut right = descriptor();
    right.domain = "a".into();
    right.kind = "bc".into();
    assert_ne!(left.identity(), right.identity());
    assert_ne!(left.cluster_key(), right.cluster_key());
}

#[test]
fn validate_rejects_unknown_schema_and_empty_fields() {
    let mut schema = descriptor();
    schema.schema_version = 2;
    assert_eq!(
        schema.validate(),
        Err(PodIdentityError::UnsupportedSchema {
            expected: POD_IDENTITY_SCHEMA_VERSION,
            actual: 2
        })
    );
    let mut domain = descriptor();
    domain.domain.clear();
    assert_eq!(
        domain.validate(),
        Err(PodIdentityError::EmptyField { field: "domain" })
    );
    let mut predicate = descriptor();
    predicate.predicate = Some(String::new());
    assert_eq!(
        predicate.validate(),
        Err(PodIdentityError::EmptyField { field: "predicate" })
    );
    let mut alias = descriptor();
    alias.aliases.insert(String::new());
    assert_eq!(
        alias.validate(),
        Err(PodIdentityError::EmptyField { field: "alias" })
    );
    assert_eq!(descriptor().validate(), Ok(()));
}

#[test]
fn cluster_key_display_matches_hex() {
    let key: ClusterKey = descriptor().cluster_key();
    assert_eq!(key.to_string(), key.to_hex());
}
