use std::fmt;

use crate::error::PgError;

/// A lowercase SQL identifier: `[a-z][a-z0-9_]{0,39}`. Anything this type holds
/// can be interpolated into DDL without quoting and without injection.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Identifier(String);

impl Identifier {
    /// Validate and retain an ASCII identifier matching `[a-z][a-z0-9_]{0,39}`.
    /// Returns `PgError::InvalidIdentifier` for any other value.
    pub fn new(value: &str) -> Result<Self, PgError> {
        check_identifier(value)?;
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Refuse anything that is not a plain lowercase identifier.
pub fn check_identifier(value: &str) -> Result<(), PgError> {
    let mut chars = value.chars();
    let valid = value.len() <= 40
        && chars.next().is_some_and(|first| first.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if valid {
        Ok(())
    } else {
        Err(PgError::InvalidIdentifier {
            value: value.to_owned(),
        })
    }
}

/// The three schemas one substrate instance uses, each with a different
/// contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchemaSet {
    /// Ledger projection: rebuildable from the ledger, governed by the
    /// watermark, written only by the projector.
    pub projection: Identifier,
    /// Derived caches (search documents, embeddings): computed outside the
    /// projection, dropped and rebuilt with it.
    pub derived: Identifier,
    /// Non-authoritative working state (branches, fast-memory journals,
    /// lineage, labels): not derived from the ledger, never dropped by a
    /// rebuild, with its own backup and retention.
    pub work: Identifier,
}

impl SchemaSet {
    /// `<prefix>_projection`, `<prefix>_derived` and `<prefix>_work`.
    pub fn with_prefix(prefix: &str) -> Result<Self, PgError> {
        if prefix.len() > 29 {
            return Err(PgError::InvalidIdentifier {
                value: prefix.to_owned(),
            });
        }
        Ok(Self {
            projection: Identifier::new(&format!("{prefix}_projection"))?,
            derived: Identifier::new(&format!("{prefix}_derived"))?,
            work: Identifier::new(&format!("{prefix}_work"))?,
        })
    }
}

/// How to reach the substrate.
///
/// The connection string is referenced by the name of the environment
/// variable that holds it, never embedded: a config file that names
/// `PTR_PG_DSN` can be committed, one that holds a password cannot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PgConfig {
    pub dsn_variable: String,
    /// Schemas of this instance. Separate prefixes are separate instances,
    /// which is how tests and side-by-side rebuilds stay isolated.
    pub schemas: SchemaSet,
}

impl PgConfig {
    /// Resolve the connection string from the environment.
    ///
    /// Returns the original string, including surrounding whitespace. An unset,
    /// non-Unicode, or whitespace-only value becomes `PgError::MissingDsn`.
    pub fn resolve_dsn(&self) -> Result<String, PgError> {
        match std::env::var(&self.dsn_variable) {
            Ok(dsn) if !dsn.trim().is_empty() => Ok(dsn),
            _ => Err(PgError::MissingDsn {
                variable: self.dsn_variable.clone(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_plain_lowercase_identifiers_are_accepted() {
        for good in ["ptr", "ptr_test_01", "a"] {
            assert!(Identifier::new(good).is_ok(), "{good}");
        }
        for bad in [
            "",
            "1ptr",
            "Ptr",
            "ptr-test",
            "ptr; drop table x",
            "ptr\"",
            "p.t",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ] {
            assert!(Identifier::new(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_prefix_yields_three_distinct_schemas() {
        let set = SchemaSet::with_prefix("ptr").unwrap();
        assert_eq!(set.projection.as_str(), "ptr_projection");
        assert_eq!(set.derived.as_str(), "ptr_derived");
        assert_eq!(set.work.as_str(), "ptr_work");
        assert!(SchemaSet::with_prefix("Ptr").is_err());
        assert!(SchemaSet::with_prefix(&"p".repeat(30)).is_err());
    }

    #[test]
    fn an_unset_dsn_variable_is_a_typed_refusal() {
        let config = PgConfig {
            dsn_variable: "PTR_PG_TEST_VARIABLE_THAT_IS_NEVER_SET".into(),
            schemas: SchemaSet::with_prefix("ptr").unwrap(),
        };
        assert_eq!(
            config.resolve_dsn().unwrap_err().code(),
            "PTR_PG_MISSING_DSN"
        );
    }
}
