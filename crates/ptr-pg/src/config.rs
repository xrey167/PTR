use std::fmt;

use crate::error::PgError;

/// A lowercase SQL identifier: `[a-z][a-z0-9_]{0,39}` that is not a keyword
/// PostgreSQL reserves. Anything this type holds can be interpolated into DDL
/// without quoting and without injection.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Identifier(String);

impl Identifier {
    /// Validate and retain an ASCII identifier matching `[a-z][a-z0-9_]{0,39}`
    /// that is not one of [`RESERVED_KEYWORDS`]. Returns
    /// `PgError::InvalidIdentifier` for any other value.
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

/// The keywords PostgreSQL 16 to 18 reserve: those `pg_get_keywords()` lists
/// as reserved (`R`) or as reserved except as a function or type name (`T`).
/// Neither kind may stand unquoted where a schema, table or index name goes,
/// so `CREATE SCHEMA select` is a syntax error. Unreserved and column-name
/// keywords (`U`, `C`) may, and are accepted.
pub const RESERVED_KEYWORDS: [&str; 101] = [
    "all",
    "analyse",
    "analyze",
    "and",
    "any",
    "array",
    "as",
    "asc",
    "asymmetric",
    "authorization",
    "binary",
    "both",
    "case",
    "cast",
    "check",
    "collate",
    "collation",
    "column",
    "concurrently",
    "constraint",
    "create",
    "cross",
    "current_catalog",
    "current_date",
    "current_role",
    "current_schema",
    "current_time",
    "current_timestamp",
    "current_user",
    "default",
    "deferrable",
    "desc",
    "distinct",
    "do",
    "else",
    "end",
    "except",
    "false",
    "fetch",
    "for",
    "foreign",
    "freeze",
    "from",
    "full",
    "grant",
    "group",
    "having",
    "ilike",
    "in",
    "initially",
    "inner",
    "intersect",
    "into",
    "is",
    "isnull",
    "join",
    "lateral",
    "leading",
    "left",
    "like",
    "limit",
    "localtime",
    "localtimestamp",
    "natural",
    "not",
    "notnull",
    "null",
    "offset",
    "on",
    "only",
    "or",
    "order",
    "outer",
    "overlaps",
    "placing",
    "primary",
    "references",
    "returning",
    "right",
    "select",
    "session_user",
    "similar",
    "some",
    "symmetric",
    "system_user",
    "table",
    "tablesample",
    "then",
    "to",
    "trailing",
    "true",
    "union",
    "unique",
    "user",
    "using",
    "variadic",
    "verbose",
    "when",
    "where",
    "window",
    "with",
];

/// Refuse anything that is not a plain lowercase identifier, or that is a
/// keyword PostgreSQL reserves ([`RESERVED_KEYWORDS`]).
pub fn check_identifier(value: &str) -> Result<(), PgError> {
    let mut chars = value.chars();
    let valid = value.len() <= 40
        && chars.next().is_some_and(|first| first.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && !RESERVED_KEYWORDS.contains(&value);
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
///
/// Build one with [`SchemaSet::with_prefix`]; the fields are public to read.
/// `PgSubstrate::connect_with` refuses any set that is not `with_prefix` of
/// some prefix ([`SchemaSet::check`]): the three names must be distinct and
/// belong to this instance alone. A set composed from other names could name
/// another instance's schema, which a rebuild of this one would drop, its
/// projector would delete that instance's search documents and fast-memory
/// writes from, and its migrations would change, all under a migration lock
/// that instance never takes.
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

    /// Refuse a set that is not [`Self::with_prefix`] of some prefix, with
    /// `PgError::InvalidSchemaSet`.
    ///
    /// Each name ends in its class's suffix and the three share the prefix
    /// before it, so the names are pairwise distinct, a set is determined by
    /// its projection schema's name, and two accepted sets are either the
    /// same instance or share no schema at all.
    pub fn check(&self) -> Result<(), PgError> {
        let own = self
            .projection
            .as_str()
            .strip_suffix("_projection")
            .and_then(|prefix| Self::with_prefix(prefix).ok());
        if own.as_ref() == Some(self) {
            Ok(())
        } else {
            Err(PgError::InvalidSchemaSet {
                projection: self.projection.to_string(),
                derived: self.derived.to_string(),
                work: self.work.to_string(),
            })
        }
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
    /// Schemas of this instance, made with [`SchemaSet::with_prefix`]: a set
    /// composed by hand is refused on connect unless it is one prefix's.
    /// Separate prefixes are separate instances that share no schema, which
    /// is how tests and side-by-side rebuilds stay isolated.
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
    fn keywords_postgresql_reserves_are_refused_as_identifiers() {
        for reserved in [
            "select",
            "user",
            "table",
            "all",
            "authorization",
            "join",
            "current_schema",
            "system_user",
        ] {
            assert_eq!(
                Identifier::new(reserved),
                Err(PgError::InvalidIdentifier {
                    value: reserved.into()
                }),
                "{reserved}"
            );
        }
        // Unreserved and column-name keywords may name a schema unquoted.
        for allowed in ["schema", "data", "name", "vector", "between", "values"] {
            assert!(Identifier::new(allowed).is_ok(), "{allowed}");
        }
        // A reserved word is a prefix like any other: the names it yields
        // end in their class's suffix.
        assert!(SchemaSet::with_prefix("select").is_ok());
        let mut sorted = RESERVED_KEYWORDS;
        sorted.sort_unstable();
        assert_eq!(sorted, RESERVED_KEYWORDS, "kept in order to be reviewable");
    }

    #[test]
    fn a_schema_set_is_accepted_only_as_the_three_schemas_of_one_prefix() {
        let mine = SchemaSet::with_prefix("ptr_a").unwrap();
        let theirs = SchemaSet::with_prefix("ptr_b").unwrap();
        assert_eq!(mine.check(), Ok(()));
        let name = |value: &str| Identifier::new(value).unwrap();
        for composed in [
            // Another instance's work schema as this one's derived cache,
            // which a rebuild drops.
            SchemaSet {
                derived: theirs.work.clone(),
                ..mine.clone()
            },
            // Another instance's working state or cache, which this one's
            // projector deletes from.
            SchemaSet {
                work: theirs.work.clone(),
                ..mine.clone()
            },
            SchemaSet {
                derived: theirs.derived.clone(),
                ..mine.clone()
            },
            // One schema in every role, and two roles swapped.
            SchemaSet {
                projection: mine.projection.clone(),
                derived: mine.projection.clone(),
                work: mine.projection.clone(),
            },
            SchemaSet {
                projection: mine.derived.clone(),
                derived: mine.projection.clone(),
                work: mine.work.clone(),
            },
            // Distinct names that no prefix yields.
            SchemaSet {
                projection: name("ptr_projection"),
                derived: name("cache"),
                work: name("state"),
            },
        ] {
            let refused = composed.check().unwrap_err();
            assert_eq!(refused.code(), "PTR_PG_INVALID_SCHEMA_SET", "{composed:?}");
            assert_eq!(
                refused,
                PgError::InvalidSchemaSet {
                    projection: composed.projection.to_string(),
                    derived: composed.derived.to_string(),
                    work: composed.work.to_string(),
                }
            );
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
