use sha2::{Digest, Sha256};

use crate::config::SchemaSet;
use crate::error::PgError;

/// Which schema a migration catalog builds. Each class is migrated and tracked
/// in its own schema, so the projection and the derived caches can be dropped
/// and rebuilt without touching working state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SchemaClass {
    Projection,
    Derived,
    Work,
}

impl SchemaClass {
    pub const ALL: [SchemaClass; 3] = [Self::Projection, Self::Derived, Self::Work];

    pub fn placeholder(self) -> &'static str {
        match self {
            Self::Projection => "{{projection}}",
            Self::Derived => "{{derived}}",
            Self::Work => "{{work}}",
        }
    }

    pub fn catalog(self) -> &'static [Migration] {
        match self {
            Self::Projection => PROJECTION_MIGRATIONS,
            Self::Derived => DERIVED_MIGRATIONS,
            Self::Work => WORK_MIGRATIONS,
        }
    }
}

/// One schema migration. The SQL names schemas through placeholders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Migration {
    pub version: u32,
    pub name: &'static str,
    pub sql: &'static str,
}

pub const PROJECTION_MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    name: "projection",
    sql: include_str!("../migrations/projection/0001_projection.sql"),
}];

pub const DERIVED_MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    name: "search",
    sql: include_str!("../migrations/derived/0001_search.sql"),
}];

pub const WORK_MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        name: "branch",
        sql: include_str!("../migrations/work/0001_branch.sql"),
    },
    Migration {
        version: 2,
        name: "fastmem",
        sql: include_str!("../migrations/work/0002_fastmem.sql"),
    },
    Migration {
        version: 3,
        name: "lineage",
        sql: include_str!("../migrations/work/0003_lineage.sql"),
    },
    Migration {
        version: 4,
        name: "labeling",
        sql: include_str!("../migrations/work/0004_labeling.sql"),
    },
    Migration {
        version: 5,
        name: "triage_policy",
        sql: include_str!("../migrations/work/0005_triage_policy.sql"),
    },
    Migration {
        version: 6,
        name: "labeling_adapter",
        sql: include_str!("../migrations/work/0006_labeling_adapter.sql"),
    },
    Migration {
        version: 7,
        name: "adapter_interference",
        sql: include_str!("../migrations/work/0007_adapter_interference.sql"),
    },
    Migration {
        version: 8,
        name: "branch_touched_inputs",
        sql: include_str!("../migrations/work/0008_branch_touched_inputs.sql"),
    },
];

impl Migration {
    /// The SQL with line endings normalised to LF. A checkout that converted
    /// them (Windows, `core.autocrlf`) must not change a migration's identity.
    fn normalised(&self) -> String {
        self.sql.replace("\r\n", "\n")
    }

    /// SHA-256 of the normalised migration before schema substitution, so
    /// every instance built from one catalog records the same checksum.
    pub fn checksum(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"ptr-pg/migration/v1");
        hasher.update(self.version.to_le_bytes());
        hasher.update(self.normalised().as_bytes());
        hasher.finalize().into()
    }

    /// The SQL with every schema placeholder substituted. `Identifier`
    /// guarantees the names need no quoting.
    pub fn render(&self, schemas: &SchemaSet) -> String {
        self.normalised()
            .replace(
                SchemaClass::Projection.placeholder(),
                schemas.projection.as_str(),
            )
            .replace(SchemaClass::Derived.placeholder(), schemas.derived.as_str())
            .replace(SchemaClass::Work.placeholder(), schemas.work.as_str())
    }
}

/// Refuse a catalog whose versions are not `1..=n` in order, or whose SQL
/// never names its own schema.
pub fn check_catalog(class: SchemaClass, catalog: &[Migration]) -> Result<(), PgError> {
    for (index, migration) in catalog.iter().enumerate() {
        if migration.version as usize != index + 1 {
            return Err(PgError::InvalidCatalog {
                message: "versions must be 1..=n without gaps, in order",
            });
        }
        if !migration.sql.contains(class.placeholder()) {
            return Err(PgError::InvalidCatalog {
                message: "every migration must address its own schema through its placeholder",
            });
        }
        if migration.name.is_empty() {
            return Err(PgError::InvalidCatalog {
                message: "every migration needs a name",
            });
        }
    }
    Ok(())
}

/// What applying a catalog to a schema with `applied` migrations must do.
///
/// Refuses when an applied migration's checksum differs from the catalog
/// (drift) or when the schema has a version the catalog lacks (it was migrated
/// by a newer build). Otherwise returns the pending migrations in order.
pub fn plan_migrations<'a>(
    class: SchemaClass,
    catalog: &'a [Migration],
    applied: &[(u32, [u8; 32])],
) -> Result<Vec<&'a Migration>, PgError> {
    check_catalog(class, catalog)?;
    for (version, checksum) in applied {
        let Some(known) = catalog.iter().find(|m| m.version == *version) else {
            return Err(PgError::UnknownMigration { version: *version });
        };
        if known.checksum() != *checksum {
            return Err(PgError::MigrationDrift { version: *version });
        }
    }
    Ok(catalog
        .iter()
        .filter(|migration| !applied.iter().any(|(v, _)| *v == migration.version))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shipped_catalog_is_well_formed_and_renders_completely() {
        let schemas = SchemaSet::with_prefix("ptr_x").unwrap();
        for class in SchemaClass::ALL {
            check_catalog(class, class.catalog()).unwrap();
            for migration in class.catalog() {
                let rendered = migration.render(&schemas);
                assert!(!rendered.contains("{{"), "{class:?} {}", migration.name);
            }
        }
    }

    #[test]
    fn a_fresh_schema_gets_every_migration_in_order() {
        let pending = plan_migrations(SchemaClass::Work, WORK_MIGRATIONS, &[]).unwrap();
        let versions: Vec<u32> = pending.iter().map(|m| m.version).collect();
        assert_eq!(
            versions,
            (1..=WORK_MIGRATIONS.len() as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn an_edited_applied_migration_is_drift() {
        let applied = [(1, [0u8; 32])];
        assert_eq!(
            plan_migrations(SchemaClass::Work, WORK_MIGRATIONS, &applied).unwrap_err(),
            PgError::MigrationDrift { version: 1 }
        );
    }

    #[test]
    fn a_migration_from_a_newer_build_is_refused() {
        let applied = [(1, WORK_MIGRATIONS[0].checksum()), (99, [0u8; 32])];
        assert_eq!(
            plan_migrations(SchemaClass::Work, WORK_MIGRATIONS, &applied).unwrap_err(),
            PgError::UnknownMigration { version: 99 }
        );
    }

    #[test]
    fn crlf_line_endings_do_not_change_a_checksum() {
        let lf = Migration {
            version: 1,
            name: "x",
            sql: "CREATE TABLE {{work}}.t (a int);\nSELECT 1;\n",
        };
        let crlf = Migration {
            sql: "CREATE TABLE {{work}}.t (a int);\r\nSELECT 1;\r\n",
            ..lf
        };
        assert_eq!(lf.checksum(), crlf.checksum());
    }

    #[test]
    fn a_catalog_with_a_gap_is_refused() {
        let broken = [WORK_MIGRATIONS[0], WORK_MIGRATIONS[2]];
        assert!(check_catalog(SchemaClass::Work, &broken).is_err());
    }
}
