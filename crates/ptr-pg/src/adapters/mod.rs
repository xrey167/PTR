//! Async adapters over `tokio-postgres`. Driver types never leave this module:
//! every public signature uses PTR types, and every driver error becomes a
//! [`PgError`].

mod branch;
mod events;
mod fastmem;
mod lineage;
mod metrics;
mod policy;
mod projection;
mod search;

use tokio_postgres::config::Host;
use tokio_postgres::{Client, Config, IsolationLevel, NoTls, Transaction};

use crate::capability::Capabilities;
use crate::config::{PgConfig, SchemaSet};
use crate::error::PgError;
use crate::migrate::{plan_migrations, SchemaClass};

pub use branch::BranchOutcome;
pub use events::ProjectionEventRow;
pub use fastmem::{FastMemoryCheckpoint, FastMemoryRecord};
pub use projection::ProjectionApply;
pub use search::{
    EmbeddingSpace, HybridQuery, SearchDocument, SearchResults, LEXICAL_BACKEND, VECTOR_BACKEND,
};

/// A connection to one substrate instance (one [`SchemaSet`]).
pub struct PgSubstrate {
    client: Client,
    schemas: SchemaSet,
    connection: tokio::task::JoinHandle<()>,
}

/// What a migration run applied, per schema class.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MigrationReport {
    pub projection: Vec<u32>,
    pub derived: Vec<u32>,
    pub work: Vec<u32>,
}

impl PgSubstrate {
    /// Connect with the connection string named by `config`.
    pub async fn connect(config: &PgConfig) -> Result<Self, PgError> {
        let dsn = config.resolve_dsn()?;
        Self::connect_with(&dsn, config.schemas.clone()).await
    }

    /// Connect with an explicit connection string.
    ///
    /// This build links no TLS connector, so it refuses any target that is not
    /// a loopback address or a Unix socket rather than send credentials in the
    /// clear. When `hostaddr` is given the driver connects to it and uses
    /// `host` only as a name, so every `hostaddr` must be a loopback address
    /// as well. Must be called inside a Tokio runtime.
    pub async fn connect_with(dsn: &str, schemas: SchemaSet) -> Result<Self, PgError> {
        let config: Config =
            dsn.parse()
                .map_err(|error: tokio_postgres::Error| PgError::Connection {
                    message: error.to_string(),
                })?;
        check_targets(&config)?;
        let (client, connection) = config.connect(NoTls).await.map_err(database)?;
        let connection = tokio::spawn(async move {
            // A closed connection surfaces as an error on the next statement.
            let _ = connection.await;
        });
        Ok(Self {
            client,
            schemas,
            connection,
        })
    }

    pub fn schemas(&self) -> &SchemaSet {
        &self.schemas
    }

    /// What this server offers. Never creates an extension: installing one is
    /// an operator's decision (and usually needs a superuser).
    pub async fn capabilities(&self) -> Result<Capabilities, PgError> {
        let version: i32 = self
            .client
            .query_one("SELECT current_setting('server_version_num')::int", &[])
            .await
            .map_err(database)?
            .get(0);
        let rows = self
            .client
            .query("SELECT extname, extversion FROM pg_extension", &[])
            .await
            .map_err(database)?;
        let installed: Vec<(String, String)> =
            rows.iter().map(|row| (row.get(0), row.get(1))).collect();
        Ok(Capabilities::from_catalog(
            u32::try_from(version).map_err(|_| PgError::OutOfRange {
                field: "server_version_num",
            })?,
            installed.iter().map(|(n, v)| (n.as_str(), v.as_str())),
        ))
    }

    /// Create the three schemas if needed and apply every pending migration.
    ///
    /// Runs under a session advisory lock keyed by the schema prefix, so two
    /// processes migrating one instance serialize. Refuses a server without the
    /// capabilities the schema needs, an applied migration whose checksum
    /// changed, and a schema migrated by a newer build.
    pub async fn migrate(&mut self) -> Result<MigrationReport, PgError> {
        self.capabilities().await?.check_supported()?;
        self.lock_migrations().await?;
        let result = self.migrate_locked().await;
        let unlock = self.unlock_migrations().await;
        let report = result?;
        unlock?;
        Ok(report)
    }

    /// Drop the projection and the derived caches and recreate them empty.
    /// Working state is untouched. The caller replays the ledger afterwards.
    ///
    /// The drop and the recreation run under the migration lock, so another
    /// migrator cannot interleave, and a lock wait that fails leaves the
    /// schemas as they were rather than dropped.
    pub async fn rebuild_projection(&mut self) -> Result<MigrationReport, PgError> {
        self.capabilities().await?.check_supported()?;
        self.lock_migrations().await?;
        let result = self.rebuild_locked().await;
        let unlock = self.unlock_migrations().await;
        let report = result?;
        unlock?;
        Ok(report)
    }

    /// Drop all three schemas. For tests and decommissioning only: working
    /// state is not recoverable from the ledger.
    pub async fn drop_all(self) -> Result<(), PgError> {
        let SchemaSet {
            projection,
            derived,
            work,
        } = self.schemas.clone();
        self.client
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS {derived} CASCADE; \
                 DROP SCHEMA IF EXISTS {projection} CASCADE; \
                 DROP SCHEMA IF EXISTS {work} CASCADE;"
            ))
            .await
            .map_err(database)?;
        drop(self.client);
        let _ = self.connection.await;
        Ok(())
    }

    /// A read-committed transaction, whatever the session's default.
    ///
    /// The row-lock ordering between the projector and every writer of derived
    /// or working rows relies on each statement taking a fresh snapshot after
    /// the locks it waited for were released. Under repeatable read a
    /// projector's delete would miss a row committed while it waited, so the
    /// level is pinned here rather than inherited from
    /// `default_transaction_isolation`.
    pub(crate) async fn read_committed(&mut self) -> Result<Transaction<'_>, PgError> {
        self.client
            .build_transaction()
            .isolation_level(IsolationLevel::ReadCommitted)
            .start()
            .await
            .map_err(database)
    }

    /// The session advisory lock every schema change of this instance takes.
    /// Its wait has no timeout: a second migrator waits for the first.
    async fn lock_migrations(&self) -> Result<(), PgError> {
        let lock_key = format!("ptr-pg:{}", self.schemas.projection);
        self.client
            .execute("SELECT pg_advisory_lock(hashtext($1))", &[&lock_key])
            .await
            .map_err(database)?;
        Ok(())
    }

    async fn unlock_migrations(&self) -> Result<(), PgError> {
        let lock_key = format!("ptr-pg:{}", self.schemas.projection);
        self.client
            .execute("SELECT pg_advisory_unlock(hashtext($1))", &[&lock_key])
            .await
            .map_err(database)?;
        Ok(())
    }

    async fn rebuild_locked(&mut self) -> Result<MigrationReport, PgError> {
        let projection = self.schemas.projection.clone();
        let derived = self.schemas.derived.clone();
        let transaction = self.client.transaction().await.map_err(database)?;
        transaction
            .batch_execute(&format!(
                "SET LOCAL lock_timeout = '10s'; \
                 DROP SCHEMA IF EXISTS {derived} CASCADE; \
                 DROP SCHEMA IF EXISTS {projection} CASCADE;"
            ))
            .await
            .map_err(database)?;
        transaction.commit().await.map_err(database)?;
        self.migrate_locked().await
    }

    async fn migrate_locked(&mut self) -> Result<MigrationReport, PgError> {
        let mut report = MigrationReport::default();
        for class in SchemaClass::ALL {
            let schema = self.schema_of(class).to_owned();
            // SET LOCAL: the timeout bounds this transaction's DDL only and
            // never leaks into the session the projector uses afterwards.
            let transaction = self.client.transaction().await.map_err(database)?;
            transaction
                .batch_execute(&format!(
                    "SET LOCAL lock_timeout = '10s';
                     CREATE SCHEMA IF NOT EXISTS {schema};
                     CREATE TABLE IF NOT EXISTS {schema}.schema_migration (
                         version integer PRIMARY KEY,
                         name text NOT NULL,
                         checksum bytea NOT NULL,
                         applied_at timestamptz NOT NULL DEFAULT now()
                     );"
                ))
                .await
                .map_err(database)?;
            transaction.commit().await.map_err(database)?;
            let rows = self
                .client
                .query(
                    &format!("SELECT version, checksum FROM {schema}.schema_migration"),
                    &[],
                )
                .await
                .map_err(database)?;
            let mut applied = Vec::with_capacity(rows.len());
            for row in rows {
                let version: i32 = row.get(0);
                let checksum: Vec<u8> = row.get(1);
                let checksum: [u8; 32] = checksum.try_into().map_err(|_| PgError::CorruptRow {
                    table: "schema_migration",
                    reason: "checksum is not 32 bytes".into(),
                })?;
                applied.push((version as u32, checksum));
            }
            let pending = plan_migrations(class, class.catalog(), &applied)?;
            for migration in pending {
                let transaction = self.client.transaction().await.map_err(database)?;
                transaction
                    .batch_execute("SET LOCAL lock_timeout = '10s'")
                    .await
                    .map_err(database)?;
                transaction
                    .batch_execute(&migration.render(&self.schemas))
                    .await
                    .map_err(database)?;
                transaction
                    .execute(
                        &format!(
                            "INSERT INTO {schema}.schema_migration (version, name, checksum) \
                             VALUES ($1, $2, $3)"
                        ),
                        &[
                            &(migration.version as i32),
                            &migration.name,
                            &migration.checksum().to_vec(),
                        ],
                    )
                    .await
                    .map_err(database)?;
                transaction.commit().await.map_err(database)?;
                match class {
                    SchemaClass::Projection => report.projection.push(migration.version),
                    SchemaClass::Derived => report.derived.push(migration.version),
                    SchemaClass::Work => report.work.push(migration.version),
                }
            }
        }
        Ok(report)
    }

    fn schema_of(&self, class: SchemaClass) -> &str {
        match class {
            SchemaClass::Projection => self.schemas.projection.as_str(),
            SchemaClass::Derived => self.schemas.derived.as_str(),
            SchemaClass::Work => self.schemas.work.as_str(),
        }
    }
}

/// Refuse a connection target that is not local: every `hostaddr` must be a
/// loopback address, and every `host` a Unix socket, `localhost` or a
/// loopback address.
fn check_targets(config: &Config) -> Result<(), PgError> {
    for address in config.get_hostaddrs() {
        if !address.is_loopback() {
            return Err(PgError::TlsRequired {
                host: address.to_string(),
            });
        }
    }
    for host in config.get_hosts() {
        check_loopback(host)?;
    }
    Ok(())
}

fn check_loopback(host: &Host) -> Result<(), PgError> {
    match host {
        #[cfg(unix)]
        Host::Unix(_) => Ok(()),
        Host::Tcp(name) => {
            let loopback = name == "localhost"
                || name
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.is_loopback());
            if loopback {
                Ok(())
            } else {
                Err(PgError::TlsRequired { host: name.clone() })
            }
        }
    }
}

/// Convert a driver error into a typed refusal carrying only SQLSTATE and
/// message.
pub(crate) fn database(error: tokio_postgres::Error) -> PgError {
    match error.as_db_error() {
        Some(db) => PgError::Database {
            sqlstate: db.code().code().to_owned(),
            message: db.message().to_owned(),
        },
        None if error.is_closed() => PgError::Connection {
            message: error.to_string(),
        },
        None => PgError::Connection {
            message: error.to_string(),
        },
    }
}

/// Refuse a string a PostgreSQL `text` column cannot hold.
pub(crate) fn check_text(field: &'static str, value: &str) -> Result<(), PgError> {
    if value.contains('\0') {
        return Err(PgError::InvalidText { field });
    }
    Ok(())
}

/// `u64` to a signed column, refusing rather than wrapping.
pub(crate) fn to_i64(value: u64, field: &'static str) -> Result<i64, PgError> {
    i64::try_from(value).map_err(|_| PgError::OutOfRange { field })
}

/// A signed column back to `u64`, refusing a negative value.
pub(crate) fn to_u64(value: i64, table: &'static str) -> Result<u64, PgError> {
    u64::try_from(value).map_err(|_| PgError::CorruptRow {
        table,
        reason: format!("negative value {value}"),
    })
}

pub(crate) fn digest_from(bytes: Vec<u8>, table: &'static str) -> Result<[u8; 32], PgError> {
    bytes.try_into().map_err(|_| PgError::CorruptRow {
        table,
        reason: "digest is not 32 bytes".into(),
    })
}
