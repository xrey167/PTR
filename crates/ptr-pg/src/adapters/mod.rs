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

use std::time::Duration;

use tokio::task::JoinHandle;
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

/// A connection to one substrate instance (one [`SchemaSet`], the three
/// schemas of one prefix: [`connect_with`](Self::connect_with) refuses any
/// other set).
pub struct PgSubstrate {
    client: Client,
    schemas: SchemaSet,
    connection: JoinHandle<()>,
    /// The configuration `client` was opened with, already checked by
    /// [`check_targets`]. Schema changes open their lock session from it.
    config: Config,
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
    /// `schemas` must be [`SchemaSet::with_prefix`] of some prefix, or
    /// [`PgError::InvalidSchemaSet`] is returned before anything is parsed
    /// or opened ([`SchemaSet::check`]). Every schema this substrate
    /// migrates, rebuilds, drops or writes is therefore its own: no other
    /// instance shares one, and the migration lock, keyed by the projection
    /// schema's name, is the lock of all three.
    ///
    /// This build links no TLS connector, so it refuses any target that is not
    /// a loopback address or a Unix socket rather than send credentials in the
    /// clear. When `hostaddr` is given the driver connects to it and uses
    /// `host` only as a name, so every `hostaddr` must be a loopback address
    /// as well. Must be called inside a Tokio runtime.
    pub async fn connect_with(dsn: &str, schemas: SchemaSet) -> Result<Self, PgError> {
        schemas.check()?;
        let config: Config =
            dsn.parse()
                .map_err(|error: tokio_postgres::Error| PgError::Connection {
                    message: error.to_string(),
                })?;
        let (client, connection) = open_session(&config).await?;
        Ok(Self {
            client,
            schemas,
            connection,
            config,
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
    /// Runs under a session advisory lock keyed by the projection schema's
    /// name, `<prefix>_projection`. Since the set is one prefix's, that name
    /// determines all three schemas, so two processes migrating one instance
    /// serialize and no other instance's migration touches these schemas.
    /// The lock is held by a
    /// session of its own, opened to the same checked target: a migration
    /// whose future is cancelled or unwinds releases it when that session
    /// closes, while this substrate's session stays open, and a retry takes it
    /// afresh. A migration cancelled while it waits for the lock leaves no
    /// session behind either: the lock is taken with `pg_try_advisory_lock`,
    /// retried while another session holds it, so no statement of it waits on
    /// the lock. The migrations themselves run on this substrate's session,
    /// each in its own transaction, so one in flight at a cancellation either
    /// commits with its `schema_migration` row or rolls back; and each commits
    /// only after the lock's session has confirmed that it still holds the
    /// lock, so a lock lost mid-migration rolls that migration back with
    /// [`PgError::MigrationLockLost`] instead of letting a second migrator run
    /// beside it. Refuses a server without the capabilities the schema needs,
    /// an applied migration whose checksum changed, and a schema migrated by a
    /// newer build.
    pub async fn migrate(&mut self) -> Result<MigrationReport, PgError> {
        self.capabilities().await?.check_supported()?;
        let lock = MigrationLock::acquire(&self.config, &self.schemas).await?;
        let result = self.migrate_locked(&lock).await;
        let release = lock.release().await;
        let report = result?;
        release?;
        Ok(report)
    }

    /// Drop the projection and the derived caches and recreate them empty.
    /// Working state is untouched: the two schemas dropped are this
    /// instance's own, which no other instance shares (see
    /// [`connect_with`](Self::connect_with)). The caller replays the ledger
    /// afterwards.
    ///
    /// The drop and the recreation run under the migration lock, so another
    /// migrator cannot interleave, and a lock wait that fails leaves the
    /// schemas as they were rather than dropped. The lock is released as for
    /// [`migrate`](Self::migrate) when the rebuild's future is cancelled or
    /// unwinds: a drop that has not committed is rolled back with its
    /// transaction, and schemas dropped before the cancellation are recreated
    /// by the next migration. The drop, like every migration, commits only
    /// while the lock is confirmed held, and is rolled back with
    /// [`PgError::MigrationLockLost`] otherwise.
    pub async fn rebuild_projection(&mut self) -> Result<MigrationReport, PgError> {
        self.capabilities().await?.check_supported()?;
        let lock = MigrationLock::acquire(&self.config, &self.schemas).await?;
        let result = self.rebuild_locked(&lock).await;
        let release = lock.release().await;
        let report = result?;
        release?;
        Ok(report)
    }

    /// Drop all three schemas. For tests and decommissioning only: working
    /// state is not recoverable from the ledger.
    ///
    /// Takes no migration lock, so it must not run while any other session
    /// migrates, rebuilds or uses this instance; the caller serializes it.
    /// It drops exactly this instance's three schemas, which no other
    /// instance shares (see [`connect_with`](Self::connect_with)).
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

    async fn rebuild_locked(&mut self, lock: &MigrationLock) -> Result<MigrationReport, PgError> {
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
        lock.confirm().await?;
        transaction.commit().await.map_err(database)?;
        self.migrate_locked(lock).await
    }

    /// Every transaction here confirms the lock right before it commits; an
    /// error drops the open transaction, which rolls it back.
    async fn migrate_locked(&mut self, lock: &MigrationLock) -> Result<MigrationReport, PgError> {
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
            lock.confirm().await?;
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
                lock.confirm().await?;
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

/// How long a migration waits between two attempts to take the migration
/// lock while another session holds it.
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(100);

/// The session advisory lock every schema change of an instance takes, held
/// by a session of its own.
///
/// PostgreSQL releases a session lock when its session ends. Dropping the
/// guard drops that session's client, which ends its connection task and with
/// it the server session, so a schema change whose future is cancelled or
/// unwinds after taking the lock releases it as soon as the session closes,
/// never leaving it with the substrate's session. Every acquisition opens a
/// fresh session, so a retry never re-enters a lock an earlier attempt still
/// holds, and one unlock always releases it. The key is `ptr-pg:` and the
/// projection schema's name, the one earlier builds took on the substrate's
/// session, so they serialize with this one; since a substrate's schemas are
/// one prefix's, the key covers all three.
///
/// The session runs nothing while the schema change does, so it is kept
/// from ending for being idle, and the change confirms on it that the lock is
/// still held before each commit (see [`Self::confirm`]).
struct MigrationLock {
    client: Client,
    connection: JoinHandle<()>,
    key: String,
}

impl MigrationLock {
    /// Open a session to the target of `config`, which [`open_session`]
    /// checks as it checks every other, and take the lock of `schemas` there.
    /// The wait has no timeout: a second migrator waits for the first.
    ///
    /// The lock is taken with `pg_try_advisory_lock`, retried every
    /// [`LOCK_RETRY_INTERVAL`] while another session holds it, rather than
    /// waited for with `pg_advisory_lock`. A future dropped while it waits
    /// (a caller's timeout, an aborted task) thus has no statement
    /// outstanding, or only a try that answers at once, so its connection
    /// task ends the session as soon as the client is dropped. A blocking
    /// wait would keep the connection open until its answer arrived, and the
    /// server does not notice a closed client while a backend waits on a
    /// lock: every cancelled attempt would leave a session queued on the lock
    /// until the holder released it.
    ///
    /// Before taking the lock the session sets `idle_session_timeout` to
    /// zero, whatever the connection string or the server configures: it is
    /// idle while the schema change runs on the substrate's session, and a
    /// session ended for being idle would take the lock with it.
    async fn acquire(config: &Config, schemas: &SchemaSet) -> Result<Self, PgError> {
        let (client, connection) = open_session(config).await?;
        client
            .batch_execute("SET idle_session_timeout = 0")
            .await
            .map_err(database)?;
        let key = format!("ptr-pg:{}", schemas.projection);
        loop {
            let taken: bool = client
                .query_one("SELECT pg_try_advisory_lock(hashtext($1))", &[&key])
                .await
                .map_err(database)?
                .get(0);
            if taken {
                break;
            }
            tokio::time::sleep(LOCK_RETRY_INTERVAL).await;
        }
        Ok(Self {
            client,
            connection,
            key,
        })
    }

    /// Confirm on the lock's session that it still holds the lock, right
    /// before a schema change commits. A session that was terminated or lost
    /// its connection no longer holds it, and another migrator may already
    /// have taken it, so the change must roll back rather than commit:
    /// refuses with [`PgError::MigrationLockLost`] whenever the lock cannot be
    /// confirmed. A session ended between this answer and the commit, a
    /// window of one round trip, still lets that one change commit; the next
    /// confirmation then refuses.
    async fn confirm(&self) -> Result<(), PgError> {
        let held = self
            .client
            .query_one(
                "SELECT EXISTS (SELECT 1 FROM pg_locks \
                 WHERE locktype = 'advisory' AND pid = pg_backend_pid() AND granted \
                   AND objsubid = 1 \
                   AND database = (SELECT oid FROM pg_database \
                                   WHERE datname = current_database()) \
                   AND ((classid::bigint << 32) | objid::bigint) = hashtext($1)::bigint)",
                &[&self.key],
            )
            .await
            .map(|row| row.get::<_, bool>(0));
        match held {
            Ok(true) => Ok(()),
            Ok(false) | Err(_) => Err(PgError::MigrationLockLost),
        }
    }

    /// Unlock explicitly, then close the session and wait for its connection
    /// to end. The session closes even when the unlock fails, which releases
    /// the lock all the same.
    async fn release(self) -> Result<(), PgError> {
        let unlocked = self
            .client
            .execute("SELECT pg_advisory_unlock(hashtext($1))", &[&self.key])
            .await
            .map_err(database);
        drop(self.client);
        let _ = self.connection.await;
        unlocked.map(|_| ())
    }
}

/// Open a session to a target [`check_targets`] accepts, its connection driven
/// by a task of its own. Every session the substrate opens goes through here,
/// so none of them reaches a non-loopback target.
async fn open_session(config: &Config) -> Result<(Client, JoinHandle<()>), PgError> {
    check_targets(config)?;
    let (client, connection) = config.connect(NoTls).await.map_err(database)?;
    let connection = tokio::spawn(async move {
        // A closed connection surfaces as an error on the next statement.
        let _ = connection.await;
    });
    Ok((client, connection))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_migration_lock_session_is_refused_a_target_that_is_not_loopback() {
        let schemas = SchemaSet::with_prefix("ptr").unwrap();
        for (dsn, host) in [
            ("host=db.example.com user=ptr", "db.example.com"),
            ("host=localhost hostaddr=192.0.2.2 user=ptr", "192.0.2.2"),
        ] {
            let config: Config = dsn.parse().unwrap();
            let refused = MigrationLock::acquire(&config, &schemas)
                .await
                .err()
                .expect("a remote lock session must be refused");
            assert_eq!(refused, PgError::TlsRequired { host: host.into() }, "{dsn}");
        }
    }
}
