//! Server access shared by the PostgreSQL experiment harnesses: a fresh
//! substrate instance per case, a raw session for crash injection, cleanup.

use std::time::Duration;

use ptr_pg::{PgSubstrate, SchemaSet};
use tokio_postgres::{Client, NoTls};

/// Names the environment variable holding the server's connection string. The
/// string itself never appears in a manifest or a result file.
pub const DSN_VARIABLE: &str = "PTR_PG_EXPERIMENT_DSN";

pub fn dsn() -> String {
    match std::env::var(DSN_VARIABLE) {
        Ok(dsn) if !dsn.trim().is_empty() => dsn,
        _ => {
            eprintln!(
                "{DSN_VARIABLE} must name a loopback PostgreSQL 16+ server with pgvector 0.8+, \
                 e.g. {DSN_VARIABLE}=\"host=127.0.0.1 port=5432 user=postgres\""
            );
            std::process::exit(2);
        }
    }
}

/// The message an injected commit fault raises, which the harnesses require
/// in the error an operation returns: any other error is not the fault.
pub const COMMIT_FAULT: &str = "injected commit fault";

const FAULT_TRIGGER: &str = "ptr_bench_commit_fault";

/// Whether `error` is the injected commit fault, as the server reported it.
pub fn is_commit_fault(error: &ptr_pg::PgError) -> bool {
    matches!(error, ptr_pg::PgError::Database { message, .. } if message == COMMIT_FAULT)
}

/// One substrate instance: its schemas and the connection string that tags
/// its sessions with `application_name = prefix`, so a crash can target them.
pub struct Instance {
    pub prefix: String,
    pub schemas: SchemaSet,
    pub tagged_dsn: String,
}

impl Instance {
    pub fn new(tag: &str, seed: u64, case: usize) -> Self {
        // A process runs one seed, and the process id keeps concurrent runs
        // apart, so the seed's low digits suffice; the prefix stays within
        // the identifier bound for any seed and case count.
        let prefix = format!("{tag}_{}_{case}_{}", seed % 1_000_000, std::process::id());
        let schemas = SchemaSet::with_prefix(&prefix).expect("experiment prefix is an identifier");
        let tagged_dsn = tag_dsn(&dsn(), &prefix);
        Self {
            prefix,
            schemas,
            tagged_dsn,
        }
    }

    /// Drop anything a previous run left under this prefix, then connect and
    /// migrate.
    pub async fn create(&self, raw: &Client) -> PgSubstrate {
        let prefix = &self.prefix;
        raw.batch_execute(&format!(
            "DROP SCHEMA IF EXISTS {prefix}_derived CASCADE; \
             DROP SCHEMA IF EXISTS {prefix}_projection CASCADE; \
             DROP SCHEMA IF EXISTS {prefix}_work CASCADE;"
        ))
        .await
        .expect("drop leftover experiment schemas");
        let mut substrate = self.connect().await;
        substrate
            .migrate()
            .await
            .expect("migrate experiment instance");
        substrate
    }

    /// A new session on the same schemas, as a restarted process would open.
    pub async fn connect(&self) -> PgSubstrate {
        PgSubstrate::connect_with(&self.tagged_dsn, self.schemas.clone())
            .await
            .expect("connect experiment instance")
    }

    /// Make every transaction that inserts or updates a row of `table`, in the
    /// instance's `class` schema (`projection` or `work`), fail at COMMIT
    /// after all of its statements succeeded, the way a deferred constraint or
    /// any other error raised at commit does. A deferred constraint trigger
    /// raises [`COMMIT_FAULT`]; its function lives in the work schema, so
    /// dropping the instance removes it. Armed and disarmed from the raw
    /// session between operations, so no other session holds the table.
    pub async fn arm_commit_fault(&self, raw: &Client, class: &str, table: &str) {
        let prefix = &self.prefix;
        raw.batch_execute(&format!(
            "CREATE OR REPLACE FUNCTION {prefix}_work.{FAULT_TRIGGER}() RETURNS trigger \
             LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION '{COMMIT_FAULT}'; END $$; \
             DROP TRIGGER IF EXISTS {FAULT_TRIGGER} ON {prefix}_{class}.{table}; \
             CREATE CONSTRAINT TRIGGER {FAULT_TRIGGER} \
             AFTER INSERT OR UPDATE ON {prefix}_{class}.{table} \
             DEFERRABLE INITIALLY DEFERRED FOR EACH ROW \
             EXECUTE FUNCTION {prefix}_work.{FAULT_TRIGGER}();"
        ))
        .await
        .expect("arm the commit fault");
    }

    pub async fn disarm_commit_fault(&self, raw: &Client, class: &str, table: &str) {
        let prefix = &self.prefix;
        raw.batch_execute(&format!(
            "DROP TRIGGER IF EXISTS {FAULT_TRIGGER} ON {prefix}_{class}.{table};"
        ))
        .await
        .expect("disarm the commit fault");
    }

    /// Terminate every server session of this instance: a crash of the
    /// substrate's process as the server sees it, mid-statement or
    /// mid-transaction. Returns how many sessions were terminated.
    pub async fn kill_sessions(&self, raw: &Client) -> u64 {
        raw.query(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
             WHERE application_name = $1 AND pid <> pg_backend_pid()",
            &[&self.prefix],
        )
        .await
        .expect("terminate instance sessions")
        .len() as u64
    }
}

/// Tag every session a connection string opens with `application_name`, in
/// either form libpq accepts: key/value pairs or a URL.
pub fn tag_dsn(dsn: &str, application_name: &str) -> String {
    let trimmed = dsn.trim();
    if trimmed.starts_with("postgres://") || trimmed.starts_with("postgresql://") {
        let separator = if trimmed.contains('?') { '&' } else { '?' };
        format!("{trimmed}{separator}application_name={application_name}")
    } else {
        format!("{trimmed} application_name={application_name}")
    }
}

/// How long a crashed instance's sessions may take to disappear.
const SESSION_DEADLINE: Duration = Duration::from_secs(60);

/// Crash the process that runs `task`: terminate its server sessions (a kill
/// the server sees mid-statement or mid-transaction) or drop the task where
/// it stands (a client crash). Every other session of the instance must
/// already be closed, as it is when the whole process dies.
///
/// Returns what the task returned if it finished before the crash reached it
/// (`None` when it was dropped first), so the caller can hold an
/// acknowledgement to what the server kept: an operation reported done must
/// be found committed. Returns an error if the server has not released every
/// session of the instance within a minute, instead of waiting forever.
pub async fn crash_task<T: Send + 'static>(
    instance: &Instance,
    raw: &Client,
    task: tokio::task::JoinHandle<T>,
    kill: bool,
    delay: Duration,
) -> Result<Option<T>, String> {
    tokio::time::sleep(delay).await;
    let returned = if kill {
        instance.kill_sessions(raw).await;
        task.await.ok()
    } else {
        task.abort();
        task.await.ok()
    };
    let started = std::time::Instant::now();
    loop {
        let alive: i64 = raw
            .query_one(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name = $1",
                &[&instance.prefix],
            )
            .await
            .map_err(|error| format!("counting sessions failed: {error}"))?
            .get(0);
        if alive == 0 {
            return Ok(returned);
        }
        if started.elapsed() > SESSION_DEADLINE {
            return Err(format!(
                "{alive} sessions of {} outlived the crash by a minute",
                instance.prefix
            ));
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

/// A raw session, with pgvector created once under a transaction lock (the
/// same guard the integration tests use).
pub async fn raw_client() -> Client {
    let (client, connection) = tokio_postgres::connect(&dsn(), NoTls)
        .await
        .expect("connect raw experiment session");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .batch_execute(
            "BEGIN; \
             SELECT pg_advisory_xact_lock(7402301); \
             CREATE EXTENSION IF NOT EXISTS vector; \
             COMMIT;",
        )
        .await
        .expect("pgvector must be installed or installable");
    client
}

/// The server's version and the settings a timing or a durability claim
/// depends on, recorded with every result: a run against a server with
/// `fsync` off measures no flush cost. A session kill or a dropped client
/// cannot lose a committed transaction whatever these say.
pub async fn server_version(raw: &Client) -> String {
    raw.query_one(
        "SELECT current_setting('server_version') || ' / pgvector ' || \
         coalesce((SELECT extversion FROM pg_extension WHERE extname = 'vector'), 'none') || \
         ' / fsync=' || current_setting('fsync') || \
         ' synchronous_commit=' || current_setting('synchronous_commit') || \
         ' full_page_writes=' || current_setting('full_page_writes')",
        &[],
    )
    .await
    .expect("read server version")
    .get(0)
}

/// Escape a string for a JSON value.
pub fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_connection_string_forms_are_tagged() {
        assert_eq!(
            tag_dsn("host=127.0.0.1 user=postgres", "l004_17_0_9"),
            "host=127.0.0.1 user=postgres application_name=l004_17_0_9"
        );
        assert_eq!(
            tag_dsn("postgres://postgres@127.0.0.1/db", "x"),
            "postgres://postgres@127.0.0.1/db?application_name=x"
        );
        assert_eq!(
            tag_dsn("postgresql://127.0.0.1/db?sslmode=disable", "x"),
            "postgresql://127.0.0.1/db?sslmode=disable&application_name=x"
        );
    }

    #[test]
    fn a_prefix_fits_the_identifier_bound_for_any_seed() {
        let instance_prefix = format!("l004b_{}_{}_{}", u64::MAX % 1_000_000, 99_999, 4_194_304);
        assert!(
            SchemaSet::with_prefix(&instance_prefix).is_ok(),
            "{instance_prefix}"
        );
    }
}
