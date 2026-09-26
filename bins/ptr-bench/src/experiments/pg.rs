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

/// One substrate instance: its schemas and the connection string that tags
/// its sessions with `application_name = prefix`, so a crash can target them.
pub struct Instance {
    pub prefix: String,
    pub schemas: SchemaSet,
    pub tagged_dsn: String,
}

impl Instance {
    pub fn new(tag: &str, seed: u64, case: usize) -> Self {
        let prefix = format!("{tag}_{seed}_{case}_{}", std::process::id());
        let schemas = SchemaSet::with_prefix(&prefix).expect("experiment prefix is an identifier");
        let tagged_dsn = format!("{} application_name={prefix}", dsn());
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

/// Crash the process that runs `task`: terminate its server sessions (a kill
/// the server sees mid-statement or mid-transaction) or drop the task where
/// it stands (a client crash). Every other session of the instance must
/// already be closed, as it is when the whole process dies. Returns once the
/// server has released every session of the instance, so a restarted process
/// never waits on a dead one's locks.
pub async fn crash_task<T: Send + 'static>(
    instance: &Instance,
    raw: &Client,
    task: tokio::task::JoinHandle<T>,
    kill: bool,
    delay: Duration,
) {
    tokio::time::sleep(delay).await;
    if kill {
        instance.kill_sessions(raw).await;
        // Whatever the task returns, its session is gone or suspect.
        let _ = task.await;
    } else {
        task.abort();
        let _ = task.await;
    }
    loop {
        let alive: i64 = raw
            .query_one(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name = $1",
                &[&instance.prefix],
            )
            .await
            .expect("count instance sessions")
            .get(0);
        if alive == 0 {
            return;
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

/// The server's version string, recorded with every result.
pub async fn server_version(raw: &Client) -> String {
    raw.query_one(
        "SELECT current_setting('server_version') || ' / pgvector ' || \
         coalesce((SELECT extversion FROM pg_extension WHERE extname = 'vector'), 'none')",
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
