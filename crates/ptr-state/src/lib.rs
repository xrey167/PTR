use ptr_ledger::{CommittedEvent, LedgerEvent};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplyOutcome {
    Applied,
    Duplicate,
    OutOfOrder,
    Gap,
}

#[derive(Clone, Debug, Default)]
pub struct MaterializedState {
    pub values: BTreeMap<String, String>,
    pub last_applied: u64,
}

impl MaterializedState {
    pub fn try_apply(&mut self, committed: &CommittedEvent) -> ApplyOutcome {
        let index = committed.index.0;
        if index == self.last_applied {
            return ApplyOutcome::Duplicate;
        }
        if index < self.last_applied {
            return ApplyOutcome::OutOfOrder;
        }
        if index != self.last_applied.saturating_add(1) {
            return ApplyOutcome::Gap;
        }

        for (key, value) in materialized_entries(&committed.event) {
            self.values.insert(key, value);
        }
        self.last_applied = index;
        ApplyOutcome::Applied
    }

    pub fn apply(&mut self, committed: &CommittedEvent) {
        let _ = self.try_apply(committed);
    }
}

fn materialized_entries(event: &LedgerEvent) -> Vec<(String, String)> {
    match event {
        // Lifecycle materialization records the position, not a second copy of
        // semantic payloads. Their authoritative replay belongs to ptr-semdb.
        LedgerEvent::SemanticDeltaCommitted { revision, .. } => {
            vec![("semdb:revision".into(), revision.0.to_string())]
        }
        LedgerEvent::CapsuleCommitted {
            project,
            capsule,
            generation,
        } => vec![
            (
                format!("capsule:{capsule}:generation"),
                generation.0.to_string(),
            ),
            (format!("capsule:{capsule}:project"), project.to_string()),
        ],
        LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
            vec![(format!("capsule:{capsule}:generation"), new.0.to_string())]
        }
        LedgerEvent::Revoked {
            subject,
            generation,
        } => vec![(format!("revoked:{subject}"), generation.0.to_string())],
        LedgerEvent::HardConstraintCommitted { key, generation } => {
            vec![(format!("constraint:{key}"), generation.0.to_string())]
        }
        LedgerEvent::VerifierAttested { subject, passed } => vec![(
            format!("verifier:{subject}"),
            if *passed { "pass" } else { "fail" }.to_owned(),
        )],
        LedgerEvent::ProcedurePromoted { id, generation } => vec![(
            format!("procedure:{id}:generation"),
            generation.0.to_string(),
        )],
        LedgerEvent::ProcedureRevoked { id, generation } => {
            vec![(format!("revoked:procedure:{id}"), generation.0.to_string())]
        }
        LedgerEvent::SnapshotCommitted { revision, covers } => vec![
            ("snapshot:last_revision".into(), revision.to_string()),
            ("snapshot:covers_commit".into(), covers.0.to_string()),
        ],
    }
}

#[cfg(feature = "turso-backend")]
pub struct TursoMaterializedState {
    _database: turso::Database,
    connection: turso::Connection,
    last_applied: u64,
}

#[cfg(feature = "turso-backend")]
impl TursoMaterializedState {
    pub async fn open(path: &str) -> Result<Self, String> {
        let database = turso::Builder::new_local(path)
            .build()
            .await
            .map_err(|error| error.to_string())?;
        let connection = database.connect().map_err(|error| error.to_string())?;
        connection
            .execute_batch(
                "
                CREATE TABLE IF NOT EXISTS ptr_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    commit_index INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ptr_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_applied INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO ptr_meta(id, last_applied) VALUES (1, 0);
                ",
            )
            .await
            .map_err(|error| error.to_string())?;

        let mut rows = connection
            .query("SELECT last_applied FROM ptr_meta WHERE id = 1", ())
            .await
            .map_err(|error| error.to_string())?;
        let last_applied = match rows.next().await.map_err(|error| error.to_string())? {
            Some(row) => {
                let value: i64 = row.get(0).map_err(|error| error.to_string())?;
                u64::try_from(value)
                    .map_err(|_| "negative last_applied in Turso state".to_owned())?
            }
            None => 0,
        };

        Ok(Self {
            _database: database,
            connection,
            last_applied,
        })
    }

    pub fn last_applied(&self) -> u64 {
        self.last_applied
    }

    pub async fn get(&self, key: &str) -> Result<Option<String>, String> {
        let mut rows = self
            .connection
            .query("SELECT value FROM ptr_state WHERE key = ?1", (key,))
            .await
            .map_err(|error| error.to_string())?;
        match rows.next().await.map_err(|error| error.to_string())? {
            Some(row) => row
                .get::<String>(0)
                .map(Some)
                .map_err(|error| error.to_string()),
            None => Ok(None),
        }
    }

    pub async fn try_apply(&mut self, committed: &CommittedEvent) -> Result<ApplyOutcome, String> {
        let index = committed.index.0;
        if index == self.last_applied {
            return Ok(ApplyOutcome::Duplicate);
        }
        if index < self.last_applied {
            return Ok(ApplyOutcome::OutOfOrder);
        }
        if index != self.last_applied.saturating_add(1) {
            return Ok(ApplyOutcome::Gap);
        }

        let tx = self
            .connection
            .transaction()
            .await
            .map_err(|error| error.to_string())?;

        for (key, value) in materialized_entries(&committed.event) {
            tx.execute(
                "
                INSERT INTO ptr_state(key, value, commit_index)
                VALUES (?1, ?2, ?3)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    commit_index = excluded.commit_index
                ",
                (key, value, index as i64),
            )
            .await
            .map_err(|error| error.to_string())?;
        }

        tx.execute(
            "UPDATE ptr_meta SET last_applied = ?1 WHERE id = 1",
            (index as i64,),
        )
        .await
        .map_err(|error| error.to_string())?;
        tx.commit().await.map_err(|error| error.to_string())?;
        self.last_applied = index;
        Ok(ApplyOutcome::Applied)
    }
}
