//! The projection event log: what each applied commit projected, readable in
//! commit order by consumers that keep their own committed offset.

use ptr_types::CommitIndex;

use super::{check_text, database, to_i64, to_u64, PgSubstrate};
use crate::error::PgError;

/// One row of the projection event log.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionEventRow {
    pub commit_index: CommitIndex,
    pub ordinal: u16,
    pub topic: String,
    pub subject: String,
    /// The commit index and the key/value entries the commit projected.
    pub payload: serde_json::Value,
}

impl PgSubstrate {
    /// Events of commits after `offset`, in commit order, up to the watermark.
    ///
    /// The projector writes a commit's events in the transaction that advances
    /// the watermark and applies commit `n + 1` only after `n` has committed,
    /// so a consumer that resumes from the last index it processed never skips
    /// a row.
    pub async fn events_after(
        &self,
        offset: CommitIndex,
        limit: u32,
    ) -> Result<Vec<ProjectionEventRow>, PgError> {
        let projection = &self.schemas.projection;
        let rows = self
            .client
            .query(
                &format!(
                    "SELECT e.commit_index, e.ordinal, e.topic, e.subject, e.payload::text \
                     FROM {projection}.projection_event e \
                     JOIN {projection}.projection_watermark w ON w.id = 1 \
                     WHERE e.commit_index > $1 AND e.commit_index <= w.last_applied \
                     ORDER BY e.commit_index, e.ordinal LIMIT $2"
                ),
                &[&to_i64(offset.0, "offset")?, &i64::from(limit)],
            )
            .await
            .map_err(database)?;
        rows.into_iter()
            .map(|row| {
                let ordinal: i16 = row.get(1);
                let payload: String = row.get(4);
                Ok(ProjectionEventRow {
                    commit_index: CommitIndex(to_u64(row.get(0), "projection_event")?),
                    ordinal: u16::try_from(ordinal).map_err(|_| PgError::CorruptRow {
                        table: "projection_event",
                        reason: format!("negative ordinal {ordinal}"),
                    })?,
                    topic: row.get(2),
                    subject: row.get(3),
                    payload: serde_json::from_str(&payload).map_err(|error| {
                        PgError::CorruptRow {
                            table: "projection_event",
                            reason: error.to_string(),
                        }
                    })?,
                })
            })
            .collect()
    }

    /// The last commit index `consumer` recorded as processed; `0` for a
    /// consumer that never committed.
    pub async fn consumer_offset(&self, consumer: &str) -> Result<CommitIndex, PgError> {
        check_text("event_consumer.consumer", consumer)?;
        let projection = &self.schemas.projection;
        let row = self
            .client
            .query_opt(
                &format!("SELECT committed FROM {projection}.event_consumer WHERE consumer = $1"),
                &[&consumer],
            )
            .await
            .map_err(database)?;
        match row {
            Some(row) => Ok(CommitIndex(to_u64(row.get(0), "event_consumer")?)),
            None => Ok(CommitIndex(0)),
        }
    }

    /// Record that `consumer` processed every event up to `offset`.
    ///
    /// Offsets only move forward (a late, smaller commit is a no-op), and an
    /// offset beyond the watermark is refused: a consumer cannot have processed
    /// events the projection has not written. Offsets live in the projection
    /// schema and restart at zero after a rebuild, so consumers must be
    /// idempotent per commit index.
    pub async fn commit_consumer(
        &self,
        consumer: &str,
        offset: CommitIndex,
    ) -> Result<(), PgError> {
        check_text("event_consumer.consumer", consumer)?;
        let projection = &self.schemas.projection;
        let offset_value = to_i64(offset.0, "offset")?;
        let written = self
            .client
            .execute(
                &format!(
                    "INSERT INTO {projection}.event_consumer (consumer, committed) \
                     SELECT $1, $2 FROM {projection}.projection_watermark w \
                     WHERE w.id = 1 AND $2 <= w.last_applied \
                     ON CONFLICT (consumer) DO UPDATE \
                     SET committed = greatest({projection}.event_consumer.committed, \
                                              EXCLUDED.committed)"
                ),
                &[&consumer, &offset_value],
            )
            .await
            .map_err(database)?;
        if written == 0 {
            let watermark = self.watermark().await?;
            return Err(PgError::ProjectionBehind {
                watermark: watermark.index.0,
                fence: offset.0,
            });
        }
        Ok(())
    }
}
