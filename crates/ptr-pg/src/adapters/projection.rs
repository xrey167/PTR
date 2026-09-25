//! The projector: the only writer of the projection schema.
//!
//! It applies one committed record per transaction, in commit order, behind the
//! watermark row lock. Every record arrives with the anchor the ledger computed
//! for it, and the projector recomputes that anchor from the one it stored for
//! the previous record: if they differ, the projection's history is not the
//! ledger's and the record is refused. A re-delivered record is accepted only
//! when its anchor equals the one stored for its index.

use ptr_ledger::integrity::{chain_anchors, LogAnchor};
use ptr_ledger::CommittedEvent;
use ptr_state::{classify_next, projection_entries, ApplyOutcome};
use ptr_types::{CommitIndex, Generation, Revision};
use tokio_postgres::Transaction;

use super::{database, digest_from, to_i64, to_u64, PgSubstrate};
use crate::config::SchemaSet;
use crate::error::PgError;
use crate::event::{event_payload, event_subject, event_topic, lifecycle_change, LifecycleChange};

/// What applying one record did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionApply {
    /// `Applied` when the record advanced the watermark; otherwise why it was
    /// not applied. A duplicate or out-of-order record is reported only after
    /// its anchor matched the stored one.
    pub outcome: ApplyOutcome,
    /// Derived search documents deleted because their generation stopped
    /// being live.
    pub dropped_documents: u64,
    /// Fast-memory journal writes deleted because their source generation was
    /// revoked, and checkpoints deleted because they folded one of them.
    pub revoked_writes: u64,
    pub dropped_checkpoints: u64,
}

impl ProjectionApply {
    fn refused(outcome: ApplyOutcome) -> Self {
        Self {
            outcome,
            dropped_documents: 0,
            revoked_writes: 0,
            dropped_checkpoints: 0,
        }
    }
}

impl PgSubstrate {
    /// The anchor of the last applied record: [`LogAnchor::empty`] before the
    /// first.
    pub async fn watermark(&self) -> Result<LogAnchor, PgError> {
        let projection = &self.schemas.projection;
        let row = self
            .client
            .query_one(
                &format!(
                    "SELECT last_applied, last_anchor FROM {projection}.projection_watermark \
                     WHERE id = 1"
                ),
                &[],
            )
            .await
            .map_err(database)?;
        anchor_from_row(row.get(0), row.get(1))
    }

    /// Check the projection against the ledger's current head.
    ///
    /// Returns how many commits the projection is behind. Refuses when the
    /// projection is ahead of the ledger (a discarded ledger tail or a
    /// projection restored from a later backup) and when both are at the same
    /// index with different anchors. A projection that is behind is verified
    /// record by record as it catches up.
    pub async fn check_against_ledger(&self, ledger_head: LogAnchor) -> Result<u64, PgError> {
        let watermark = self.watermark().await?;
        let (projection, ledger) = (watermark.index.0, ledger_head.index.0);
        if ledger < projection {
            return Err(PgError::ProjectionAhead { projection, ledger });
        }
        if ledger == projection && ledger_head.digest != watermark.digest {
            return Err(PgError::ForeignHistory { index: projection });
        }
        Ok(ledger - projection)
    }

    /// Apply one committed record with the anchor the ledger computed for it.
    pub async fn apply_committed(
        &mut self,
        committed: &CommittedEvent,
        anchor: LogAnchor,
    ) -> Result<ProjectionApply, PgError> {
        let index = committed.index.0;
        if anchor.index != committed.index {
            return Err(PgError::InvalidRecord {
                index,
                reason: format!("the anchor is for index {}", anchor.index.0),
            });
        }
        let schemas = self.schemas.clone();
        let transaction = self.client.transaction().await.map_err(database)?;
        let current = lock_watermark(&transaction, &schemas).await?;

        if let Some(refusal) = classify_next(current.index.0, index) {
            if matches!(refusal, ApplyOutcome::Duplicate | ApplyOutcome::OutOfOrder) {
                let stored = applied_anchor(&transaction, &schemas, index).await?;
                if stored != Some(anchor.digest) {
                    return Err(PgError::ForeignHistory { index });
                }
            }
            transaction.rollback().await.map_err(database)?;
            return Ok(ProjectionApply::refused(refusal));
        }

        let recomputed = chain_anchors(std::slice::from_ref(committed), current)
            .map_err(|error| PgError::InvalidRecord {
                index,
                reason: error.to_string(),
            })?
            .pop()
            .ok_or_else(|| PgError::InvalidRecord {
                index,
                reason: "the encoder produced no anchor".into(),
            })?;
        if recomputed != anchor {
            // The record does not chain from the anchor this projection stored:
            // either it is not the ledger's record at this index or the
            // projection's own prefix is not the ledger's.
            return Err(PgError::ForeignHistory { index });
        }

        let mut report = ProjectionApply::refused(ApplyOutcome::Applied);
        let commit = to_i64(index, "commit_index")?;
        let projection = schemas.projection.as_str();
        let derived = schemas.derived.as_str();
        let work = schemas.work.as_str();

        for (key, value) in projection_entries(committed) {
            transaction
                .execute(
                    &format!(
                        "INSERT INTO {projection}.state_entry (key, value, commit_index) \
                         VALUES ($1, $2, $3) \
                         ON CONFLICT (key) DO UPDATE \
                         SET value = EXCLUDED.value, commit_index = EXCLUDED.commit_index"
                    ),
                    &[&key, &value, &commit],
                )
                .await
                .map_err(database)?;
        }

        match lifecycle_change(committed) {
            LifecycleChange::SetLive {
                target,
                generation,
                project,
            } => {
                let generation = to_i64(generation.0, "generation")?;
                let project = project.map(|project| project.0);
                // Updating the row waits for every derived-cache writer that
                // holds it FOR SHARE, so none can index the old generation
                // after the delete below.
                transaction
                    .execute(
                        &format!(
                            "INSERT INTO {projection}.live_generation \
                             (target, generation, project, commit_index) \
                             VALUES ($1, $2, $3, $4) \
                             ON CONFLICT (target) DO UPDATE \
                             SET generation = EXCLUDED.generation, \
                                 project = coalesce(EXCLUDED.project, \
                                                    {projection}.live_generation.project), \
                                 commit_index = EXCLUDED.commit_index"
                        ),
                        &[&target, &generation, &project, &commit],
                    )
                    .await
                    .map_err(database)?;
                report.dropped_documents = transaction
                    .execute(
                        &format!(
                            "DELETE FROM {derived}.search_document \
                             WHERE capsule = $1 AND generation <> $2"
                        ),
                        &[&target, &generation],
                    )
                    .await
                    .map_err(database)?;
            }
            LifecycleChange::Tombstone {
                subject,
                generation,
            } => {
                let generation = to_i64(generation.0, "generation")?;
                // Take the target's row lock before anything is deleted; a
                // tombstone committed before the target is live has no row.
                transaction
                    .execute(
                        &format!(
                            "UPDATE {projection}.live_generation SET commit_index = $2 \
                             WHERE target = $1"
                        ),
                        &[&subject, &commit],
                    )
                    .await
                    .map_err(database)?;
                transaction
                    .execute(
                        &format!(
                            "INSERT INTO {projection}.tombstone (subject, generation, commit_index) \
                             VALUES ($1, $2, $3) ON CONFLICT (subject, generation) DO NOTHING"
                        ),
                        &[&subject, &generation, &commit],
                    )
                    .await
                    .map_err(database)?;
                report.dropped_documents = transaction
                    .execute(
                        &format!(
                            "DELETE FROM {derived}.search_document \
                             WHERE capsule = $1 AND generation = $2"
                        ),
                        &[&subject, &generation],
                    )
                    .await
                    .map_err(database)?;
                // Exact revocation of fast memories: delete every journal write
                // derived from the revoked generation and every checkpoint that
                // folded one of them. What remains refolds bit-identically to a
                // memory that never saw the writes.
                let row = transaction
                    .query_one(
                        &format!(
                            "WITH removed AS ( \
                                 DELETE FROM {work}.fastmem_write \
                                 WHERE source_key = $1 AND source_generation = $2 \
                                 RETURNING memory, seq \
                             ), first_removed AS ( \
                                 SELECT memory, min(seq) AS seq FROM removed GROUP BY memory \
                             ), dropped AS ( \
                                 DELETE FROM {work}.fastmem_checkpoint c \
                                 USING first_removed f \
                                 WHERE c.memory = f.memory AND c.applied_seq >= f.seq \
                                 RETURNING 1 \
                             ) \
                             SELECT (SELECT count(*) FROM removed), (SELECT count(*) FROM dropped)"
                        ),
                        &[&subject, &generation],
                    )
                    .await
                    .map_err(database)?;
                report.revoked_writes = to_u64(row.get(0), "fastmem_write")?;
                report.dropped_checkpoints = to_u64(row.get(1), "fastmem_checkpoint")?;
            }
            LifecycleChange::Revision { base, revision } => {
                transaction
                    .execute(
                        &format!(
                            "INSERT INTO {projection}.semantic_revision \
                             (revision, base_revision, commit_index) VALUES ($1, $2, $3)"
                        ),
                        &[
                            &to_i64(revision, "revision")?,
                            &to_i64(base, "base_revision")?,
                            &commit,
                        ],
                    )
                    .await
                    .map_err(database)?;
            }
            LifecycleChange::None => {}
        }

        let payload = event_payload(committed).to_string();
        transaction
            .execute(
                &format!(
                    "INSERT INTO {projection}.projection_event \
                     (commit_index, ordinal, topic, subject, payload) \
                     VALUES ($1, 0, $2, $3, $4::text::jsonb)"
                ),
                &[
                    &commit,
                    &event_topic(&committed.event),
                    &event_subject(committed),
                    &payload,
                ],
            )
            .await
            .map_err(database)?;
        transaction
            .execute(
                &format!(
                    "INSERT INTO {projection}.applied_commit (commit_index, anchor) VALUES ($1, $2)"
                ),
                &[&commit, &anchor.digest.to_vec()],
            )
            .await
            .map_err(database)?;
        transaction
            .execute(
                &format!(
                    "UPDATE {projection}.projection_watermark \
                     SET last_applied = $1, last_anchor = $2 WHERE id = 1"
                ),
                &[&commit, &anchor.digest.to_vec()],
            )
            .await
            .map_err(database)?;
        // Delivered on commit only, so a listener never wakes for a record
        // that was rolled back.
        transaction
            .execute(
                "SELECT pg_notify($1, $2)",
                &[&notify_channel(&schemas), &index.to_string()],
            )
            .await
            .map_err(database)?;
        transaction.commit().await.map_err(database)?;
        Ok(report)
    }

    /// Replay a complete log from its first record, computing each record's
    /// anchor from [`LogAnchor::empty`] with the ledger's own encoder. This is
    /// the rebuild path: after [`PgSubstrate::rebuild_projection`] the
    /// projection is empty and replays from index 1; records already applied
    /// are checked against their stored anchors and skipped.
    pub async fn replay(&mut self, log: &[CommittedEvent]) -> Result<u64, PgError> {
        let anchors =
            chain_anchors(log, LogAnchor::empty()).map_err(|error| PgError::InvalidRecord {
                index: log.first().map_or(0, |first| first.index.0),
                reason: error.to_string(),
            })?;
        let mut applied = 0;
        for (committed, anchor) in log.iter().zip(anchors) {
            let report = self.apply_committed(committed, anchor).await?;
            match report.outcome {
                ApplyOutcome::Applied => applied += 1,
                ApplyOutcome::Duplicate | ApplyOutcome::OutOfOrder => {}
                ApplyOutcome::Gap => {
                    return Err(PgError::InvalidRecord {
                        index: committed.index.0,
                        reason: "the log skips an index".into(),
                    })
                }
            }
        }
        Ok(applied)
    }

    /// The projected value of `key`, refusing unless the projection has
    /// applied at least `fence`. One statement, so the watermark and the value
    /// come from one snapshot.
    pub async fn state_value(
        &self,
        key: &str,
        fence: CommitIndex,
    ) -> Result<Option<String>, PgError> {
        let projection = &self.schemas.projection;
        let row = self
            .client
            .query_one(
                &format!(
                    "SELECT w.last_applied, s.value FROM {projection}.projection_watermark w \
                     LEFT JOIN {projection}.state_entry s ON s.key = $1 WHERE w.id = 1"
                ),
                &[&key],
            )
            .await
            .map_err(database)?;
        check_fence(row.get(0), fence)?;
        Ok(row.get(1))
    }

    /// The live generation of a lifecycle target (a capsule id,
    /// `constraint:<key>` or `procedure:<id>`), as of at least `fence`.
    pub async fn live_generation(
        &self,
        target: &str,
        fence: CommitIndex,
    ) -> Result<Option<Generation>, PgError> {
        let projection = &self.schemas.projection;
        let row = self
            .client
            .query_one(
                &format!(
                    "SELECT w.last_applied, l.generation FROM {projection}.projection_watermark w \
                     LEFT JOIN {projection}.live_generation l ON l.target = $1 WHERE w.id = 1"
                ),
                &[&target],
            )
            .await
            .map_err(database)?;
        check_fence(row.get(0), fence)?;
        row.get::<_, Option<i64>>(1)
            .map(|generation| to_u64(generation, "live_generation").map(Generation))
            .transpose()
    }

    /// Whether `(target, generation)` is admissible as of at least `fence`:
    /// the target's live generation *and* not in the revocation set, the same
    /// rule the runtime applies.
    pub async fn is_admissible(
        &self,
        target: &str,
        generation: Generation,
        fence: CommitIndex,
    ) -> Result<bool, PgError> {
        let projection = &self.schemas.projection;
        let generation = to_i64(generation.0, "generation")?;
        let row = self
            .client
            .query_one(
                &format!(
                    "SELECT w.last_applied, \
                        EXISTS (SELECT 1 FROM {projection}.live_generation l \
                                WHERE l.target = $1 AND l.generation = $2) \
                        AND NOT EXISTS (SELECT 1 FROM {projection}.tombstone t \
                                        WHERE t.subject = $1 AND t.generation = $2) \
                     FROM {projection}.projection_watermark w WHERE w.id = 1"
                ),
                &[&target, &generation],
            )
            .await
            .map_err(database)?;
        check_fence(row.get(0), fence)?;
        Ok(row.get(1))
    }

    /// The commit that published `revision`, if the projection has seen it.
    pub async fn revision_commit(
        &self,
        revision: Revision,
    ) -> Result<Option<CommitIndex>, PgError> {
        let projection = &self.schemas.projection;
        let row = self
            .client
            .query_opt(
                &format!(
                    "SELECT commit_index FROM {projection}.semantic_revision WHERE revision = $1"
                ),
                &[&to_i64(revision.0, "revision")?],
            )
            .await
            .map_err(database)?;
        row.map(|row| to_u64(row.get(0), "semantic_revision").map(CommitIndex))
            .transpose()
    }

    /// The `LISTEN` channel the projector notifies with each applied index.
    pub fn projection_channel(&self) -> String {
        notify_channel(&self.schemas)
    }
}

fn notify_channel(schemas: &SchemaSet) -> String {
    format!("{}_event", schemas.projection)
}

async fn lock_watermark(
    transaction: &Transaction<'_>,
    schemas: &SchemaSet,
) -> Result<LogAnchor, PgError> {
    let projection = &schemas.projection;
    let row = transaction
        .query_one(
            &format!(
                "SELECT last_applied, last_anchor FROM {projection}.projection_watermark \
                 WHERE id = 1 FOR UPDATE"
            ),
            &[],
        )
        .await
        .map_err(database)?;
    anchor_from_row(row.get(0), row.get(1))
}

async fn applied_anchor(
    transaction: &Transaction<'_>,
    schemas: &SchemaSet,
    index: u64,
) -> Result<Option<[u8; 32]>, PgError> {
    let projection = &schemas.projection;
    let row = transaction
        .query_opt(
            &format!("SELECT anchor FROM {projection}.applied_commit WHERE commit_index = $1"),
            &[&to_i64(index, "commit_index")?],
        )
        .await
        .map_err(database)?;
    row.map(|row| digest_from(row.get(0), "applied_commit"))
        .transpose()
}

fn anchor_from_row(last_applied: i64, anchor: Option<Vec<u8>>) -> Result<LogAnchor, PgError> {
    let index = to_u64(last_applied, "projection_watermark")?;
    match anchor {
        None if index == 0 => Ok(LogAnchor::empty()),
        Some(digest) if index > 0 => Ok(LogAnchor {
            index: CommitIndex(index),
            digest: digest_from(digest, "projection_watermark")?,
        }),
        _ => Err(PgError::CorruptRow {
            table: "projection_watermark",
            reason: "anchor presence does not match the applied index".into(),
        }),
    }
}

fn check_fence(watermark: i64, fence: CommitIndex) -> Result<(), PgError> {
    let watermark = to_u64(watermark, "projection_watermark")?;
    if watermark < fence.0 {
        return Err(PgError::ProjectionBehind {
            watermark,
            fence: fence.0,
        });
    }
    Ok(())
}
