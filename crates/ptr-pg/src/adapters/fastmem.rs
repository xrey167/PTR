//! Fast-memory journals and checkpoints in the work schema.
//!
//! The journal stores each write exactly as the writer composed it (the
//! [`WriteRequest`], before key normalisation) as little-endian `f32` bit
//! patterns, so [`ptr_fastmem::FastMemory::restore`] re-admits the same bits and
//! refolds to the same state. Revoking a source generation deletes its writes
//! here in the projector's transaction; see `projection.rs`.

use ptr_fastmem::{
    binding_digest_of, check_config, decode_state, encode_state, validate_write, Decay, FastMemory,
    FastMemoryConfig, FastMemoryError, FastWeightState, IdentifierCodebook, SourceRef,
    WriteRequest, WriteSeq,
};
use ptr_types::{Generation, PrincipalId};
use tokio_postgres::{IsolationLevel, Row, Transaction};

use super::{check_text, database, digest_from, to_i64, to_u64, PgSubstrate};
use crate::error::PgError;

/// A registered fast memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FastMemoryRecord {
    pub id: String,
    /// The memory's owner, stored as given: that it is the principal the
    /// caller's execution session admitted is the caller's obligation.
    pub principal: PrincipalId,
    pub thread: String,
    pub config: FastMemoryConfig,
    /// Digest of the sealed key projection; the memory's keys mean nothing
    /// under any other.
    pub projection_digest: [u8; 32],
    /// Seed of the identifier codebook. Stored in a signed column by bit
    /// pattern, so every `u64` round-trips.
    pub codebook_seed: u64,
}

impl FastMemoryRecord {
    /// The identifier codebook the memory's values are codes of: this seed,
    /// with codes as long as one value of the configuration. A memory
    /// rebuilt from storage ([`PgSubstrate::restore_memory`]) is bound to it,
    /// so its readouts decode against no other codebook.
    ///
    /// # Errors
    /// Refuses a configuration whose values no codebook can code (it fails
    /// [`check_config`], which every stored record passes).
    pub fn codebook(&self) -> Result<IdentifierCodebook, FastMemoryError> {
        IdentifierCodebook::new(self.codebook_seed, self.config.value_len())
    }
}

/// A stored checkpoint: a state its writer declared to be the fold of the
/// journal prefix up to `applied`, bound to the digest of the writes it
/// claims to fold.
///
/// The substrate checks the binding against the stored prefix when the
/// checkpoint is stored and again when it is read; it never refolds the
/// journal to check the state cells. That they are the fold of the bound
/// writes is the writer's obligation, met by passing
/// [`ptr_fastmem::FastMemory::state`] together with
/// [`ptr_fastmem::FastMemory::binding_digest`] of one memory.
#[derive(Clone, Debug, PartialEq)]
pub struct FastMemoryCheckpoint {
    pub applied: WriteSeq,
    pub binding_digest: [u8; 32],
    pub state: FastWeightState,
}

impl PgSubstrate {
    /// Register a memory. One memory per principal and thread.
    ///
    /// The configuration must pass [`check_config`], the check
    /// [`ptr_fastmem::FastMemory`] and [`validate_write`] apply, or
    /// `PgError::InvalidMemory` is returned before anything is written. The
    /// table bounds each dimension but not their product, so without this a
    /// memory whose state exceeds `MAX_STATE_CELLS` would register and then
    /// refuse every write.
    pub async fn create_memory(&self, record: &FastMemoryRecord) -> Result<(), PgError> {
        check_text("fastmem_memory.id", &record.id)?;
        check_text("fastmem_memory.principal", &record.principal.0)?;
        check_text("fastmem_memory.thread", &record.thread)?;
        let work = &self.schemas.work;
        let config = &record.config;
        check_config(config).map_err(|_| PgError::InvalidMemory {
            memory: record.id.clone(),
            reason: "the configuration is outside the supported ranges",
        })?;
        self.client
            .execute(
                &format!(
                    "INSERT INTO {work}.fastmem_memory \
                     (id, principal, thread, heads, key_dim, value_dim, checkpoint_interval, \
                      max_writes, projection_digest, codebook_seed) \
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"
                ),
                &[
                    &record.id,
                    &record.principal.0,
                    &record.thread,
                    &dimension(config.heads, "heads")?,
                    &dimension(config.key_dim, "key_dim")?,
                    &dimension(config.value_dim, "value_dim")?,
                    &dimension(config.checkpoint_interval as usize, "checkpoint_interval")?,
                    &dimension(config.max_writes as usize, "max_writes")?,
                    &record.projection_digest.to_vec(),
                    // Bit-pattern reinterpretation, reversed on load.
                    &(record.codebook_seed as i64),
                ],
            )
            .await
            .map_err(database)?;
        Ok(())
    }

    /// Load a memory's registration and configuration, or `None` if absent.
    ///
    /// The stored configuration is returned only if it passes
    /// [`check_config`], as every loader here requires: the table bounds each
    /// dimension but not their product, so a row registered before
    /// [`create_memory`](Self::create_memory) checked the configuration, or
    /// written around it, can hold a state beyond `MAX_STATE_CELLS` that no
    /// write could ever be appended to.
    ///
    /// # Errors
    /// Propagates database errors and reports corrupt stored dimensions, a
    /// configuration outside the supported ranges or a malformed projection
    /// digest as `PgError::CorruptRow` naming `fastmem_memory`.
    pub async fn load_memory(&self, id: &str) -> Result<Option<FastMemoryRecord>, PgError> {
        let row = self
            .client
            .query_opt(&memory_query(self.schemas.work.as_str()), &[&id])
            .await
            .map_err(database)?;
        row.map(|row| memory_record(id, &row)).transpose()
    }

    /// Rebuild a registered memory from its stored journal, or `None` if it
    /// is not registered.
    ///
    /// The registration and the journal are read in one read-only
    /// repeatable-read snapshot, and the journal is restored with
    /// [`FastMemory::restore`] under the configuration and the identifier
    /// codebook the registration records ([`FastMemoryRecord::codebook`]).
    /// The stored `codebook_seed` thus binds every memory loaded here: its
    /// readouts carry that codebook, and `ptr_fastmem::decode_readout`
    /// refuses fact codes from any other.
    ///
    /// # Errors
    /// Propagates database errors; a registration [`load_memory`](Self::load_memory)
    /// refuses, or a journal row that is malformed or that
    /// [`FastMemory::restore`] refuses (out of order, beyond the configured
    /// journal, failing the write rules), is a `PgError::CorruptRow`.
    pub async fn restore_memory(&mut self, id: &str) -> Result<Option<FastMemory>, PgError> {
        let work = self.schemas.work.clone();
        let transaction = self
            .client
            .build_transaction()
            .isolation_level(IsolationLevel::RepeatableRead)
            .read_only(true)
            .start()
            .await
            .map_err(database)?;
        let Some(row) = transaction
            .query_opt(&memory_query(work.as_str()), &[&id])
            .await
            .map_err(database)?
        else {
            return Ok(None);
        };
        let record = memory_record(id, &row)?;
        let codebook = record.codebook().map_err(|error| {
            corrupt_memory(&format!("the stored codebook is not supported: {error}"))
        })?;
        let journal = transaction
            .query(&journal_query(work.as_str()), &[&id])
            .await
            .map_err(database)?
            .iter()
            .map(journal_entry)
            .collect::<Result<Vec<_>, _>>()?;
        transaction.commit().await.map_err(database)?;
        FastMemory::restore(record.config, codebook, journal)
            .map(Some)
            .map_err(|error| corrupt(&format!("the stored journal does not restore: {error}")))
    }

    /// Append one write to a memory's journal.
    ///
    /// The source generation must be admissible when the write lands: its
    /// live-generation row is held `FOR SHARE` for the transaction, so a
    /// revocation either commits first (and the append is refused) or waits
    /// for the append and then deletes it. The memory row is then locked
    /// `FOR UPDATE`, and the journal's length and last sequence number are
    /// read in a statement issued after that lock was granted, so they include
    /// every append committed before: the sequence number must exceed every
    /// journaled one, and the journal must have room.
    pub async fn append_write(
        &mut self,
        memory: &str,
        seq: WriteSeq,
        request: &WriteRequest,
    ) -> Result<(), PgError> {
        let schemas = self.schemas.clone();
        let (projection, work) = (schemas.projection.as_str(), schemas.work.as_str());
        let source = &request.source;
        let generation = to_i64(source.generation.0, "source_generation")?;
        let refuse = |reason| PgError::InvalidWrite {
            memory: memory.to_owned(),
            reason,
        };
        let (decay_kind, decay_cells) = match &request.decay {
            Decay::None => ("none", None),
            Decay::Scalar(factor) => ("scalar", Some(cells(std::slice::from_ref(factor)))),
            Decay::PerChannel(factors) => ("per_channel", Some(cells(factors))),
        };

        check_text("fastmem_write.memory", memory)?;
        check_text("fastmem_write.source_key", &source.key)?;
        let transaction = self.read_committed().await?;
        let live = transaction
            .query_opt(
                &format!(
                    "SELECT generation FROM {projection}.live_generation \
                     WHERE target = $1 FOR SHARE"
                ),
                &[&source.key],
            )
            .await
            .map_err(database)?;
        let revoked: bool = transaction
            .query_one(
                &format!(
                    "SELECT EXISTS (SELECT 1 FROM {projection}.tombstone \
                                    WHERE subject = $1 AND generation = $2)"
                ),
                &[&source.key, &generation],
            )
            .await
            .map_err(database)?
            .get(0);
        if live.map(|row| row.get::<_, i64>(0)) != Some(generation) || revoked {
            return Err(PgError::NotLive {
                target: source.key.clone(),
                generation: source.generation.0,
            });
        }
        // Lock the memory row so concurrent appends to one journal serialize,
        // then read the journal in a separate statement: its snapshot is taken
        // after the lock was granted, so it includes the append that held it.
        // Reading both in the locking statement would use the snapshot from
        // before the wait and let two appends pass the same checks.
        let config = load_config(&transaction, work, memory, true)
            .await?
            .ok_or_else(|| refuse("the memory is not registered"))?;
        validate_write(&config, request)
            .map_err(|_| refuse("the write does not satisfy the memory configuration"))?;
        let journal = transaction
            .query_one(
                &format!(
                    "SELECT count(*), coalesce(max(seq), 0) FROM {work}.fastmem_write \
                     WHERE memory = $1"
                ),
                &[&memory],
            )
            .await
            .map_err(database)?;
        let (count, last): (i64, i64) = (journal.get(0), journal.get(1));
        if count >= i64::from(config.max_writes) {
            return Err(refuse("the journal is full"));
        }
        let seq_value = to_i64(seq.0, "seq")?;
        if seq_value <= last {
            return Err(refuse("the sequence number does not follow the journal"));
        }
        transaction
            .execute(
                &format!(
                    "INSERT INTO {work}.fastmem_write \
                     (memory, seq, source_key, source_generation, input_digest, key_cells, \
                      value_cells, beta, decay_kind, decay_cells) \
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"
                ),
                &[
                    &memory,
                    &seq_value,
                    &source.key,
                    &generation,
                    &source.input_digest.to_vec(),
                    &cells(&request.key),
                    &cells(&request.value),
                    &request.beta,
                    &decay_kind,
                    &decay_cells,
                ],
            )
            .await
            .map_err(database)?;
        transaction.commit().await.map_err(database)
    }

    /// The journal in sequence order, ready for
    /// [`ptr_fastmem::FastMemory::restore`].
    pub async fn load_journal(
        &self,
        memory: &str,
    ) -> Result<Vec<(WriteSeq, WriteRequest)>, PgError> {
        self.client
            .query(&journal_query(self.schemas.work.as_str()), &[&memory])
            .await
            .map_err(database)?
            .iter()
            .map(journal_entry)
            .collect()
    }

    /// Store a checkpoint as `PTRFW001` bytes bound to `binding_digest`.
    ///
    /// The state must match the registered configuration, its applied write
    /// must be journaled, and `binding_digest` must match the stored prefix
    /// according to [`binding_digest_of`]. The caller supplies the state cells;
    /// this method does not refold the journal to verify them, so a state
    /// that folds other writes than `binding_digest` names (a revoked one,
    /// say) is stored all the same and later handed out by
    /// [`latest_checkpoint`](Self::latest_checkpoint). The caller must pass
    /// the state and the binding digest of one
    /// [`ptr_fastmem::FastMemory`], never a digest computed from the stored
    /// journal beside a state folded from another.
    /// The memory row is locked, so no
    /// append interleaves, and the prefix rows are held `FOR SHARE`: a
    /// revocation that deletes one of them either commits first (and this
    /// checkpoint no longer matches the prefix) or waits and then deletes this
    /// checkpoint with the write.
    pub async fn put_checkpoint(
        &mut self,
        memory: &str,
        binding_digest: [u8; 32],
        state: &FastWeightState,
    ) -> Result<(), PgError> {
        check_text("fastmem_checkpoint.memory", memory)?;
        let work = self.schemas.work.clone();
        let refuse = |reason| PgError::InvalidCheckpoint {
            memory: memory.to_owned(),
            reason,
        };
        let applied = to_i64(state.applied().0, "applied_seq")?;
        if applied == 0 {
            return Err(refuse("an empty state needs no checkpoint"));
        }
        let transaction = self.read_committed().await?;
        let config = load_config(&transaction, work.as_str(), memory, true)
            .await?
            .ok_or_else(|| refuse("the memory is not registered"))?;
        if config != *state.config() {
            return Err(refuse("the state's shape differs from the memory's"));
        }
        let prefix = journal_prefix(&transaction, work.as_str(), memory, applied, true).await?;
        if prefix.last().map(|(seq, _)| seq.0) != Some(state.applied().0) {
            return Err(refuse("the applied write is not in the journal"));
        }
        let recomputed = binding_digest_of(prefix.iter().map(|(seq, source)| (*seq, source)));
        if recomputed != binding_digest {
            return Err(refuse(
                "the binding does not match the stored journal prefix",
            ));
        }
        transaction
            .execute(
                &format!(
                    "INSERT INTO {work}.fastmem_checkpoint (memory, applied_seq, binding_digest, state) \
                     VALUES ($1, $2, $3, $4)"
                ),
                &[&memory, &applied, &binding_digest.to_vec(), &encode_state(state)],
            )
            .await
            .map_err(database)?;
        transaction.commit().await.map_err(database)
    }

    /// The newest checkpoint whose binding matches the stored journal prefix,
    /// decoded and integrity-checked within the read snapshot.
    ///
    /// Checkpoints are read with their prefix in one repeatable-read snapshot
    /// and each one's binding is recomputed from the prefix; one that no longer
    /// matches (it names a write that has since been removed) is skipped.
    /// Returns `None` if no matching checkpoint exists. State cells are not
    /// compared with a refold: they are the fold of the bound writes only if
    /// the writer met [`put_checkpoint`](Self::put_checkpoint)'s obligation.
    /// Lifecycle admission is still required at use.
    ///
    /// # Errors
    /// Propagates database errors and returns `PgError::CorruptRow` for
    /// malformed bindings, invalid state encodings in a matching checkpoint,
    /// or a decoded sequence number that disagrees with its row. These errors
    /// are not skipped in favor of an older checkpoint.
    pub async fn latest_checkpoint(
        &mut self,
        memory: &str,
    ) -> Result<Option<FastMemoryCheckpoint>, PgError> {
        let work = self.schemas.work.clone();
        let transaction = self
            .client
            .build_transaction()
            .isolation_level(tokio_postgres::IsolationLevel::RepeatableRead)
            .read_only(true)
            .start()
            .await
            .map_err(database)?;
        let rows = transaction
            .query(
                &format!(
                    "SELECT applied_seq, binding_digest, state FROM {work}.fastmem_checkpoint \
                     WHERE memory = $1 ORDER BY applied_seq DESC"
                ),
                &[&memory],
            )
            .await
            .map_err(database)?;
        let newest = rows.first().map(|row| row.get::<_, i64>(0)).unwrap_or(0);
        let prefix = journal_prefix(&transaction, work.as_str(), memory, newest, false).await?;
        for row in rows {
            let applied_value: i64 = row.get(0);
            let binding_digest = digest_from(row.get(1), "fastmem_checkpoint")?;
            let folded: Vec<&(WriteSeq, SourceRef)> = prefix
                .iter()
                .filter(|(seq, _)| seq.0 as i64 <= applied_value)
                .collect();
            let applied = WriteSeq(to_u64(applied_value, "fastmem_checkpoint")?);
            let current = folded.last().map(|(seq, _)| *seq) == Some(applied)
                && binding_digest_of(folded.iter().map(|(seq, source)| (*seq, source)))
                    == binding_digest;
            if !current {
                continue;
            }
            let bytes: Vec<u8> = row.get(2);
            let state = decode_state(&bytes).map_err(|error| PgError::CorruptRow {
                table: "fastmem_checkpoint",
                reason: error.to_string(),
            })?;
            if state.applied() != applied {
                return Err(corrupt_checkpoint(
                    "the state's applied sequence differs from the row",
                ));
            }
            transaction.commit().await.map_err(database)?;
            return Ok(Some(FastMemoryCheckpoint {
                applied,
                binding_digest,
                state,
            }));
        }
        transaction.commit().await.map_err(database)?;
        Ok(None)
    }
}

/// The registration of one memory, `$1`, as [`memory_record`] reads it.
fn memory_query(work: &str) -> String {
    format!(
        "SELECT principal, thread, heads, key_dim, value_dim, checkpoint_interval, \
                max_writes, projection_digest, codebook_seed \
         FROM {work}.fastmem_memory WHERE id = $1"
    )
}

/// One row of [`memory_query`], its configuration checked by
/// [`stored_config`].
fn memory_record(id: &str, row: &Row) -> Result<FastMemoryRecord, PgError> {
    let config = stored_config(row.get(2), row.get(3), row.get(4), row.get(5), row.get(6))?;
    Ok(FastMemoryRecord {
        id: id.to_owned(),
        principal: PrincipalId(row.get(0)),
        thread: row.get(1),
        config,
        projection_digest: digest_from(row.get(7), "fastmem_memory")?,
        codebook_seed: row.get::<_, i64>(8) as u64,
    })
}

/// The journal of one memory, `$1`, in sequence order, as [`journal_entry`]
/// reads it.
fn journal_query(work: &str) -> String {
    format!(
        "SELECT seq, source_key, source_generation, input_digest, key_cells, \
                value_cells, beta, decay_kind, decay_cells \
         FROM {work}.fastmem_write WHERE memory = $1 ORDER BY seq"
    )
}

/// One row of [`journal_query`] as the write request it stores.
fn journal_entry(row: &Row) -> Result<(WriteSeq, WriteRequest), PgError> {
    let decay_kind: String = row.get(7);
    let decay_cells: Option<Vec<u8>> = row.get(8);
    let decay = match (decay_kind.as_str(), decay_cells) {
        ("none", None) => Decay::None,
        ("scalar", Some(bytes)) => match floats(&bytes)?.as_slice() {
            [factor] => Decay::Scalar(*factor),
            _ => return Err(corrupt("a scalar decay is not one value")),
        },
        ("per_channel", Some(bytes)) => Decay::PerChannel(floats(&bytes)?),
        _ => return Err(corrupt("decay kind and cells disagree")),
    };
    Ok((
        WriteSeq(to_u64(row.get(0), "fastmem_write")?),
        WriteRequest {
            source: SourceRef {
                key: row.get(1),
                generation: Generation(to_u64(row.get(2), "fastmem_write")?),
                input_digest: digest_from(row.get(3), "fastmem_write")?,
            },
            key: floats(&row.get::<_, Vec<u8>>(4))?,
            value: floats(&row.get::<_, Vec<u8>>(5))?,
            beta: row.get(6),
            decay,
        },
    ))
}

/// A memory's shape, optionally locking its row. The shape is checked as
/// [`PgSubstrate::load_memory`] checks it, so an append or a checkpoint never
/// proceeds against a configuration no memory can have.
async fn load_config(
    transaction: &Transaction<'_>,
    work: &str,
    memory: &str,
    lock: bool,
) -> Result<Option<FastMemoryConfig>, PgError> {
    let locking = if lock { " FOR UPDATE" } else { "" };
    let row = transaction
        .query_opt(
            &format!(
                "SELECT heads, key_dim, value_dim, checkpoint_interval, max_writes \
                 FROM {work}.fastmem_memory WHERE id = $1{locking}"
            ),
            &[&memory],
        )
        .await
        .map_err(database)?;
    let Some(row) = row else {
        return Ok(None);
    };
    stored_config(row.get(0), row.get(1), row.get(2), row.get(3), row.get(4)).map(Some)
}

/// Rebuild a stored configuration and refuse it, as a corrupt
/// `fastmem_memory` row, unless it passes [`check_config`]. Every loader goes
/// through here.
fn stored_config(
    heads: i32,
    key_dim: i32,
    value_dim: i32,
    checkpoint_interval: i32,
    max_writes: i32,
) -> Result<FastMemoryConfig, PgError> {
    let count = |value: i32| {
        u32::try_from(value).map_err(|_| corrupt_memory("negative journal or checkpoint bound"))
    };
    let config = FastMemoryConfig {
        heads: unsigned(heads)?,
        key_dim: unsigned(key_dim)?,
        value_dim: unsigned(value_dim)?,
        checkpoint_interval: count(checkpoint_interval)?,
        max_writes: count(max_writes)?,
    };
    check_config(&config).map_err(|error| PgError::CorruptRow {
        table: "fastmem_memory",
        reason: format!("the stored configuration is not supported: {error}"),
    })?;
    Ok(config)
}

/// The sources of the journal writes up to `applied`, in sequence order,
/// optionally holding the rows `FOR SHARE`.
async fn journal_prefix(
    transaction: &Transaction<'_>,
    work: &str,
    memory: &str,
    applied: i64,
    lock: bool,
) -> Result<Vec<(WriteSeq, SourceRef)>, PgError> {
    let locking = if lock { " FOR SHARE" } else { "" };
    let rows = transaction
        .query(
            &format!(
                "SELECT seq, source_key, source_generation, input_digest \
                 FROM {work}.fastmem_write WHERE memory = $1 AND seq <= $2 \
                 ORDER BY seq{locking}"
            ),
            &[&memory, &applied],
        )
        .await
        .map_err(database)?;
    rows.into_iter()
        .map(|row| {
            Ok((
                WriteSeq(to_u64(row.get(0), "fastmem_write")?),
                SourceRef {
                    key: row.get(1),
                    generation: Generation(to_u64(row.get(2), "fastmem_write")?),
                    input_digest: digest_from(row.get(3), "fastmem_write")?,
                },
            ))
        })
        .collect()
}

fn dimension(value: usize, field: &'static str) -> Result<i32, PgError> {
    i32::try_from(value).map_err(|_| PgError::OutOfRange { field })
}

fn unsigned(value: i32) -> Result<usize, PgError> {
    usize::try_from(value).map_err(|_| corrupt_memory("negative dimension"))
}

fn cells(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn floats(bytes: &[u8]) -> Result<Vec<f32>, PgError> {
    if bytes.len() % 4 != 0 {
        return Err(corrupt("cell bytes are not a whole number of f32 values"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn corrupt(reason: &str) -> PgError {
    PgError::CorruptRow {
        table: "fastmem_write",
        reason: reason.to_owned(),
    }
}

fn corrupt_memory(reason: &str) -> PgError {
    PgError::CorruptRow {
        table: "fastmem_memory",
        reason: reason.to_owned(),
    }
}

fn corrupt_checkpoint(reason: &str) -> PgError {
    PgError::CorruptRow {
        table: "fastmem_checkpoint",
        reason: reason.to_owned(),
    }
}
