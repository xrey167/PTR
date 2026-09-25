//! Fast-memory journals and checkpoints in the work schema.
//!
//! The journal stores each write exactly as the writer composed it (the
//! [`WriteRequest`], before key normalisation) as little-endian `f32` bit
//! patterns, so [`ptr_fastmem::FastMemory::restore`] re-admits the same bits and
//! refolds to the same state. Revoking a source generation deletes its writes
//! here in the projector's transaction; see `projection.rs`.

use ptr_fastmem::{
    decode_state, encode_state, Decay, FastMemoryConfig, FastWeightState, SourceRef, WriteRequest,
    WriteSeq,
};
use ptr_types::{Generation, PrincipalId};

use super::{database, digest_from, to_i64, to_u64, PgSubstrate};
use crate::error::PgError;

/// A registered fast memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FastMemoryRecord {
    pub id: String,
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

/// A stored checkpoint: a fold of the journal prefix up to `applied`, bound to
/// the digest of the writes it folds.
#[derive(Clone, Debug, PartialEq)]
pub struct FastMemoryCheckpoint {
    pub applied: WriteSeq,
    pub binding_digest: [u8; 32],
    pub state: FastWeightState,
}

impl PgSubstrate {
    /// Register a memory. One memory per principal and thread.
    pub async fn create_memory(&self, record: &FastMemoryRecord) -> Result<(), PgError> {
        let work = &self.schemas.work;
        let config = &record.config;
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

    pub async fn load_memory(&self, id: &str) -> Result<Option<FastMemoryRecord>, PgError> {
        let work = &self.schemas.work;
        let row = self
            .client
            .query_opt(
                &format!(
                    "SELECT principal, thread, heads, key_dim, value_dim, checkpoint_interval, \
                            max_writes, projection_digest, codebook_seed \
                     FROM {work}.fastmem_memory WHERE id = $1"
                ),
                &[&id],
            )
            .await
            .map_err(database)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let config = FastMemoryConfig {
            heads: unsigned(row.get(2))?,
            key_dim: unsigned(row.get(3))?,
            value_dim: unsigned(row.get(4))?,
            checkpoint_interval: unsigned(row.get(5))? as u32,
            max_writes: unsigned(row.get(6))? as u32,
        };
        Ok(Some(FastMemoryRecord {
            id: id.to_owned(),
            principal: PrincipalId(row.get(0)),
            thread: row.get(1),
            config,
            projection_digest: digest_from(row.get(7), "fastmem_memory")?,
            codebook_seed: row.get::<_, i64>(8) as u64,
        }))
    }

    /// Append one write to a memory's journal.
    ///
    /// The source generation must be admissible when the write lands: its
    /// live-generation row is held `FOR SHARE` for the transaction, so a
    /// revocation either commits first (and the append is refused) or waits
    /// for the append and then deletes it. The sequence number must exceed
    /// every journaled one, and the journal must have room.
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

        let transaction = self.client.transaction().await.map_err(database)?;
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
        // Lock the memory row so concurrent appends to one journal serialize.
        let header = transaction
            .query_opt(
                &format!(
                    "SELECT m.max_writes, \
                            (SELECT count(*) FROM {work}.fastmem_write w WHERE w.memory = m.id), \
                            (SELECT coalesce(max(seq), 0) FROM {work}.fastmem_write w \
                             WHERE w.memory = m.id) \
                     FROM {work}.fastmem_memory m WHERE m.id = $1 FOR UPDATE"
                ),
                &[&memory],
            )
            .await
            .map_err(database)?
            .ok_or_else(|| refuse("the memory is not registered"))?;
        let (max_writes, count, last): (i32, i64, i64) =
            (header.get(0), header.get(1), header.get(2));
        if count >= i64::from(max_writes) {
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
        let work = &self.schemas.work;
        let rows = self
            .client
            .query(
                &format!(
                    "SELECT seq, source_key, source_generation, input_digest, key_cells, \
                            value_cells, beta, decay_kind, decay_cells \
                     FROM {work}.fastmem_write WHERE memory = $1 ORDER BY seq"
                ),
                &[&memory],
            )
            .await
            .map_err(database)?;
        rows.into_iter()
            .map(|row| {
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
            })
            .collect()
    }

    /// Store a checkpoint as `PTRFW001` bytes bound to `binding_digest`.
    pub async fn put_checkpoint(
        &self,
        memory: &str,
        binding_digest: [u8; 32],
        state: &FastWeightState,
    ) -> Result<(), PgError> {
        let work = &self.schemas.work;
        self.client
            .execute(
                &format!(
                    "INSERT INTO {work}.fastmem_checkpoint (memory, applied_seq, binding_digest, state) \
                     VALUES ($1, $2, $3, $4)"
                ),
                &[
                    &memory,
                    &to_i64(state.applied().0, "applied_seq")?,
                    &binding_digest.to_vec(),
                    &encode_state(state),
                ],
            )
            .await
            .map_err(database)?;
        Ok(())
    }

    /// The newest checkpoint of a memory, decoded and integrity-checked.
    pub async fn latest_checkpoint(
        &self,
        memory: &str,
    ) -> Result<Option<FastMemoryCheckpoint>, PgError> {
        let work = &self.schemas.work;
        let row = self
            .client
            .query_opt(
                &format!(
                    "SELECT applied_seq, binding_digest, state FROM {work}.fastmem_checkpoint \
                     WHERE memory = $1 ORDER BY applied_seq DESC LIMIT 1"
                ),
                &[&memory],
            )
            .await
            .map_err(database)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let bytes: Vec<u8> = row.get(2);
        let state = decode_state(&bytes).map_err(|error| PgError::CorruptRow {
            table: "fastmem_checkpoint",
            reason: error.to_string(),
        })?;
        let applied = WriteSeq(to_u64(row.get(0), "fastmem_checkpoint")?);
        if state.applied() != applied {
            return Err(corrupt_checkpoint(
                "the state's applied sequence differs from the row",
            ));
        }
        Ok(Some(FastMemoryCheckpoint {
            applied,
            binding_digest: digest_from(row.get(1), "fastmem_checkpoint")?,
            state,
        }))
    }
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
