use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use crate::config::{check_config, FastMemoryConfig};
use crate::error::FastMemoryError;
use crate::projection::IdentifierCodebook;
use crate::state::{FastWeightState, Readout};
use crate::write::{admit_write, MemoryWrite, Query, SourceRef, WriteRequest, WriteSeq};

/// A fast-weight memory with its complete write journal.
///
/// The journal, not the matrices, is the memory's record: the matrices are a
/// fold over the journal that is kept current for reads and checkpointed so a
/// rebuild never starts from the beginning. Because every write names the
/// semantic input it came from, revoking that input removes its writes from the
/// journal and refolds the state from the last checkpoint that precedes them.
/// The result is bit-identical to a memory that never received those writes.
///
/// A memory is bound to the [`IdentifierCodebook`] its values are codes of,
/// fixed when it is created or restored: writers compose values with
/// [`Self::codebook`], every readout carries it, [`Self::fact_codes`] derives
/// candidates from it, and [`crate::decode_readout`] refuses a fact code from
/// any other codebook.
///
/// Its keys are only meaningful under the key projection that produced them
/// ([`crate::SeededProjection::digest`]). A memory created or restored with
/// that digest ([`Self::with_projection`], [`Self::restore_with_projection`])
/// reads only queries stating the same digest ([`Query::project`],
/// [`Query::with_projection`]) and refuses any other query of the right shape
/// (`ProjectionMismatch`), whose readout would be crosstalk. A memory created
/// or restored without one ([`Self::new`], [`Self::restore`]) states no
/// projection and reads a query of its head shape from any projection, so a
/// store that records a memory's projection must restore it with that
/// projection for the check to apply. Write keys are not checked against the
/// projection: composing them with the one the memory names is the writer's
/// obligation.
#[derive(Clone, Debug)]
pub struct FastMemory {
    config: FastMemoryConfig,
    codebook: IdentifierCodebook,
    projection: Option<[u8; 32]>,
    state: FastWeightState,
    writes: Vec<MemoryWrite>,
    checkpoints: Vec<FastWeightState>,
    next_seq: u64,
    /// The first sequence number this memory never takes: `u64::MAX` unless
    /// [`Self::with_sequence_limit`] lowered it.
    seq_limit: u64,
}

/// What one write did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WriteReceipt {
    pub seq: WriteSeq,
    /// `beta * ||v - (A S)^T k||` summed over heads: how much the memory had
    /// to change. A derived signal for replay selection and annotation
    /// priority; it is recomputed on refold, never stored as a fact.
    pub surprise: f32,
}

/// What one revocation removed and what it cost to refold.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RevocationReport {
    /// Writes removed from the journal.
    pub removed: usize,
    /// Writes refolded after the restart point.
    pub replayed: usize,
    /// The checkpoint the refold started from; `WriteSeq(0)` is the empty state.
    pub restarted_from: WriteSeq,
}

impl FastMemory {
    /// An empty memory whose values are codes of `codebook`, bound to no key
    /// projection: it reads a query of its head shape from any projection.
    /// [`Self::with_projection`] binds one.
    ///
    /// # Errors
    /// Refuses a configuration outside the supported ranges, then a codebook
    /// whose codes are not as long as a value (`DimensionMismatch` naming
    /// `codebook`): no code of it could be written or decoded.
    pub fn new(
        config: FastMemoryConfig,
        codebook: IdentifierCodebook,
    ) -> Result<Self, FastMemoryError> {
        Self::empty(config, codebook, None)
    }

    /// An empty memory whose values are codes of `codebook` and whose keys
    /// are projected by the key projection with digest `projection`
    /// ([`crate::SeededProjection::digest`]): [`Self::read_admitted`] refuses
    /// a query that does not state that digest.
    ///
    /// # Errors
    /// Those of [`Self::new`].
    pub fn with_projection(
        config: FastMemoryConfig,
        codebook: IdentifierCodebook,
        projection: [u8; 32],
    ) -> Result<Self, FastMemoryError> {
        Self::empty(config, codebook, Some(projection))
    }

    fn empty(
        config: FastMemoryConfig,
        codebook: IdentifierCodebook,
        projection: Option<[u8; 32]>,
    ) -> Result<Self, FastMemoryError> {
        check_config(&config)?;
        if codebook.len() != config.value_len() {
            return Err(FastMemoryError::DimensionMismatch {
                field: "codebook",
                expected: config.value_len(),
                actual: codebook.len(),
            });
        }
        Ok(Self {
            config,
            codebook,
            projection,
            state: FastWeightState::empty(config),
            writes: Vec::new(),
            checkpoints: Vec::new(),
            next_seq: 1,
            seq_limit: u64::MAX,
        })
    }

    /// Rebuild a memory from a stored journal. Sequence numbers must strictly
    /// increase; gaps are expected where writes were revoked.
    ///
    /// Use original write requests, with sequence numbers starting at one or
    /// higher and below `u64::MAX`, so the restored memory can number its next
    /// write. Every write is admitted by the same rules as [`Self::write`].
    /// `codebook` must be the one the journal's values are codes of, as its
    /// store records it (`ptr-pg` keeps its seed with the memory). The
    /// restored memory is bound to no key projection, like one from
    /// [`Self::new`]; [`Self::restore_with_projection`] binds the one a store
    /// records.
    /// Returns configuration, codebook, journal-capacity, sequence-order,
    /// sequence-exhaustion, or write-validation errors; no partially restored
    /// memory is returned.
    pub fn restore<I>(
        config: FastMemoryConfig,
        codebook: IdentifierCodebook,
        journal: I,
    ) -> Result<Self, FastMemoryError>
    where
        I: IntoIterator<Item = (WriteSeq, WriteRequest)>,
    {
        Self::restore_into(Self::new(config, codebook)?, journal)
    }

    /// [`Self::restore`], binding the restored memory to the key projection
    /// with digest `projection`, as [`Self::with_projection`] does: the
    /// digest its store recorded for the memory's keys.
    ///
    /// # Errors
    /// Those of [`Self::restore`].
    pub fn restore_with_projection<I>(
        config: FastMemoryConfig,
        codebook: IdentifierCodebook,
        projection: [u8; 32],
        journal: I,
    ) -> Result<Self, FastMemoryError>
    where
        I: IntoIterator<Item = (WriteSeq, WriteRequest)>,
    {
        Self::restore_into(
            Self::with_projection(config, codebook, projection)?,
            journal,
        )
    }

    fn restore_into<I>(mut memory: Self, journal: I) -> Result<Self, FastMemoryError>
    where
        I: IntoIterator<Item = (WriteSeq, WriteRequest)>,
    {
        for (seq, request) in journal {
            if seq.0 < memory.next_seq {
                return Err(FastMemoryError::OutOfOrderWrite {
                    expected: memory.next_seq,
                    actual: seq.0,
                });
            }
            memory.check_capacity()?;
            let next_seq = memory.successor(seq)?;
            let write = admit_write(&memory.config, seq, request)?;
            memory.next_seq = next_seq;
            let _surprise = memory.fold(write);
        }
        Ok(memory)
    }

    /// This memory, taking no sequence number at or above `limit`: a write
    /// that would take one is refused as `SequenceExhausted` before anything
    /// changes, as a write that would take `u64::MAX` always is. A limit
    /// only ever lowers: the memory keeps the smaller of `limit` and the one
    /// it had.
    ///
    /// A store whose sequence column holds fewer numbers than `u64` restores
    /// a memory with its own limit (`ptr-pg` journals no number at or above
    /// `i64::MAX`), so the memory refuses the write the store would refuse
    /// rather than fold one it cannot journal. Its sequence space is then
    /// exhausted after `limit - 1`, however much room its journal has.
    ///
    /// # Errors
    /// `SequenceExhausted` naming the largest sequence number the memory has
    /// taken (restored, since revoked, or at or below a mark
    /// [`Self::with_sequence_high_water`] set) when that is at or above
    /// `limit`: the memory already holds, or has handed out, a number the
    /// limit excludes.
    pub fn with_sequence_limit(mut self, limit: WriteSeq) -> Result<Self, FastMemoryError> {
        let taken = self.next_seq - 1;
        if taken > 0 && taken >= limit.0 {
            return Err(FastMemoryError::SequenceExhausted { seq: taken });
        }
        self.seq_limit = self.seq_limit.min(limit.0);
        Ok(self)
    }

    /// This memory, numbering its next write above `high_water`: the
    /// largest sequence number its store ever journaled for it, including
    /// the numbers of writes revoked since, which the store no longer holds.
    ///
    /// A memory keeps every number it took ([`Self::revoke`] leaves its next
    /// number where it was), but [`Self::restore`] can only number the next
    /// write after the last one its journal holds. Restored from a journal
    /// whose last writes were revoked, it would hand their numbers out
    /// again, although the live memory never takes them again. A store that
    /// keeps its journal's high-water mark restores with it (`ptr-pg` does),
    /// so a restored memory numbers its next write as the live one does. A
    /// mark only ever raises: one at or below a number the memory already
    /// took changes nothing.
    ///
    /// # Errors
    /// `SequenceExhausted` naming `high_water` when it is at or above the
    /// memory's limit ([`Self::sequence_limit`]): the memory never takes
    /// that number, so no store journaled it for this memory.
    pub fn with_sequence_high_water(
        mut self,
        high_water: WriteSeq,
    ) -> Result<Self, FastMemoryError> {
        let next_seq = self.successor(high_water)?;
        self.next_seq = self.next_seq.max(next_seq);
        Ok(self)
    }

    /// The first sequence number this memory never takes: `u64::MAX`, or
    /// the lower limit [`Self::with_sequence_limit`] set.
    pub fn sequence_limit(&self) -> WriteSeq {
        WriteSeq(self.seq_limit)
    }

    pub fn config(&self) -> &FastMemoryConfig {
        &self.config
    }

    /// The codebook this memory's values are codes of. Writers compose each
    /// value as `codebook().code_for(capsule, generation)`.
    pub fn codebook(&self) -> IdentifierCodebook {
        self.codebook
    }

    /// The digest of the key projection this memory is bound to, or `None`
    /// for a memory from [`Self::new`] or [`Self::restore`], which reads
    /// queries from any projection.
    pub fn projection_digest(&self) -> Option<[u8; 32]> {
        self.projection
    }

    pub fn state(&self) -> &FastWeightState {
        &self.state
    }

    /// The admitted writes in write order, with keys normalised per head.
    ///
    /// These are not the writes as composed: re-admitting a normalised key
    /// normalises it again, which can change its bits, so this list must not
    /// be passed back to [`FastMemory::restore`]. A store keeps each
    /// [`WriteRequest`] as the caller composed it, as `ptr-pg` does.
    pub fn writes(&self) -> &[MemoryWrite] {
        &self.writes
    }

    /// Every semantic input the current state depends on. This is the input set
    /// a neural-state declaration for this memory must list; an input missing
    /// from it would be a dependency admission cannot see.
    pub fn sources(&self) -> BTreeSet<&SourceRef> {
        self.writes.iter().map(MemoryWrite::source).collect()
    }

    /// Admit and fold one write.
    ///
    /// Assigns the next sequence number and returns it with the write's
    /// surprise. Successful writes are journaled and may create a checkpoint.
    /// The source's lifecycle and input digest are not checked here.
    ///
    /// # Errors
    /// Rejects a full journal, a write that would take sequence number
    /// `u64::MAX` (it has no successor, so its journal could not be restored)
    /// or one at or above a lower limit [`Self::with_sequence_limit`] set
    /// (both `SequenceExhausted`), invalid vector lengths, nonfinite vector entries, value cells beyond
    /// [`crate::MAX_VALUE_MAGNITUDE`] (the bound that keeps every fold of
    /// admitted writes finite), all-zero key heads, or
    /// strength/decay factors outside `(0, 1]`. Validation errors leave the
    /// journal, state, and next sequence unchanged.
    pub fn write(&mut self, request: WriteRequest) -> Result<WriteReceipt, FastMemoryError> {
        self.check_capacity()?;
        let seq = WriteSeq(self.next_seq);
        let next_seq = self.successor(seq)?;
        let write = admit_write(&self.config, seq, request)?;
        self.next_seq = next_seq;
        let surprise = self.fold(write);
        Ok(WriteReceipt { seq, surprise })
    }

    /// Read the memory, deciding admission at this use.
    ///
    /// Every write the state depends on is checked with `admissible`, which
    /// the caller answers from its lifecycle view and the current input
    /// digests. If any is not admissible the read is refused: the state must
    /// first be refolded without those writes ([`Self::revoke`]). The decision
    /// is only as current as the answers `admissible` gives: a revocation that
    /// commits after the caller's lifecycle view was taken (always possible for
    /// an external authority, which a synchronous predicate can only consult
    /// beforehand) is not seen, so decoded candidates must still pass the
    /// lifecycle check at use. The readout is a derived value, never evidence;
    /// it enters reasoning only through decoding into search candidates, and
    /// carries this memory's codebook so it is decoded against no other.
    ///
    /// # Errors
    /// First refuses a query normalised for another head shape
    /// (`DimensionMismatch` naming `query_heads` or `query_key_dim`): its
    /// components would be split into heads it was not normalised for, and the
    /// readout would be plausible but wrong. Then, if this memory is bound to
    /// a key projection, refuses a query that states another projection or
    /// none (`ProjectionMismatch`): scored against keys it was not projected
    /// like, it would read crosstalk as weights. Then refuses a state that
    /// depends on an inadmissible source (`Denied`).
    pub fn read_admitted<F>(&self, query: &Query, admissible: F) -> Result<Readout, FastMemoryError>
    where
        F: Fn(&SourceRef) -> bool,
    {
        query.check_shape(&self.config)?;
        if let Some(expected) = self.projection {
            if query.projection_digest() != Some(expected) {
                return Err(FastMemoryError::ProjectionMismatch {
                    expected,
                    actual: query.projection_digest(),
                });
            }
        }
        let denied = self
            .writes
            .iter()
            .filter(|write| !admissible(write.source()))
            .count();
        if denied > 0 {
            return Err(FastMemoryError::Denied { sources: denied });
        }
        Ok(Readout::new(
            self.state.read(query),
            self.state.applied(),
            self.codebook,
        ))
    }

    /// Digest of the ordered set of writes the state folds; see
    /// [`binding_digest_of`].
    pub fn binding_digest(&self) -> [u8; 32] {
        binding_digest_of(
            self.writes
                .iter()
                .map(|write| (write.seq(), write.source())),
        )
    }

    /// Remove every write whose source matches `is_revoked` and refold.
    ///
    /// This is exact revocation — the refolded state is bit-identical to one
    /// that never received the removed writes — not erasure: copies of the
    /// removed writes may survive in storage the caller keeps (journals,
    /// checkpoints, database pages, backups), and erasing those is the
    /// storage's obligation, audited separately.
    pub fn revoke<F>(&mut self, is_revoked: F) -> RevocationReport
    where
        F: Fn(&SourceRef) -> bool,
    {
        let Some(first) = self
            .writes
            .iter()
            .position(|write| is_revoked(write.source()))
        else {
            return RevocationReport {
                removed: 0,
                replayed: 0,
                restarted_from: self.state.applied(),
            };
        };
        let first_seq = self.writes[first].seq();
        let before = self.writes.len();
        self.writes.retain(|write| !is_revoked(write.source()));
        let removed = before - self.writes.len();

        // A checkpoint taken before the first revoked write contains none of the
        // revoked writes; every later one may.
        self.checkpoints
            .retain(|checkpoint| checkpoint.applied() < first_seq);
        let restart = self
            .checkpoints
            .last()
            .cloned()
            .unwrap_or_else(|| FastWeightState::empty(self.config));
        let restarted_from = restart.applied();
        self.state = restart;

        let mut replayed = 0;
        let interval = self.config.checkpoint_interval as usize;
        for (position, write) in self.writes.iter().enumerate() {
            if write.seq() <= restarted_from {
                continue;
            }
            let _surprise = self.state.apply(write);
            replayed += 1;
            if (position + 1) % interval == 0 {
                self.checkpoints.push(self.state.clone());
            }
        }
        RevocationReport {
            removed,
            replayed,
            restarted_from,
        }
    }

    /// Refold the whole journal from the empty state, ignoring checkpoints. Used
    /// to audit that the incrementally maintained state is the fold it claims.
    pub fn refold_from_journal(&self) -> FastWeightState {
        let mut state = FastWeightState::empty(self.config);
        for write in &self.writes {
            let _surprise = state.apply(write);
        }
        state
    }

    fn check_capacity(&self) -> Result<(), FastMemoryError> {
        if self.writes.len() >= self.config.max_writes as usize {
            return Err(FastMemoryError::JournalFull {
                limit: self.config.max_writes,
            });
        }
        Ok(())
    }

    fn fold(&mut self, write: MemoryWrite) -> f32 {
        let surprise = self.state.apply(&write);
        self.writes.push(write);
        if self.writes.len() % self.config.checkpoint_interval as usize == 0 {
            self.checkpoints.push(self.state.clone());
        }
        surprise
    }

    /// The sequence number after `seq`, refusing a `seq` at or above the
    /// memory's limit before anything changes. The limit is at most
    /// `u64::MAX`, so the successor never overflows: a write or restore that
    /// took `u64::MAX` would leave no number for the next write, and a
    /// journal ending there could not be restored.
    fn successor(&self, seq: WriteSeq) -> Result<u64, FastMemoryError> {
        if seq.0 >= self.seq_limit {
            return Err(FastMemoryError::SequenceExhausted { seq: seq.0 });
        }
        Ok(seq.0 + 1)
    }
}

/// Digest of an ordered set of folded writes: the sequence number, source
/// key, generation and input digest of each, in order.
///
/// A checkpoint is bound to this digest, so a state that folds a write the
/// journal no longer holds is never admitted as that journal's fold. It binds
/// which writes were folded, identified by their sources; it does not cover
/// the key, value, strength or gate bits, which the journal's own storage has
/// to keep intact. Storage adapters recompute it from their journal rows with
/// this function, so the encoding exists once.
pub fn binding_digest_of<'a, I>(writes: I) -> [u8; 32]
where
    I: IntoIterator<Item = (WriteSeq, &'a SourceRef)>,
{
    let mut hasher = Sha256::new();
    hasher.update(b"ptr-fastmem/binding/v1");
    for (seq, source) in writes {
        hasher.update(seq.0.to_le_bytes());
        hasher.update((source.key.len() as u64).to_le_bytes());
        hasher.update(source.key.as_bytes());
        hasher.update(source.generation.0.to_le_bytes());
        hasher.update(source.input_digest);
    }
    hasher.finalize().into()
}
