use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use crate::config::{check_config, FastMemoryConfig};
use crate::error::FastMemoryError;
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
#[derive(Clone, Debug)]
pub struct FastMemory {
    config: FastMemoryConfig,
    state: FastWeightState,
    writes: Vec<MemoryWrite>,
    checkpoints: Vec<FastWeightState>,
    next_seq: u64,
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
    /// An empty memory.
    pub fn new(config: FastMemoryConfig) -> Result<Self, FastMemoryError> {
        check_config(&config)?;
        Ok(Self {
            config,
            state: FastWeightState::empty(config),
            writes: Vec::new(),
            checkpoints: Vec::new(),
            next_seq: 1,
        })
    }

    /// Rebuild a memory from a stored journal. Sequence numbers must strictly
    /// increase; gaps are expected where writes were revoked.
    ///
    /// Use original write requests, with sequence numbers starting at one or
    /// higher and below `u64::MAX`, so the restored memory can number its next
    /// write. Every write is admitted by the same rules as [`Self::write`].
    /// Returns configuration, journal-capacity, sequence-order,
    /// sequence-exhaustion, or write-validation errors; no partially restored
    /// memory is returned.
    pub fn restore<I>(config: FastMemoryConfig, journal: I) -> Result<Self, FastMemoryError>
    where
        I: IntoIterator<Item = (WriteSeq, WriteRequest)>,
    {
        let mut memory = Self::new(config)?;
        for (seq, request) in journal {
            if seq.0 < memory.next_seq {
                return Err(FastMemoryError::OutOfOrderWrite {
                    expected: memory.next_seq,
                    actual: seq.0,
                });
            }
            memory.check_capacity()?;
            let next_seq = successor(seq)?;
            let write = admit_write(&memory.config, seq, request)?;
            memory.next_seq = next_seq;
            let _surprise = memory.fold(write);
        }
        Ok(memory)
    }

    pub fn config(&self) -> &FastMemoryConfig {
        &self.config
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
    /// `u64::MAX` (it has no successor, so its journal could not be restored),
    /// invalid vector lengths, nonfinite vector entries, value cells beyond
    /// [`crate::MAX_VALUE_MAGNITUDE`] (the bound that keeps every fold of
    /// admitted writes finite), all-zero key heads, or
    /// strength/decay factors outside `(0, 1]`. Validation errors leave the
    /// journal, state, and next sequence unchanged.
    pub fn write(&mut self, request: WriteRequest) -> Result<WriteReceipt, FastMemoryError> {
        self.check_capacity()?;
        let seq = WriteSeq(self.next_seq);
        let next_seq = successor(seq)?;
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
    /// it enters reasoning only through decoding into search candidates.
    ///
    /// # Errors
    /// First refuses a query normalised for another head shape
    /// (`DimensionMismatch` naming `query_heads` or `query_key_dim`): its
    /// components would be split into heads it was not normalised for, and the
    /// readout would be plausible but wrong. Then refuses a state that depends
    /// on an inadmissible source (`Denied`).
    pub fn read_admitted<F>(&self, query: &Query, admissible: F) -> Result<Readout, FastMemoryError>
    where
        F: Fn(&SourceRef) -> bool,
    {
        query.check_shape(&self.config)?;
        let denied = self
            .writes
            .iter()
            .filter(|write| !admissible(write.source()))
            .count();
        if denied > 0 {
            return Err(FastMemoryError::Denied { sources: denied });
        }
        Ok(self.state.read(query))
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
}

/// The sequence number after `seq`. A journaled write must leave a successor
/// free, so `u64::MAX` is refused before anything changes: a write or restore
/// that took it would leave no number for the next write, and a journal ending
/// there could not be restored.
fn successor(seq: WriteSeq) -> Result<u64, FastMemoryError> {
    seq.0
        .checked_add(1)
        .ok_or(FastMemoryError::SequenceExhausted { seq: seq.0 })
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
