//! `DurableKnowledgeStore`: a [`KnowledgeStore`] whose every successful mutation is journaled and
//! replayed on open.
//!
//! The wrapped store keeps its exact semantics: each mutating method first runs on the in-memory
//! store (which validates and either changes nothing or applies completely) and, only on success,
//! appends the operation to the journal. If the append fails, the in-memory store is rebuilt from
//! the intact journal records, so [`DurableKnowledgeStore::store`] never exposes a change whose
//! call returned an error. A rejection that wrote nothing (record or file size bound) leaves the
//! wrapper usable; an I/O failure poisons it because the file tail is unknown, and reopening
//! recovers the durable state (a frame that reached the disk before the failure is part of it).
//! If the rebuild itself fails the store is emptied and stays poisoned. Read access goes through
//! [`DurableKnowledgeStore::store`].
//!
//! Journal payload encoding (tag byte, then little-endian fields from `codec`):
//! `1 Raw`, `2 Register`, `3 Admit`, `4 Activate`, `5 Transition`, `6 Invalidate`, plus the
//! snapshot-only `7 RestoreCurrent`, `8 RestoreLifecycle`, `9 RestoreRevision` that compaction
//! uses to reproduce lifecycle records and revision counters exactly.

use crate::codec::{CodecError, Reader, Writer};
use crate::journal::{JournalError, JournalKind, JournalOptions, MemoryJournal};
use crate::knowledge::{KnowledgeStore, KnowledgeStoreError, LifecycleRecord};
use crate::retention::{
    ConflictState, Freshness, KnowledgeKey, KnowledgeLifecycle, KnowledgeObject, RawEvent,
    Relation, ToolCall, ToolResult,
};
use ptr_types::{
    EntityId, EventRole, Generation, KnowledgeObjectId, NamespaceId, Probability, ProvenanceRef,
    RawEventId, RetrievalKey, SessionId, Timestamp,
};
use std::path::Path;

/// Options of a durable store.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DurableOptions {
    pub journal: JournalOptions,
    /// Compact automatically once this many records were appended since the last compaction.
    pub compact_after_records: Option<u64>,
}

impl Default for DurableOptions {
    fn default() -> Self {
        Self {
            journal: JournalOptions::default(),
            compact_after_records: Some(10_000),
        }
    }
}

/// Failures of [`DurableKnowledgeStore`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DurableKnowledgeError {
    /// The operation was rejected by the knowledge store; nothing was changed or journaled.
    Store(KnowledgeStoreError),
    Journal(JournalError),
    /// A journal record passed its CRC but could not be decoded.
    Decode {
        record_index: usize,
        error: CodecError,
    },
    /// A journaled operation no longer applies on replay (the journal does not match the
    /// semantics of this store).
    ReplayDiverged {
        record_index: usize,
        error: KnowledgeStoreError,
    },
    /// The journal write failed with an I/O error (or compaction left the journal unusable).
    /// Reopen the journal.
    Poisoned,
}

impl From<JournalError> for DurableKnowledgeError {
    fn from(error: JournalError) -> Self {
        Self::Journal(error)
    }
}

enum Op {
    Raw(RawEvent),
    Register(KnowledgeObject),
    Admit(KnowledgeObject),
    Activate(KnowledgeObjectId, Generation),
    Transition(KnowledgeObjectId, KnowledgeLifecycle),
    Invalidate(KnowledgeObjectId),
    RestoreCurrent(KnowledgeObjectId, Generation),
    RestoreLifecycle(LifecycleRecord),
    RestoreRevision(KnowledgeObjectId, u64),
}

/// Journaled knowledge store.
pub struct DurableKnowledgeStore {
    store: KnowledgeStore,
    journal: MemoryJournal,
    options: DurableOptions,
    poisoned: bool,
    recovered_truncated_bytes: u64,
    compaction_failures: u64,
    last_compaction_error: Option<DurableKnowledgeError>,
}

impl DurableKnowledgeStore {
    /// Opens the journal at `path` (creating it) and replays it into a fresh store.
    pub fn open(
        path: impl AsRef<Path>,
        options: DurableOptions,
    ) -> Result<Self, DurableKnowledgeError> {
        let (journal, recovery) =
            MemoryJournal::open(path, JournalKind::Knowledge, options.journal)?;
        let mut store = KnowledgeStore::default();
        for (record_index, payload) in recovery.records.iter().enumerate() {
            let op = decode_op(payload).map_err(|error| DurableKnowledgeError::Decode {
                record_index,
                error,
            })?;
            apply_replay(&mut store, op).map_err(|error| {
                DurableKnowledgeError::ReplayDiverged {
                    record_index,
                    error,
                }
            })?;
        }
        Ok(Self {
            store,
            journal,
            options,
            poisoned: false,
            recovered_truncated_bytes: recovery.truncated_bytes,
            compaction_failures: 0,
            last_compaction_error: None,
        })
    }

    /// Number of automatic compactions that failed after their triggering mutation was already
    /// durable. Such a failure never fails the mutation; compaction is retried on later commits.
    pub fn compaction_failures(&self) -> u64 {
        self.compaction_failures
    }

    /// The most recent automatic-compaction failure, cleared by the next successful compaction.
    pub fn last_compaction_error(&self) -> Option<&DurableKnowledgeError> {
        self.last_compaction_error.as_ref()
    }

    /// Read-only access to the wrapped store (all query methods of [`KnowledgeStore`]).
    pub fn store(&self) -> &KnowledgeStore {
        &self.store
    }

    /// Bytes of a torn tail removed when this store was opened.
    pub fn recovered_truncated_bytes(&self) -> u64 {
        self.recovered_truncated_bytes
    }

    pub fn state_digest(&self) -> [u8; 32] {
        self.store.state_digest()
    }

    fn guard(&self) -> Result<(), DurableKnowledgeError> {
        if self.poisoned {
            Err(DurableKnowledgeError::Poisoned)
        } else {
            Ok(())
        }
    }

    fn persist(&mut self, op: &Op) -> Result<(), DurableKnowledgeError> {
        if let Err(error) = self.journal.append(&encode_op(op)) {
            // The in-memory store already applied the operation; make reads agree with the
            // journal again before reporting the failure.
            self.rebuild_from_journal();
            if matches!(error, JournalError::Io { .. } | JournalError::Poisoned) {
                self.poisoned = true;
            }
            return Err(error.into());
        }
        if let Some(limit) = self.options.compact_after_records {
            if self.journal.appended_since_rewrite() >= limit {
                // The operation is already journaled and applied: a failed rewrite must not be
                // reported as a failed operation. Record it; the next commit retries.
                if let Err(error) = self.compact() {
                    self.compaction_failures += 1;
                    self.last_compaction_error = Some(error);
                }
            }
        }
        Ok(())
    }

    /// Replaces the in-memory store with a replay of the intact journal records, so that after a
    /// failed append no reader sees a change the caller was told did not happen. If the journal
    /// cannot be read or replayed, the store is emptied and the wrapper stays poisoned rather
    /// than exposing state that may be ahead of disk.
    fn rebuild_from_journal(&mut self) {
        let rebuilt = self.journal.read_records().ok().and_then(|records| {
            let mut store = KnowledgeStore::default();
            for payload in &records {
                apply_replay(&mut store, decode_op(payload).ok()?).ok()?;
            }
            Some(store)
        });
        match rebuilt {
            Some(store) => self.store = store,
            None => {
                self.store = KnowledgeStore::default();
                self.poisoned = true;
            }
        }
    }

    /// Appends immutable raw history. Returns whether the event was new.
    pub fn append_raw(&mut self, event: RawEvent) -> Result<bool, DurableKnowledgeError> {
        self.guard()?;
        let inserted = self
            .store
            .append_raw(event.clone())
            .map_err(DurableKnowledgeError::Store)?;
        if inserted {
            self.persist(&Op::Raw(event))?;
        }
        Ok(inserted)
    }

    pub fn register_object(
        &mut self,
        object: KnowledgeObject,
    ) -> Result<(), DurableKnowledgeError> {
        self.guard()?;
        self.store
            .register_object(object.clone())
            .map_err(DurableKnowledgeError::Store)?;
        self.persist(&Op::Register(object))
    }

    pub fn admit_object(&mut self, object: KnowledgeObject) -> Result<(), DurableKnowledgeError> {
        self.guard()?;
        self.store
            .admit_object(object.clone())
            .map_err(DurableKnowledgeError::Store)?;
        self.persist(&Op::Admit(object))
    }

    pub fn activate_generation(
        &mut self,
        id: &KnowledgeObjectId,
        generation: Generation,
    ) -> Result<(), DurableKnowledgeError> {
        self.guard()?;
        self.store
            .activate_generation(id, generation)
            .map_err(DurableKnowledgeError::Store)?;
        self.persist(&Op::Activate(id.clone(), generation))
    }

    pub fn transition(
        &mut self,
        id: &KnowledgeObjectId,
        to: KnowledgeLifecycle,
    ) -> Result<(), DurableKnowledgeError> {
        self.guard()?;
        self.store
            .transition(id, to)
            .map_err(DurableKnowledgeError::Store)?;
        self.persist(&Op::Transition(id.clone(), to))
    }

    pub fn invalidate(&mut self, id: KnowledgeObjectId) -> Result<(), DurableKnowledgeError> {
        self.guard()?;
        self.store
            .invalidate(id.clone())
            .map_err(DurableKnowledgeError::Store)?;
        self.persist(&Op::Invalidate(id))
    }

    /// Rewrites the journal as a minimal snapshot of the current state (temp file + rename).
    /// State after reopening is identical to the state before compaction.
    pub fn compact(&mut self) -> Result<(), DurableKnowledgeError> {
        self.guard()?;
        let export = self.store.export();
        let mut ops: Vec<Op> = Vec::new();
        ops.extend(export.raw.into_iter().map(Op::Raw));
        ops.extend(export.objects.into_iter().map(Op::Register));
        ops.extend(
            export
                .current
                .into_iter()
                .map(|(id, g)| Op::RestoreCurrent(id, g)),
        );
        ops.extend(export.lifecycle.into_iter().map(Op::RestoreLifecycle));
        ops.extend(
            export
                .revisions
                .into_iter()
                .map(|(id, r)| Op::RestoreRevision(id, r)),
        );
        let payloads: Vec<Vec<u8>> = ops.iter().map(encode_op).collect();
        match self.journal.rewrite(&payloads) {
            Ok(()) => {
                self.last_compaction_error = None;
                Ok(())
            }
            Err(JournalError::Poisoned) => Err(DurableKnowledgeError::Poisoned),
            Err(error) => Err(error.into()),
        }
    }
}

fn apply_replay(store: &mut KnowledgeStore, op: Op) -> Result<(), KnowledgeStoreError> {
    match op {
        Op::Raw(event) => store.append_raw(event).map(|_| ()),
        Op::Register(object) => store.register_object(object),
        Op::Admit(object) => store.admit_object(object),
        Op::Activate(id, generation) => store.activate_generation(&id, generation),
        Op::Transition(id, to) => store.transition(&id, to),
        Op::Invalidate(id) => store.invalidate(id),
        Op::RestoreCurrent(id, generation) => {
            store.restore_current(id, generation);
            Ok(())
        }
        Op::RestoreLifecycle(record) => {
            store.restore_lifecycle(record);
            Ok(())
        }
        Op::RestoreRevision(id, revision) => {
            store.restore_revision(id, revision);
            Ok(())
        }
    }
}

fn role_tag(role: EventRole) -> u8 {
    match role {
        EventRole::System => 0,
        EventRole::User => 1,
        EventRole::Assistant => 2,
        EventRole::Tool => 3,
    }
}

fn role_from(tag: u8) -> Result<EventRole, CodecError> {
    match tag {
        0 => Ok(EventRole::System),
        1 => Ok(EventRole::User),
        2 => Ok(EventRole::Assistant),
        3 => Ok(EventRole::Tool),
        tag => Err(CodecError::InvalidTag {
            field: "event role",
            tag,
        }),
    }
}

fn lifecycle_from(tag: u8) -> Result<KnowledgeLifecycle, CodecError> {
    match tag {
        0 => Ok(KnowledgeLifecycle::Hot),
        1 => Ok(KnowledgeLifecycle::Warm),
        2 => Ok(KnowledgeLifecycle::Cold),
        3 => Ok(KnowledgeLifecycle::Pod),
        4 => Ok(KnowledgeLifecycle::Archived),
        5 => Ok(KnowledgeLifecycle::Invalidated),
        tag => Err(CodecError::InvalidTag {
            field: "lifecycle",
            tag,
        }),
    }
}

fn freshness_tag(value: Freshness) -> u8 {
    match value {
        Freshness::Current => 0,
        Freshness::Aging => 1,
        Freshness::Stale => 2,
    }
}

fn freshness_from(tag: u8) -> Result<Freshness, CodecError> {
    match tag {
        0 => Ok(Freshness::Current),
        1 => Ok(Freshness::Aging),
        2 => Ok(Freshness::Stale),
        tag => Err(CodecError::InvalidTag {
            field: "freshness",
            tag,
        }),
    }
}

fn conflict_tag(value: ConflictState) -> u8 {
    match value {
        ConflictState::None => 0,
        ConflictState::Suspected => 1,
        ConflictState::Confirmed => 2,
    }
}

fn conflict_from(tag: u8) -> Result<ConflictState, CodecError> {
    match tag {
        0 => Ok(ConflictState::None),
        1 => Ok(ConflictState::Suspected),
        2 => Ok(ConflictState::Confirmed),
        tag => Err(CodecError::InvalidTag {
            field: "conflict",
            tag,
        }),
    }
}

fn write_key(w: &mut Writer, key: &KnowledgeKey) {
    w.str(&key.logical_id.0);
    w.u64(key.generation.0);
}

fn read_key(r: &mut Reader<'_>) -> Result<KnowledgeKey, CodecError> {
    Ok(KnowledgeKey {
        logical_id: KnowledgeObjectId(r.string()?),
        generation: Generation(r.u64()?),
    })
}

fn write_event(w: &mut Writer, event: &RawEvent) {
    w.str(&event.id.0);
    w.str(&event.session.0);
    w.u8(role_tag(event.role));
    w.bytes(&event.content);
    w.u64(event.timestamp.0);
    w.raw(&event.digest);
    match &event.tool_call {
        Some(call) => {
            w.bool(true);
            w.str(&call.name);
            w.bytes(&call.arguments);
        }
        None => w.bool(false),
    }
    match &event.tool_result {
        Some(result) => {
            w.bool(true);
            w.bytes(&result.output);
        }
        None => w.bool(false),
    }
}

fn read_event(r: &mut Reader<'_>) -> Result<RawEvent, CodecError> {
    let id = RawEventId(r.string()?);
    let session = SessionId(r.string()?);
    let role = role_from(r.u8()?)?;
    let content = r.bytes()?;
    let timestamp = Timestamp(r.u64()?);
    let digest = r.array32()?;
    let tool_call = if r.bool()? {
        Some(ToolCall {
            name: r.string()?,
            arguments: r.bytes()?,
        })
    } else {
        None
    };
    let tool_result = if r.bool()? {
        Some(ToolResult { output: r.bytes()? })
    } else {
        None
    };
    Ok(RawEvent {
        id,
        session,
        role,
        content,
        timestamp,
        digest,
        tool_call,
        tool_result,
    })
}

fn write_strings<'a>(w: &mut Writer, items: impl ExactSizeIterator<Item = &'a String>) {
    w.count(items.len());
    for item in items {
        w.str(item);
    }
}

fn read_strings(r: &mut Reader<'_>) -> Result<Vec<String>, CodecError> {
    let count = r.count()?;
    (0..count).map(|_| r.string()).collect()
}

fn write_object(w: &mut Writer, object: &KnowledgeObject) {
    w.str(&object.id.0);
    w.str(&object.namespace.0);
    write_strings(w, object.entities.iter().map(|e| &e.0));
    w.count(object.relations.len());
    for relation in &object.relations {
        w.str(&relation.predicate);
        w.str(&relation.target.0);
    }
    write_strings(w, object.sources.iter().map(|s| &s.0));
    w.count(object.provenance.len());
    for provenance in &object.provenance {
        w.str(&provenance.source.0);
        w.opt_str(provenance.note.as_deref());
    }
    w.u64(object.generation.0);
    match &object.supersedes {
        Some(key) => {
            w.bool(true);
            write_key(w, key);
        }
        None => w.bool(false),
    }
    w.count(object.dependencies.len());
    for key in &object.dependencies {
        write_key(w, key);
    }
    w.f32(object.confidence.get());
    w.f32(object.relevance.get());
    w.u8(freshness_tag(object.freshness));
    w.u8(conflict_tag(object.conflict));
    w.u8(crate::knowledge::lifecycle_tag(object.lifecycle));
    write_strings(w, object.retrieval_keys.iter().map(|k| &k.0));
    w.u32(object.token_cost);
}

fn read_probability(r: &mut Reader<'_>) -> Result<Probability, CodecError> {
    Probability::new(r.f32()?).ok_or(CodecError::InvalidValue("probability outside 0..=1"))
}

fn read_object(r: &mut Reader<'_>) -> Result<KnowledgeObject, CodecError> {
    let id = KnowledgeObjectId(r.string()?);
    let namespace = NamespaceId(r.string()?);
    let entities = read_strings(r)?.into_iter().map(EntityId).collect();
    let relation_count = r.count()?;
    let mut relations = Vec::with_capacity(relation_count);
    for _ in 0..relation_count {
        relations.push(Relation {
            predicate: r.string()?,
            target: KnowledgeObjectId(r.string()?),
        });
    }
    let sources = read_strings(r)?.into_iter().map(RawEventId).collect();
    let provenance_count = r.count()?;
    let mut provenance = Vec::with_capacity(provenance_count);
    for _ in 0..provenance_count {
        provenance.push(ProvenanceRef {
            source: ptr_types::EvidenceId(r.string()?),
            note: r.opt_string()?,
        });
    }
    let generation = Generation(r.u64()?);
    let supersedes = if r.bool()? { Some(read_key(r)?) } else { None };
    let dependency_count = r.count()?;
    let mut dependencies = Vec::with_capacity(dependency_count);
    for _ in 0..dependency_count {
        dependencies.push(read_key(r)?);
    }
    let confidence = read_probability(r)?;
    let relevance = read_probability(r)?;
    let freshness = freshness_from(r.u8()?)?;
    let conflict = conflict_from(r.u8()?)?;
    let lifecycle = lifecycle_from(r.u8()?)?;
    let retrieval_keys = read_strings(r)?.into_iter().map(RetrievalKey).collect();
    let token_cost = r.u32()?;
    Ok(KnowledgeObject {
        id,
        namespace,
        entities,
        relations,
        sources,
        provenance,
        generation,
        supersedes,
        dependencies,
        confidence,
        relevance,
        freshness,
        conflict,
        lifecycle,
        retrieval_keys,
        token_cost,
    })
}

fn encode_op(op: &Op) -> Vec<u8> {
    let mut w = Writer::new();
    match op {
        Op::Raw(event) => {
            w.u8(1);
            write_event(&mut w, event);
        }
        Op::Register(object) => {
            w.u8(2);
            write_object(&mut w, object);
        }
        Op::Admit(object) => {
            w.u8(3);
            write_object(&mut w, object);
        }
        Op::Activate(id, generation) => {
            w.u8(4);
            w.str(&id.0);
            w.u64(generation.0);
        }
        Op::Transition(id, to) => {
            w.u8(5);
            w.str(&id.0);
            w.u8(crate::knowledge::lifecycle_tag(*to));
        }
        Op::Invalidate(id) => {
            w.u8(6);
            w.str(&id.0);
        }
        Op::RestoreCurrent(id, generation) => {
            w.u8(7);
            w.str(&id.0);
            w.u64(generation.0);
        }
        Op::RestoreLifecycle(record) => {
            w.u8(8);
            write_key(&mut w, &record.key);
            w.u8(crate::knowledge::lifecycle_tag(record.lifecycle));
            w.u64(record.revision);
        }
        Op::RestoreRevision(id, revision) => {
            w.u8(9);
            w.str(&id.0);
            w.u64(*revision);
        }
    }
    w.finish()
}

fn decode_op(payload: &[u8]) -> Result<Op, CodecError> {
    let mut r = Reader::new(payload);
    let op = match r.u8()? {
        1 => Op::Raw(read_event(&mut r)?),
        2 => Op::Register(read_object(&mut r)?),
        3 => Op::Admit(read_object(&mut r)?),
        4 => Op::Activate(KnowledgeObjectId(r.string()?), Generation(r.u64()?)),
        5 => Op::Transition(KnowledgeObjectId(r.string()?), lifecycle_from(r.u8()?)?),
        6 => Op::Invalidate(KnowledgeObjectId(r.string()?)),
        7 => Op::RestoreCurrent(KnowledgeObjectId(r.string()?), Generation(r.u64()?)),
        8 => {
            let key = read_key(&mut r)?;
            let lifecycle = lifecycle_from(r.u8()?)?;
            Op::RestoreLifecycle(LifecycleRecord {
                key,
                lifecycle,
                revision: r.u64()?,
            })
        }
        9 => Op::RestoreRevision(KnowledgeObjectId(r.string()?), r.u64()?),
        tag => {
            return Err(CodecError::InvalidTag {
                field: "knowledge op",
                tag,
            })
        }
    };
    r.finish()?;
    Ok(op)
}
