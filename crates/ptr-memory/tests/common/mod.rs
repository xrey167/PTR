#![allow(dead_code)]
//! Shared helpers for the integration tests.

use ptr_memory::{
    ConflictState, Freshness, KnowledgeKey, KnowledgeLifecycle, KnowledgeObject, MemoryClass,
    MemoryRecordId, NewMemoryRecord, RawEvent, Support, SupportKind,
};
use ptr_types::{
    EntityId, EventRole, Generation, KnowledgeObjectId, NamespaceId, Probability, RawEventId,
    Revision, SessionId, Timestamp,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique temp journal path, removed (with the compaction sibling) on drop.
pub struct TempJournal(pub PathBuf);

impl TempJournal {
    pub fn new(tag: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "ptr-memory-it-{tag}-{}-{n}.journal",
            std::process::id()
        )))
    }
}

impl Drop for TempJournal {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let mut tmp = self.0.as_os_str().to_owned();
        tmp.push(".compact.tmp");
        let _ = std::fs::remove_file(PathBuf::from(tmp));
    }
}

pub fn raw_event(id: &str, content: &str) -> RawEvent {
    RawEvent::new(
        RawEventId::from(id),
        SessionId::from("s1"),
        EventRole::User,
        content.as_bytes().to_vec(),
        Timestamp(1),
    )
}

pub fn knowledge_object(
    id: &str,
    generation: u64,
    sources: &[&str],
    dependencies: Vec<KnowledgeKey>,
) -> KnowledgeObject {
    KnowledgeObject {
        id: KnowledgeObjectId::from(id),
        namespace: NamespaceId::from("ptr.test"),
        entities: vec![EntityId::from("entity")],
        relations: vec![],
        sources: sources.iter().map(|s| RawEventId::from(*s)).collect(),
        provenance: vec![],
        generation: Generation(generation),
        supersedes: (generation > 1).then(|| KnowledgeKey {
            logical_id: KnowledgeObjectId::from(id),
            generation: Generation(generation - 1),
        }),
        dependencies,
        confidence: Probability::new(0.75).unwrap(),
        relevance: Probability::new(0.5).unwrap(),
        freshness: Freshness::Current,
        conflict: ConflictState::None,
        lifecycle: KnowledgeLifecycle::Hot,
        retrieval_keys: vec![],
        token_cost: 7,
    }
}

pub fn key(id: &str, generation: u64) -> KnowledgeKey {
    KnowledgeKey {
        logical_id: KnowledgeObjectId::from(id),
        generation: Generation(generation),
    }
}

pub fn rid(id: &str) -> MemoryRecordId {
    MemoryRecordId::from(id)
}

pub fn observation(source: &str, revision: u64) -> Support {
    Support {
        source: source.to_owned(),
        kind: SupportKind::Observation {
            revision: Revision(revision),
        },
    }
}

pub fn derived(source: &str) -> Support {
    Support {
        source: source.to_owned(),
        kind: SupportKind::Derived,
    }
}

pub fn new_record(id: &str, class: MemoryClass, supports: Vec<Support>) -> NewMemoryRecord {
    NewMemoryRecord {
        id: rid(id),
        class,
        body: format!("body of {id}"),
        sensitive: false,
        pinned: false,
        supports,
        derived_from: vec![],
    }
}

pub fn derived_record(id: &str, class: MemoryClass, parents: &[&str]) -> NewMemoryRecord {
    NewMemoryRecord {
        derived_from: parents.iter().map(|p| rid(p)).collect(),
        ..new_record(id, class, vec![])
    }
}

/// Small deterministic generator for property-style sequences.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}
