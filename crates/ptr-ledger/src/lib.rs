pub mod acknowledged;
pub mod anchor;
mod file;
pub mod integrity;
pub use acknowledged::{AcknowledgedError, AcknowledgedLedger, Split, TailPolicy, TailRecovery};
pub use file::{FileLedger, LegacyLog, RecoverableLog};

use ptr_types::{CapsuleId, CommitIndex, Generation, ProjectId, Revision};
use std::io;
#[cfg(feature = "raft-engine-backend")]
use std::path::Path;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LedgerEvent {
    /// Opaque versioned semantic transaction. ptr-runtime validates its schema
    /// and both revisions before append and again during replay.
    SemanticDeltaCommitted {
        base_revision: Revision,
        revision: Revision,
        encoded_delta: Vec<u8>,
    },
    CapsuleCommitted {
        project: ProjectId,
        capsule: CapsuleId,
        generation: Generation,
    },
    CapsuleSuperseded {
        capsule: CapsuleId,
        old: Generation,
        new: Generation,
    },
    Revoked {
        subject: String,
        generation: Generation,
    },
    HardConstraintCommitted {
        key: String,
        generation: Generation,
    },
    VerifierAttested {
        subject: String,
        passed: bool,
    },
    ProcedurePromoted {
        id: String,
        generation: Generation,
    },
    ProcedureRevoked {
        id: String,
        generation: Generation,
    },
    SnapshotCommitted {
        revision: u64,
        covers: CommitIndex,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedEvent {
    pub index: CommitIndex,
    pub event: LedgerEvent,
}

pub trait Ledger {
    fn append(&mut self, event: LedgerEvent) -> CommitIndex;
    fn events(&self) -> &[CommittedEvent];
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryLedger {
    events: Vec<CommittedEvent>,
}

impl Ledger for InMemoryLedger {
    fn append(&mut self, event: LedgerEvent) -> CommitIndex {
        let index = CommitIndex(self.events.len() as u64 + 1);
        self.events.push(CommittedEvent { index, event });
        index
    }

    fn events(&self) -> &[CommittedEvent] {
        &self.events
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactionBarrier {
    pub snapshot_covers: CommitIndex,
    pub all_consumers_caught_up: bool,
    pub unresolved_revocations: usize,
}

impl CompactionBarrier {
    pub fn safe(&self) -> bool {
        self.all_consumers_caught_up && self.unresolved_revocations == 0
    }
}

fn encode_event(event: &LedgerEvent) -> Vec<u8> {
    let mut out = Vec::new();
    match event {
        LedgerEvent::SemanticDeltaCommitted {
            base_revision,
            revision,
            encoded_delta,
        } => {
            out.push(8);
            put_u64(&mut out, base_revision.0);
            put_u64(&mut out, revision.0);
            put_bytes(&mut out, encoded_delta);
        }
        LedgerEvent::CapsuleCommitted {
            project,
            capsule,
            generation,
        } => {
            out.push(0);
            put_string(&mut out, &project.to_string());
            put_string(&mut out, &capsule.to_string());
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::CapsuleSuperseded { capsule, old, new } => {
            out.push(1);
            put_string(&mut out, &capsule.to_string());
            put_u64(&mut out, old.0);
            put_u64(&mut out, new.0);
        }
        LedgerEvent::Revoked {
            subject,
            generation,
        } => {
            out.push(2);
            put_string(&mut out, subject);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::HardConstraintCommitted { key, generation } => {
            out.push(3);
            put_string(&mut out, key);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::VerifierAttested { subject, passed } => {
            out.push(4);
            put_string(&mut out, subject);
            out.push(u8::from(*passed));
        }
        LedgerEvent::ProcedurePromoted { id, generation } => {
            out.push(5);
            put_string(&mut out, id);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::ProcedureRevoked { id, generation } => {
            out.push(6);
            put_string(&mut out, id);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::SnapshotCommitted { revision, covers } => {
            out.push(7);
            put_u64(&mut out, *revision);
            put_u64(&mut out, covers.0);
        }
    }
    out
}

fn decode_event(payload: &[u8]) -> io::Result<LedgerEvent> {
    let mut cursor = Cursor::new(payload);
    let tag = cursor.u8()?;
    let event = match tag {
        8 => LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            encoded_delta: cursor.bytes()?.to_vec(),
        },
        0 => LedgerEvent::CapsuleCommitted {
            project: ProjectId(cursor.string()?),
            capsule: CapsuleId(cursor.string()?),
            generation: Generation(cursor.u64()?),
        },
        1 => LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId(cursor.string()?),
            old: Generation(cursor.u64()?),
            new: Generation(cursor.u64()?),
        },
        2 => LedgerEvent::Revoked {
            subject: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        3 => LedgerEvent::HardConstraintCommitted {
            key: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        4 => LedgerEvent::VerifierAttested {
            subject: cursor.string()?,
            passed: match cursor.u8()? {
                0 => false,
                1 => true,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid bool byte {other}"),
                    ))
                }
            },
        },
        5 => LedgerEvent::ProcedurePromoted {
            id: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        6 => LedgerEvent::ProcedureRevoked {
            id: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        7 => LedgerEvent::SnapshotCommitted {
            revision: cursor.u64()?,
            covers: CommitIndex(cursor.u64()?),
        },
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown ledger event tag {other}"),
            ))
        }
    };

    if !cursor.finished() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "ledger event contains trailing bytes",
        ));
    }

    Ok(event)
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    let length = u32::try_from(bytes.len()).expect("string length fits u32");
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(bytes);
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn u8(&mut self) -> io::Result<u8> {
        let value = *self.bytes.get(self.offset).ok_or_else(truncated)?;
        self.offset += 1;
        Ok(value)
    }

    fn u32(&mut self) -> io::Result<u32> {
        let end = self.offset.saturating_add(4);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
    }

    fn u64(&mut self) -> io::Result<u64> {
        let end = self.offset.saturating_add(8);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
    }

    fn bytes(&mut self) -> io::Result<&'a [u8]> {
        let length = self.u32()? as usize;
        let end = self.offset.checked_add(length).ok_or_else(truncated)?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn string(&mut self) -> io::Result<String> {
        String::from_utf8(self.bytes()?.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 string"))
    }
}

fn truncated() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "truncated ledger record")
}

#[cfg(feature = "raft-engine-backend")]
mod raft_engine_backend {
    use super::*;
    use raft_engine::{Config, Engine, LogBatch};

    const GROUP_ID: u64 = 1;
    const KEY_PREFIX: &[u8] = b"ptr/event/";

    pub struct RaftEngineLedger {
        engine: Engine,
        events: Vec<CommittedEvent>,
    }

    impl RaftEngineLedger {
        pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
            let config = Config {
                dir: path.as_ref().to_string_lossy().into_owned(),
                ..Default::default()
            };
            let engine = Engine::open(config).map_err(engine_error)?;

            let mut encoded = Vec::<(Vec<u8>, Vec<u8>)>::new();
            engine
                .scan_raw_messages(GROUP_ID, None, None, false, |key, value| {
                    if key.starts_with(KEY_PREFIX) {
                        encoded.push((key.to_vec(), value.to_vec()));
                    }
                    true
                })
                .map_err(engine_error)?;
            encoded.sort_by(|left, right| left.0.cmp(&right.0));

            let mut events = Vec::with_capacity(encoded.len());
            for (offset, (key, payload)) in encoded.into_iter().enumerate() {
                let index = parse_event_key(&key)?;
                let expected = offset as u64 + 1;
                if index != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("raft-engine ledger gap: expected {expected}, found {index}"),
                    ));
                }
                events.push(CommittedEvent {
                    index: CommitIndex(index),
                    event: decode_event(&payload)?,
                });
            }

            Ok(Self { engine, events })
        }

        pub fn append_durable(&mut self, event: LedgerEvent) -> io::Result<CommitIndex> {
            let index = CommitIndex(self.events.len() as u64 + 1);
            let mut batch = LogBatch::default();
            batch.put(
                GROUP_ID,
                event_key(index).into_bytes(),
                encode_event(&event),
            );
            self.engine.write(&mut batch, true).map_err(engine_error)?;
            self.events.push(CommittedEvent { index, event });
            Ok(index)
        }

        pub fn events(&self) -> &[CommittedEvent] {
            &self.events
        }

        pub fn sync(&self) -> io::Result<()> {
            self.engine.sync().map_err(engine_error)
        }

        pub fn purge_expired_files(&self) -> io::Result<Vec<u64>> {
            self.engine.purge_expired_files().map_err(engine_error)
        }
    }

    fn event_key(index: CommitIndex) -> String {
        format!("ptr/event/{:020}", index.0)
    }

    fn parse_event_key(key: &[u8]) -> io::Result<u64> {
        if !key.starts_with(KEY_PREFIX) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected raft-engine ledger key prefix",
            ));
        }
        let suffix = std::str::from_utf8(&key[KEY_PREFIX.len()..])
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "ledger key is not UTF-8"))?;
        suffix
            .parse::<u64>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid ledger event index"))
    }

    fn engine_error(error: raft_engine::Error) -> io::Error {
        io::Error::other(error.to_string())
    }
}

#[cfg(feature = "raft-engine-backend")]
pub use raft_engine_backend::RaftEngineLedger;

#[cfg(feature = "raft-rs-backend")]
mod raft_rs_backend {
    use super::*;
    use raft::prelude::{ConfState, Config as RaftConfig, Entry, EntryType, RawNode};
    use raft::storage::MemStorage;
    use raft::StateRole;

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct RaftCommitReceipt {
        pub raft_index: u64,
        pub commit_index: CommitIndex,
    }

    pub struct SingleNodeRaftConsensus {
        node: RawNode<MemStorage>,
        committed: Vec<CommittedEvent>,
    }

    impl SingleNodeRaftConsensus {
        pub fn new(node_id: u64) -> Result<Self, String> {
            let storage = MemStorage::new_with_conf_state(ConfState::from((vec![node_id], vec![])));
            let config = RaftConfig {
                id: node_id,
                election_tick: 10,
                heartbeat_tick: 3,
                max_size_per_msg: 1024 * 1024,
                max_inflight_msgs: 256,
                applied: 0,
                ..Default::default()
            };
            config.validate().map_err(|error| error.to_string())?;

            let logger = slog::Logger::root(slog::Discard, slog::o!());
            let node =
                RawNode::new(&config, storage, &logger).map_err(|error| error.to_string())?;
            let mut consensus = Self {
                node,
                committed: Vec::new(),
            };
            consensus
                .node
                .campaign()
                .map_err(|error| error.to_string())?;
            consensus.drain_ready()?;
            if consensus.node.raft.state != StateRole::Leader {
                return Err("single-node raft group failed to become leader".into());
            }
            Ok(consensus)
        }

        pub fn is_leader(&self) -> bool {
            self.node.raft.state == StateRole::Leader
        }

        pub fn committed_events(&self) -> &[CommittedEvent] {
            &self.committed
        }

        pub fn propose(&mut self, event: LedgerEvent) -> Result<RaftCommitReceipt, String> {
            let before = self.committed.len();
            self.node
                .propose(Vec::new(), encode_event(&event))
                .map_err(|error| error.to_string())?;
            self.drain_ready()?;

            let committed = self
                .committed
                .get(before)
                .ok_or_else(|| "proposal was not committed in single-node raft group".to_owned())?;
            Ok(RaftCommitReceipt {
                raft_index: self.node.raft.raft_log.committed,
                commit_index: committed.index,
            })
        }

        fn drain_ready(&mut self) -> Result<(), String> {
            while self.node.has_ready() {
                let store = self.node.raft.raft_log.store.clone();
                let mut ready = self.node.ready();

                if !ready.messages().is_empty() {
                    ready.take_messages();
                }

                if !ready.snapshot().is_empty() {
                    store
                        .wl()
                        .apply_snapshot(ready.snapshot().clone())
                        .map_err(|error| error.to_string())?;
                }

                let committed = ready.take_committed_entries();

                if !ready.entries().is_empty() {
                    store
                        .wl()
                        .append(ready.entries())
                        .map_err(|error| error.to_string())?;
                }

                if let Some(hard_state) = ready.hs() {
                    store.wl().set_hardstate(hard_state.clone());
                }

                if !ready.persisted_messages().is_empty() {
                    ready.take_persisted_messages();
                }

                self.apply_committed(committed)?;

                let mut light = self.node.advance(ready);
                if let Some(commit) = light.commit_index() {
                    store.wl().mut_hard_state().set_commit(commit);
                }
                self.apply_committed(light.take_committed_entries())?;
                self.node.advance_apply();
            }
            Ok(())
        }

        fn apply_committed(&mut self, entries: Vec<Entry>) -> Result<(), String> {
            for entry in entries {
                if entry.data.is_empty() || entry.get_entry_type() != EntryType::EntryNormal {
                    continue;
                }
                let event = decode_event(entry.data.as_ref()).map_err(|error| error.to_string())?;
                let index = CommitIndex(self.committed.len() as u64 + 1);
                self.committed.push(CommittedEvent { index, event });
            }
            Ok(())
        }
    }
}

#[cfg(feature = "raft-rs-backend")]
pub use raft_rs_backend::{RaftCommitReceipt, SingleNodeRaftConsensus};
