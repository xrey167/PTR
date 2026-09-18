use ptr_types::{CapsuleId, CommitIndex, Generation, ProjectId};

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub struct FileLedger {
    path: PathBuf,
    file: File,
    events: Vec<CommittedEvent>,
}

impl FileLedger {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;

        file.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let mut offset = 0usize;
        let mut last_good = 0usize;
        let mut events = Vec::new();

        while offset < bytes.len() {
            if bytes.len() - offset < 4 {
                break;
            }
            let length = u32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .expect("four byte record prefix"),
            ) as usize;
            offset += 4;

            if bytes.len() - offset < length {
                offset -= 4;
                break;
            }

            let payload = &bytes[offset..offset + length];
            let event = decode_event(payload)?;
            let index = CommitIndex(events.len() as u64 + 1);
            events.push(CommittedEvent { index, event });
            offset += length;
            last_good = offset;
        }

        if last_good != bytes.len() {
            file.set_len(last_good as u64)?;
            file.sync_data()?;
        }
        file.seek(SeekFrom::End(0))?;

        Ok(Self { path, file, events })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn events(&self) -> &[CommittedEvent] {
        &self.events
    }

    pub fn append_durable(&mut self, event: LedgerEvent) -> io::Result<CommitIndex> {
        let payload = encode_event(&event);
        let length = u32::try_from(payload.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "ledger record too large"))?;

        self.file.write_all(&length.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.flush()?;
        self.file.sync_data()?;

        let index = CommitIndex(self.events.len() as u64 + 1);
        self.events.push(CommittedEvent { index, event });
        Ok(index)
    }
}

fn encode_event(event: &LedgerEvent) -> Vec<u8> {
    let mut out = Vec::new();
    match event {
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
    let bytes = value.as_bytes();
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

    fn string(&mut self) -> io::Result<String> {
        let length = self.u32()? as usize;
        let end = self.offset.saturating_add(length);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 string"))
    }
}

fn truncated() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "truncated ledger record")
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LedgerEvent {
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
