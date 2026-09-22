//! The identity a stored model artifact has to carry with it.
//!
//! Weights are a table of numbers whose meaning is entirely external to them. A
//! checkpoint trained when code 4 meant `Resource` keeps loading after code 4 has
//! come to mean something else: the tensors have the right shape, the outputs look
//! plausible, and the model is now reading a different taxonomy than the one it
//! learned. Nothing fails.
//!
//! So an artifact records the assignment it was produced under, and a reader that
//! disagrees refuses. The assignment is stored **verbatim**, not as a digest: this
//! crate has no dependencies and therefore no hash, and a byte comparison is
//! exact where a digest is only probably exact. A reader that wants a fingerprint
//! hashes [`Codebook::canonical_bytes`] itself, which is what `ptr-runtime` does
//! to fill a `StateBinding`.
//!
//! This is a commitment, not authentication. It detects a mismatch between an
//! artifact and a build. It cannot detect someone who rewrites the header, and
//! nothing here should be read as claiming otherwise.
use crate::{CodeFamily, Codebook, CodebookVersion};
use std::fmt;

/// Marks the start of a PTR checkpoint. Eight bytes so a truncated or foreign
/// file is rejected before anything else is interpreted.
const MAGIC: &[u8; 8] = b"PTRCKPT\x00";

/// The only header layout this build writes and reads.
pub const FORMAT_V1: u16 = 1;

/// Why a checkpoint was refused.
///
/// Every variant is a refusal, and none of them falls back to a default. A
/// checkpoint that loads under the wrong assignment is the failure this module
/// exists to prevent, so "close enough" is not available.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckpointError {
    /// The bytes do not begin with the checkpoint magic.
    NotACheckpoint,
    /// A header layout this build does not know. Not guessed at: a layout is not
    /// forward-compatible just because its first fields happen to parse.
    UnknownFormat { format: u16 },
    /// The bytes end inside a field.
    Truncated { field: &'static str },
    /// A string field is not UTF-8.
    NotUtf8 { field: &'static str },
    /// Bytes after the declared payload. Refused rather than ignored, because
    /// trailing data means the writer and the reader disagree about the layout.
    TrailingBytes { extra: usize },
    /// The declared payload length does not match what is there.
    PayloadLength { declared: u64, found: usize },
    /// A code family this build does not define.
    UnknownFamily { name: String },
    /// The same family recorded twice, so its size is ambiguous.
    DuplicateFamily { family: CodeFamily },
    /// A codebook version this build has no tables for.
    UnknownCodebookVersion { version: CodebookVersion },
    /// The stored assignment differs from this build's assignment for the same
    /// version. Tables were edited without a version bump, and every code in the
    /// artifact now means something this build cannot reproduce.
    CodebookMoved { version: CodebookVersion },
    /// The artifact's table for a family is not the size this build's codebook
    /// requires. A table shorter than its family cannot represent its last
    /// members; a longer one carries rows that denote nothing.
    TableSize {
        family: CodeFamily,
        stored: u16,
        required: u16,
    },
    /// A family the reader requires is not recorded at all.
    MissingTable { family: CodeFamily },
}

impl CheckpointError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotACheckpoint => "PTR_CKPT_NOT_A_CHECKPOINT",
            Self::UnknownFormat { .. } => "PTR_CKPT_UNKNOWN_FORMAT",
            Self::Truncated { .. } => "PTR_CKPT_TRUNCATED",
            Self::NotUtf8 { .. } => "PTR_CKPT_NOT_UTF8",
            Self::TrailingBytes { .. } => "PTR_CKPT_TRAILING_BYTES",
            Self::PayloadLength { .. } => "PTR_CKPT_PAYLOAD_LENGTH",
            Self::UnknownFamily { .. } => "PTR_CKPT_UNKNOWN_FAMILY",
            Self::DuplicateFamily { .. } => "PTR_CKPT_DUPLICATE_FAMILY",
            Self::UnknownCodebookVersion { .. } => "PTR_CKPT_UNKNOWN_CODEBOOK_VERSION",
            Self::CodebookMoved { .. } => "PTR_CKPT_CODEBOOK_MOVED",
            Self::TableSize { .. } => "PTR_CKPT_TABLE_SIZE",
            Self::MissingTable { .. } => "PTR_CKPT_MISSING_TABLE",
        }
    }
}

impl fmt::Display for CheckpointError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for CheckpointError {}

/// One embedded code family and the table size the weights were built for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TableSize {
    pub family: CodeFamily,
    pub rows: u16,
}

/// The identity recorded in front of a checkpoint's weights.
///
/// Not a description of the architecture: only what a reader needs in order to
/// decide whether the codes inside the weights still mean what they meant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointHeader {
    /// Which model wrote this, as a stable name.
    pub model: String,
    /// Codebook version the codes belong to.
    pub codebook: CodebookVersion,
    /// That version's complete assignment, verbatim, as
    /// [`Codebook::canonical_bytes`] produced it.
    pub codebook_bytes: Vec<u8>,
    /// Embedded table sizes, in the order the writer recorded them.
    pub tables: Vec<TableSize>,
}

impl CheckpointHeader {
    /// Record the assignment a model's tables were built from.
    ///
    /// `tables` are the sizes the weights actually have, so they are supplied by
    /// the writer rather than derived here: deriving them from the codebook would
    /// make the header agree with the codebook by construction and check nothing.
    pub fn new(model: &str, book: &Codebook, tables: &[TableSize]) -> Self {
        Self {
            model: model.to_owned(),
            codebook: book.version(),
            codebook_bytes: book.canonical_bytes(),
            tables: tables.to_vec(),
        }
    }

    /// The recorded size of one family's table.
    pub fn table(&self, family: CodeFamily) -> Option<u16> {
        self.tables
            .iter()
            .find(|table| table.family == family)
            .map(|table| table.rows)
    }

    /// Check this header against a build's codebook.
    ///
    /// `required` names the families the reader expects to find, each of which
    /// must be recorded and must be exactly its cardinality in `book`. A family
    /// the reader does not name is left alone: a model that embeds fewer families
    /// than the kernel defines is not thereby wrong.
    pub fn verify(&self, book: &Codebook, required: &[CodeFamily]) -> Result<(), CheckpointError> {
        if self.codebook != book.version() {
            return Err(CheckpointError::UnknownCodebookVersion {
                version: self.codebook,
            });
        }
        if self.codebook_bytes != book.canonical_bytes() {
            return Err(CheckpointError::CodebookMoved {
                version: self.codebook,
            });
        }
        for family in required {
            let stored = self
                .table(*family)
                .ok_or(CheckpointError::MissingTable { family: *family })?;
            let cardinality = book.cardinality(*family);
            if stored != cardinality {
                return Err(CheckpointError::TableSize {
                    family: *family,
                    stored,
                    required: cardinality,
                });
            }
        }
        Ok(())
    }

    /// Serialize the header followed by `payload`.
    pub fn write(&self, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(payload.len() + 128);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_V1.to_le_bytes());
        put_str(&mut out, &self.model);
        out.extend_from_slice(&self.codebook.0.to_le_bytes());
        out.extend_from_slice(&(self.codebook_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.codebook_bytes);
        out.extend_from_slice(&(self.tables.len() as u16).to_le_bytes());
        for table in &self.tables {
            put_str(&mut out, table.family.name());
            out.extend_from_slice(&table.rows.to_le_bytes());
        }
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// Read a header and its payload.
    ///
    /// The header is parsed but not judged; call [`CheckpointHeader::verify`] to
    /// decide whether this build may interpret the payload at all.
    pub fn read(bytes: &[u8]) -> Result<(Self, &[u8]), CheckpointError> {
        let mut cursor = Cursor { bytes, at: 0 };
        if cursor.take(MAGIC.len(), "magic")? != MAGIC {
            return Err(CheckpointError::NotACheckpoint);
        }
        let format = cursor.u16("format")?;
        if format != FORMAT_V1 {
            return Err(CheckpointError::UnknownFormat { format });
        }
        let model = cursor.string("model")?;
        let codebook = CodebookVersion(cursor.u32("codebook_version")?);
        let assignment_len = cursor.u32("codebook_bytes_len")? as usize;
        let codebook_bytes = cursor.take(assignment_len, "codebook_bytes")?.to_vec();
        let table_count = cursor.u16("table_count")? as usize;
        let mut tables: Vec<TableSize> = Vec::with_capacity(table_count);
        for _ in 0..table_count {
            let name = cursor.string("table_family")?;
            let family =
                CodeFamily::from_name(&name).ok_or(CheckpointError::UnknownFamily { name })?;
            if tables.iter().any(|table| table.family == family) {
                return Err(CheckpointError::DuplicateFamily { family });
            }
            tables.push(TableSize {
                family,
                rows: cursor.u16("table_rows")?,
            });
        }
        let declared = cursor.u64("payload_len")?;
        let remaining = cursor.remaining();
        if remaining.len() as u64 != declared {
            if (remaining.len() as u64) < declared {
                return Err(CheckpointError::PayloadLength {
                    declared,
                    found: remaining.len(),
                });
            }
            return Err(CheckpointError::TrailingBytes {
                extra: remaining.len() - declared as usize,
            });
        }
        Ok((
            Self {
                model,
                codebook,
                codebook_bytes,
                tables,
            },
            remaining,
        ))
    }
}

/// Append a length-prefixed string.
fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u16).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

/// A position in the byte stream that refuses to read past the end.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    /// Take exactly `len` bytes, or say which field ran out.
    fn take(&mut self, len: usize, field: &'static str) -> Result<&'a [u8], CheckpointError> {
        let end = self
            .at
            .checked_add(len)
            .ok_or(CheckpointError::Truncated { field })?;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or(CheckpointError::Truncated { field })?;
        self.at = end;
        Ok(slice)
    }

    /// Read a little-endian `u16`.
    fn u16(&mut self, field: &'static str) -> Result<u16, CheckpointError> {
        let bytes = self.take(2, field)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    /// Read a little-endian `u32`.
    fn u32(&mut self, field: &'static str) -> Result<u32, CheckpointError> {
        let bytes = self.take(4, field)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a little-endian `u64`.
    fn u64(&mut self, field: &'static str) -> Result<u64, CheckpointError> {
        let bytes = self.take(8, field)?;
        let mut value = [0_u8; 8];
        value.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(value))
    }

    /// Read a length-prefixed UTF-8 string.
    fn string(&mut self, field: &'static str) -> Result<String, CheckpointError> {
        let len = self.u16(field)? as usize;
        let bytes = self.take(len, field)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| CheckpointError::NotUtf8 { field })
    }

    /// Everything not yet read.
    fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.at..]
    }
}
