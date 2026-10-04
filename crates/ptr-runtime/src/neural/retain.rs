//! Binding, sealing and writing a checkpoint as one operation with one outcome.
//!
//! A checkpoint carries the assignment it was trained under. It does not carry what
//! it was trained *from* — the committed position, the semantic values it read, the
//! generations that were live — because those are facts about a journal. Until this
//! existed the artifact and its binding were produced by different programs: a
//! trainer saved weights, and `ptrctl seal` bound them later, so a checkpoint could
//! sit on disk, and reach a loader, having never been bound at all.
//!
//! [`PtrRuntime::seal_checkpoint`] is the one place that joins them, and both the
//! command-line step and a trainer that saves as it trains call it, so there is one
//! rule for what a sealed save is rather than two copies that can drift.
//!
//! Three properties, each a way the step could have been written and is not:
//!
//! - **It refuses before it writes.** Binding and sealing happen in memory first. A
//!   declaration naming a key the journal does not hold, or an artifact whose
//!   assignment this build cannot reproduce, leaves nothing behind: a sealed file
//!   beside a checkpoint reads as a successful binding to anything that finds it.
//! - **Each file is installed whole.** It is written beside its destination,
//!   synchronized, renamed over it and the directory synchronized, so a crash leaves
//!   either the old file or the new one and never a prefix of either.
//! - **The set is undone when it cannot be completed.** Files cannot be renamed
//!   into place as a group, so the anchor goes last and a failure removes every file
//!   this call had already installed. The anchor's presence therefore marks a
//!   complete save: a sealed artifact whose anchor was never retained cannot be
//!   reopened, and a checkpoint with no seal is exactly what this exists to stop.
//!
//! The anchor is written to a **separate** file on purpose. A digest read back out
//! of the artifact it describes proves nothing; the anchor is evidence only while it
//! is retained somewhere the artifact cannot rewrite.
use super::{NeuralAnchor, StateDeclaration};
use crate::{PtrRuntime, RuntimeError};
use ptr_ledger::integrity::LogAnchor;
use ptr_types::{CodeFamily, CodebookVersion, CommitIndex};
use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

/// Where a sealed save goes.
#[derive(Clone, Copy, Debug)]
pub struct SealTargets<'a> {
    /// Where to write the checkpoint itself, when sealing is part of saving it.
    ///
    /// `None` when the checkpoint is already on disk and only its seal is being
    /// produced, which is what `ptrctl seal` does: it reads an artifact another
    /// program saved, and rewriting that file would make it the author of bytes it
    /// only checked.
    pub checkpoint: Option<&'a Path>,
    /// The sealed artifact: binding, payload and digest.
    pub sealed: &'a Path,
    /// The retained anchor, in the text form [`retained_anchor`] writes.
    pub anchor: &'a Path,
}

/// Why a checkpoint was not saved, bound and sealed.
///
/// The first two are refusals that leave nothing written. The third is a failure
/// part-way through writing, after which every file this call had installed has
/// been removed again.
#[derive(Debug)]
pub enum SealCheckpointError {
    /// The runtime refused to bind the artifact to its committed state.
    Bind(RuntimeError),
    /// The binding could not be serialized into a sealed state.
    Seal(RuntimeError),
    /// A file could not be written.
    Write { path: PathBuf, error: io::Error },
}

impl SealCheckpointError {
    /// Stable diagnostic code for this failure.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Bind(_) => "PTR_SEAL_UNBINDABLE",
            Self::Seal(_) => "PTR_SEAL_UNSEALABLE",
            Self::Write { .. } => "PTR_SEAL_WRITE",
        }
    }
}

impl std::fmt::Display for SealCheckpointError {
    /// Render the stable failure code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for SealCheckpointError {}

impl PtrRuntime {
    /// Bind `checkpoint` to this runtime's committed state, seal it, and write the
    /// sealed artifact and its anchor — and the checkpoint too, when
    /// [`SealTargets::checkpoint`] names where.
    ///
    /// Binding goes through [`PtrRuntime::bind_checkpoint`], so a sealed save is
    /// admitted by exactly the rules any other neural state is. `required` names the
    /// code families the caller needs the artifact to agree about, as there.
    ///
    /// Returns the anchor a catalog has to retain. Nothing is written unless binding
    /// and sealing both succeeded, and a write that fails removes what this call had
    /// already written. A destination that held a file before is not restored: what
    /// it held belonged to another save, and putting it back beside this save's
    /// other files would pair an artifact with an anchor that does not describe it.
    pub fn seal_checkpoint(
        &self,
        checkpoint: &[u8],
        declaration: &StateDeclaration,
        required: &[CodeFamily],
        targets: &SealTargets<'_>,
    ) -> Result<NeuralAnchor, SealCheckpointError> {
        let state = self
            .bind_checkpoint(checkpoint, declaration, required)
            .map_err(SealCheckpointError::Bind)?;
        let sealed = state.seal().map_err(SealCheckpointError::Seal)?;
        let anchor = sealed.anchor();
        let anchor_text = retained_anchor(targets.sealed, anchor);

        let mut files: Vec<(&Path, &[u8])> = Vec::with_capacity(3);
        if let Some(path) = targets.checkpoint {
            files.push((path, checkpoint));
        }
        files.push((targets.sealed, sealed.bytes()));
        // Last, so that its presence marks a complete set.
        files.push((targets.anchor, anchor_text.as_bytes()));

        let mut installed: Vec<&Path> = Vec::with_capacity(files.len());
        for (path, bytes) in files {
            if let Err(error) = write_atomically(path, bytes) {
                for done in installed {
                    let _ = fs::remove_file(done);
                }
                return Err(SealCheckpointError::Write {
                    path: path.to_path_buf(),
                    error,
                });
            }
            installed.push(path);
        }
        Ok(anchor)
    }
}

/// The scratch name a file is written under before it is renamed into place.
///
/// Beside the destination, so the rename stays within one directory and therefore
/// one filesystem, which is what makes it atomic.
fn scratch_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".sealing");
    path.with_file_name(name)
}

/// Install `bytes` at `path` whole or not at all.
fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let scratch = scratch_path(path);
    let written = (|| {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&scratch)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&scratch, path)
    })();
    if let Err(error) = written {
        // A scratch file is never authoritative; leaving it would only be litter
        // that the next save truncates anyway.
        let _ = fs::remove_file(&scratch);
        return Err(error);
    }
    sync_parent(path)
}

/// Make a rename durable: on Unix the directory entry is only on disk once the
/// directory itself is synchronized. Other platforms offer file-sync only.
fn sync_parent(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// An anchor read back from its retained text, with the artifact it names.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetainedAnchor {
    /// The sealed artifact the anchor says it describes, as it was written.
    pub sealed: String,
    pub anchor: NeuralAnchor,
}

/// Why retained anchor text could not be read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetainedAnchorError {
    /// A line that is neither a comment nor `key = value`.
    Malformed { line: usize },
    /// A key this format does not have.
    UnknownField { field: String },
    /// A key given twice, so neither value is the record.
    DuplicateField { field: String },
    /// A field the anchor cannot be rebuilt without.
    MissingField { field: &'static str },
    /// A value that does not parse as its field's type.
    InvalidValue { field: &'static str },
}

impl RetainedAnchorError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Malformed { .. } => "PTR_ANCHOR_TEXT_MALFORMED",
            Self::UnknownField { .. } => "PTR_ANCHOR_TEXT_UNKNOWN_FIELD",
            Self::DuplicateField { .. } => "PTR_ANCHOR_TEXT_DUPLICATE_FIELD",
            Self::MissingField { .. } => "PTR_ANCHOR_TEXT_MISSING_FIELD",
            Self::InvalidValue { .. } => "PTR_ANCHOR_TEXT_INVALID_VALUE",
        }
    }
}

impl std::fmt::Display for RetainedAnchorError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for RetainedAnchorError {}

const ANCHOR_FIELDS: [&str; 5] = [
    "sealed",
    "codebook",
    "journal_index",
    "journal_digest",
    "digest",
];

/// The retained form of an anchor: what it describes, and the fields that identify
/// it.
///
/// It is a small TOML document so that tools outside Rust read it with an ordinary
/// parser — the training pipeline's `require_seal` does — and every value is
/// written in the one spelling [`parse_retained_anchor`] accepts back.
pub fn retained_anchor(sealed: &Path, anchor: NeuralAnchor) -> String {
    let mut text = String::new();
    text.push_str("# Retained outside the artifact it describes. A digest read back\n");
    text.push_str("# out of that artifact would prove nothing.\n");
    let _ = writeln!(text, "sealed = {:?}", sealed.display().to_string());
    let _ = writeln!(text, "codebook = {}", anchor.codebook.0);
    let _ = writeln!(text, "journal_index = {}", anchor.journal.index.0);
    let _ = writeln!(text, "journal_digest = {:?}", hex(&anchor.journal.digest));
    let _ = writeln!(text, "digest = {:?}", hex(&anchor.digest));
    text
}

/// Read an anchor back from the text [`retained_anchor`] wrote.
///
/// Strict on purpose: an anchor is the trusted half of a seal, so a field given
/// twice, a field this format does not have and a digest of the wrong length are
/// each refused rather than resolved by a guess. It reads exactly what the writer
/// writes and is not a general TOML parser.
pub fn parse_retained_anchor(text: &str) -> Result<RetainedAnchor, RetainedAnchorError> {
    let mut values: [Option<&str>; 5] = [None; 5];
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or(RetainedAnchorError::Malformed { line: number + 1 })?;
        let key = key.trim();
        let slot = ANCHOR_FIELDS
            .iter()
            .position(|field| *field == key)
            .ok_or_else(|| RetainedAnchorError::UnknownField {
                field: key.to_owned(),
            })?;
        if values[slot].replace(value.trim()).is_some() {
            return Err(RetainedAnchorError::DuplicateField {
                field: key.to_owned(),
            });
        }
    }
    let field = |slot: usize| {
        values[slot].ok_or(RetainedAnchorError::MissingField {
            field: ANCHOR_FIELDS[slot],
        })
    };
    let invalid = |slot: usize| RetainedAnchorError::InvalidValue {
        field: ANCHOR_FIELDS[slot],
    };

    let sealed = unquote(field(0)?).ok_or_else(|| invalid(0))?;
    let codebook = field(1)?.parse::<u32>().map_err(|_| invalid(1))?;
    let journal_index = field(2)?.parse::<u64>().map_err(|_| invalid(2))?;
    let journal_digest = unquote(field(3)?)
        .and_then(|text| unhex(&text))
        .ok_or_else(|| invalid(3))?;
    let digest = unquote(field(4)?)
        .and_then(|text| unhex(&text))
        .ok_or_else(|| invalid(4))?;
    Ok(RetainedAnchor {
        sealed,
        anchor: NeuralAnchor {
            journal: LogAnchor {
                index: CommitIndex(journal_index),
                digest: journal_digest,
            },
            codebook: CodebookVersion(codebook),
            digest,
        },
    })
}

/// A basic TOML string as `{:?}` writes it for the paths and digests here. Escapes
/// other than `\\` and `\"` are refused rather than half-understood.
fn unquote(value: &str) -> Option<String> {
    let inner = value.strip_prefix('"')?.strip_suffix('"')?;
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next()? {
                escaped @ ('\\' | '"') => out.push(escaped),
                _ => return None,
            },
            '"' => return None,
            other => out.push(other),
        }
    }
    Some(out)
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// Exactly 64 lowercase hex digits, the only spelling [`hex`] produces.
fn unhex(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(text.as_bytes().chunks(2)) {
        *slot = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor() -> NeuralAnchor {
        NeuralAnchor {
            journal: LogAnchor {
                index: CommitIndex(7),
                digest: [0xAB; 32],
            },
            codebook: CodebookVersion(1),
            digest: [0x01; 32],
        }
    }

    #[test]
    fn retained_anchor_text_reads_back_as_the_anchor_it_was_written_from() {
        let text = retained_anchor(Path::new("out/model \"x\".sealed"), anchor());
        let read = parse_retained_anchor(&text).expect("the writer's own output parses");
        assert_eq!(read.anchor, anchor());
        assert_eq!(read.sealed, "out/model \"x\".sealed");
    }

    /// The control for the test above: a reader that accepted anything would pass
    /// it too, so each kind of damage has to be refused, by its own code.
    #[test]
    fn retained_anchor_text_that_is_damaged_is_refused_by_what_is_wrong_with_it() {
        let good = retained_anchor(Path::new("model.sealed"), anchor());
        let cases: [(String, &str); 6] = [
            (
                good.replace("codebook = 1", "codebook = one"),
                "PTR_ANCHOR_TEXT_INVALID_VALUE",
            ),
            (
                good.replace(&"01".repeat(32), &"01".repeat(31)),
                "PTR_ANCHOR_TEXT_INVALID_VALUE",
            ),
            (
                good.replace(&"01".repeat(32), &"0A".repeat(32)),
                "PTR_ANCHOR_TEXT_INVALID_VALUE",
            ),
            (
                format!("{good}codebook = 1\n"),
                "PTR_ANCHOR_TEXT_DUPLICATE_FIELD",
            ),
            (
                format!("{good}extra = 1\n"),
                "PTR_ANCHOR_TEXT_UNKNOWN_FIELD",
            ),
            (
                good.lines()
                    .filter(|line| !line.starts_with("journal_index"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                "PTR_ANCHOR_TEXT_MISSING_FIELD",
            ),
        ];
        for (text, code) in cases {
            let error = parse_retained_anchor(&text).expect_err(code);
            assert_eq!(error.code(), code, "{text}");
        }
        assert_eq!(
            parse_retained_anchor("no equals sign here")
                .expect_err("malformed")
                .code(),
            "PTR_ANCHOR_TEXT_MALFORMED"
        );
    }
}
