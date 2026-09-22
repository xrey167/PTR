//! Bind a checkpoint to committed history and seal it, as one step with saving it.
//!
//! A checkpoint carries the assignment it was produced under. It does not carry
//! what it was produced *from*: the committed position, the semantic values that
//! were read, the generations that were live. Those are facts about a journal, so
//! only a process holding that journal can supply them — which is why the artifact
//! and its binding have until now been produced by different programs, and why a
//! checkpoint could reach a loader having never been bound at all.
//!
//! This is the step that joins them. It refuses before it writes: a declaration
//! naming a key the journal does not have, or an artifact whose assignment this
//! build cannot reproduce, leaves no output behind, because a half-written seal
//! beside a checkpoint is worse than no seal — it looks like the binding
//! succeeded.
//!
//! The anchor is written to a **separate** file on purpose. A digest read back out
//! of the artifact it describes proves nothing; the anchor is only evidence while
//! it is retained somewhere the artifact cannot rewrite.
use ptr_config::PtrConfig;
use ptr_runtime::neural::{NeuralAnchor, StateDeclaration};
use ptr_runtime::PtrRuntime;
use ptr_types::{CodeFamily, CodebookVersion, ProvenanceRef};
use serde::Deserialize;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// What the trainer declares about the checkpoint it just wrote.
///
/// Everything here is a claim the trainer makes; none of it is trusted. Each
/// semantic key is resolved against the journal's committed value and each target
/// against its live generation, so a declaration that names something the journal
/// does not have is a refusal rather than an empty entry.
#[derive(Deserialize)]
struct DeclarationFile {
    /// Codebook version the weights were trained under.
    codebook: u32,
    /// Semantic keys read while producing the state.
    #[serde(default)]
    reading: Vec<String>,
    /// Lifecycle targets whose generation constrained production.
    #[serde(default)]
    under: Vec<String>,
    /// Code families the reader requires the artifact to record.
    #[serde(default)]
    requires: Vec<String>,
    #[serde(default)]
    provenance: Vec<ProvenanceFile>,
}

#[derive(Deserialize)]
struct ProvenanceFile {
    source: String,
    note: Option<String>,
}

/// Where the sealing step writes, and what it reads.
pub struct SealRequest {
    pub journal: PathBuf,
    pub checkpoint: PathBuf,
    pub declaration: PathBuf,
    pub out: PathBuf,
    pub anchor: PathBuf,
}

/// Bind `checkpoint` against the journal and write the sealed artifact.
///
/// Returns the anchor a catalog has to retain. Nothing is written unless every
/// step succeeded.
pub fn seal(request: &SealRequest) -> Result<NeuralAnchor, String> {
    let declaration_text = read(&request.declaration)?;
    let declared: DeclarationFile = toml::from_str(&declaration_text).map_err(|error| {
        format!(
            "invalid declaration {}: {error}",
            show(&request.declaration)
        )
    })?;

    let mut declaration = StateDeclaration::at(CodebookVersion(declared.codebook));
    for key in &declared.reading {
        declaration = declaration.reading(key.clone());
    }
    for target in &declared.under {
        declaration = declaration.under(target.clone());
    }
    for entry in &declared.provenance {
        declaration = declaration.produced_by(ProvenanceRef {
            source: entry.source.as_str().into(),
            note: entry.note.clone(),
        });
    }

    let mut required = Vec::new();
    for name in &declared.requires {
        // An unknown family is refused rather than skipped: a reader that asks for
        // a family this build does not define is asking for a guarantee nobody can
        // give, and silently requiring nothing is how that becomes invisible.
        let family = CodeFamily::from_name(name)
            .ok_or_else(|| format!("unknown code family in declaration: {name}"))?;
        required.push(family);
    }

    let bytes = std::fs::read(&request.checkpoint)
        .map_err(|error| format!("cannot read {}: {error}", show(&request.checkpoint)))?;

    let runtime = PtrRuntime::open_durable(PtrConfig::default(), &request.journal)
        .map_err(|error| format!("cannot open journal {}: {error:?}", show(&request.journal)))?;

    let state = runtime
        .bind_checkpoint(&bytes, &declaration, &required)
        .map_err(|error| format!("refused to bind {}: {error:?}", show(&request.checkpoint)))?;

    let sealed = state
        .seal()
        .map_err(|error| format!("cannot seal {}: {error:?}", show(&request.checkpoint)))?;

    // The two files have to arrive together or not at all. An anchor describing a
    // sealed file that is not there points at nothing; a sealed file whose anchor
    // was never retained cannot be reopened and still *reads* as a successful
    // binding to anything that finds it, which is the worse of the two.
    //
    // They cannot be written atomically as a pair, so the second failing undoes
    // the first.
    std::fs::write(&request.out, sealed.bytes())
        .map_err(|error| format!("cannot write {}: {error}", show(&request.out)))?;
    let anchor = sealed.anchor();
    if let Err(error) = std::fs::write(&request.anchor, anchor_toml(&request.out, anchor)) {
        let _ = std::fs::remove_file(&request.out);
        return Err(format!("cannot write {}: {error}", show(&request.anchor)));
    }

    Ok(anchor)
}

/// The retained form of an anchor: what it describes, and the three fields that
/// identify it.
fn anchor_toml(sealed: &Path, anchor: NeuralAnchor) -> String {
    let mut text = String::new();
    text.push_str("# Retained outside the artifact it describes. A digest read back\n");
    text.push_str("# out of that artifact would prove nothing.\n");
    let _ = writeln!(text, "sealed = {:?}", show(sealed));
    let _ = writeln!(text, "codebook = {}", anchor.codebook.0);
    let _ = writeln!(text, "journal_index = {}", anchor.journal.index.0);
    let _ = writeln!(text, "journal_digest = {:?}", hex(&anchor.journal.digest));
    let _ = writeln!(text, "digest = {:?}", hex(&anchor.digest));
    text
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|error| format!("cannot read {}: {error}", show(path)))
}

fn show(path: &Path) -> String {
    path.display().to_string()
}
