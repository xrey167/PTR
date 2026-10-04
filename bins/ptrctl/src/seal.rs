//! Bind a checkpoint to committed history and seal it, as one step with saving it.
//!
//! A checkpoint carries the assignment it was produced under. It does not carry
//! what it was produced *from*: the committed position, the semantic values that
//! were read, the generations that were live. Those are facts about a journal, so
//! only a process holding that journal can supply them — which is why the artifact
//! and its binding have until now been produced by different programs, and why a
//! checkpoint could reach a loader having never been bound at all.
//!
//! This command is the step that joins them for a checkpoint another program has
//! already saved. What the step *is* — bind, seal, write the sealed artifact and its
//! separately retained anchor, refusing before writing and leaving nothing behind
//! when a write fails — lives in `ptr_runtime::neural` as
//! `PtrRuntime::seal_checkpoint`, because a trainer that seals as it saves
//! (`ptr-a0-bridge`'s `save_bound_checkpoint`) has to do exactly the same thing.
//! What is left here is reading the declaration file and opening the journal.
use ptr_config::PtrConfig;
use ptr_runtime::neural::{NeuralAnchor, SealCheckpointError, SealTargets, StateDeclaration};
use ptr_runtime::PtrRuntime;
use ptr_types::{CodeFamily, CodebookVersion, ProvenanceRef};
use serde::Deserialize;
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

    // Binding, sealing and writing both files is the runtime's one sealed-save step,
    // which a trainer that seals as it saves calls too, so this command and that
    // trainer cannot come to disagree about what a seal is. It refuses before it
    // writes, installs each file whole, and removes the sealed artifact again if
    // its anchor cannot be retained. The checkpoint itself is not rewritten: this
    // command only checks bytes another program saved.
    runtime
        .seal_checkpoint(
            &bytes,
            &declaration,
            &required,
            &SealTargets {
                checkpoint: None,
                sealed: &request.out,
                anchor: &request.anchor,
            },
        )
        .map_err(|error| match error {
            SealCheckpointError::Bind(error) => {
                format!("refused to bind {}: {error:?}", show(&request.checkpoint))
            }
            SealCheckpointError::Seal(error) => {
                format!("cannot seal {}: {error:?}", show(&request.checkpoint))
            }
            SealCheckpointError::Write { path, error } => {
                format!("cannot write {}: {error}", show(&path))
            }
        })
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|error| format!("cannot read {}: {error}", show(path)))
}

fn show(path: &Path) -> String {
    path.display().to_string()
}
