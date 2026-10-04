//! The edge between the PTR runtime and Burn A0.
//!
//! Issue #15's Gate 3 asks to "connect actual semantic payloads to the model"
//! (issue #23's G1). Before this crate the channel existed at both ends and nothing
//! joined them: `SemanticSnapshot::slot_vector` had no caller outside its tests,
//! nothing outside `ptr-burn-a0` built a [`SlotValues`], and no PTR crate depended
//! on A0. This crate is that caller, and the only thing it adds is the joining:
//!
//! - **Payloads** come from one [`SemanticSnapshot`] through
//!   [`SemanticSnapshot::slot_vector`], and reach the model through A0's own
//!   [`SlotValues::new`]. There is no second encoding anywhere; the definition is
//!   the one in `ptr-types`, and the width is the model's.
//! - **Admission** comes from the runtime's committed lifecycle state:
//!   [`PtrRuntime::generation_validity`] for each slot's target and generation,
//!   narrowed into a [`ValidityMask`] that A0 applies by construction.
//! - **Binding** happens as the checkpoint is saved. [`save_bound_checkpoint`]
//!   writes the checkpoint, its sealed binding and the retained anchor through
//!   [`PtrRuntime::seal_checkpoint`], the same step `ptrctl seal` runs, so a
//!   checkpoint this path writes is never on disk unbound (issue #23's C9).
//!
//! It lives in A0's workspace rather than the root one so that it shares A0's
//! lockfile and vendor patches: burn resolves for it exactly as for A0.
//!
//! What it does **not** establish: anything about model quality. A slot vector
//! preserves identity, not similarity (see `ptr_types::SlotEncoding`); whether a
//! model learns anything useful from it is a separate, open question. Role,
//! epistemic state, provenance bucket and confidence are still the caller's to
//! supply — the runtime has no committed source for them — and only the payload and
//! the admission are taken from committed state here.
use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{
    admission_bias, save, CheckpointIoError, CodeGrid, CodeGridError, PtrA0, PtrA0Output,
    PtrSlotMetadata, SlotValueError, SlotValues, EMBEDDED_FAMILIES,
};
use ptr_ledger::integrity::LogAnchor;
use ptr_runtime::neural::{NeuralAnchor, SealCheckpointError, SealTargets, StateDeclaration};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_semdb::{SemanticSnapshot, SlotVectorError};
use ptr_types::{
    EpistemicState, Generation, Probability, ProvenanceRef, Revision, SemanticRole, SlotEncoding,
    ValidityMask,
};
use std::collections::BTreeSet;
use std::path::Path;

/// The lifecycle target and generation a slot was produced under.
///
/// Admission is decided from this and nothing else: the slot is admitted only when
/// the runtime reports exactly this generation of this target as live now.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Lifecycle<'a> {
    pub target: &'a str,
    pub generation: Generation,
}

/// One slot as the caller describes it.
///
/// `key` and `lifecycle` name committed facts and are resolved against the runtime;
/// the other four are the caller's, exactly as they were for every A0 call before
/// this crate, because nothing committed records them.
#[derive(Clone, Copy, Debug)]
pub struct SlotSpec<'a> {
    /// The semantic key whose committed value is this slot's payload.
    pub key: &'a str,
    pub lifecycle: Lifecycle<'a>,
    pub role: SemanticRole,
    pub epistemic: EpistemicState,
    /// Research-local provenance bucket; must index the model's provenance table.
    pub provenance_bucket: usize,
    pub confidence: Probability,
}

/// Why a runtime-fed forward pass or a bound save did not happen.
///
/// Every variant is a refusal; none pads, substitutes or skips a slot, because a
/// slot the model reads that nobody committed is a payload it believes it has seen.
#[derive(Debug)]
pub enum BridgeError {
    /// No rows, or rows with no slots.
    Empty,
    /// The raw path and the typed path describe different batch sizes.
    BatchMismatch { tokens: usize, slots: usize },
    /// A slot's committed value could not become a slot vector — most often because
    /// nothing is committed at its key.
    SlotVector {
        row: usize,
        slot: usize,
        error: SlotVectorError,
    },
    /// The vectors did not form one batch.
    SlotValues(SlotValueError),
    /// A role or epistemic state has no code in the model's codebook, or the rows
    /// are ragged.
    Codes(CodeGridError),
    /// A provenance bucket past the end of the model's table.
    ProvenanceBucket {
        row: usize,
        slot: usize,
        bucket: usize,
        rows: usize,
    },
    /// The runtime could not vouch for its own position (it is fenced).
    Position(RuntimeError),
    /// The runtime committed something between the forward passes a save describes
    /// and the save itself, so binding now would pin facts the model never read.
    PositionMoved { read: LogAnchor, now: LogAnchor },
    /// Two read sets from different runtime positions cannot describe one state.
    MixedPositions { left: LogAnchor, right: LogAnchor },
    /// The model could not be serialized.
    Checkpoint(CheckpointIoError),
    /// Binding, sealing or writing the save failed; see [`SealCheckpointError`].
    Seal(SealCheckpointError),
}

impl BridgeError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Empty => "PTR_A0_BRIDGE_EMPTY",
            Self::BatchMismatch { .. } => "PTR_A0_BRIDGE_BATCH_MISMATCH",
            Self::SlotVector { error, .. } => error.code(),
            Self::SlotValues(_) => "PTR_A0_BRIDGE_SLOT_VALUES",
            Self::Codes(_) => "PTR_A0_BRIDGE_CODES",
            Self::ProvenanceBucket { .. } => "PTR_A0_BRIDGE_PROVENANCE_BUCKET",
            Self::Position(_) => "PTR_A0_BRIDGE_POSITION",
            Self::PositionMoved { .. } => "PTR_A0_BRIDGE_POSITION_MOVED",
            Self::MixedPositions { .. } => "PTR_A0_BRIDGE_MIXED_POSITIONS",
            Self::Checkpoint(_) => "PTR_A0_BRIDGE_CHECKPOINT",
            Self::Seal(error) => error.code(),
        }
    }
}

impl std::fmt::Display for BridgeError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for BridgeError {}

/// Build one batch of slot values from committed payloads.
///
/// `keys` is the batch, one row of semantic keys per example. Every vector comes
/// from [`SemanticSnapshot::slot_vector`] — a key the snapshot does not hold is
/// refused, never encoded as zeros — and the batch is assembled by A0's own
/// [`SlotValues::new`], so this function adds no definition of its own.
pub fn slot_values_from(
    snapshot: &SemanticSnapshot,
    keys: &[&[&str]],
    encoding: SlotEncoding,
    width: usize,
    device: &Device,
) -> Result<SlotValues, BridgeError> {
    let mut rows = Vec::with_capacity(keys.len());
    for (row, row_keys) in keys.iter().enumerate() {
        let mut vectors = Vec::with_capacity(row_keys.len());
        for (slot, key) in row_keys.iter().enumerate() {
            let vector = snapshot
                .slot_vector(key, encoding, width)
                .map_err(|error| BridgeError::SlotVector { row, slot, error })?;
            vectors.push(vector);
        }
        rows.push(vectors);
    }
    let rows: Vec<&[_]> = rows.iter().map(Vec::as_slice).collect();
    SlotValues::new(&rows, device).map_err(BridgeError::SlotValues)
}

/// Decide each slot's admission from the runtime's committed lifecycle state.
///
/// A slot is admitted only when [`PtrRuntime::generation_validity`] reports its
/// generation `Live`. Superseded and revoked generations are excluded by the
/// kernel's rule ([`ValidityMask::admits_validity`]); a generation the authority
/// does not know at all is excluded too, because "cannot say" must not read as
/// "admitted". The mask starts admitting everything and is only ever narrowed.
pub fn admission_from(runtime: &PtrRuntime, slots: &[Lifecycle<'_>]) -> ValidityMask {
    let mut mask = ValidityMask::admitting_all(slots.len());
    for (slot, lifecycle) in slots.iter().enumerate() {
        let admitted = runtime
            .generation_validity(lifecycle.target, lifecycle.generation)
            .is_some_and(ValidityMask::admits_validity);
        if !admitted {
            mask.exclude(slot)
                .expect("the slot index comes from the same slice the mask was sized by");
        }
    }
    mask
}

/// What a set of forward passes read from one runtime position.
///
/// It is what a binding has to name, recorded by the bridge as it fed the model
/// rather than claimed afterwards by a trainer: the semantic keys of every
/// **admitted** slot and the targets they were admitted under. An excluded slot
/// contributes nothing — it cannot reach anything the model produces, and its
/// target may well be revoked, which a binding could not name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadSet {
    position: LogAnchor,
    keys: BTreeSet<String>,
    targets: BTreeSet<String>,
}

impl ReadSet {
    /// An empty read set at the runtime's current position, for a model that read
    /// nothing from it.
    pub fn at(runtime: &PtrRuntime) -> Result<Self, BridgeError> {
        Ok(Self {
            position: runtime.journal_anchor().map_err(BridgeError::Position)?,
            keys: BTreeSet::new(),
            targets: BTreeSet::new(),
        })
    }

    /// The committed position every read was made at.
    pub fn position(&self) -> LogAnchor {
        self.position
    }

    /// Semantic keys of every admitted slot.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.keys.iter().map(String::as_str)
    }

    /// Lifecycle targets of every admitted slot.
    pub fn targets(&self) -> impl Iterator<Item = &str> {
        self.targets.iter().map(String::as_str)
    }

    /// Fold another pass's reads in. Refused across positions: a binding names one.
    pub fn merge(&mut self, other: &ReadSet) -> Result<(), BridgeError> {
        if self.position != other.position {
            return Err(BridgeError::MixedPositions {
                left: self.position,
                right: other.position,
            });
        }
        self.keys.extend(other.keys.iter().cloned());
        self.targets.extend(other.targets.iter().cloned());
        Ok(())
    }

    /// The declaration a binding is resolved from: these reads, at `model`'s
    /// codebook, with the given provenance.
    pub fn declaration(&self, model: &PtrA0, provenance: &[ProvenanceRef]) -> StateDeclaration {
        let mut declaration = StateDeclaration::at(model.codebook().version());
        for key in &self.keys {
            declaration = declaration.reading(key.clone());
        }
        for target in &self.targets {
            declaration = declaration.under(target.clone());
        }
        for entry in provenance {
            declaration = declaration.produced_by(entry.clone());
        }
        declaration
    }
}

/// A forward pass fed from the runtime, with what it read.
pub struct RuntimeForward {
    pub output: PtrA0Output,
    /// The semantic revision every payload was read at.
    pub revision: Revision,
    /// The admission each row was governed by, as the runtime decided it.
    pub masks: Vec<ValidityMask>,
    /// What this pass read, for binding a checkpoint trained on it.
    pub read: ReadSet,
}

/// Run A0 on payloads and admission taken from the runtime's committed state.
///
/// One snapshot supplies every payload, and the lifecycle answers come from the
/// same `&PtrRuntime`, so nothing can commit between them: the payloads and the
/// admission describe one position, which is recorded in [`RuntimeForward::read`].
///
/// `token_ids` is the raw path, unchanged by this crate. `slots` has one row per
/// example and must be rectangular.
///
/// # Panics
///
/// As [`PtrA0::forward`] does if a token id lies outside the model's vocabulary:
/// the raw path is passed through, not checked here. Everything the bridge itself
/// supplies is checked first and refused as a [`BridgeError`].
pub fn forward_from_runtime(
    runtime: &PtrRuntime,
    model: &PtrA0,
    token_ids: Tensor<2, Int>,
    slots: &[&[SlotSpec<'_>]],
    device: &Device,
) -> Result<RuntimeForward, BridgeError> {
    if slots.is_empty() || slots[0].is_empty() {
        return Err(BridgeError::Empty);
    }
    let [tokens, _] = token_ids.dims();
    if tokens != slots.len() {
        return Err(BridgeError::BatchMismatch {
            tokens,
            slots: slots.len(),
        });
    }
    let position = runtime.journal_anchor().map_err(BridgeError::Position)?;
    let snapshot = runtime.snapshot();

    let keys: Vec<Vec<&str>> = slots
        .iter()
        .map(|row| row.iter().map(|slot| slot.key).collect())
        .collect();
    let keys: Vec<&[&str]> = keys.iter().map(Vec::as_slice).collect();
    // Ragged rows are refused here, by A0's own constructor, before anything below
    // indexes them as a rectangle.
    let slot_values =
        slot_values_from(&snapshot, &keys, model.encoding(), model.d_model(), device)?;

    let roles: Vec<Vec<SemanticRole>> = slots
        .iter()
        .map(|row| row.iter().map(|slot| slot.role).collect())
        .collect();
    let roles: Vec<&[SemanticRole]> = roles.iter().map(Vec::as_slice).collect();
    let states: Vec<Vec<EpistemicState>> = slots
        .iter()
        .map(|row| row.iter().map(|slot| slot.epistemic).collect())
        .collect();
    let states: Vec<&[EpistemicState]> = states.iter().map(Vec::as_slice).collect();
    let book = model.codebook();
    let slot_types = CodeGrid::new(&book, &roles, device).map_err(BridgeError::Codes)?;
    let epistemic = CodeGrid::new(&book, &states, device).map_err(BridgeError::Codes)?;

    let width = slots[0].len();
    let provenance_rows = model.provenance_rows();
    let mut provenance = Vec::with_capacity(slots.len() * width);
    let mut confidence = Vec::with_capacity(slots.len() * width);
    for (row, row_slots) in slots.iter().enumerate() {
        for (slot, spec) in row_slots.iter().enumerate() {
            if spec.provenance_bucket >= provenance_rows {
                return Err(BridgeError::ProvenanceBucket {
                    row,
                    slot,
                    bucket: spec.provenance_bucket,
                    rows: provenance_rows,
                });
            }
            provenance.push(spec.provenance_bucket as i64);
            confidence.push(spec.confidence.get());
        }
    }

    let mut read = ReadSet {
        position,
        keys: BTreeSet::new(),
        targets: BTreeSet::new(),
    };
    let mut masks = Vec::with_capacity(slots.len());
    for row in slots {
        let lifecycle: Vec<Lifecycle<'_>> = row.iter().map(|slot| slot.lifecycle).collect();
        let mask = admission_from(runtime, &lifecycle);
        for slot in mask.admitted_slots() {
            read.keys.insert(row[slot].key.to_owned());
            read.targets.insert(row[slot].lifecycle.target.to_owned());
        }
        masks.push(mask);
    }

    let metadata = PtrSlotMetadata {
        epistemic,
        provenance_ids: Tensor::<1, Int>::from_data(provenance.as_slice(), device)
            .reshape([slots.len(), width]),
        confidence: Tensor::<1>::from_data(confidence.as_slice(), device)
            .reshape([slots.len(), width]),
        admission: admission_bias(&masks, device),
    };
    let output = model.forward(token_ids, &slot_types, &slot_values, metadata);
    Ok(RuntimeForward {
        output,
        revision: snapshot.revision,
        masks,
        read,
    })
}

/// Where a bound save writes its three files.
#[derive(Clone, Copy, Debug)]
pub struct CheckpointPaths<'a> {
    /// The checkpoint itself: A0's header and weights.
    pub checkpoint: &'a Path,
    /// The sealed binding of that checkpoint to committed state.
    pub sealed: &'a Path,
    /// The anchor, retained separately from the sealed artifact it describes.
    pub anchor: &'a Path,
}

/// Save `model` already bound to what it read: checkpoint, seal and anchor, as one
/// operation.
///
/// This is issue #23's C9 — a trainer that writes a checkpoint binds it. It
/// serializes the model with A0's [`save`] and hands the bytes to
/// [`PtrRuntime::seal_checkpoint`], the same step `ptrctl seal` runs, with A0's
/// [`EMBEDDED_FAMILIES`] as the families the artifact must agree about. Binding
/// and sealing happen before anything is written, so a refusal leaves no file at
/// all — in particular no unbound checkpoint.
///
/// The declaration is the bridge's record of what the forward passes read
/// ([`ReadSet`]), not a claim the trainer composes. The save is refused if the
/// runtime has committed anything since those reads: binding then would pin values
/// and generations the model never saw.
///
/// Not to be confused with A0's own `save_bound`, which binds a checkpoint to an
/// M002-v5 experiment contract rather than to committed history.
pub fn save_bound_checkpoint(
    runtime: &PtrRuntime,
    model: &PtrA0,
    read: &ReadSet,
    provenance: &[ProvenanceRef],
    paths: &CheckpointPaths<'_>,
) -> Result<NeuralAnchor, BridgeError> {
    let now = runtime.journal_anchor().map_err(BridgeError::Position)?;
    if now != read.position {
        return Err(BridgeError::PositionMoved {
            read: read.position,
            now,
        });
    }
    let bytes = save(model).map_err(BridgeError::Checkpoint)?;
    runtime
        .seal_checkpoint(
            &bytes,
            &read.declaration(model, provenance),
            &EMBEDDED_FAMILIES,
            &SealTargets {
                checkpoint: Some(paths.checkpoint),
                sealed: paths.sealed,
                anchor: paths.anchor,
            },
        )
        .map_err(BridgeError::Seal)
}
