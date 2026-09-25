use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{
        activation::{gelu, softmax},
        Int,
    },
};
mod checkpoint;
pub use checkpoint::{header, load, save, CheckpointIoError, EMBEDDED_FAMILIES, MODEL};

use core::marker::PhantomData;
use ptr_types::{
    CodeFamily, Codebook, CodebookError, CodebookVersion, CognitiveType, EncodingVersion,
    EpistemicState, ReasoningOperator, SemanticRole, SlotEncoding, SlotVector, ValidityMask,
};

/// Why a batch of codes could not be built.
///
/// Every variant is a refusal, and none of them substitutes a code. A defaulted
/// code is how a foreign or stale identity reaches an embedding table without
/// anyone noticing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodeGridError {
    /// The codebook refused a member: it has no code in this version.
    Codebook(CodebookError),
    /// Rows disagree on how many slots they carry, so they cannot index one
    /// tensor. Not padded, because padding invents slots.
    Ragged {
        row: usize,
        expected: usize,
        found: usize,
    },
    /// No rows, or rows with no slots.
    Empty,
}

impl From<CodebookError> for CodeGridError {
    /// Carry a codebook refusal through unchanged.
    fn from(error: CodebookError) -> Self {
        Self::Codebook(error)
    }
}

impl core::fmt::Display for CodeGridError {
    /// Render a stable refusal code.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Codebook(error) => error.fmt(f),
            Self::Ragged { .. } => f.write_str("PTR_A0_RAGGED_CODE_GRID"),
            Self::Empty => f.write_str("PTR_A0_EMPTY_CODE_GRID"),
        }
    }
}

impl std::error::Error for CodeGridError {}

/// A rectangular batch of codebook codes for one family, fixed by that family's
/// type.
///
/// Built only through [`Codebook`], so a code the assignment does not define
/// cannot reach an embedding table. A bare `Tensor<2, Int>` can hold anything: an
/// index past the end of the table, a code minted under another codebook version,
/// or a code from another family that happens to be in range. An embedding lookup
/// answers all three without complaint, and the model is then wrong about what it
/// is looking at rather than visibly broken.
///
/// The family is a type parameter rather than a field, so passing epistemic codes
/// where semantic roles belong does not compile. The version travels with the
/// codes for the same reason: a code alone identifies nothing.
#[derive(Clone, Debug)]
pub struct CodeGrid<T: CognitiveType> {
    codebook: CodebookVersion,
    ids: Tensor<2, Int>,
    rows: usize,
    columns: usize,
    family: PhantomData<T>,
}

impl<T: CognitiveType> CodeGrid<T> {
    /// Assign codes to one batch of members.
    ///
    /// `rows` is the batch; every row must name the same number of slots.
    ///
    /// ```
    /// use burn::prelude::*;
    /// use ptr_burn_a0::CodeGrid;
    /// use ptr_types::{Codebook, SemanticRole};
    ///
    /// let device = Device::flex();
    /// let roles: [&[SemanticRole]; 1] = [&[SemanticRole::Goal, SemanticRole::Action]];
    /// let grid = CodeGrid::new(&Codebook::V1, &roles, &device).unwrap();
    /// assert_eq!(grid.dims(), [1, 2]);
    /// assert_eq!(grid.codebook(), Codebook::V1.version());
    /// ```
    pub fn new(book: &Codebook, rows: &[&[T]], device: &Device) -> Result<Self, CodeGridError> {
        let columns = rows.first().map_or(0, |row| row.len());
        if rows.is_empty() || columns == 0 {
            return Err(CodeGridError::Empty);
        }
        let mut values: Vec<i32> = Vec::with_capacity(rows.len() * columns);
        for (index, row) in rows.iter().enumerate() {
            if row.len() != columns {
                return Err(CodeGridError::Ragged {
                    row: index,
                    expected: columns,
                    found: row.len(),
                });
            }
            for member in row.iter() {
                values.push(i32::from(book.code_of(*member)?.index()));
            }
        }
        Ok(Self {
            codebook: book.version(),
            ids: Tensor::<1, Int>::from_data(values.as_slice(), device)
                .reshape([rows.len(), columns]),
            rows: rows.len(),
            columns,
            family: PhantomData,
        })
    }

    /// The single family these codes belong to.
    pub fn family(&self) -> CodeFamily {
        T::FAMILY
    }

    /// The codebook version that assigned them.
    pub fn codebook(&self) -> CodebookVersion {
        self.codebook
    }

    /// `[rows, slots]`.
    pub fn dims(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    /// The codes as an index tensor.
    pub fn ids(&self) -> Tensor<2, Int> {
        self.ids.clone()
    }
}

/// Why a batch of slot values could not be built.
///
/// Every variant is a refusal. None of them pads, truncates or substitutes, because
/// a slot value nobody committed is a payload the model believes it has seen.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SlotValueError {
    /// No rows, or rows with no slots.
    Empty,
    /// Rows disagree on how many slots they carry, so they cannot index one tensor.
    Ragged {
        row: usize,
        expected: usize,
        found: usize,
    },
    /// Two vectors in one batch were produced by different encoding definitions.
    ///
    /// Refused rather than mixed: a batch is one tensor, and a network cannot be
    /// told that some of its rows mean something else.
    MixedEncodings {
        expected: EncodingVersion,
        found: EncodingVersion,
    },
    /// Two vectors in one batch have different widths.
    MixedWidths { expected: usize, found: usize },
}

impl core::fmt::Display for SlotValueError {
    /// Render a stable refusal code.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => f.write_str("PTR_A0_EMPTY_SLOT_VALUES"),
            Self::Ragged { .. } => f.write_str("PTR_A0_RAGGED_SLOT_VALUES"),
            Self::MixedEncodings { .. } => f.write_str("PTR_A0_MIXED_SLOT_ENCODINGS"),
            Self::MixedWidths { .. } => f.write_str("PTR_A0_MIXED_SLOT_WIDTHS"),
        }
    }
}

impl std::error::Error for SlotValueError {}

/// A rectangular batch of slot values: what each slot in each row *is*.
///
/// This argument used to be a bare `Tensor<3>`, and it was the only input to
/// [`PtrA0::forward`] without a type, a version or any provenance — while slot
/// *identity* next to it could only be a [`CodeGrid`] built through a [`Codebook`].
/// It was also, at every call site in the repository, `Tensor::<3>::zeros`: the
/// channel was wired into `slots = slot_values + typed_metadata` and never carried a
/// value.
///
/// So it gets identity's treatment. A `SlotValues` is built only from
/// [`SlotVector`]s, which `ptr-types` produces only through a named
/// [`SlotEncoding`], which `ptr-semdb` feeds only from a *committed* value. A tensor
/// full of numbers nobody can account for no longer satisfies this signature.
#[derive(Clone, Debug)]
pub struct SlotValues {
    encoding: EncodingVersion,
    values: Tensor<3>,
    rows: usize,
    columns: usize,
    width: usize,
}

impl SlotValues {
    /// Assemble one batch of slot values.
    ///
    /// `rows` is the batch; every row must carry the same number of slots, and every
    /// vector the same width and encoding.
    ///
    /// ```
    /// use burn::prelude::*;
    /// use ptr_burn_a0::SlotValues;
    /// use ptr_types::{SlotEncoding, TypeId};
    ///
    /// let device = Device::flex();
    /// let book = SlotEncoding::V1;
    /// let goal = book.encode(&TypeId::from("Text"), b"ship the gate", 4).unwrap();
    /// let claim = book.encode(&TypeId::from("Document"), b"evidence", 4).unwrap();
    /// let values = SlotValues::new(&[&[goal, claim]], &device).unwrap();
    /// assert_eq!(values.dims(), [1, 2, 4]);
    /// assert_eq!(values.encoding(), SlotEncoding::V1.version());
    /// ```
    pub fn new(rows: &[&[SlotVector]], device: &Device) -> Result<Self, SlotValueError> {
        let columns = rows.first().map_or(0, |row| row.len());
        if rows.is_empty() || columns == 0 {
            return Err(SlotValueError::Empty);
        }
        let first = &rows[0][0];
        let encoding = first.encoding();
        let width = first.width();
        let mut values: Vec<f32> = Vec::with_capacity(rows.len() * columns * width);
        for (index, row) in rows.iter().enumerate() {
            if row.len() != columns {
                return Err(SlotValueError::Ragged {
                    row: index,
                    expected: columns,
                    found: row.len(),
                });
            }
            for vector in row.iter() {
                if vector.encoding() != encoding {
                    return Err(SlotValueError::MixedEncodings {
                        expected: encoding,
                        found: vector.encoding(),
                    });
                }
                if vector.width() != width {
                    return Err(SlotValueError::MixedWidths {
                        expected: width,
                        found: vector.width(),
                    });
                }
                values.extend_from_slice(vector.values());
            }
        }
        Ok(Self {
            encoding,
            values: Tensor::<1>::from_data(values.as_slice(), device).reshape([
                rows.len(),
                columns,
                width,
            ]),
            rows: rows.len(),
            columns,
            width,
        })
    }

    /// The definition that produced every vector in this batch.
    pub fn encoding(&self) -> EncodingVersion {
        self.encoding
    }

    /// `[rows, slots, width]`.
    pub fn dims(&self) -> [usize; 3] {
        [self.rows, self.columns, self.width]
    }

    /// The values as a tensor.
    pub fn values(&self) -> Tensor<3> {
        self.values.clone()
    }
}

/// The name under which the provenance width is recorded in the kernel.
pub const PROVENANCE_EXCEPTION: &str = "provenance_bucket_count";

/// The recorded width for [`PROVENANCE_EXCEPTION`], read from the kernel rather
/// than written here.
///
/// A literal in this file is a number with no home: invisible to Python, to a
/// dataset builder and to every check in the repository, so nothing could compare
/// it to anything. Resolving it from [`ptr_types::EXCEPTIONS`] gives it one home,
/// and the `const` panics at compile time if the record is ever removed — a
/// missing exception is a build failure rather than a silent fallback to 64.
pub const PROVENANCE_BUCKET_COUNT: usize = match ptr_types::exception_width(PROVENANCE_EXCEPTION) {
    Some(width) => width as usize,
    None => panic!("the kernel records no width for provenance_bucket_count"),
};

#[derive(Clone, Debug)]
pub struct PtrA0Config {
    pub vocab_size: usize,
    /// Provenance bucketing is **not** a codebook family: it is a research-local
    /// hashing of sources with no kernel taxonomy behind it, so it has no members
    /// to assign codes to. Its width is a *recorded exception* rather than a
    /// number invented here — [`ptr_types::EXCEPTIONS`] holds it and
    /// `datasets/generated/codebook.json` carries it, so a dataset builder and a
    /// model read one value instead of each picking their own. See
    /// `docs/architecture/26-cognitive-codebook.md`.
    ///
    /// It remains settable, because the exception is a *recorded default* and not
    /// a constraint: a experiment may legitimately bucket differently. What stops
    /// two widths meeting silently is the embedding's own shape — a checkpoint
    /// written at one width is refused by a model built at another, asserted in
    /// `tests/checkpoint.rs`.
    pub provenance_bucket_count: usize,
    pub d_model: usize,
    pub latent_steps: usize,
    /// Whether typed metadata biases attention between slots and raw tokens: the
    /// M002 mechanism. On by default. Switched off, the pair bias is zero in both
    /// directions and `metadata_bias` takes no part in the forward pass, which is
    /// the only difference; every parameter is still built, from the same random
    /// draws, so two arms of one seed start from identical weights.
    ///
    /// Like `latent_steps`, this is an architecture choice that the checkpoint
    /// header does not record: a checkpoint is loaded under whatever the config
    /// passed to [`load`] says.
    pub typed_attention: bool,
    /// Whether the slot->raw attention query reads the typed slot state (payload
    /// plus role, epistemic state, provenance and confidence) or the payload hash
    /// alone. On by default. Switched off, metadata can still steer which raw
    /// tokens a slot reads, but only through the typed attention bias, which is
    /// what the study's sufficiency pair isolates. Not recorded in the
    /// checkpoint header, like `typed_attention`.
    pub typed_query: bool,
    /// Whether each latent refinement step applies `gelu`. On by default.
    /// Switched off, the step is `slots + latent_refine(slots)`: the same
    /// parameters and depth with no nonlinearity, which separates "the per-slot
    /// nonlinearity is used" from "the extra parameters are used". Not recorded
    /// in the checkpoint header.
    pub latent_nonlinearity: bool,
    /// Whether the operator router is excluded from training. Off by default.
    /// It is applied by [`PtrA0Config::init`] after every parameter has been
    /// drawn, so the router's initial weights are the same as in any other arm
    /// of the seed, and it never affects [`load`]. The forward pass is unchanged:
    /// the router still produces the logits, from fixed weights. A training-time
    /// choice, not recorded in the checkpoint.
    pub frozen_router: bool,
    codebook: Codebook,
    /// The slot-encoding definition this model's inputs are produced by.
    ///
    /// Not a table width, so nothing about the tensors records it — which is why the
    /// checkpoint header carries it explicitly and `forward` asserts it.
    encoding: SlotEncoding,
}

impl PtrA0Config {
    /// Size the typed tables from the current frozen codebook.
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self::at_codebook(vocab_size, d_model, Codebook::V1)
    }

    /// Size the typed tables from a named codebook.
    ///
    /// The three typed widths are no longer arguments. They were, and the numbers
    /// the callers chose disagreed with the kernel: eight slot types against nine
    /// semantic roles, eight epistemic states against six, four operators against
    /// eleven. Rows that correspond to nothing train on nothing, and a table
    /// shorter than its family silently folds two members onto one code.
    pub fn at_codebook(vocab_size: usize, d_model: usize, codebook: Codebook) -> Self {
        Self {
            vocab_size,
            provenance_bucket_count: PROVENANCE_BUCKET_COUNT,
            d_model,
            latent_steps: 0,
            typed_attention: true,
            typed_query: true,
            latent_nonlinearity: true,
            frozen_router: false,
            codebook,
            encoding: SlotEncoding::V1,
        }
    }

    /// The slot encoding this model's inputs are produced by.
    pub fn encoding(&self) -> SlotEncoding {
        self.encoding
    }

    /// The codebook this model's tables are sized by.
    pub fn codebook(&self) -> Codebook {
        self.codebook
    }

    /// Slot-type table size: one row per [`SemanticRole`] in this codebook.
    pub fn slot_type_count(&self) -> usize {
        usize::from(self.codebook.cardinality_of::<SemanticRole>())
    }

    /// Epistemic table size: one row per [`EpistemicState`] in this codebook.
    pub fn epistemic_count(&self) -> usize {
        usize::from(self.codebook.cardinality_of::<EpistemicState>())
    }

    /// Router width: one logit per [`ReasoningOperator`] in this codebook.
    pub fn operator_count(&self) -> usize {
        usize::from(self.codebook.cardinality_of::<ReasoningOperator>())
    }

    pub fn with_provenance_buckets(mut self, provenance_bucket_count: usize) -> Self {
        self.provenance_bucket_count = provenance_bucket_count;
        self
    }

    pub fn with_latent_steps(mut self, latent_steps: usize) -> Self {
        self.latent_steps = latent_steps;
        self
    }

    pub fn with_typed_attention(mut self, typed_attention: bool) -> Self {
        self.typed_attention = typed_attention;
        self
    }

    pub fn with_typed_query(mut self, typed_query: bool) -> Self {
        self.typed_query = typed_query;
        self
    }

    pub fn with_latent_nonlinearity(mut self, latent_nonlinearity: bool) -> Self {
        self.latent_nonlinearity = latent_nonlinearity;
        self
    }

    pub fn with_frozen_router(mut self, frozen_router: bool) -> Self {
        self.frozen_router = frozen_router;
        self
    }

    /// Build a model with every parameter drawn now, in declaration order, so the
    /// same seed gives the same weights whatever the model later reads first.
    pub fn init(&self, device: &Device) -> PtrA0 {
        let mut model = self.init_lazy(device);
        model.materialize();
        // Only after every draw: excluding a parameter from training reads it, and
        // reading a lazy parameter early would shift every draw after it.
        if self.frozen_router {
            model.router = model.router.no_grad();
        }
        model
    }

    /// Build a model whose parameters are drawn on first read, as Burn does by
    /// default. Only for a model whose parameters are about to be replaced, as
    /// [`load`] replaces them: drawing them first would cost a full random model
    /// and advance the global generator as a side effect of loading.
    pub(crate) fn init_lazy(&self, device: &Device) -> PtrA0 {
        let linear = || LinearConfig::new(self.d_model, self.d_model).init(device);
        PtrA0 {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            slot_type_embedding: EmbeddingConfig::new(self.slot_type_count(), self.d_model)
                .init(device),
            epistemic_embedding: EmbeddingConfig::new(self.epistemic_count(), self.d_model)
                .init(device),
            provenance_embedding: EmbeddingConfig::new(self.provenance_bucket_count, self.d_model)
                .init(device),
            confidence_projection: LinearConfig::new(1, self.d_model).init(device),
            metadata_bias: LinearConfig::new(self.d_model, 1).init(device),
            slot_query: linear(),
            raw_key: linear(),
            raw_value: linear(),
            slot_output: linear(),
            raw_query: linear(),
            slot_key: linear(),
            slot_value: linear(),
            raw_output: linear(),
            latent_refine: linear(),
            router: LinearConfig::new(self.d_model, self.operator_count()).init(device),
            d_model: self.d_model,
            operator_count: self.operator_count(),
            latent_steps: self.latent_steps,
            typed_attention: self.typed_attention,
            typed_query: self.typed_query,
            latent_nonlinearity: self.latent_nonlinearity,
            codebook_version: self.codebook.version().0,
            encoding_version: self.encoding.version().0,
        }
    }
}

#[derive(Module, Debug)]
pub struct PtrA0 {
    token_embedding: Embedding,
    slot_type_embedding: Embedding,
    epistemic_embedding: Embedding,
    provenance_embedding: Embedding,
    confidence_projection: Linear,
    metadata_bias: Linear,
    slot_query: Linear,
    raw_key: Linear,
    raw_value: Linear,
    slot_output: Linear,
    raw_query: Linear,
    slot_key: Linear,
    slot_value: Linear,
    raw_output: Linear,
    latent_refine: Linear,
    router: Linear,
    d_model: usize,
    operator_count: usize,
    latent_steps: usize,
    typed_attention: bool,
    typed_query: bool,
    latent_nonlinearity: bool,
    /// Codebook version the tables were sized by. Burn records constants as empty,
    /// so this does **not** survive a saved record: an artifact that has to carry
    /// the identity carries it in its own header.
    codebook_version: u32,
    /// Slot-encoding version the inputs are produced by, recorded for the same
    /// reason and with the same caveat.
    encoding_version: u32,
}

impl PtrA0 {
    /// Draw every parameter now, in declaration order.
    ///
    /// Burn initializes a parameter lazily, the first time it is read, from one
    /// global generator. Left lazy, the draws follow the order in which a forward
    /// pass first reads the parameters, so two models built from one seed start
    /// from different weights as soon as one of them skips a module (the typed
    /// attention switch skips `metadata_bias`), and every later parameter shifts.
    /// An ablation compares arms of one seed, so the draw order has to be fixed
    /// here rather than left to the forward pass.
    fn materialize(&self) {
        for embedding in [
            &self.token_embedding,
            &self.slot_type_embedding,
            &self.epistemic_embedding,
            &self.provenance_embedding,
        ] {
            let _ = embedding.weight.val();
        }
        for linear in [
            &self.confidence_projection,
            &self.metadata_bias,
            &self.slot_query,
            &self.raw_key,
            &self.raw_value,
            &self.slot_output,
            &self.raw_query,
            &self.slot_key,
            &self.slot_value,
            &self.raw_output,
            &self.latent_refine,
            &self.router,
        ] {
            let _ = linear.weight.val();
            if let Some(bias) = &linear.bias {
                let _ = bias.val();
            }
        }
    }

    /// The codebook this model's tables were sized by.
    ///
    /// The lookup cannot fail: a [`PtrA0`] only comes from [`PtrA0Config::init`],
    /// which was given a [`Codebook`], and a `Codebook` only exists for a version
    /// this build defines.
    pub fn codebook(&self) -> Codebook {
        Codebook::at(CodebookVersion(self.codebook_version))
            .expect("the version came from a Codebook this build accepted")
    }

    /// The slot encoding this model's inputs are produced by.
    ///
    /// The lookup cannot fail: a [`PtrA0`] only comes from [`PtrA0Config::init`],
    /// which was given a [`SlotEncoding`], and one only exists for a version this
    /// build defines.
    pub fn encoding(&self) -> SlotEncoding {
        SlotEncoding::at(EncodingVersion(self.encoding_version))
            .expect("the version came from a SlotEncoding this build accepted")
    }

    /// Router width, one logit per operator in that codebook.
    pub fn operator_count(&self) -> usize {
        self.operator_count
    }

    /// Rows in the slot-type table, read from the weights.
    ///
    /// Not the number the config asked for: what a checkpoint has to record is
    /// what the tensors actually are.
    pub fn slot_type_rows(&self) -> usize {
        self.slot_type_embedding.weight.dims()[0]
    }

    /// Rows in the epistemic table, read from the weights.
    pub fn epistemic_rows(&self) -> usize {
        self.epistemic_embedding.weight.dims()[0]
    }
}

pub struct PtrSlotMetadata {
    /// Epistemic state per slot, as codebook codes.
    pub epistemic: CodeGrid<EpistemicState>,
    /// Research-local provenance bucket per slot. Not a codebook family; see
    /// [`PtrA0Config::provenance_bucket_count`].
    pub provenance_ids: Tensor<2, Int>,
    pub confidence: Tensor<2>,
    /// Additive attention bias from [`ValidityMask`]: `0.0` where the slot is
    /// admitted, negative infinity where it is not.
    ///
    /// Lifecycle validity enters here and nowhere else. It used to be a learned
    /// embedding summed into the metadata, which made it a hint the rest of the
    /// network could outvote — and the fact being outvoted was "this generation
    /// was revoked". Build it with [`admission_bias`].
    pub admission: Tensor<2>,
}

pub struct PtrA0Output {
    pub raw: Tensor<3>,
    /// Per-slot states, including the rows of excluded slots: they are computed
    /// but nothing the model produces depends on them.
    pub slots: Tensor<3>,
    pub router_logits: Tensor<2>,
    /// The admission that governed this forward pass, returned so a consumer
    /// pooling `slots` cannot lose it.
    pub admission: Tensor<2>,
}

/// Build the attention bias for a batch of per-slot masks.
///
/// The mask itself is computed from committed lifecycle state by `ptr-types`;
/// this only moves it onto the device. Every row must govern the same number of
/// slots, because they index one tensor.
pub fn admission_bias(masks: &[ValidityMask], device: &Device) -> Tensor<2> {
    let slots = masks.first().map(ValidityMask::len).unwrap_or(0);
    assert!(
        masks.iter().all(|mask| mask.len() == slots),
        "every mask in a batch governs the same slots"
    );
    let values: Vec<f32> = masks
        .iter()
        .flat_map(ValidityMask::attention_bias)
        .collect();
    Tensor::<1>::from_data(values.as_slice(), device).reshape([masks.len(), slots])
}

impl PtrA0 {
    /// Run the typed and raw paths.
    ///
    /// Slot identity arrives as [`CodeGrid<SemanticRole>`] and epistemic state as
    /// [`CodeGrid<EpistemicState>`], so the two cannot be swapped: the swap is a
    /// type error, not a plausible-looking output.
    ///
    /// ```
    /// use burn::{prelude::*, tensor::Int};
    /// use ptr_burn_a0::{admission_bias, CodeGrid, PtrA0Config, PtrSlotMetadata, SlotValues};
    /// use ptr_types::{
    ///     Codebook, EpistemicState, SemanticRole, SlotEncoding, TypeId, Validity, ValidityMask,
    /// };
    ///
    /// let device = Device::flex();
    /// let book = Codebook::V1;
    /// let model = PtrA0Config::new(8, 12).init(&device);
    /// let roles: [&[SemanticRole]; 1] = [&[SemanticRole::Goal, SemanticRole::Claim]];
    /// let states: [&[EpistemicState]; 1] = [&[EpistemicState::Observed, EpistemicState::Assumed]];
    /// let slot_types = CodeGrid::new(&book, &roles, &device).unwrap();
    /// let epistemic = CodeGrid::new(&book, &states, &device).unwrap();
    /// let encoding = SlotEncoding::V1;
    /// let goal = encoding.encode(&TypeId::from("Text"), b"ship the gate", 12).unwrap();
    /// let claim = encoding.encode(&TypeId::from("Document"), b"the evidence", 12).unwrap();
    /// let slot_values = SlotValues::new(&[&[goal, claim]], &device).unwrap();
    /// let output = model.forward(
    ///     Tensor::<2, Int>::from_data([[1, 2]], &device),
    ///     &slot_types,
    ///     &slot_values,
    ///     PtrSlotMetadata {
    ///         epistemic,
    ///         provenance_ids: Tensor::<2, Int>::zeros([1, 2], &device),
    ///         confidence: Tensor::<2>::ones([1, 2], &device),
    ///         admission: admission_bias(
    ///             &[ValidityMask::from_validities(&[Validity::Live; 2])],
    ///             &device,
    ///         ),
    ///     },
    /// );
    /// assert_eq!(output.router_logits.dims(), [1, model.operator_count()]);
    /// ```
    ///
    /// Swapping the two grids does not compile:
    ///
    /// ```compile_fail
    /// use burn::{prelude::*, tensor::Int};
    /// use ptr_burn_a0::{admission_bias, CodeGrid, PtrA0Config, PtrSlotMetadata, SlotValues};
    /// use ptr_types::{
    ///     Codebook, EpistemicState, SemanticRole, SlotEncoding, TypeId, Validity, ValidityMask,
    /// };
    ///
    /// let device = Device::flex();
    /// let book = Codebook::V1;
    /// let model = PtrA0Config::new(8, 12).init(&device);
    /// let roles: [&[SemanticRole]; 1] = [&[SemanticRole::Goal, SemanticRole::Claim]];
    /// let states: [&[EpistemicState]; 1] = [&[EpistemicState::Observed, EpistemicState::Assumed]];
    /// let slot_types = CodeGrid::new(&book, &roles, &device).unwrap();
    /// let epistemic = CodeGrid::new(&book, &states, &device).unwrap();
    /// let goal = SlotEncoding::V1.encode(&TypeId::from("Text"), b"x", 12).unwrap();
    /// let slot_values = SlotValues::new(&[&[goal.clone(), goal]], &device).unwrap();
    /// let _ = model.forward(
    ///     Tensor::<2, Int>::from_data([[1, 2]], &device),
    ///     &epistemic,
    ///     &slot_values,
    ///     PtrSlotMetadata {
    ///         epistemic: slot_types,
    ///         provenance_ids: Tensor::<2, Int>::zeros([1, 2], &device),
    ///         confidence: Tensor::<2>::ones([1, 2], &device),
    ///         admission: admission_bias(
    ///             &[ValidityMask::from_validities(&[Validity::Live; 2])],
    ///             &device,
    ///         ),
    ///     },
    /// );
    /// ```
    pub fn forward(
        &self,
        token_ids: Tensor<2, Int>,
        slot_types: &CodeGrid<SemanticRole>,
        slot_values: &SlotValues,
        metadata: PtrSlotMetadata,
    ) -> PtrA0Output {
        let [batch, sequence] = token_ids.dims();
        let [slot_batch, slot_count] = slot_types.dims();
        assert_eq!(
            batch, slot_batch,
            "raw and typed paths need the same batch size"
        );
        // Codes minted under another assignment index these tables just as well as
        // the right ones, and the result is a model confidently reading the wrong
        // taxonomy. With a single frozen version this cannot be reached from
        // outside; it is here so that adding a version fails loudly.
        let version = self.codebook().version();
        assert_eq!(
            slot_types.codebook(),
            version,
            "slot codes come from another codebook version"
        );
        assert_eq!(
            metadata.epistemic.codebook(),
            version,
            "epistemic codes come from another codebook version"
        );
        // Vectors from another definition index nothing — they are simply different
        // numbers — so no shape would reveal the mismatch and this assert is the only
        // thing that can. Same reasoning as the codebook asserts above, one step
        // earlier in the pipeline.
        assert_eq!(
            slot_values.encoding(),
            self.encoding().version(),
            "slot values come from another slot-encoding version"
        );
        assert_eq!(
            slot_values.dims(),
            [batch, slot_count, self.d_model],
            "slot values must cover every slot at the model's width"
        );
        assert_eq!(metadata.epistemic.dims(), [batch, slot_count]);
        assert_eq!(metadata.provenance_ids.dims(), [batch, slot_count]);
        assert_eq!(metadata.confidence.dims(), [batch, slot_count]);
        assert_eq!(metadata.admission.dims(), [batch, slot_count]);

        let raw = self.token_embedding.forward(token_ids);
        let slot_type = self.slot_type_embedding.forward(slot_types.ids());
        let epistemic = self.epistemic_embedding.forward(metadata.epistemic.ids());
        let provenance = self.provenance_embedding.forward(metadata.provenance_ids);
        let confidence = self
            .confidence_projection
            .forward(metadata.confidence.unsqueeze_dim::<3>(2));

        // Validity is deliberately absent from this sum. It is admission, not a
        // feature, and it is applied below where it cannot be weighed.
        let typed_metadata = slot_type + epistemic + provenance + confidence;
        let slots = slot_values.values() + typed_metadata.clone();

        // The typed query reads the whole slot state; the blind one only the
        // payload, so metadata can steer this read only through the typed bias.
        let slot_query = if self.typed_query {
            self.slot_query.forward(slots.clone())
        } else {
            self.slot_query.forward(slot_values.values())
        };
        let raw_key = self.raw_key.forward(raw.clone());
        // Switched off, the pair bias is zero in both directions below and
        // `metadata_bias` is never applied, so it receives no gradient.
        let cross_bias = if self.typed_attention {
            typed_cross_bias(self.metadata_bias.forward(typed_metadata), raw_key.clone())
        } else {
            Tensor::<3>::zeros([batch, slot_count, sequence], &raw_key.device())
        };
        let raw_value = self.raw_value.forward(raw.clone());
        let slot_scores = slot_query
            .matmul(raw_key.transpose())
            .div_scalar((self.d_model as f32).sqrt())
            + cross_bias.clone();
        let slot_weights = softmax(slot_scores, 2);
        let slot_context = slot_weights.matmul(raw_value);
        let mut slots = slots + self.slot_output.forward(slot_context);

        let raw_query = self.raw_query.forward(raw.clone());
        let slot_key = self.slot_key.forward(slots.clone());
        let slot_value = self.slot_value.forward(slots.clone());
        // The admission bias is added last and is negative infinity for an
        // excluded slot, so no score this network can produce reaches it: after
        // the softmax its weight is exactly zero, not merely small.
        //
        // A row that admits nothing would be a softmax over nothing but negative
        // infinity, which is NaN and would poison the whole batch. Such a row
        // gets a finite bias so the softmax stays defined, and its context is
        // dropped afterwards instead: with no admissible typed state there is
        // nothing to attend to, which is an answer rather than a crash.
        let admitted = metadata.admission.clone().equal_elem(0.0);
        let any_admitted = admitted.clone().float().sum_dim(1);
        let nothing_admitted = any_admitted.clone().equal_elem(0.0);
        let bias = metadata
            .admission
            .clone()
            .reshape([batch, 1, slot_count])
            .expand([batch, sequence, slot_count])
            .mask_fill(
                nothing_admitted
                    .clone()
                    .reshape([batch, 1, 1])
                    .expand([batch, sequence, slot_count]),
                0.0,
            );
        let raw_scores = raw_query
            .matmul(slot_key.transpose())
            .div_scalar((self.d_model as f32).sqrt())
            + cross_bias.transpose()
            + bias;
        let raw_weights = softmax(raw_scores, 2);
        let raw_context = raw_weights.matmul(slot_value).mask_fill(
            nothing_admitted
                .clone()
                .reshape([batch, 1, 1])
                .expand([batch, sequence, self.d_model]),
            0.0,
        );
        let raw = raw + self.raw_output.forward(raw_context);

        for _ in 0..self.latent_steps {
            let refined = self.latent_refine.forward(slots.clone());
            let delta = if self.latent_nonlinearity {
                gelu(refined)
            } else {
                refined
            };
            slots = slots + delta;
        }

        // Attention is not the only way a slot reaches the output: the router
        // averages over slots, so an excluded one would contribute through the
        // mean however it was attended to. Weight the sum by admission instead,
        // and divide by the admitted count rather than by every slot.
        let admitted = admitted.float().reshape([batch, slot_count, 1]);
        let admitted_count = admitted.clone().sum_dim(1).clamp_min(1.0);
        let router_logits = (self.router.forward(slots.clone())
            * admitted.expand([batch, slot_count, self.operator_count]))
        .sum_dim(1)
        .reshape([batch, self.operator_count])
            / admitted_count
                .reshape([batch, 1])
                .expand([batch, self.operator_count]);

        PtrA0Output {
            raw,
            slots,
            router_logits,
            admission: metadata.admission,
        }
    }
}

// A query-only constant cancels in softmax. Pair the learned slot scalar with
// each raw key's summary so this rank-one bias actually varies across keys.
fn typed_cross_bias(slot_bias: Tensor<3>, raw_keys: Tensor<3>) -> Tensor<3> {
    slot_bias.matmul(raw_keys.mean_dim(2).transpose())
}

#[cfg(test)]
mod typed_attention_switch_tests {
    //! The M002 switch removes exactly the typed pair bias. Each test compares a
    //! model with its own clone, so the two differ only where a test makes them.
    use super::*;
    use burn::nn::Initializer;
    use ptr_types::{TypeId, Validity};

    const WIDTH: usize = 8;

    fn inputs(
        device: &Device,
    ) -> (
        Tensor<2, Int>,
        CodeGrid<SemanticRole>,
        SlotValues,
        PtrSlotMetadata,
    ) {
        let roles: [&[SemanticRole]; 2] = [
            &[
                SemanticRole::Goal,
                SemanticRole::Evidence,
                SemanticRole::Constraint,
            ],
            &[
                SemanticRole::Claim,
                SemanticRole::Resource,
                SemanticRole::Action,
            ],
        ];
        let states: [&[EpistemicState]; 2] = [
            &[
                EpistemicState::Observed,
                EpistemicState::Hypothesis,
                EpistemicState::Verified,
            ],
            &[
                EpistemicState::Assumed,
                EpistemicState::Inferred,
                EpistemicState::Unknown,
            ],
        ];
        let vectors: Vec<Vec<SlotVector>> = (0..2)
            .map(|row| {
                (0..3)
                    .map(|slot| {
                        SlotEncoding::V1
                            .encode(
                                &TypeId::from("Fact"),
                                format!("{row}/{slot}").as_bytes(),
                                WIDTH,
                            )
                            .expect("a small payload")
                    })
                    .collect()
            })
            .collect();
        let rows: Vec<&[SlotVector]> = vectors.iter().map(Vec::as_slice).collect();
        (
            Tensor::<2, Int>::from_data([[1, 5, 9, 2], [7, 3, 3, 8]], device),
            CodeGrid::new(&Codebook::V1, &roles, device).expect("v1 roles"),
            SlotValues::new(&rows, device).expect("a rectangular batch"),
            PtrSlotMetadata {
                epistemic: CodeGrid::new(&Codebook::V1, &states, device).expect("v1 states"),
                provenance_ids: Tensor::<2, Int>::from_data([[0, 1, 2], [2, 1, 0]], device),
                confidence: Tensor::<2>::from_data([[0.9, 0.4, 0.7], [0.2, 1.0, 0.5]], device),
                admission: admission_bias(
                    &[
                        ValidityMask::from_validities(&[Validity::Live; 3]),
                        ValidityMask::from_validities(&[
                            Validity::Live,
                            Validity::Revoked,
                            Validity::Live,
                        ]),
                    ],
                    device,
                ),
            },
        )
    }

    fn run(model: &PtrA0, device: &Device) -> PtrA0Output {
        let (tokens, roles, values, metadata) = inputs(device);
        model.forward(tokens, &roles, &values, metadata)
    }

    fn largest_difference(left: Tensor<2>, right: Tensor<2>) -> f32 {
        (left - right).abs().max().into_scalar()
    }

    fn config() -> PtrA0Config {
        PtrA0Config::new(16, WIDTH)
            .with_provenance_buckets(4)
            .with_latent_steps(1)
    }

    #[test]
    fn switching_it_off_equals_a_zero_pair_bias_and_nothing_else() {
        let device = Device::flex();
        device.seed(7);
        let mut on = config().init(&device);
        // A metadata_bias that outputs zero makes the pair bias zero, which is what
        // the switch claims to do; everything else stays as initialized.
        on.metadata_bias = LinearConfig::new(WIDTH, 1)
            .with_initializer(Initializer::Zeros)
            .init(&device);
        let mut off = on.clone();
        off.typed_attention = false;

        let (a, b) = (run(&on, &device), run(&off, &device));
        assert!(largest_difference(a.router_logits, b.router_logits) < 1.0e-6);
        let slots: f32 = (a.slots - b.slots).abs().max().into_scalar();
        let raw: f32 = (a.raw - b.raw).abs().max().into_scalar();
        assert!(slots < 1.0e-6 && raw < 1.0e-6, "slots {slots}, raw {raw}");
    }

    #[test]
    fn switching_it_off_changes_what_a_trained_bias_does() {
        let device = Device::flex();
        // One model and its clone, not two inits from one seed: the generator is
        // shared by every test thread, so a second init could draw other weights
        // and make the two differ for a reason that is not the switch.
        let on = config().init(&device);
        let mut off = on.clone();
        off.typed_attention = false;

        let difference = largest_difference(
            run(&on, &device).router_logits,
            run(&off, &device).router_logits,
        );
        assert!(
            difference > 1.0e-5,
            "the switch must matter when the bias is not zero: {difference}"
        );
    }

    #[test]
    fn switched_off_the_bias_layer_gets_no_gradient_and_the_rest_still_learns() {
        let device = Device::flex().autodiff();
        for typed_attention in [true, false] {
            device.seed(7);
            let model = config().with_typed_attention(typed_attention).init(&device);
            let gradients = run(&model, &device).router_logits.sum().backward();
            let bias = model.metadata_bias.weight.grad(&gradients);
            assert_eq!(
                bias.is_some(),
                typed_attention,
                "metadata_bias gradient with typed_attention={typed_attention}"
            );
            let router = model
                .router
                .weight
                .grad(&gradients)
                .expect("the router always learns");
            let magnitude: f32 = router.abs().sum().into_scalar();
            assert!(magnitude > 0.0);
        }
    }

    #[test]
    fn it_is_on_by_default() {
        assert!(PtrA0Config::new(16, WIDTH).typed_attention);
    }
}

#[cfg(test)]
mod ablation_mechanism_tests {
    //! Tests T3 (switch completeness), T5 (the dead raw->slot branch) and the
    //! frozen-router half of T2 from the A0 ablation study design. Each builds
    //! one model per configuration and never compares two models drawn from the
    //! same seed, so tests running in parallel threads cannot disturb them.
    use super::*;
    use burn::{
        nn::{loss::CrossEntropyLossConfig, Initializer},
        optim::{AdamConfig, GradientsParams},
    };
    use ptr_types::{TypeId, Validity};

    const WIDTH: usize = 8;

    fn metadata(
        device: &Device,
        roles: [SemanticRole; 3],
    ) -> (CodeGrid<SemanticRole>, PtrSlotMetadata) {
        let rows: [&[SemanticRole]; 1] = [&roles];
        let states: [&[EpistemicState]; 1] = [&[
            EpistemicState::Observed,
            EpistemicState::Hypothesis,
            EpistemicState::Verified,
        ]];
        (
            CodeGrid::new(&Codebook::V1, &rows, device).expect("v1 roles"),
            PtrSlotMetadata {
                epistemic: CodeGrid::new(&Codebook::V1, &states, device).expect("v1 states"),
                provenance_ids: Tensor::<2, Int>::from_data([[0, 1, 2]], device),
                confidence: Tensor::<2>::from_data([[0.9, 0.4, 0.7]], device),
                admission: admission_bias(
                    &[ValidityMask::from_validities(&[Validity::Live; 3])],
                    device,
                ),
            },
        )
    }

    fn values(device: &Device) -> SlotValues {
        let vectors: Vec<SlotVector> = (0..3)
            .map(|slot| {
                SlotEncoding::V1
                    .encode(
                        &TypeId::from("Entity"),
                        format!("e{slot}").as_bytes(),
                        WIDTH,
                    )
                    .expect("a small payload")
            })
            .collect();
        SlotValues::new(&[vectors.as_slice()], device).expect("one row")
    }

    fn logits(
        model: &PtrA0,
        tokens: [i64; 4],
        roles: [SemanticRole; 3],
        device: &Device,
    ) -> Tensor<2> {
        let (roles, metadata) = metadata(device, roles);
        let tokens = Tensor::<2, Int>::from_data([tokens], device);
        model
            .forward(tokens, &roles, &values(device), metadata)
            .router_logits
    }

    fn largest(tensor: Tensor<2>) -> f32 {
        tensor.abs().max().into_scalar()
    }

    fn config() -> PtrA0Config {
        PtrA0Config::new(16, WIDTH).with_provenance_buckets(4)
    }

    /// Every distinct model configuration the study's arms use.
    fn arm_configs() -> Vec<(&'static str, PtrA0Config)> {
        let full = config().with_latent_steps(2);
        vec![
            ("full", full.clone()),
            (
                "no-typed-attention",
                full.clone().with_typed_attention(false),
            ),
            (
                "blind-query-k0",
                full.clone().with_typed_query(false).with_latent_steps(0),
            ),
            (
                "blind-query-k0-no-typed-attention",
                full.clone()
                    .with_typed_query(false)
                    .with_latent_steps(0)
                    .with_typed_attention(false),
            ),
            ("latent-0", full.clone().with_latent_steps(0)),
            (
                "latent-linear",
                full.clone().with_latent_nonlinearity(false),
            ),
            ("latent-1", full.clone().with_latent_steps(1)),
            ("latent-4", full.clone().with_latent_steps(4)),
            ("frozen-router", full.with_frozen_router(true)),
        ]
    }

    /// T3a: with no latent step, the refinement layer takes no part.
    #[test]
    fn with_no_latent_step_the_logits_ignore_latent_refine() {
        let device = Device::flex();
        let model = config().with_latent_steps(0).init(&device);
        let mut zeroed = model.clone();
        zeroed.latent_refine = LinearConfig::new(WIDTH, WIDTH)
            .with_initializer(Initializer::Zeros)
            .init(&device);
        let roles = [
            SemanticRole::Goal,
            SemanticRole::Claim,
            SemanticRole::Action,
        ];
        let difference = largest(
            logits(&model, [1, 2, 3, 4], roles, &device)
                - logits(&zeroed, [1, 2, 3, 4], roles, &device),
        );
        assert_eq!(difference, 0.0);
    }

    /// T3c: one latent step adds exactly gelu(latent_refine(slots)) to the slots,
    /// and exactly latent_refine(slots) with the nonlinearity switched off.
    #[test]
    fn one_latent_step_adds_exactly_its_delta_with_or_without_the_gelu() {
        let device = Device::flex();
        let roles = [
            SemanticRole::Goal,
            SemanticRole::Claim,
            SemanticRole::Action,
        ];
        let slots_of = |model: &PtrA0| {
            let (roles, metadata) = metadata(&device, roles);
            let tokens = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], &device);
            model
                .forward(tokens, &roles, &values(&device), metadata)
                .slots
        };
        let stepped = config().with_latent_steps(1).init(&device);
        let mut before = stepped.clone();
        before.latent_steps = 0;
        let start = slots_of(&before);
        let refined = stepped.latent_refine.forward(start.clone());
        for (nonlinearity, delta) in [(true, gelu(refined.clone())), (false, refined)] {
            let mut model = stepped.clone();
            model.latent_nonlinearity = nonlinearity;
            let difference: f32 = (slots_of(&model) - (start.clone() + delta))
                .abs()
                .max()
                .into_scalar();
            assert!(
                difference < 1.0e-6,
                "nonlinearity {nonlinearity}: {difference}"
            );
        }
        let mut linear = stepped.clone();
        linear.latent_nonlinearity = false;
        let switched: f32 = (slots_of(&stepped) - slots_of(&linear))
            .abs()
            .max()
            .into_scalar();
        assert!(
            switched > 1.0e-4,
            "the gelu must change the step: {switched}"
        );
    }

    /// T3b: with the typed query, the typed bias and the latent steps all off,
    /// a slot's role can only add to its own vote: the effect of changing one role
    /// is the same whatever the raw tokens are. This is why the blind-query arm
    /// without typed attention cannot gate raw context by role.
    #[test]
    fn with_every_metadata_read_off_a_role_only_shifts_the_logits() {
        let device = Device::flex();
        let model = config()
            .with_latent_steps(0)
            .with_typed_query(false)
            .with_typed_attention(false)
            .init(&device);
        let a = [
            SemanticRole::Goal,
            SemanticRole::Claim,
            SemanticRole::Action,
        ];
        let b = [
            SemanticRole::Evidence,
            SemanticRole::Claim,
            SemanticRole::Action,
        ];
        let effect =
            |tokens| logits(&model, tokens, a, &device) - logits(&model, tokens, b, &device);
        let first = effect([1, 2, 3, 4]);
        let second = effect([9, 15, 7, 11]);
        assert!(
            largest(first.clone()) > 1.0e-4,
            "the role must matter at all"
        );
        assert!(
            largest(first - second) < 1.0e-5,
            "the role's effect must not depend on raw input"
        );

        // The control: with the typed query back on, the same change does depend
        // on the raw input, so the invariance above is the switches' doing.
        let typed = config()
            .with_latent_steps(0)
            .with_typed_attention(false)
            .init(&device);
        let effect =
            |tokens| logits(&typed, tokens, a, &device) - logits(&typed, tokens, b, &device);
        assert!(largest(effect([1, 2, 3, 4]) - effect([9, 15, 7, 11])) > 1.0e-5);
    }

    /// T5: the raw->slot update never reaches the router, in any arm. Pinned so
    /// that a future change which connects it is seen rather than assumed.
    #[test]
    fn the_raw_to_slot_branch_gets_no_gradient_in_any_arm() {
        let device = Device::flex().autodiff();
        for (arm, arm_config) in arm_configs() {
            let model = arm_config.init(&device);
            let roles = [
                SemanticRole::Goal,
                SemanticRole::Claim,
                SemanticRole::Action,
            ];
            let gradients = logits(&model, [1, 2, 3, 4], roles, &device)
                .sum()
                .backward();
            for (name, layer) in [
                ("raw_query", &model.raw_query),
                ("slot_key", &model.slot_key),
                ("slot_value", &model.slot_value),
                ("raw_output", &model.raw_output),
            ] {
                assert!(
                    layer.weight.grad(&gradients).is_none(),
                    "{arm}: {name} got a gradient"
                );
            }
            assert!(
                model.slot_query.weight.grad(&gradients).is_some(),
                "{arm}: slot_query must learn"
            );
        }
    }

    /// T2, frozen router: one Adam step moves the upstream layers and leaves the
    /// router's weights bit for bit as they were; without the switch it moves them.
    #[test]
    fn a_frozen_router_does_not_move_while_the_rest_learns() {
        let device = Device::flex().autodiff();
        for frozen in [true, false] {
            let model = config()
                .with_latent_steps(2)
                .with_frozen_router(frozen)
                .init(&device);
            let router_before = model.router.weight.val().into_data();
            let query_before = model.slot_query.weight.val().into_data();
            let roles = [
                SemanticRole::Goal,
                SemanticRole::Claim,
                SemanticRole::Action,
            ];
            let loss = CrossEntropyLossConfig::new().init(&device).forward(
                logits(&model, [1, 2, 3, 4], roles, &device),
                Tensor::<1, Int>::from_data([3], &device),
            );
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            let mut optimizer = AdamConfig::new().init();
            let model = optimizer.step(0.01, model, gradients);
            assert_eq!(
                model.router.weight.val().into_data() == router_before,
                frozen,
                "router unchanged must mean frozen (frozen={frozen})"
            );
            assert_ne!(model.slot_query.weight.val().into_data(), query_before);
        }
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;

    #[test]
    fn row_constant_bias_cancels_but_pair_bias_changes_attention_and_has_gradient() {
        let device = Device::flex().autodiff();
        let scores = Tensor::<3>::from_data([[[0.0_f32, 0.0, 0.0]]], &device);
        let query_bias = Tensor::<3>::from_data([[[2.0_f32]]], &device).require_grad();
        let raw_keys = Tensor::<3>::from_data([[[0.0_f32], [1.0], [2.0]]], &device);
        let unchanged = softmax(scores.clone() + query_bias.clone().expand([1, 1, 3]), 2);
        let baseline = softmax(scores.clone(), 2);
        let cancellation_error: f32 = (unchanged - baseline.clone()).abs().max().into_scalar();
        assert!(cancellation_error < 1.0e-6);
        let weights = softmax(scores + typed_cross_bias(query_bias.clone(), raw_keys), 2);
        let difference: f32 = (weights.clone() - baseline).abs().max().into_scalar();
        assert!(difference > 0.1, "pair bias must affect attention");
        let selected = weights.slice([0..1, 0..1, 2..3]).sum();
        let gradients = selected.backward();
        let gradient = query_bias
            .grad(&gradients)
            .expect("bias gradient is connected");
        let magnitude: f32 = gradient.abs().sum().into_scalar();
        assert!(magnitude.is_finite() && magnitude > 1.0e-4);
    }
}

/// The codebook-version guard in `forward`, entered.
///
/// `26-cognitive-codebook.md` recorded this comparison as unreachable: one frozen
/// version means [`Codebook::at`] refuses every other, and [`CodeGrid`]'s version
/// is private, so no test outside this crate can build a foreign grid. A guard
/// nobody has ever entered is a guard whose behaviour is a claim — the repository's
/// own rule, that a test which has never failed is not evidence it can, applies
/// just as well to a branch never taken.
///
/// These tests enter it from inside the module, where the field is reachable. Two
/// details make them evidence rather than decoration:
///
/// * each names its guard's **full** message. Both guards say "another codebook
///   version", so a shorter `expected` would let a test aimed at the slot guard
///   pass when the epistemic one fired.
/// * `forward` also panics on a batch mismatch and on three dimension checks, so a
///   `should_panic` with no message at all passes on any of five unrelated faults.
#[cfg(test)]
mod codebook_guard_tests {
    use super::*;
    use ptr_types::Validity;

    const VOCAB: usize = 16;
    const D_MODEL: usize = 8;

    /// A version this build has no tables for. `Codebook::at` refuses it, which is
    /// exactly why the guard cannot be reached from outside.
    const FOREIGN: CodebookVersion = CodebookVersion(2);

    /// Relabel a well-formed grid as another version, leaving its codes alone.
    ///
    /// This is the whole hazard in one function: the codes are valid, the tensor is
    /// the right shape, and only the assignment they were minted under has changed.
    /// Nothing downstream could notice.
    fn relabel<T: CognitiveType>(grid: CodeGrid<T>, codebook: CodebookVersion) -> CodeGrid<T> {
        CodeGrid { codebook, ..grid }
    }

    fn slot_types(device: &Device) -> CodeGrid<SemanticRole> {
        let roles: [&[SemanticRole]; 1] = [&[SemanticRole::Goal, SemanticRole::Action]];
        CodeGrid::new(&Codebook::V1, &roles, device).expect("assigned in v1")
    }

    fn epistemic(device: &Device) -> CodeGrid<EpistemicState> {
        let states: [&[EpistemicState]; 1] = [&[EpistemicState::Observed, EpistemicState::Assumed]];
        CodeGrid::new(&Codebook::V1, &states, device).expect("assigned in v1")
    }

    fn metadata(epistemic: CodeGrid<EpistemicState>, device: &Device) -> PtrSlotMetadata {
        PtrSlotMetadata {
            epistemic,
            provenance_ids: Tensor::<2, Int>::zeros([1, 2], device),
            confidence: Tensor::<2>::ones([1, 2], device),
            admission: admission_bias(
                &[ValidityMask::from_validities(&[Validity::Live; 2])],
                device,
            ),
        }
    }

    /// Two committed payloads, one per slot.
    fn slot_values(device: &Device) -> SlotValues {
        let encoding = SlotEncoding::V1;
        let goal = encoding
            .encode(&ptr_types::TypeId::from("Text"), b"a goal", D_MODEL)
            .unwrap();
        let claim = encoding
            .encode(&ptr_types::TypeId::from("Document"), b"a claim", D_MODEL)
            .unwrap();
        SlotValues::new(&[&[goal, claim]], device).unwrap()
    }

    fn run(slot_types: CodeGrid<SemanticRole>, epistemic: CodeGrid<EpistemicState>) {
        let device = Device::flex();
        let model = PtrA0Config::new(VOCAB, D_MODEL)
            .with_provenance_buckets(8)
            .init(&device);
        let tokens = Tensor::<2, Int>::from_data([[1, 2, 3]], &device);
        let values = slot_values(&device);
        let _ = model.forward(tokens, &slot_types, &values, metadata(epistemic, &device));
    }

    /// The control. Without it, the two refusals below are consistent with a
    /// `forward` that panics on this input whatever the versions say.
    #[test]
    fn matching_versions_pass_through_the_guard() {
        let device = Device::flex();
        run(slot_types(&device), epistemic(&device));
    }

    #[test]
    #[should_panic(expected = "slot codes come from another codebook version")]
    fn slot_codes_from_another_version_are_refused() {
        let device = Device::flex();
        run(relabel(slot_types(&device), FOREIGN), epistemic(&device));
    }

    #[test]
    #[should_panic(expected = "epistemic codes come from another codebook version")]
    fn epistemic_codes_from_another_version_are_refused() {
        let device = Device::flex();
        run(slot_types(&device), relabel(epistemic(&device), FOREIGN));
    }

    /// Relabel a well-formed batch of slot values as another encoding definition,
    /// leaving the numbers alone.
    ///
    /// The whole hazard in one function, exactly as `relabel` is for codes: the
    /// values are real, the tensor is the right shape, and only the definition that
    /// produced them has changed. Unlike a code, a slot value indexes nothing, so
    /// there is no table bound and no shape for anything downstream to notice.
    fn reencode(values: SlotValues, encoding: EncodingVersion) -> SlotValues {
        SlotValues { encoding, ..values }
    }

    /// Slot values at a width that is not the model's `d_model`.
    fn at_width(width: usize, device: &Device) -> SlotValues {
        let encoding = SlotEncoding::V1;
        let vectors: Vec<_> = ["a goal", "a claim"]
            .iter()
            .map(|payload| {
                encoding
                    .encode(
                        &ptr_types::TypeId::from("Document"),
                        payload.as_bytes(),
                        width,
                    )
                    .unwrap()
            })
            .collect();
        SlotValues::new(&[vectors.as_slice()], device).unwrap()
    }

    /// Run `forward` with a given batch of slot values, everything else well formed.
    fn run_values(values: &SlotValues) {
        let device = Device::flex();
        let model = PtrA0Config::new(VOCAB, D_MODEL)
            .with_provenance_buckets(8)
            .init(&device);
        let tokens = Tensor::<2, Int>::from_data([[1, 2, 3]], &device);
        let _ = model.forward(
            tokens,
            &slot_types(&device),
            values,
            metadata(epistemic(&device), &device),
        );
    }

    #[test]
    #[should_panic(expected = "slot values come from another slot-encoding version")]
    fn slot_values_from_another_encoding_version_are_refused() {
        let device = Device::flex();
        run_values(&reencode(slot_values(&device), EncodingVersion(2)));
    }

    /// A width that is not `d_model` must be refused **by name**.
    ///
    /// Asserting only that it panics would prove nothing: the tensor addition that
    /// follows panics on a mismatched last dimension all by itself, so a bare
    /// `should_panic` here passes whether this guard exists or not. That is how a
    /// first version of this test managed to survive deleting the guard it was
    /// written for.
    #[test]
    #[should_panic(expected = "slot values must cover every slot at the model's width")]
    fn slot_values_at_another_width_are_refused_by_name() {
        let device = Device::flex();
        run_values(&at_width(D_MODEL + 1, &device));
    }

    /// The control for both: the well-formed batch this module builds passes.
    #[test]
    fn well_formed_slot_values_pass_through_the_guards() {
        let device = Device::flex();
        run_values(&slot_values(&device));
    }

    /// The two guards are distinguishable, which is what makes the two tests above
    /// separate evidence rather than one property asserted twice.
    #[test]
    fn the_two_guards_do_not_share_a_message() {
        let device = Device::flex();
        let slot = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run(relabel(slot_types(&device), FOREIGN), epistemic(&device))
        }))
        .expect_err("the slot guard fires");
        let epistemic_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run(slot_types(&device), relabel(epistemic(&device), FOREIGN))
        }))
        .expect_err("the epistemic guard fires");

        let message = |payload: &Box<dyn std::any::Any + Send>| -> String {
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .expect("a string panic payload")
        };
        assert_ne!(message(&slot), message(&epistemic_panic));
    }
}
