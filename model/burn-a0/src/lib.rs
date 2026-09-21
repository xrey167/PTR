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
    CodeFamily, Codebook, CodebookError, CodebookVersion, CognitiveType, EpistemicState,
    ReasoningOperator, SemanticRole, ValidityMask,
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
    codebook: Codebook,
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
            codebook,
        }
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

    pub fn init(&self, device: &Device) -> PtrA0 {
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
            codebook_version: self.codebook.version().0,
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
    /// Codebook version the tables were sized by. Burn records constants as empty,
    /// so this does **not** survive a saved record: an artifact that has to carry
    /// the identity carries it in its own header.
    codebook_version: u32,
}

impl PtrA0 {
    /// The codebook this model's tables were sized by.
    ///
    /// The lookup cannot fail: a [`PtrA0`] only comes from [`PtrA0Config::init`],
    /// which was given a [`Codebook`], and a `Codebook` only exists for a version
    /// this build defines.
    pub fn codebook(&self) -> Codebook {
        Codebook::at(CodebookVersion(self.codebook_version))
            .expect("the version came from a Codebook this build accepted")
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
    /// use ptr_burn_a0::{admission_bias, CodeGrid, PtrA0Config, PtrSlotMetadata};
    /// use ptr_types::{Codebook, EpistemicState, SemanticRole, Validity, ValidityMask};
    ///
    /// let device = Device::flex();
    /// let book = Codebook::V1;
    /// let model = PtrA0Config::new(8, 12).init(&device);
    /// let roles: [&[SemanticRole]; 1] = [&[SemanticRole::Goal, SemanticRole::Claim]];
    /// let states: [&[EpistemicState]; 1] = [&[EpistemicState::Observed, EpistemicState::Assumed]];
    /// let slot_types = CodeGrid::new(&book, &roles, &device).unwrap();
    /// let epistemic = CodeGrid::new(&book, &states, &device).unwrap();
    /// let output = model.forward(
    ///     Tensor::<2, Int>::from_data([[1, 2]], &device),
    ///     &slot_types,
    ///     Tensor::<3>::zeros([1, 2, 12], &device),
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
    /// use ptr_burn_a0::{admission_bias, CodeGrid, PtrA0Config, PtrSlotMetadata};
    /// use ptr_types::{Codebook, EpistemicState, SemanticRole, Validity, ValidityMask};
    ///
    /// let device = Device::flex();
    /// let book = Codebook::V1;
    /// let model = PtrA0Config::new(8, 12).init(&device);
    /// let roles: [&[SemanticRole]; 1] = [&[SemanticRole::Goal, SemanticRole::Claim]];
    /// let states: [&[EpistemicState]; 1] = [&[EpistemicState::Observed, EpistemicState::Assumed]];
    /// let slot_types = CodeGrid::new(&book, &roles, &device).unwrap();
    /// let epistemic = CodeGrid::new(&book, &states, &device).unwrap();
    /// let _ = model.forward(
    ///     Tensor::<2, Int>::from_data([[1, 2]], &device),
    ///     &epistemic,
    ///     Tensor::<3>::zeros([1, 2, 12], &device),
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
        slot_values: Tensor<3>,
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
        let slots = slot_values + typed_metadata.clone();

        let typed_bias = self.metadata_bias.forward(typed_metadata);
        let slot_query = self.slot_query.forward(slots.clone());
        let raw_key = self.raw_key.forward(raw.clone());
        let cross_bias = typed_cross_bias(typed_bias, raw_key.clone());
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
            let delta = gelu(self.latent_refine.forward(slots.clone()));
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
