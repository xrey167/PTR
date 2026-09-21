use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{
        activation::{gelu, softmax},
        Int,
    },
};
use ptr_types::ValidityMask;

#[derive(Clone, Debug)]
pub struct PtrA0Config {
    pub vocab_size: usize,
    pub slot_type_count: usize,
    pub epistemic_count: usize,
    pub provenance_bucket_count: usize,
    pub d_model: usize,
    pub operator_count: usize,
    pub latent_steps: usize,
}

impl PtrA0Config {
    pub fn new(
        vocab_size: usize,
        slot_type_count: usize,
        d_model: usize,
        operator_count: usize,
    ) -> Self {
        Self {
            vocab_size,
            slot_type_count,
            epistemic_count: 8,
            provenance_bucket_count: 64,
            d_model,
            operator_count,
            latent_steps: 0,
        }
    }

    pub fn with_metadata_sizes(
        mut self,
        epistemic_count: usize,
        provenance_bucket_count: usize,
    ) -> Self {
        self.epistemic_count = epistemic_count;
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
            slot_type_embedding: EmbeddingConfig::new(self.slot_type_count, self.d_model)
                .init(device),
            epistemic_embedding: EmbeddingConfig::new(self.epistemic_count, self.d_model)
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
            router: LinearConfig::new(self.d_model, self.operator_count).init(device),
            d_model: self.d_model,
            operator_count: self.operator_count,
            latent_steps: self.latent_steps,
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
}

pub struct PtrSlotMetadata {
    pub epistemic_ids: Tensor<2, Int>,
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
    pub fn forward(
        &self,
        token_ids: Tensor<2, Int>,
        slot_type_ids: Tensor<2, Int>,
        slot_values: Tensor<3>,
        metadata: PtrSlotMetadata,
    ) -> PtrA0Output {
        let [batch, sequence] = token_ids.dims();
        let [slot_batch, slot_count] = slot_type_ids.dims();
        assert_eq!(
            batch, slot_batch,
            "raw and typed paths need the same batch size"
        );
        assert_eq!(metadata.epistemic_ids.dims(), [batch, slot_count]);
        assert_eq!(metadata.provenance_ids.dims(), [batch, slot_count]);
        assert_eq!(metadata.confidence.dims(), [batch, slot_count]);
        assert_eq!(metadata.admission.dims(), [batch, slot_count]);

        let raw = self.token_embedding.forward(token_ids);
        let slot_type = self.slot_type_embedding.forward(slot_type_ids);
        let epistemic = self.epistemic_embedding.forward(metadata.epistemic_ids);
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
