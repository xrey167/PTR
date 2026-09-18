use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{activation::softmax, Int},
};

#[derive(Clone, Debug)]
pub struct PtrA0Config {
    pub vocab_size: usize,
    pub slot_type_count: usize,
    pub d_model: usize,
    pub operator_count: usize,
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
            d_model,
            operator_count,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> PtrA0<B> {
        let linear = || LinearConfig::new(self.d_model, self.d_model).init(device);
        PtrA0 {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            slot_type_embedding: EmbeddingConfig::new(self.slot_type_count, self.d_model)
                .init(device),
            slot_query: linear(),
            raw_key: linear(),
            raw_value: linear(),
            slot_output: linear(),
            raw_query: linear(),
            slot_key: linear(),
            slot_value: linear(),
            raw_output: linear(),
            router: LinearConfig::new(self.d_model, self.operator_count).init(device),
            d_model: self.d_model,
            operator_count: self.operator_count,
        }
    }
}

#[derive(Module, Debug)]
pub struct PtrA0<B: Backend> {
    token_embedding: Embedding<B>,
    slot_type_embedding: Embedding<B>,
    slot_query: Linear<B>,
    raw_key: Linear<B>,
    raw_value: Linear<B>,
    slot_output: Linear<B>,
    raw_query: Linear<B>,
    slot_key: Linear<B>,
    slot_value: Linear<B>,
    raw_output: Linear<B>,
    router: Linear<B>,
    d_model: usize,
    operator_count: usize,
}

pub struct PtrA0Output<B: Backend> {
    pub raw: Tensor<B, 3>,
    pub slots: Tensor<B, 3>,
    pub router_logits: Tensor<B, 2>,
}

impl<B: Backend> PtrA0<B> {
    pub fn forward(
        &self,
        token_ids: Tensor<B, 2, Int>,
        slot_type_ids: Tensor<B, 2, Int>,
        slot_values: Tensor<B, 3>,
    ) -> PtrA0Output<B> {
        let [batch, _sequence] = token_ids.dims();
        let [slot_batch, slot_count] = slot_type_ids.dims();
        assert_eq!(batch, slot_batch, "raw and typed paths need the same batch size");

        let raw = self.token_embedding.forward(token_ids);
        let typed = self.slot_type_embedding.forward(slot_type_ids);
        let slots = slot_values + typed;

        let slot_query = self.slot_query.forward(slots.clone());
        let raw_key = self.raw_key.forward(raw.clone());
        let raw_value = self.raw_value.forward(raw.clone());
        let slot_scores = slot_query
            .matmul(raw_key.transpose())
            .div_scalar((self.d_model as f32).sqrt());
        let slot_weights = softmax(slot_scores, 2);
        let slot_context = slot_weights.matmul(raw_value);
        let slots = slots + self.slot_output.forward(slot_context);

        let raw_query = self.raw_query.forward(raw.clone());
        let slot_key = self.slot_key.forward(slots.clone());
        let slot_value = self.slot_value.forward(slots.clone());
        let raw_scores = raw_query
            .matmul(slot_key.transpose())
            .div_scalar((self.d_model as f32).sqrt());
        let raw_weights = softmax(raw_scores, 2);
        let raw_context = raw_weights.matmul(slot_value);
        let raw = raw + self.raw_output.forward(raw_context);

        let router_logits = self
            .router
            .forward(slots.clone())
            .mean_dim(1)
            .reshape([batch, self.operator_count]);

        debug_assert_eq!(slots.dims()[1], slot_count);
        PtrA0Output {
            raw,
            slots,
            router_logits,
        }
    }
}
