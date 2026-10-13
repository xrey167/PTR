//! Tiny mixture-of-experts MLP for the reference Qwen2 stack (experiment R010 substrate).
//!
//! A MoE layer replaces the dense MLP of one decoder layer with a router
//! (`Linear(n_experts x hidden)`, no bias) and `n_experts` SwiGLU experts
//! (`gate`/`up`: `expert_intermediate x hidden`, `down`: `hidden x expert_intermediate`).
//!
//! Routing, per token:
//!
//! * softmax over **all** router logits (live or not);
//! * the top-`k` experts by probability among the **live** ones, the lowest index winning
//!   ties, with the probabilities used **as is** (no renormalisation);
//! * the layer output is `sum_k p_k * expert_k(x)` in selection order.
//!
//! A revoked or retired expert is masked from the selection (its score is `-inf`). The
//! softmax denominator still contains it, so masking an expert that the token would not have
//! been routed to leaves every output bit unchanged, and masking one that it would have been
//! routed to hands its slot to the next live expert at that expert's original probability.
//!
//! [`Qwen2RoutingMode::Dropless`] has no capacity limit, so the routing of a token depends on
//! the token's own hidden state only and is invariant to what else runs in the same batch.
//! [`Qwen2RoutingMode::BatchVariantCapacity`] is the **negative control**: each expert accepts
//! at most `capacity` tokens per batch step and later tokens routed to a full expert are
//! dropped, so a request's output depends on the load the co-batched requests put on the same
//! experts ([`Qwen2PagedStack::prefill_cobatched_in_regime`](crate::Qwen2PagedStack)).
//!
//! Expert and router weights come from their own seeded domain
//! (`seed`, `"ptr.qwen2.moe"`, layer, expert, generation), so the dense reference weights of
//! the stack are unchanged. The dense `gate`/`up`/`down` of a layer that carries a MoE are
//! not used by the forward pass; they stay in the weights and in the chain digest.

use crate::qwen::{silu, Qwen2LinearWeights};
use crate::qwen_stack::{domain_seed, Qwen2StackConfig, SeededStream};
use crate::KvBackendError;

const MOE_DOMAIN: &str = "ptr.qwen2.moe";

/// How tokens are admitted to experts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen2RoutingMode {
    /// No capacity limit: per-token routing, independent of co-batched requests.
    Dropless,
    /// Negative control: at most `capacity` tokens per expert per batch step; the rest are
    /// dropped (that expert contributes nothing for them).
    BatchVariantCapacity { capacity: usize },
}

/// Whether a stack's output for a request can depend on the other requests of a batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutingDeterminism {
    /// No MoE layer: nothing is routed.
    NotRouted,
    /// MoE with per-token routing: outputs are independent of the batch composition.
    BatchInvariant,
    /// MoE whose drops depend on the co-batched load: reuse of its KV is approximate.
    BatchVariant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen2MoeError {
    NoLayers,
    LayerOutOfRange {
        layer: usize,
        num_layers: usize,
    },
    DuplicateLayer(usize),
    ZeroExperts,
    ZeroExpertIntermediate,
    /// `top_k` is zero or larger than the number of experts.
    InvalidTopK {
        top_k: usize,
        experts: usize,
    },
    ZeroCapacity,
    NotAMoeLayer(usize),
    ExpertOutOfRange {
        layer: usize,
        expert: usize,
    },
    /// Every expert of the layer is masked.
    NoLiveExpert {
        layer: usize,
    },
    /// A weight shape or the stack hidden size does not match.
    Shape,
    NonFinite,
}

impl std::fmt::Display for Qwen2MoeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Qwen2MoeError {}

impl From<KvBackendError> for Qwen2MoeError {
    fn from(_: KvBackendError) -> Self {
        Self::Shape
    }
}

/// What to build.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen2MoeSpec {
    /// Seed of the MoE domain (independent of the dense `weight_seed` stream).
    pub seed: u64,
    /// Decoder layers whose MLP is a MoE.
    pub moe_layers: Vec<usize>,
    pub experts: usize,
    pub expert_intermediate: usize,
    pub top_k: usize,
    pub mode: Qwen2RoutingMode,
}

impl Qwen2MoeSpec {
    /// R010 pin: a MoE in every layer, 4 experts of intermediate size 64, top-2, dropless.
    pub fn pinned(seed: u64, num_layers: usize) -> Self {
        Self {
            seed,
            moe_layers: (0..num_layers).collect(),
            experts: 4,
            expert_intermediate: 64,
            top_k: 2,
            mode: Qwen2RoutingMode::Dropless,
        }
    }
}

/// One expert: SwiGLU weights, the generation of the pod that owns the slot, and whether it
/// may be routed to.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2Expert {
    pub gate: Qwen2LinearWeights,
    pub up: Qwen2LinearWeights,
    pub down: Qwen2LinearWeights,
    pub generation: u64,
    pub live: bool,
}

impl Qwen2Expert {
    fn apply(&self, input: &[f32]) -> Result<Vec<f32>, KvBackendError> {
        let gate = self.gate.apply(input)?;
        let up = self.up.apply(input)?;
        let activated: Vec<f32> = gate
            .into_iter()
            .zip(up)
            .map(|(gate, up)| silu(gate) * up)
            .collect();
        self.down.apply(&activated)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2MoeLayer {
    /// `n_experts x hidden`, no bias.
    pub router: Qwen2LinearWeights,
    pub experts: Vec<Qwen2Expert>,
}

/// The MoE layers of a stack plus the routing rule.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2Moe {
    seed: u64,
    top_k: usize,
    mode: Qwen2RoutingMode,
    expert_intermediate: usize,
    layers: Vec<Option<Qwen2MoeLayer>>,
}

/// Tokens admitted per `(layer, expert)` in one batch step.
#[derive(Clone, Debug)]
pub(crate) struct ExpertLoads {
    loads: Vec<Vec<usize>>,
}

impl ExpertLoads {
    pub(crate) fn new(moe: &Qwen2Moe) -> Self {
        Self {
            loads: moe
                .layers
                .iter()
                .map(|layer| vec![0; layer.as_ref().map_or(0, |l| l.experts.len())])
                .collect(),
        }
    }
}

/// How often each expert was selected and how often a selection was dropped for capacity,
/// per layer, over the tokens of a run.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RoutingCounts {
    selected: Vec<Vec<u64>>,
    dropped: Vec<Vec<u64>>,
}

impl RoutingCounts {
    pub(crate) fn new(moe: &Qwen2Moe) -> Self {
        let shape: Vec<Vec<u64>> = moe
            .layers
            .iter()
            .map(|layer| vec![0; layer.as_ref().map_or(0, |l| l.experts.len())])
            .collect();
        Self {
            selected: shape.clone(),
            dropped: shape,
        }
    }

    /// Tokens routed to `expert` of `layer` (counting drops as selected).
    pub fn selected(&self, layer: usize, expert: usize) -> u64 {
        self.selected
            .get(layer)
            .and_then(|row| row.get(expert))
            .copied()
            .unwrap_or(0)
    }

    /// Selections that were dropped by the capacity rule.
    pub fn dropped(&self, layer: usize, expert: usize) -> u64 {
        self.dropped
            .get(layer)
            .and_then(|row| row.get(expert))
            .copied()
            .unwrap_or(0)
    }

    pub fn total_dropped(&self) -> u64 {
        self.dropped.iter().flatten().sum()
    }

    pub fn layer_selected(&self, layer: usize) -> &[u64] {
        self.selected.get(layer).map_or(&[], Vec::as_slice)
    }

    /// The expert of `layer` selected most often, the lowest index on ties; `None` when the
    /// layer is not a MoE layer or nothing was selected.
    pub fn most_routed(&self, layer: usize) -> Option<usize> {
        let row = self.selected.get(layer)?;
        let mut best: Option<(usize, u64)> = None;
        for (expert, count) in row.iter().enumerate() {
            if *count > 0 && best.is_none_or(|(_, top)| *count > top) {
                best = Some((expert, *count));
            }
        }
        best.map(|(expert, _)| expert)
    }

    pub(crate) fn merge_selection(&mut self, layer: usize, expert: usize, dropped: bool) {
        self.selected[layer][expert] += 1;
        if dropped {
            self.dropped[layer][expert] += 1;
        }
    }
}

impl Qwen2Moe {
    pub fn seeded(config: &Qwen2StackConfig, spec: &Qwen2MoeSpec) -> Result<Self, Qwen2MoeError> {
        config.validate().map_err(|_| Qwen2MoeError::Shape)?;
        if spec.moe_layers.is_empty() {
            return Err(Qwen2MoeError::NoLayers);
        }
        if spec.experts == 0 {
            return Err(Qwen2MoeError::ZeroExperts);
        }
        if spec.expert_intermediate == 0 {
            return Err(Qwen2MoeError::ZeroExpertIntermediate);
        }
        if spec.top_k == 0 || spec.top_k > spec.experts {
            return Err(Qwen2MoeError::InvalidTopK {
                top_k: spec.top_k,
                experts: spec.experts,
            });
        }
        if matches!(
            spec.mode,
            Qwen2RoutingMode::BatchVariantCapacity { capacity: 0 }
        ) {
            return Err(Qwen2MoeError::ZeroCapacity);
        }
        let hidden = config.layer.hidden_size;
        let mut layers: Vec<Option<Qwen2MoeLayer>> = (0..config.num_layers).map(|_| None).collect();
        for layer in &spec.moe_layers {
            let slot = layers
                .get_mut(*layer)
                .ok_or(Qwen2MoeError::LayerOutOfRange {
                    layer: *layer,
                    num_layers: config.num_layers,
                })?;
            if slot.is_some() {
                return Err(Qwen2MoeError::DuplicateLayer(*layer));
            }
            let mut router_stream = SeededStream::new(domain_seed(
                MOE_DOMAIN,
                &[
                    &spec.seed.to_le_bytes(),
                    b"router",
                    &(*layer as u64).to_le_bytes(),
                ],
            ));
            let router = router_stream.linear(spec.experts, hidden, false);
            let experts = (0..spec.experts)
                .map(|expert| {
                    seeded_expert(
                        spec.seed,
                        *layer,
                        expert,
                        1,
                        hidden,
                        spec.expert_intermediate,
                    )
                })
                .collect();
            *slot = Some(Qwen2MoeLayer { router, experts });
        }
        Ok(Self {
            seed: spec.seed,
            top_k: spec.top_k,
            mode: spec.mode,
            expert_intermediate: spec.expert_intermediate,
            layers,
        })
    }

    pub(crate) fn validate_for(&self, config: &Qwen2StackConfig) -> Result<(), Qwen2MoeError> {
        if self.layers.len() != config.num_layers {
            return Err(Qwen2MoeError::Shape);
        }
        let hidden = config.layer.hidden_size;
        let finite = |linear: &Qwen2LinearWeights| linear.weight.iter().all(|v| v.is_finite());
        for layer in self.layers.iter().flatten() {
            layer.router.validate()?;
            if (layer.router.output_size, layer.router.input_size) != (layer.experts.len(), hidden)
                || layer.experts.is_empty()
                || self.top_k == 0
                || self.top_k > layer.experts.len()
            {
                return Err(Qwen2MoeError::Shape);
            }
            if !finite(&layer.router) {
                return Err(Qwen2MoeError::NonFinite);
            }
            for expert in &layer.experts {
                for linear in [&expert.gate, &expert.up, &expert.down] {
                    linear.validate()?;
                    if !finite(linear) {
                        return Err(Qwen2MoeError::NonFinite);
                    }
                }
                if (
                    expert.gate.input_size,
                    expert.up.input_size,
                    expert.down.output_size,
                ) != (hidden, hidden, hidden)
                    || expert.gate.output_size != expert.up.output_size
                    || expert.down.input_size != expert.gate.output_size
                {
                    return Err(Qwen2MoeError::Shape);
                }
            }
        }
        Ok(())
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn mode(&self) -> Qwen2RoutingMode {
        self.mode
    }

    pub fn expert_intermediate(&self) -> usize {
        self.expert_intermediate
    }

    pub fn is_moe_layer(&self, layer: usize) -> bool {
        self.layer(layer).is_some()
    }

    pub fn layer(&self, layer: usize) -> Option<&Qwen2MoeLayer> {
        self.layers.get(layer).and_then(Option::as_ref)
    }

    pub fn routing_determinism(&self) -> RoutingDeterminism {
        match self.mode {
            Qwen2RoutingMode::Dropless => RoutingDeterminism::BatchInvariant,
            Qwen2RoutingMode::BatchVariantCapacity { .. } => RoutingDeterminism::BatchVariant,
        }
    }

    pub fn with_mode(mut self, mode: Qwen2RoutingMode) -> Result<Self, Qwen2MoeError> {
        if matches!(mode, Qwen2RoutingMode::BatchVariantCapacity { capacity: 0 }) {
            return Err(Qwen2MoeError::ZeroCapacity);
        }
        self.mode = mode;
        Ok(self)
    }

    fn expert_mut(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<&mut Qwen2Expert, Qwen2MoeError> {
        let slot = self
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or(Qwen2MoeError::NotAMoeLayer(layer))?;
        slot.experts
            .get_mut(expert)
            .ok_or(Qwen2MoeError::ExpertOutOfRange { layer, expert })
    }

    /// Mask (`live = false`) or unmask an expert; weights and generation are unchanged.
    pub fn set_expert_live(
        mut self,
        layer: usize,
        expert: usize,
        live: bool,
    ) -> Result<Self, Qwen2MoeError> {
        self.expert_mut(layer, expert)?.live = live;
        Ok(self)
    }

    /// Replace expert `(layer, expert)` by the one the seeded domain yields for
    /// `generation` (new weights, `generation` recorded, live).
    pub fn with_expert_generation(
        mut self,
        config: &Qwen2StackConfig,
        layer: usize,
        expert: usize,
        generation: u64,
    ) -> Result<Self, Qwen2MoeError> {
        let (seed, intermediate) = (self.seed, self.expert_intermediate);
        *self.expert_mut(layer, expert)? = seeded_expert(
            seed,
            layer,
            expert,
            generation,
            config.layer.hidden_size,
            intermediate,
        );
        Ok(self)
    }

    /// Same weights, a different generation (a supersession that did not change the bits).
    pub fn with_expert_generation_only(
        mut self,
        layer: usize,
        expert: usize,
        generation: u64,
    ) -> Result<Self, Qwen2MoeError> {
        self.expert_mut(layer, expert)?.generation = generation;
        Ok(self)
    }

    /// Run the MoE of `layer` on one token's post-attention hidden state.
    ///
    /// `loads` is the per-step load shared by co-batched requests; without it each token
    /// sees fresh loads, so a lone request never drops while `capacity >= 1`.
    pub(crate) fn apply(
        &self,
        layer: usize,
        input: &[f32],
        loads: Option<&mut ExpertLoads>,
        trace: Option<&mut RoutingCounts>,
    ) -> Result<Vec<f32>, Qwen2MoeError> {
        let moe = self
            .layer(layer)
            .ok_or(Qwen2MoeError::NotAMoeLayer(layer))?;
        let logits = moe.router.apply(input)?;
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f32> = logits.iter().map(|logit| (logit - maximum).exp()).collect();
        let denominator: f32 = exponentials.iter().sum();
        let probabilities: Vec<f32> = exponentials
            .iter()
            .map(|value| value / denominator)
            .collect();

        let live = moe.experts.iter().filter(|expert| expert.live).count();
        if live == 0 {
            return Err(Qwen2MoeError::NoLiveExpert { layer });
        }
        let mut chosen: Vec<usize> = Vec::with_capacity(self.top_k);
        for _ in 0..self.top_k.min(live) {
            let mut best: Option<usize> = None;
            for (index, expert) in moe.experts.iter().enumerate() {
                if !expert.live || chosen.contains(&index) {
                    continue;
                }
                // Strict comparison: the lowest index wins ties.
                if best.is_none_or(|top| probabilities[index] > probabilities[top]) {
                    best = Some(index);
                }
            }
            chosen.push(best.expect("a live unchosen expert exists while chosen < live"));
        }

        let mut fresh;
        let loads = match loads {
            Some(loads) => loads,
            None => {
                fresh = ExpertLoads::new(self);
                &mut fresh
            }
        };
        let mut trace = trace;
        let hidden = moe.router.input_size;
        let mut output = vec![0.0_f32; hidden];
        for expert_index in chosen {
            let admitted = match self.mode {
                Qwen2RoutingMode::Dropless => true,
                Qwen2RoutingMode::BatchVariantCapacity { capacity } => {
                    loads.loads[layer][expert_index] < capacity
                }
            };
            loads.loads[layer][expert_index] += 1;
            if let Some(trace) = trace.as_deref_mut() {
                trace.merge_selection(layer, expert_index, !admitted);
            }
            if !admitted {
                continue;
            }
            let contribution = moe.experts[expert_index].apply(input)?;
            let weight = probabilities[expert_index];
            for (sum, value) in output.iter_mut().zip(contribution) {
                *sum += weight * value;
            }
        }
        Ok(output)
    }
}

fn seeded_expert(
    seed: u64,
    layer: usize,
    expert: usize,
    generation: u64,
    hidden: usize,
    intermediate: usize,
) -> Qwen2Expert {
    let mut stream = SeededStream::new(domain_seed(
        MOE_DOMAIN,
        &[
            &seed.to_le_bytes(),
            b"expert",
            &(layer as u64).to_le_bytes(),
            &(expert as u64).to_le_bytes(),
            &generation.to_le_bytes(),
        ],
    ));
    Qwen2Expert {
        gate: stream.linear(intermediate, hidden, false),
        up: stream.linear(intermediate, hidden, false),
        down: stream.linear(hidden, intermediate, false),
        generation,
        live: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KvKeyEncoding, Qwen2Activation, Qwen2LayerConfig, Qwen2QkNorm, Qwen2RopeScaling};

    fn config() -> Qwen2StackConfig {
        Qwen2StackConfig {
            layer: Qwen2LayerConfig {
                hidden_size: 8,
                intermediate_size: 16,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                max_position_embeddings: 32,
                rope_theta: 10_000.0,
                rope_scaling: Qwen2RopeScaling::None,
                rms_norm_eps: 1e-6,
                hidden_act: Qwen2Activation::Silu,
                sliding_window: None,
            },
            num_layers: 1,
            vocab_size: 11,
            key_encoding: KvKeyEncoding::PreRope,
            qk_norm: Qwen2QkNorm::Disabled,
        }
    }

    fn moe_with_tied_router() -> Qwen2Moe {
        let spec = Qwen2MoeSpec {
            expert_intermediate: 8,
            ..Qwen2MoeSpec::pinned(3, 1)
        };
        let mut moe = Qwen2Moe::seeded(&config(), &spec).unwrap();
        // Zero router: every probability is exactly 1/4, a four-way tie.
        moe.layers[0].as_mut().unwrap().router.weight.fill(0.0);
        moe
    }

    fn input() -> Vec<f32> {
        (0..8).map(|i| 0.1 * i as f32 - 0.3).collect()
    }

    #[test]
    fn ties_go_to_the_lowest_index_and_probabilities_are_not_renormalised() {
        let moe = moe_with_tied_router();
        let mut counts = RoutingCounts::new(&moe);
        let output = moe.apply(0, &input(), None, Some(&mut counts)).unwrap();
        assert_eq!(counts.layer_selected(0), &[1, 1, 0, 0]);
        let layer = moe.layer(0).unwrap();
        let first = layer.experts[0].apply(&input()).unwrap();
        let second = layer.experts[1].apply(&input()).unwrap();
        // Weight 0.25 each (softmax of four equal logits), not 0.5 after renormalising.
        let expected: Vec<f32> = first
            .iter()
            .zip(&second)
            .map(|(a, b)| (0.0 + 0.25 * a) + 0.25 * b)
            .collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn a_masked_expert_hands_its_slot_to_the_next_index_at_its_own_probability() {
        let moe = moe_with_tied_router().set_expert_live(0, 0, false).unwrap();
        let mut counts = RoutingCounts::new(&moe);
        let output = moe.apply(0, &input(), None, Some(&mut counts)).unwrap();
        assert_eq!(counts.layer_selected(0), &[0, 1, 1, 0]);
        let layer = moe.layer(0).unwrap();
        let first = layer.experts[1].apply(&input()).unwrap();
        let second = layer.experts[2].apply(&input()).unwrap();
        let expected: Vec<f32> = first
            .iter()
            .zip(&second)
            .map(|(a, b)| (0.0 + 0.25 * a) + 0.25 * b)
            .collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn capacity_counts_loads_across_a_shared_step_and_resets_per_step() {
        let moe = moe_with_tied_router()
            .with_mode(Qwen2RoutingMode::BatchVariantCapacity { capacity: 1 })
            .unwrap();
        let mut loads = ExpertLoads::new(&moe);
        let mut counts = RoutingCounts::new(&moe);
        let first = moe
            .apply(0, &input(), Some(&mut loads), Some(&mut counts))
            .unwrap();
        // The second token of the same step finds experts 0 and 1 full and is dropped.
        let second = moe
            .apply(0, &input(), Some(&mut loads), Some(&mut counts))
            .unwrap();
        assert!(first.iter().any(|v| *v != 0.0));
        assert!(second.iter().all(|v| *v == 0.0));
        assert_eq!(counts.dropped(0, 0), 1);
        assert_eq!(counts.dropped(0, 1), 1);
        assert_eq!(counts.total_dropped(), 2);
        // Fresh loads (the next step) admit it again.
        let mut next_step = ExpertLoads::new(&moe);
        assert_eq!(
            moe.apply(0, &input(), Some(&mut next_step), None).unwrap(),
            first
        );
    }
}
