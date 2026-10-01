//! Training one arm and scoring it. Everything written to stdout is a
//! deterministic function of the data, the arm, the seeds and the step count;
//! timing goes to stderr, so two runs can be compared byte for byte.

use crate::arms::{Arm, Batch};
use crate::batch::{self, Payloads};
use crate::data::{Dataset, Example, Split, TEST_SPLITS};
use crate::rng::{permutation, splitmix64, SplitMix64};
use burn::{
    module::{AutodiffModule, Module, ModuleVisitor, Param},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    tensor::{activation::softmax, Gradients, Int, TensorData},
};
use ptr_burn_a0::{PlainTransformer, PlainTransformerConfig, PtrA0};
use std::time::Instant;

pub const BATCH: usize = 128;
pub const WARMUP: usize = 100;
pub const EVAL_BATCH: usize = 500;
const OPERATORS: usize = 11;
const ECE_BINS: usize = 15;
/// Parameters are counted as effective when any of the first this many
/// batches gives them a non-zero gradient.
const EFFECTIVE_WINDOW: usize = 10;

fn apply_soft_metadata_dropout(inputs: &mut batch::Inputs, probability: f32, seed: u64) {
    if probability <= 0.0 {
        return;
    }
    let [rows, slots] = inputs.metadata.confidence.dims();
    let device = inputs.metadata.confidence.device();
    let mut rng = SplitMix64::new(seed);
    let keep: Vec<i32> = (0..rows * slots)
        .map(|_| i32::from(rng.u01() >= f64::from(probability)))
        .collect();
    let keep = Tensor::<2, Int>::from_data(TensorData::new(keep, [rows, slots]), &device);
    inputs.metadata.confidence = inputs.metadata.confidence.clone() * keep.clone().float();
    inputs.metadata.provenance_ids = inputs.metadata.provenance_ids.clone() * keep;
}

fn jensen_shannon(left_logits: Tensor<2>, right_logits: Tensor<2>) -> Tensor<1> {
    let left = softmax(left_logits, 1);
    let right = softmax(right_logits, 1);
    let midpoint = (left.clone() + right.clone()) * 0.5;
    let epsilon = 1.0e-7;
    let left_kl = (left.clone() * ((left + epsilon).log() - (midpoint.clone() + epsilon).log()))
        .sum_dim(1)
        .mean();
    let right_kl = (right.clone() * ((right + epsilon).log() - (midpoint + epsilon).log()))
        .sum_dim(1)
        .mean();
    (left_kl + right_kl) * 0.5
}

pub struct Schedule {
    pub steps: usize,
    pub peak: f64,
}

impl Schedule {
    /// Return the learning rate for a zero-based step: linear warm-up for steps
    /// 0 through 99, then cosine decay toward 0.1 x peak.
    /// For `steps > 100`, the floor is reached at `step == steps`, one step after
    /// the training loop's last update. A schedule shorter than 100 steps uses
    /// only warm-up when queried within its training range.
    pub fn at(&self, step: usize) -> f64 {
        if step < WARMUP {
            return self.peak * (step + 1) as f64 / WARMUP as f64;
        }
        let span = (self.steps - WARMUP).max(1) as f64;
        let progress = ((step - WARMUP) as f64 / span).min(1.0);
        0.1 * self.peak + 0.9 * self.peak * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

pub struct Seeds {
    pub init: u64,
    pub order: u64,
}

/// What training one arm produced, besides the model.
pub struct Trained {
    pub model: PtrA0,
    pub rows: Vec<String>,
    pub nan: bool,
    pub milliseconds_per_step: f64,
}

/// Output of the token-only matched baseline. It deliberately uses the same
/// schedule, batch order and scoring contract as the PTR arms, while exposing a
/// separate model type so the comparison cannot accidentally consume typed
/// metadata.
pub struct PlainTrained {
    pub model: PlainTransformer,
    pub rows: Vec<String>,
    pub nan: bool,
    pub milliseconds_per_step: f64,
}

/// Sums |gradient| per parameter element over the first batches.
struct GradientTally<'a> {
    gradients: &'a Gradients,
    sums: &'a mut Vec<Option<Tensor<1>>>,
    position: usize,
}

impl ModuleVisitor for GradientTally<'_> {
    /// Accumulate absolute gradients per element, retaining slots without gradients.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if self.sums.len() <= self.position {
            self.sums.push(None);
        }
        if let Some(gradient) = param.grad(self.gradients) {
            let elements = gradient.shape().num_elements();
            let flat = gradient.abs().reshape([elements]);
            let slot = &mut self.sums[self.position];
            *slot = Some(match slot.take() {
                Some(sum) => sum + flat,
                None => flat,
            });
        }
        self.position += 1;
    }
}

/// Count parameter elements with positive accumulated gradient magnitude.
fn effective(sums: &[Option<Tensor<1>>]) -> usize {
    sums.iter()
        .flatten()
        .map(|sum| {
            let nonzero: i64 = sum.clone().greater_elem(0.0).int().sum().into_scalar();
            nonzero as usize
        })
        .sum()
}

/// Encode a switch as the string label used in study rows.
fn on(flag: bool) -> &'static str {
    if flag {
        "on"
    } else {
        "off"
    }
}

/// Train `arm` on full batches of 128, reseeding model initialization with `seeds.init`.
///
/// `seeds.order` controls epoch shuffles; each epoch drops its incomplete batch.
/// Returns the model, JSON rows, whether any loss was non-finite, and elapsed
/// milliseconds per step (including validation). When `validate` is true, a row
/// is collected every `max(steps / 10, 1)` steps. Non-finite losses mark `nan` but
/// do not stop training. Payload width must match `d_model`; a nonzero step count
/// requires at least 128 training examples or the epoch calculation panics.
pub fn train(
    arm: &Arm,
    data: &Dataset,
    payloads: &Payloads,
    d_model: usize,
    schedule: &Schedule,
    seeds: &Seeds,
    validate: bool,
) -> Trained {
    let device = Device::flex().autodiff();
    // Every arm builds the same modules, so after this seed every arm of a run
    // starts from the same weights (tests/seeded_init_arms.rs).
    device.seed(seeds.init);
    let mut model = arm.config(d_model).init(&device);
    let total_params = model.num_params();
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new()
        .with_smoothing((arm.label_smoothing > 0.0).then_some(arm.label_smoothing))
        .init(&device);
    let train_split = data.split("train");
    let examples = &train_split.examples;
    let per_epoch = examples.len() / BATCH;
    let eval_every = (schedule.steps / 10).max(1);

    let mut rows = Vec::new();
    let mut order: Vec<usize> = Vec::new();
    let mut sums: Vec<Option<Tensor<1>>> = Vec::new();
    let mut window_loss = 0.0_f64;
    let mut window_steps = 0usize;
    let mut last_loss = f64::NAN;
    let mut nan = false;
    let started = Instant::now();

    for step in 0..schedule.steps {
        let epoch = step / per_epoch;
        let position = step % per_epoch;
        if position == 0 {
            let mut rng = SplitMix64::new(splitmix64(seeds.order ^ epoch as u64));
            order = permutation(examples.len(), &mut rng);
        }
        let chosen: Vec<&Example> = order[position * BATCH..(position + 1) * BATCH]
            .iter()
            .map(|&i| &examples[i])
            .collect();
        let inputs = batch::build(&chosen, arm.batch, payloads, &device);
        let output = model.forward(
            inputs.tokens,
            &inputs.roles,
            &inputs.values,
            inputs.metadata,
        );
        let labels = batch::labels(&chosen, &device);
        let classification = loss_fn.forward(output.router_logits.clone(), labels);
        let loss = if arm.consistency_weight > 0.0 {
            let mut dropped = batch::build(&chosen, arm.batch, payloads, &device);
            apply_soft_metadata_dropout(
                &mut dropped,
                arm.metadata_dropout,
                splitmix64(seeds.order ^ step as u64 ^ 0x4D_45_54_41),
            );
            let dropped_logits = model
                .forward(
                    dropped.tokens,
                    &dropped.roles,
                    &dropped.values,
                    dropped.metadata,
                )
                .router_logits;
            classification
                + jensen_shannon(output.router_logits, dropped_logits) * arm.consistency_weight
        } else {
            classification
        };
        let value: f32 = loss.clone().into_scalar();
        let value = f64::from(value);
        if !value.is_finite() {
            nan = true;
        }
        window_loss += value;
        window_steps += 1;
        last_loss = value;
        let gradients = loss.backward();
        if step < EFFECTIVE_WINDOW {
            let mut tally = GradientTally {
                gradients: &gradients,
                sums: &mut sums,
                position: 0,
            };
            model.visit(&mut tally);
        }
        let gradients = GradientsParams::from_grads(gradients, &model);
        model = optimizer.step(schedule.at(step), model, gradients);

        if validate && (step + 1) % eval_every == 0 {
            let checkpoint = (step + 1) / eval_every;
            let score = score(&model.valid(), arm, data.split("val"), payloads);
            rows.push(format!(
                r#"{{"row":"val","arm":"{}","checkpoint":"{checkpoint}","val_accuracy":{},"train_loss":{}}}"#,
                arm.name,
                score.accuracy(),
                window_loss / window_steps.max(1) as f64
            ));
            window_loss = 0.0;
            window_steps = 0;
        }
    }
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    // Static forward-pass estimate used only for matched-compute validation.
    // Both M002-v5 arms execute this exact graph, including the factorized
    // branch; Off zeros its result only after the branch has run.
    let sequence = 72_u64;
    let slots = 6_u64;
    let width = d_model as u64;
    let rank = arm.typed_attention_rank as u64;
    let model_flops = 8 * sequence * width * width
        + 8 * slots * width * width
        + 4 * slots * sequence * width
        + 2 * slots * sequence * rank
        + 2 * slots * width * OPERATORS as u64;
    rows.insert(
        0,
        format!(
            r#"{{"row":"meta","arm":"{}","typed_attention":"{}","typed_attention_mode":"{}","typed_attention_rank":{},"typed_attention_limit":{},"typed_query":"{}","latent_steps":"{}","latent_nonlinearity":"{}","frozen_router":"{}","router_mode":"{}","router_logit_scale":{},"label_smoothing":{},"metadata_dropout":{},"consistency_weight":{},"batch":"{}","lr":{},"steps":{},"total_params":{},"estimated_flops_per_example":{},"effective_params":{},"final_train_loss":{},"nan":{}}}"#,
            arm.name,
            on(arm.typed_attention),
            arm.typed_attention_mode.name(),
            arm.typed_attention_rank,
            arm.typed_attention_limit,
            on(arm.typed_query),
            arm.latent_steps,
            on(arm.latent_nonlinearity),
            on(arm.frozen_router),
            arm.router_mode.name(),
            arm.router_logit_scale,
            arm.label_smoothing,
            arm.metadata_dropout,
            arm.consistency_weight,
            arm.batch.name(),
            schedule.peak,
            schedule.steps,
            total_params,
            model_flops,
            effective(&sums),
            if last_loss.is_finite() { last_loss } else { -1.0 },
            u8::from(nan),
        ),
    );
    Trained {
        model,
        rows,
        nan,
        milliseconds_per_step: elapsed / schedule.steps.max(1) as f64,
    }
}

/// Train the token-only one-layer Transformer baseline on the identical raw
/// token stream. `Batch::ContentFree { masked: false }` is used only as the
/// shared token loader; no slot or metadata tensor reaches the model.
pub fn train_plain(
    data: &Dataset,
    d_model: usize,
    schedule: &Schedule,
    seeds: &Seeds,
    validate: bool,
) -> PlainTrained {
    let device = Device::flex().autodiff();
    device.seed(seeds.init);
    let mut model = PlainTransformerConfig::new(216, d_model, OPERATORS).init(&device);
    let payloads = Payloads::new(d_model);
    let total_params = model.num_params();
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let examples = &data.split("train").examples;
    let per_epoch = examples.len() / BATCH;
    let eval_every = (schedule.steps / 10).max(1);
    let mut rows = Vec::new();
    let mut order = Vec::new();
    let mut window_loss = 0.0_f64;
    let mut window_steps = 0usize;
    let mut last_loss = f64::NAN;
    let mut nan = false;
    let started = Instant::now();

    for step in 0..schedule.steps {
        let epoch = step / per_epoch;
        let position = step % per_epoch;
        if position == 0 {
            let mut rng = SplitMix64::new(splitmix64(seeds.order ^ epoch as u64));
            order = permutation(examples.len(), &mut rng);
        }
        let chosen: Vec<&Example> = order[position * BATCH..(position + 1) * BATCH]
            .iter()
            .map(|&i| &examples[i])
            .collect();
        let inputs = batch::build(
            &chosen,
            Batch::ContentFree { masked: false },
            &payloads,
            &device,
        );
        let loss = loss_fn.forward(
            model.forward(inputs.tokens),
            batch::labels(&chosen, &device),
        );
        let value: f32 = loss.clone().into_scalar();
        let value = f64::from(value);
        nan |= !value.is_finite();
        window_loss += value;
        window_steps += 1;
        last_loss = value;
        let gradients = loss.backward();
        let gradients = GradientsParams::from_grads(gradients, &model);
        model = optimizer.step(schedule.at(step), model, gradients);
        if validate && (step + 1) % eval_every == 0 {
            let checkpoint = (step + 1) / eval_every;
            let score = score_plain(&model.valid(), data.split("val"), d_model);
            rows.push(format!(
                r#"{{"row":"val","arm":"plain-transformer","checkpoint":"{checkpoint}","val_accuracy":{},"train_loss":{}}}"#,
                score.accuracy(),
                window_loss / window_steps.max(1) as f64
            ));
            window_loss = 0.0;
            window_steps = 0;
        }
    }
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    let sequence = 72_u64;
    let latent_slots = 6_u64;
    let model_flops = 4 * sequence * 2 * d_model as u64 * d_model as u64
        + 6 * latent_slots * 2 * d_model as u64 * d_model as u64
        + 4 * 2 * sequence * latent_slots * d_model as u64
        + 2 * latent_slots * d_model as u64 * OPERATORS as u64;
    rows.insert(
        0,
        format!(
            r#"{{"row":"meta","arm":"plain-transformer","architecture":"token-to-six-latent-cross-attention","sequence":{},"d_model":{},"latent_slots":{},"batch":{},"lr":{},"steps":{},"total_params":{},"estimated_flops_per_example":{},"final_train_loss":{},"nan":{}}}"#,
            sequence,
            d_model,
            latent_slots,
            BATCH,
            schedule.peak,
            schedule.steps,
            total_params,
            model_flops,
            if last_loss.is_finite() { last_loss } else { -1.0 },
            u8::from(nan),
        ),
    );
    PlainTrained {
        model,
        rows,
        nan,
        milliseconds_per_step: elapsed / schedule.steps.max(1) as f64,
    }
}

/// Counts over one split.
pub struct Score {
    pub n: usize,
    pub correct: usize,
    pub nll: f64,
    pub ece: f64,
    pub temperature: f64,
    pub threshold: f64,
    pub covered: usize,
    pub covered_correct: usize,
    /// One lowercase hex digit per example: the predicted operator code.
    pub predictions: String,
    /// Comma-separated `Unknown` or forced-choice operator codes.
    pub decisions: String,
}

/// Forward-only scoring for the plain Transformer using the same metrics and
/// prediction encoding as the PTR arms.
pub fn score_plain(model: &PlainTransformer, split: &Split, d_model: usize) -> Score {
    let device = Device::flex();
    let payloads = Payloads::new(d_model);
    let mut correct = 0;
    let mut nll = 0.0_f64;
    let mut predictions = String::with_capacity(split.examples.len());
    let mut bins = [(0usize, 0.0_f64, 0usize); ECE_BINS];
    for chunk in split.examples.chunks(EVAL_BATCH) {
        let chosen: Vec<&Example> = chunk.iter().collect();
        let inputs = batch::build(
            &chosen,
            Batch::ContentFree { masked: false },
            &payloads,
            &device,
        );
        let logits: Vec<f32> = model
            .forward(inputs.tokens)
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32 logits");
        for (example, row) in chosen.iter().zip(logits.chunks(OPERATORS)) {
            let best = checked_argmax(row);
            let max = f64::from(row[best]);
            let normalizer: f64 = row.iter().map(|&v| (f64::from(v) - max).exp()).sum();
            nll -= f64::from(row[example.label]) - max - normalizer.ln();
            let confidence = 1.0 / normalizer;
            let hit = best == example.label;
            correct += usize::from(hit);
            let bin = ((confidence * ECE_BINS as f64) as usize).min(ECE_BINS - 1);
            bins[bin].0 += 1;
            bins[bin].1 += confidence;
            bins[bin].2 += usize::from(hit);
            predictions.push(char::from_digit(best as u32, 16).expect("a hex digit"));
        }
    }
    let n = split.examples.len();
    let ece = bins
        .iter()
        .filter(|bin| bin.0 > 0)
        .map(|&(count, confidence, hits)| {
            let count_f = count as f64;
            (count_f / n as f64) * (hits as f64 / count_f - confidence / count_f).abs()
        })
        .sum();
    Score {
        n,
        correct,
        nll: nll / n.max(1) as f64,
        ece,
        temperature: 1.0,
        threshold: 0.0,
        covered: n,
        covered_correct: correct,
        decisions: predictions.clone(),
        predictions,
    }
}

impl Score {
    /// Return the fraction of correct predictions, or zero for an empty split.
    pub fn accuracy(&self) -> f64 {
        self.correct as f64 / self.n.max(1) as f64
    }

    pub fn coverage(&self) -> f64 {
        self.covered as f64 / self.n.max(1) as f64
    }

    pub fn selective_error(&self) -> f64 {
        if self.covered == 0 {
            1.0
        } else {
            1.0 - self.covered_correct as f64 / self.covered as f64
        }
    }
}

/// Forward-only scoring in batches of 500. The prediction is the argmax of the
/// logits, an exact tie going to the lowest code.
/// Returns counts, mean negative log-likelihood in natural-log units, expected
/// calibration error over 15 equal-width confidence bins, and prediction codes.
/// An empty split produces zero counts and metrics and an empty prediction string.
pub fn score(model: &PtrA0, arm: &Arm, split: &Split, payloads: &Payloads) -> Score {
    score_calibrated(model, arm, split, payloads, 1.0, 0.0)
}

fn collect_logits(
    model: &PtrA0,
    arm: &Arm,
    split: &Split,
    payloads: &Payloads,
) -> Vec<(usize, Vec<f32>)> {
    let device = Device::flex();
    let mut out = Vec::with_capacity(split.examples.len());
    for chunk in split.examples.chunks(EVAL_BATCH) {
        let chosen: Vec<&Example> = chunk.iter().collect();
        let inputs = batch::build(&chosen, arm.batch, payloads, &device);
        let logits: Vec<f32> = model
            .forward(
                inputs.tokens,
                &inputs.roles,
                &inputs.values,
                inputs.metadata,
            )
            .router_logits
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32 logits");
        for (example, row) in chosen.iter().zip(logits.chunks(OPERATORS)) {
            out.push((example.label, row.to_vec()));
        }
    }
    out
}

fn score_rows(rows: &[(usize, Vec<f32>)], temperature: f64, threshold: f64) -> Score {
    assert!(temperature.is_finite() && temperature > 0.0);
    assert!(threshold.is_finite() && (0.0..=1.0).contains(&threshold));
    let mut correct = 0;
    let mut covered = 0;
    let mut covered_correct = 0;
    let mut nll = 0.0_f64;
    let mut predictions = String::with_capacity(rows.len());
    let mut decisions = String::with_capacity(rows.len());
    let mut bins = [(0usize, 0.0_f64, 0usize); ECE_BINS];
    for (label, raw) in rows {
        let scaled: Vec<f64> = raw
            .iter()
            .map(|value| f64::from(*value) / temperature)
            .collect();
        assert!(scaled.iter().all(|value| value.is_finite()));
        let best = checked_argmax(raw);
        let max = scaled[best];
        let normalizer: f64 = scaled.iter().map(|value| (value - max).exp()).sum();
        let confidence = 1.0 / normalizer;
        nll -= scaled[*label] - max - normalizer.ln();
        let hit = best == *label;
        correct += usize::from(hit);
        let is_covered = confidence >= threshold;
        covered += usize::from(is_covered);
        covered_correct += usize::from(is_covered && hit);
        let bin = ((confidence * ECE_BINS as f64) as usize).min(ECE_BINS - 1);
        bins[bin].0 += 1;
        bins[bin].1 += confidence;
        bins[bin].2 += usize::from(hit);
        let encoded = char::from_digit(best as u32, 16).expect("a hex digit");
        predictions.push(encoded);
        if !decisions.is_empty() {
            decisions.push(',');
        }
        if is_covered {
            decisions.push(encoded);
        } else {
            decisions.push_str("Unknown");
        }
    }
    let n = rows.len();
    let ece = bins
        .iter()
        .filter(|bin| bin.0 > 0)
        .map(|&(count, confidence, hits)| {
            let count_f = count as f64;
            (count_f / n as f64) * (hits as f64 / count_f - confidence / count_f).abs()
        })
        .sum();
    Score {
        n,
        correct,
        nll: nll / n.max(1) as f64,
        ece,
        temperature,
        threshold,
        covered,
        covered_correct,
        predictions,
        decisions,
    }
}

pub fn score_calibrated(
    model: &PtrA0,
    arm: &Arm,
    split: &Split,
    payloads: &Payloads,
    temperature: f64,
    threshold: f64,
) -> Score {
    score_rows(
        &collect_logits(model, arm, split, payloads),
        temperature,
        threshold,
    )
}

/// Deterministically fit one scalar temperature on validation NLL.
pub fn fit_temperature(model: &PtrA0, arm: &Arm, split: &Split, payloads: &Payloads) -> f64 {
    let rows = collect_logits(model, arm, split, payloads);
    (25..=400)
        .map(|step| step as f64 / 100.0)
        .map(|temperature| {
            let score = score_rows(&rows, temperature, 0.0);
            (temperature, score.nll)
        })
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.total_cmp(&right.0))
        })
        .expect("validation is non-empty")
        .0
}

/// Threshold that retains at least 90% of validation examples.
pub fn fit_coverage_threshold(
    model: &PtrA0,
    arm: &Arm,
    split: &Split,
    payloads: &Payloads,
    temperature: f64,
) -> f64 {
    let rows = collect_logits(model, arm, split, payloads);
    let mut confidence: Vec<f64> = rows
        .iter()
        .map(|(_label, raw)| {
            let scaled: Vec<f64> = raw
                .iter()
                .map(|value| f64::from(*value) / temperature)
                .collect();
            let best = checked_argmax(raw);
            let max = scaled[best];
            1.0 / scaled.iter().map(|value| (value - max).exp()).sum::<f64>()
        })
        .collect();
    confidence.sort_by(|left, right| right.total_cmp(left));
    let retained = ((confidence.len() as f64) * 0.90).ceil() as usize;
    confidence[retained.saturating_sub(1).min(confidence.len() - 1)]
}

/// The final row and PRED line for every test split.
pub fn evaluate(trained: &Trained, arm: &Arm, data: &Dataset, payloads: &Payloads) -> Vec<String> {
    let model = trained.model.valid();
    let mut out = Vec::new();
    for name in TEST_SPLITS {
        let result = score(&model, arm, data.split(name), payloads);
        out.push(format!(
            r#"{{"row":"final","arm":"{}","split":"{name}","n":{},"correct":{},"accuracy":{},"nll":{},"ece15":{}}}"#,
            arm.name,
            result.n,
            result.correct,
            result.accuracy(),
            result.nll,
            result.ece
        ));
        out.push(format!("PRED {} {name} {}", arm.name, result.predictions));
    }
    out
}

/// Calibrated M002-v5 rows. Temperature and the abstention threshold are fit on
/// validation only, then held fixed for IID and compositional OOD evaluation.
pub fn evaluate_v5(
    trained: &Trained,
    arm: &Arm,
    data: &Dataset,
    payloads: &Payloads,
) -> Vec<String> {
    let model = trained.model.valid();
    let validation = data.split("val");
    let temperature = fit_temperature(&model, arm, validation, payloads);
    let threshold = fit_coverage_threshold(&model, arm, validation, payloads, temperature);
    let latency_p95_ms = inference_latency_p95_ms(&model, arm, data.split("test_ood"), payloads);
    let mut out = vec![
        format!(
            r#"{{"row":"calibration","arm":"{}","temperature":{},"coverage_target":0.9,"confidence_threshold":{}}}"#,
            arm.name, temperature, threshold
        ),
        format!(
            r#"{{"row":"performance-v5","arm":"{}","latency_p95_ms":{}}}"#,
            arm.name, latency_p95_ms
        ),
    ];
    for name in ["test_iid", "test_ood"] {
        let result = score_calibrated(
            &model,
            arm,
            data.split(name),
            payloads,
            temperature,
            threshold,
        );
        out.push(format!(
            r#"{{"row":"final-v5","arm":"{}","split":"{name}","n":{},"correct":{},"accuracy":{},"nll":{},"ece15":{},"temperature":{},"confidence_threshold":{},"covered":{},"coverage":{},"covered_correct":{},"selective_error":{},"abstained":{}}}"#,
            arm.name,
            result.n,
            result.correct,
            result.accuracy(),
            result.nll,
            result.ece,
            result.temperature,
            result.threshold,
            result.covered,
            result.coverage(),
            result.covered_correct,
            result.selective_error(),
            result.n - result.covered,
        ));
        out.push(format!("PRED {} {name} {}", arm.name, result.predictions));
        out.push(format!("DECISION {} {name} {}", arm.name, result.decisions));
    }
    out
}

/// Fixed-shape forward latency for the M002-v5 matched-compute gate. A warmup is
/// discarded, then 20 identical batches are measured and the nearest-rank p95
/// is reported per example. The run record binds the hardware profile.
fn inference_latency_p95_ms(model: &PtrA0, arm: &Arm, split: &Split, payloads: &Payloads) -> f64 {
    let device = Device::flex();
    let chosen: Vec<&Example> = split.examples.iter().take(EVAL_BATCH).collect();
    assert!(!chosen.is_empty(), "latency split is non-empty");
    let run = || {
        let inputs = batch::build(&chosen, arm.batch, payloads, &device);
        model
            .forward(
                inputs.tokens,
                &inputs.roles,
                &inputs.values,
                inputs.metadata,
            )
            .router_logits
            .into_data()
    };
    let _ = run();
    let mut samples = Vec::with_capacity(20);
    for _ in 0..20 {
        let started = Instant::now();
        let _ = run();
        samples.push(started.elapsed().as_secs_f64() * 1000.0 / chosen.len() as f64);
    }
    samples.sort_by(f64::total_cmp);
    samples[((samples.len() as f64 * 0.95).ceil() as usize).saturating_sub(1)]
}

/// Final rows and prediction strings for every held-out split of the baseline.
pub fn evaluate_plain(trained: &PlainTrained, data: &Dataset, d_model: usize) -> Vec<String> {
    let model = trained.model.valid();
    TEST_SPLITS
        .iter()
        .flat_map(|name| {
            let result = score_plain(&model, data.split(name), d_model);
            [
                format!(
                    r#"{{"row":"final","arm":"plain-transformer","split":"{name}","n":{} ,"correct":{},"accuracy":{},"nll":{},"ece15":{}}}"#,
                    result.n,
                    result.correct,
                    result.accuracy(),
                    result.nll,
                    result.ece
                ),
                format!("PRED plain-transformer {name} {}", result.predictions),
            ]
        })
        .collect()
}

/// Refuse numerical failures before converting logits to valid operator codes.
/// A panic makes the process fail, including a failure after its final update.
fn checked_argmax(row: &[f32]) -> usize {
    assert_eq!(row.len(), OPERATORS, "one logit per operator");
    assert!(
        row.iter().all(|value| value.is_finite()),
        "non-finite router logit; refusing to emit predictions"
    );
    let mut best = 0;
    for k in 1..OPERATORS {
        if row[k] > row[best] {
            best = k;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_logits_never_become_operator_predictions() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for position in 0..OPERATORS {
                let mut row = [0.0; OPERATORS];
                row[position] = invalid;
                assert!(std::panic::catch_unwind(|| checked_argmax(&row)).is_err());
            }
        }
    }

    #[test]
    fn finite_argmax_keeps_the_lowest_code_on_ties() {
        let mut row = [0.0; OPERATORS];
        assert_eq!(checked_argmax(&row), 0);
        row[3] = 2.0;
        row[7] = 2.0;
        assert_eq!(checked_argmax(&row), 3);
        row[7] = 3.0;
        assert_eq!(checked_argmax(&row), 7);
    }
}
