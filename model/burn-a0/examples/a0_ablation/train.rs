//! Training one arm and scoring it. Everything written to stdout is a
//! deterministic function of the data, the arm, the seeds and the step count;
//! timing goes to stderr, so two runs can be compared byte for byte.

use crate::arms::Arm;
use crate::batch::{self, Payloads};
use crate::data::{Dataset, Example, Split, TEST_SPLITS};
use crate::rng::{permutation, splitmix64, SplitMix64};
use burn::{
    module::{AutodiffModule, Module, ModuleVisitor, Param},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    tensor::Gradients,
};
use ptr_burn_a0::PtrA0;
use std::time::Instant;

pub const BATCH: usize = 128;
pub const WARMUP: usize = 100;
pub const EVAL_BATCH: usize = 500;
const OPERATORS: usize = 11;
const ECE_BINS: usize = 15;
/// Parameters are counted as effective when any of the first this many
/// batches gives them a non-zero gradient.
const EFFECTIVE_WINDOW: usize = 10;

pub struct Schedule {
    pub steps: usize,
    pub peak: f64,
}

impl Schedule {
    /// Linear warm-up over the first 100 steps, then cosine decay to 0.1 x peak
    /// at the last step.
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

/// Sums |gradient| per parameter element over the first batches.
struct GradientTally<'a> {
    gradients: &'a Gradients,
    sums: &'a mut Vec<Option<Tensor<1>>>,
    position: usize,
}

impl ModuleVisitor for GradientTally<'_> {
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

fn effective(sums: &[Option<Tensor<1>>]) -> usize {
    sums.iter()
        .flatten()
        .map(|sum| {
            let nonzero: i64 = sum.clone().greater_elem(0.0).int().sum().into_scalar();
            nonzero as usize
        })
        .sum()
}

fn on(flag: bool) -> &'static str {
    if flag {
        "on"
    } else {
        "off"
    }
}

/// Train `arm` on the train split for `schedule.steps` steps. A validation row
/// is written every steps/10 steps when `validate` holds.
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
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
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
        let loss = loss_fn.forward(output.router_logits, batch::labels(&chosen, &device));
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
    rows.insert(
        0,
        format!(
            r#"{{"row":"meta","arm":"{}","typed_attention":"{}","typed_query":"{}","latent_steps":"{}","latent_nonlinearity":"{}","frozen_router":"{}","batch":"{}","lr":{},"steps":{},"total_params":{},"effective_params":{},"final_train_loss":{},"nan":{}}}"#,
            arm.name,
            on(arm.typed_attention),
            on(arm.typed_query),
            arm.latent_steps,
            on(arm.latent_nonlinearity),
            on(arm.frozen_router),
            arm.batch.name(),
            schedule.peak,
            schedule.steps,
            total_params,
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

/// Counts over one split.
pub struct Score {
    pub n: usize,
    pub correct: usize,
    pub nll: f64,
    pub ece: f64,
    /// One lowercase hex digit per example: the predicted operator code.
    pub predictions: String,
}

impl Score {
    pub fn accuracy(&self) -> f64 {
        self.correct as f64 / self.n.max(1) as f64
    }
}

/// Forward-only scoring in batches of 500. The prediction is the argmax of the
/// logits, an exact tie going to the lowest code.
pub fn score(model: &PtrA0, arm: &Arm, split: &Split, payloads: &Payloads) -> Score {
    let device = Device::flex();
    let mut correct = 0;
    let mut nll = 0.0_f64;
    let mut predictions = String::with_capacity(split.examples.len());
    let mut bins = [(0usize, 0.0_f64, 0usize); ECE_BINS];
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
            let mut best = 0;
            for k in 1..OPERATORS {
                if row[k] > row[best] {
                    best = k;
                }
            }
            let max = f64::from(row[best]);
            let normalizer: f64 = row.iter().map(|&v| (f64::from(v) - max).exp()).sum();
            let log_probability = f64::from(row[example.label]) - max - normalizer.ln();
            nll -= log_probability;
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
        predictions,
    }
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
