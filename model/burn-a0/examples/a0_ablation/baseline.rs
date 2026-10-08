//! The `baseline-v1` phases: the successor study M001-v5 / M002-v10 / M003-v2
//! against the pinned plain transformer.
//!
//! - `baseline-v1`: per fold, train the study's A0 arms and the plain
//!   transformer in one process on identical data, order, steps and schedule,
//!   each arm at its OWN learning rate from `--lr-file`, then score
//!   `test_iid` and `test_ood` (validation fits the temperature and thresholds).
//! - `baseline-v1-sweep`: validation-only learning-rate sweep of any arm,
//!   including plain, over [`GRID`] at one seed. The test splits are never read.
//! - `baseline-v1-select`: the fixed selection rule (best mean validation
//!   accuracy, tie lower mean validation NLL, tie smaller lr) over sweep rows,
//!   writing the `arm<TAB>lr` file `baseline-v1` consumes.
//!
//! # Row schema
//!
//! `final-baseline-v1`, one row per (fold, arm, split in `test_iid`, `test_ood`):
//! `study fold arm split seed n correct accuracy` then
//! - raw logits (T = 1), the quantities the gates use: `raw_nll raw_ece15`
//!   (ECE over 15 equal-width bins), `raw_confidence_threshold raw_coverage
//!   raw_selective_error` (threshold fit on validation at T = 1);
//! - temperature-fitted diagnostics (T fit on validation NLL only, grid
//!   0.25..=4.0): `fitted_temperature fitted_nll fitted_ece15`;
//! - the evaluate_v5 abstention contract at the fitted T: `confidence_threshold
//!   covered coverage covered_correct selective_error abstained`;
//! - cost: `total_params flops_per_example` (corrected for the arm's own
//!   `latent_steps`) `flops_per_example_legacy` (the historical formula, which
//!   assumes two refinement passes) `train_wall_ms ms_per_step
//!   latency_p95_ms` (per example, 500-example batches of `test_ood`);
//! - `latent_steps` (plain: 1, its single fixed GELU refinement pass, not a
//!   recurrence), `explicit_token_count` (total non-PAD tokens, PAD = 0, of the
//!   split's RAW input as stored in the dataset) and `explicit_token_mean`.
//!   These counts are identical across arms by construction. This study does
//!   NOT test token savings; the columns exist so the claim cannot be made by
//!   accident. Typed arms additionally zero the attribute tokens of non-live
//!   facts inside the model input (admission); that is not counted here.
//!
//! `meta-baseline-v1` (one per fold and arm) carries the arm's switches,
//! learning rate, `final_train_loss`, `effective_params` (A0 only) and `nan`.
//! `mask-share-baseline-v1` (M001-v5, when both `no-semantic-slots-masked` and
//! `no-semantic-slots` ran) carries the A0-v2 mask share: masked minus unmasked
//! accuracy, plus the fraction of the full-minus-unmasked gap when `full` ran and
//! the share of non-live fact slots in the split. `PRED` and `DECISION` lines
//! follow each split as in paired-v5, suffixed ` FOLD <fold>`.

use crate::arms::{Arm, Study, PLAIN_ARM};
use crate::batch::Payloads;
use crate::data::{Dataset, Split};
use crate::train::{self, Schedule, Seeds};
use burn::module::{AutodiffModule, Module};
use ptr_types::Validity;
use std::collections::{BTreeMap, BTreeSet};

pub const ORDER_SALT: u64 = 0x0B_A7C4;
pub const SWEEP_SALT: u64 = 0x5EE9;
pub const D_MODEL: usize = 48;
pub const STEPS: usize = 1500;
/// The learning-rate grid, fixed before any sweep ran.
pub const GRID: [f64; 4] = [0.0025, 0.005, 0.0125, 0.025];

/// Usage text for `--help` / `--phase help`.
pub const USAGE: &str = "\
a0_ablation baseline-v1 phases (all flags are `--flag value`)

  --phase baseline-v1 --study STUDY --seed N --data DIR --folds FOLDS
        --lr-file PATH --out PATH [--arms A,B,..] [--steps 1500] [--d-model 48]
        [--rank 8] [--bias-limit 1.0] [--metadata-dropout 0.0]
      Per fold: train the study's A0 arms and `plain`, same data order, then
      write final-baseline-v1 / meta-baseline-v1 rows (schema: baseline.rs docs).
      --lr-file lines `arm<TAB>lr`; a missing entry for a selected arm is an error.
  --phase baseline-v1-sweep --study STUDY --seed N --data DIR --folds FOLDS
        --out PATH [--arms A,B,..] [--steps 1500] [--d-model 48]
        [--rank 8] [--bias-limit 1.0] [--metadata-dropout 0.0]
      Validation-only sweep over lr in [0.0025, 0.005, 0.0125, 0.025]; init seed
      is seed ^ 0x5EE9. No test split is read.
  --phase baseline-v1-select --sweep PATH[,PATH..] --out LRFILE
      Best mean val accuracy, tie lower val NLL, tie smaller lr, per arm.

STUDY: m001-v5 | m002-v10 | m003-v2     (arms: see `Study::arm_names`, plus `plain`)
FOLDS: fold=FNV64[,fold=FNV64..]        (FNV64 lowercase hex; names unique)
--out files are created exclusively; an existing file is never overwritten.
";

/// Typed failure of the learning-rate file.
#[derive(Clone, Debug, PartialEq)]
pub enum RateError {
    Missing { arm: String },
    Invalid { arm: String, value: String },
    Malformed { line: String },
}

impl std::fmt::Display for RateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { arm } => write!(f, "the lr file names no rate for {arm}"),
            Self::Invalid { arm, value } => {
                write!(f, "lr for {arm} must be finite and positive, got {value:?}")
            }
            Self::Malformed { line } => write!(f, "lr file line is not `arm<TAB>lr`: {line:?}"),
        }
    }
}

/// Parse `arm<TAB>lr[<TAB>extra..]`; `#` lines, the `arm` header and blanks are
/// skipped, a repeated arm is a malformed file (no silent last-wins).
pub fn parse_rates(text: &str) -> Result<BTreeMap<String, f64>, RateError> {
    let mut rates = BTreeMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("arm\t") || line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split('\t');
        let arm = fields.next().unwrap_or_default().trim().to_owned();
        let value = fields.next().unwrap_or_default().trim();
        if arm.is_empty() || value.is_empty() {
            return Err(RateError::Malformed {
                line: line.to_owned(),
            });
        }
        let lr: f64 = value.parse().map_err(|_| RateError::Invalid {
            arm: arm.clone(),
            value: value.to_owned(),
        })?;
        if !lr.is_finite() || lr <= 0.0 {
            return Err(RateError::Invalid {
                arm,
                value: value.to_owned(),
            });
        }
        if rates.insert(arm.clone(), lr).is_some() {
            return Err(RateError::Malformed {
                line: format!("duplicate arm {arm}"),
            });
        }
    }
    Ok(rates)
}

/// The rate of `arm`; no fallback to another arm's rate exists.
pub fn rate_for(rates: &BTreeMap<String, f64>, arm: &str) -> Result<f64, RateError> {
    rates.get(arm).copied().ok_or_else(|| RateError::Missing {
        arm: arm.to_owned(),
    })
}

/// Parse the ordered `fold=fnv64` list (hex digest, unique non-empty names).
pub fn parse_folds(text: &str) -> Result<Vec<(String, u64)>, String> {
    let mut folds: Vec<(String, u64)> = Vec::new();
    for item in text.split(',') {
        let (fold, digest) = item
            .split_once('=')
            .ok_or_else(|| format!("--folds item {item:?} must be fold=fnv64"))?;
        if fold.is_empty() || folds.iter().any(|(seen, _)| seen == fold) {
            return Err(format!(
                "--folds contains an empty or duplicate fold {fold:?}"
            ));
        }
        let digest = u64::from_str_radix(digest, 16)
            .map_err(|_| format!("--folds digest for {fold} is not hexadecimal"))?;
        folds.push((fold.to_owned(), digest));
    }
    Ok(folds)
}

/// What a run trains: resolved arms, whether plain is included, shared knobs.
#[derive(Clone, Debug)]
pub struct Plan {
    pub study: Study,
    pub arms: Vec<Arm>,
    pub plain: bool,
    pub seed: u64,
    pub steps: usize,
    pub d_model: usize,
}

/// Knobs applied to every A0 arm of a plan, as paired-v5 applied them.
#[derive(Clone, Copy, Debug)]
pub struct Knobs {
    pub rank: usize,
    pub bias_limit: f32,
    pub metadata_dropout: f32,
}

impl Knobs {
    pub fn validate(self, d_model: usize) -> Result<Self, String> {
        if self.rank == 0 || self.rank > d_model {
            return Err("rank must be in 1..=d-model".to_owned());
        }
        if !self.bias_limit.is_finite() || self.bias_limit <= 0.0 {
            return Err("bias-limit must be positive".to_owned());
        }
        if !self.metadata_dropout.is_finite() || !(0.0..=1.0).contains(&self.metadata_dropout) {
            return Err("metadata-dropout must be in [0,1]".to_owned());
        }
        Ok(self)
    }
}

impl Plan {
    /// Resolve `--arms` (or the whole study table plus plain) and apply `knobs`.
    pub fn new(
        study: Study,
        selection: Option<&[&str]>,
        seed: u64,
        steps: usize,
        d_model: usize,
        knobs: Knobs,
    ) -> Result<Self, String> {
        let knobs = knobs.validate(d_model)?;
        if steps == 0 {
            return Err("--steps must be positive".to_owned());
        }
        let (mut arms, plain) = match selection {
            Some(names) => study.select(names).map_err(|e| e.to_string())?,
            None => (study.arms(), true),
        };
        for arm in &mut arms {
            arm.typed_attention_rank = knobs.rank;
            arm.typed_attention_limit = knobs.bias_limit;
            arm.metadata_dropout = knobs.metadata_dropout;
        }
        Ok(Self {
            study,
            arms,
            plain,
            seed,
            steps,
            d_model,
        })
    }

    /// Every selected arm name, plain last.
    pub fn names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.arms.iter().map(|arm| arm.name).collect();
        if self.plain {
            names.push(PLAIN_ARM);
        }
        names
    }

    /// The learning rate of every selected arm, or the first missing/invalid one.
    pub fn rates(
        &self,
        rates: &BTreeMap<String, f64>,
    ) -> Result<BTreeMap<&'static str, f64>, RateError> {
        self.names()
            .into_iter()
            .map(|name| rate_for(rates, name).map(|lr| (name, lr)))
            .collect()
    }
}

/// Pull `"key":value` out of one of this module's own flat JSON rows.
pub fn json_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\":");
    let start = line.find(&marker)? + marker.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    Some(rest[..end].trim_matches('"'))
}

fn finite_rows(rows: &[(usize, Vec<f32>)]) -> bool {
    rows.iter()
        .all(|(_, row)| row.iter().all(|v| v.is_finite()))
}

fn explicit_tokens(split: &Split) -> usize {
    split
        .examples
        .iter()
        .map(|example| example.tokens.iter().filter(|&&t| t != 0).count())
        .sum()
}

fn nonlive_share(split: &Split) -> f64 {
    let facts = split.examples.iter().flat_map(|e| e.facts.iter());
    let (total, nonlive) = facts.fold((0usize, 0usize), |(t, n), fact| {
        (t + 1, n + usize::from(fact.validity != Validity::Live))
    });
    nonlive as f64 / total.max(1) as f64
}

/// `(gold label, logits)` per example.
type LogitRows = Vec<(usize, Vec<f32>)>;

/// One trained model plus how to read its logits, hiding the model kind.
struct Fitted<'a> {
    name: &'static str,
    logits: Box<dyn Fn(&Split) -> LogitRows + 'a>,
    latency: Box<dyn Fn(&Split) -> f64 + 'a>,
    meta: String,
    params: usize,
    flops: u64,
    flops_adjusted: i64,
    latent_steps: usize,
    ms_per_step: f64,
    nan: bool,
}

/// Train one A0 arm at `lr` with the plan's schedule and the shared seeds.
fn train_a0<'a>(
    plan: &Plan,
    arm: &'a Arm,
    data: &Dataset,
    payloads: &'a Payloads,
    lr: f64,
) -> Fitted<'a> {
    let trained = train::train(
        arm,
        data,
        payloads,
        plan.d_model,
        &Schedule {
            steps: plan.steps,
            peak: lr,
        },
        &Seeds {
            init: plan.seed,
            order: plan.seed ^ ORDER_SALT,
        },
        false,
    );
    let params = trained.model.num_params();
    let meta = trained.rows.first().cloned().unwrap_or_default();
    let model = trained.model.valid();
    let (arm_l, arm_p) = (*arm, *arm);
    let model = std::rc::Rc::new(model);
    let model_l = model.clone();
    Fitted {
        name: arm.name,
        logits: Box::new(move |split| train::collect_logits(&model_l, &arm_l, split, payloads)),
        latency: Box::new(move |split| {
            train::inference_latency_p95_ms(&model, &arm_p, split, payloads)
        }),
        meta,
        params,
        flops: train::a0_flops_per_example(arm, plan.d_model),
        flops_adjusted: train::a0_flops_latent_adjusted(arm, plan.d_model),
        latent_steps: arm.latent_steps,
        ms_per_step: trained.milliseconds_per_step,
        nan: trained.nan,
    }
}

/// Train the plain transformer at `lr` with the plan's schedule and shared seeds.
fn train_plain_arm<'a>(plan: &Plan, data: &Dataset, lr: f64) -> Fitted<'a> {
    let d_model = plan.d_model;
    let trained = train::train_plain(
        data,
        d_model,
        &Schedule {
            steps: plan.steps,
            peak: lr,
        },
        &Seeds {
            init: plan.seed,
            order: plan.seed ^ ORDER_SALT,
        },
        false,
    );
    let params = trained.model.num_params();
    let meta = trained.rows.first().cloned().unwrap_or_default();
    let model = std::rc::Rc::new(trained.model.valid());
    let model_l = model.clone();
    let flops = train::plain_flops_per_example(d_model);
    Fitted {
        name: PLAIN_ARM,
        logits: Box::new(move |split| train::collect_logits_plain(&model_l, split, d_model)),
        latency: Box::new(move |split| {
            train::inference_latency_p95_ms_plain(&model, split, d_model)
        }),
        meta,
        params,
        flops,
        flops_adjusted: flops as i64,
        latent_steps: 1,
        ms_per_step: trained.milliseconds_per_step,
        nan: trained.nan,
    }
}

/// Run `plan` on one fold, passing every output line to `sink`. `rates` must
/// already hold every selected arm (see [`Plan::rates`]). Returns false when any
/// arm diverged (its rows are then marked `nan` and carry no scores).
pub fn run_fold(
    plan: &Plan,
    rates: &BTreeMap<&'static str, f64>,
    fold: &str,
    data: &Dataset,
    sink: &mut dyn FnMut(&str),
) -> Result<bool, String> {
    let payloads = Payloads::new(plan.d_model);
    let study = plan.study.id();
    let mut finite = true;
    let mut accuracy: BTreeMap<(&str, &str), f64> = BTreeMap::new();
    for name in plan.names() {
        let lr = *rates.get(name).ok_or_else(|| {
            RateError::Missing {
                arm: name.to_owned(),
            }
            .to_string()
        })?;
        let arm = plan.arms.iter().find(|arm| arm.name == name);
        let fitted = match arm {
            Some(arm) => train_a0(plan, arm, data, &payloads, lr),
            None => train_plain_arm(plan, data, lr),
        };
        eprintln!(
            "timing phase=baseline-v1 study={study} fold={fold} arm={name} ms_per_step={:.3}",
            fitted.ms_per_step
        );
        let loss = json_field(&fitted.meta, "final_train_loss").unwrap_or("-1");
        let effective = json_field(&fitted.meta, "effective_params").unwrap_or("0");
        let switches = match arm {
            Some(arm) => format!(
                r#""kind":"a0","typed_attention_mode":"{}","typed_attention_rank":{},"typed_attention_limit":{},"typed_query":{},"latent_nonlinearity":{},"router_mode":"{}","label_smoothing":{},"metadata_dropout":{},"consistency_weight":{},"batch":"{}","effective_params":{effective}"#,
                arm.typed_attention_mode.name(),
                arm.typed_attention_rank,
                arm.typed_attention_limit,
                arm.typed_query,
                arm.latent_nonlinearity,
                arm.router_mode.name(),
                arm.label_smoothing,
                arm.metadata_dropout,
                arm.consistency_weight,
                arm.batch.name(),
            ),
            None => {
                r#""kind":"plain","architecture":"token-to-six-latent-cross-attention""#.to_owned()
            }
        };
        let wall_ms = fitted.ms_per_step * plan.steps as f64;
        sink(&format!(
            r#"{{"row":"meta-baseline-v1","study":"{study}","fold":"{fold}","arm":"{name}","seed":{},"lr":{lr},"steps":{},"batch_size":{},"d_model":{},"latent_steps":{},{switches},"total_params":{},"flops_per_example":{},"flops_per_example_legacy":{},"train_wall_ms":{wall_ms},"ms_per_step":{},"final_train_loss":{loss},"nan":{}}}"#,
            plan.seed,
            plan.steps,
            train::BATCH,
            plan.d_model,
            fitted.latent_steps,
            fitted.params,
            fitted.flops_adjusted,
            fitted.flops,
            fitted.ms_per_step,
            u8::from(fitted.nan),
        ));
        let validation = (fitted.logits)(data.split("val"));
        if fitted.nan || !finite_rows(&validation) {
            finite = false;
            continue;
        }
        let fitted_t = train::temperature_from_rows(&validation);
        let fitted_thr = train::threshold_from_rows(&validation, fitted_t);
        let raw_thr = train::threshold_from_rows(&validation, 1.0);
        let latency = (fitted.latency)(data.split("test_ood"));
        for split_name in ["test_iid", "test_ood"] {
            let split = data.split(split_name);
            let rows = (fitted.logits)(split);
            if !finite_rows(&rows) {
                finite = false;
                continue;
            }
            let raw = train::score_rows(&rows, 1.0, raw_thr);
            let calibrated = train::score_rows(&rows, fitted_t, fitted_thr);
            let tokens = explicit_tokens(split);
            sink(&format!(
                r#"{{"row":"final-baseline-v1","study":"{study}","fold":"{fold}","arm":"{name}","split":"{split_name}","seed":{},"n":{},"correct":{},"accuracy":{},"raw_nll":{},"raw_ece15":{},"raw_confidence_threshold":{raw_thr},"raw_coverage":{},"raw_selective_error":{},"fitted_temperature":{fitted_t},"fitted_nll":{},"fitted_ece15":{},"confidence_threshold":{fitted_thr},"covered":{},"coverage":{},"covered_correct":{},"selective_error":{},"abstained":{},"total_params":{},"flops_per_example":{},"flops_per_example_legacy":{},"train_wall_ms":{wall_ms},"ms_per_step":{},"latency_p95_ms":{latency},"latent_steps":{},"explicit_token_count":{tokens},"explicit_token_mean":{}}}"#,
                plan.seed,
                raw.n,
                raw.correct,
                raw.accuracy(),
                raw.nll,
                raw.ece,
                raw.coverage(),
                raw.selective_error(),
                calibrated.nll,
                calibrated.ece,
                calibrated.covered,
                calibrated.coverage(),
                calibrated.covered_correct,
                calibrated.selective_error(),
                calibrated.n - calibrated.covered,
                fitted.params,
                fitted.flops_adjusted,
                fitted.flops,
                fitted.ms_per_step,
                fitted.latent_steps,
                tokens as f64 / raw.n.max(1) as f64,
            ));
            sink(&format!(
                "PRED {name} {split_name} {} FOLD {fold}",
                calibrated.predictions
            ));
            sink(&format!(
                "DECISION {name} {split_name} {} FOLD {fold}",
                calibrated.decisions
            ));
            accuracy.insert((fitted.name, split_name), raw.accuracy());
        }
    }
    for split_name in ["test_iid", "test_ood"] {
        let masked = accuracy.get(&("no-semantic-slots-masked", split_name));
        let unmasked = accuracy.get(&("no-semantic-slots", split_name));
        if let (Some(masked), Some(unmasked)) = (masked, unmasked) {
            let gap = match accuracy.get(&("full", split_name)) {
                Some(full) if full - unmasked > 0.0 => {
                    format!("{}", (masked - unmasked) / (full - unmasked))
                }
                _ => "null".to_owned(),
            };
            sink(&format!(
                r#"{{"row":"mask-share-baseline-v1","study":"{study}","fold":"{fold}","split":"{split_name}","masked_accuracy":{masked},"unmasked_accuracy":{unmasked},"mask_share":{},"mask_share_of_full_gap":{gap},"nonlive_slot_share":{}}}"#,
                masked - unmasked,
                nonlive_share(data.split(split_name)),
            ));
        }
    }
    Ok(finite)
}

/// Validation-only sweep of `plan`'s arms over [`GRID`] on one fold. Only the
/// `train` and `val` splits are read; a dataset without test splits is enough.
pub fn sweep_fold(
    plan: &Plan,
    fold: &str,
    data: &Dataset,
    sink: &mut dyn FnMut(&str),
) -> Result<bool, String> {
    let payloads = Payloads::new(plan.d_model);
    let study = plan.study.id();
    let init = plan.seed ^ SWEEP_SALT;
    let mut finite = true;
    for name in plan.names() {
        let arm = plan.arms.iter().find(|arm| arm.name == name);
        for lr in GRID {
            let schedule = Schedule {
                steps: plan.steps,
                peak: lr,
            };
            let seeds = Seeds {
                init,
                order: init ^ ORDER_SALT,
            };
            let (nan, ms_per_step, rows) = match arm {
                Some(arm) => {
                    let t =
                        train::train(arm, data, &payloads, plan.d_model, &schedule, &seeds, false);
                    let model = t.model.valid();
                    let rows = (!t.nan)
                        .then(|| train::collect_logits(&model, arm, data.split("val"), &payloads));
                    (t.nan, t.milliseconds_per_step, rows)
                }
                None => {
                    let t = train::train_plain(data, plan.d_model, &schedule, &seeds, false);
                    let model = t.model.valid();
                    let rows = (!t.nan).then(|| {
                        train::collect_logits_plain(&model, data.split("val"), plan.d_model)
                    });
                    (t.nan, t.milliseconds_per_step, rows)
                }
            };
            let scored = rows.filter(|rows| finite_rows(rows));
            let diverged = nan || scored.is_none();
            finite &= !diverged;
            let (accuracy, nll, ece) = match &scored {
                Some(rows) if !diverged => {
                    let s = train::score_rows(rows, 1.0, 0.0);
                    (s.accuracy(), s.nll, s.ece)
                }
                _ => (0.0, 0.0, 0.0),
            };
            eprintln!(
                "timing phase=baseline-v1-sweep study={study} fold={fold} arm={name} lr={lr} ms_per_step={ms_per_step:.3}"
            );
            sink(&format!(
                r#"{{"row":"sweep-baseline-v1","study":"{study}","fold":"{fold}","arm":"{name}","seed":{},"lr":{lr},"steps":{},"val_accuracy":{accuracy},"val_nll":{nll},"val_ece15":{ece},"ms_per_step":{ms_per_step},"nan":{}}}"#,
                plan.seed,
                plan.steps,
                u8::from(diverged),
            ));
        }
    }
    Ok(finite)
}

/// One grid point of one arm on one fold, as read back from a sweep row.
#[derive(Clone, Debug, PartialEq)]
pub struct SweepPoint {
    pub arm: String,
    pub fold: String,
    pub lr: f64,
    pub val_accuracy: f64,
    pub val_nll: f64,
    pub diverged: bool,
}

/// A candidate learning rate with its fold-mean validation metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub lr: f64,
    pub val_accuracy: f64,
    pub val_nll: f64,
}

/// The selected learning rate of one arm.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub arm: String,
    pub best: Candidate,
}

/// Typed failure of sweep-row reading and selection.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectError {
    BadRow(String),
    NoRows,
    MixedRuns,
    IncompleteGrid { arm: String },
    OffGrid { arm: String, lr: f64 },
    DuplicatePoint { arm: String, fold: String, lr: f64 },
    FoldMismatch { arm: String },
    AllDiverged { arm: String },
}

impl std::fmt::Display for SelectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadRow(row) => write!(f, "unreadable sweep row: {row}"),
            Self::NoRows => write!(f, "no sweep rows"),
            Self::MixedRuns => write!(f, "sweep rows mix studies or seeds"),
            Self::IncompleteGrid { arm } => {
                write!(f, "{arm}: the sweep does not cover the whole lr grid")
            }
            Self::OffGrid { arm, lr } => write!(f, "{arm}: lr {lr} is not on the grid"),
            Self::DuplicatePoint { arm, fold, lr } => {
                write!(f, "{arm}: duplicate sweep row for fold {fold}, lr {lr}")
            }
            Self::FoldMismatch { arm } => {
                write!(f, "{arm}: the grid points were not run on the same folds")
            }
            Self::AllDiverged { arm } => write!(f, "{arm}: every grid point diverged"),
        }
    }
}

/// Read `sweep-baseline-v1` rows (other lines are skipped); the study and seed
/// must be unique over the input.
pub fn parse_sweep(text: &str) -> Result<(String, u64, Vec<SweepPoint>), SelectError> {
    let mut run: Option<(String, u64)> = None;
    let mut points = Vec::new();
    for line in text
        .lines()
        .filter(|l| l.contains(r#""row":"sweep-baseline-v1""#))
    {
        let bad = || SelectError::BadRow(line.to_owned());
        let get = |key: &str| json_field(line, key).ok_or_else(bad);
        let study = get("study")?.to_owned();
        let seed: u64 = get("seed")?.parse().map_err(|_| bad())?;
        match &run {
            None => run = Some((study, seed)),
            Some(first) if *first != (study.clone(), seed) => return Err(SelectError::MixedRuns),
            Some(_) => {}
        }
        let number =
            |key: &str| -> Result<f64, SelectError> { get(key)?.parse::<f64>().map_err(|_| bad()) };
        points.push(SweepPoint {
            arm: get("arm")?.to_owned(),
            fold: get("fold")?.to_owned(),
            lr: number("lr")?,
            val_accuracy: number("val_accuracy")?,
            val_nll: number("val_nll")?,
            diverged: get("nan")? != "0",
        });
    }
    let (study, seed) = run.ok_or(SelectError::NoRows)?;
    Ok((study, seed, points))
}

/// The fixed rule: higher accuracy wins, then lower NLL, then smaller lr.
pub fn select_best(candidates: &[Candidate]) -> Option<Candidate> {
    candidates.iter().copied().reduce(|best, next| {
        let order = next
            .val_accuracy
            .total_cmp(&best.val_accuracy)
            .then_with(|| best.val_nll.total_cmp(&next.val_nll))
            .then_with(|| best.lr.total_cmp(&next.lr));
        if order.is_gt() {
            next
        } else {
            best
        }
    })
}

/// Apply [`select_best`] per arm over fold means. Every arm must cover the whole
/// [`GRID`] on one common fold set; a grid point that diverged on any fold is
/// ineligible.
pub fn select_rates(points: &[SweepPoint]) -> Result<Vec<Selection>, SelectError> {
    let mut by_arm: BTreeMap<&str, Vec<&SweepPoint>> = BTreeMap::new();
    for point in points {
        by_arm.entry(&point.arm).or_default().push(point);
    }
    if by_arm.is_empty() {
        return Err(SelectError::NoRows);
    }
    let mut out = Vec::new();
    for (arm, arm_points) in by_arm {
        let mut seen: BTreeSet<(&str, u64)> = BTreeSet::new();
        let mut folds_of: BTreeMap<u64, BTreeSet<&str>> = BTreeMap::new();
        for point in &arm_points {
            if !GRID.contains(&point.lr) {
                return Err(SelectError::OffGrid {
                    arm: arm.to_owned(),
                    lr: point.lr,
                });
            }
            if !seen.insert((&point.fold, point.lr.to_bits())) {
                return Err(SelectError::DuplicatePoint {
                    arm: arm.to_owned(),
                    fold: point.fold.clone(),
                    lr: point.lr,
                });
            }
            folds_of
                .entry(point.lr.to_bits())
                .or_default()
                .insert(&point.fold);
        }
        if GRID.iter().any(|g| !folds_of.contains_key(&g.to_bits())) {
            return Err(SelectError::IncompleteGrid {
                arm: arm.to_owned(),
            });
        }
        let first = folds_of.values().next().expect("grid is non-empty");
        if folds_of.values().any(|folds| folds != first) {
            return Err(SelectError::FoldMismatch {
                arm: arm.to_owned(),
            });
        }
        let mut candidates = Vec::new();
        for lr in GRID {
            let at: Vec<&&SweepPoint> = arm_points.iter().filter(|p| p.lr == lr).collect();
            if at.iter().any(|p| p.diverged) {
                continue;
            }
            let n = at.len() as f64;
            candidates.push(Candidate {
                lr,
                val_accuracy: at.iter().map(|p| p.val_accuracy).sum::<f64>() / n,
                val_nll: at.iter().map(|p| p.val_nll).sum::<f64>() / n,
            });
        }
        let best = select_best(&candidates).ok_or_else(|| SelectError::AllDiverged {
            arm: arm.to_owned(),
        })?;
        out.push(Selection {
            arm: arm.to_owned(),
            best,
        });
    }
    Ok(out)
}

/// The `arm<TAB>lr` file (extra audit columns after lr are ignored by readers).
pub fn render_lr_file(study: &str, seed: u64, selections: &[Selection]) -> String {
    let mut text = format!(
        "# baseline-v1-select: study={study} seed={seed}; best mean val accuracy, tie lower mean val NLL, tie smaller lr\narm\tlr\tval_accuracy\tval_nll\n"
    );
    for s in selections {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            s.arm, s.best.lr, s.best.val_accuracy, s.best.val_nll
        ));
    }
    text
}
