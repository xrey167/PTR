//! The A0 mechanism ablation study (research/falsification/A0-ablations-v1).
//!
//! Phases:
//! - `list-arms`: the arm table as JSON;
//! - `self-test`: T7 (arm order and repetition do not change an arm's result) and
//!   T8 (this binary's rule on hand-checked cases);
//! - `calibrate`: the full arm only, train and val only, any seed (the study uses
//!   0, which is not a declared seed);
//! - `sweep`: one learning rate for every arm of an experiment, val only, with the
//!   salted seeds init = seed ^ 0x5EE9 and order = init ^ 0x0BA7C4;
//! - `eval`: listed arms of one experiment, each at its selected learning rate,
//!   with validation checkpoints, then every test split, with seeds init = seed and
//!   order = seed ^ 0x0BA7C4.
//! - `paired`: one selected A0 arm and the matched plain baseline at the same
//!   seed, step count and learning rate, for the frozen M001/M002 comparison.
//!
//! Stdout carries one JSON object per line (string fields name the row, numeric
//! fields are its metrics) and `PRED <arm> <split> <hex digit per example>` lines.
//! Timing goes to stderr only, so stdout is comparable byte for byte.

mod arms;
mod batch;
mod data;
mod rng;
mod rule;
mod train;

use arms::Arm;
use batch::Payloads;
use data::Dataset;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use train::{Schedule, Seeds};

const ORDER_SALT: u64 = 0x0B_A7C4;
const SWEEP_SALT: u64 = 0x5EE9;
const DEFAULT_D_MODEL: usize = 32;

struct Args {
    values: BTreeMap<String, String>,
}

impl Args {
    /// Read `--flag value` pairs, rejecting positional, duplicate, or valueless flags.
    fn parse() -> Result<Self, String> {
        let mut values = BTreeMap::new();
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let key = flag
                .strip_prefix("--")
                .ok_or_else(|| format!("expected --flag, got {flag:?}"))?;
            let value = args
                .next()
                .ok_or_else(|| format!("--{key} needs a value"))?;
            if values.insert(key.to_owned(), value).is_some() {
                return Err(format!("--{key} given twice"));
            }
        }
        Ok(Self { values })
    }

    /// Return a required option's value or an error naming the missing flag.
    fn get(&self, key: &str) -> Result<&str, String> {
        self.values
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("--{key} is required"))
    }

    /// Parse a required option, returning an error if it is absent or invalid.
    fn number<T: std::str::FromStr>(&self, key: &str) -> Result<T, String> {
        self.get(key)?
            .parse()
            .map_err(|_| format!("--{key} is not a number"))
    }

    /// Use the default only for an absent option; invalid supplied values are errors.
    fn number_or<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.values.get(key) {
            Some(_) => self.number(key),
            None => Ok(default),
        }
    }
}

/// Write and flush one stdout line, panicking on write or flush failure.
fn emit(line: &str) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{line}").expect("stdout is writable");
    out.flush().expect("stdout flushes");
}

/// Load `--data` against hexadecimal `--data-fnv64` and emit its identity row.
/// Returns argument or dataset-loading errors; output failures panic.
fn load(args: &Args) -> Result<Dataset, String> {
    let fnv = u64::from_str_radix(args.get("data-fnv64")?, 16)
        .map_err(|_| "--data-fnv64 is not hex".to_owned())?;
    let dataset = data::load(&PathBuf::from(args.get("data")?), fnv)?;
    let mut row = format!(r#"{{"row":"data","data_fnv64":"{:016x}""#, dataset.fnv64);
    for split in &dataset.splits {
        row.push_str(&format!(
            r#","label_fnv64_{}":"{:016x}""#,
            split.name, split.label_fnv64
        ));
    }
    row.push('}');
    emit(&row);
    Ok(dataset)
}

/// Parse the ordered `fold=digest` list used by M002-v5.
fn v2_folds(args: &Args) -> Result<Vec<(String, u64)>, String> {
    let mut folds = Vec::new();
    for item in args.get("folds")?.split(',') {
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
    if folds.is_empty() {
        return Err("--folds must contain at least one fold".to_owned());
    }
    Ok(folds)
}

/// Load one operator-routing v2 fold and emit its bound identity.
fn load_v2(args: &Args, fold: &str, fnv: u64) -> Result<Dataset, String> {
    let directory = PathBuf::from(args.get("data")?).join(fold);
    let dataset = data::load_v2_fold(&directory, fnv)?;
    let mut row = format!(
        r#"{{"row":"data-v2","fold":"{fold}","data_fnv64":"{:016x}","hard_validity_violations":0"#,
        dataset.fnv64
    );
    for split in &dataset.splits {
        row.push_str(&format!(
            r#", "label_fnv64_{}":"{:016x}""#,
            split.name, split.label_fnv64
        ));
    }
    row.push('}');
    emit(&row);
    Ok(dataset)
}

/// Attach the fold identity to every JSON evidence row emitted by a training
/// run. Prediction strings remain human/debug evidence and receive an explicit
/// trailing fold label instead.
fn emit_fold(line: &str, fold: &str) {
    if line.starts_with('{') && line.ends_with('}') {
        emit(&format!(
            r#"{},"fold":"{}"}}"#,
            &line[..line.len() - 1],
            fold
        ));
    } else {
        emit(&format!("{line} FOLD {fold}"));
    }
}

/// Arms named by `--arms`, every one of which must belong to `--experiment`.
/// If `--arms` is absent, selects all arms for the experiment. Returns an error
/// for a missing experiment, unknown arm, mismatched experiment, or empty selection.
fn arms_of(args: &Args) -> Result<Vec<Arm>, String> {
    let experiment = args.get("experiment")?;
    let arms: Vec<Arm> = match args.values.get("arms") {
        Some(list) => list.split(',').map(arms::find).collect::<Result<_, _>>()?,
        None => arms::ARMS
            .iter()
            .copied()
            .filter(|arm| arm.experiment == experiment)
            .collect(),
    };
    for arm in &arms {
        if arm.experiment != experiment {
            return Err(format!(
                "arm {} belongs to {}, not {experiment}",
                arm.name, arm.experiment
            ));
        }
    }
    if arms.is_empty() {
        return Err(format!("no arms for {experiment}"));
    }
    Ok(arms)
}

/// `arm<TAB>lr` lines; lines starting with `#` or the header `arm` are skipped.
/// Blank lines are skipped and later entries replace earlier rates for an arm.
/// Returns an error for file I/O or a missing or unparseable rate; parsed rates
/// are not checked for positivity or finiteness.
fn learning_rates(path: &str) -> Result<BTreeMap<String, f64>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let mut rates = BTreeMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("arm\t") || line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split('\t');
        let arm = fields.next().unwrap_or_default().to_owned();
        let lr: f64 = fields
            .next()
            .and_then(|lr| lr.parse().ok())
            .ok_or_else(|| format!("{path}: no learning rate for {arm:?}"))?;
        rates.insert(arm, lr);
    }
    Ok(rates)
}

/// Report the arm's elapsed milliseconds per training step to stderr.
fn timing(phase: &str, arm: &Arm, trained: &train::Trained) {
    eprintln!(
        "timing phase={phase} arm={} ms_per_step={:.3}",
        arm.name, trained.milliseconds_per_step
    );
}

/// Train the full arm and emit metadata and validation rows, without test scoring.
/// Returns false for any non-finite training loss, or an argument or data error.
fn calibrate(args: &Args) -> Result<bool, String> {
    let data = load(args)?;
    let d_model = args.number_or("d-model", DEFAULT_D_MODEL)?;
    let payloads = Payloads::new(d_model);
    let seed: u64 = args.number("seed")?;
    let arm = arms::find("full")?;
    let schedule = Schedule {
        steps: args.number("steps")?,
        peak: args.number("lr")?,
    };
    let trained = train::train(
        &arm,
        &data,
        &payloads,
        d_model,
        &schedule,
        &Seeds {
            init: seed,
            order: seed ^ ORDER_SALT,
        },
        true,
    );
    timing("calibrate", &arm, &trained);
    trained.rows.iter().for_each(|row| emit(row));
    Ok(!trained.nan)
}

/// Train selected arms at one rate with salted seeds and emit validation scores.
/// Returns argument, arm-selection, or data errors. Non-finite training losses
/// appear in the rows; the phase still returns true when all arms finish.
fn sweep(args: &Args) -> Result<bool, String> {
    let data = load(args)?;
    let d_model = args.number_or("d-model", DEFAULT_D_MODEL)?;
    let payloads = Payloads::new(d_model);
    let init = args.number::<u64>("seed")? ^ SWEEP_SALT;
    let lr: f64 = args.number("lr")?;
    let steps: usize = args.number("steps")?;
    for arm in arms_of(args)? {
        let trained = train::train(
            &arm,
            &data,
            &payloads,
            d_model,
            &Schedule { steps, peak: lr },
            &Seeds {
                init,
                order: init ^ ORDER_SALT,
            },
            false,
        );
        timing("sweep", &arm, &trained);
        trained.rows.iter().for_each(|row| emit(row));
        let score = train::score(
            &burn::module::AutodiffModule::valid(&trained.model),
            &arm,
            data.split("val"),
            &payloads,
        );
        emit(&format!(
            r#"{{"row":"sweep","arm":"{}","lr":{lr},"val_accuracy":{},"nan":{}}}"#,
            arm.name,
            score.accuracy(),
            u8::from(trained.nan)
        ));
    }
    Ok(true)
}

/// Train selected arms at their file-provided rates and emit validation and test results.
/// Returns false if any training loss is non-finite. Argument, data, arm-selection,
/// and rate-file errors propagate, including a missing rate for a selected arm.
fn eval(args: &Args) -> Result<bool, String> {
    let data = load(args)?;
    let d_model = args.number_or("d-model", DEFAULT_D_MODEL)?;
    let payloads = Payloads::new(d_model);
    let seed: u64 = args.number("seed")?;
    let steps: usize = args.number("steps")?;
    let rates = learning_rates(args.get("lr-file")?)?;
    let mut finite = true;
    for arm in arms_of(args)? {
        let lr = *rates
            .get(arm.name)
            .ok_or_else(|| format!("the lr file names no rate for {}", arm.name))?;
        let trained = train::train(
            &arm,
            &data,
            &payloads,
            d_model,
            &Schedule { steps, peak: lr },
            &Seeds {
                init: seed,
                order: seed ^ ORDER_SALT,
            },
            true,
        );
        timing("eval", &arm, &trained);
        finite &= !trained.nan;
        trained.rows.iter().for_each(|row| emit(row));
        train::evaluate(&trained, &arm, &data, &payloads)
            .iter()
            .for_each(|line| emit(line));
    }
    Ok(finite)
}

/// Run the matched token-only baseline. This shares the frozen data loader,
/// schedule, batch order and output schema with the A0 arms, but its model has
/// no typed input path and no PTR-specific recurrence/router.
fn plain(args: &Args) -> Result<bool, String> {
    let data = load(args)?;
    let d_model = args.number_or("d-model", DEFAULT_D_MODEL)?;
    let seed: u64 = args.number("seed")?;
    let schedule = Schedule {
        steps: args.number("steps")?,
        peak: args.number("lr")?,
    };
    let trained = train::train_plain(
        &data,
        d_model,
        &schedule,
        &Seeds {
            init: seed,
            order: seed ^ ORDER_SALT,
        },
        true,
    );
    eprintln!(
        "timing phase=plain arm=plain-transformer ms_per_step={:.3}",
        trained.milliseconds_per_step
    );
    trained.rows.iter().for_each(|row| emit(row));
    train::evaluate_plain(&trained, &data, d_model)
        .iter()
        .for_each(|line| emit(line));
    Ok(!trained.nan)
}

/// Run one frozen A0 arm beside the matched plain baseline. The command keeps
/// both arms in one process so the runner records one atomic paired outcome.
fn paired(args: &Args) -> Result<bool, String> {
    let data = load(args)?;
    let d_model = args.number_or("d-model", DEFAULT_D_MODEL)?;
    let payloads = Payloads::new(d_model);
    let seed: u64 = args.number("seed")?;
    let steps: usize = args.number("steps")?;
    let rates = learning_rates(args.get("lr-file")?)?;
    let arm = arms_of(args)?
        .into_iter()
        .next()
        .ok_or("paired requires one A0 arm")?;
    let lr = *rates
        .get(arm.name)
        .ok_or_else(|| format!("the lr file names no rate for {}", arm.name))?;
    let schedule = Schedule { steps, peak: lr };
    let trained = train::train(
        &arm,
        &data,
        &payloads,
        d_model,
        &schedule,
        &Seeds {
            init: seed,
            order: seed ^ ORDER_SALT,
        },
        true,
    );
    timing("paired", &arm, &trained);
    let mut finite = !trained.nan;
    trained.rows.iter().for_each(|row| emit(row));
    train::evaluate(&trained, &arm, &data, &payloads)
        .iter()
        .for_each(|line| emit(line));

    let plain = train::train_plain(
        &data,
        d_model,
        &schedule,
        &Seeds {
            init: seed,
            order: seed ^ ORDER_SALT,
        },
        true,
    );
    eprintln!(
        "timing phase=paired arm=plain-transformer ms_per_step={:.3}",
        plain.milliseconds_per_step
    );
    finite &= !plain.nan;
    plain.rows.iter().for_each(|row| emit(row));
    train::evaluate_plain(&plain, &data, d_model)
        .iter()
        .for_each(|line| emit(line));
    Ok(finite)
}

/// Run both M002-v5 arms and every requested fold atomically for one seed.
fn paired_v5(args: &Args) -> Result<bool, String> {
    let d_model = args.number_or("d-model", DEFAULT_D_MODEL)?;
    let payloads = Payloads::new(d_model);
    let seed: u64 = args.number("seed")?;
    let schedule = Schedule {
        steps: args.number("steps")?,
        peak: args.number("lr")?,
    };
    let rank = args.number_or("rank", 16usize)?;
    let bias_limit = args.number_or("bias-limit", 2.0f32)?;
    let metadata_dropout = args.number_or("metadata-dropout", 0.10f32)?;
    let mut arms = arms_of(args)?;
    let names: Vec<&str> = arms.iter().map(|arm| arm.name).collect();
    let experiment = args.get("experiment")?;
    let expected_names = arms::paired_names(experiment)
        .ok_or_else(|| format!("paired-v5 has no registered matched pair for {experiment}"))?;
    if names.as_slice() != expected_names {
        return Err(format!(
            "paired-v5 requires arms {} in that order, got {}",
            expected_names.join(","),
            names.join(",")
        ));
    }
    if rank == 0 || rank > d_model || !bias_limit.is_finite() || bias_limit <= 0.0 {
        return Err("rank must be in 1..=d-model and bias-limit must be positive".to_owned());
    }
    if !metadata_dropout.is_finite() || !(0.0..=1.0).contains(&metadata_dropout) {
        return Err("metadata-dropout must be in [0,1]".to_owned());
    }
    for arm in &mut arms {
        arm.typed_attention_rank = rank;
        arm.typed_attention_limit = bias_limit;
        arm.metadata_dropout = metadata_dropout;
    }
    emit(&format!(
        r#"{{"row":"run-v5","seed":{seed},"d_model":{d_model},"batch_size":{},"steps":{},"lr":{},"rank":{rank},"bias_limit":{bias_limit},"metadata_dropout":{metadata_dropout}}}"#,
        train::BATCH,
        schedule.steps,
        schedule.peak,
    ));
    let mut finite = true;
    for (fold, digest) in v2_folds(args)? {
        let data = load_v2(args, &fold, digest)?;
        for arm in &arms {
            let trained = train::train(
                arm,
                &data,
                &payloads,
                d_model,
                &schedule,
                &Seeds {
                    init: seed,
                    order: seed ^ ORDER_SALT,
                },
                true,
            );
            timing("paired-v5", arm, &trained);
            finite &= !trained.nan;
            trained.rows.iter().for_each(|row| emit_fold(row, &fold));
            train::evaluate_v5(&trained, arm, &data, &payloads)
                .iter()
                .for_each(|line| emit_fold(line, &fold));
        }
    }
    Ok(finite)
}

#[cfg(test)]
mod paired_arm_tests {
    use super::*;

    fn arguments(experiment: &str, arms: &str) -> Args {
        Args {
            values: BTreeMap::from([
                ("experiment".to_owned(), experiment.to_owned()),
                ("arms".to_owned(), arms.to_owned()),
            ]),
        }
    }

    #[test]
    fn v7_has_its_own_explicit_matched_pair() {
        let selected = arms_of(&arguments(
            "M002-v7",
            "factorized-v2-v7,factorized-v2-off-v7",
        ))
        .expect("v7 pair selects before dataset loading");
        assert_eq!(
            selected.iter().map(|arm| arm.name).collect::<Vec<_>>(),
            arms::paired_names("M002-v7").expect("v7 is registered")
        );
        assert!(selected.iter().all(|arm| arm.experiment == "M002-v7"));
        assert_eq!(
            selected[0].typed_attention_mode,
            ptr_burn_a0::TypedAttentionMode::FactorizedV2
        );
        assert_eq!(
            selected[1].typed_attention_mode,
            ptr_burn_a0::TypedAttentionMode::Off
        );
    }

    #[test]
    fn v7_cannot_borrow_v5_or_v6_identity() {
        let v5_names = arms_of(&arguments("M002-v7", "factorized-v2,factorized-v2-off"))
            .expect_err("v7 must not select v5 arms");
        assert!(v5_names.contains("belongs to M002-v5"));
        let unsupported = arms::paired_names("M002-v6");
        assert!(unsupported.is_none(), "superseded v6 has no runner pair");
    }

    #[test]
    fn v8_has_its_own_explicit_matched_pair() {
        let selected = arms_of(&arguments(
            "M002-v8",
            "factorized-v2-v8,factorized-v2-off-v8",
        ))
        .expect("v8 pair selects before dataset loading");
        assert_eq!(
            selected.iter().map(|arm| arm.name).collect::<Vec<_>>(),
            arms::paired_names("M002-v8").expect("v8 is registered")
        );
        assert!(selected.iter().all(|arm| arm.experiment == "M002-v8"));
        assert_eq!(
            selected[0].typed_attention_mode,
            ptr_burn_a0::TypedAttentionMode::FactorizedV2
        );
        assert_eq!(
            selected[1].typed_attention_mode,
            ptr_burn_a0::TypedAttentionMode::Off
        );
        let cross_study = arms_of(&arguments(
            "M002-v8",
            "factorized-v2-v7,factorized-v2-off-v7",
        ))
        .expect_err("v8 must not select v7 arms");
        assert!(cross_study.contains("belongs to M002-v7"));
    }
}

/// T8: the rule on cases whose labels were worked out by hand (the working is in
/// each case's comment), independent of the generator.
fn rule_cases() -> Result<(), String> {
    use ptr_types::{
        EpistemicState as E, ReasoningOperator as O, SemanticRole as R, Validity as V,
    };
    use rule::RuleFact;
    let fact = |role, epistemic, bucket, validity| RuleFact {
        role,
        epistemic,
        bucket,
        validity,
    };
    let code = |op| usize::from(ptr_types::Codebook::V1.code_of(op).expect("v1 op").index());
    let unknown_bonus_winner = O::Probabilistic;
    let cases: Vec<(&str, Vec<RuleFact>, usize, usize, O)> = vec![
        // Goal/Verified, cb 4: Search 1.25*1.55*2 = 3.875, Optimization 1.74375.
        (
            "one goal",
            vec![fact(R::Goal, E::Verified, 4, V::Live)],
            0,
            2,
            O::Search,
        ),
        // Budget 0.4 costs Search 0.2 and Optimization 0.4: 3.675 still leads.
        (
            "tight budget",
            vec![fact(R::Goal, E::Verified, 4, V::Live)],
            0,
            0,
            O::Search,
        ),
        // A cb-0 fact adds nothing, not even a penalty: all z = 0, and the tie
        // goes to the cheapest operator, Semantic (0.2).
        (
            "all zero",
            vec![fact(R::Goal, E::Verified, 0, V::Live)],
            0,
            2,
            O::Semantic,
        ),
        // Goal and Procedure, both Observed, cb 2 (G 1): Search = Symbolic = 2.6
        // exactly; the tie goes to Symbolic (cost 0.4 < 0.6).
        (
            "exact tie",
            vec![
                fact(R::Goal, E::Observed, 2, V::Live),
                fact(R::Procedure, E::Observed, 2, V::Live),
            ],
            0,
            2,
            O::Symbolic,
        ),
        // A revoked Evidence fact is not admitted; the Goal fact alone decides.
        (
            "revoked ignored",
            vec![
                fact(R::Goal, E::Verified, 4, V::Live),
                fact(R::Evidence, E::Observed, 4, V::Revoked),
            ],
            2,
            2,
            O::Search,
        ),
        // Evidence/Hypothesis under tabular: Statistical 0.8*2 = 1.6 beats
        // Probabilistic 0.8*0.9 + 0.85 = 1.57.
        (
            "evidence hypothesis",
            vec![fact(R::Evidence, E::Hypothesis, 2, V::Live)],
            0,
            2,
            O::Statistical,
        ),
        // Evidence/Unknown: Statistical 0.6 loses to Probabilistic 0.27 + 0.35 = 0.62.
        (
            "unknown bonus",
            vec![fact(R::Evidence, E::Unknown, 2, V::Live)],
            0,
            2,
            unknown_bonus_winner,
        ),
        // Claim under interventional: Deductive 2*1.3 = 2.6, Causal 0.9*1.3 = 1.17.
        (
            "claim",
            vec![fact(R::Claim, E::Observed, 2, V::Live)],
            2,
            2,
            O::Deductive,
        ),
    ];
    for (name, facts, regime, budget, expected) in cases {
        let got = rule::label(&facts, regime, budget);
        if got != code(expected) {
            return Err(format!(
                "T8 {name}: rule gives code {got}, hand gives {expected:?}"
            ));
        }
    }
    Ok(())
}

/// T7: an arm's rows do not depend on which arms ran before it in the process,
/// or on the process having run it before.
fn arm_order(args: &Args) -> Result<(), String> {
    let data = load(args)?;
    let payloads = Payloads::new(DEFAULT_D_MODEL);
    let schedule = Schedule {
        steps: 20,
        peak: 0.005,
    };
    let seeds = Seeds {
        init: 7,
        order: 7 ^ ORDER_SALT,
    };
    let run = |name: &str| -> Result<String, String> {
        let arm = arms::find(name)?;
        let trained = train::train(
            &arm,
            &data,
            &payloads,
            DEFAULT_D_MODEL,
            &schedule,
            &seeds,
            false,
        );
        let score = train::score(
            &burn::module::AutodiffModule::valid(&trained.model),
            &arm,
            data.split("val"),
            &payloads,
        );
        Ok(format!(
            "{}\n{}",
            trained.rows.join("\n"),
            score.predictions
        ))
    };
    let first = run("full")?;
    let _ = run("latent-0")?;
    let after_other = run("full")?;
    let _ = run("no-semantic-slots")?;
    let again = run("full")?;
    if first != after_other || first != again {
        return Err("T7: the full arm's rows depend on what ran before it".into());
    }
    Ok(())
}

fn self_test(args: &Args) -> Result<bool, String> {
    rule_cases()?;
    emit(r#"{"row":"self-test","test":"T8","passed":1}"#);
    arm_order(args)?;
    emit(r#"{"row":"self-test","test":"T7","passed":1}"#);
    Ok(true)
}

/// Dispatch the study phase; exit 2 on a returned error or 3 on a false phase result.
fn main() {
    let result = Args::parse().and_then(|args| match args.get("phase")? {
        "list-arms" => {
            let arms: Vec<String> = arms::ARMS.iter().map(Arm::json).collect();
            emit(&format!("[{}]", arms.join(",")));
            Ok(true)
        }
        "self-test" => self_test(&args),
        "calibrate" => calibrate(&args),
        "sweep" => sweep(&args),
        "eval" => eval(&args),
        "plain" => plain(&args),
        "paired" => paired(&args),
        "paired-v5" => paired_v5(&args),
        other => Err(format!("unknown phase {other:?}")),
    });
    match result {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("a0_ablation: a loss was not finite");
            std::process::exit(3);
        }
        Err(error) => {
            eprintln!("a0_ablation: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod rule_case_tests {
    #[test]
    fn fixed_rule_cases_hold() {
        super::rule_cases().expect("T8 hand-computed rule cases");
    }
}
