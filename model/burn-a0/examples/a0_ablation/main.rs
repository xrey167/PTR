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

    fn get(&self, key: &str) -> Result<&str, String> {
        self.values
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("--{key} is required"))
    }

    fn number<T: std::str::FromStr>(&self, key: &str) -> Result<T, String> {
        self.get(key)?
            .parse()
            .map_err(|_| format!("--{key} is not a number"))
    }

    fn number_or<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.values.get(key) {
            Some(_) => self.number(key),
            None => Ok(default),
        }
    }
}

fn emit(line: &str) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{line}").expect("stdout is writable");
    out.flush().expect("stdout flushes");
}

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

/// Arms named by `--arms`, every one of which must belong to `--experiment`.
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

fn timing(phase: &str, arm: &Arm, trained: &train::Trained) {
    eprintln!(
        "timing phase={phase} arm={} ms_per_step={:.3}",
        arm.name, trained.milliseconds_per_step
    );
}

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
            O::Probabilistic,
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
