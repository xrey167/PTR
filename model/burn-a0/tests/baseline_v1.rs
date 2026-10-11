//! The baseline-v1 runner (examples/a0_ablation/baseline.rs): study arm tables,
//! typed errors, matched compute, the val-only sweep, the selection rule, the
//! plain analogue of the calibrated evaluation and a tiny end-to-end run.
//!
//! The runner is an example binary, so its modules are compiled into this test
//! crate by path; `crate::` inside them then resolves to this file.
#![allow(dead_code)]

#[path = "../examples/a0_ablation/arms.rs"]
mod arms;
#[path = "../examples/a0_ablation/baseline.rs"]
mod baseline;
#[path = "../examples/a0_ablation/batch.rs"]
mod batch;
#[path = "../examples/a0_ablation/data.rs"]
mod data;
#[path = "../examples/a0_ablation/rng.rs"]
mod rng;
#[path = "../examples/a0_ablation/rule.rs"]
mod rule;
#[path = "../examples/a0_ablation/train.rs"]
mod train;

use arms::{Study, StudyError, PLAIN_ARM};
use baseline::{Candidate, Knobs, Plan, SelectError, SweepPoint, GRID};
use burn::{module::Module, prelude::Device};
use data::{Dataset, Example, Fact, Split};
use ptr_burn_a0::{PlainTransformerConfig, TypedAttentionMode};
use rng::SplitMix64;
use std::collections::BTreeMap;

const D_MODEL: usize = 48;

/// Model initialisation draws from the device's global generator, so tests that
/// build or train models must not interleave (tests/ablation_switches.rs).
static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    DEVICE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn knobs() -> Knobs {
    Knobs {
        rank: 8,
        bias_limit: 1.0,
        metadata_dropout: 0.0,
    }
}

/// Deterministic synthetic split: random labels, tokens (PAD = 0 included) and facts.
fn synthetic_split(name: &str, n: usize, seed: u64) -> Split {
    let tables = data::tables();
    let mut rng = SplitMix64::new(seed);
    let mut pick = |len: usize| (rng.u01() * len as f64) as usize % len;
    let examples = (0..n)
        .map(|_| Example {
            label: pick(11),
            tokens: (0..36)
                .map(|_| {
                    if pick(5) == 0 {
                        0
                    } else {
                        1 + pick(215) as i64
                    }
                })
                .collect(),
            // Like the frozen folds, every example keeps one live fact (slot 0);
            // an example admitting nothing is not part of the benchmark.
            facts: (0..data::SLOTS)
                .map(|slot| Fact {
                    role: tables.roles[pick(tables.roles.len())],
                    epistemic: tables.states[pick(tables.states.len())],
                    confidence: pick(100) as f32 / 100.0,
                    bucket: pick(5),
                    validity: if slot == 0 {
                        ptr_types::Validity::Live
                    } else {
                        tables.validities[pick(tables.validities.len())]
                    },
                    entity: pick(256) as u32,
                })
                .collect(),
        })
        .collect();
    Split {
        name: name.to_owned(),
        examples,
        label_fnv64: 0,
    }
}

fn dataset(names: &[&str]) -> Dataset {
    Dataset {
        fnv64: 0,
        splits: names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let n = if *name == "train" { 256 } else { 64 };
                synthetic_split(name, n, 100 + i as u64)
            })
            .collect(),
    }
}

fn params(arm: &arms::Arm) -> usize {
    arm.config(D_MODEL).init(&Device::flex()).num_params()
}

#[test]
fn every_study_table_resolves_with_the_calibrated_router() {
    let expected: [(Study, &[&str]); 3] = [
        (
            Study::M001V5,
            &["full", "no-semantic-slots-masked", "no-semantic-slots"],
        ),
        (
            Study::M002V10,
            &["factorized-v2", "factorized-v2-off", "full"],
        ),
        (
            Study::M003V2,
            &["full", "latent-0", "latent-1", "latent-linear"],
        ),
    ];
    for (study, names) in expected {
        assert_eq!(study.arm_names(), names);
        let arms = study.arms();
        assert_eq!(arms.iter().map(|a| a.name).collect::<Vec<_>>(), names);
        for arm in &arms {
            assert_eq!(arm.experiment, study.id());
            assert_eq!(arm.router_mode, ptr_burn_a0::RouterMode::CalibratedCosineV2);
        }
        assert_eq!(Study::from_id(study.id()), Ok(study));
    }
    let (arms, plain) = Study::M003V2
        .select(&["latent-1", PLAIN_ARM, "full"])
        .expect("a subset with plain resolves");
    assert!(plain);
    assert_eq!(
        arms.iter().map(|a| a.name).collect::<Vec<_>>(),
        ["latent-1", "full"]
    );
    let (_, plain) = Study::M002V10.select(&["full"]).expect("without plain");
    assert!(!plain);
}

#[test]
fn unknown_and_foreign_names_are_typed_errors() {
    assert_eq!(
        Study::from_id("m002-v9"),
        Err(StudyError::UnknownStudy("m002-v9".to_owned()))
    );
    assert!(matches!(
        Study::from_id("M001-v5"),
        Err(StudyError::UnknownStudy(_))
    ));
    // A name from another study is not borrowed.
    assert_eq!(
        Study::M001V5.arm("latent-0").unwrap_err(),
        StudyError::UnknownArm {
            study: "m001-v5",
            arm: "latent-0".to_owned()
        }
    );
    assert!(matches!(
        Study::M003V2.arm("factorized-v2"),
        Err(StudyError::UnknownArm { .. })
    ));
    // The historical per-experiment suffixed names do not resolve either.
    assert!(Study::M002V10.arm("factorized-v2-v9").is_err());
    // Plain is not an A0 arm; it is selected, never resolved.
    assert!(Study::M001V5.arm(PLAIN_ARM).is_err());
    assert_eq!(
        Study::M003V2.select(&["full", "full"]).unwrap_err(),
        StudyError::DuplicateArm {
            study: "m003-v2",
            arm: "full".to_owned()
        }
    );
    assert_eq!(
        Study::M003V2.select(&[]).unwrap_err(),
        StudyError::EmptySelection { study: "m003-v2" }
    );
    let message = StudyError::UnknownStudy("x".to_owned()).to_string();
    assert!(message.contains("m001-v5") && message.contains("m003-v2"));
    assert!(Plan::new(Study::M001V5, Some(&["nope"]), 17, 10, D_MODEL, knobs()).is_err());
}

#[test]
fn m002_pair_differs_only_in_the_typed_attention_switch() {
    let _device = serial();
    let on = Study::M002V10.arm("factorized-v2").unwrap();
    let off = Study::M002V10.arm("factorized-v2-off").unwrap();
    assert_eq!(on.typed_attention_mode, TypedAttentionMode::FactorizedV2);
    assert_eq!(off.typed_attention_mode, TypedAttentionMode::Off);
    assert_eq!(params(&on), params(&off), "identical parameter count");
    assert_eq!(
        train::a0_flops_per_example(&on, D_MODEL),
        train::a0_flops_per_example(&off, D_MODEL)
    );
    assert_eq!(
        train::a0_flops_latent_adjusted(&on, D_MODEL),
        train::a0_flops_latent_adjusted(&off, D_MODEL)
    );
    let normalize = |arm: arms::Arm| arms::Arm {
        name: "",
        typed_attention: true,
        typed_attention_mode: TypedAttentionMode::Off,
        ..arm
    };
    let (a, b) = (normalize(on), normalize(off));
    assert_eq!(a.json(), b.json());
    assert_eq!(
        (
            a.latent_steps,
            a.label_smoothing,
            a.consistency_weight,
            a.batch
        ),
        (
            b.latent_steps,
            b.label_smoothing,
            b.consistency_weight,
            b.batch
        )
    );
}

#[test]
fn latent_linear_matches_the_parameters_and_compute_of_full() {
    let _device = serial();
    let full = Study::M003V2.arm("full").unwrap();
    let linear = Study::M003V2.arm("latent-linear").unwrap();
    assert_eq!(full.latent_steps, 2);
    assert!(full.latent_nonlinearity && !linear.latent_nonlinearity);
    assert_eq!(linear.latent_steps, full.latent_steps);
    assert_eq!(params(&linear), params(&full));
    assert_eq!(
        train::a0_flops_latent_adjusted(&linear, D_MODEL),
        train::a0_flops_latent_adjusted(&full, D_MODEL)
    );
    // The latent-step arms share parameters too (the layer exists at k = 0) and
    // the adjusted FLOPs move by exactly one projection per step.
    let zero = Study::M003V2.arm("latent-0").unwrap();
    let one = Study::M003V2.arm("latent-1").unwrap();
    assert_eq!(params(&zero), params(&full));
    assert_eq!(params(&one), params(&full));
    let step = 2 * 6 * (D_MODEL as i64) * (D_MODEL as i64);
    assert_eq!(
        train::a0_flops_latent_adjusted(&full, D_MODEL)
            - train::a0_flops_latent_adjusted(&one, D_MODEL),
        step
    );
    assert_eq!(
        train::a0_flops_latent_adjusted(&full, D_MODEL)
            - train::a0_flops_latent_adjusted(&zero, D_MODEL),
        2 * step
    );
    // The legacy formula is k-blind (it always assumes two passes).
    assert_eq!(
        train::a0_flops_per_example(&zero, D_MODEL),
        train::a0_flops_per_example(&full, D_MODEL)
    );
}

#[test]
fn study_arms_stay_within_one_percent_of_plain_parameters() {
    let _device = serial();
    let plain = PlainTransformerConfig::new(216, D_MODEL, 11)
        .init(&Device::flex())
        .num_params();
    for study in Study::ALL {
        let plan = Plan::new(study, None, 17, 10, D_MODEL, knobs()).unwrap();
        for arm in &plan.arms {
            let relative = (params(arm) as f64 / plain as f64 - 1.0).abs();
            eprintln!(
                "params {} {}: {} vs plain {plain}",
                study.id(),
                arm.name,
                params(arm)
            );
            assert!(relative <= 0.01, "{} {}: {relative}", study.id(), arm.name);
        }
    }
}

#[test]
fn learning_rates_have_no_fallback() {
    let text = "# c\narm\tlr\nfull\t0.005\tx\nplain\t0.0125\n";
    let rates = baseline::parse_rates(text).unwrap();
    assert_eq!(baseline::rate_for(&rates, "plain"), Ok(0.0125));
    assert_eq!(
        baseline::rate_for(&rates, "latent-0"),
        Err(baseline::RateError::Missing {
            arm: "latent-0".to_owned()
        })
    );
    let plan = Plan::new(Study::M003V2, None, 17, 10, D_MODEL, knobs()).unwrap();
    let error = plan.rates(&rates).unwrap_err();
    assert!(matches!(error, baseline::RateError::Missing { .. }));
    assert!(error.to_string().contains("no rate"));
    for bad in [
        "full\tabc",
        "full\t0",
        "full\t-1",
        "full\tnan",
        "full\tinf",
        "full",
        "full\t0.1\nfull\t0.2",
    ] {
        assert!(baseline::parse_rates(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn fold_tokens_are_validated_like_paired_v5() {
    let ok = baseline::parse_folds(
        "evidence-temporal=8c3bc2324e0eb770,claim-interventional=fd11dbf9270b2f39",
    )
    .unwrap();
    assert_eq!(ok[0], ("evidence-temporal".to_owned(), 0x8c3bc2324e0eb770));
    for bad in ["", "a", "a=zz", "=1f", "a=1,a=2", "a=1,,b=2"] {
        assert!(baseline::parse_folds(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn the_sweep_reads_train_and_val_only() {
    let _device = serial();
    // No test split exists: touching one would panic in `Dataset::split`.
    let data = dataset(&["train", "val"]);
    let plan = Plan::new(
        Study::M003V2,
        Some(&["latent-linear", PLAIN_ARM]),
        17,
        2,
        D_MODEL,
        knobs(),
    )
    .unwrap();
    let mut rows = Vec::new();
    let finite =
        baseline::sweep_fold(&plan, "fold-a", &data, &mut |l| rows.push(l.to_owned())).unwrap();
    assert!(finite);
    assert_eq!(rows.len(), 2 * GRID.len());
    for row in &rows {
        assert!(row.contains(r#""row":"sweep-baseline-v1""#));
        assert!(!row.contains("test") && !row.contains("ood"), "{row}");
        for key in ["val_accuracy", "val_nll", "val_ece15"] {
            assert!(baseline::json_field(row, key)
                .unwrap()
                .parse::<f64>()
                .unwrap()
                .is_finite());
        }
    }
    let lrs: Vec<f64> = rows[..4]
        .iter()
        .map(|r| baseline::json_field(r, "lr").unwrap().parse().unwrap())
        .collect();
    assert_eq!(lrs, GRID);
    // The rows feed the selection straight back.
    let (study, seed, points) = baseline::parse_sweep(&rows.join("\n")).unwrap();
    assert_eq!((study.as_str(), seed), ("m003-v2", 17));
    let selected = baseline::select_rates(&points).unwrap();
    assert_eq!(
        selected.iter().map(|s| s.arm.as_str()).collect::<Vec<_>>(),
        ["latent-linear", "plain"]
    );
    assert!(GRID.contains(&selected[0].best.lr));
}

fn candidate(lr: f64, val_accuracy: f64, val_nll: f64) -> Candidate {
    Candidate {
        lr,
        val_accuracy,
        val_nll,
    }
}

#[test]
fn selection_breaks_ties_by_nll_then_smaller_lr() {
    // Best accuracy wins even with worse NLL.
    let best = baseline::select_best(&[
        candidate(0.0025, 0.50, 0.1),
        candidate(0.005, 0.60, 9.0),
        candidate(0.0125, 0.55, 0.1),
    ])
    .unwrap();
    assert_eq!(best.lr, 0.005);
    // Accuracy tie: lower NLL.
    let best =
        baseline::select_best(&[candidate(0.0025, 0.6, 2.0), candidate(0.025, 0.6, 1.0)]).unwrap();
    assert_eq!(best.lr, 0.025);
    // Full tie: smaller lr, in any input order.
    let tied = [
        candidate(0.0125, 0.6, 1.0),
        candidate(0.0025, 0.6, 1.0),
        candidate(0.005, 0.6, 1.0),
    ];
    assert_eq!(baseline::select_best(&tied).unwrap().lr, 0.0025);
    let mut reversed = tied;
    reversed.reverse();
    assert_eq!(baseline::select_best(&reversed).unwrap().lr, 0.0025);
    assert!(baseline::select_best(&[]).is_none());
}

fn point(arm: &str, fold: &str, lr: f64, accuracy: f64, nll: f64, diverged: bool) -> SweepPoint {
    SweepPoint {
        arm: arm.to_owned(),
        fold: fold.to_owned(),
        lr,
        val_accuracy: accuracy,
        val_nll: nll,
        diverged,
    }
}

#[test]
fn selection_averages_folds_excludes_divergence_and_writes_the_lr_file() {
    let mut points = Vec::new();
    for (i, lr) in GRID.iter().enumerate() {
        // Two folds; the mean accuracy peaks at the third grid point.
        let acc = [0.4, 0.5, 0.7, 0.9][i];
        points.push(point("plain", "a", *lr, acc, 1.0, false));
        points.push(point("plain", "b", *lr, acc, 1.0, i == 3));
        points.push(point("full", "a", *lr, 0.5, 1.0, false));
        points.push(point("full", "b", *lr, 0.5, 1.0, false));
    }
    let selected = baseline::select_rates(&points).unwrap();
    let by_arm: BTreeMap<_, _> = selected
        .iter()
        .map(|s| (s.arm.as_str(), s.best.lr))
        .collect();
    // plain: the 0.025 point diverged on fold b, so 0.0125 wins.
    assert_eq!(by_arm["plain"], 0.0125);
    // full: complete tie -> smallest lr.
    assert_eq!(by_arm["full"], 0.0025);
    let text = baseline::render_lr_file("m001-v5", 17, &selected);
    let rates = baseline::parse_rates(&text).unwrap();
    assert_eq!(rates["plain"], 0.0125);
    assert_eq!(rates["full"], 0.0025);
    assert!(text.contains("arm\tlr"));
}

#[test]
fn selection_rejects_incomplete_or_inconsistent_sweeps() {
    let grid_points = |arm: &str| -> Vec<SweepPoint> {
        GRID.iter()
            .map(|lr| point(arm, "a", *lr, 0.5, 1.0, false))
            .collect()
    };
    assert_eq!(baseline::select_rates(&[]), Err(SelectError::NoRows));
    let mut missing = grid_points("full");
    missing.pop();
    assert!(matches!(
        baseline::select_rates(&missing),
        Err(SelectError::IncompleteGrid { .. })
    ));
    let mut off = grid_points("full");
    off[0].lr = 0.001;
    assert!(matches!(
        baseline::select_rates(&off),
        Err(SelectError::OffGrid { .. })
    ));
    let mut duplicate = grid_points("full");
    duplicate.push(duplicate[0].clone());
    assert!(matches!(
        baseline::select_rates(&duplicate),
        Err(SelectError::DuplicatePoint { .. })
    ));
    let mut mismatch = grid_points("full");
    mismatch.push(point("full", "b", GRID[0], 0.5, 1.0, false));
    assert!(matches!(
        baseline::select_rates(&mismatch),
        Err(SelectError::FoldMismatch { .. })
    ));
    let diverged: Vec<_> = GRID
        .iter()
        .map(|lr| point("full", "a", *lr, 0.0, 0.0, true))
        .collect();
    assert!(matches!(
        baseline::select_rates(&diverged),
        Err(SelectError::AllDiverged { .. })
    ));
    assert!(baseline::parse_sweep("").is_err());
    let a = r#"{"row":"sweep-baseline-v1","study":"m001-v5","fold":"a","arm":"full","seed":17,"lr":0.005,"val_accuracy":0.5,"val_nll":1,"nan":0}"#;
    let b = a.replace("17", "29");
    assert_eq!(
        baseline::parse_sweep(&format!("{a}\n{b}")),
        Err(SelectError::MixedRuns)
    );
}

#[test]
fn plain_calibrated_evaluation_agrees_with_raw_at_unit_temperature() {
    let _device = serial();
    let data = dataset(&["train", "val", "test_iid", "test_ood"]);
    let schedule = train::Schedule {
        steps: 2,
        peak: 0.005,
    };
    let seeds = train::Seeds {
        init: 7,
        order: 7 ^ baseline::ORDER_SALT,
    };
    let trained = train::train_plain(&data, D_MODEL, &schedule, &seeds, false);
    let model = module_valid(&trained);
    for name in ["val", "test_iid", "test_ood"] {
        let split = data.split(name);
        let raw = train::score_plain(&model, split, D_MODEL);
        let rows = train::collect_logits_plain(&model, split, D_MODEL);
        let unit = train::score_rows(&rows, 1.0, 0.0);
        assert_eq!((raw.n, raw.correct), (unit.n, unit.correct));
        assert!((raw.nll - unit.nll).abs() < 1e-12, "{name}");
        assert!((raw.ece - unit.ece).abs() < 1e-12, "{name}");
        assert_eq!(raw.predictions, unit.predictions);
    }
    // The v5-style rows reuse the very same fitted values the runner reports.
    let v5 = train::evaluate_plain_v5(&trained, &data, D_MODEL);
    let calibration = v5
        .iter()
        .find(|l| l.contains(r#""row":"calibration""#))
        .unwrap();
    let temperature: f64 = baseline::json_field(calibration, "temperature")
        .unwrap()
        .parse()
        .unwrap();
    assert!((0.25..=4.0).contains(&temperature));
    let plan = Plan::new(Study::M001V5, Some(&[PLAIN_ARM]), 7, 2, D_MODEL, knobs()).unwrap();
    let rates = plan
        .rates(&BTreeMap::from([(PLAIN_ARM.to_owned(), 0.005)]))
        .unwrap();
    let mut rows = Vec::new();
    baseline::run_fold(&plan, &rates, "f", &data, &mut |l| rows.push(l.to_owned())).unwrap();
    for split in ["test_iid", "test_ood"] {
        let ours = rows
            .iter()
            .find(|l| {
                l.contains(&format!(r#""split":"{split}""#)) && l.contains("final-baseline-v1")
            })
            .unwrap();
        let theirs = v5
            .iter()
            .find(|l| l.contains(&format!(r#""split":"{split}""#)) && l.contains("final-v5"))
            .unwrap();
        assert_eq!(
            baseline::json_field(ours, "fitted_temperature"),
            Some(format!("{temperature}").as_str())
        );
        for (a, b) in [
            ("fitted_nll", "nll"),
            ("fitted_ece15", "ece15"),
            ("coverage", "coverage"),
            ("selective_error", "selective_error"),
        ] {
            assert_eq!(
                baseline::json_field(ours, a),
                baseline::json_field(theirs, b),
                "{split} {a}"
            );
        }
        // Raw gate quantities: at T = 1 the runner's raw columns are the unit-T scores.
        let unit = train::score_rows(
            &train::collect_logits_plain(&model, data.split(split), D_MODEL),
            1.0,
            0.0,
        );
        let raw_nll: f64 = baseline::json_field(ours, "raw_nll")
            .unwrap()
            .parse()
            .unwrap();
        assert!((raw_nll - unit.nll).abs() < 1e-9);
    }
}

fn module_valid(trained: &train::PlainTrained) -> ptr_burn_a0::PlainTransformer {
    use burn::module::AutodiffModule;
    trained.model.valid()
}

const FINAL_COLUMNS: [&str; 32] = [
    "study",
    "fold",
    "arm",
    "split",
    "seed",
    "n",
    "correct",
    "accuracy",
    "raw_nll",
    "raw_ece15",
    "raw_confidence_threshold",
    "raw_coverage",
    "raw_selective_error",
    "fitted_temperature",
    "fitted_nll",
    "fitted_ece15",
    "confidence_threshold",
    "covered",
    "coverage",
    "covered_correct",
    "selective_error",
    "abstained",
    "total_params",
    "flops_per_example",
    "flops_per_example_legacy",
    "train_wall_ms",
    "ms_per_step",
    "latency_p95_ms",
    "latent_steps",
    "explicit_token_count",
    "explicit_token_mean",
    "row",
];

#[test]
fn baseline_v1_smoke_emits_every_required_column_finite_for_each_study() {
    let _device = serial();
    let data = dataset(&["train", "val", "test_iid", "test_ood"]);
    for study in Study::ALL {
        let plan = Plan::new(study, None, 17, 2, D_MODEL, knobs()).unwrap();
        let rates: BTreeMap<String, f64> = plan
            .names()
            .into_iter()
            .map(|n| (n.to_owned(), 0.005))
            .collect();
        let rates = plan.rates(&rates).unwrap();
        let mut rows = Vec::new();
        let finite = baseline::run_fold(&plan, &rates, "fold-x", &data, &mut |l| {
            rows.push(l.to_owned())
        })
        .unwrap();
        assert!(finite, "{}: {rows:#?}", study.id());
        let finals: Vec<&String> = rows
            .iter()
            .filter(|r| r.contains(r#""row":"final-baseline-v1""#))
            .collect();
        assert_eq!(finals.len(), plan.names().len() * 2, "{}", study.id());
        let expected_tokens: usize = data
            .split("test_ood")
            .examples
            .iter()
            .map(|e| e.tokens.iter().filter(|&&t| t != 0).count())
            .sum();
        for row in &finals {
            for column in FINAL_COLUMNS.iter() {
                let value = baseline::json_field(row, column)
                    .unwrap_or_else(|| panic!("{column} missing in {row}"));
                if !matches!(*column, "study" | "fold" | "arm" | "split" | "row") {
                    assert!(
                        value.parse::<f64>().expect(column).is_finite(),
                        "{column}={value}"
                    );
                }
            }
            if row.contains(r#""split":"test_ood""#) {
                // Identical across arms; the study does not test token savings.
                assert_eq!(
                    baseline::json_field(row, "explicit_token_count"),
                    Some(expected_tokens.to_string().as_str())
                );
            }
            let nll: f64 = baseline::json_field(row, "raw_nll")
                .unwrap()
                .parse()
                .unwrap();
            assert!(nll > 0.0);
        }
        let metas = rows
            .iter()
            .filter(|r| r.contains("meta-baseline-v1"))
            .count();
        assert_eq!(metas, plan.names().len());
        let preds = rows.iter().filter(|r| r.starts_with("PRED ")).count();
        assert_eq!(preds, plan.names().len() * 2);
        assert!(rows.iter().any(|r| r.contains(r#""arm":"plain""#)));
        let masks = rows
            .iter()
            .filter(|r| r.contains("mask-share-baseline-v1"))
            .count();
        assert_eq!(masks, if study == Study::M001V5 { 2 } else { 0 });
    }
}

#[test]
fn a_missing_rate_fails_before_any_training() {
    let _device = serial();
    let plan = Plan::new(Study::M001V5, None, 17, 2, D_MODEL, knobs()).unwrap();
    let partial = BTreeMap::from([("full".to_owned(), 0.005), (PLAIN_ARM.to_owned(), 0.005)]);
    assert!(plan.rates(&partial).is_err());
    // run_fold itself also refuses a rate map that lacks a selected arm.
    let data = dataset(&["train", "val", "test_iid", "test_ood"]);
    let ok = BTreeMap::from([("full", 0.005)]);
    let mut rows = Vec::new();
    let error =
        baseline::run_fold(&plan, &ok, "f", &data, &mut |l| rows.push(l.to_owned())).unwrap_err();
    assert!(error.contains("no rate"));
}
