//! The pilot: choosing the memory shape and the abstention thresholds on the pilot seeds only.
//!
//! `ptr-bench fast-weight-memory calibrate 1001 1002 ...` runs the pilot seeds through every arm
//! with no abstention, pools the queries and, for each arm, picks the operating point by the rule
//! below. It refuses any seed that is not a pilot seed, so a confirmatory seed is never used to
//! tune. Its output is a JSON object; the chosen values are then written into `params.rs` and
//! `config.toml` by hand, and the drift test keeps the two equal.
//!
//! The rule, fixed before the pilot was run:
//!
//! - an arm's *matched operating point* is the smallest score threshold (permille, steps of 5)
//!   at which the arm answers at most `MATCHED_FALSE_RECALL_PERMILLE` (5 percent) of the pooled
//!   unanswerable pilot queries. The recency buffer and the hybrid arms abstain on the cosine
//!   between the query and their top hit; fast memory abstains on the decode score, with a lead
//!   over the runner-up (`min_margin`) taken from {25, 50, 100, 200} permille, the margin that
//!   gives the highest pooled update-aware recall@1 at its matched score threshold (ties: the
//!   smaller margin). A nonzero margin is part of the mechanism's contract (it is the memory's
//!   guard against an ambiguous readout), not a free knob.
//! - the memory shape is the plan's (8 heads, key_dim 32, value_dim 64, scalar decay 0.999) unless
//!   another shape of the grid below reaches a pooled recall@1 at its matched point that is at
//!   least 100 permille higher; then the shape with the highest recall@1 is used and the plan's
//!   shape is reported as invalid for this pool.

use super::arms::{self, Policy, Ranked};
use super::fast::{self, FastConfig};
use super::hybrid;
use super::oracle::Truth;
use super::params;
use super::pool::Pool;
use super::session::{QueryKind, Session};
use crate::experiments::layered_kv::emit::MetricList;
use crate::experiments::layered_kv::emit::{Json, Tally};

/// The memory shapes tried (the first is the plan's).
const GRID: [(usize, usize, usize, i64); 9] = [
    (8, 32, 64, 999),
    (8, 32, 64, 1000),
    (8, 64, 64, 999),
    (16, 64, 64, 999),
    (8, 128, 64, 999),
    (32, 64, 64, 999),
    (8, 256, 64, 999),
    (8, 384, 64, 999),
    (16, 384, 64, 999),
];
const MARGINS_MILLI: [i64; 4] = [25, 50, 100, 200];
const STEP_MILLI: usize = 5;
/// A shape replaces the plan's only if it recalls this much more (permille) at its matched point.
const REPLACEMENT_GAIN_PERMILLE: i64 = 100;

/// One pooled query: the arm's candidates, whether it is answerable and the right row.
struct Item {
    ranked: Ranked,
    answerable: bool,
    expected: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
struct Point {
    min_score_milli: i64,
    min_margin_milli: i64,
    recall_permille: i64,
    answered_unanswerable_permille: i64,
    answerable: i64,
    unanswerable: i64,
}

impl Point {
    fn json(&self) -> Json {
        Json::obj([
            ("min_score_milli", Json::Int(self.min_score_milli)),
            ("min_margin_milli", Json::Int(self.min_margin_milli)),
            ("recall_at_1_permille", Json::Int(self.recall_permille)),
            (
                "answered_unanswerable_permille",
                Json::Int(self.answered_unanswerable_permille),
            ),
            ("answerable_queries", Json::Int(self.answerable)),
            ("unanswerable_queries", Json::Int(self.unanswerable)),
        ])
    }
}

fn evaluate(items: &[Item], score_milli: i64, margin_milli: i64) -> Point {
    let policy = Policy::milli(score_milli, margin_milli);
    let (mut answerable, mut unanswerable, mut correct, mut answered) = (0, 0, 0, 0);
    for item in items {
        let answer = policy.apply(&item.ranked);
        if item.answerable {
            answerable += 1;
            if !answer.is_empty() && answer.first() == item.expected.as_ref() {
                correct += 1;
            }
        } else {
            unanswerable += 1;
            if !answer.is_empty() {
                answered += 1;
            }
        }
    }
    Point {
        min_score_milli: score_milli,
        min_margin_milli: margin_milli,
        recall_permille: super::metrics::permille(correct, answerable),
        answered_unanswerable_permille: super::metrics::permille(answered, unanswerable),
        answerable,
        unanswerable,
    }
}

/// The smallest score threshold at the given margin whose pooled false-answer rate on the
/// unanswerable queries is within the matched rate.
fn matched(items: &[Item], margin_milli: i64) -> Option<Point> {
    (0..=1000)
        .step_by(STEP_MILLI)
        .map(|score| evaluate(items, score, margin_milli))
        .find(|point| point.answered_unanswerable_permille <= params::MATCHED_FALSE_RECALL_PERMILLE)
}

fn best_fast(items: &[Item]) -> Option<Point> {
    let mut best: Option<Point> = None;
    for margin in MARGINS_MILLI {
        if let Some(point) = matched(items, margin) {
            if best
                .as_ref()
                .is_none_or(|current| point.recall_permille > current.recall_permille)
            {
                best = Some(point);
            }
        }
    }
    best
}

struct Pilot {
    items: Vec<Item>,
}

fn items_of(session: &Session, truth: &Truth, ranked: &[Ranked]) -> Vec<Item> {
    session
        .queries
        .iter()
        .zip(ranked)
        .map(|(query, ranked)| Item {
            ranked: ranked.clone(),
            answerable: query.kind.answerable(),
            expected: if query.kind.answerable() {
                truth.current(query.cue)
            } else {
                None
            },
        })
        .collect()
}

pub fn run(seeds: &[u64]) -> Result<String, String> {
    for seed in seeds {
        if !params::PILOT_SEEDS.contains(seed) {
            return Err(format!(
                "calibration uses the pilot seeds {:?} only, not {seed}",
                params::PILOT_SEEDS
            ));
        }
    }
    let pool = Pool::load()?;
    let mut bridge = super::spawn_driver()?;
    let mut recency = Pilot { items: Vec::new() };
    let mut live = Pilot { items: Vec::new() };
    let mut append = Pilot { items: Vec::new() };
    let mut shapes: Vec<Pilot> = GRID.iter().map(|_| Pilot { items: Vec::new() }).collect();
    let mut stale_scores: Vec<Vec<f32>> = GRID.iter().map(|_| Vec::new()).collect();
    let mut readback_failures: Vec<i64> = GRID.iter().map(|_| 0).collect();
    let mut errors: Vec<i64> = GRID.iter().map(|_| 0).collect();
    for &seed in seeds {
        let session = Session::generate(&pool, seed, params::Shape::PINNED)?;
        let truth = Truth::fold(&session.events);
        let (ranked, _) = arms::recency(&pool, &session);
        recency.items.extend(items_of(&session, &truth, &ranked));
        let run = hybrid::run(&mut bridge, &pool, &session, hybrid::Index::Live)
            .map_err(|error| error.to_string())?;
        live.items.extend(items_of(&session, &truth, &run.queries));
        let run = hybrid::run(&mut bridge, &pool, &session, hybrid::Index::Append)
            .map_err(|error| error.to_string())?;
        append
            .items
            .extend(items_of(&session, &truth, &run.queries));
        for (index, &(heads, key_dim, value_dim, decay)) in GRID.iter().enumerate() {
            let config = FastConfig {
                heads,
                key_dim,
                value_dim,
                decay_permille: decay,
                ..FastConfig::pinned()
            };
            let mut tally = Tally::new(MetricList {
                names: &[],
                grouped: &[],
            });
            let fast_run = fast::run(&pool, &session, &config, Policy::milli(0, 0), &mut tally);
            let ranked: Vec<Ranked> = fast_run
                .queries
                .iter()
                .map(|query| query.ranked.clone())
                .collect();
            shapes[index]
                .items
                .extend(items_of(&session, &truth, &ranked));
            readback_failures[index] += tally.get("readback_failures");
            errors[index] += tally.get("product_errors") + tally.get("product_panics");
            for (query, run) in session.queries.iter().zip(&fast_run.queries) {
                if query.kind == QueryKind::Updated {
                    if let Some(stale) = query.stale.first() {
                        if let Some(entry) = run.scored.iter().find(|entry| entry.0 == *stale) {
                            stale_scores[index].push(entry.1);
                        }
                    }
                }
            }
        }
        eprintln!("calibrate: pilot seed {seed} done");
    }
    bridge.quit().map_err(|error| error.to_string())?;

    let mut shape_rows = Vec::new();
    let mut best: Vec<Option<Point>> = Vec::new();
    for (index, &(heads, key_dim, value_dim, decay)) in GRID.iter().enumerate() {
        let point = best_fast(&shapes[index].items);
        stale_scores[index].sort_by(|a, b| a.total_cmp(b));
        let scores = &stale_scores[index];
        let permille = |value: f32| Json::Int((value * 1000.0).round() as i64);
        shape_rows.push(Json::obj([
            ("heads", Json::Int(heads as i64)),
            ("key_dim", Json::Int(key_dim as i64)),
            ("value_dim", Json::Int(value_dim as i64)),
            ("decay_permille", Json::Int(decay)),
            ("readback_failures", Json::Int(readback_failures[index])),
            ("product_errors_or_panics", Json::Int(errors[index])),
            (
                "stale_score_max_permille",
                scores.last().map_or(Json::Null, |value| permille(*value)),
            ),
            (
                "stale_score_p99_permille",
                scores
                    .get(scores.len() * 99 / 100)
                    .map_or(Json::Null, |value| permille(*value)),
            ),
            ("chosen", point.as_ref().map_or(Json::Null, Point::json)),
        ]));
        best.push(point);
    }
    let plan_recall = best[0].as_ref().map_or(-1, |point| point.recall_permille);
    let (mut selected, mut selected_recall) = (0, plan_recall);
    for (index, point) in best.iter().enumerate().skip(1) {
        if let Some(point) = point {
            if point.recall_permille >= plan_recall + REPLACEMENT_GAIN_PERMILLE
                && point.recall_permille > selected_recall
            {
                selected = index;
                selected_recall = point.recall_permille;
            }
        }
    }
    let one = |pilot: &Pilot| matched(&pilot.items, 0).map_or(Json::Null, |point| point.json());
    let (heads, key_dim, value_dim, decay) = GRID[selected];
    let object = Json::obj([
        ("calibration", Json::str("M008 pilot")),
        (
            "pilot_seeds",
            Json::ints(seeds.iter().map(|seed| *seed as i64)),
        ),
        (
            "matched_false_recall_permille",
            Json::Int(params::MATCHED_FALSE_RECALL_PERMILLE),
        ),
        ("shapes", Json::Arr(shape_rows)),
        (
            "selected_memory",
            Json::obj([
                ("grid_index", Json::Int(selected as i64)),
                ("plan_shape_replaced", Json::Bool(selected != 0)),
                ("heads", Json::Int(heads as i64)),
                ("key_dim", Json::Int(key_dim as i64)),
                ("value_dim", Json::Int(value_dim as i64)),
                ("decay_permille", Json::Int(decay)),
                (
                    "fast",
                    best[selected].as_ref().map_or(Json::Null, Point::json),
                ),
            ]),
        ),
        ("recency", one(&recency)),
        ("hybrid_live", one(&live)),
        ("hybrid_append", one(&append)),
    ]);
    Ok(object.render())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(score: f32, runner_up: f32, answerable: bool, right: bool) -> Item {
        Item {
            ranked: Ranked {
                rows: vec![if right { 7 } else { 8 }],
                scores: vec![score],
                runner_up,
            },
            answerable,
            expected: answerable.then_some(7),
        }
    }

    #[test]
    fn the_matched_threshold_is_the_smallest_one_within_the_false_answer_rate() {
        // Twenty unanswerable queries scoring 0.30, 0.31, ... 0.49 and ten answerable ones above.
        let mut items: Vec<Item> = (0..20)
            .map(|i| item(0.30 + i as f32 * 0.01, 0.0, false, false))
            .collect();
        items.extend((0..10).map(|_| item(0.9, 0.0, true, true)));
        let point = matched(&items, 0).unwrap();
        // At most 5 percent of 20 is one query: only the top unanswerable score (0.49) may pass.
        assert!(point.answered_unanswerable_permille <= 50);
        assert!((485..=495).contains(&point.min_score_milli), "{point:?}");
        assert_eq!(point.recall_permille, 1000);
        // A threshold one step lower admits two.
        assert!(evaluate(&items, point.min_score_milli - 5, 0).answered_unanswerable_permille > 50);
    }

    #[test]
    fn the_margin_with_the_highest_recall_wins_and_a_tie_goes_to_the_smaller() {
        let mut items: Vec<Item> = (0..20).map(|_| item(0.1, 0.0, false, false)).collect();
        // Right answers with a lead of 0.3: every grid margin keeps them.
        items.extend((0..10).map(|_| item(0.8, 0.5, true, true)));
        let point = best_fast(&items).unwrap();
        assert_eq!(point.min_margin_milli, 25);
        assert_eq!(point.recall_permille, 1000);
        // Answers with a lead of 0.15 survive margins 25, 50 and 100 only.
        let mut items: Vec<Item> = (0..20).map(|_| item(0.1, 0.0, false, false)).collect();
        items.extend((0..10).map(|_| item(0.8, 0.65, true, true)));
        let point = best_fast(&items).unwrap();
        assert_eq!(point.min_margin_milli, 25);
        assert_eq!(evaluate(&items, 300, 200).recall_permille, 0);
    }

    #[test]
    fn a_confirmatory_seed_is_refused() {
        for seed in params::SEEDS {
            assert!(run(&[seed]).is_err());
        }
        assert!(run(&[1001, 17]).is_err());
    }
}
