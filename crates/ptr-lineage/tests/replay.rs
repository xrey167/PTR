use std::collections::BTreeSet;

use ptr_lineage::{LineageError, ModelTime, NewSample, ReplayParams, ReplayPool, Split};

fn params() -> ReplayParams {
    ReplayParams {
        initial_stability: 100.0,
        growth: 1.5,
        lapse_factor: 0.3,
        min_stability: 1.0,
        difficulty_step: 1.0,
        lapse_loss: 0.5,
        max_lapses: 3,
    }
}

fn sample(id: &str, stratum: &str) -> NewSample {
    NewSample {
        id: id.into(),
        stratum: stratum.into(),
        split: Split::Train,
    }
}

fn pool(ids: &[(&str, &str)]) -> ReplayPool {
    let mut pool = ReplayPool::new(params()).unwrap();
    for (id, stratum) in ids {
        pool.insert(sample(id, stratum), ModelTime(0.0)).unwrap();
    }
    pool
}

#[test]
fn a_held_out_sample_can_never_enter_the_pool() {
    let mut pool = ReplayPool::new(params()).unwrap();
    let held_out = NewSample {
        split: Split::HeldOut,
        ..sample("h1", "t")
    };
    assert_eq!(
        pool.insert(held_out, ModelTime(0.0)).unwrap_err(),
        LineageError::HeldOutSample { id: "h1".into() }
    );
    assert!(pool.is_empty());
}

#[test]
fn a_lapsed_sample_outranks_a_retained_one_at_the_same_model_time() {
    let mut pool = pool(&[("kept", "t"), ("lost", "t")]);
    pool.record_probe("kept", 0.1, ModelTime(50.0)).unwrap();
    pool.record_probe("lost", 2.0, ModelTime(50.0)).unwrap();
    let now = ModelTime(80.0);
    let kept = pool.priority(pool.get("kept").unwrap(), now);
    let lost = pool.priority(pool.get("lost").unwrap(), now);
    assert!(lost > kept, "lost {lost} kept {kept}");
    assert!(pool.get("lost").unwrap().memory.stability < params().initial_stability);
    assert!(pool.get("kept").unwrap().memory.stability > params().initial_stability);
}

#[test]
fn nothing_is_forgotten_while_the_model_clock_stands_still() {
    let pool = pool(&[("a", "t")]);
    assert_eq!(pool.priority(pool.get("a").unwrap(), ModelTime(0.0)), 0.0);
}

#[test]
fn a_draw_is_distinct_reproducible_and_covers_every_stratum() {
    let ids: Vec<(String, String)> = (0..60)
        .map(|i| (format!("s{i}"), format!("task{}", i % 3)))
        .collect();
    let refs: Vec<(&str, &str)> = ids.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let pool = pool(&refs);
    let now = ModelTime(500.0);
    let first = pool.sample(12, now, 42);
    assert_eq!(first.len(), 12);
    assert_eq!(first.iter().collect::<BTreeSet<_>>().len(), 12);
    assert_eq!(first, pool.sample(12, now, 42));
    assert_ne!(first, pool.sample(12, now, 43));
    let strata: BTreeSet<&str> = first
        .iter()
        .map(|id| pool.get(id).unwrap().stratum.as_str())
        .collect();
    assert_eq!(strata.len(), 3);
}

#[test]
fn forgotten_samples_are_drawn_far_more_often_than_retained_ones() {
    let ids: Vec<String> = (0..40).map(|i| format!("s{i}")).collect();
    let mut pool = ReplayPool::new(params()).unwrap();
    for id in &ids {
        pool.insert(sample(id, "t"), ModelTime(0.0)).unwrap();
    }
    // Half the samples lapse, half are retained, all at the same time.
    for (i, id) in ids.iter().enumerate() {
        let loss = if i % 2 == 0 { 3.0 } else { 0.0 };
        pool.record_probe(id, loss, ModelTime(10.0)).unwrap();
    }
    let now = ModelTime(60.0);
    let mut lapsed = 0;
    let mut retained = 0;
    for seed in 0..200 {
        for id in pool.sample(5, now, seed) {
            let index: usize = id[1..].parse().unwrap();
            if index % 2 == 0 {
                lapsed += 1;
            } else {
                retained += 1;
            }
        }
    }
    assert!(lapsed > 3 * retained, "lapsed {lapsed} retained {retained}");
}

#[test]
fn asking_for_more_than_the_pool_returns_the_whole_pool_once() {
    let pool = pool(&[("a", "x"), ("b", "y")]);
    let drawn = pool.sample(10, ModelTime(5.0), 1);
    assert_eq!(drawn.len(), 2);
}

#[test]
fn a_sample_that_keeps_lapsing_is_withheld_pending_a_label_audit() {
    let mut pool = pool(&[("noisy", "t"), ("fine", "t")]);
    for step in 1..=3 {
        pool.record_probe("noisy", 5.0, ModelTime(step as f64 * 10.0))
            .unwrap();
    }
    let audit: Vec<&str> = pool.needing_audit().map(|s| s.id.as_str()).collect();
    assert_eq!(audit, vec!["noisy"]);
    let drawn = pool.sample(2, ModelTime(100.0), 7);
    assert_eq!(drawn, vec!["fine"]);
}

#[test]
fn a_sample_probed_at_this_model_time_is_not_drawn() {
    let mut pool = pool(&[("a", "t"), ("b", "t")]);
    pool.record_probe("a", 0.0, ModelTime(40.0)).unwrap();
    let drawn = pool.sample(2, ModelTime(40.0), 1);
    assert_eq!(drawn, vec!["b"]);
}

#[test]
fn nonfinite_probes_leave_the_sample_memory_unchanged() {
    let mut pool = pool(&[("a", "task")]);
    pool.record_probe("a", 0.1, ModelTime(10.0)).unwrap();
    let before = pool.get("a").unwrap().clone();
    for loss in [f64::NEG_INFINITY, f64::NAN, f64::INFINITY] {
        assert_eq!(
            pool.record_probe("a", loss, ModelTime(20.0)),
            Err(LineageError::NonFinite {
                field: "probe loss"
            })
        );
        assert_eq!(pool.get("a"), Some(&before));
    }
}

#[test]
fn replay_returns_nothing_for_zero_budget_or_an_entirely_ineligible_pool() {
    let pool = pool(&[("a", "task"), ("b", "task")]);
    assert!(pool.sample(0, ModelTime(100.0), 7).is_empty());
    assert!(pool.sample(10, ModelTime(0.0), 7).is_empty());
    assert!(ReplayPool::new(params())
        .unwrap()
        .sample(10, ModelTime(100.0), 7)
        .is_empty());
}

#[test]
fn a_probe_earlier_than_the_last_one_is_refused_and_changes_nothing() {
    let mut pool = pool(&[("a", "task")]);
    pool.record_probe("a", 0.1, ModelTime(50.0)).unwrap();
    let before = pool.get("a").unwrap().clone();
    let priority = pool.priority(&before, ModelTime(60.0));
    for loss in [0.1, 3.0] {
        assert_eq!(
            pool.record_probe("a", loss, ModelTime(10.0)),
            Err(LineageError::InvalidParameter {
                field: "model time",
                message: "must not precede the sample's last probe",
            })
        );
        assert_eq!(pool.get("a"), Some(&before));
        assert_eq!(
            pool.priority(pool.get("a").unwrap(), ModelTime(60.0)),
            priority
        );
    }
    // A probe at the same model time is not a step backward.
    pool.record_probe("a", 0.1, ModelTime(50.0)).unwrap();
}

#[test]
fn a_nonfinite_model_time_is_refused_and_changes_nothing() {
    let mut pool = pool(&[("a", "task")]);
    pool.record_probe("a", 0.1, ModelTime(10.0)).unwrap();
    let before = pool.get("a").unwrap().clone();
    for now in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for loss in [0.1, 3.0] {
            assert_eq!(
                pool.record_probe("a", loss, ModelTime(now)),
                Err(LineageError::NonFinite {
                    field: "model time"
                })
            );
            assert_eq!(pool.get("a"), Some(&before));
        }
        assert_eq!(
            pool.insert(sample("b", "task"), ModelTime(now)),
            Err(LineageError::NonFinite {
                field: "model time"
            })
        );
        assert!(pool.get("b").is_none());
    }
    // The sample is still drawable once the clock moves on.
    assert_eq!(pool.sample(1, ModelTime(1e9), 1), vec!["a"]);
}

#[test]
fn a_probe_whose_update_would_overflow_the_stability_is_refused_and_changes_nothing() {
    // Finite parameters and model times: one stability interval after
    // insertion, a successful probe multiplies the stability by about seven.
    let huge = ReplayParams {
        initial_stability: f64::MAX / 2.0,
        growth: 100.0,
        ..params()
    };
    let mut pool = ReplayPool::new(huge).unwrap();
    pool.insert(sample("a", "task"), ModelTime(0.0)).unwrap();
    let before = pool.get("a").unwrap().clone();
    let now = ModelTime(huge.initial_stability);
    assert_eq!(
        pool.record_probe("a", 0.1, now),
        Err(LineageError::NonFinite { field: "stability" })
    );
    assert_eq!(pool.get("a"), Some(&before));
    // The sample keeps a positive priority instead of looking retained forever.
    assert!(pool.priority(pool.get("a").unwrap(), now) > 0.0);
    assert_eq!(pool.sample(1, now, 1), vec!["a"]);
    // A lapse only shrinks the stability, so it is still recorded.
    pool.record_probe("a", 3.0, now).unwrap();
    let lapsed = pool.get("a").unwrap().memory;
    assert!(lapsed.stability.is_finite() && lapsed.stability < before.memory.stability);
    assert_eq!(lapsed.lapses, 1);
}
