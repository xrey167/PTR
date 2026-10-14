mod common;

use common::{backend, caps, option, prob};
use ptr_router::{
    BackendCapability, ConstrainedSelector, ConstraintError, Criterion, Infeasibility, Objective,
    Offer, RouteConstraints, RouteOption, Weights, MAX_OPTIONS,
};

fn selector(constraints: RouteConstraints) -> ConstrainedSelector {
    ConstrainedSelector::new(constraints, Objective::default())
}

#[test]
fn every_infeasibility_reason_is_typed() {
    let constraints = RouteConstraints {
        max_cost_units: 10,
        max_latency_ms: 100,
        min_capabilities: caps(&[BackendCapability::Code, BackendCapability::ToolUse]),
        max_uncertainty: prob(0.2),
    };
    let bad = option("bad", 11, 101, 0.3, &[BackendCapability::Code]);
    let reasons = constraints.violations(&bad);
    assert_eq!(
        reasons,
        vec![
            Infeasibility::CostExceeded {
                required: 11,
                limit: 10
            },
            Infeasibility::LatencyExceeded {
                estimate: 101,
                limit: 100
            },
            Infeasibility::MissingCapabilities {
                missing: caps(&[BackendCapability::ToolUse])
            },
            Infeasibility::UncertaintyTooHigh {
                estimate: prob(0.3),
                limit: prob(0.2)
            },
        ]
    );
    let good = option(
        "good",
        10,
        100,
        0.2,
        &[BackendCapability::Code, BackendCapability::ToolUse],
    );
    assert!(constraints.violations(&good).is_empty());
}

#[test]
fn infeasible_for_all_lists_each_option_with_its_reasons() {
    let constraints = RouteConstraints {
        max_cost_units: 5,
        ..RouteConstraints::unbounded()
    };
    let offers = vec![
        Offer::eligible(option("b", 9, 1, 0.1, &[])),
        Offer::ineligible(option("a", 1, 1, 0.1, &[])),
        Offer::eligible(option("c", 6, 1, 0.1, &[])),
    ];
    let Err(ConstraintError::Infeasible { rejections }) = selector(constraints).select(&offers)
    else {
        panic!("expected Infeasible");
    };
    let ids: Vec<&str> = rejections.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, ["a", "b", "c"]);
    assert_eq!(rejections[0].reasons, vec![Infeasibility::NotEligible]);
    assert_eq!(
        rejections[1].reasons,
        vec![Infeasibility::CostExceeded {
            required: 9,
            limit: 5
        }]
    );
    assert_eq!(rejections[2].reasons.len(), 1);
}

#[test]
fn ineligible_and_infeasible_both_reported() {
    let constraints = RouteConstraints {
        max_latency_ms: 1,
        ..RouteConstraints::unbounded()
    };
    let offers = vec![Offer::ineligible(option("x", 1, 50, 0.1, &[]))];
    let Err(ConstraintError::Infeasible { rejections }) = selector(constraints).select(&offers)
    else {
        panic!("expected Infeasible");
    };
    assert_eq!(rejections[0].reasons.len(), 2);
    assert_eq!(rejections[0].reasons[0], Infeasibility::NotEligible);
}

#[test]
fn offer_list_errors() {
    let s = selector(RouteConstraints::unbounded());
    assert_eq!(s.select(&[]), Err(ConstraintError::NoOptions));
    let dup = vec![
        Offer::eligible(option("a", 1, 1, 0.1, &[])),
        Offer::eligible(option("a", 2, 2, 0.2, &[])),
    ];
    assert_eq!(
        s.select(&dup),
        Err(ConstraintError::DuplicateOption { id: backend("a") })
    );
    let many: Vec<_> = (0..=MAX_OPTIONS)
        .map(|i| Offer::eligible(option(&format!("o{i}"), 1, 1, 0.1, &[])))
        .collect();
    assert_eq!(
        s.select(&many),
        Err(ConstraintError::TooManyOptions {
            count: MAX_OPTIONS + 1
        })
    );
}

#[test]
fn ineligible_option_is_never_chosen_even_if_best_on_every_criterion() {
    let offers = vec![
        Offer::ineligible(option("perfect", 0, 0, 0.0, &[])),
        Offer::eligible(option("ok", 5, 5, 0.5, &[])),
    ];
    let chosen = selector(RouteConstraints::unbounded())
        .select(&offers)
        .unwrap();
    assert_eq!(chosen.chosen.id().as_str(), "ok");
    assert_eq!((chosen.eligible, chosen.feasible), (1, 1));
}

#[test]
fn lexicographic_order_and_weighted_sum_differ_deterministically() {
    let offers = vec![
        Offer::eligible(option("cheap", 1, 900, 0.5, &[])),
        Offer::eligible(option("fast", 9, 10, 0.5, &[])),
        Offer::eligible(option("sure", 5, 100, 0.1, &[])),
    ];
    let pick = |objective| {
        ConstrainedSelector::new(RouteConstraints::unbounded(), objective)
            .select(&offers)
            .unwrap()
            .chosen
            .id()
            .as_str()
            .to_owned()
    };
    assert_eq!(
        pick(Objective::Lexicographic([
            Criterion::Cost,
            Criterion::Latency,
            Criterion::Uncertainty
        ])),
        "cheap"
    );
    assert_eq!(
        pick(Objective::Lexicographic([
            Criterion::Latency,
            Criterion::Cost,
            Criterion::Uncertainty
        ])),
        "fast"
    );
    assert_eq!(pick(Objective::default()), "sure");
    // 1 + 900 = 901, 9 + 10 = 19, 5 + 100 = 105.
    assert_eq!(
        pick(Objective::Weighted(Weights {
            cost: 1,
            latency: 1,
            uncertainty: 0
        })),
        "fast"
    );
}

#[test]
fn choice_is_independent_of_offer_order() {
    let base: Vec<_> = (0..6)
        .map(|i| {
            Offer::eligible(option(
                &format!("m{i}"),
                (i % 3) as u32,
                (i % 2) as u32,
                0.25,
                &[],
            ))
        })
        .collect();
    let s = selector(RouteConstraints::unbounded());
    let first = s.select(&base).unwrap().chosen;
    for rotation in 1..base.len() {
        let mut rotated = base.clone();
        rotated.rotate_left(rotation);
        assert_eq!(s.select(&rotated).unwrap().chosen, first);
    }
}

/// Deterministic pseudo-random stream (xorshift) so the sweep needs no dependency.
struct Stream(u64);
impl Stream {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn random_offers(stream: &mut Stream, count: usize) -> Vec<Offer<RouteOption>> {
    (0..count)
        .map(|i| {
            let mask: Vec<BackendCapability> = BackendCapability::ALL
                .into_iter()
                .filter(|_| stream.next().is_multiple_of(2))
                .collect();
            let o = option(
                &format!("opt{i:02}"),
                (stream.next() % 20) as u32,
                (stream.next() % 200) as u32,
                (stream.next() % 101) as f32 / 100.0,
                &mask,
            );
            Offer {
                option: o,
                eligible: !stream.next().is_multiple_of(3),
            }
        })
        .collect()
}

#[test]
fn constraints_only_shrink_the_eligible_set() {
    let mut stream = Stream(0x9e37_79b9_7f4a_7c15);
    for _ in 0..300 {
        let offers = random_offers(&mut stream, 12);
        let constraints = RouteConstraints {
            max_cost_units: (stream.next() % 25) as u32,
            max_latency_ms: (stream.next() % 250) as u32,
            min_capabilities: caps(&[BackendCapability::ALL[(stream.next() % 5) as usize]]),
            max_uncertainty: prob((stream.next() % 101) as f32 / 100.0),
        };
        let s = selector(constraints);
        let feasible = s.feasible(&offers);
        let eligible: Vec<&RouteOption> = offers
            .iter()
            .filter(|o| o.eligible)
            .map(|o| &o.option)
            .collect();
        assert!(feasible.len() <= eligible.len());
        for option in &feasible {
            assert!(eligible.contains(option), "feasible option not eligible");
            assert!(constraints.violations(option).is_empty());
        }
        // The unconstrained feasible set is exactly the eligible set.
        let unbounded = selector(RouteConstraints::unbounded());
        assert_eq!(unbounded.feasible(&offers), eligible);
        match s.select(&offers) {
            Ok(selection) => {
                assert!(feasible.contains(&&selection.chosen));
                assert_eq!(selection.feasible, feasible.len());
            }
            Err(ConstraintError::Infeasible { rejections }) => {
                assert!(feasible.is_empty());
                assert_eq!(rejections.len(), offers.len());
            }
            Err(other) => panic!("unexpected {other:?}"),
        }
    }
}

#[test]
fn tightening_never_grows_and_loosening_never_shrinks_the_feasible_set() {
    let mut stream = Stream(0x1234_5678_9abc_def1);
    for _ in 0..200 {
        let offers = random_offers(&mut stream, 10);
        let tight = RouteConstraints {
            max_cost_units: 10,
            max_latency_ms: 100,
            min_capabilities: caps(&[BackendCapability::Code]),
            max_uncertainty: prob(0.5),
        };
        let tighter = tight.requiring(caps(&[BackendCapability::Reasoning]));
        let loose = RouteConstraints {
            max_cost_units: 20,
            max_latency_ms: 200,
            ..tight
        };
        let ids = |c: RouteConstraints| -> Vec<String> {
            selector(c)
                .feasible(&offers)
                .iter()
                .map(|o| o.id().as_str().to_owned())
                .collect()
        };
        let (a, b, c) = (ids(tighter), ids(tight), ids(loose));
        assert!(a.iter().all(|id| b.contains(id)));
        assert!(b.iter().all(|id| c.contains(id)));
    }
}
