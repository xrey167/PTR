//! Estimator tests on synthetic logs with a known value. They build logs with
//! `LoggedDecision::replayed`, which is crate-private (a caller outside the crate cannot build a
//! log of its own), so they live here and not in `tests/`.

use ptr_types::PodId;

use crate::findings::{AdmissibleScorer, Exploration, FindingCode, RouteClass, ScoringContext};
use crate::learning::{LoggedDecision, RewardModel, VerifierReward};
use crate::router::{RouteDistribution, RouteError};
use crate::{compare_policies, estimate_policy, Confidence, Estimator, OpeError};

fn pod(id: &str) -> PodId {
    PodId::from(id)
}

fn repair_class() -> RouteClass {
    RouteClass::Repair(FindingCode::new("wrong_answer"))
}

/// A seeded generator for the simulated world (not the router's).
struct World(u64);

impl World {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

// --- the estimators -------------------------------------------------------------------------

/// A target policy that prefers its weights as given.
struct Fixed(Vec<(&'static str, f64)>);

impl AdmissibleScorer for Fixed {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        Ok(context
            .admissible
            .iter()
            .map(|p| {
                let w = self
                    .0
                    .iter()
                    .find(|(n, _)| *n == p.0)
                    .map_or(0.0, |(_, w)| *w);
                (p.clone(), w)
            })
            .collect())
    }
}

struct Constant(f64);

impl RewardModel for Constant {
    fn predict(&self, _: &RouteClass, _: &PodId) -> f64 {
        self.0
    }
}

struct Truth;

impl RewardModel for Truth {
    fn predict(&self, _: &RouteClass, p: &PodId) -> f64 {
        match p.0.as_str() {
            "a" => 0.2,
            "b" => 0.5,
            _ => 0.8,
        }
    }
}

fn synthetic_log(n: usize, world: &mut World) -> Vec<LoggedDecision> {
    let logging =
        RouteDistribution::new(vec![(pod("a"), 0.5), (pod("b"), 0.3), (pod("c"), 0.2)]).unwrap();
    let support: Vec<(PodId, u64)> = ["a", "b", "c"]
        .iter()
        .map(|p| (pod(p), logging.width(&pod(p))))
        .collect();
    let floor = Exploration::new(0.1).unwrap();
    (0..n)
        .map(|i| {
            let draw = world.next() >> 11;
            let mut upper = 0;
            let mut chosen = pod("c");
            for (p, width) in &support {
                upper += width;
                if draw < upper {
                    chosen = p.clone();
                    break;
                }
            }
            let truth = Truth.predict(&repair_class(), &chosen);
            LoggedDecision::replayed(
                i as u64,
                repair_class(),
                support.clone(),
                chosen,
                VerifierReward::replayed(world.unit() < truth),
                floor,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn ips_snips_and_dr_are_unbiased_on_a_log_with_known_value_and_cover() {
    // Target mixture at epsilon 0.1 over weights (0.1, 0.2, 0.7): pi = 0.9 w + 0.1 / 3.
    let target = Fixed(vec![("a", 0.1), ("b", 0.2), ("c", 0.7)]);
    let pi = [
        0.9 * 0.1 + 0.1 / 3.0,
        0.9 * 0.2 + 0.1 / 3.0,
        0.9 * 0.7 + 0.1 / 3.0,
    ];
    let truth = pi[0] * 0.2 + pi[1] * 0.5 + pi[2] * 0.8;
    let reps = 40;
    let mut world = World(77);
    let mut sums = [0.0_f64; 3];
    let mut squares = [0.0_f64; 3];
    let mut covered = [0_usize; 3];
    let wrong_model = Constant(0.5);
    for _ in 0..reps {
        let log = synthetic_log(3000, &mut world);
        // DR with a deliberately wrong reward model is still unbiased: the propensities are right.
        let estimate = estimate_policy(&log, &target, &wrong_model).unwrap();
        let values = [estimate.ips, estimate.snips, estimate.doubly_robust];
        let errors = [
            estimate.se_ips,
            estimate.se_snips,
            estimate.se_doubly_robust,
        ];
        for k in 0..3 {
            sums[k] += values[k];
            squares[k] += values[k] * values[k];
            if (values[k] - truth).abs() <= 1.96 * errors[k] {
                covered[k] += 1;
            }
        }
        assert!(estimate.effective_sample_size > 1000.0);
    }
    for k in 0..3 {
        let mean = sums[k] / reps as f64;
        let variance = squares[k] / reps as f64 - mean * mean;
        let se_of_mean = (variance / reps as f64).sqrt();
        assert!(
            (mean - truth).abs() < 3.0 * se_of_mean + 1e-3,
            "estimator {k}: mean {mean} truth {truth} se {se_of_mean}"
        );
        assert!(
            covered[k] >= 34,
            "estimator {k} covered {} of {reps}",
            covered[k]
        );
    }
}

#[test]
fn doubly_robust_with_a_good_model_has_a_smaller_error_than_ips() {
    let target = Fixed(vec![("a", 0.1), ("b", 0.2), ("c", 0.7)]);
    let mut world = World(5);
    let log = synthetic_log(3000, &mut world);
    let good = estimate_policy(&log, &target, &Truth).unwrap();
    assert!(good.se_doubly_robust < good.se_ips, "{good:?}");
}

#[test]
fn the_comparison_of_a_policy_with_itself_is_exactly_zero() {
    let target = Fixed(vec![("a", 0.1), ("b", 0.2), ("c", 0.7)]);
    let mut world = World(8);
    let log = synthetic_log(500, &mut world);
    for estimator in [Estimator::Ips, Estimator::Snips, Estimator::DoublyRobust] {
        let c = compare_policies(
            &log,
            &target,
            &target,
            &Truth,
            estimator,
            Confidence::ONE_SIDED_95,
        )
        .unwrap();
        assert!(c.difference.abs() < 1e-12);
        assert!(c.difference_se.abs() < 1e-12);
    }
}

#[test]
fn estimation_refuses_what_it_cannot_trust() {
    let target = Fixed(vec![("a", 1.0), ("b", 1.0), ("c", 1.0)]);
    assert_eq!(
        estimate_policy(&[], &target, &Truth).unwrap_err(),
        OpeError::EmptyLog
    );
    let mut world = World(1);
    let log = synthetic_log(50, &mut world);

    // A policy that names a Pod outside the logged admissible set.
    struct Escaping;
    impl AdmissibleScorer for Escaping {
        fn weights(&self, _: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
            Ok(vec![(PodId::from("zzz"), 1.0)])
        }
    }
    assert!(matches!(
        estimate_policy(&log, &Escaping, &Truth).unwrap_err(),
        OpeError::PolicyOutsideAdmissibleSet { .. }
    ));

    // A policy with no mass at all.
    struct Silent;
    impl AdmissibleScorer for Silent {
        fn weights(&self, _: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
            Ok(Vec::new())
        }
    }
    assert!(matches!(
        estimate_policy(&log, &Silent, &Truth).unwrap_err(),
        OpeError::PolicyFailed { .. }
    ));

    // A reward model outside [0, 1].
    assert_eq!(
        estimate_policy(&log, &target, &Constant(1.5)).unwrap_err(),
        OpeError::RewardModelOutOfRange
    );
    assert_eq!(
        estimate_policy(&log, &target, &Constant(f64::NAN)).unwrap_err(),
        OpeError::RewardModelOutOfRange
    );

    // A log whose logger had no floor: a propensity far below epsilon / n.
    let support = vec![
        (pod("a"), 1_u64 << 20),
        (pod("b"), (1_u64 << 53) - (1 << 20)),
    ];
    let thin = LoggedDecision::replayed(
        0,
        repair_class(),
        support,
        pod("a"),
        VerifierReward::NOT_CLEARED,
        Exploration::new(0.1).unwrap(),
    )
    .unwrap();
    assert_eq!(
        estimate_policy(&[thin], &target, &Truth).unwrap_err(),
        OpeError::BelowFloor { seq: 0 }
    );
    // A log without any exploration floor cannot even be built.
    assert!(LoggedDecision::replayed(
        0,
        repair_class(),
        vec![(pod("a"), 1_u64 << 53)],
        pod("a"),
        VerifierReward::NOT_CLEARED,
        Exploration::NONE,
    )
    .is_err());
}
