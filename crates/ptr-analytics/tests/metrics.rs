use ptr_analytics::{
    binomial_cdf, brier_score, clopper_pearson_upper, expected_calibration_error,
    krippendorff_alpha_nominal, weighted_rate, wilson_interval, Binning, Metric, MetricRow,
    RunningMoments, StatsError, WeightedOutcome,
};

#[test]
fn a_metric_row_turns_into_an_interval_that_contains_its_point() {
    let row = MetricRow {
        group: String::new(),
        numerator: 12,
        denominator: 80,
    };
    let rate = wilson_interval(row.numerator, row.denominator, 1.96).unwrap();
    assert!(rate.low <= rate.point && rate.point <= rate.high);
    // The one-sided exact bound at the same confidence is above the point.
    let upper = clopper_pearson_upper(row.numerator, row.denominator, 0.025).unwrap();
    assert!(upper > rate.point);
    assert_eq!(Metric::AdjudicatedHarmRate.name(), "adjudicated_harm_rate");
}

#[test]
fn scaling_all_importance_weights_preserves_the_estimate_and_interval() {
    let observations = [
        WeightedOutcome {
            weight: 1.0,
            success: false,
        },
        WeightedOutcome {
            weight: 3.0,
            success: true,
        },
    ];
    let base = weighted_rate(&observations, 1.96).unwrap();
    let scaled = observations.map(|o| WeightedOutcome {
        weight: o.weight * 8.0,
        ..o
    });
    assert_eq!(weighted_rate(&scaled, 1.96).unwrap(), base);
    assert_eq!(base.estimate, 0.75);
    // Dividing by the largest weight rounds the others once.
    assert!(
        (base.effective_n - 1.6).abs() < 1e-12,
        "{}",
        base.effective_n
    );
}

#[test]
fn weighted_rates_reject_empty_samples_and_invalid_weights_or_quantiles() {
    assert_eq!(
        weighted_rate(&[], 1.96),
        Err(StatsError::Empty {
            field: "observations"
        })
    );
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            weighted_rate(
                &[WeightedOutcome {
                    weight: invalid,
                    success: true
                }],
                1.96
            ),
            Err(StatsError::InvalidParameter {
                field: "weight",
                ..
            })
        ));
        assert!(matches!(
            weighted_rate(
                &[WeightedOutcome {
                    weight: 1.0,
                    success: true
                }],
                invalid
            ),
            Err(StatsError::InvalidParameter { field: "z", .. })
        ));
    }
}

#[test]
fn empty_binomial_samples_distinguish_an_undefined_rate_from_a_conservative_bound() {
    assert_eq!(
        wilson_interval(0, 0, 1.96),
        Err(StatsError::Empty { field: "trials" })
    );
    assert_eq!(clopper_pearson_upper(0, 0, 0.05), Ok(1.0));
    for invalid in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            clopper_pearson_upper(0, 0, invalid),
            Err(StatsError::InvalidParameter { field: "delta", .. })
        ));
    }
    assert_eq!(
        wilson_interval(2, 1, 1.96),
        Err(StatsError::InvalidCount {
            successes: 2,
            trials: 1
        })
    );
    assert_eq!(
        clopper_pearson_upper(2, 1, 0.05),
        Err(StatsError::InvalidCount {
            successes: 2,
            trials: 1
        })
    );
}

#[test]
fn calibration_handles_more_bins_than_items_and_confidence_one() {
    let predictions = [vec![0.75, 0.25], vec![0.0, 1.0]];
    // Squared distances: 0.125 and 2; calibration gaps: 0.25 and 1.
    assert_eq!(brier_score(&predictions, &[0, 0]), Ok(1.0625));
    for binning in [Binning::EqualMass(5), Binning::EqualWidth(5)] {
        assert_eq!(
            expected_calibration_error(&predictions, &[0, 0], binning),
            Ok(0.625)
        );
    }
}

#[test]
fn calibration_reports_the_invalid_item_and_alignment_errors() {
    for bad in [
        vec![],
        vec![0.5, -0.5, 1.0],
        vec![f64::NAN, 0.0],
        vec![f64::INFINITY, 0.0],
        vec![0.2, 0.2],
        vec![1.0000005, 0.0],
    ] {
        let predictions = [vec![1.0, 0.0], bad];
        assert_eq!(
            brier_score(&predictions, &[0, 0]),
            Err(StatsError::InvalidDistribution { item: 1 })
        );
        assert_eq!(
            expected_calibration_error(&predictions, &[0, 0], Binning::EqualMass(2)),
            Err(StatsError::InvalidDistribution { item: 1 })
        );
    }
    assert_eq!(
        brier_score(&[], &[]),
        Err(StatsError::Empty { field: "truth" })
    );
    assert_eq!(
        brier_score(&[], &[0]),
        Err(StatsError::LengthMismatch {
            expected: 1,
            actual: 0
        })
    );
    assert_eq!(
        brier_score(&[vec![1.0, 0.0]], &[2]),
        Err(StatsError::InvalidDistribution { item: 0 })
    );
    for bins in [Binning::EqualMass(0), Binning::EqualWidth(0)] {
        assert!(matches!(
            expected_calibration_error(&[vec![1.0, 0.0]], &[0], bins),
            Err(StatsError::InvalidParameter { field: "bins", .. })
        ));
    }
}

#[test]
fn empty_moment_summaries_are_merge_identities_and_one_sample_has_no_variance() {
    let empty = RunningMoments::default();
    let mut one = empty;
    one.push(17.0).unwrap();
    assert_eq!(one.count(), 1);
    assert_eq!(one.mean(), Some(17.0));
    assert_eq!(one.sample_variance(), None);
    assert_eq!(empty.merge(&one), Ok(one));
    assert_eq!(one.merge(&empty), Ok(one));
    let two = one.merge(&one).unwrap();
    assert_eq!(two.count(), 2);
    assert_eq!(two.mean(), Some(17.0));
    assert_eq!(two.sample_variance(), Some(0.0));
}

#[test]
fn agreement_can_be_negative_and_unpaired_ratings_add_no_evidence() {
    let mut units = vec![vec![Some(0), Some(1)], vec![Some(0), Some(1)]];
    let expected = -0.5;
    assert_eq!(krippendorff_alpha_nominal(&units), Some(expected));
    units.extend([vec![Some(99), None], vec![None, None], vec![]]);
    assert_eq!(krippendorff_alpha_nominal(&units), Some(expected));
    assert_eq!(krippendorff_alpha_nominal(&[vec![Some(1)], vec![]]), None);
}

#[test]
fn agreement_is_undefined_for_a_single_category_whatever_the_unit_sizes() {
    // Four raters weight each pair by 1/3, which no binary float holds exactly.
    let units = vec![
        vec![Some(7); 4],
        vec![Some(7); 3],
        vec![Some(7); 4],
        vec![Some(7), None],
    ];
    assert_eq!(krippendorff_alpha_nominal(&units), None);
}

#[test]
fn the_clopper_pearson_bound_holds_for_a_confidence_below_one_half() {
    // One success in two trials: P(X <= 1) = 1 - p^2, so the bound at
    // delta = 0.9 is sqrt(0.1), below the observed rate of 0.5.
    let bound = clopper_pearson_upper(1, 2, 0.9).unwrap();
    assert!((bound - 0.1_f64.sqrt()).abs() < 1e-9, "{bound}");
    assert!((binomial_cdf(1, 2, bound).unwrap() - 0.9).abs() < 1e-9);
    // Every accepted delta yields the root, above and below the observed rate.
    for delta in [0.01, 0.25, 0.5, 0.75, 0.99] {
        let bound = clopper_pearson_upper(3, 10, delta).unwrap();
        assert!(
            (binomial_cdf(3, 10, bound).unwrap() - delta).abs() < 1e-9,
            "{delta}"
        );
    }
}

#[test]
fn a_calibration_slice_reweighted_by_its_rate_estimates_the_population_rate() {
    // 1 in 10 eligible branches is audited (weight 10); audited branches are
    // harmful 2 times in 20.
    let audited: Vec<_> = (0..20)
        .map(|i| WeightedOutcome {
            weight: 10.0,
            success: i < 2,
        })
        .collect();
    let rate = weighted_rate(&audited, 1.96).unwrap();
    assert!((rate.estimate - 0.1).abs() < 1e-12);
    assert!((rate.effective_n - 20.0).abs() < 1e-9);
}

#[test]
fn importance_weights_at_either_end_of_the_float_range_give_the_rate_of_a_moderate_copy() {
    let moderate = [
        WeightedOutcome {
            weight: 1.0,
            success: false,
        },
        WeightedOutcome {
            weight: 3.0,
            success: true,
        },
    ];
    let expected = weighted_rate(&moderate, 1.96).unwrap();
    // Exact power-of-two copies: the sum of the huge pair overflows, the
    // squares of the tiny pair underflow to zero.
    let tiny = f64::MIN_POSITIVE * 0.5_f64.powi(48);
    for factor in [2.0_f64.powi(1022), tiny] {
        let copy = moderate.map(|o| WeightedOutcome {
            weight: o.weight * factor,
            ..o
        });
        assert_eq!(weighted_rate(&copy, 1.96), Ok(expected), "{factor:e}");
    }
    // One maximal weight is one effective observation, like any single weight.
    let single = |weight| {
        weighted_rate(
            &[WeightedOutcome {
                weight,
                success: true,
            }],
            1.96,
        )
    };
    let maximal = single(f64::MAX).unwrap();
    assert_eq!(maximal, single(1.0).unwrap());
    assert_eq!((maximal.estimate, maximal.effective_n), (1.0, 1.0));
    assert!(maximal.low > 0.0 && maximal.high == 1.0);
}

#[test]
fn a_quantile_too_large_for_a_finite_interval_is_refused() {
    // The square of this z overflows; the clamps to [0, 1] used to turn the
    // resulting NaN into a vacuous interval.
    let refused = Err(StatsError::InvalidParameter {
        field: "z",
        message: "is too large for a finite interval",
    });
    assert_eq!(wilson_interval(8, 10, 1e200).map(|_| ()), refused);
    assert_eq!(
        weighted_rate(
            &[WeightedOutcome {
                weight: 1.0,
                success: true
            }],
            1e200
        )
        .map(|_| ()),
        refused
    );
    // A large z whose square is finite still gives an interval.
    let wide = wilson_interval(8, 10, 1e150).unwrap();
    assert!(wide.low <= wide.point && wide.point <= wide.high);
}

#[test]
fn a_moment_summary_refuses_what_would_overflow_it_and_stays_unchanged() {
    let mut moments = RunningMoments::default();
    moments.push(f64::MAX).unwrap();
    let before = moments;
    // The deviation of -MAX from MAX is not finite; nor is a NaN or infinity.
    for value in [-f64::MAX, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            moments.push(value),
            Err(StatsError::InvalidParameter {
                field: "value",
                message: "would make the mean or the sum of squared deviations non-finite",
            }),
            "{value}"
        );
        assert_eq!(moments, before);
    }
    // Finite values whose squared deviations overflow the sum of squares.
    let mut high = RunningMoments::default();
    high.push(1e200).unwrap();
    let mut low = RunningMoments::default();
    low.push(-1e200).unwrap();
    assert_eq!(
        high.merge(&low),
        Err(StatsError::InvalidParameter {
            field: "summary",
            message: "would make the mean or the sum of squared deviations non-finite",
        })
    );
    let unchanged = high;
    assert!(high.push(-1e200).is_err());
    assert_eq!(high, unchanged);
    // Values within range are still summarised.
    moments.push(f64::MAX).unwrap();
    assert_eq!(moments.mean(), Some(f64::MAX));
    assert_eq!(moments.sample_variance(), Some(0.0));
}

#[test]
fn merging_summaries_is_refused_only_when_the_merged_summary_itself_would_overflow() {
    // Chan's cross term delta^2 * n1 * n2 / n is finite here, but the product
    // delta^2 * n1 * n2 alone is not: 1e304 * 1e6 in the first case, and
    // delta^2 = 2.25e308 before the division by two in the second.
    for (left_values, right_values) in [
        (vec![0.0; 1000], vec![1e152; 1000]),
        (vec![0.0], vec![1.5e154]),
    ] {
        let mut all = RunningMoments::default();
        let mut left = RunningMoments::default();
        let mut right = RunningMoments::default();
        for &value in &left_values {
            all.push(value).unwrap();
            left.push(value).unwrap();
        }
        for &value in &right_values {
            all.push(value).unwrap();
            right.push(value).unwrap();
        }
        for merged in [left.merge(&right), right.merge(&left)] {
            let merged = merged.unwrap();
            assert_eq!(merged.count(), all.count());
            let (mean, expected_mean) = (merged.mean().unwrap(), all.mean().unwrap());
            assert!(
                (mean - expected_mean).abs() <= 1e-12 * expected_mean,
                "{mean}"
            );
            let (variance, expected_variance) = (
                merged.sample_variance().unwrap(),
                all.sample_variance().unwrap(),
            );
            assert!(variance.is_finite());
            assert!(
                (variance - expected_variance).abs() <= 1e-9 * expected_variance,
                "{variance} {expected_variance}"
            );
        }
    }
}

#[test]
fn a_moment_summary_refuses_to_count_past_u64_max_and_stays_unchanged() {
    // Doubling one value 63 times counts 2^63 values; adding every smaller
    // power of two on the way counts u64::MAX.
    let mut one = RunningMoments::default();
    one.push(1.0).unwrap();
    let (mut power, mut full) = (one, one);
    for _ in 0..63 {
        power = power.merge(&power).unwrap();
        full = full.merge(&power).unwrap();
    }
    assert_eq!(power.count(), 1 << 63);
    assert_eq!(full.count(), u64::MAX);
    let overflow = |field| StatsError::InvalidParameter {
        field,
        message: "would count more values than a u64 holds",
    };
    assert_eq!(power.merge(&power), Err(overflow("summary")));
    assert_eq!(full.merge(&one), Err(overflow("summary")));
    assert_eq!(one.merge(&full), Err(overflow("summary")));
    let before = full;
    assert_eq!(full.push(1.0), Err(overflow("value")));
    assert_eq!(full, before);
    assert_eq!(overflow("value").code(), "PTR_STATS_INVALID_PARAMETER");
    // The full summary is still exact and still merges with an empty one.
    assert_eq!(full.mean(), Some(1.0));
    assert_eq!(full.sample_variance(), Some(0.0));
    assert_eq!(full.merge(&RunningMoments::default()), Ok(full));
}

#[test]
fn the_binomial_cdf_refuses_a_p_that_is_not_a_probability() {
    // Clamped to an endpoint, each of these would answer a confident 0 or 1.
    for p in [f64::NAN, -0.5, 1.5, f64::NEG_INFINITY, f64::INFINITY] {
        for k in [0, 3, 10] {
            assert_eq!(
                binomial_cdf(k, 10, p),
                Err(StatsError::InvalidParameter {
                    field: "p",
                    message: "must lie in [0, 1]",
                }),
                "{p} {k}"
            );
        }
    }
    // The endpoints are probabilities.
    assert_eq!(binomial_cdf(3, 10, 0.0), Ok(1.0));
    assert_eq!(binomial_cdf(3, 10, 1.0), Ok(0.0));
    assert_eq!(binomial_cdf(10, 10, 1.0), Ok(1.0));
}

#[test]
fn no_successes_at_a_p_too_small_to_invert_have_probability_one_not_nan() {
    // Below 1 / f64::MAX (about 5.56e-309) the ratio (1 - p) / p overflows.
    // For k = 0 no ratio is summed, yet it was formed anyway, and its
    // infinite quotient with a zero low part made the value NaN.
    let tiny = [
        f64::from_bits(1),
        1e-310,
        5.5e-309,
        5.56e-309,
        f64::MIN_POSITIVE,
        1e-300,
    ];
    for p in tiny {
        for n in [1, 10, 1_000_000, u64::MAX] {
            // (1 - p)^n = exp(n ln(1 - p)) rounds to one for all of these.
            assert_eq!(binomial_cdf(0, n, p), Ok(1.0), "0 of {n} at {p:e}");
            for k in [1, 2] {
                if k < n {
                    assert_eq!(binomial_cdf(k, n, p), Ok(1.0), "{k} of {n} at {p:e}");
                }
            }
        }
    }
    // Where n p is not negligible the one term is still (1 - p)^n.
    let p = 1e-300;
    let n = u64::MAX;
    assert_eq!(binomial_cdf(0, n, p), Ok((n as f64 * (-p).ln_1p()).exp()));
}

#[test]
fn a_bin_count_beyond_memory_is_computed_on_the_occupied_bins_alone() {
    let predictions = [vec![0.75, 0.25], vec![0.0, 1.0], vec![0.6, 0.4]];
    let truth = [0, 0, 1];
    // Each prediction in a bin of its own: gaps 0.6, 0.25 and 1 by confidence.
    let singletons =
        expected_calibration_error(&predictions, &truth, Binning::EqualMass(3)).unwrap();
    assert!((singletons - 1.85 / 3.0).abs() < 1e-12, "{singletons}");
    for count in [4, 1 << 60, usize::MAX] {
        assert_eq!(
            expected_calibration_error(&predictions, &truth, Binning::EqualMass(count)),
            Ok(singletons),
            "{count}"
        );
    }
    for count in [1 << 60, usize::MAX] {
        assert_eq!(
            expected_calibration_error(&predictions, &truth, Binning::EqualWidth(count)),
            Ok(singletons),
            "{count}"
        );
    }
    assert_eq!(
        expected_calibration_error(&[vec![1.0, 0.0]], &[0], Binning::EqualWidth(usize::MAX)),
        Ok(0.0)
    );
}

#[test]
fn a_wilson_interval_contains_its_point_with_no_successes_or_only_successes() {
    // At 0 and at n successes the Wilson interval touches its point in exact
    // arithmetic; the centre and half-width used to round apart and leave
    // the point one rounding outside (6 of 6 and 0 of 11 at z = 1.96).
    for z in [1.0, 1.645, 1.96, 2.576] {
        for trials in 1..=3000 {
            let none = wilson_interval(0, trials, z).unwrap();
            let all = wilson_interval(trials, trials, z).unwrap();
            assert_eq!((none.point, none.low), (0.0, 0.0), "0/{trials} at {z}");
            assert!(none.high > 0.0, "0/{trials} at {z}");
            assert_eq!(
                (all.point, all.high),
                (1.0, 1.0),
                "{trials}/{trials} at {z}"
            );
            assert!(all.low < 1.0, "{trials}/{trials} at {z}");
        }
    }
    assert_eq!(wilson_interval(6, 6, 1.96).unwrap().high, 1.0);
    assert_eq!(wilson_interval(0, 11, 1.96).unwrap().low, 0.0);
    assert_eq!(wilson_interval(0, 65, 1.0).unwrap().low, 0.0);
    // Interior points stay strictly inside.
    for trials in 2..=400 {
        for successes in 1..trials {
            let rate = wilson_interval(successes, trials, 1.96).unwrap();
            assert!(
                rate.low < rate.point && rate.point < rate.high,
                "{successes}/{trials}: {rate:?}"
            );
        }
    }
}

#[test]
fn a_weighted_rate_with_no_successes_or_only_successes_contains_its_estimate() {
    for count in 1..=200 {
        for weight in [1.0, 3.0, 0.1] {
            let observations = |success| vec![WeightedOutcome { weight, success }; count];
            let none = weighted_rate(&observations(false), 1.96).unwrap();
            let all = weighted_rate(&observations(true), 1.96).unwrap();
            assert_eq!((none.estimate, none.low), (0.0, 0.0), "{count} x {weight}");
            assert_eq!((all.estimate, all.high), (1.0, 1.0), "{count} x {weight}");
        }
    }
}

#[test]
fn a_clopper_pearson_bound_on_millions_of_trials_with_one_failure_solves_the_closed_form() {
    // P(X <= n - 1) = 1 - p^n, so the bound is (1 - delta)^(1 / n). Summed
    // from zero, that tail has n terms, and the bisection evaluates it 100
    // times.
    let trials: u64 = 5_000_000;
    for delta in [0.05, 0.5, 0.95] {
        let bound = clopper_pearson_upper(trials - 1, trials, delta).unwrap();
        let exact = ((-delta).ln_1p() / trials as f64).exp();
        assert!((bound - exact).abs() < 1e-12, "{delta}: {bound} {exact}");
    }
    let p = 1.0 - 1e-7;
    let tail = binomial_cdf(trials - 1, trials, p).unwrap();
    assert!(
        (tail - (1.0 - p.powf(trials as f64))).abs() < 1e-12,
        "{tail}"
    );
}

#[test]
fn binomial_tails_of_millions_of_trials_match_references_computed_at_fifty_digits() {
    // (k, n, p, P(X <= k)): each reference is the exact tail at the f64 `p`,
    // summed at 50 significant digits and rounded. For an even n at p = 1/2
    // it is also 1/2 + P(X = n/2)/2.
    let references = [
        (5_000_000, 10_000_000, 0.5, 0.500_126_156_622_947_1),
        (30_000, 3_000_000, 0.01, 0.501_535_542_400_828_6),
        (29_500, 3_000_000, 0.01, 0.001_833_567_565_795_332_4),
        (31_000, 3_000_000, 0.01, 0.999_999_996_150_658_3),
        (12, 2_000_000, 1e-5, 0.039_011_287_848_817_69),
    ];
    for (k, n, p, reference) in references {
        let tail = binomial_cdf(k, n, p).unwrap();
        assert!(
            (tail - reference).abs() <= 1e-13 * reference,
            "{k} of {n} at {p}: {tail}"
        );
    }
    // The bound at the centre is the root of its own tail.
    for delta in [0.05, 0.5] {
        let bound = clopper_pearson_upper(5_000_000, 10_000_000, delta).unwrap();
        let tail = binomial_cdf(5_000_000, 10_000_000, bound).unwrap();
        assert!((tail - delta).abs() < 1e-9, "{delta}: {bound} {tail}");
    }
}

#[test]
fn a_binomial_tail_near_the_largest_count_is_summed_on_the_side_its_exact_count_picks() {
    // With p = 1 - 2^-53, failures number about 2048 in u64::MAX trials. As
    // an f64, u64::MAX - 3000 rounds to the mean, which used to pick the
    // upper tail and return the rounding of its complement instead of a
    // tail near 1e-86. References are exact tails summed at 80 digits.
    let (n, p, q) = (u64::MAX, 1.0 - 2.0_f64.powi(-53), 2.0_f64.powi(-53));
    let references = [
        (n - 3000, p, 2.718_230_685_978_389e-86),
        (n - 2100, p, 0.127_780_095_439_163_77),
        (n - 2000, p, 0.858_203_579_962_535_4),
        (1500, q, 2.804_905_285_645_475_3e-37),
        (2000, q, 0.146_856_473_677_153_38),
    ];
    for (k, p, reference) in references {
        let tail = binomial_cdf(k, n, p).unwrap();
        assert!(
            (tail - reference).abs() <= 1e-12 * reference,
            "{k} at {p}: {tail}"
        );
    }
}

#[test]
fn binomial_tails_at_ordinary_probabilities_stay_within_the_documented_error() {
    // (k, n, p, P(X <= k)): exact tails at the f64 `p`, summed at 70 digits
    // and rounded to the nearest f64. At p = 0.3 or 0.45 the rounding of
    // 1 - p, of the ratio q / p reused by every term and of n p and n q
    // used to compound, to about 1e-13 absolute near the centre and 8e-12
    // relative in the far tails; the rustdoc of binomial_cdf states the
    // figures these must meet.
    let centre = [
        (4_500_000, 10_000_000, 0.45, 0.500_131_018_582_184),
        (3_000_000, 10_000_000, 0.3, 0.500_156_001_245_883_2),
        (
            3_414_944,
            10_000_000,
            0.341_449_488_349_846_06,
            0.617_861_286_993_527_5,
        ),
        (1_350_000, 3_000_000, 0.45, 0.500_239_206_094_156_2),
    ];
    for (k, n, p, reference) in centre {
        let tail = binomial_cdf(k, n, p).unwrap();
        assert!(
            (tail - reference).abs() < 1e-14,
            "{k} of {n} at {p}: {tail}"
        );
    }
    let far = [
        (1_318_118, 3_000_000, 0.45, 2.108_970_683_372_145_2e-300),
        (
            3_273_683,
            10_000_000,
            0.332_732_453_339_112_33,
            7.208_910_012_437_18e-285,
        ),
        (1_332_766, 3_000_000, 0.45, 2.358_265_733_900_953e-89),
    ];
    for (k, n, p, reference) in far {
        let tail = binomial_cdf(k, n, p).unwrap();
        assert!(
            (tail - reference).abs() < 5e-13 * reference,
            "{k} of {n} at {p}: {tail}"
        );
    }
}
