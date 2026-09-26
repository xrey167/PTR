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
    assert!((binomial_cdf(1, 2, bound) - 0.9).abs() < 1e-9);
    // Every accepted delta yields the root, above and below the observed rate.
    for delta in [0.01, 0.25, 0.5, 0.75, 0.99] {
        let bound = clopper_pearson_upper(3, 10, delta).unwrap();
        assert!((binomial_cdf(3, 10, bound) - delta).abs() < 1e-9, "{delta}");
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
