use ptr_analytics::{
    clopper_pearson_upper, weighted_rate, wilson_interval, Metric, MetricRow, WeightedOutcome,
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
