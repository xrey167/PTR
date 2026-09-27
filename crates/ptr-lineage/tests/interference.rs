use ptr_lineage::{
    activation_interference, chance_overlap, column_basis, measure_interference, subspace_overlap,
    AdapterId, InterferenceReport, LayerInterference, LayerUpdate, LineageError, Matrix,
};

/// A rank-1 update writing along output axis `out_axis` and reading input axis
/// `in_axis` of a 4x4 layer.
fn rank_one(layer: &str, out_axis: usize, in_axis: usize) -> LayerUpdate {
    let mut b = vec![0.0; 4];
    b[out_axis] = 1.0;
    let mut a = vec![0.0; 4];
    a[in_axis] = 2.0;
    LayerUpdate::new(
        layer,
        Matrix::new(4, 1, b).unwrap(),
        Matrix::new(1, 4, a).unwrap(),
    )
    .unwrap()
}

fn candidate() -> AdapterId {
    AdapterId::from("candidate")
}

#[test]
fn adapters_on_orthogonal_subspaces_do_not_interfere() {
    let earlier = vec![(AdapterId::from("a1"), vec![rank_one("q", 0, 0)])];
    let report = measure_interference(&candidate(), &[rank_one("q", 1, 1)], &earlier).unwrap();
    assert!(report.max_overlap() < 1e-12);
    assert!(report.within(0.01));
}

#[test]
fn sharing_an_output_direction_is_full_output_overlap_and_names_the_culprit() {
    let earlier = vec![
        (AdapterId::from("a1"), vec![rank_one("q", 2, 0)]),
        (AdapterId::from("a2"), vec![rank_one("q", 1, 3)]),
    ];
    let report = measure_interference(&candidate(), &[rank_one("q", 1, 1)], &earlier).unwrap();
    let layer = &report.layers[0];
    assert!((layer.output_overlap - 1.0).abs() < 1e-12);
    assert!(layer.input_overlap.abs() < 1e-12);
    assert_eq!(layer.worst, Some(AdapterId::from("a2")));
    assert!(!report.within(0.5));
}

#[test]
fn a_report_names_the_candidate_it_was_measured_for() {
    let earlier = vec![(AdapterId::from("a1"), vec![rank_one("q", 1, 1)])];
    let updates = [rank_one("q", 1, 1)];
    for id in ["c1", "c2"] {
        let report = measure_interference(&AdapterId::from(id), &updates, &earlier).unwrap();
        assert_eq!(report.candidate, AdapterId::from(id));
        assert_eq!(report.layers[0].worst, Some(AdapterId::from("a1")));
    }
}

#[test]
fn different_layers_never_interfere() {
    let earlier = vec![(AdapterId::from("a1"), vec![rank_one("k", 1, 1)])];
    let report = measure_interference(&candidate(), &[rank_one("q", 1, 1)], &earlier).unwrap();
    assert!(report.max_overlap() < 1e-12);
    assert_eq!(report.layers[0].worst, None);
}

#[test]
fn a_rank_mismatch_between_b_and_a_is_refused() {
    let b = Matrix::new(4, 2, vec![0.0; 8]).unwrap();
    let a = Matrix::new(1, 4, vec![0.0; 4]).unwrap();
    assert!(LayerUpdate::new("q", b, a).is_err());
}

#[test]
fn a_rank_deficient_update_is_measured_on_its_product_not_its_factors() {
    // B = [e1, 0] and A = [e1; e2]: delta_W = e1 e1^T reads only e1, although
    // row(A) is the plane {e1, e2}.
    let b = Matrix::new(3, 2, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let a = Matrix::new(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let candidate = LayerUpdate::new("q_proj", b, a).unwrap();
    assert_eq!(candidate.input_basis().rank(), 1);
    assert_eq!(candidate.output_basis().rank(), 1);
    // An earlier adapter reading {e1, e3} and writing {e2, e3}.
    let earlier_b = Matrix::new(3, 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
    let earlier_a = Matrix::new(2, 3, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let earlier = LayerUpdate::new("q_proj", earlier_b, earlier_a).unwrap();
    let report = measure_interference(
        &AdapterId::from("candidate"),
        &[candidate],
        &[(AdapterId::from("earlier"), vec![earlier])],
    )
    .unwrap();
    let layer = &report.layers[0];
    assert!((layer.input_overlap - 1.0).abs() < 1e-12, "{layer:?}");
    assert!(layer.output_overlap.abs() < 1e-12, "{layer:?}");
}

#[test]
fn factors_that_lose_rank_together_yield_the_rank_of_the_product() {
    // B = [b, b] and A = [a1; a2]: delta_W = b (a1 + a2)^T has rank one.
    let b = Matrix::new(2, 2, vec![1.0, 1.0, 2.0, 2.0]).unwrap();
    let a = Matrix::new(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let update = LayerUpdate::new("v_proj", b, a).unwrap();
    assert_eq!(update.input_basis().rank(), 1);
    assert_eq!(update.output_basis().rank(), 1);
}

fn matrix(rows: usize, cols: usize, data: &[f64]) -> Matrix {
    Matrix::new(rows, cols, data.to_vec()).unwrap()
}

#[test]
fn update_subspaces_are_measured_whatever_the_scale_of_the_factors() {
    // Squared norms of 1e200 overflow and of 1e-170 underflow; a subspace
    // does not depend on the scale of the vectors that span it.
    for scale in [1.0, 1e200, 1e-170, f64::MAX] {
        assert_eq!(
            column_basis(&matrix(2, 1, &[scale, 0.0])).rank(),
            1,
            "{scale}"
        );
        let line = column_basis(&matrix(2, 1, &[scale, scale]));
        assert_eq!(line.rank(), 1, "{scale}");
        assert!((subspace_overlap(&line, &line).unwrap() - 1.0).abs() < 1e-12);
    }
    // A column below the rank tolerance of the largest one is dropped, and
    // the largest one is kept.
    assert_eq!(
        column_basis(&matrix(2, 2, &[1e200, 0.0, 0.0, 1.0])).rank(),
        1
    );
    for scale in [1e200, 1e-170] {
        let update = || {
            LayerUpdate::new("q", matrix(2, 1, &[scale, 0.0]), matrix(1, 2, &[1.0, 0.0])).unwrap()
        };
        let report = measure_interference(
            &candidate(),
            &[update()],
            &[(AdapterId::from("earlier"), vec![update()])],
        )
        .unwrap();
        assert!((report.max_overlap() - 1.0).abs() < 1e-12, "{report:?}");
        assert!(!report.within(0.1));
        for (b, a) in [(1.0, scale), (scale, scale)] {
            // Large on both sides, B A itself (1e400, 1e-340) is out of range.
            let update =
                LayerUpdate::new("q", matrix(2, 1, &[b, 0.0]), matrix(1, 2, &[a, 0.0])).unwrap();
            let (output, input) = (update.output_basis(), update.input_basis());
            assert_eq!((output.rank(), input.rank()), (1, 1), "{b} {a}");
            assert!((subspace_overlap(&output, &output).unwrap() - 1.0).abs() < 1e-12);
            assert!((subspace_overlap(&input, &input).unwrap() - 1.0).abs() < 1e-12);
        }
    }
}

#[test]
fn a_product_entry_beyond_the_largest_finite_value_is_refused_and_a_finite_one_is_computed() {
    let update = LayerUpdate::new("q", matrix(1, 1, &[1e200]), matrix(1, 1, &[1e200])).unwrap();
    assert_eq!(
        update.delta_weight(),
        Err(LineageError::NonFinite {
            field: "matrix product"
        })
    );
    // MAX + MAX - MAX overflows as a running sum, but the entry is MAX.
    let row = matrix(1, 3, &[f64::MAX, f64::MAX, -f64::MAX]);
    let ones = matrix(3, 1, &[1.0, 1.0, 1.0]);
    assert_eq!(row.multiply(&ones).unwrap().as_slice(), &[f64::MAX]);
    assert_eq!(matrix(1, 1, &[1e200]).frobenius(), 1e200);
    assert_eq!(matrix(1, 1, &[1e-200]).frobenius(), 1e-200);
    let big = 2f64.powi(600);
    assert_eq!(matrix(1, 2, &[3.0 * big, 4.0 * big]).frobenius(), 5.0 * big);
    assert_eq!(
        matrix(1, 2, &[f64::MAX, f64::MAX]).frobenius(),
        f64::INFINITY
    );
}

#[test]
fn activation_interference_does_not_depend_on_the_scale_of_updates_or_inputs() {
    let update =
        |value: f64| LayerUpdate::new("q", matrix(1, 1, &[value]), matrix(1, 1, &[value])).unwrap();
    // Equal updates move the output equally: the ratio is one, although the
    // effects overflow (1e100 and 1e200), square to infinity (1e300) or
    // square to zero (1e-300).
    for (value, input) in [(1e100, 1e100), (1e-100, 1e-100), (1e200, 1.0), (1.0, 1.0)] {
        assert_eq!(
            activation_interference(&update(value), &update(value), &matrix(1, 1, &[input])),
            Ok(Some(1.0)),
            "{value} {input}"
        );
    }
    // A ratio that is itself beyond f64::MAX is refused.
    assert_eq!(
        activation_interference(&update(1e200), &update(1e-200), &matrix(1, 1, &[1.0])),
        Err(LineageError::NonFinite {
            field: "activation interference"
        })
    );
    // An earlier update that does not move the inputs gives no ratio.
    let silent = LayerUpdate::new("q", matrix(1, 1, &[0.0]), matrix(1, 1, &[1.0])).unwrap();
    assert_eq!(
        activation_interference(&update(1e200), &silent, &matrix(1, 1, &[1.0])),
        Ok(None)
    );
}

#[test]
fn activation_interference_keeps_an_input_or_factor_entry_far_below_the_largest_of_its_matrix() {
    let reads_second = |weight: f64| {
        LayerUpdate::new("q", matrix(1, 1, &[1.0]), matrix(1, 2, &[0.0, weight])).unwrap()
    };
    // Both updates read only the second input, 1e-30 beside an input of 1e300
    // that neither reads: dividing the activations by the power of two at or
    // below 1e300 would flush it to zero and report no earlier effect at all.
    assert_eq!(
        activation_interference(
            &reads_second(2.0),
            &reads_second(1.0),
            &matrix(2, 1, &[1e300, 1e-30])
        ),
        Ok(Some(2.0))
    );
    // The same when the effects themselves (2e-400 and 1e-400) are below the
    // range of f64, so the direct product is zero as well.
    assert_eq!(
        activation_interference(
            &reads_second(2e-100),
            &reads_second(1e-100),
            &matrix(2, 1, &[1e300, 1e-300])
        ),
        Ok(Some(2.0))
    );
    // A factor entry: the earlier update reaches its output only through the
    // 1e-30 of B = diag(1e300, 1e-30).
    let earlier = LayerUpdate::new(
        "q",
        matrix(2, 2, &[1e300, 0.0, 0.0, 1e-30]),
        matrix(2, 1, &[0.0, 1.0]),
    )
    .unwrap();
    let candidate = LayerUpdate::new("q", matrix(2, 1, &[0.0, 1.0]), matrix(1, 1, &[1.0])).unwrap();
    let ratio = activation_interference(&candidate, &earlier, &matrix(1, 1, &[1.0]))
        .unwrap()
        .unwrap();
    assert!((ratio / 1e30 - 1.0).abs() < 1e-15, "{ratio}");
}

#[test]
fn a_product_entry_whose_large_terms_cancel_keeps_the_small_term_that_decides_it() {
    // MAX + MAX overflows as a running sum, and dividing the row by 2^1023
    // would flush 1e-20 to zero; with an unbounded exponent range the sum
    // returns to zero and ends at 1e-20.
    let row = matrix(1, 5, &[f64::MAX, f64::MAX, -f64::MAX, -f64::MAX, 1e-20]);
    let ones = matrix(5, 1, &[1.0; 5]);
    assert_eq!(row.multiply(&ones).unwrap().as_slice(), &[1e-20]);
}

#[test]
fn a_product_entry_whose_terms_underflow_is_the_sum_they_make_with_an_unbounded_exponent_range() {
    // 2^-1074, the smallest positive subnormal.
    let unit = 5e-324;
    // Each term 1e-162 * 1e-162 = 1e-324 rounds to zero on its own, so the
    // direct product is zero; the in-order sum with an unbounded exponent
    // range is 3e-324, which rounds to the smallest subnormal.
    let row = matrix(1, 3, &[1e-162; 3]);
    let column = matrix(3, 1, &[1e-162; 3]);
    assert_eq!(row.multiply(&column).unwrap().as_slice(), &[unit]);
    // Each term 3 * 2^-540 * 2^-535 = 1.5 units rounds to 2 on its own, so
    // the direct product is 4 units, not the 3 the terms make.
    let row = matrix(1, 2, &[3.0 * 2f64.powi(-540); 2]);
    let column = matrix(2, 1, &[2f64.powi(-535); 2]);
    assert_eq!(row.multiply(&column).unwrap().as_slice(), &[3.0 * unit]);
    // The update is the same one its bases and a consolidation see: not
    // erased by delta_weight while the bases keep it.
    let update = LayerUpdate::new(
        "q",
        matrix(2, 3, &[1e-162, 1e-162, 1e-162, 0.0, 0.0, 0.0]),
        matrix(3, 1, &[1e-162; 3]),
    )
    .unwrap();
    let delta = update.delta_weight().unwrap();
    assert_eq!(delta.as_slice(), &[unit, 0.0]);
    assert_eq!(
        (update.output_basis().rank(), update.input_basis().rank()),
        (1, 1)
    );
    assert_eq!(update.output_basis(), column_basis(&delta));
    assert_eq!(
        ptr_lineage::ties_merge(&[delta.as_slice().to_vec()], 1.0).unwrap(),
        vec![unit, 0.0]
    );
    // An entry of one term is that term rounded once, as the f64 product is:
    // (1 + 2^-52) 2^-537 * (1 - 2^-53) 2^-538 is (1 + 2^-53 - 2^-105) 2^-1075,
    // just above half the smallest subnormal, so it is 5e-324. Rounded first
    // to 53 bits it is 2^-1075, exactly half, which rounds to even: zero.
    let b = (1.0 + f64::EPSILON) * 2f64.powi(-537);
    let a = (1.0 - f64::EPSILON / 2.0) * 2f64.powi(-538);
    assert_eq!(b * a, unit);
    let single = LayerUpdate::new("q", matrix(1, 1, &[b]), matrix(1, 1, &[a])).unwrap();
    assert_eq!(single.b().multiply(single.a()).unwrap().as_slice(), &[unit]);
    let delta = single.delta_weight().unwrap();
    assert_eq!(delta.as_slice(), &[unit]);
    assert_eq!(
        (single.output_basis().rank(), single.input_basis().rank()),
        (1, 1)
    );
    assert_eq!(
        ptr_lineage::ties_merge(&[delta.as_slice().to_vec()], 1.0).unwrap(),
        vec![unit]
    );
    // So is every entry of a rank-1 update built the same way: the f64
    // outer product of its factors.
    let (column, row) = ([b, -b, b], [a, -a, a, a]);
    let rank_one = LayerUpdate::new("q", matrix(3, 1, &column), matrix(1, 4, &row)).unwrap();
    let outer: Vec<f64> = column
        .iter()
        .flat_map(|b| row.iter().map(move |a| b * a))
        .collect();
    assert!(outer.iter().all(|entry| entry.abs() == unit));
    assert_eq!(
        rank_one.delta_weight().unwrap().as_slice(),
        outer.as_slice()
    );
    assert_eq!(
        (
            rank_one.output_basis().rank(),
            rank_one.input_basis().rank()
        ),
        (1, 1)
    );
    // An entry of several terms is their in-order sum rounded once: 2^-1075
    // + 2^-1200 is 2^-1075 at 53 bits, exactly half the smallest subnormal,
    // but the sum itself is above half and rounds to it, and 2^-1075 -
    // 2^-1200 is below half and rounds to zero.
    let row = matrix(1, 2, &[2f64.powi(-537), 2f64.powi(-600)]);
    let column = matrix(2, 1, &[2f64.powi(-538), 2f64.powi(-600)]);
    assert_eq!(row.multiply(&column).unwrap().as_slice(), &[unit]);
    let column = matrix(2, 1, &[2f64.powi(-538), -2f64.powi(-600)]);
    assert_eq!(row.multiply(&column).unwrap().as_slice(), &[0.0]);
    // An interference ratio below the normal range is rounded once, as the
    // f64 quotient of the two effects is: (5 * 2^50 + 7) units over five
    // is 2^50 + 1.4 units, not 2^50 + 2 by way of 2^50 + 1.5.
    let total = (5.0 * 2f64.powi(50) + 7.0) * unit;
    let scalar =
        |value: f64| LayerUpdate::new("q", matrix(1, 1, &[value]), matrix(1, 1, &[1.0])).unwrap();
    assert_eq!(
        activation_interference(&scalar(total), &scalar(5.0), &matrix(1, 1, &[1.0])),
        Ok(Some(total / 5.0))
    );
    assert_eq!(total / 5.0, (2f64.powi(50) + 1.0) * unit);
}

#[test]
fn a_report_with_a_nan_overlap_is_not_within_any_limit() {
    let layer = |overlap: f64| LayerInterference {
        layer: "q".into(),
        output_overlap: overlap,
        input_overlap: 0.0,
        output_chance: 0.0,
        input_chance: 0.0,
        worst: None,
    };
    // Folding with f64::max would drop the NaN and report no overlap.
    let report = InterferenceReport {
        candidate: candidate(),
        layers: vec![layer(0.2), layer(f64::NAN), layer(0.1)],
    };
    assert!(report.max_overlap().is_nan());
    assert!(!report.within(0.1));
    assert!(!report.within(1.0));
    let measured = InterferenceReport {
        candidate: candidate(),
        layers: vec![layer(0.2)],
    };
    assert_eq!(measured.max_overlap(), 0.2);
    assert!(measured.within(0.2));
    assert!(!measured.within(f64::NAN));
}

#[test]
fn a_report_with_an_overlap_outside_the_unit_interval_is_not_within_any_limit() {
    let layer = |output_overlap: f64, input_overlap: f64| LayerInterference {
        layer: "q".into(),
        output_overlap,
        input_overlap,
        output_chance: 0.0,
        input_chance: 0.0,
        worst: None,
    };
    // No measured overlap lies outside [0, 1]. Folded from zero, a negative
    // one would vanish and its layer pass as no overlap at all, and one
    // above one would pass any limit above it.
    for (output, input) in [
        (-0.25, 0.0),
        (0.0, -f64::MIN_POSITIVE),
        (f64::NEG_INFINITY, 0.1),
        (1.5, 0.0),
        (0.0, f64::INFINITY),
    ] {
        let report = InterferenceReport {
            candidate: candidate(),
            layers: vec![layer(0.1, 0.1), layer(output, input)],
        };
        assert!(report.max_overlap().is_nan(), "{output} {input}");
        for limit in [0.1, 1.0, 2.0, f64::INFINITY] {
            assert!(!report.within(limit), "{output} {input} within {limit}");
        }
    }
    // Both ends of the interval are overlaps, and so is a signed zero.
    let ends = InterferenceReport {
        candidate: candidate(),
        layers: vec![layer(0.0, 1.0), layer(-0.0, 0.5)],
    };
    assert_eq!(ends.max_overlap(), 1.0);
    assert!(ends.within(1.0));
}

#[test]
fn activation_interference_compares_two_updates_of_one_layer_only() {
    let inputs = matrix(4, 1, &[1.0; 4]);
    assert_eq!(
        activation_interference(&rank_one("q", 0, 0), &rank_one("q", 0, 0), &inputs),
        Ok(Some(1.0))
    );
    // Same shapes on different layers: the ratio would compare changes to
    // two different outputs, and the activations belong to one of them.
    assert_eq!(
        activation_interference(&rank_one("q", 0, 0), &rank_one("k", 0, 0), &inputs),
        Err(LineageError::LayerMismatch {
            expected: "k".into(),
            actual: "q".into(),
        })
    );
    // One layer name, but outputs of different sizes: not one layer.
    let wider = LayerUpdate::new("q", matrix(5, 1, &[1.0; 5]), matrix(1, 4, &[1.0; 4])).unwrap();
    assert_eq!(
        activation_interference(&wider, &rank_one("q", 0, 0), &inputs),
        Err(LineageError::ShapeMismatch {
            field: "layer outputs",
            expected: 4,
            actual: 5,
        })
    );
}

#[test]
fn an_adapter_that_updates_a_layer_twice_is_refused_before_anything_is_measured() {
    let duplicate = |adapter: &str| {
        Err(LineageError::DuplicateLayer {
            adapter: adapter.into(),
            layer: "q".into(),
        })
    };
    let once = vec![(AdapterId::from("a1"), vec![rank_one("q", 0, 0)])];
    // Each row reports one layer: two updates of "q" would give two rows,
    // neither of them the overlap of the layer's update, their sum.
    assert_eq!(
        measure_interference(
            &candidate(),
            &[rank_one("q", 1, 1), rank_one("q", 2, 2)],
            &once
        ),
        duplicate("candidate")
    );
    // The same for an earlier adapter, within one entry or across two.
    let twice = vec![(
        AdapterId::from("a1"),
        vec![rank_one("q", 0, 0), rank_one("q", 3, 3)],
    )];
    let split = vec![
        (AdapterId::from("a1"), vec![rank_one("q", 0, 0)]),
        (AdapterId::from("a1"), vec![rank_one("q", 3, 3)]),
    ];
    for earlier in [twice, split] {
        assert_eq!(
            measure_interference(&candidate(), &[rank_one("q", 1, 1)], &earlier),
            duplicate("a1")
        );
    }
    // One adapter's layers may be listed across entries, and different
    // adapters update the same layer.
    let spread = vec![
        (AdapterId::from("a1"), vec![rank_one("q", 0, 0)]),
        (AdapterId::from("a1"), vec![rank_one("k", 0, 0)]),
        (AdapterId::from("a2"), vec![rank_one("q", 1, 1)]),
    ];
    let report = measure_interference(
        &candidate(),
        &[rank_one("q", 1, 1), rank_one("k", 2, 2)],
        &spread,
    )
    .unwrap();
    assert_eq!(report.layers.len(), 2);
    assert_eq!(report.layers[0].worst, Some(AdapterId::from("a2")));
}

#[test]
fn chance_overlap_refuses_subspaces_of_different_dimensions() {
    let plane_line = column_basis(&matrix(2, 1, &[1.0, 0.0]));
    let mut axis = vec![0.0; 100];
    axis[0] = 1.0;
    let space_line = column_basis(&matrix(100, 1, &axis));
    let mismatch = |expected, actual| {
        Err(LineageError::ShapeMismatch {
            field: "subspace dimension",
            expected,
            actual,
        })
    };
    assert_eq!(chance_overlap(&plane_line, &space_line), mismatch(2, 100));
    assert_eq!(chance_overlap(&space_line, &plane_line), mismatch(100, 2));
    assert_eq!(subspace_overlap(&plane_line, &space_line), mismatch(2, 100));
    assert_eq!(chance_overlap(&space_line, &space_line), Ok(0.01));
}

#[test]
fn a_layer_update_exposes_its_checked_factors() {
    let b = matrix(2, 1, &[1.0, 2.0]);
    let a = matrix(1, 3, &[3.0, 0.0, -1.0]);
    let update = LayerUpdate::new("q_proj", b.clone(), a.clone()).unwrap();
    assert_eq!(update.layer(), "q_proj");
    assert_eq!((update.b(), update.a()), (&b, &a));
}

#[test]
fn update_bases_keep_a_component_that_cancellation_makes_the_whole_product_depend_on() {
    // B A = [MAX - MAX; MIN_POSITIVE] = MIN_POSITIVE e2. Dividing B by the
    // power of two at or below MAX flushes its lower-left entry to zero, and
    // a rank tolerance relative to B drops that direction as well, so the
    // factors alone leave no update at all.
    let cancelling = LayerUpdate::new(
        "q",
        matrix(2, 2, &[f64::MAX, f64::MAX, f64::MIN_POSITIVE, 0.0]),
        matrix(2, 1, &[1.0, -1.0]),
    )
    .unwrap();
    // The same update, factored without cancellation.
    let direct = LayerUpdate::new(
        "q",
        matrix(2, 1, &[0.0, f64::MIN_POSITIVE]),
        matrix(1, 1, &[1.0]),
    )
    .unwrap();
    assert_eq!(cancelling.delta_weight(), direct.delta_weight());
    for update in [&cancelling, &direct] {
        assert_eq!(
            (update.output_basis().rank(), update.input_basis().rank()),
            (1, 1)
        );
    }
    assert_eq!(cancelling.output_basis(), direct.output_basis());
    assert_eq!(cancelling.input_basis(), direct.input_basis());
    assert_eq!(
        cancelling.output_basis(),
        column_basis(&matrix(2, 1, &[0.0, 1.0]))
    );
    // Nothing extreme is needed: 1e-12 is below the rank tolerance of B, not
    // of B A = [0; 1e-12], and another factorization of that product
    // (B G, G^-1 A with G = diag(2, 1/2)) gives the same bases.
    let near_one =
        |b: &[f64], a: &[f64]| LayerUpdate::new("q", matrix(2, 2, b), matrix(2, 1, a)).unwrap();
    let first = near_one(&[1.0, 1.0, 1e-12, 0.0], &[1.0, -1.0]);
    let second = near_one(&[2.0, 0.5, 2e-12, 0.0], &[0.5, -2.0]);
    assert_eq!(first.delta_weight(), second.delta_weight());
    for update in [&first, &second] {
        assert_eq!(update.output_basis(), direct.output_basis());
        assert_eq!(update.input_basis(), direct.input_basis());
    }
    // An update that writes exactly where an earlier one does is not
    // measured as harmless.
    let report = measure_interference(
        &candidate(),
        &[cancelling],
        &[(AdapterId::from("earlier"), vec![direct])],
    )
    .unwrap();
    let layer = &report.layers[0];
    assert!((layer.output_overlap - 1.0).abs() < 1e-12, "{layer:?}");
    assert!((layer.input_overlap - 1.0).abs() < 1e-12, "{layer:?}");
    assert!(!report.within(0.5));
}

#[test]
fn a_layer_reports_the_chance_level_of_the_comparison_that_produced_each_overlap() {
    // A 10 x 16 layer. The candidate writes e1 and reads e1.
    let axis = |dim: usize, index: usize| {
        let mut vector = vec![0.0; dim];
        vector[index] = 1.0;
        vector
    };
    let candidate_update = LayerUpdate::new(
        "q",
        matrix(10, 1, &axis(10, 0)),
        matrix(1, 16, &axis(16, 0)),
    )
    .unwrap();
    // X is rank one: it writes (e1 + e2) / sqrt 2, an output overlap of 0.5
    // at a chance level of 1/10, and reads e2.
    let mut diagonal = vec![0.0; 10];
    diagonal[0] = std::f64::consts::FRAC_1_SQRT_2;
    diagonal[1] = std::f64::consts::FRAC_1_SQRT_2;
    let x = LayerUpdate::new("q", matrix(10, 1, &diagonal), matrix(1, 16, &axis(16, 1))).unwrap();
    // Y is rank eight: it writes e3..e10 and reads e9..e16, overlapping the
    // candidate nowhere, at chance levels of 8/10 and 8/16.
    let mut writes = vec![0.0; 80];
    let mut reads = vec![0.0; 128];
    for inner in 0..8 {
        writes[(inner + 2) * 8 + inner] = 1.0;
        reads[inner * 16 + inner + 8] = 1.0;
    }
    let y = LayerUpdate::new("q", matrix(10, 8, &writes), matrix(8, 16, &reads)).unwrap();
    assert_eq!((y.output_basis().rank(), y.input_basis().rank()), (8, 8));
    for earlier in [
        vec![
            (AdapterId::from("x"), vec![x.clone()]),
            (AdapterId::from("y"), vec![y.clone()]),
        ],
        vec![
            (AdapterId::from("y"), vec![y.clone()]),
            (AdapterId::from("x"), vec![x.clone()]),
        ],
    ] {
        let report = measure_interference(
            &candidate(),
            std::slice::from_ref(&candidate_update),
            &earlier,
        )
        .unwrap();
        let layer = &report.layers[0];
        assert!((layer.output_overlap - 0.5).abs() < 1e-12, "{layer:?}");
        // X's chance level, not Y's 0.8: the overlap is five times its own
        // chance level, not below the largest one.
        assert_eq!(layer.output_chance, 0.1, "{layer:?}");
        // Neither reads e1; of two equal overlaps the one furthest above its
        // chance level is kept, here X's at 1/16 rather than Y's 1/2.
        assert!(layer.input_overlap.abs() < 1e-12, "{layer:?}");
        assert_eq!(layer.input_chance, 1.0 / 16.0, "{layer:?}");
        assert_eq!(layer.worst, Some(AdapterId::from("x")));
    }
}
