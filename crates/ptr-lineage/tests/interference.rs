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

#[test]
fn adapters_on_orthogonal_subspaces_do_not_interfere() {
    let earlier = vec![(AdapterId::from("a1"), vec![rank_one("q", 0, 0)])];
    let report = measure_interference(&[rank_one("q", 1, 1)], &earlier).unwrap();
    assert!(report.max_overlap() < 1e-12);
    assert!(report.within(0.01));
}

#[test]
fn sharing_an_output_direction_is_full_output_overlap_and_names_the_culprit() {
    let earlier = vec![
        (AdapterId::from("a1"), vec![rank_one("q", 2, 0)]),
        (AdapterId::from("a2"), vec![rank_one("q", 1, 3)]),
    ];
    let report = measure_interference(&[rank_one("q", 1, 1)], &earlier).unwrap();
    let layer = &report.layers[0];
    assert!((layer.output_overlap - 1.0).abs() < 1e-12);
    assert!(layer.input_overlap.abs() < 1e-12);
    assert_eq!(layer.worst, Some(AdapterId::from("a2")));
    assert!(!report.within(0.5));
}

#[test]
fn different_layers_never_interfere() {
    let earlier = vec![(AdapterId::from("a1"), vec![rank_one("k", 1, 1)])];
    let report = measure_interference(&[rank_one("q", 1, 1)], &earlier).unwrap();
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
    let report =
        measure_interference(&[candidate], &[(AdapterId::from("earlier"), vec![earlier])]).unwrap();
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
        let report =
            measure_interference(&[update()], &[(AdapterId::from("earlier"), vec![update()])])
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
        layers: vec![layer(0.2), layer(f64::NAN), layer(0.1)],
    };
    assert!(report.max_overlap().is_nan());
    assert!(!report.within(0.1));
    assert!(!report.within(1.0));
    let measured = InterferenceReport {
        layers: vec![layer(0.2)],
    };
    assert_eq!(measured.max_overlap(), 0.2);
    assert!(measured.within(0.2));
    assert!(!measured.within(f64::NAN));
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
