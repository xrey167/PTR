use ptr_lineage::{measure_interference, AdapterId, LayerUpdate, Matrix};

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
