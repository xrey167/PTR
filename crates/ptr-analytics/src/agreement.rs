use std::collections::BTreeMap;

/// Krippendorff's alpha for nominal data.
///
/// `units[u]` holds the values raters assigned to unit `u`; `None` is a missing
/// rating, and an abstention must be passed as `None`, never as a category (a
/// shared "abstain" value would inflate agreement). Units with fewer than two
/// ratings are not pairable and are ignored. With coincidences
/// `o_ck = sum_u (c-k pairs in u) / (m_u - 1)`, marginals `n_c = sum_k o_ck`
/// and `n = sum_c n_c`:
///
/// ```text
/// alpha = [(n - 1) sum_c o_cc - sum_c n_c (n_c - 1)] / [n (n - 1) - sum_c n_c (n_c - 1)]
/// ```
///
/// Returns `None` when alpha is undefined: no pairable unit, or a single
/// category in all pairable ratings. Agreement among correlated automatic
/// labelers measures their redundancy, not their accuracy; apply this to
/// independent human annotators.
pub fn krippendorff_alpha_nominal(units: &[Vec<Option<usize>>]) -> Option<f64> {
    let mut coincidence: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for unit in units {
        let values: Vec<usize> = unit.iter().flatten().copied().collect();
        let m = values.len();
        if m < 2 {
            continue;
        }
        let weight = 1.0 / (m as f64 - 1.0);
        for (i, &c) in values.iter().enumerate() {
            for (j, &k) in values.iter().enumerate() {
                if i != j {
                    *coincidence.entry((c, k)).or_default() += weight;
                }
            }
        }
    }
    let mut marginals: BTreeMap<usize, f64> = BTreeMap::new();
    for (&(c, _), &count) in &coincidence {
        *marginals.entry(c).or_default() += count;
    }
    let n: f64 = marginals.values().sum();
    let diagonal: f64 = coincidence
        .iter()
        .filter(|((c, k), _)| c == k)
        .map(|(_, count)| count)
        .sum();
    let marginal_pairs: f64 = marginals.values().map(|nc| nc * (nc - 1.0)).sum();
    let denominator = n * (n - 1.0) - marginal_pairs;
    if n <= 1.0 || marginals.len() == 1 {
        return None;
    }
    Some(((n - 1.0) * diagonal - marginal_pairs) / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_matches_a_hand_computed_coincidence_matrix() {
        // o_aa = 2, o_ab = o_ba = 1, o_bb = 4; n_a = 3, n_b = 5, n = 8:
        // [(7)(6) - (6 + 20)] / [56 - 26] = 16/30.
        let units = vec![
            vec![Some(0), Some(0)],
            vec![Some(0), Some(1)],
            vec![Some(1), Some(1)],
            vec![Some(1), Some(1)],
        ];
        let alpha = krippendorff_alpha_nominal(&units).unwrap();
        assert!((alpha - 16.0 / 30.0).abs() < 1e-12);
    }

    #[test]
    fn perfect_agreement_is_one_and_missing_ratings_are_ignored() {
        let units = vec![
            vec![Some(0), Some(0), None],
            vec![Some(1), None, Some(1)],
            vec![Some(2)],
        ];
        assert_eq!(krippendorff_alpha_nominal(&units), Some(1.0));
    }

    #[test]
    fn single_category_with_fractional_coincidences_is_undefined() {
        for ratings in [4, 7, 11] {
            let units = vec![vec![Some(0); ratings], vec![Some(0); ratings + 1]];
            assert_eq!(krippendorff_alpha_nominal(&units), None);
        }
    }

    #[test]
    fn alpha_is_undefined_without_variation() {
        let units = vec![vec![Some(0), Some(0)], vec![Some(0), Some(0)]];
        assert_eq!(krippendorff_alpha_nominal(&units), None);
    }
}
