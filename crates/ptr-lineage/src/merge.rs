use crate::error::LineageError;
use crate::scale;

/// Merge task vectors (flattened full updates `delta_W = B A` of equal length,
/// never LoRA factors — see [`crate::LayerUpdate::delta_weight`]) with TIES
/// (Yadav et al., "TIES-Merging"): trim each vector to its `density` fraction
/// of largest-magnitude entries, elect a sign per coordinate from the summed
/// trimmed values, and average only the entries that agree with the elected
/// sign.
///
/// Finite vectors always merge to a finite vector, whatever their magnitudes:
/// every merged entry lies between the smallest and the largest entry it
/// averages. The election and the mean are the direct in-order sums wherever
/// no partial sum overflows; a column whose running sum would overflow (two
/// entries of `f64::MAX`, or `1e308, 1e308, -1e308, -1e308, -1e308`, whose
/// true sum is negative) is summed again divided by a power of two, which
/// changes no sign and cannot overflow.
///
/// This is the consolidation step of a lineage: several adapters' deltas
/// become one, so serving cost and chain depth stop growing. The merged update
/// generally has rank up to the sum of the input ranks; refactoring it to a
/// low-rank adapter (truncated SVD, reporting the Eckart-Young residual) is the
/// trainer's step, where the tensors live. The merged adapter is a new
/// candidate and passes the same gate as a trained one; the merge itself
/// proves nothing about retained ability.
pub fn ties_merge(vectors: &[Vec<f64>], density: f64) -> Result<Vec<f64>, LineageError> {
    let Some(first) = vectors.first() else {
        return Err(LineageError::Empty {
            field: "task vectors",
        });
    };
    if !(density.is_finite() && density > 0.0 && density <= 1.0) {
        return Err(LineageError::InvalidParameter {
            field: "density",
            message: "must lie in (0, 1]",
        });
    }
    let len = first.len();
    for vector in vectors {
        if vector.len() != len {
            return Err(LineageError::ShapeMismatch {
                field: "task vector",
                expected: len,
                actual: vector.len(),
            });
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(LineageError::NonFinite {
                field: "task vector",
            });
        }
    }
    let trimmed: Vec<Vec<f64>> = vectors.iter().map(|v| trim(v, density)).collect();
    Ok((0..len)
        .map(|index| {
            let column: Vec<f64> = trimmed.iter().map(|v| v[index]).collect();
            // Only the sign of the sum is used, and scaling by a power of two
            // keeps it.
            let (sum, _) = scale::sum(&column);
            if sum == 0.0 {
                return 0.0;
            }
            // Not empty: a sum of one sign has a term of that sign.
            let agreeing: Vec<f64> = column
                .into_iter()
                .filter(|value| *value != 0.0 && value.signum() == sum.signum())
                .collect();
            scale::mean(&agreeing)
        })
        .collect())
}

/// Keep the `ceil(density * len)` largest-magnitude entries; ties at the cut
/// are broken by position so the result is deterministic.
fn trim(vector: &[f64], density: f64) -> Vec<f64> {
    let keep = ((density * vector.len() as f64).ceil() as usize).min(vector.len());
    let mut order: Vec<usize> = (0..vector.len()).collect();
    order.sort_by(|&a, &b| {
        vector[b]
            .abs()
            .total_cmp(&vector[a].abs())
            .then_with(|| a.cmp(&b))
    });
    let mut out = vec![0.0; vector.len()];
    for &index in order.iter().take(keep) {
        out[index] = vector[index];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disagreeing_signs_do_not_cancel_the_majority() {
        // Coordinate 0: +2 and +4 agree, -1 disagrees; elected sign is + and
        // the mean of the agreeing entries is 3.
        let merged = ties_merge(&[vec![2.0, 0.0], vec![4.0, 1.0], vec![-1.0, 1.0]], 1.0).unwrap();
        assert_eq!(merged, vec![3.0, 1.0]);
    }

    #[test]
    fn trimming_drops_small_entries_before_the_election() {
        // With density 0.5 each vector keeps one of two entries.
        let merged = ties_merge(&[vec![5.0, -0.1], vec![0.2, 3.0]], 0.5).unwrap();
        assert_eq!(merged, vec![5.0, 3.0]);
    }

    #[test]
    fn ragged_vectors_are_refused() {
        assert!(matches!(
            ties_merge(&[vec![1.0], vec![1.0, 2.0]], 1.0),
            Err(LineageError::ShapeMismatch { .. })
        ));
    }
}
