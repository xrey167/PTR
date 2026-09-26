use crate::error::LineageError;
use crate::lineage::AdapterId;
use crate::scale;

/// A dense row-major matrix of `f64`.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Build a matrix from row-major entries.
    ///
    /// # Errors
    /// Rejects zero dimensions, a length other than `rows * cols`, or nonfinite
    /// entries. A dimension product that does not fit in `usize` is refused
    /// with `LineageError::InvalidParameter` rather than wrapped, so no length
    /// can match it.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, LineageError> {
        if rows == 0 || cols == 0 {
            return Err(LineageError::Empty { field: "matrix" });
        }
        let expected = rows
            .checked_mul(cols)
            .ok_or(LineageError::InvalidParameter {
                field: "matrix dimensions",
                message: "rows * cols overflows usize",
            })?;
        if data.len() != expected {
            return Err(LineageError::ShapeMismatch {
                field: "matrix data",
                expected,
                actual: data.len(),
            });
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(LineageError::NonFinite { field: "matrix" });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    fn column(&self, index: usize) -> Vec<f64> {
        (0..self.rows)
            .map(|row| self.data[row * self.cols + index])
            .collect()
    }

    /// Entry at `(row, col)`.
    ///
    /// # Panics
    /// Panics if `row` or `col` is outside the matrix. Both are checked: a
    /// column index past the last would otherwise read an entry of the next
    /// row.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(
            row < self.rows && col < self.cols,
            "entry ({row}, {col}) is outside a {}x{} matrix",
            self.rows,
            self.cols
        );
        self.data[row * self.cols + col]
    }

    /// Row-major entries.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// `self * other`.
    ///
    /// # Errors
    /// Returns `LineageError::ShapeMismatch` unless `self` has as many columns
    /// as `other` has rows, and `LineageError::NonFinite` when an entry of the
    /// product exceeds `f64::MAX`, so the result is a matrix
    /// [`Matrix::new`] would accept. An entry that is finite although its
    /// running sum overflows (`MAX + MAX - MAX`) is not refused: such entries
    /// are recomputed from both factors divided by powers of two and scaled
    /// back; every other entry is the direct sum of products.
    pub fn multiply(&self, other: &Matrix) -> Result<Matrix, LineageError> {
        if self.cols != other.rows {
            return Err(LineageError::ShapeMismatch {
                field: "matrix product",
                expected: self.cols,
                actual: other.rows,
            });
        }
        let mut product = self.product(other);
        if product.data.iter().all(|cell| cell.is_finite()) {
            return Ok(product);
        }
        let (left, left_exponent) = self.rescaled();
        let (right, right_exponent) = other.rescaled();
        // Entries below two in magnitude: no sum of products overflows.
        let rescaled = left.product(&right);
        for (cell, small) in product.data.iter_mut().zip(rescaled.data) {
            if !cell.is_finite() {
                *cell = scale::times_power_of_two(small, left_exponent + right_exponent);
            }
        }
        if product.data.iter().any(|cell| !cell.is_finite()) {
            return Err(LineageError::NonFinite {
                field: "matrix product",
            });
        }
        Ok(product)
    }

    /// The product of matrices of matching shapes, as floating-point sums of
    /// products in order; an entry may overflow.
    fn product(&self, other: &Matrix) -> Matrix {
        let mut data = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for k in 0..self.cols {
                let left = self.data[i * self.cols + k];
                if left == 0.0 {
                    continue;
                }
                let row = &other.data[k * other.cols..(k + 1) * other.cols];
                for (cell, &right) in data[i * other.cols..(i + 1) * other.cols]
                    .iter_mut()
                    .zip(row)
                {
                    *cell += left * right;
                }
            }
        }
        Matrix {
            rows: self.rows,
            cols: other.cols,
            data,
        }
    }

    /// Frobenius norm, computed on the matrix divided by a power of two so
    /// that no square overflows or underflows: `[1e200]` has norm `1e200`
    /// and `[1e-200]` norm `1e-200`. It is infinite only when the norm itself
    /// exceeds `f64::MAX`.
    pub fn frobenius(&self) -> f64 {
        let (scaled, exponent) = self.rescaled();
        scale::times_power_of_two(scaled.sum_of_squares().sqrt(), exponent)
    }

    fn sum_of_squares(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum()
    }

    /// This matrix divided by the power of two at or below its largest
    /// magnitude, and that power's exponent. Every entry of the result is
    /// below two in magnitude; a zero matrix is returned unchanged with
    /// exponent zero. A positive scale changes no span, basis or ratio of
    /// norms, and the division is exact unless an entry underflows.
    fn rescaled(&self) -> (Matrix, i32) {
        let Some(exponent) = scale::exponent(self.data.iter().copied()) else {
            return (self.clone(), 0);
        };
        let factor = scale::power_of_two(exponent);
        (
            Matrix {
                rows: self.rows,
                cols: self.cols,
                data: self.data.iter().map(|x| x / factor).collect(),
            },
            exponent,
        )
    }

    fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.data.len());
        for col in 0..self.cols {
            data.extend(self.column(col));
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }
}

/// An orthonormal basis of a subspace of `R^dim`.
#[derive(Clone, Debug, PartialEq)]
pub struct Basis {
    dim: usize,
    vectors: Vec<Vec<f64>>,
}

impl Basis {
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Dimension of the subspace.
    pub fn rank(&self) -> usize {
        self.vectors.len()
    }
}

/// Relative size below which a column is treated as linearly dependent.
const RANK_TOLERANCE: f64 = 1e-10;

/// Orthonormal basis of the column space of `matrix`, by modified Gram-Schmidt
/// with one full reorthogonalisation pass ("twice is enough"). A column whose
/// remainder is negligible relative to the largest column of the matrix is
/// dropped, so the basis has the numerical rank of the matrix; a tolerance
/// relative to each column's own norm would keep a column that is only
/// rounding noise of the others.
///
/// The column space does not depend on the scale of the matrix, so the matrix
/// is first divided by the power of two at or below its largest entry: finite
/// entries of any magnitude (`1e200`, `1e-170`) then give the same basis as
/// the same matrix near one, where squared norms would otherwise overflow or
/// underflow and drop every column.
pub fn column_basis(matrix: &Matrix) -> Basis {
    let (matrix, _) = matrix.rescaled();
    let matrix = &matrix;
    let scale = (0..matrix.cols)
        .map(|index| norm(&matrix.column(index)))
        .fold(0.0, f64::max);
    let mut vectors: Vec<Vec<f64>> = Vec::new();
    for index in 0..matrix.cols {
        let mut column = matrix.column(index);
        let original = norm(&column);
        if original == 0.0 {
            continue;
        }
        for _ in 0..2 {
            for base in &vectors {
                let projection = dot(&column, base);
                for (cell, &b) in column.iter_mut().zip(base) {
                    *cell -= projection * b;
                }
            }
        }
        let remainder = norm(&column);
        if remainder <= RANK_TOLERANCE * scale {
            continue;
        }
        column.iter_mut().for_each(|cell| *cell /= remainder);
        vectors.push(column);
    }
    Basis {
        dim: matrix.rows,
        vectors,
    }
}

/// Cosines of the principal angles between two subspaces, largest first.
///
/// They are the singular values of `Qa^T Qb` for orthonormal bases `Qa`, `Qb`
/// (Björck and Golub). There are `min(rank_a, rank_b)` of them, each in
/// `[0, 1]`; `1` means the subspaces share a direction and `0` that they are
/// orthogonal along it.
pub fn principal_cosines(left: &Basis, right: &Basis) -> Result<Vec<f64>, LineageError> {
    if left.dim != right.dim {
        return Err(LineageError::ShapeMismatch {
            field: "subspace dimension",
            expected: left.dim,
            actual: right.dim,
        });
    }
    let count = left.rank().min(right.rank());
    if count == 0 {
        return Ok(Vec::new());
    }
    // cross[i][j] = <left_i, right_j>; gram = cross^T cross is rank_b x rank_b.
    let cross: Vec<Vec<f64>> = left
        .vectors
        .iter()
        .map(|a| right.vectors.iter().map(|b| dot(a, b)).collect())
        .collect();
    let size = right.rank();
    let mut gram = vec![vec![0.0; size]; size];
    for (i, gram_row) in gram.iter_mut().enumerate() {
        for (j, cell) in gram_row.iter_mut().enumerate() {
            *cell = cross.iter().map(|row| row[i] * row[j]).sum();
        }
    }
    let mut eigenvalues = symmetric_eigenvalues(gram);
    eigenvalues.sort_by(|a, b| b.total_cmp(a));
    Ok(eigenvalues
        .into_iter()
        .take(count)
        .map(|value| value.max(0.0).sqrt().min(1.0))
        .collect())
}

/// Normalised subspace overlap `||Qa^T Qb||_F^2 / min(rank_a, rank_b)`: the mean
/// squared principal cosine, `0` for orthogonal subspaces and `1` when the
/// smaller one lies inside the larger. Compare it with [`chance_overlap`]: two
/// random subspaces overlap by that much with no interference at all.
pub fn subspace_overlap(left: &Basis, right: &Basis) -> Result<f64, LineageError> {
    let cosines = principal_cosines(left, right)?;
    if cosines.is_empty() {
        return Ok(0.0);
    }
    Ok(cosines.iter().map(|c| c * c).sum::<f64>() / cosines.len() as f64)
}

/// Expected normalised overlap of two uniformly random subspaces of the given
/// ranks in `R^dim`: `E ||Qa^T Qb||_F^2 = rank_a * rank_b / dim`, normalised by
/// the smaller rank, which is `max(rank_a, rank_b) / dim`. Overlaps near this
/// level carry no evidence of interference; ranks that differ are only
/// comparable after subtracting it.
///
/// # Errors
/// Returns `LineageError::ShapeMismatch` when the subspaces live in spaces
/// of different dimensions, as [`principal_cosines`] does: they have no
/// overlap, by chance or otherwise.
pub fn chance_overlap(left: &Basis, right: &Basis) -> Result<f64, LineageError> {
    if left.dim != right.dim {
        return Err(LineageError::ShapeMismatch {
            field: "subspace dimension",
            expected: left.dim,
            actual: right.dim,
        });
    }
    if left.rank() == 0 || right.rank() == 0 || left.dim == 0 {
        return Ok(0.0);
    }
    Ok(left.rank().max(right.rank()) as f64 / left.dim as f64)
}

/// One layer of a low-rank update `delta_W = B A`, with `B` of shape
/// `d_out x r` and `A` of shape `r x d_in`.
///
/// The fields are private so that [`LayerUpdate::new`], which checks that the
/// factors' inner dimensions agree, is the only way to build one:
///
/// ```compile_fail
/// use ptr_lineage::{LayerUpdate, Matrix};
/// let b = Matrix::new(2, 2, vec![1.0; 4]).unwrap();
/// let a = Matrix::new(3, 1, vec![1.0; 3]).unwrap();
/// let misaligned = LayerUpdate { layer: "q".into(), b, a };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LayerUpdate {
    layer: String,
    b: Matrix,
    a: Matrix,
}

impl LayerUpdate {
    /// Describe the update `B * A` for a layer. Returns
    /// `LineageError::ShapeMismatch` unless `B`'s column count equals `A`'s
    /// row count; rank-deficient factors are accepted.
    pub fn new(layer: impl Into<String>, b: Matrix, a: Matrix) -> Result<Self, LineageError> {
        if b.cols != a.rows {
            return Err(LineageError::ShapeMismatch {
                field: "adapter rank",
                expected: b.cols,
                actual: a.rows,
            });
        }
        Ok(Self {
            layer: layer.into(),
            b,
            a,
        })
    }

    /// The layer this update modifies.
    pub fn layer(&self) -> &str {
        &self.layer
    }

    /// The `d_out x r` factor `B`.
    pub fn b(&self) -> &Matrix {
        &self.b
    }

    /// The `r x d_in` factor `A`.
    pub fn a(&self) -> &Matrix {
        &self.a
    }

    /// Orthonormal basis of the column space of `delta_W = B A`: the outputs
    /// the update can write.
    pub fn output_basis(&self) -> Basis {
        self.update_bases().0
    }

    /// Orthonormal basis of the row space of `delta_W = B A`: the inputs the
    /// update reads.
    pub fn input_basis(&self) -> Basis {
        self.update_bases().1
    }

    /// Bases of the column and row spaces of `B A`, computed without forming
    /// the `d_out x d_in` product.
    ///
    /// With `Q` an orthonormal basis of `col(B)` and `M = (Q^T B) A`, which is
    /// only `rank(B) x d_in`, `B A = Q M`; so `row(B A) = row(M)` and
    /// `col(B A) = Q col(M)`. Taking `col(B)` and `row(A)` directly instead is
    /// wrong whenever the factors are rank-deficient together: `B = [b, b]`,
    /// `A = [a1; a2]` gives `B A = b (a1 + a2)^T`, whose row space is one
    /// direction, not the plane `row(A)`. Since the overlap is normalised by
    /// the smaller rank, a too-large subspace can understate overlap as well
    /// as overstate it.
    ///
    /// Both factors are first divided by powers of two, which changes neither
    /// space, so `M` stays in range for factors of any finite magnitude.
    fn update_bases(&self) -> (Basis, Basis) {
        let (b, _) = self.b.rescaled();
        let (a, _) = self.a.rescaled();
        let (d_out, d_in) = (b.rows, a.cols);
        let q = column_basis(&b);
        let rank = b.cols;
        // M = (Q^T B) A, one row per basis vector of col(B).
        let m_rows: Vec<Vec<f64>> = q
            .vectors
            .iter()
            .map(|basis_vector| {
                let r: Vec<f64> = (0..rank)
                    .map(|column| dot(basis_vector, &b.column(column)))
                    .collect();
                (0..d_in)
                    .map(|input| {
                        r.iter()
                            .enumerate()
                            .map(|(inner, coefficient)| coefficient * a.get(inner, input))
                            .sum()
                    })
                    .collect()
            })
            .collect();
        if m_rows.is_empty() {
            return (
                Basis {
                    dim: d_out,
                    vectors: Vec::new(),
                },
                Basis {
                    dim: d_in,
                    vectors: Vec::new(),
                },
            );
        }
        let m = Matrix {
            rows: m_rows.len(),
            cols: d_in,
            data: m_rows.concat(),
        };
        let input = column_basis(&m.transpose());
        // An orthonormal basis of col(M) in coordinates of Q, mapped through Q:
        // orthonormal because Q's columns are.
        let coordinates = column_basis(&m);
        let output = coordinates
            .vectors
            .iter()
            .map(|coefficients| {
                (0..d_out)
                    .map(|row| {
                        coefficients
                            .iter()
                            .zip(&q.vectors)
                            .map(|(coefficient, basis_vector)| coefficient * basis_vector[row])
                            .sum()
                    })
                    .collect()
            })
            .collect();
        (
            Basis {
                dim: d_out,
                vectors: output,
            },
            input,
        )
    }

    /// The full update `delta_W = B A`. Merging and comparing adapters must
    /// work on this product, never on the factors: `B A = (B G)(G^-1 A)` for
    /// every invertible `G`, so any operation on `A` and `B` separately
    /// depends on an arbitrary choice of basis.
    ///
    /// # Errors
    /// Returns `LineageError::NonFinite` when an entry of the product exceeds
    /// `f64::MAX` (finite factors of `1e200` multiply to `1e400`), rather
    /// than a matrix holding infinity; see [`Matrix::multiply`].
    pub fn delta_weight(&self) -> Result<Matrix, LineageError> {
        self.b.multiply(&self.a)
    }
}

/// Data-dependent interference of a candidate update with an earlier one on
/// the earlier task's inputs: `||dW_new X||_F / ||dW_old X||_F`, where the
/// columns of `activations` (`d_in x m`) are held-out inputs of the earlier
/// task at this layer. Geometry alone ignores which input directions the
/// earlier task actually uses; this measures how much the new update moves the
/// layer's output on exactly those inputs, relative to the change the earlier
/// update made. `None` when the earlier update does not move them at all.
///
/// The ratio does not depend on the scale of either update or of the
/// activations, and it is computed that way: every factor and both effects
/// are divided by powers of two, whose exponents are added back to the ratio
/// alone. Finite inputs whose effects would overflow or underflow (updates
/// and inputs of `1e100`, or of `1e-100`) therefore still give their ratio.
///
/// # Errors
/// Returns `LineageError::ShapeMismatch` when the activations do not have
/// one row per input of both updates, and `LineageError::NonFinite` when the
/// ratio itself exceeds `f64::MAX`.
pub fn activation_interference(
    candidate: &LayerUpdate,
    earlier: &LayerUpdate,
    activations: &Matrix,
) -> Result<Option<f64>, LineageError> {
    let (inputs, _) = activations.rescaled();
    let (new_norm, new_exponent) = rescaled_effect_norm(candidate, &inputs)?;
    let (old_norm, old_exponent) = rescaled_effect_norm(earlier, &inputs)?;
    if old_norm == 0.0 {
        return Ok(None);
    }
    let ratio = scale::times_power_of_two(new_norm / old_norm, new_exponent - old_exponent);
    if !ratio.is_finite() {
        return Err(LineageError::NonFinite {
            field: "activation interference",
        });
    }
    Ok(Some(ratio))
}

/// `||B A X||_F` as `(norm, exponent)`, the norm of the effect being
/// `norm * 2^exponent` up to the scale of `X`. With `B`, `A` and the effect
/// each divided by the power of two at or below its largest entry, the norm
/// is zero or between `2^-52` and `2 sqrt(entries)`, so a ratio of two of
/// them is finite and nonzero whatever the magnitudes.
fn rescaled_effect_norm(update: &LayerUpdate, inputs: &Matrix) -> Result<(f64, i32), LineageError> {
    let (b, b_exponent) = update.b.rescaled();
    let (a, a_exponent) = update.a.rescaled();
    let (effect, effect_exponent) = b.multiply(&a.multiply(inputs)?)?.rescaled();
    Ok((
        effect.sum_of_squares().sqrt(),
        b_exponent + a_exponent + effect_exponent,
    ))
}

/// Worst overlap of one layer of a candidate adapter with any earlier adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerInterference {
    pub layer: String,
    pub output_overlap: f64,
    pub input_overlap: f64,
    /// Chance level of the output-side overlap for these ranks.
    pub output_chance: f64,
    /// Chance level of the input-side overlap for these ranks.
    pub input_chance: f64,
    /// The earlier adapter with the largest overlap on this layer.
    pub worst: Option<AdapterId>,
}

/// Per-layer interference of a candidate with an existing lineage.
#[derive(Clone, Debug, PartialEq)]
pub struct InterferenceReport {
    /// The adapter whose updates were measured. [`measure_interference`]
    /// sets it, and a store records the report as this adapter's evidence
    /// only (`ptr-pg` refuses it under any other).
    pub candidate: AdapterId,
    pub layers: Vec<LayerInterference>,
}

impl InterferenceReport {
    /// The largest overlap on any layer, output or input side, or NaN when
    /// any overlap is NaN. [`measure_interference`] never reports one, but a
    /// report built or loaded by hand can, and folding it away with
    /// `f64::max` would let an unmeasured layer pass as no overlap at all.
    pub fn max_overlap(&self) -> f64 {
        self.layers
            .iter()
            .flat_map(|layer| [layer.output_overlap, layer.input_overlap])
            .fold(0.0, |worst, overlap| {
                if overlap.is_nan() || overlap > worst {
                    overlap
                } else {
                    worst
                }
            })
    }

    /// Whether every layer stays within `max_overlap`. False when any
    /// overlap, or `max_overlap` itself, is NaN: what cannot be compared is
    /// not within the limit.
    pub fn within(&self, max_overlap: f64) -> bool {
        self.max_overlap() <= max_overlap
    }
}

/// Measure how much the update subspaces of adapter `candidate` overlap those
/// of earlier adapters, layer by layer. The report names `candidate`, so it
/// cannot be taken for another adapter's evidence.
///
/// This is the quantity orthogonal-subspace continual-learning methods
/// (O-LoRA, InfLoRA) drive towards zero. It is measured between subspaces, not
/// against the null space of a summed update: the null space of `sum_i dW_i` is
/// not the intersection of the individual null spaces, so an update inside it
/// can still overlap every earlier adapter.
pub fn measure_interference(
    candidate: &AdapterId,
    updates: &[LayerUpdate],
    earlier: &[(AdapterId, Vec<LayerUpdate>)],
) -> Result<InterferenceReport, LineageError> {
    let mut layers = Vec::with_capacity(updates.len());
    for update in updates {
        let output = update.output_basis();
        let input = update.input_basis();
        let mut worst: Option<AdapterId> = None;
        let mut output_overlap: f64 = 0.0;
        let mut input_overlap: f64 = 0.0;
        let mut output_chance: f64 = 0.0;
        let mut input_chance: f64 = 0.0;
        for (id, updates) in earlier {
            for previous in updates.iter().filter(|p| p.layer == update.layer) {
                let previous_output = previous.output_basis();
                let previous_input = previous.input_basis();
                let out = subspace_overlap(&output, &previous_output)?;
                let inp = subspace_overlap(&input, &previous_input)?;
                output_chance = output_chance.max(chance_overlap(&output, &previous_output)?);
                input_chance = input_chance.max(chance_overlap(&input, &previous_input)?);
                if out.max(inp) > output_overlap.max(input_overlap) {
                    worst = Some(id.clone());
                }
                output_overlap = output_overlap.max(out);
                input_overlap = input_overlap.max(inp);
            }
        }
        layers.push(LayerInterference {
            layer: update.layer.clone(),
            output_overlap,
            input_overlap,
            output_chance,
            input_chance,
            worst,
        });
    }
    Ok(InterferenceReport {
        candidate: candidate.clone(),
        layers,
    })
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(l, r)| l * r).sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

/// Eigenvalues of a small symmetric matrix by the cyclic Jacobi method.
fn symmetric_eigenvalues(mut matrix: Vec<Vec<f64>>) -> Vec<f64> {
    let size = matrix.len();
    for _sweep in 0..100 {
        let off: f64 = (0..size)
            .flat_map(|i| (0..size).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| matrix[i][j] * matrix[i][j])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..size {
            for q in (p + 1)..size {
                if matrix[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (matrix[q][q] - matrix[p][p]) / (2.0 * matrix[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in matrix.iter_mut() {
                    let (kp, kq) = (row[p], row[q]);
                    row[p] = c * kp - s * kq;
                    row[q] = s * kp + c * kq;
                }
                // p < q, so row p lies in the first half and row q starts the second.
                let (upper, lower) = matrix.split_at_mut(q);
                for (pk, qk) in upper[p].iter_mut().zip(lower[0].iter_mut()) {
                    let (left, right) = (*pk, *qk);
                    *pk = c * left - s * right;
                    *qk = s * left + c * right;
                }
            }
        }
    }
    (0..size).map(|i| matrix[i][i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, cols: usize, data: &[f64]) -> Matrix {
        Matrix::new(rows, cols, data.to_vec()).unwrap()
    }

    #[test]
    fn identical_subspaces_overlap_fully_and_orthogonal_ones_not_at_all() {
        let a = column_basis(&matrix(3, 1, &[1.0, 0.0, 0.0]));
        let b = column_basis(&matrix(3, 1, &[2.0, 0.0, 0.0]));
        let c = column_basis(&matrix(3, 1, &[0.0, 5.0, 0.0]));
        assert!((subspace_overlap(&a, &b).unwrap() - 1.0).abs() < 1e-12);
        assert!(subspace_overlap(&a, &c).unwrap().abs() < 1e-12);
    }

    #[test]
    fn a_45_degree_line_has_cosine_one_over_root_two() {
        let a = column_basis(&matrix(2, 1, &[1.0, 0.0]));
        let b = column_basis(&matrix(2, 1, &[1.0, 1.0]));
        let cosines = principal_cosines(&a, &b).unwrap();
        assert!((cosines[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn a_dependent_column_does_not_add_rank() {
        let basis = column_basis(&matrix(
            3,
            3,
            &[1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ));
        assert_eq!(basis.rank(), 2);
    }

    #[test]
    fn a_plane_contains_its_own_line() {
        // span{e1, e2} and span{e1 + e2}: the line lies inside the plane.
        let plane = column_basis(&matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
        let line = column_basis(&matrix(3, 1, &[1.0, 1.0, 0.0]));
        assert!((subspace_overlap(&plane, &line).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn chance_overlap_is_the_larger_rank_over_the_dimension() {
        let line = column_basis(&matrix(4, 1, &[1.0, 0.0, 0.0, 0.0]));
        let plane = column_basis(&matrix(4, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]));
        assert_eq!(chance_overlap(&line, &plane).unwrap(), 0.5);
    }

    #[test]
    fn the_product_of_factors_is_the_update() {
        let b = matrix(2, 1, &[1.0, 2.0]);
        let a = matrix(1, 3, &[3.0, 0.0, -1.0]);
        let update = LayerUpdate::new("q", b, a).unwrap();
        assert_eq!(
            update.delta_weight().unwrap().as_slice(),
            &[3.0, 0.0, -1.0, 6.0, 0.0, -2.0]
        );
    }

    #[test]
    fn activation_interference_is_zero_on_inputs_the_new_update_ignores() {
        // The earlier update reads input axis 0; the candidate reads axis 1.
        let earlier =
            LayerUpdate::new("q", matrix(2, 1, &[1.0, 0.0]), matrix(1, 2, &[1.0, 0.0])).unwrap();
        let candidate =
            LayerUpdate::new("q", matrix(2, 1, &[1.0, 0.0]), matrix(1, 2, &[0.0, 1.0])).unwrap();
        // The earlier task's inputs lie on axis 0.
        let inputs = matrix(2, 3, &[1.0, 2.0, -1.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            activation_interference(&candidate, &earlier, &inputs).unwrap(),
            Some(0.0)
        );
        // Although both write to the same output direction.
        assert!(
            (subspace_overlap(&candidate.output_basis(), &earlier.output_basis()).unwrap() - 1.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn overflowing_dimensions_are_refused_rather_than_wrapped() {
        let overflow = Err(LineageError::InvalidParameter {
            field: "matrix dimensions",
            message: "rows * cols overflows usize",
        });
        // 2 * (usize::MAX / 2 + 1) wraps to exactly 0, so an unchecked product
        // would match an empty buffer (or panic in a debug build).
        assert_eq!(Matrix::new(2, usize::MAX / 2 + 1, Vec::new()), overflow);
        assert_eq!(Matrix::new(usize::MAX, usize::MAX, vec![0.0]), overflow);
        // The largest representable product is still only a length check.
        assert_eq!(
            Matrix::new(1, usize::MAX, vec![0.0]),
            Err(LineageError::ShapeMismatch {
                field: "matrix data",
                expected: usize::MAX,
                actual: 1,
            })
        );
    }

    #[test]
    #[should_panic(expected = "entry (0, 2) is outside a 2x2 matrix")]
    fn a_column_past_the_last_panics_rather_than_reading_the_next_row() {
        // Row-major, (0, 2) is the flat index of (1, 0).
        let _ = matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]).get(0, 2);
    }

    #[test]
    fn jacobi_recovers_known_eigenvalues() {
        let mut values = symmetric_eigenvalues(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        values.sort_by(f64::total_cmp);
        assert!((values[0] - 1.0).abs() < 1e-12);
        assert!((values[1] - 3.0).abs() < 1e-12);
    }
}
