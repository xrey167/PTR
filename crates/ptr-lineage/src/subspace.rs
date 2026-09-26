use crate::error::LineageError;
use crate::lineage::AdapterId;
use crate::scale::{self, Wide};

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
    /// are recomputed as the same in-order sum of products with an exponent
    /// range `f64` does not bound, so large terms that cancel leave the small
    /// ones that decide the entry (`MAX + MAX - MAX - MAX + 1e-20` is
    /// `1e-20`); every other entry is the direct sum of products.
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
        let (left, right) = (WideMatrix::of(self), WideMatrix::of(other));
        for (index, cell) in product.data.iter_mut().enumerate() {
            if !cell.is_finite() {
                *cell = left
                    .entry_of_product(&right, index / other.cols, index % other.cols)
                    .to_f64();
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

    /// The direct product when it is exactly the product with an unbounded
    /// exponent range: no entry overflows, and the smallest nonzero
    /// magnitudes of the two factors multiply to at least
    /// `f64::MIN_POSITIVE`, so no product of two entries underflows and the
    /// sums of such products lose nothing either. `None` otherwise.
    fn product_in_range(&self, other: &Matrix) -> Option<Matrix> {
        let smallest = |matrix: &Matrix| {
            matrix
                .data
                .iter()
                .filter(|value| **value != 0.0)
                .map(|value| scale::exact_exponent(*value))
                .min()
        };
        if let (Some(left), Some(right)) = (smallest(self), smallest(other)) {
            // 2^left * 2^right is at most the smallest nonzero product.
            if left + right < -1022 {
                return None;
            }
        }
        let product = self.product(other);
        product
            .data
            .iter()
            .all(|cell| cell.is_finite())
            .then_some(product)
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
        let mut matrix = self.clone();
        let exponent = matrix.rescale();
        (matrix, exponent)
    }

    /// [`Matrix::rescaled`], when no nonzero entry is more than
    /// `2^MAX_FACTOR_SPREAD` times smaller than the largest; `None`
    /// otherwise. Every nonzero entry of the result is then at least
    /// `2^-400` in magnitude, so the division is exact and no product of two
    /// entries underflows.
    fn rescaled_within_spread(&self) -> Option<Matrix> {
        let (smallest, largest) = self
            .data
            .iter()
            .filter(|value| **value != 0.0)
            .map(|value| scale::exact_exponent(*value))
            .fold((i32::MAX, i32::MIN), |(low, high), exponent| {
                (low.min(exponent), high.max(exponent))
            });
        if smallest <= largest && largest - smallest > MAX_FACTOR_SPREAD {
            return None;
        }
        Some(self.rescaled().0)
    }

    /// Divide this matrix in place as [`Matrix::rescaled`] does, and return
    /// the exponent of the power of two it was divided by.
    fn rescale(&mut self) -> i32 {
        let Some(exponent) = scale::exponent(self.data.iter().copied()) else {
            return 0;
        };
        let factor = scale::power_of_two(exponent);
        self.data.iter_mut().for_each(|x| *x /= factor);
        exponent
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

/// Largest spread, in powers of two, between the largest and the smallest
/// nonzero magnitude of either factor for which the bases of an update are
/// computed from its factors ([`LayerUpdate::input_basis`]). Factors held as
/// `f32` or `bf16` spread over at most 277.
const MAX_FACTOR_SPREAD: i32 = 400;

/// Largest ratio, on either side of an update, between the size the
/// product's largest column (row) could have without cancellation and the
/// size it has, for which its bases are computed from its factors
/// ([`LayerUpdate::input_basis`]).
const MAX_CANCELLATION: f64 = 256.0;

/// A row is left out of the orthonormal basis of a factor ([`RowSpan`]) only
/// when what orthogonalisation leaves of it is at most this fraction,
/// `2^-50`, of its own norm: a few units in the last place.
const FACTOR_TOLERANCE: f64 = 1.0 / (1u64 << 50) as f64;

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
    row_basis(matrix.transpose())
}

/// Orthonormal basis of the span of the rows of `matrix`: [`column_basis`] of
/// its transpose, with the same arithmetic on the same values in the same
/// order, but reading each vector from contiguous memory. The matrix is
/// divided in place.
fn row_basis(mut matrix: Matrix) -> Basis {
    matrix.rescale();
    let dim = matrix.cols;
    let scale = matrix.data.chunks_exact(dim).map(norm).fold(0.0, f64::max);
    let mut vectors: Vec<Vec<f64>> = Vec::new();
    for row in matrix.data.chunks_exact(dim) {
        let original = norm(row);
        if original == 0.0 {
            continue;
        }
        let mut vector = row.to_vec();
        for _ in 0..2 {
            for base in &vectors {
                let projection = dot(&vector, base);
                for (cell, &b) in vector.iter_mut().zip(base) {
                    *cell -= projection * b;
                }
            }
        }
        let remainder = norm(&vector);
        if remainder <= RANK_TOLERANCE * scale {
            continue;
        }
        vector.iter_mut().for_each(|cell| *cell /= remainder);
        vectors.push(vector);
    }
    Basis { dim, vectors }
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
    /// the update can write. It is [`column_basis`] of the product, computed
    /// as described under [`LayerUpdate::input_basis`].
    pub fn output_basis(&self) -> Basis {
        self.update_bases().0
    }

    /// Orthonormal basis of the row space of `delta_W = B A`: the inputs the
    /// update reads. It is [`column_basis`] of the transposed product.
    ///
    /// Both bases are the Gram-Schmidt of [`column_basis`] over the columns
    /// (rows) of `B A` in order, with its rank tolerance relative to the
    /// largest of them; they are never `col(B)` and `row(A)`, which are
    /// wrong whenever the factors are rank-deficient together: `B = [b, b]`,
    /// `A = [a1; a2]` gives `B A = b (a1 + a2)^T`, whose row space is one
    /// direction, not the plane `row(A)`; since the overlap is normalised by
    /// the smaller rank, a too-large subspace can understate overlap as well
    /// as overstate it. That Gram-Schmidt runs on one of two
    /// representations of the product, with `r` the inner dimension of the
    /// factors.
    ///
    /// **From the factors**, in `O((d_out + d_in) r^2)` time and
    /// `O((d_out + d_in) r)` memory. Each factor is divided by the power of
    /// two at or below its largest entry, and orthonormal bases `Q_B` of
    /// `col(B)` and `Q_A` of `row(A)` are built by Gram-Schmidt with one
    /// reorthogonalisation pass, leaving a column of `B` (row of `A`) out
    /// only when what is left of it is at most `2^-50` of its own norm, never
    /// for being small beside another. The Gram-Schmidt of the product then
    /// runs on the coordinates of its columns in `Q_B` (rows in `Q_A`),
    /// which have the lengths and inner products of the columns (rows)
    /// themselves up to rounding, and the result is mapped back through
    /// `Q_B` (`Q_A`). This representation is used only where the factors are
    /// smaller than the product, `(d_out + d_in) r < d_out d_in`, and both
    /// checks hold:
    ///
    /// 1. no nonzero entry of either factor is more than `2^400` times
    ///    smaller than the largest entry of that factor, so the division is
    ///    exact and no product of two factor entries underflows;
    /// 2. on each side, the largest column (row) of the product, as its
    ///    coordinates give it, is at least `2^-8` of the largest size a
    ///    column (row) could have without cancellation: `sum_k |b_k| |a_kj|`
    ///    for column `j` and `sum_k |b_ik| |a_k|` for row `i`, with `b_k` the
    ///    columns of `B` and `a_k` the rows of `A`.
    ///
    /// Its rounding, like that of the direct product, is bounded relative to
    /// those sizes without cancellation, which check 2 keeps within `2^8` of
    /// the largest column (row), where the rank tolerance applies. The bases
    /// agree with those taken from the formed product up to that rounding,
    /// not bit for bit; the rank can differ only for a column (row) whose
    /// remainder lies within it of the tolerance.
    ///
    /// **From the formed product** otherwise, in `O(d_out d_in r)` time with
    /// memory for a few `d_out x d_in` matrices, as
    /// [`LayerUpdate::delta_weight`] takes. Every entry of `B A` is the
    /// in-order sum of products, rounded as `f64` rounds each step but with
    /// an exponent range `f64` does not bound (the direct product wherever
    /// it provably is exactly that, as for [`activation_interference`]), and
    /// the product is then divided by the power of two at or below its
    /// largest entry, which is exact except for entries more than `2^1022`
    /// times smaller than the largest (rounded to subnormal values or zero),
    /// far below the rank tolerance. Such bases depend on the factors only
    /// through that product: factorizations whose products are computed as
    /// the same matrix give the same bases, and a direction the product
    /// keeps after large terms cancel is kept however large they were. It
    /// is the representation used where factors cancel: `B = [MAX, MAX;
    /// MIN_POSITIVE, 0]` and `A = [1; -1]` multiply to `MIN_POSITIVE e2`,
    /// which dividing `B` by its largest power of two flushes, and which a
    /// rank tolerance relative to `B` drops, leaving no update at all; they
    /// fail both checks.
    pub fn input_basis(&self) -> Basis {
        self.update_bases().1
    }

    /// The output and input bases: from the factors where they are smaller
    /// than the product and both checks under [`LayerUpdate::input_basis`]
    /// hold, from the formed product otherwise.
    fn update_bases(&self) -> (Basis, Basis) {
        let (d_out, rank, d_in) = (
            self.b.rows as u128,
            self.b.cols as u128,
            self.a.cols as u128,
        );
        let thinner = (d_out + d_in) * rank < d_out * d_in;
        thinner
            .then(|| self.factored_bases())
            .flatten()
            .unwrap_or_else(|| self.product_bases())
    }

    /// The output and input bases, from the formed product.
    fn product_bases(&self) -> (Basis, Basis) {
        let product = self.product_at_scale();
        (row_basis(product.transpose()), row_basis(product))
    }

    /// The output and input bases, from the factors, or `None` when either
    /// check under [`LayerUpdate::input_basis`] fails.
    fn factored_bases(&self) -> Option<(Basis, Basis)> {
        let b = self.b.rescaled_within_spread()?;
        let a = self.a.rescaled_within_spread()?;
        let (d_out, d_in) = (b.rows, a.cols);
        // The rows of B^T are the columns of B.
        let b_columns = b.transpose();
        let (col_b, row_a) = (RowSpan::of(&b_columns), RowSpan::of(&a));
        // Column j of B A is sum_k A[k][j] b_k; row i is sum_k B[i][k] a_k.
        let (columns, output_uncancelled) = col_b.combinations(&a);
        let (rows, input_uncancelled) = row_a.combinations(&b_columns);
        if output_uncancelled == 0.0 {
            // Every term b_ik a_kj is zero (then input_uncancelled is zero
            // too), and so is the product.
            let empty = |dim| Basis {
                dim,
                vectors: Vec::new(),
            };
            return Some((empty(d_out), empty(d_in)));
        }
        let columns = within_cancellation(columns, output_uncancelled)?;
        let rows = within_cancellation(rows, input_uncancelled)?;
        Some((
            col_b.lift(&row_basis(columns)),
            row_a.lift(&row_basis(rows)),
        ))
    }

    /// `B A`, every entry the in-order sum of products with an exponent range
    /// `f64` does not bound (the direct product where it provably is exactly
    /// that sum, the wide one otherwise), brought to scale by [`at_scale`],
    /// which treats equal entries alike whichever way they were computed.
    fn product_at_scale(&self) -> Matrix {
        match self.b.product_in_range(&self.a) {
            Some(direct) => at_scale(
                direct.rows,
                direct.cols,
                direct.data.iter().copied().map(Wide::new),
            ),
            None => {
                let wide = WideMatrix::of(&self.b).times(&WideMatrix::of(&self.a));
                at_scale(wide.rows, wide.cols, wide.data.iter().copied())
            }
        }
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
/// Each effect `B (A X)` and its norm are computed as `f64` computes them,
/// rounding every step the same way, but with an exponent range `f64` does
/// not bound; only the ratio is rounded into range. Finite inputs whose
/// effects would overflow or underflow (updates and inputs of `1e100`, or of
/// `1e-100`) therefore still give their ratio, and an entry far smaller than
/// the largest of its matrix keeps its effect: an input of `1e-30` beside
/// one of `1e300` that no update reads still moves the output by `1e-30`.
/// Where no intermediate result leaves the range of `f64` the effects are
/// the direct products.
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
    let new_norm = effect_norm(candidate, activations)?;
    let old_norm = effect_norm(earlier, activations)?;
    if old_norm.is_zero() {
        return Ok(None);
    }
    let ratio = new_norm.ratio(old_norm);
    if !ratio.is_finite() {
        return Err(LineageError::NonFinite {
            field: "activation interference",
        });
    }
    Ok(Some(ratio))
}

/// `||B (A X)||_F`, every product entry the in-order sum of products and the
/// norm the root of the in-order sum of squares, rounded as `f64` rounds
/// them but with an unbounded exponent range. The direct products are used
/// when they provably are exactly that ([`Matrix::product_in_range`]).
fn effect_norm(update: &LayerUpdate, inputs: &Matrix) -> Result<Wide, LineageError> {
    if update.a.cols != inputs.rows {
        return Err(LineageError::ShapeMismatch {
            field: "matrix product",
            expected: update.a.cols,
            actual: inputs.rows,
        });
    }
    let direct = update
        .a
        .product_in_range(inputs)
        .and_then(|reads| update.b.product_in_range(&reads));
    let effect = match direct {
        Some(effect) => WideMatrix::of(&effect),
        None => WideMatrix::of(&update.b)
            .times(&WideMatrix::of(&update.a).times(&WideMatrix::of(inputs))),
    };
    Ok(effect
        .data
        .iter()
        .fold(Wide::ZERO, |sum, &entry| sum + entry * entry)
        .sqrt())
}

/// A matrix of [`Wide`] entries, for products whose entries leave the range
/// of `f64`.
struct WideMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Wide>,
}

impl WideMatrix {
    fn of(matrix: &Matrix) -> Self {
        Self {
            rows: matrix.rows,
            cols: matrix.cols,
            data: matrix.data.iter().copied().map(Wide::new).collect(),
        }
    }

    /// Entry `(row, col)` of `self * other`: the in-order sum of products,
    /// skipping zero terms as [`Matrix::product`] does.
    fn entry_of_product(&self, other: &WideMatrix, row: usize, col: usize) -> Wide {
        (0..self.cols)
            .map(|inner| self.data[row * self.cols + inner])
            .enumerate()
            .filter(|(_, left)| !left.is_zero())
            .fold(Wide::ZERO, |sum, (inner, left)| {
                sum + left * other.data[inner * other.cols + col]
            })
    }

    /// `self * other`, for matching shapes.
    fn times(&self, other: &WideMatrix) -> WideMatrix {
        debug_assert_eq!(self.cols, other.rows);
        let data = (0..self.rows * other.cols)
            .map(|index| self.entry_of_product(other, index / other.cols, index % other.cols))
            .collect();
        WideMatrix {
            rows: self.rows,
            cols: other.cols,
            data,
        }
    }
}

/// The `rows x cols` matrix of `entries` divided by the power of two at or
/// below their largest magnitude (subnormal magnitudes included), as `f64`
/// entries below two in magnitude: exact for entries within `2^1022` of the
/// largest, and rounded once, to a subnormal value or zero, for smaller ones.
/// All-zero entries stay zero. The result depends only on the values of the
/// entries, and a positive scale changes no span or basis.
fn at_scale<I>(rows: usize, cols: usize, entries: I) -> Matrix
where
    I: Iterator<Item = Wide> + Clone,
{
    let shift = entries
        .clone()
        .filter_map(Wide::exponent)
        .max()
        .map_or(0, |largest| -largest);
    Matrix {
        rows,
        cols,
        data: entries
            .map(|entry| entry.times_power_of_two(shift).to_f64())
            .collect(),
    }
}

/// The rows `x_k` of a matrix in an orthonormal basis `q_l` of their span:
/// `x_k = sum_l coefficients[k][l] q_l + e_k`, by modified Gram-Schmidt with
/// one reorthogonalisation pass. `e_k` is rounding error, except where what
/// orthogonalisation left of `x_k` was at most [`FACTOR_TOLERANCE`] of
/// `|x_k|` and was left out rather than made a basis vector: a row is never
/// left out for being small beside another row.
struct RowSpan {
    /// The dimension of the rows.
    dim: usize,
    /// `|x_k|` for each row.
    norms: Vec<f64>,
    /// One entry per row, each holding at least one coefficient per basis
    /// vector.
    coefficients: Vec<Vec<f64>>,
    /// The orthonormal basis.
    basis: Vec<Vec<f64>>,
}

impl RowSpan {
    fn of(matrix: &Matrix) -> Self {
        let dim = matrix.cols;
        let mut norms = Vec::with_capacity(matrix.rows);
        let mut coefficients = vec![vec![0.0; matrix.rows]; matrix.rows];
        let mut basis: Vec<Vec<f64>> = Vec::new();
        for (row, (x, coefficients)) in matrix
            .data
            .chunks_exact(dim)
            .zip(coefficients.iter_mut())
            .enumerate()
        {
            let original = norm(x);
            norms.push(original);
            if original == 0.0 {
                continue;
            }
            let mut vector = x.to_vec();
            for _ in 0..2 {
                for (coefficient, base) in coefficients.iter_mut().zip(&basis) {
                    let projection = dot(&vector, base);
                    *coefficient += projection;
                    for (cell, &b) in vector.iter_mut().zip(base) {
                        *cell -= projection * b;
                    }
                }
            }
            let remainder = norm(&vector);
            if remainder <= FACTOR_TOLERANCE * original {
                continue;
            }
            vector.iter_mut().for_each(|cell| *cell /= remainder);
            // At most one basis vector per row so far, so this is in range.
            debug_assert!(basis.len() <= row);
            coefficients[basis.len()] = remainder;
            basis.push(vector);
        }
        Self {
            dim,
            norms,
            coefficients,
            basis,
        }
    }

    /// For each column `j` of `weights`, whose rows match the rows `x_k`:
    /// the coordinates in this basis of `sum_k weights[k][j] x_k`, summed
    /// over `k` in order, as one row of the returned matrix; and the largest
    /// over `j` of `sum_k |weights[k][j]| |x_k|`, the size such a
    /// combination could have without cancellation.
    fn combinations(&self, weights: &Matrix) -> (Matrix, f64) {
        debug_assert_eq!(weights.rows, self.norms.len());
        let (count, rank) = (weights.cols, self.basis.len());
        let mut data = vec![0.0; count * rank];
        let mut uncancelled = vec![0.0; count];
        for ((row, coefficients), size) in weights
            .data
            .chunks_exact(count)
            .zip(&self.coefficients)
            .zip(&self.norms)
        {
            for (j, &weight) in row.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                uncancelled[j] += weight.abs() * size;
                for (cell, &coefficient) in
                    data[j * rank..(j + 1) * rank].iter_mut().zip(coefficients)
                {
                    *cell += weight * coefficient;
                }
            }
        }
        let largest = uncancelled.into_iter().fold(0.0, f64::max);
        (
            Matrix {
                rows: count,
                cols: rank,
                data,
            },
            largest,
        )
    }

    /// The vectors `sum_l w_l q_l` for the vectors `w` of an orthonormal
    /// basis of coordinates in this basis: orthonormal as well, up to
    /// rounding, since the `q_l` are.
    fn lift(&self, coordinates: &Basis) -> Basis {
        let vectors = coordinates
            .vectors
            .iter()
            .map(|weights| {
                let mut vector = vec![0.0; self.dim];
                for (&weight, base) in weights.iter().zip(&self.basis) {
                    for (cell, &q) in vector.iter_mut().zip(base) {
                        *cell += weight * q;
                    }
                }
                vector
            })
            .collect();
        Basis {
            dim: self.dim,
            vectors,
        }
    }
}

/// `coordinates` divided by the power of two at or below its largest entry,
/// when its largest row is at least `1 / MAX_CANCELLATION` of `uncancelled`,
/// the largest size a row could have without cancellation; `None`
/// otherwise. The matrix must have at least one column.
fn within_cancellation(mut coordinates: Matrix, uncancelled: f64) -> Option<Matrix> {
    let exponent = coordinates.rescale();
    let largest = coordinates
        .data
        .chunks_exact(coordinates.cols)
        .map(norm)
        .fold(0.0, f64::max);
    let uncancelled = scale::times_power_of_two(uncancelled, -exponent);
    (uncancelled <= MAX_CANCELLATION * largest).then_some(coordinates)
}

/// Worst overlap of one layer of a candidate adapter with any earlier adapter.
///
/// Each side is reported as one comparison: the largest overlap with any
/// earlier update on the layer and the chance level of that same comparison,
/// so the two can be compared with each other. The output and the input side
/// may come from different earlier updates.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerInterference {
    pub layer: String,
    /// The largest [`subspace_overlap`] of the output spaces with an earlier
    /// update on this layer; zero when there is none.
    pub output_overlap: f64,
    /// The largest [`subspace_overlap`] of the input spaces with an earlier
    /// update on this layer; zero when there is none.
    pub input_overlap: f64,
    /// [`chance_overlap`] of the comparison that produced `output_overlap`:
    /// of earlier updates with that overlap, the lowest chance level, which
    /// the overlap exceeds the most. Zero when no earlier update is on this
    /// layer.
    pub output_chance: f64,
    /// [`chance_overlap`] of the comparison that produced `input_overlap`,
    /// chosen as for `output_chance`.
    pub input_chance: f64,
    /// The first earlier adapter, in the order given, with the largest
    /// overlap on this layer on either side; `None` when every overlap is
    /// zero.
    pub worst: Option<AdapterId>,
}

/// Per-layer interference of a candidate with an existing lineage.
#[derive(Clone, Debug, PartialEq)]
pub struct InterferenceReport {
    /// The adapter whose updates were measured. [`measure_interference`]
    /// sets it, and a store records the report as this adapter's evidence
    /// only (`ptr-pg` refuses it under any other). That catches a report
    /// passed with the wrong adapter; it proves no provenance, since the
    /// field is public and a report relabelled or built by hand names
    /// whatever it was given.
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
/// of earlier adapters, layer by layer. The report names `candidate`, so a
/// store can refuse it under any other adapter.
///
/// This is the quantity orthogonal-subspace continual-learning methods
/// (O-LoRA, InfLoRA) drive towards zero. It is measured between subspaces, not
/// against the null space of a summed update: the null space of `sum_i dW_i` is
/// not the intersection of the individual null spaces, so an update inside it
/// can still overlap every earlier adapter.
///
/// The bases of each update are computed once, and only for earlier updates
/// on a layer the candidate updates. Their cost is that of
/// [`LayerUpdate::input_basis`]: `O((d_out + d_in) r^2)` time and
/// `O((d_out + d_in) r)` memory for an update whose bases come from its
/// factors, and `O(d_out d_in r)` time with memory for a few `d_out x d_in`
/// matrices for one whose factors are not smaller than the product, cancel,
/// or spread over more than `2^400`, whose bases come from the formed
/// product.
pub fn measure_interference(
    candidate: &AdapterId,
    updates: &[LayerUpdate],
    earlier: &[(AdapterId, Vec<LayerUpdate>)],
) -> Result<InterferenceReport, LineageError> {
    let previous: Vec<(&AdapterId, &str, (Basis, Basis))> = earlier
        .iter()
        .flat_map(|(id, previous)| previous.iter().map(move |update| (id, update)))
        .filter(|(_, previous)| updates.iter().any(|update| update.layer == previous.layer))
        .map(|(id, previous)| (id, previous.layer.as_str(), previous.update_bases()))
        .collect();
    let mut layers = Vec::with_capacity(updates.len());
    for update in updates {
        let (output, input) = update.update_bases();
        let mut worst: Option<AdapterId> = None;
        // (overlap, chance level) of the comparison reported on each side.
        let mut output_side: Option<(f64, f64)> = None;
        let mut input_side: Option<(f64, f64)> = None;
        let largest = |side: Option<(f64, f64)>| side.map_or(0.0, |(overlap, _)| overlap);
        for (id, _, (previous_output, previous_input)) in previous
            .iter()
            .filter(|(_, layer, _)| *layer == update.layer)
        {
            let out = (
                subspace_overlap(&output, previous_output)?,
                chance_overlap(&output, previous_output)?,
            );
            let inp = (
                subspace_overlap(&input, previous_input)?,
                chance_overlap(&input, previous_input)?,
            );
            if out.0.max(inp.0) > largest(output_side).max(largest(input_side)) {
                worst = Some((*id).clone());
            }
            output_side = Some(stronger(output_side, out));
            input_side = Some(stronger(input_side, inp));
        }
        let (output_overlap, output_chance) = output_side.unwrap_or((0.0, 0.0));
        let (input_overlap, input_chance) = input_side.unwrap_or((0.0, 0.0));
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

/// Of the comparison kept so far on one side of a layer and a new one, each
/// `(overlap, chance level)`, the one to report: the larger overlap, and of
/// equal overlaps the lower chance level, which the overlap exceeds the most.
/// The kept one wins a full tie, so the first in order is reported.
fn stronger(kept: Option<(f64, f64)>, new: (f64, f64)) -> (f64, f64) {
    match kept {
        Some(kept) if kept.0 > new.0 || (kept.0 == new.0 && kept.1 <= new.1) => kept,
        _ => new,
    }
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
    fn the_wide_product_is_the_direct_product_wherever_that_stays_in_range() {
        // Deterministic entries of mixed signs and magnitudes, with zeros.
        let entries = |count: usize, seed: u64| -> Vec<f64> {
            let mut state = seed;
            (0..count)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
                    match state % 5 {
                        0 => 0.0,
                        1 => unit * 1e-150,
                        2 => -unit * 1e150,
                        _ => unit - 0.5,
                    }
                })
                .collect()
        };
        let left = matrix(3, 4, &entries(12, 1));
        let right = matrix(4, 5, &entries(20, 2));
        let direct = left.product_in_range(&right).expect("in range");
        let wide = WideMatrix::of(&left).times(&WideMatrix::of(&right));
        let wide: Vec<f64> = wide.data.iter().map(|entry| entry.to_f64()).collect();
        assert_eq!(wide.as_slice(), direct.as_slice());
        assert_eq!(left.multiply(&right).unwrap(), direct);
        // Out of range, the direct product is not taken for the exact one.
        assert_eq!(
            matrix(1, 1, &[1e-200]).product_in_range(&matrix(1, 1, &[1e-200])),
            None
        );
        assert_eq!(
            matrix(1, 1, &[1e200]).product_in_range(&matrix(1, 1, &[1e200])),
            None
        );
    }

    #[test]
    fn equal_products_give_equal_bases_whether_formed_directly_or_wide() {
        let tiny = f64::MIN_POSITIVE;
        // B A = [tiny / 2; tiny / 4], subnormal after the normal terms cancel:
        // formed directly, since every term is at least MIN_POSITIVE.
        let cancelling = LayerUpdate::new(
            "q",
            matrix(2, 2, &[1.5 * tiny, tiny, 1.25 * tiny, tiny]),
            matrix(2, 1, &[1.0, -1.0]),
        )
        .unwrap();
        // The same product from subnormal factors, formed wide.
        let subnormal = LayerUpdate::new(
            "q",
            matrix(2, 1, &[tiny / 2.0, tiny / 4.0]),
            matrix(1, 1, &[1.0]),
        )
        .unwrap();
        assert!(cancelling.b.product_in_range(&cancelling.a).is_some());
        assert!(subnormal.b.product_in_range(&subnormal.a).is_none());
        assert_eq!(cancelling.product_at_scale(), subnormal.product_at_scale());
        assert_eq!(cancelling.product_at_scale().as_slice(), &[1.0, 0.5]);
        assert_eq!(cancelling.update_bases(), subnormal.update_bases());
        assert_eq!(cancelling.output_basis().rank(), 1);
    }

    /// Deterministic entries in `[-1, 1)`.
    fn pseudo_random(count: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    /// Equal ranks and a full overlap on each side, and orthonormal vectors
    /// in the first pair of bases.
    fn assert_same_spaces(bases: &(Basis, Basis), expected: &(Basis, Basis)) {
        for (basis, expected) in [(&bases.0, &expected.0), (&bases.1, &expected.1)] {
            assert_eq!(
                (basis.dim(), basis.rank()),
                (expected.dim(), expected.rank())
            );
            if basis.rank() > 0 {
                let overlap = subspace_overlap(basis, expected).unwrap();
                assert!((overlap - 1.0).abs() < 1e-12, "{overlap}");
            }
            for (i, u) in basis.vectors.iter().enumerate() {
                for (j, v) in basis.vectors.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((dot(u, v) - expected).abs() < 1e-12, "{i} {j}");
                }
            }
        }
    }

    #[test]
    fn bases_from_the_factors_are_those_of_the_formed_product_whatever_the_shape_or_rank() {
        // (d_out, r, d_in), including an inner dimension above both sides.
        for (d_out, r, d_in) in [(7, 3, 5), (2, 4, 3), (40, 6, 25), (1, 2, 9), (9, 1, 1)] {
            let b = pseudo_random(d_out * r, 1);
            let a = pseudo_random(r * d_in, 2);
            let update = LayerUpdate::new("q", matrix(d_out, r, &b), matrix(r, d_in, &a)).unwrap();
            let factored = update
                .factored_bases()
                .expect("random factors do not cancel");
            assert_same_spaces(&factored, &update.product_bases());
            let rank = d_out.min(r).min(d_in);
            assert_eq!((factored.0.rank(), factored.1.rank()), (rank, rank));
        }
        // Rank-deficient factors: a column of B twice another, and a zero
        // row of A, so B A has rank r - 2.
        let (d_out, r, d_in) = (12, 4, 10);
        let mut b = pseudo_random(d_out * r, 3);
        for row in 0..d_out {
            b[row * r + 1] = 2.0 * b[row * r];
        }
        let mut a = pseudo_random(r * d_in, 4);
        a[3 * d_in..].fill(0.0);
        let update = LayerUpdate::new("q", matrix(d_out, r, &b), matrix(r, d_in, &a)).unwrap();
        let factored = update.factored_bases().expect("no cancellation");
        assert_same_spaces(&factored, &update.product_bases());
        assert_eq!((factored.0.rank(), factored.1.rank()), (2, 2));
        // The rank tolerance is relative to the product, not to the factors:
        // with B = diag(1, s) and A = I, a direction of s = 1e-9 is kept and
        // one of 1e-11 is not, whatever the path, although the factors keep
        // both.
        for (small, rank) in [(1e-9, 2), (1e-11, 1)] {
            let update = LayerUpdate::new(
                "q",
                matrix(2, 2, &[1.0, 0.0, 0.0, small]),
                matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            )
            .unwrap();
            let factored = update.factored_bases().expect("no cancellation");
            assert_same_spaces(&factored, &update.product_bases());
            assert_eq!((factored.0.rank(), factored.1.rank()), (rank, rank));
        }
    }

    #[test]
    fn factors_that_cancel_or_spread_too_far_are_measured_on_their_formed_product() {
        // An 8 x 8 update of rank 2, (8 + 8) 2 < 8 * 8, so thin enough to be
        // measured from its factors: B holds `top` in its first two rows and
        // zeros below, and A the all-ones row times `first` and `second`, so
        // B A = (first b_0 + second b_1) 1^T.
        let thin = |top: [f64; 4], first: f64, second: f64| {
            let mut b = vec![0.0; 16];
            b[..4].copy_from_slice(&top);
            let mut a = vec![first; 8];
            a.extend([second; 8]);
            LayerUpdate::new("q", matrix(8, 2, &b), matrix(2, 8, &a)).unwrap()
        };
        let from_the_factors = |update: &LayerUpdate| {
            let factored = update.factored_bases().expect("within both checks");
            assert_same_spaces(&factored, &update.product_bases());
            assert_eq!(update.update_bases(), factored);
        };
        let from_the_product = |update: &LayerUpdate| {
            assert_eq!(update.factored_bases(), None);
            assert_eq!(update.update_bases(), update.product_bases());
        };
        // B A = (1 - (1 - 2^-7)) e1 1^T = 2^-7 e1 1^T, 255 times below the
        // size of its terms, (2 - 2^-7) e1 1^T: from the factors. With 2^-8,
        // 511 times below: on the formed product.
        from_the_factors(&thin([1.0, -1.0, 0.0, 0.0], 1.0, 1.0 - 2f64.powi(-7)));
        from_the_product(&thin([1.0, -1.0, 0.0, 0.0], 1.0, 1.0 - 2f64.powi(-8)));
        // Terms that cancel exactly leave no update.
        let cancelled = thin([1.0, -1.0, 0.0, 0.0], 1.0, 1.0);
        from_the_product(&cancelled);
        assert_eq!(
            (
                cancelled.output_basis().rank(),
                cancelled.input_basis().rank()
            ),
            (0, 0)
        );
        // b_0 = e1 + 1e-12 e2 and b_1 = e1: B A = 1e-12 e2 1^T, 2e12 below its
        // terms, of which a basis of col(B) keeps only rounding.
        let steep = thin([1.0, 1.0, 1e-12, 0.0], 1.0, -1.0);
        from_the_product(&steep);
        let mut e2 = vec![0.0; 8];
        e2[1] = 1.0;
        assert_eq!(steep.output_basis().vectors, vec![e2.clone()]);
        // A factor spread over 400 powers of two is measured from the
        // factors, and one spread over 401 on the formed product; so is
        // b_0 = MAX e1 + MIN_POSITIVE e2, b_1 = MAX e1, spread over 2045.
        let spread = |exponent: i32| thin([1.0, 2f64.powi(exponent), 0.0, 0.0], 1.0, 1.0);
        from_the_factors(&spread(-400));
        from_the_product(&spread(-401));
        let extreme = thin([f64::MAX, f64::MAX, f64::MIN_POSITIVE, 0.0], 1.0, -1.0);
        from_the_product(&extreme);
        assert_eq!(extreme.output_basis().vectors, vec![e2]);
    }

    #[test]
    fn bases_come_from_the_factors_only_where_the_factors_are_smaller_than_the_product() {
        // (d_out + d_in) r against d_out d_in: 20 >= 6, 20 >= 9, 36 >= 35 and
        // 10 >= 9 are formed, 390 < 1000 and 8 < 16 are not.
        for (d_out, r, d_in, thinner) in [
            (2, 4, 3, false),
            (1, 2, 9, false),
            (7, 3, 5, false),
            (9, 1, 1, false),
            (40, 6, 25, true),
            (4, 1, 4, true),
        ] {
            let b = pseudo_random(d_out * r, 5);
            let a = pseudo_random(r * d_in, 6);
            let update = LayerUpdate::new("q", matrix(d_out, r, &b), matrix(r, d_in, &a)).unwrap();
            let factored = update
                .factored_bases()
                .expect("random factors do not cancel");
            let expected = if thinner {
                factored
            } else {
                update.product_bases()
            };
            assert_eq!(update.update_bases(), expected, "{d_out} {r} {d_in}");
        }
    }

    #[test]
    fn an_update_whose_every_term_is_zero_has_empty_bases_from_the_factors() {
        // B's nonzero column meets A's zero row, and B's zero column A's
        // nonzero row: every b_ik a_kj is zero.
        let update = LayerUpdate::new(
            "q",
            matrix(3, 2, &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]),
            matrix(2, 2, &[0.0, 0.0, 4.0, 5.0]),
        )
        .unwrap();
        let factored = update.factored_bases().expect("nothing cancels");
        assert_eq!(factored, update.product_bases());
        assert_eq!(
            (
                factored.0.dim(),
                factored.0.rank(),
                factored.1.dim(),
                factored.1.rank()
            ),
            (3, 0, 2, 0)
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
