//! Minimal row-major `f32` matrix and a deterministic seeded initializer.
//!
//! This is a numerical reference, not a performance kernel: every operation is a
//! plain loop with a fixed summation order, so results are reproducible bit for
//! bit on one platform. Transcendental functions (`exp`, `tanh`) come from the
//! platform `libm` and may differ in the last ulp across platforms.

use std::fmt;

/// Upper bound on the element count of one matrix (Invariant 8: bounded inputs).
pub const MAX_ELEMENTS: usize = 1 << 24;

/// A shape or bounds violation. Nothing is broadcast or truncated silently.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShapeError {
    /// A matrix dimension is zero.
    ZeroDimension { rows: usize, cols: usize },
    /// The element count exceeds [`MAX_ELEMENTS`].
    TooLarge { rows: usize, cols: usize },
    /// The data length is not `rows * cols`.
    DataLength {
        rows: usize,
        cols: usize,
        len: usize,
    },
    /// Inner dimensions of a matrix product disagree.
    MatMul {
        left: (usize, usize),
        right: (usize, usize),
    },
    /// Two operands of an element-wise operation differ in shape.
    Elementwise {
        op: &'static str,
        left: (usize, usize),
        right: (usize, usize),
    },
    /// A column range lies outside the matrix or is empty.
    ColumnRange {
        start: usize,
        end: usize,
        cols: usize,
    },
    /// A row index lies outside the matrix.
    RowOutOfRange { row: usize, rows: usize },
    /// A bias / gain was expected to be a `1 x cols` row vector.
    NotRowVector {
        expected_cols: usize,
        found: (usize, usize),
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { rows, cols } => write!(f, "zero dimension {rows}x{cols}"),
            Self::TooLarge { rows, cols } => write!(f, "matrix {rows}x{cols} exceeds bound"),
            Self::DataLength { rows, cols, len } => {
                write!(f, "data length {len} is not {rows}x{cols}")
            }
            Self::MatMul { left, right } => write!(f, "matmul {left:?} x {right:?}"),
            Self::Elementwise { op, left, right } => write!(f, "{op}: {left:?} vs {right:?}"),
            Self::ColumnRange { start, end, cols } => {
                write!(f, "column range {start}..{end} of {cols}")
            }
            Self::RowOutOfRange { row, rows } => write!(f, "row {row} of {rows}"),
            Self::NotRowVector {
                expected_cols,
                found,
            } => write!(f, "expected 1x{expected_cols}, found {found:?}"),
        }
    }
}

impl std::error::Error for ShapeError {}

/// Row-major dense `f32` matrix with non-zero, bounded dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    /// Build a matrix from row-major data.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, ShapeError> {
        Self::check_dims(rows, cols)?;
        if data.len() != rows * cols {
            return Err(ShapeError::DataLength {
                rows,
                cols,
                len: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    fn check_dims(rows: usize, cols: usize) -> Result<(), ShapeError> {
        if rows == 0 || cols == 0 {
            return Err(ShapeError::ZeroDimension { rows, cols });
        }
        match rows.checked_mul(cols) {
            Some(n) if n <= MAX_ELEMENTS => Ok(()),
            _ => Err(ShapeError::TooLarge { rows, cols }),
        }
    }

    /// All-zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, ShapeError> {
        Self::filled(rows, cols, 0.0)
    }

    /// Matrix with every element equal to `value`.
    pub fn filled(rows: usize, cols: usize, value: f32) -> Result<Self, ShapeError> {
        Self::check_dims(rows, cols)?;
        Ok(Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        })
    }

    /// Identity matrix of size `n`.
    pub fn identity(n: usize) -> Result<Self, ShapeError> {
        let mut m = Self::zeros(n, n)?;
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        Ok(m)
    }

    /// Row count.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Column count.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Row-major element slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable row-major element slice (shape cannot change through it).
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Element at `(r, c)`, if in range.
    pub fn get(&self, r: usize, c: usize) -> Option<f32> {
        if r < self.rows && c < self.cols {
            self.data.get(r * self.cols + c).copied()
        } else {
            None
        }
    }

    /// One row, if in range.
    pub fn row(&self, r: usize) -> Option<&[f32]> {
        if r < self.rows {
            self.data.get(r * self.cols..(r + 1) * self.cols)
        } else {
            None
        }
    }

    /// One mutable row, if in range.
    pub fn row_mut(&mut self, r: usize) -> Option<&mut [f32]> {
        if r < self.rows {
            let cols = self.cols;
            self.data.get_mut(r * cols..(r + 1) * cols)
        } else {
            None
        }
    }

    /// Whether every element is finite.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }

    /// Matrix product with a fixed `i, p, j` accumulation order.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, ShapeError> {
        if self.cols != rhs.rows {
            return Err(ShapeError::MatMul {
                left: self.shape(),
                right: rhs.shape(),
            });
        }
        let (k, n) = (self.cols, rhs.cols);
        let mut out = vec![0.0_f32; self.rows * n];
        for (a_row, out_row) in self.data.chunks_exact(k).zip(out.chunks_exact_mut(n)) {
            for (&a, b_row) in a_row.iter().zip(rhs.data.chunks_exact(n)) {
                for (o, &b) in out_row.iter_mut().zip(b_row) {
                    *o += a * b;
                }
            }
        }
        Self::new(self.rows, n, out)
    }

    /// Element-wise sum of equally shaped matrices.
    pub fn add(&self, rhs: &Self) -> Result<Self, ShapeError> {
        if self.shape() != rhs.shape() {
            return Err(ShapeError::Elementwise {
                op: "add",
                left: self.shape(),
                right: rhs.shape(),
            });
        }
        let data = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a + b)
            .collect();
        Self::new(self.rows, self.cols, data)
    }

    /// Add a `1 x cols` row vector to every row.
    pub fn add_row_vector(&self, bias: &Self) -> Result<Self, ShapeError> {
        if bias.shape() != (1, self.cols) {
            return Err(ShapeError::NotRowVector {
                expected_cols: self.cols,
                found: bias.shape(),
            });
        }
        let mut out = self.clone();
        for row in out.data.chunks_exact_mut(self.cols) {
            for (x, b) in row.iter_mut().zip(&bias.data) {
                *x += b;
            }
        }
        Ok(out)
    }

    /// Multiply every element by `factor`.
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * factor).collect(),
        }
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut data = vec![0.0_f32; self.data.len()];
        for (r, row) in self.data.chunks_exact(self.cols).enumerate() {
            for (c, &x) in row.iter().enumerate() {
                data[c * self.rows + r] = x;
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Row-wise softmax with max-subtraction.
    ///
    /// `-inf` entries receive exactly zero weight. A row whose entries are all
    /// `-inf` has no defined distribution and yields all zeros; callers that need
    /// a distribution must reject such rows first (the attention layer does).
    pub fn softmax_rows(&self) -> Self {
        let mut out = self.clone();
        for row in out.data.chunks_exact_mut(self.cols) {
            softmax_in_place(row);
        }
        out
    }

    /// Row-wise layer normalisation with `1 x cols` gain and bias.
    pub fn layer_norm(&self, gamma: &Self, beta: &Self, eps: f32) -> Result<Self, ShapeError> {
        for p in [gamma, beta] {
            if p.shape() != (1, self.cols) {
                return Err(ShapeError::NotRowVector {
                    expected_cols: self.cols,
                    found: p.shape(),
                });
            }
        }
        let n = self.cols as f32;
        let mut out = self.clone();
        for row in out.data.chunks_exact_mut(self.cols) {
            let mean = row.iter().sum::<f32>() / n;
            let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
            let inv = 1.0 / (var + eps).sqrt();
            for ((x, g), b) in row.iter_mut().zip(&gamma.data).zip(&beta.data) {
                *x = (*x - mean) * inv * g + b;
            }
        }
        Ok(out)
    }

    /// Element-wise GELU (tanh approximation).
    pub fn gelu(&self) -> Self {
        self.map(gelu)
    }

    /// Element-wise hyperbolic tangent.
    pub fn tanh(&self) -> Self {
        self.map(f32::tanh)
    }

    fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| f(*x)).collect(),
        }
    }

    /// Copy of columns `start..end`.
    pub fn cols_range(&self, start: usize, end: usize) -> Result<Self, ShapeError> {
        if start >= end || end > self.cols {
            return Err(ShapeError::ColumnRange {
                start,
                end,
                cols: self.cols,
            });
        }
        let width = end - start;
        let mut data = Vec::with_capacity(self.rows * width);
        for row in self.data.chunks_exact(self.cols) {
            data.extend_from_slice(&row[start..end]);
        }
        Self::new(self.rows, width, data)
    }

    /// Overwrite columns starting at `start` with `block` (same row count).
    pub fn set_cols(&mut self, start: usize, block: &Self) -> Result<(), ShapeError> {
        let end = start.saturating_add(block.cols);
        if block.rows != self.rows || end > self.cols {
            return Err(ShapeError::ColumnRange {
                start,
                end,
                cols: self.cols,
            });
        }
        let cols = self.cols;
        for (dst, src) in self
            .data
            .chunks_exact_mut(cols)
            .zip(block.data.chunks_exact(block.cols))
        {
            dst[start..end].copy_from_slice(src);
        }
        Ok(())
    }
}

/// In-place softmax with max-subtraction; all-`-inf` input becomes all zeros.
pub fn softmax_in_place(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        row.iter_mut().for_each(|x| *x = 0.0);
        return;
    }
    let mut sum = 0.0_f32;
    for x in row.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in row.iter_mut() {
        *x /= sum;
    }
}

/// Softmax of a vector (see [`softmax_in_place`]).
pub fn softmax(values: &[f32]) -> Vec<f32> {
    let mut out = values.to_vec();
    softmax_in_place(&mut out);
    out
}

/// Logistic function.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// GELU, tanh approximation.
pub fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2 / pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// FNV-1a 64-bit hash (used for stable sub-seeds, not for security).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// SplitMix64 generator: tiny, deterministic, platform independent.
#[derive(Clone, Debug)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    /// Generator for `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generator for a named sub-stream of `seed`, independent of init order.
    pub fn for_name(seed: u64, name: &str) -> Self {
        Self::new(seed ^ fnv1a64(name.as_bytes()).rotate_left(17))
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 24 bits of resolution.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Uniform integer in `0..n` (`0` when `n == 0`).
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// Xavier/Glorot uniform initialisation: limit `sqrt(6 / (rows + cols))`.
pub fn xavier_uniform(rows: usize, cols: usize, rng: &mut SeededRng) -> Result<Matrix, ShapeError> {
    let limit = (6.0 / (rows + cols) as f32).sqrt();
    uniform_matrix(rows, cols, limit, rng)
}

/// Uniform initialisation in `[-limit, limit)`.
pub fn uniform_matrix(
    rows: usize,
    cols: usize,
    limit: f32,
    rng: &mut SeededRng,
) -> Result<Matrix, ShapeError> {
    Matrix::check_dims(rows, cols)?;
    let data = (0..rows * cols)
        .map(|_| rng.uniform(-limit, limit))
        .collect();
    Matrix::new(rows, cols, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: usize, cols: usize, d: &[f32]) -> Matrix {
        Matrix::new(rows, cols, d.to_vec()).unwrap()
    }

    #[test]
    fn matmul_matches_hand_computation() {
        // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
        let c = m(2, 2, &[1., 2., 3., 4.])
            .matmul(&m(2, 2, &[5., 6., 7., 8.]))
            .unwrap();
        assert_eq!(c.as_slice(), &[19., 22., 43., 50.]);
    }

    #[test]
    fn rectangular_matmul_and_transpose() {
        let a = m(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let c = a.matmul(&a.transpose()).unwrap();
        assert_eq!(c.shape(), (2, 2));
        assert_eq!(c.as_slice(), &[14., 32., 32., 77.]);
    }

    #[test]
    fn shape_errors_are_typed() {
        let a = m(2, 3, &[0.; 6]);
        assert!(matches!(a.matmul(&a), Err(ShapeError::MatMul { .. })));
        assert!(matches!(
            a.add(&m(3, 2, &[0.; 6])),
            Err(ShapeError::Elementwise { .. })
        ));
        assert!(matches!(
            Matrix::new(2, 2, vec![0.; 3]),
            Err(ShapeError::DataLength { .. })
        ));
        assert!(matches!(
            Matrix::zeros(0, 2),
            Err(ShapeError::ZeroDimension { .. })
        ));
        assert!(matches!(
            Matrix::zeros(MAX_ELEMENTS, 2),
            Err(ShapeError::TooLarge { .. })
        ));
        assert!(matches!(
            a.add_row_vector(&m(2, 3, &[0.; 6])),
            Err(ShapeError::NotRowVector { .. })
        ));
        assert!(matches!(
            a.cols_range(2, 2),
            Err(ShapeError::ColumnRange { .. })
        ));
        assert!(matches!(
            a.cols_range(0, 4),
            Err(ShapeError::ColumnRange { .. })
        ));
        let mut b = a.clone();
        assert!(matches!(
            b.set_cols(2, &m(2, 2, &[0.; 4])),
            Err(ShapeError::ColumnRange { .. })
        ));
        assert!(matches!(
            a.layer_norm(&m(1, 2, &[1., 1.]), &m(1, 3, &[0.; 3]), 1e-5),
            Err(ShapeError::NotRowVector { .. })
        ));
        assert!(a.row(2).is_none());
        assert!(a.get(0, 3).is_none());
    }

    #[test]
    fn softmax_rows_sum_to_one_and_survive_large_logits() {
        let x = m(2, 3, &[1000., 1001., 1002., -1., 0., 1.]);
        let s = x.softmax_rows();
        for r in 0..2 {
            let row = s.row(r).unwrap();
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!(row.iter().all(|p| p.is_finite() && *p > 0.0));
        }
    }

    #[test]
    fn softmax_gives_exact_zero_to_neg_inf_and_zeros_to_all_masked() {
        let s = softmax(&[0.0, f32::NEG_INFINITY, 0.0]);
        assert_eq!(s[1], 0.0);
        assert_eq!(s[0], 0.5);
        assert_eq!(softmax(&[f32::NEG_INFINITY; 2]), vec![0.0, 0.0]);
    }

    #[test]
    fn layer_norm_zero_mean_unit_variance() {
        let x = m(1, 4, &[1., 2., 3., 4.]);
        let g = Matrix::filled(1, 4, 1.0).unwrap();
        let b = Matrix::zeros(1, 4).unwrap();
        let y = x.layer_norm(&g, &b, 1e-9).unwrap();
        let row = y.row(0).unwrap();
        assert!(row.iter().sum::<f32>().abs() < 1e-5);
        let var = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-4);
    }

    #[test]
    fn gelu_and_tanh_reference_points() {
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-5);
        assert!((gelu(-1.0) + 0.158_808).abs() < 1e-5);
        let t = m(1, 2, &[0.0, 1.0]).tanh();
        assert!((t.as_slice()[1] - 0.761_594_2).abs() < 1e-6);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn column_slicing_roundtrip() {
        let a = m(2, 4, &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let right = a.cols_range(2, 4).unwrap();
        assert_eq!(right.as_slice(), &[3., 4., 7., 8.]);
        let mut z = Matrix::zeros(2, 4).unwrap();
        z.set_cols(2, &right).unwrap();
        assert_eq!(z.as_slice(), &[0., 0., 3., 4., 0., 0., 7., 8.]);
    }

    #[test]
    fn rng_is_deterministic_and_seed_sensitive() {
        let mut a = SeededRng::new(42);
        let mut b = SeededRng::new(42);
        let mut c = SeededRng::new(43);
        let (x, y, z) = (a.next_u64(), b.next_u64(), c.next_u64());
        assert_eq!(x, y);
        assert_ne!(x, z);
        // SplitMix64 reference value for seed 0 (first output).
        assert_eq!(SeededRng::new(0).next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_ne!(
            SeededRng::for_name(1, "a").next_u64(),
            SeededRng::for_name(1, "b").next_u64()
        );
    }

    #[test]
    fn xavier_is_bounded_and_roughly_centered() {
        let mut rng = SeededRng::new(9);
        let w = xavier_uniform(64, 32, &mut rng).unwrap();
        let limit = (6.0_f32 / 96.0).sqrt();
        assert!(w.as_slice().iter().all(|x| x.abs() <= limit));
        let mean = w.as_slice().iter().sum::<f32>() / w.as_slice().len() as f32;
        assert!(mean.abs() < 0.02);
        let w2 = xavier_uniform(64, 32, &mut SeededRng::new(9)).unwrap();
        assert_eq!(w, w2);
    }

    #[test]
    fn below_is_bounded() {
        let mut rng = SeededRng::new(1);
        assert!((0..100).all(|_| rng.below(7) < 7));
        assert_eq!(rng.below(0), 0);
    }
}
