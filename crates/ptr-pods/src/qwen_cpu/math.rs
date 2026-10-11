//! Numeric kernels: bf16 dot products, row-parallel matmul, norms and rotary tables.

#[inline]
pub(super) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Round-to-nearest-even `f32` to bf16 bits (used by tests that build synthetic checkpoints).
#[cfg(test)]
pub(super) fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

const LANES: usize = 8;

/// Below this many multiply-adds a product runs on the calling thread.
const PARALLEL_MIN_WORK: usize = 1 << 16;

/// Dot product of a bf16 row with an `f32` vector, eight independent accumulators.
pub(super) fn dot_bf16(row: &[u16], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len(), x.len());
    let mut acc = [0.0_f32; LANES];
    let (row_blocks, row_rest) = row.as_chunks::<LANES>();
    let (x_blocks, x_rest) = x.as_chunks::<LANES>();
    for (r, v) in row_blocks.iter().zip(x_blocks) {
        for lane in 0..LANES {
            acc[lane] += bf16_to_f32(r[lane]) * v[lane];
        }
    }
    let mut total: f32 = acc.iter().sum();
    for (r, v) in row_rest.iter().zip(x_rest) {
        total += bf16_to_f32(*r) * v;
    }
    total
}

/// Dot product of two `f32` vectors, eight independent accumulators.
pub(super) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0_f32; LANES];
    let (a_blocks, a_rest) = a.as_chunks::<LANES>();
    let (b_blocks, b_rest) = b.as_chunks::<LANES>();
    for (l, r) in a_blocks.iter().zip(b_blocks) {
        for lane in 0..LANES {
            acc[lane] += l[lane] * r[lane];
        }
    }
    let mut total: f32 = acc.iter().sum();
    for (l, r) in a_rest.iter().zip(b_rest) {
        total += l * r;
    }
    total
}

/// A dense `rows x cols` matrix of bf16 values, row-major (the layout of a Hugging Face linear
/// weight: one row per output feature).
pub(super) struct Bf16Matrix {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) data: Vec<u16>,
}

impl Bf16Matrix {
    /// `out[r * t + i] = row_r . xs[i] (+ bias[r])` for `t` input vectors of length `cols`
    /// stored back to back in `xs`. `out` has `rows * t` entries. Rows are split across
    /// `threads` scoped threads; each owns a contiguous slice of `out`.
    pub(super) fn matmul(
        &self,
        xs: &[f32],
        t: usize,
        bias: Option<&[f32]>,
        threads: usize,
        out: &mut [f32],
    ) {
        debug_assert_eq!(xs.len(), t * self.cols);
        debug_assert_eq!(out.len(), self.rows * t);
        let work = self.rows * self.cols * t;
        let threads = if work < PARALLEL_MIN_WORK {
            1
        } else {
            threads.clamp(1, self.rows)
        };
        let rows_per = self.rows.div_ceil(threads);
        if threads == 1 {
            self.matmul_rows(0, xs, t, bias, out);
            return;
        }
        std::thread::scope(|scope| {
            for (chunk_index, chunk) in out.chunks_mut(rows_per * t).enumerate() {
                scope.spawn(move || {
                    self.matmul_rows(chunk_index * rows_per, xs, t, bias, chunk);
                });
            }
        });
    }

    fn matmul_rows(
        &self,
        first_row: usize,
        xs: &[f32],
        t: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let rows = out.len() / t;
        if t == 1 {
            for (local, slot) in out.iter_mut().enumerate() {
                let r = first_row + local;
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                *slot = dot_bf16(row, xs) + bias.map_or(0.0, |b| b[r]);
            }
            return;
        }
        let mut converted = vec![0.0_f32; self.cols];
        for local in 0..rows {
            let r = first_row + local;
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            for (dst, src) in converted.iter_mut().zip(row) {
                *dst = bf16_to_f32(*src);
            }
            let offset = bias.map_or(0.0, |b| b[r]);
            for i in 0..t {
                out[local * t + i] =
                    dot_f32(&converted, &xs[i * self.cols..(i + 1) * self.cols]) + offset;
            }
        }
    }
}

/// `x * rsqrt(mean(x^2) + eps) * weight`, in `f32`.
pub(super) fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for ((o, v), w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * scale * w;
    }
}

#[inline]
pub(super) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// In-place softmax with the usual max shift.
pub(super) fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in values.iter_mut() {
        *v /= sum;
    }
}

/// `cos` and `sin` of `position * theta^(-2i/head_dim)` for `i < head_dim / 2`, computed in
/// `f64` and rounded once.
pub(super) fn rope_angles(position: usize, head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(half);
    let mut sin = Vec::with_capacity(half);
    for i in 0..half {
        let inv_freq = theta.powf(-(2.0 * i as f64) / head_dim as f64);
        let angle = position as f64 * inv_freq;
        cos.push(angle.cos() as f32);
        sin.push(angle.sin() as f32);
    }
    (cos, sin)
}

/// Rotate every head of `x` (`heads * head_dim` values) with the "rotate half" pairing used by
/// Hugging Face Qwen2: element `i` pairs with element `i + head_dim / 2`.
pub(super) fn apply_rope(x: &mut [f32], head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for head in x.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos[i] - b * sin[i];
            head[i + half] = b * cos[i] + a * sin[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trips_representable_values() {
        for value in [0.0_f32, 1.0, -2.5, 0.15625, 1024.0] {
            assert_eq!(bf16_to_f32(f32_to_bf16(value)), value);
        }
    }

    #[test]
    fn matmul_matches_a_naive_product_with_and_without_threads() {
        // Large enough that four threads actually split the work (see `PARALLEL_MIN_WORK`).
        let (rows, cols, t) = (301, 257, 3);
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) / 4.0)
            .collect();
        let matrix = Bf16Matrix {
            rows,
            cols,
            data: weights.iter().map(|w| f32_to_bf16(*w)).collect(),
        };
        let xs: Vec<f32> = (0..t * cols)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) / 3.0)
            .collect();
        let bias: Vec<f32> = (0..rows).map(|r| r as f32 * 0.01).collect();
        let mut expected = vec![0.0_f32; rows * t];
        for r in 0..rows {
            for i in 0..t {
                let mut s = bias[r];
                for c in 0..cols {
                    s += bf16_to_f32(matrix.data[r * cols + c]) * xs[i * cols + c];
                }
                expected[r * t + i] = s;
            }
        }
        for threads in [1, 4] {
            let mut out = vec![0.0_f32; rows * t];
            matrix.matmul(&xs, t, Some(&bias), threads, &mut out);
            for (a, b) in out.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-3, "{a} vs {b}");
            }
            let mut single = vec![0.0_f32; rows];
            matrix.matmul(&xs[..cols], 1, Some(&bias), threads, &mut single);
            for r in 0..rows {
                assert!((single[r] - expected[r * t]).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn rope_at_position_zero_is_the_identity_and_preserves_norms() {
        let head_dim = 8;
        let original: Vec<f32> = (0..16).map(|i| i as f32 * 0.3 - 1.0).collect();
        let mut x = original.clone();
        let (cos, sin) = rope_angles(0, head_dim, 10_000.0);
        apply_rope(&mut x, head_dim, &cos, &sin);
        assert_eq!(x, original);
        let (cos, sin) = rope_angles(37, head_dim, 10_000.0);
        apply_rope(&mut x, head_dim, &cos, &sin);
        let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
        assert!((norm(&x) - norm(&original)).abs() < 1e-3);
    }

    #[test]
    fn softmax_sums_to_one_and_rms_norm_scales_to_unit_rms() {
        let mut values = vec![1.0, 2.0, 3.0, 1000.0];
        softmax(&mut values);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let mut out = vec![0.0; 4];
        rms_norm(&[3.0, 4.0, 0.0, 0.0], &[1.0; 4], 0.0, &mut out);
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5);
    }
}
