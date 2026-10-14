//! Affine layer shared by the attention, MLP and head modules.

use crate::tensor::{xavier_uniform, Matrix, SeededRng, ShapeError};

/// `y = x W + b` with `W: in x out` and `b: 1 x out`.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    pub(crate) w: Matrix,
    pub(crate) b: Matrix,
}

impl Linear {
    /// Xavier-initialised weights from the named sub-stream of `seed`, zero bias.
    pub(crate) fn init(
        seed: u64,
        name: &str,
        inputs: usize,
        outputs: usize,
    ) -> Result<Self, ShapeError> {
        let mut rng = SeededRng::for_name(seed, name);
        Ok(Self {
            w: xavier_uniform(inputs, outputs, &mut rng)?,
            b: Matrix::zeros(1, outputs)?,
        })
    }

    /// Weight matrix (`in x out`).
    pub fn weight(&self) -> &Matrix {
        &self.w
    }

    /// Bias row (`1 x out`).
    pub fn bias(&self) -> &Matrix {
        &self.b
    }

    /// Apply to every row of `x`.
    pub fn forward(&self, x: &Matrix) -> Result<Matrix, ShapeError> {
        x.matmul(&self.w)?.add_row_vector(&self.b)
    }

    /// Apply to one vector.
    pub fn forward_vec(&self, x: &[f32]) -> Result<Vec<f32>, ShapeError> {
        let row = Matrix::new(1, x.len(), x.to_vec())?;
        Ok(self.forward(&row)?.as_slice().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_is_xw_plus_b() {
        let l = Linear {
            w: Matrix::new(2, 2, vec![1., 2., 3., 4.]).unwrap(),
            b: Matrix::new(1, 2, vec![10., 20.]).unwrap(),
        };
        // [1,1] W = [4,6]; + b = [14,26]
        assert_eq!(l.forward_vec(&[1., 1.]).unwrap(), vec![14., 26.]);
        assert!(l.forward_vec(&[1., 1., 1.]).is_err());
    }

    #[test]
    fn init_is_deterministic_per_name() {
        let a = Linear::init(1, "x", 3, 2).unwrap();
        assert_eq!(a, Linear::init(1, "x", 3, 2).unwrap());
        assert_ne!(a, Linear::init(1, "y", 3, 2).unwrap());
    }
}
