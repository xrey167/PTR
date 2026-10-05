//! Conversions between plain Python values and `ptr-lineage` types.

use ptr_lineage::{LayerUpdate, LineageError, Matrix};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pyo3::create_exception!(
    ptr_py,
    PtrLineageError,
    PyValueError,
    "A ptr-lineage refusal; the message is `<RUST_ERROR_CODE>: <Display text>`."
);

/// Map a Rust refusal to `PtrLineageError`, keeping its code and text.
pub(crate) fn lineage_error(error: LineageError) -> PyErr {
    PtrLineageError::new_err(format!("{}: {}", error.code(), error))
}

/// Build a checked `Matrix` from row-major nested lists. Ragged rows are refused
/// here, because `Matrix::new` takes a flat buffer and could not see them.
pub(crate) fn matrix(rows: &[Vec<f64>]) -> PyResult<Matrix> {
    let cols = rows.first().map_or(0, Vec::len);
    if let Some((index, row)) = rows.iter().enumerate().find(|(_, row)| row.len() != cols) {
        return Err(PyValueError::new_err(format!(
            "matrix is ragged: row {index} has {} entries, row 0 has {cols}",
            row.len()
        )));
    }
    let data: Vec<f64> = rows.iter().flatten().copied().collect();
    Matrix::new(rows.len(), cols, data).map_err(lineage_error)
}

/// One layer update as given from Python: `(layer, B, A)` with `delta_W = B A`.
pub(crate) type RawUpdate = (String, Vec<Vec<f64>>, Vec<Vec<f64>>);

pub(crate) fn layer_update(raw: RawUpdate) -> PyResult<LayerUpdate> {
    let (layer, b, a) = raw;
    LayerUpdate::new(layer, matrix(&b)?, matrix(&a)?).map_err(lineage_error)
}

pub(crate) fn layer_updates(raw: Vec<RawUpdate>) -> PyResult<Vec<LayerUpdate>> {
    raw.into_iter().map(layer_update).collect()
}
