//! Conversions between plain Python values and `ptr-lineage` types.

use ptr_lineage::{LayerUpdate, LineageError, Matrix};
use ptr_types::VerificationLevel;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pyo3::create_exception!(
    ptr_py,
    PtrLineageError,
    PyValueError,
    "A ptr-lineage refusal; the message is `<RUST_ERROR_CODE>: <Display text>`."
);

pyo3::create_exception!(
    ptr_py,
    PtrVerifierError,
    PyValueError,
    "A ptr-verifier refusal (an unknown role, an unusable task or a bad revision)."
);

pyo3::create_exception!(
    ptr_py,
    PtrRouterError,
    PyValueError,
    "A ptr-router refusal (bad exploration, registry, request, log or estimator input)."
);

pyo3::create_exception!(
    ptr_py,
    PtrWorkspaceError,
    PyValueError,
    "A ptr-pods bounded-workspace refusal (bad configuration, submission or subscriber)."
);

pyo3::create_exception!(
    ptr_py,
    PtrWorkspaceCapacityError,
    PtrWorkspaceError,
    "A bounded-workspace capacity refusal (pending area, author quota, text size or text count)."
);

pyo3::create_exception!(
    ptr_py,
    PtrVcwError,
    PyValueError,
    "A refusal of the in-process verifier-closed loop simulation (bad configuration or loop failure)."
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

/// The stable name of a verification level.
pub(crate) fn level_name(level: VerificationLevel) -> &'static str {
    match level {
        VerificationLevel::Unverified => "unverified",
        VerificationLevel::LatentAgreement => "latent_agreement",
        VerificationLevel::SampleVerified => "sample_verified",
        VerificationLevel::FullSemantic => "full_semantic",
        VerificationLevel::Deterministic => "deterministic",
    }
}

/// Lowercase hex of a digest.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}
