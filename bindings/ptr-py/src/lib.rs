//! `ptr_py`: PTR governance checks from `ptr-lineage` for Python training code.
//!
//! These are governance checks, not training code. The gate decides; Python only
//! calls it and reads the verdict. Matrices are row-major `list[list[float]]`.
//! Every `LineageError` is raised as `PtrLineageError` (a `ValueError`) whose
//! message starts with the Rust error code.

mod convert;
mod governance;

use pyo3::prelude::*;

#[pymodule]
fn ptr_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "__doc__",
        "PTR governance checks from ptr-lineage (AIMER expert scoring, subspace interference, \
         TIES merge, forgetting gate). These are governance checks, not training code: the \
         gate decides, Python only calls it. Errors raise PtrLineageError (a ValueError) \
         whose message starts with the Rust error code, e.g. PTR_LINEAGE_EMPTY.",
    )?;
    module.add(
        "PtrLineageError",
        module.py().get_type::<convert::PtrLineageError>(),
    )?;
    module.add_function(wrap_pyfunction!(governance::aimer_score, module)?)?;
    module.add_function(wrap_pyfunction!(
        governance::rank_experts_for_pruning,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(governance::subspace_overlap, module)?)?;
    module.add_function(wrap_pyfunction!(governance::chance_overlap, module)?)?;
    module.add_function(wrap_pyfunction!(governance::measure_interference, module)?)?;
    module.add_function(wrap_pyfunction!(governance::ties_merge, module)?)?;
    module.add_function(wrap_pyfunction!(
        governance::forgetting_gate_evaluate,
        module
    )?)?;
    Ok(())
}
