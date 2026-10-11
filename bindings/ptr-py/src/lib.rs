//! `ptr_py`: PTR governance checks from `ptr-lineage` for Python training code.
//!
//! These are governance checks, not training code. The gate decides; Python only
//! calls it and reads the verdict. Matrices are row-major `list[list[float]]`.
//! Every `LineageError` is raised as `PtrLineageError` (a `ValueError`) whose
//! message starts with the Rust error code.

mod convert;
mod governance;
mod routing;
mod vcw;
mod verify;
mod workspace;

use pyo3::prelude::*;

#[pymodule]
fn ptr_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "__doc__",
        "PTR governance checks from ptr-lineage (AIMER expert scoring, subspace interference, \
         TIES merge, forgetting gate). These are governance checks, not training code: the \
         gate decides, Python only calls it. Errors raise PtrLineageError (a ValueError) \
         whose message starts with the Rust error code, e.g. PTR_LINEAGE_EMPTY. Also read-only \
         access to the VCW-MoE verifiers (verify_*, finding code table), the finding router \
         with propensity logging and off-policy estimators (FindingRouter), the bounded \
         workspace (BoundedWorkspace) and an in-process verifier-closed loop simulation \
         (simulate_vcw_run). None of them can change a gate, threshold, verifier, \
         eligibility or required level.",
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
    module.add(
        "PtrVerifierError",
        module.py().get_type::<convert::PtrVerifierError>(),
    )?;
    module.add(
        "PtrRouterError",
        module.py().get_type::<convert::PtrRouterError>(),
    )?;
    module.add(
        "PtrWorkspaceError",
        module.py().get_type::<convert::PtrWorkspaceError>(),
    )?;
    module.add(
        "PtrWorkspaceCapacityError",
        module.py().get_type::<convert::PtrWorkspaceCapacityError>(),
    )?;
    module.add(
        "PtrVcwError",
        module.py().get_type::<convert::PtrVcwError>(),
    )?;
    module.add_class::<verify::PyReport>()?;
    module.add_class::<verify::PyFinding>()?;
    module.add_class::<routing::PyFindingRouter>()?;
    module.add_class::<workspace::PyWorkspace>()?;
    module.add_function(wrap_pyfunction!(verify::verify_arithmetic, module)?)?;
    module.add_function(wrap_pyfunction!(verify::verify_json_schema, module)?)?;
    module.add_function(wrap_pyfunction!(verify::verify_code, module)?)?;
    module.add_function(wrap_pyfunction!(verify::verify_facts, module)?)?;
    module.add_function(wrap_pyfunction!(verify::finding_code_table, module)?)?;
    module.add_function(wrap_pyfunction!(verify::repair_roles, module)?)?;
    module.add_function(wrap_pyfunction!(verify::py_codes_for_role, module)?)?;
    module.add_function(wrap_pyfunction!(verify::py_repair_role_for, module)?)?;
    module.add_function(wrap_pyfunction!(routing::classify_report, module)?)?;
    module.add_function(wrap_pyfunction!(vcw::simulate_vcw_run, module)?)?;
    Ok(())
}
