//! The exposed functions. Each is a thin call into `ptr-lineage`; none adds a
//! threshold, a default or a decision of its own.

use ptr_lineage::{
    AccuracyMatrix, AdapterId, ExpertWeights, ForgettingGate, GateViolation, PublicSuite,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{layer_updates, lineage_error, matrix, RawUpdate};

/// AIMER score ||w||_1 / (sqrt(N) * ||w||_2) of one flattened expert weight vector.
///
/// Governance check, not training code. The score lies in [1/sqrt(N), 1]: one
/// nonzero weight scores 1/sqrt(N), equal magnitudes score 1. It only orders
/// experts; a prune must still pass the forgetting gate.
///
/// Raises PtrLineageError for empty, non-finite or all-zero weights.
#[pyfunction]
pub(crate) fn aimer_score(weights: Vec<f64>) -> PyResult<f64> {
    ptr_lineage::aimer_score(&weights).map_err(lineage_error)
}

/// Rank the experts of one layer for pruning, highest AIMER score first.
///
/// `experts` is a list of `(expert_id, flattened_weights)`; all weight lists must
/// have the same length. Returns the ids; equal scores keep input order.
/// Governance check, not training code: ranking is not a prune decision.
///
/// Raises PtrLineageError for no experts, unequal lengths or an unscorable expert.
#[pyfunction]
pub(crate) fn rank_experts_for_pruning(experts: Vec<(String, Vec<f64>)>) -> PyResult<Vec<String>> {
    let views: Vec<ExpertWeights<'_, String>> = experts
        .iter()
        .map(|(id, weights)| ExpertWeights {
            id: id.clone(),
            weights,
        })
        .collect();
    ptr_lineage::rank_experts_for_pruning(&views).map_err(lineage_error)
}

/// Normalised subspace overlap ||Qa^T Qb||_F^2 / min(rank_a, rank_b) of the
/// column spaces of two matrices.
///
/// `a` is a `dim x ka` and `b` a `dim x kb` row-major `list[list[float]]`; each
/// subspace is the column space of its matrix (Rust builds an orthonormal basis
/// with `column_basis`; a Basis cannot be built unchecked). 0 for orthogonal
/// subspaces, 1 when the smaller lies inside the larger. Compare with
/// `chance_overlap`. Governance check, not training code.
///
/// Raises PtrLineageError for empty or non-finite matrices or different `dim`.
#[pyfunction]
pub(crate) fn subspace_overlap(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<f64> {
    let left = ptr_lineage::column_basis(&matrix(&a)?);
    let right = ptr_lineage::column_basis(&matrix(&b)?);
    ptr_lineage::subspace_overlap(&left, &right).map_err(lineage_error)
}

/// Expected normalised overlap of two random subspaces: max(rank_a, rank_b) / dim.
///
/// Arguments are as for `subspace_overlap` (`dim x k` matrices whose column
/// spaces are the subspaces; ranks are the numerical ranks of the column spaces).
/// An overlap near this level is no evidence of interference. Governance check,
/// not training code.
///
/// Raises PtrLineageError for empty or non-finite matrices or different `dim`.
#[pyfunction]
pub(crate) fn chance_overlap(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<f64> {
    let left = ptr_lineage::column_basis(&matrix(&a)?);
    let right = ptr_lineage::column_basis(&matrix(&b)?);
    ptr_lineage::chance_overlap(&left, &right).map_err(lineage_error)
}

/// Per-layer interference of a candidate adapter's low-rank updates with earlier ones.
///
/// `candidate` is the adapter id. `updates` is a list of `(layer, B, A)` with
/// `delta_W = B A`, `B` of shape d_out x r and `A` of shape r x d_in (row-major
/// nested lists). `earlier` is a list of `(adapter_id, [(layer, B, A), ...])`.
/// Returns `{"candidate": str, "max_overlap": float, "layers": [{"layer",
/// "output_overlap", "input_overlap", "output_chance", "input_chance", "worst"}]}`
/// where `worst` is the earlier adapter id or None. Governance check, not
/// training code; deciding what overlap is acceptable is the lineage policy's job.
///
/// Raises PtrLineageError for a layer updated twice by the candidate or by one earlier
/// adapter, misaligned factors, or earlier updates of a layer with different sizes.
#[pyfunction]
pub(crate) fn measure_interference<'py>(
    py: Python<'py>,
    candidate: &str,
    updates: Vec<RawUpdate>,
    earlier: Vec<(String, Vec<RawUpdate>)>,
) -> PyResult<Bound<'py, PyDict>> {
    let updates = layer_updates(updates)?;
    let earlier = earlier
        .into_iter()
        .map(|(id, raw)| Ok((AdapterId(id), layer_updates(raw)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let report = ptr_lineage::measure_interference(&AdapterId::from(candidate), &updates, &earlier)
        .map_err(lineage_error)?;
    let layers = PyList::empty(py);
    for layer in &report.layers {
        let item = PyDict::new(py);
        item.set_item("layer", &layer.layer)?;
        item.set_item("output_overlap", layer.output_overlap)?;
        item.set_item("input_overlap", layer.input_overlap)?;
        item.set_item("output_chance", layer.output_chance)?;
        item.set_item("input_chance", layer.input_chance)?;
        item.set_item("worst", layer.worst.as_ref().map(ToString::to_string))?;
        layers.append(item)?;
    }
    let out = PyDict::new(py);
    out.set_item("candidate", report.candidate.to_string())?;
    out.set_item("max_overlap", report.max_overlap())?;
    out.set_item("layers", layers)?;
    Ok(out)
}

/// TIES merge of full task vectors (flattened delta_W = B A, equal length).
///
/// Trims each vector to its `density` fraction of largest-magnitude entries,
/// elects a sign per coordinate, and averages the entries agreeing with it.
/// `density` must lie in (0, 1]. The merged vector is a new candidate: it must
/// pass the forgetting gate like any trained adapter. Governance check, not
/// training code.
///
/// Raises PtrLineageError for no vectors, ragged or non-finite vectors, or a
/// bad density.
#[pyfunction]
pub(crate) fn ties_merge(vectors: Vec<Vec<f64>>, density: f64) -> PyResult<Vec<f64>> {
    ptr_lineage::ties_merge(&vectors, density).map_err(lineage_error)
}

/// Run the forgetting gate for one adapter candidate and return its verdict.
///
/// `accuracy` is the square accuracy matrix (`accuracy[i][j]` = accuracy on task
/// j after stage i; the last row is the final model). `public_serving` and
/// `public_candidate` are the public-suite scores of the serving and candidate
/// adapters. The four thresholds are keyword-only and required: average and
/// per-task forgetting may not exceed their maxima, backward transfer may not
/// fall below its minimum, and the public-suite drop (serving - candidate) may
/// not exceed `max_public_regression`. Equality passes each threshold. Scores need only
/// be finite, not probabilities.
///
/// Returns `{"adapter", "passed", "violations": [{"kind", "observed", "limit",
/// "task"?}], "average_accuracy", "backward_transfer", "average_forgetting",
/// "task_forgetting": [...]}`. `kind` is one of average_forgetting,
/// task_forgetting (with `task`), backward_transfer, public_regression. The gate
/// decides; this function adds no logic. Governance check, not training code.
///
/// Raises PtrLineageError for an empty, non-square or non-finite matrix, or a
/// non-finite threshold or public score.
#[pyfunction]
#[pyo3(signature = (
    adapter,
    accuracy,
    public_serving,
    public_candidate,
    *,
    max_average_forgetting,
    max_task_forgetting,
    min_backward_transfer,
    max_public_regression,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn forgetting_gate_evaluate<'py>(
    py: Python<'py>,
    adapter: &str,
    accuracy: Vec<Vec<f64>>,
    public_serving: f64,
    public_candidate: f64,
    max_average_forgetting: f64,
    max_task_forgetting: f64,
    min_backward_transfer: f64,
    max_public_regression: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let matrix = AccuracyMatrix::new(accuracy).map_err(lineage_error)?;
    let gate = ForgettingGate {
        max_average_forgetting,
        max_task_forgetting,
        min_backward_transfer,
        max_public_regression,
    };
    let public = PublicSuite {
        serving: public_serving,
        candidate: public_candidate,
    };
    let report = gate
        .evaluate(&AdapterId::from(adapter), &matrix, public)
        .map_err(lineage_error)?;
    let violations = PyList::empty(py);
    for violation in report.violations() {
        let item = PyDict::new(py);
        match violation {
            GateViolation::AverageForgetting { observed, limit } => {
                item.set_item("kind", "average_forgetting")?;
                item.set_item("observed", observed)?;
                item.set_item("limit", limit)?;
            }
            GateViolation::TaskForgetting {
                task,
                observed,
                limit,
            } => {
                item.set_item("kind", "task_forgetting")?;
                item.set_item("task", task)?;
                item.set_item("observed", observed)?;
                item.set_item("limit", limit)?;
            }
            GateViolation::BackwardTransfer { observed, limit } => {
                item.set_item("kind", "backward_transfer")?;
                item.set_item("observed", observed)?;
                item.set_item("limit", limit)?;
            }
            GateViolation::PublicRegression { observed, limit } => {
                item.set_item("kind", "public_regression")?;
                item.set_item("observed", observed)?;
                item.set_item("limit", limit)?;
            }
        }
        violations.append(item)?;
    }
    let out = PyDict::new(py);
    out.set_item("adapter", report.adapter().to_string())?;
    out.set_item("passed", report.passed())?;
    out.set_item("violations", violations)?;
    out.set_item("average_accuracy", matrix.average_accuracy())?;
    out.set_item("backward_transfer", matrix.backward_transfer())?;
    out.set_item("average_forgetting", matrix.average_forgetting())?;
    out.set_item("task_forgetting", matrix.task_forgetting())?;
    Ok(out)
}
