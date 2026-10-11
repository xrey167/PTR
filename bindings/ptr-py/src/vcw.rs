//! An in-process run of the verifier-closed workspace loop (`VcwLoop`) with simulated experts.
//!
//! No server, ledger or runtime is involved: the loop, its router and workspace run in this
//! process against seeded programmatic experts that plant and clear defect codes. The
//! verifier is the simulation's own (it reports each planted defect as a hard finding), the
//! required level is the runtime default (`sample_verified`) and the roster is the one
//! given. There is no handle on a live runtime, so this is a measurement harness for the
//! routing and budget behaviour, not a way to change any gate.

use std::collections::BTreeMap;

use ptr_router::{
    EligibilitySnapshot, Exploration, NonceSequence, RouterConfig, SamplingKey, SelfModel,
    SelfModelLimits, UniformScorer,
};
use ptr_runtime::expert::{StructurePolicy, StructurePolicyConfig};
use ptr_runtime::vcw::experts::{
    DefectDoc, DefectDocVerifier, DefectPayload, SimulatedExperts, SimulationConfig,
};
use ptr_runtime::vcw::{
    ClassSummary, ReportSummary, RequestBudget, VcwConfig, VcwLoop, VcwOutcome, VcwParts,
    VcwRequest, VcwTrace,
};
use ptr_types::{Generation, PodId, ProjectId, RequestId, VerificationLevel};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sha2::{Digest as _, Sha256};

use crate::convert::{hex, level_name, PtrVcwError};
use crate::routing::{escalation_names, fallback_name, record_dict};
use crate::verify::status_name;

fn vcw_error(error: impl std::fmt::Display) -> PyErr {
    PtrVcwError::new_err(error.to_string())
}

fn summary_dict<'py>(py: Python<'py>, summary: &ReportSummary) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("status", status_name(summary.status))?;
    out.set_item("level", level_name(summary.level))?;
    out.set_item("score", f64::from(f32::from_bits(summary.score_bits)))?;
    out.set_item("hard_codes", &summary.hard_codes)?;
    out.set_item("hard_total", summary.hard_total)?;
    out.set_item("soft_findings", summary.soft_findings)?;
    Ok(out)
}

fn trace_dict<'py>(py: Python<'py>, trace: &VcwTrace) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("request_id", &trace.request_id().0)?;
    out.set_item("digest", hex(trace.digest()))?;
    out.set_item("required", level_name(trace.required()))?;
    let budget = PyDict::new(py);
    budget.set_item("verifier_calls", trace.budget().verifier_calls.get())?;
    budget.set_item("rounds", trace.budget().rounds.get())?;
    out.set_item("budget", budget)?;
    let spent = trace.spent();
    let used = PyDict::new(py);
    used.set_item("verifier_calls", spent.verifier_calls)?;
    used.set_item("rounds", spent.rounds)?;
    used.set_item("pod_calls", spent.pod_calls)?;
    out.set_item("spent", used)?;
    out.set_item("outcome", format!("{:?}", trace.outcome()))?;
    let steps = PyList::empty(py);
    for step in trace.steps() {
        let item = PyDict::new(py);
        item.set_item("round", step.round)?;
        item.set_item("origin", format!("{:?}", step.origin))?;
        item.set_item(
            "authors",
            step.authors
                .iter()
                .map(|a| a.pod.0.clone())
                .collect::<Vec<_>>(),
        )?;
        let routes = PyList::empty(py);
        for record in &step.routes {
            routes.append(record_dict(py, record, None)?)?;
        }
        item.set_item("routes", routes)?;
        item.set_item(
            "unroutable",
            step.unroutable
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>(),
        )?;
        match &step.candidate {
            Some(candidate) => {
                let c = PyDict::new(py);
                c.set_item("report", summary_dict(py, &candidate.report)?)?;
                match &candidate.class {
                    ClassSummary::Accept => c.set_item("class", "accept")?,
                    ClassSummary::Repair(codes) => {
                        c.set_item("class", "repair")?;
                        c.set_item(
                            "codes",
                            codes
                                .iter()
                                .map(|code| code.to_string())
                                .collect::<Vec<_>>(),
                        )?;
                    }
                    ClassSummary::Fallback(reason) => {
                        c.set_item("class", "fallback")?;
                        c.set_item("reason", fallback_name(*reason))?;
                    }
                }
                c.set_item("workspace", format!("{:?}", candidate.workspace))?;
                c.set_item("workspace_revision", candidate.workspace_revision)?;
                item.set_item("candidate", c)?;
            }
            None => item.set_item("candidate", py.None())?,
        }
        item.set_item("decision", format!("{:?}", step.decision))?;
        let after = PyDict::new(py);
        after.set_item("verifier_calls", step.budget_after.verifier_calls)?;
        after.set_item("rounds", step.budget_after.rounds)?;
        after.set_item("pod_calls", step.budget_after.pod_calls)?;
        item.set_item("budget_after", after)?;
        steps.append(item)?;
    }
    out.set_item("steps", steps)?;
    Ok(out)
}

/// Run one request of the verifier-closed loop in this process with seeded simulated
/// experts, and return the run report.
///
/// `simulate_vcw_run(seed, proposers, repairers, *, error_rates, repair_success,
/// default_repair_success=1.0, decline_on_failure=False, verifier_calls=8, rounds=4,
/// exploration=0.05, request_id="req-1", task="task")`.
/// `proposers` is a list of Pod ids and `repairers` a list of `(pod, [codes])`.
/// `error_rates` maps a defect code to the probability a proposal carries it;
/// `repair_success` maps a code to the probability a repairer clears it. Draws are pure
/// functions of the seed, the Pod, the request and the code, so a run replays.
///
/// Returns `{"outcome": "Accepted" | "Exhausted" | "Escalated", "reason", "answer", "trace"}`:
/// `answer` (accepted only) has the `author`, `authors` and the verifier's `report`;
/// `trace` has the request digest, the `budget`, what was `spent`, and per-step `routes`
/// (with propensity and support), the verifier report summary, the `classify` result and
/// the workspace outcome. An exhausted or escalated run carries no candidate.
///
/// Raises PtrVcwError for an invalid rate, roster or budget, or when the loop fails.
#[pyfunction]
#[pyo3(signature = (
    seed,
    proposers,
    repairers,
    *,
    error_rates=None,
    repair_success=None,
    default_repair_success=1.0,
    decline_on_failure=false,
    verifier_calls=8,
    rounds=4,
    exploration=0.05,
    request_id="req-1",
    task="task",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_vcw_run<'py>(
    py: Python<'py>,
    seed: u64,
    proposers: Vec<String>,
    repairers: Vec<(String, Vec<String>)>,
    error_rates: Option<BTreeMap<String, f64>>,
    repair_success: Option<BTreeMap<String, f64>>,
    default_repair_success: f64,
    decline_on_failure: bool,
    verifier_calls: u32,
    rounds: u32,
    exploration: f64,
    request_id: &str,
    task: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let run = py.detach(move || {
        let mut config = SimulationConfig::new(seed);
        config.error_rates = error_rates.unwrap_or_default();
        config.repair_success = repair_success.unwrap_or_default();
        config.default_repair_success = default_repair_success;
        config.decline_on_failure = decline_on_failure;
        let mut experts = SimulatedExperts::<DefectDoc>::new(config, task).map_err(vcw_error)?;
        let mut view = EligibilitySnapshot::new();
        for pod in &proposers {
            experts
                .add_proposer(PodId::from(pod.as_str()))
                .map_err(|error| vcw_error(format!("{error:?}")))?;
            view.set(PodId::from(pod.as_str()), Generation(1));
        }
        for (pod, codes) in &repairers {
            let codes: Vec<&str> = codes.iter().map(String::as_str).collect();
            experts
                .add_repairer(PodId::from(pod.as_str()), &codes)
                .map_err(|error| vcw_error(format!("{error:?}")))?;
            view.set(PodId::from(pod.as_str()), Generation(1));
        }
        let roster = experts.roster().map_err(vcw_error)?;
        let exploration = Exploration::new(exploration).map_err(vcw_error)?;
        let mut key = [0u8; 32];
        for chunk in key.chunks_mut(8) {
            chunk.copy_from_slice(&seed.to_le_bytes());
        }
        let mut vcw = VcwLoop::new(VcwParts {
            config: VcwConfig::new("ptr-py-simulation", [3; 32]),
            roster,
            router: RouterConfig {
                exploration,
                ..RouterConfig::default()
            },
            scorer: UniformScorer,
            key: SamplingKey::from_bytes(key),
            nonces: NonceSequence::starting_at([1; 8], 0),
            self_model: SelfModel::new(SelfModelLimits::default()),
            schema: None,
            structure: StructurePolicy::new(StructurePolicyConfig::default())
                .map_err(|error| vcw_error(format!("{error:?}")))?,
        })
        .map_err(vcw_error)?;
        let request = VcwRequest {
            request_id: RequestId::from(request_id),
            project: ProjectId::from("ptr-py"),
            request_digest: Sha256::digest(request_id.as_bytes()).into(),
            required: VerificationLevel::SampleVerified,
            budget: RequestBudget::new(verifier_calls, rounds).map_err(vcw_error)?,
        };
        vcw.run(
            &request,
            &view,
            &DefectDocVerifier,
            &mut experts.guarded(&view),
        )
        .map_err(vcw_error)
    })?;
    let out = PyDict::new(py);
    match &run.outcome {
        VcwOutcome::Verified(answer) => {
            out.set_item("outcome", "Accepted")?;
            out.set_item("reason", py.None())?;
            let item = PyDict::new(py);
            item.set_item("author", &answer.author().pod.0)?;
            item.set_item(
                "authors",
                answer
                    .authors()
                    .iter()
                    .map(|a| a.pod.0.clone())
                    .collect::<Vec<_>>(),
            )?;
            item.set_item("report", summary_dict(py, answer.report())?)?;
            item.set_item(
                "defects",
                answer.payload().defects().into_iter().collect::<Vec<_>>(),
            )?;
            out.set_item("answer", item)?;
        }
        VcwOutcome::Exhausted { reason } => {
            out.set_item("outcome", "Exhausted")?;
            out.set_item("reason", format!("{reason:?}"))?;
            out.set_item("answer", py.None())?;
        }
        VcwOutcome::Escalated { reason } => {
            let (_, why) = escalation_names(*reason);
            out.set_item("outcome", "Escalated")?;
            out.set_item("reason", why)?;
            out.set_item("answer", py.None())?;
        }
    }
    out.set_item("trace", trace_dict(py, &run.trace)?)?;
    Ok(out)
}
