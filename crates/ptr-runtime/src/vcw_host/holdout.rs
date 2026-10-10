//! The held-out suite behind a structural gate: verified tasks per finding code, a clear rate
//! per code with and without a change to the roster, and the arithmetic that turns those rates
//! into the inputs of a `ptr_lineage::ForgettingGate`.
//!
//! # Mapping to the gate
//!
//! The gate speaks of tasks learned in stages. Here a task is a finding code and the stages are
//! the roster before the change and the roster after it:
//!
//! * the columns are the codes the change is not about, in code order, then the codes it is
//!   about (a Birth's code; none for a Prune or a Freeze, which get a constant control task so
//!   that every code is an "earlier" task whose loss counts as forgetting);
//! * rows before the last repeat the measurement of the serving roster; the last row is the
//!   roster after the change;
//! * so the forgetting of a code is `max(0, clear rate before - clear rate after)`, the backward
//!   transfer is the mean change, and the public suite is the mean clear rate of the codes the
//!   change is not about, before and after.
//!
//! A new Pod therefore cannot lower the clear rate of another code beyond the gate's bounds, and
//! removing a Pod cannot lower the clear rate of the codes it served, without the gate's report
//! failing. The numbers come from the suite, which runs the verifier on candidates; no number is
//! supplied by the proposer or by a model.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use ptr_lineage::{AccuracyMatrix, PublicSuite};
use ptr_router::{
    EligibilitySnapshot, FindingCode, NonceSequence, RouterConfig, SamplingKey, SelfModel,
    SelfModelLimits, UniformScorer,
};
use ptr_types::{Generation, PodId, ProjectId, RequestId, VerificationLevel};
use sha2::{Digest as _, Sha256};

use super::specialist::{Collateral, TrainedSpecialist};
use crate::expert::{StructurePolicy, StructurePolicyConfig};
use crate::vcw::experts::{DefectDoc, DefectDocVerifier, SimulatedExperts, SimulationConfig};
use crate::vcw::{RequestBudget, Roster, VcwConfig, VcwLoop, VcwOutcome, VcwParts, VcwRequest};

/// The Pods the loop can route to, and the codes each repairs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RosterView {
    pub proposers: BTreeSet<PodId>,
    pub repairers: BTreeMap<PodId, BTreeSet<FindingCode>>,
}

impl RosterView {
    /// The roster of a loop as it is now.
    pub fn of<S: ptr_router::AdmissibleScorer>(vcw: &VcwLoop<S>) -> Self {
        Self {
            proposers: vcw.proposers().clone(),
            repairers: vcw.repairers().clone(),
        }
    }

    /// The codes some repairer in the roster is registered for.
    pub fn codes(&self) -> BTreeSet<FindingCode> {
        self.repairers.values().flatten().cloned().collect()
    }
}

/// A change to the roster whose effect the suite measures.
#[derive(Clone, Copy, Debug)]
pub enum RosterChange<'a> {
    /// No change in routing (a Freeze keeps the expert routable).
    Keep,
    /// A specialist joins (Birth, Reactivate).
    Add(&'a TrainedSpecialist),
    /// Experts leave routing (Prune).
    Remove(&'a [PodId]),
    /// A specialist replaces its sources (Merge).
    Replace {
        remove: &'a [PodId],
        add: &'a TrainedSpecialist,
    },
}

impl RosterChange<'_> {
    /// The roster after the change.
    #[must_use]
    pub fn apply(&self, roster: &RosterView) -> RosterView {
        let mut next = roster.clone();
        match self {
            Self::Keep => {}
            Self::Add(specialist) => {
                next.repairers
                    .insert(specialist.pod.clone(), specialist.codes.clone());
            }
            Self::Remove(pods) => {
                for pod in *pods {
                    next.repairers.remove(pod);
                }
            }
            Self::Replace { remove, add } => {
                for pod in *remove {
                    next.repairers.remove(pod);
                }
                next.repairers.insert(add.pod.clone(), add.codes.clone());
            }
        }
        next
    }

    /// The codes the change is about: those the added specialist repairs.
    #[must_use]
    pub fn focus(&self) -> BTreeSet<FindingCode> {
        match self {
            Self::Add(specialist)
            | Self::Replace {
                add: specialist, ..
            } => specialist.codes.clone(),
            Self::Keep | Self::Remove(_) => BTreeSet::new(),
        }
    }

    /// The simulated collateral of the specialist's training, if any.
    #[must_use]
    pub fn collateral(&self) -> &[Collateral] {
        match self {
            Self::Add(specialist)
            | Self::Replace {
                add: specialist, ..
            } => &specialist.collateral,
            Self::Keep | Self::Remove(_) => &[],
        }
    }
}

/// Why a suite could not produce rates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuiteError(pub String);

impl fmt::Display for SuiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "held-out suite: {}", self.0)
    }
}

impl std::error::Error for SuiteError {}

/// Verified held-out tasks. The host's evaluator: its rates are measured with the verifier, on
/// tasks the Pods did not see in serving, never reported by the Pod being judged.
pub trait HeldOutSuite {
    /// The clear rate in `[0, 1]` of every held-out code under `roster` after `change`.
    /// The same codes must be reported with and without a change.
    ///
    /// # Errors
    /// [`SuiteError`] when the suite cannot run.
    fn clear_rates(
        &self,
        roster: &RosterView,
        change: &RosterChange<'_>,
    ) -> Result<BTreeMap<FindingCode, f64>, SuiteError>;
}

/// Why rates cannot be turned into gate inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MeasurementError {
    /// No held-out code at all.
    Empty,
    /// A rate is not a probability.
    BadRate,
    /// The suite measured different codes before and after.
    CodesDiffer,
    /// The matrix or public suite was refused by `ptr-lineage`.
    Lineage(String),
}

impl fmt::Display for MeasurementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("the held-out suite has no code"),
            Self::BadRate => f.write_str("a measured clear rate is not in [0, 1]"),
            Self::CodesDiffer => f.write_str("the suite measured other codes after the change"),
            Self::Lineage(why) => write!(f, "gate inputs refused: {why}"),
        }
    }
}

impl std::error::Error for MeasurementError {}

/// The gate inputs for a change with clear rates `before` and `after`, `focus` being the codes
/// the change is about. See the module documentation for the mapping.
///
/// # Errors
/// [`MeasurementError`] for empty, mismatched or out-of-range rates.
pub fn gate_inputs(
    before: &BTreeMap<FindingCode, f64>,
    after: &BTreeMap<FindingCode, f64>,
    focus: &BTreeSet<FindingCode>,
) -> Result<(AccuracyMatrix, PublicSuite), MeasurementError> {
    if before.is_empty() {
        return Err(MeasurementError::Empty);
    }
    if before.keys().ne(after.keys()) {
        return Err(MeasurementError::CodesDiffer);
    }
    let valid = |rate: &f64| rate.is_finite() && (0.0..=1.0).contains(rate);
    if !before.values().all(valid) || !after.values().all(valid) {
        return Err(MeasurementError::BadRate);
    }
    let others: Vec<&FindingCode> = before.keys().filter(|c| !focus.contains(*c)).collect();
    let focused: Vec<&FindingCode> = before.keys().filter(|c| focus.contains(*c)).collect();
    let mut tasks: Vec<(f64, f64)> = others
        .iter()
        .chain(focused.iter())
        .map(|code| (before[*code], after[*code]))
        .collect();
    if focused.is_empty() {
        // A constant control as the latest task, so every real code counts as an earlier one.
        tasks.push((1.0, 1.0));
    }
    let stages = tasks.len();
    let baseline: Vec<f64> = tasks.iter().map(|(b, _)| *b).collect();
    let last: Vec<f64> = tasks.iter().map(|(_, a)| *a).collect();
    let mut rows = vec![baseline; stages.saturating_sub(1)];
    rows.push(last);
    let matrix =
        AccuracyMatrix::new(rows).map_err(|error| MeasurementError::Lineage(error.to_string()))?;
    let public: Vec<usize> = if others.is_empty() {
        (0..before.len()).collect()
    } else {
        (0..others.len()).collect()
    };
    let mean = |select: &dyn Fn(&(f64, f64)) -> f64| -> f64 {
        public.iter().map(|i| select(&tasks[*i])).sum::<f64>() / public.len() as f64
    };
    Ok((
        matrix,
        PublicSuite {
            serving: mean(&|t| t.0),
            candidate: mean(&|t| t.1),
        },
    ))
}

/// A held-out suite over the simulated defect documents: for each code, `tasks_per_code` fresh
/// requests whose proposal carries exactly that defect, run through a real [`VcwLoop`] over the
/// roster (uniform routing, the simulation's `DefectDocVerifier`), counting how many end
/// `Verified`.
///
/// The "world" is the simulation configuration the live experts run under; the suite derives its
/// own per-code seed from `seed`, so its draws are independent of serving. A specialist's
/// configured skill and its simulated collateral are applied in the "after" measurement.
#[derive(Clone, Debug)]
pub struct SimulatedHeldOut {
    world: SimulationConfig,
    task: String,
    codes: Vec<String>,
    tasks_per_code: u32,
    seed: u64,
    project: ProjectId,
}

impl SimulatedHeldOut {
    /// # Errors
    /// [`SuiteError`] when no code or no task is requested, or the world is invalid.
    pub fn new(
        world: SimulationConfig,
        project: ProjectId,
        codes: &[&str],
        tasks_per_code: u32,
        seed: u64,
    ) -> Result<Self, SuiteError> {
        if codes.is_empty() || tasks_per_code == 0 || tasks_per_code > 1024 {
            return Err(SuiteError(
                "codes empty or tasks per code outside 1..=1024".into(),
            ));
        }
        world.validate().map_err(|e| SuiteError(e.to_string()))?;
        Ok(Self {
            world,
            task: "held-out".to_owned(),
            codes: codes.iter().map(|c| (*c).to_owned()).collect(),
            tasks_per_code,
            seed,
            project,
        })
    }

    fn run_code(
        &self,
        code: &str,
        roster: &RosterView,
        change: &RosterChange<'_>,
        apply_change: bool,
    ) -> Result<f64, SuiteError> {
        let roster = if apply_change {
            change.apply(roster)
        } else {
            roster.clone()
        };
        let mut config = self.world.clone();
        let derived = {
            let mut hasher = Sha256::new();
            hasher.update(self.seed.to_le_bytes());
            hasher.update(code.as_bytes());
            let digest = hasher.finalize();
            let mut head = [0_u8; 8];
            head.copy_from_slice(&digest[..8]);
            u64::from_le_bytes(head)
        };
        config.seed = derived;
        config.error_rates = BTreeMap::from([(code.to_owned(), 1.0)]);
        if apply_change {
            if let RosterChange::Add(specialist)
            | RosterChange::Replace {
                add: specialist, ..
            } = change
            {
                for (skill_code, p) in &specialist.skill {
                    config
                        .pod_repair_success
                        .insert((specialist.pod.0.clone(), skill_code.clone()), *p);
                }
            }
            for collateral in change.collateral() {
                config.pod_repair_success.insert(
                    (collateral.pod.0.clone(), collateral.code.clone()),
                    collateral.success_after,
                );
            }
        }
        let mut experts = SimulatedExperts::<DefectDoc>::new(config, &self.task)
            .map_err(|e| SuiteError(e.to_string()))?;
        let mut view = EligibilitySnapshot::new();
        for proposer in &roster.proposers {
            experts
                .add_proposer(proposer.clone())
                .map_err(|e| SuiteError(e.to_string()))?;
            view.set(proposer.clone(), Generation(1));
        }
        let mut loop_roster = Roster {
            proposers: roster.proposers.iter().cloned().collect(),
            repairers: Vec::new(),
        };
        for (pod, codes) in &roster.repairers {
            let names: Vec<&str> = codes.iter().map(FindingCode::as_str).collect();
            experts
                .add_repairer(pod.clone(), &names)
                .map_err(|e| SuiteError(e.to_string()))?;
            view.set(pod.clone(), Generation(1));
            loop_roster
                .repairers
                .push((pod.clone(), codes.iter().cloned().collect()));
        }
        let mut vcw = VcwLoop::new(VcwParts {
            config: VcwConfig::new("held-out", [7; 32]),
            roster: loop_roster,
            router: RouterConfig::default(),
            scorer: UniformScorer,
            key: SamplingKey::from_bytes(
                derived
                    .to_le_bytes()
                    .repeat(4)
                    .try_into()
                    .map_err(|_| SuiteError("key".into()))?,
            ),
            nonces: NonceSequence::starting_at([9; 8], 0),
            self_model: SelfModel::new(SelfModelLimits::default()),
            schema: None,
            structure: StructurePolicy::new(StructurePolicyConfig::default())
                .map_err(|e| SuiteError(format!("{e:?}")))?,
        })
        .map_err(|e| SuiteError(e.to_string()))?;
        let mut cleared = 0_u32;
        for k in 0..self.tasks_per_code {
            let id = format!("held-out/{}/{code}/{k}", self.seed);
            let request = VcwRequest {
                request_id: RequestId::from(id.as_str()),
                project: self.project.clone(),
                request_digest: Sha256::digest(id.as_bytes()).into(),
                required: VerificationLevel::SampleVerified,
                budget: RequestBudget::new(8, 6).map_err(|e| SuiteError(e.to_string()))?,
            };
            let mut pods = experts.guarded(&view);
            let run = vcw
                .run(&request, &view, &DefectDocVerifier, &mut pods)
                .map_err(|e| SuiteError(e.to_string()))?;
            if matches!(run.outcome, VcwOutcome::Verified(_)) {
                cleared += 1;
            }
        }
        Ok(f64::from(cleared) / f64::from(self.tasks_per_code))
    }
}

impl HeldOutSuite for SimulatedHeldOut {
    fn clear_rates(
        &self,
        roster: &RosterView,
        change: &RosterChange<'_>,
    ) -> Result<BTreeMap<FindingCode, f64>, SuiteError> {
        // `Keep` and a roster-only change are told apart by the caller; here "no change" is
        // `RosterChange::Keep`, and the serving measurement is the one with it.
        let apply = !matches!(change, RosterChange::Keep);
        self.codes
            .iter()
            .map(|code| {
                self.run_code(code, roster, change, apply)
                    .map(|rate| (FindingCode::new(code.as_str()), rate))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rates(pairs: &[(&str, f64)]) -> BTreeMap<FindingCode, f64> {
        pairs
            .iter()
            .map(|(c, r)| (FindingCode::new(*c), *r))
            .collect()
    }

    fn focus(codes: &[&str]) -> BTreeSet<FindingCode> {
        codes.iter().map(|c| FindingCode::new(*c)).collect()
    }

    #[test]
    fn a_new_code_that_costs_the_others_nothing_has_no_forgetting() {
        let before = rates(&[("a", 0.9), ("b", 0.8), ("new", 0.0)]);
        let after = rates(&[("a", 0.9), ("b", 0.8), ("new", 0.95)]);
        let (matrix, public) = gate_inputs(&before, &after, &focus(&["new"])).unwrap();
        assert_eq!(matrix.average_forgetting(), 0.0);
        assert_eq!(matrix.task_forgetting(), vec![0.0, 0.0]);
        assert!(matrix.backward_transfer().abs() < 1e-12);
        assert!((public.serving - 0.85).abs() < 1e-12);
        assert!((public.candidate - 0.85).abs() < 1e-12);
    }

    #[test]
    fn a_lost_clear_rate_is_forgetting_and_a_public_regression() {
        let before = rates(&[("a", 0.9), ("b", 0.8), ("new", 0.0)]);
        let after = rates(&[("a", 0.5), ("b", 0.8), ("new", 0.95)]);
        let (matrix, public) = gate_inputs(&before, &after, &focus(&["new"])).unwrap();
        let forgetting = matrix.task_forgetting();
        assert!((forgetting[0] - 0.4).abs() < 1e-12);
        assert_eq!(forgetting[1], 0.0);
        assert!((matrix.average_forgetting() - 0.2).abs() < 1e-12);
        assert!((matrix.backward_transfer() + 0.2).abs() < 1e-12);
        assert!((public.serving - public.candidate - 0.2).abs() < 1e-12);
    }

    #[test]
    fn removing_a_pod_without_a_focus_counts_every_code_through_a_control_task() {
        let before = rates(&[("a", 0.9), ("b", 0.8)]);
        let after = rates(&[("a", 0.9), ("b", 0.2)]);
        let (matrix, public) = gate_inputs(&before, &after, &BTreeSet::new()).unwrap();
        assert_eq!(matrix.task_forgetting().len(), 2);
        assert!((matrix.task_forgetting()[1] - 0.6).abs() < 1e-12);
        assert!((public.serving - 0.85).abs() < 1e-12);
        assert!((public.candidate - 0.55).abs() < 1e-12);
    }

    #[test]
    fn malformed_measurements_are_refused() {
        let ok = rates(&[("a", 0.5)]);
        assert_eq!(
            gate_inputs(&BTreeMap::new(), &BTreeMap::new(), &BTreeSet::new()).unwrap_err(),
            MeasurementError::Empty
        );
        assert_eq!(
            gate_inputs(&ok, &rates(&[("b", 0.5)]), &BTreeSet::new()).unwrap_err(),
            MeasurementError::CodesDiffer
        );
        assert_eq!(
            gate_inputs(&ok, &rates(&[("a", 1.5)]), &BTreeSet::new()).unwrap_err(),
            MeasurementError::BadRate
        );
        assert_eq!(
            gate_inputs(&ok, &rates(&[("a", f64::NAN)]), &BTreeSet::new()).unwrap_err(),
            MeasurementError::BadRate
        );
    }
}
