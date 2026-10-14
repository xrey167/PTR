//! Plan-vs-execution correlation.
//!
//! A [`PlanRecord`] is an ordered list of planned steps. Finished spans tagged with the
//! `ptr.plan_step` field are matched to steps by id. Rules (deterministic; spans are ordered by
//! `(start_tick, span id)`):
//!
//! - no tagged span for a planned step: `Skipped`;
//! - the *last* execution of the step (by start) ended with `SpanStatus::Error`: `Failed` (an
//!   earlier failed attempt followed by a successful retry is `Executed`);
//! - otherwise, if the step's first execution started before the first execution of some
//!   earlier-planned executed step: `OutOfOrder`;
//! - otherwise `Executed`;
//! - tagged spans whose step id is not in the plan: one `Unplanned` row per distinct id.
//!
//! Failure takes precedence over ordering. Spans without the tag are counted, not reported.
//! `operation_mismatch` is set when the step's expected operation differs from the `ptr.operation`
//! field of its last execution (informational, does not change the outcome).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{fields, Span, SpanId, SpanStatus, TraceValue};

pub const MAX_PLAN_STEPS: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedStep {
    pub id: String,
    pub expected_operation: String,
}

impl PlannedStep {
    pub fn new(id: impl Into<String>, expected_operation: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            expected_operation: expected_operation.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanError {
    EmptyStepId,
    DuplicateStepId { id: String },
    TooManySteps { limit: usize },
}

impl PlanError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::EmptyStepId => "empty_step_id",
            Self::DuplicateStepId { .. } => "duplicate_step_id",
            Self::TooManySteps { .. } => "too_many_steps",
        }
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyStepId => write!(f, "plan step id must not be empty"),
            Self::DuplicateStepId { id } => write!(f, "duplicate plan step id: {id}"),
            Self::TooManySteps { limit } => write!(f, "plan exceeds {limit} steps"),
        }
    }
}

impl std::error::Error for PlanError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanRecord {
    plan_id: String,
    steps: Vec<PlannedStep>,
}

impl PlanRecord {
    pub fn new(plan_id: impl Into<String>, steps: Vec<PlannedStep>) -> Result<Self, PlanError> {
        if steps.len() > MAX_PLAN_STEPS {
            return Err(PlanError::TooManySteps {
                limit: MAX_PLAN_STEPS,
            });
        }
        let mut seen = BTreeSet::new();
        for step in &steps {
            if step.id.is_empty() {
                return Err(PlanError::EmptyStepId);
            }
            if !seen.insert(step.id.as_str()) {
                return Err(PlanError::DuplicateStepId {
                    id: step.id.clone(),
                });
            }
        }
        Ok(Self {
            plan_id: plan_id.into(),
            steps,
        })
    }

    pub fn plan_id(&self) -> &str {
        &self.plan_id
    }

    pub fn steps(&self) -> &[PlannedStep] {
        &self.steps
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepOutcome {
    Executed,
    Skipped,
    Unplanned,
    OutOfOrder,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepReport {
    pub step_id: String,
    /// Position in the plan; `None` for unplanned executions.
    pub plan_index: Option<usize>,
    pub outcome: StepOutcome,
    pub executions: usize,
    pub spans: Vec<SpanId>,
    pub operation_mismatch: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CorrelationSummary {
    pub planned: usize,
    pub executed: usize,
    pub skipped: usize,
    pub out_of_order: usize,
    pub failed: usize,
    pub unplanned: usize,
    pub untagged_spans: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorrelationReport {
    pub plan_id: String,
    pub steps: Vec<StepReport>,
    pub summary: CorrelationSummary,
}

impl CorrelationReport {
    /// True when every planned step executed in order without failure and nothing ran unplanned.
    pub fn is_clean(&self) -> bool {
        let s = &self.summary;
        s.skipped == 0 && s.out_of_order == 0 && s.failed == 0 && s.unplanned == 0
    }

    pub fn render_text(&self) -> String {
        let mut out = format!("plan {}\n", self.plan_id);
        for step in &self.steps {
            out.push_str(&format!(
                "  {:<10} {} (executions: {})\n",
                format!("{:?}", step.outcome),
                step.step_id,
                step.executions
            ));
        }
        let s = &self.summary;
        out.push_str(&format!(
            "summary: planned={} executed={} skipped={} out_of_order={} failed={} unplanned={}\n",
            s.planned, s.executed, s.skipped, s.out_of_order, s.failed, s.unplanned
        ));
        out
    }
}

fn step_tag(span: &Span) -> Option<&str> {
    match span.fields.get(fields::PLAN_STEP) {
        Some(TraceValue::String(id)) => Some(id.as_str()),
        _ => None,
    }
}

/// Correlates `spans` (any order, any trace) with `plan`.
pub fn correlate(plan: &PlanRecord, spans: &[Span]) -> CorrelationReport {
    let mut by_step: BTreeMap<&str, Vec<&Span>> = BTreeMap::new();
    let mut untagged = 0;
    for span in spans {
        match step_tag(span) {
            Some(id) => by_step.entry(id).or_default().push(span),
            None => untagged += 1,
        }
    }
    for executions in by_step.values_mut() {
        executions.sort_by_key(|span| (span.start_tick, span.id));
    }

    let mut rows = Vec::with_capacity(plan.steps.len());
    let mut max_first_start: Option<(u64, SpanId)> = None;
    for (index, step) in plan.steps.iter().enumerate() {
        let Some(executions) = by_step.get(step.id.as_str()) else {
            rows.push(StepReport {
                step_id: step.id.clone(),
                plan_index: Some(index),
                outcome: StepOutcome::Skipped,
                executions: 0,
                spans: Vec::new(),
                operation_mismatch: false,
            });
            continue;
        };
        let first = executions[0];
        let last = executions[executions.len() - 1];
        let first_key = (first.start_tick, first.id);
        let failed = matches!(last.status, SpanStatus::Error(_));
        let out_of_order = max_first_start.is_some_and(|max| first_key < max);
        let outcome = if failed {
            StepOutcome::Failed
        } else if out_of_order {
            StepOutcome::OutOfOrder
        } else {
            StepOutcome::Executed
        };
        max_first_start = Some(max_first_start.map_or(first_key, |max| max.max(first_key)));
        let operation_mismatch = match last.fields.get(fields::OPERATION) {
            Some(TraceValue::String(op)) => *op != step.expected_operation,
            _ => false,
        };
        rows.push(StepReport {
            step_id: step.id.clone(),
            plan_index: Some(index),
            outcome,
            executions: executions.len(),
            spans: executions.iter().map(|span| span.id).collect(),
            operation_mismatch,
        });
    }

    let planned_ids: BTreeSet<&str> = plan.steps.iter().map(|step| step.id.as_str()).collect();
    for (id, executions) in &by_step {
        if !planned_ids.contains(id) {
            rows.push(StepReport {
                step_id: (*id).to_owned(),
                plan_index: None,
                outcome: StepOutcome::Unplanned,
                executions: executions.len(),
                spans: executions.iter().map(|span| span.id).collect(),
                operation_mismatch: false,
            });
        }
    }

    let count = |outcome: StepOutcome| rows.iter().filter(|r| r.outcome == outcome).count();
    let summary = CorrelationSummary {
        planned: plan.steps.len(),
        executed: count(StepOutcome::Executed),
        skipped: count(StepOutcome::Skipped),
        out_of_order: count(StepOutcome::OutOfOrder),
        failed: count(StepOutcome::Failed),
        unplanned: count(StepOutcome::Unplanned),
        untagged_spans: untagged,
    };
    CorrelationReport {
        plan_id: plan.plan_id.clone(),
        steps: rows,
        summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_validation_errors_are_typed() {
        assert_eq!(
            PlanRecord::new("p", vec![PlannedStep::new("", "op")])
                .expect_err("empty")
                .code(),
            "empty_step_id"
        );
        let dup = vec![PlannedStep::new("a", "x"), PlannedStep::new("a", "y")];
        let err = PlanRecord::new("p", dup).expect_err("dup");
        assert_eq!(err.code(), "duplicate_step_id");
        assert!(err.to_string().contains('a'));
        let many = (0..=MAX_PLAN_STEPS)
            .map(|i| PlannedStep::new(format!("s{i}"), "op"))
            .collect();
        assert_eq!(
            PlanRecord::new("p", many).expect_err("many"),
            PlanError::TooManySteps {
                limit: MAX_PLAN_STEPS
            }
        );
    }
}
