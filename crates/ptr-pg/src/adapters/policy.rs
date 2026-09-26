//! Recorded triage policies and the adjudicated calibration samples they are
//! calibrated on and evaluated against, in the work schema.

use ptr_branch::{
    AutoThreshold, BranchId, CalibrationSample, PolicyRecord, ThresholdRule, TriageDecision,
    TriageOutcome,
};

use super::{check_text, database, PgSubstrate};
use crate::error::PgError;

impl PgSubstrate {
    /// Record a policy and exactly which branches its threshold was
    /// calibrated on, in one transaction. A triage row can cite the policy
    /// only once it is recorded, and a record is never rewritten.
    ///
    /// Every calibration branch must be an adjudicated calibration-slice
    /// branch: a sample from anywhere else is not an exchangeable sample of
    /// eligible branches, and a policy calibrated on it would claim a
    /// guarantee it does not have. Such a policy is refused with
    /// `PgError::InvalidPolicy` and nothing is written.
    pub async fn record_policy(&mut self, record: &PolicyRecord) -> Result<(), PgError> {
        check_text("triage_policy.version", record.version())?;
        for branch in record.calibrated_on() {
            check_text("triage_policy_sample.branch", &branch.0)?;
        }
        let work = self.schemas.work.clone();
        let policy = record.policy();
        let threshold = match policy.threshold() {
            AutoThreshold::Never => None,
            AutoThreshold::AtLeast(threshold) => Some(threshold),
        };
        let (alpha, delta) = match record.rule() {
            ThresholdRule::Manual => (None, None),
            ThresholdRule::ConformalRiskControl { alpha } => (Some(alpha), None),
            ThresholdRule::LearnThenTest { alpha, delta } => (Some(alpha), Some(delta)),
        };
        let branches: Vec<&str> = record
            .calibrated_on()
            .iter()
            .map(|branch| branch.0.as_str())
            .collect();
        let transaction = self.read_committed().await?;
        let adjudicated: i64 = transaction
            .query_one(
                &format!(
                    "SELECT count(DISTINCT t.branch) FROM {work}.branch_triage t \
                     JOIN {work}.branch_outcome o ON o.branch = t.branch \
                     AND o.outcome IN ('adjudicated_harmful', 'adjudicated_harmless') \
                     WHERE t.calibration_slice AND t.branch = ANY($1)"
                ),
                &[&branches],
            )
            .await
            .map_err(database)?
            .get(0);
        if adjudicated != branches.len() as i64 {
            return Err(PgError::InvalidPolicy {
                version: record.version().to_owned(),
                reason: "a calibration branch is not an adjudicated calibration-slice branch",
            });
        }
        transaction
            .execute(
                &format!(
                    "INSERT INTO {work}.triage_policy \
                     (version, rule, threshold, calibration_rate, alpha, delta) \
                     VALUES ($1, $2, $3, $4, $5, $6)"
                ),
                &[
                    &record.version(),
                    &record.rule().name(),
                    &threshold,
                    &policy.calibration_rate(),
                    &alpha,
                    &delta,
                ],
            )
            .await
            .map_err(database)?;
        transaction
            .execute(
                &format!(
                    "INSERT INTO {work}.triage_policy_sample (policy_version, branch) \
                     SELECT $1, branch FROM unnest($2::text[]) AS branch"
                ),
                &[&record.version(), &branches],
            )
            .await
            .map_err(database)?;
        transaction.commit().await.map_err(database)
    }

    /// A recorded policy with its calibration branches, or `None`.
    pub async fn load_policy(&self, version: &str) -> Result<Option<PolicyRecord>, PgError> {
        check_text("triage_policy.version", version)?;
        let work = &self.schemas.work;
        let Some(row) = self
            .client
            .query_opt(
                &format!(
                    "SELECT rule, threshold, calibration_rate, alpha, delta \
                     FROM {work}.triage_policy WHERE version = $1"
                ),
                &[&version],
            )
            .await
            .map_err(database)?
        else {
            return Ok(None);
        };
        let corrupt = |reason: String| PgError::CorruptRow {
            table: "triage_policy",
            reason,
        };
        let (rule, alpha, delta): (String, Option<f64>, Option<f64>) =
            (row.get(0), row.get(3), row.get(4));
        let rule = match (rule.as_str(), alpha, delta) {
            ("manual", None, None) => ThresholdRule::Manual,
            ("conformal_risk_control", Some(alpha), None) => {
                ThresholdRule::ConformalRiskControl { alpha }
            }
            ("learn_then_test", Some(alpha), Some(delta)) => {
                ThresholdRule::LearnThenTest { alpha, delta }
            }
            (other, _, _) => return Err(corrupt(format!("rule {other:?} with its levels"))),
        };
        let threshold = match row.get::<_, Option<f32>>(1) {
            None => AutoThreshold::Never,
            Some(threshold) => AutoThreshold::AtLeast(threshold),
        };
        let branches = self
            .client
            .query(
                &format!(
                    "SELECT branch FROM {work}.triage_policy_sample \
                     WHERE policy_version = $1 ORDER BY branch"
                ),
                &[&version],
            )
            .await
            .map_err(database)?
            .into_iter()
            .map(|row| BranchId(row.get(0)))
            .collect();
        PolicyRecord::from_parts(version, threshold, row.get(2), rule, branches)
            .map(Some)
            .map_err(|error| corrupt(error.to_string()))
    }

    /// Every adjudicated calibration-slice branch as a calibration sample, in
    /// branch order. A recorded policy's
    /// [`held_out`](PolicyRecord::held_out) of these are the adjudications
    /// its harm rate may be estimated from.
    pub async fn adjudicated_samples(&self) -> Result<Vec<(BranchId, CalibrationSample)>, PgError> {
        let work = &self.schemas.work;
        let rows = self
            .client
            .query(
                &format!(
                    "SELECT t.branch, t.decision, t.eligible, t.calibration_slice, t.score, \
                            t.auto_propensity, o.outcome = 'adjudicated_harmful' \
                     FROM {work}.branch_triage t \
                     JOIN {work}.branch_outcome o ON o.branch = t.branch \
                     AND o.outcome IN ('adjudicated_harmful', 'adjudicated_harmless') \
                     WHERE t.calibration_slice ORDER BY t.branch"
                ),
                &[],
            )
            .await
            .map_err(database)?;
        rows.into_iter()
            .map(|row| {
                let branch: String = row.get(0);
                let decision = match row.get::<_, &str>(1) {
                    "auto_propose" => TriageDecision::AutoPropose,
                    "escalate" => TriageDecision::Escalate,
                    "discard" => TriageDecision::Discard,
                    other => {
                        return Err(PgError::CorruptRow {
                            table: "branch_triage",
                            reason: format!("decision {other:?}"),
                        })
                    }
                };
                let triage = TriageOutcome {
                    decision,
                    eligible: row.get(2),
                    calibration_slice: row.get(3),
                    score: row.get(4),
                    auto_propensity: row.get(5),
                };
                let sample = triage
                    .adjudicate(row.get(6))
                    .ok_or_else(|| PgError::CorruptRow {
                        table: "branch_triage",
                        reason: format!("calibration row {branch:?} is not eligible"),
                    })?;
                Ok((BranchId(branch), sample))
            })
            .collect()
    }
}
