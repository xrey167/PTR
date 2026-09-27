//! Recorded triage policies and the adjudicated calibration samples they are
//! calibrated on and evaluated against, in the work schema.

use ptr_branch::{
    AutoThreshold, BranchId, CalibrationSample, PolicyRecord, ThresholdRule, TriageDecision,
    TriageOutcome, TriagePolicy,
};

use tokio_postgres::Row;

use super::{check_text, database, PgSubstrate};
use crate::config::Identifier;
use crate::error::PgError;

impl PgSubstrate {
    /// Record a policy and exactly which branches its threshold was
    /// calibrated on, in one transaction. A triage row can cite the policy
    /// only once it is recorded, and a record is never rewritten: the policy
    /// row carries the size of its calibration set, and the database refuses
    /// any change to it, any deletion and any calibration sample added after
    /// the policy commits.
    ///
    /// Every calibration branch must be an adjudicated calibration-slice
    /// branch: a sample from anywhere else is not an exchangeable sample of
    /// eligible branches, and a policy calibrated on it would claim a
    /// guarantee it does not have. The record's rule is then rerun, inside the
    /// transaction, on those branches' stored scores and verdicts at the
    /// record's levels, and the record must be exactly what it chooses: a
    /// threshold nobody computed from the stored adjudications, or one
    /// computed from samples paired with the wrong branches, would claim the
    /// same false guarantee. Either way the policy is refused with
    /// `PgError::InvalidPolicy` and nothing is written.
    ///
    /// The statement that reads the calibration branches, their triage rows
    /// and their verdicts holds those rows `FOR SHARE` until the transaction
    /// ends, so no other transaction deletes or rewrites what the rule was
    /// rerun on before the policy commits, and once it commits its sample
    /// rows keep every calibration branch from being deleted: their foreign
    /// key is `ON DELETE RESTRICT`, which, unlike a `NO ACTION` key, refuses
    /// the delete even in a statement that writes the branch again under its
    /// id before it ends. So the triage row and verdict the rule was rerun on,
    /// which go only with their branch, stay. A branch
    /// erased by a transaction that commits while this one waits for its row
    /// is not read at all, rewritten under its id or not, and the policy is
    /// refused as naming a branch that is not an adjudicated calibration-slice
    /// branch. A concurrent erasure can also end in a deadlock that
    /// PostgreSQL resolves by failing one of the two transactions; this one
    /// then returns `PgError::Database` and writes nothing.
    ///
    /// # Errors
    /// Besides `PgError::InvalidPolicy`, refuses as `PgError::CorruptRow` a
    /// calibration branch whose stored triage row its cited policy cannot
    /// have produced, as [`adjudicated_samples`](Self::adjudicated_samples)
    /// does, writing nothing.
    pub async fn record_policy(&mut self, record: &PolicyRecord) -> Result<(), PgError> {
        check_text("triage_policy.version", record.version())?;
        for branch in record.calibrated_on() {
            check_text("triage_policy_sample.branch", &branch.0)?;
        }
        let refuse = |reason| PgError::InvalidPolicy {
            version: record.version().to_owned(),
            reason,
        };
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
        let calibration_size = i32::try_from(branches.len())
            .map_err(|_| refuse("the calibration set is larger than the table can record"))?;
        let transaction = self.read_committed().await?;
        // A manual record names no calibration branch (PolicyRecord refuses
        // one that does), so there is nothing to rerun.
        if record.rule() != ThresholdRule::Manual {
            // Locked by the statement that reads them: a row erased after
            // its snapshot is skipped once its eraser commits, never read
            // unlocked from a later snapshot.
            let samples = transaction
                .query(
                    &format!(
                        "{} AND t.branch = ANY($1) FOR SHARE OF b, t, o",
                        adjudicated_query(&work)
                    ),
                    &[&branches],
                )
                .await
                .map_err(database)?
                .iter()
                .map(adjudicated_sample)
                .collect::<Result<Vec<_>, _>>()?;
            if samples.len() != branches.len() {
                return Err(refuse(
                    "a calibration branch is not an adjudicated calibration-slice branch",
                ));
            }
            match PolicyRecord::calibrate(
                record.version(),
                record.rule(),
                policy.calibration_rate(),
                &samples,
            ) {
                Ok(chosen) if chosen == *record => {}
                _ => {
                    return Err(refuse(
                        "the rule on the stored adjudications of the calibration branches \
                         chooses another threshold",
                    ))
                }
            }
        }
        transaction
            .execute(
                &format!(
                    "INSERT INTO {work}.triage_policy \
                     (version, rule, threshold, calibration_rate, alpha, delta, \
                      calibration_size) \
                     VALUES ($1, $2, $3, $4, $5, $6, $7)"
                ),
                &[
                    &record.version(),
                    &record.rule().name(),
                    &threshold,
                    &policy.calibration_rate(),
                    &alpha,
                    &delta,
                    &calibration_size,
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

    /// A recorded policy with its calibration branches, or `None`. A policy
    /// whose stored calibration set is not the size it was recorded with is
    /// refused as a corrupt row rather than returned.
    pub async fn load_policy(&self, version: &str) -> Result<Option<PolicyRecord>, PgError> {
        check_text("triage_policy.version", version)?;
        let work = &self.schemas.work;
        let Some(row) = self
            .client
            .query_opt(
                &format!(
                    "SELECT rule, threshold, calibration_rate, alpha, delta, calibration_size \
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
        let branches: Vec<BranchId> = self
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
        let calibration_size: i32 = row.get(5);
        if usize::try_from(calibration_size).ok() != Some(branches.len()) {
            return Err(corrupt(format!(
                "{} calibration branches, recorded with {calibration_size}",
                branches.len()
            )));
        }
        PolicyRecord::from_parts(version, threshold, row.get(2), rule, branches)
            .map(Some)
            .map_err(|error| corrupt(error.to_string()))
    }

    /// Every adjudicated calibration-slice branch as a calibration sample, in
    /// branch order. A recorded policy's
    /// [`held_out`](PolicyRecord::held_out) of these are the adjudications
    /// its harm rate may be estimated from.
    ///
    /// Each row is rechecked against the policy it cites
    /// ([`TriagePolicy::explains`]), as `record_triage` and work migration 10
    /// check a row when it is written: rows stored before that check, or
    /// around it, can hold a slice row no policy logs (one under a policy
    /// with calibration rate zero, or with propensity one) or cite a version
    /// no table recorded.
    ///
    /// # Errors
    /// Refuses the whole read as `PgError::CorruptRow` rather than serve or
    /// silently skip such a row: naming `branch_triage` for a row citing an
    /// unrecorded policy or one its policy cannot have produced, and
    /// `triage_policy` for a cited policy `TriagePolicy::new` refuses. The
    /// calibration draw is not stored, so that a slice row was drawn into
    /// the slice remains its writer's word.
    pub async fn adjudicated_samples(&self) -> Result<Vec<(BranchId, CalibrationSample)>, PgError> {
        let work = &self.schemas.work;
        self.client
            .query(
                &format!("{} ORDER BY t.branch", adjudicated_query(work)),
                &[],
            )
            .await
            .map_err(database)?
            .iter()
            .map(adjudicated_sample)
            .collect()
    }
}

/// Every adjudicated calibration-slice branch with its logged triage, its
/// verdict and the policy the triage cites, as [`adjudicated_sample`] reads a
/// row; callers append further conditions, the order and any locking clause
/// (the tables are `b`, `t` and `o`; the policy, `p`, is outer-joined and
/// cannot be locked). A branch is adjudicated at most once, so there is one
/// row per branch.
fn adjudicated_query(work: &Identifier) -> String {
    format!(
        "SELECT t.branch, t.decision, t.eligible, t.calibration_slice, t.score, \
                t.auto_propensity, o.outcome = 'adjudicated_harmful', t.policy_version, \
                p.threshold, p.calibration_rate \
         FROM {work}.branch_triage t \
         JOIN {work}.branch b ON b.id = t.branch \
         JOIN {work}.branch_outcome o ON o.branch = t.branch \
         AND o.outcome IN ('adjudicated_harmful', 'adjudicated_harmless') \
         LEFT JOIN {work}.triage_policy p ON p.version = t.policy_version \
         WHERE t.calibration_slice"
    )
}

/// One row of [`adjudicated_query`] as a calibration sample keyed by its
/// branch, once the policy it cites is shown to explain it.
fn adjudicated_sample(row: &Row) -> Result<(BranchId, CalibrationSample), PgError> {
    let branch: String = row.get(0);
    let corrupt = |reason: String| PgError::CorruptRow {
        table: "branch_triage",
        reason,
    };
    let decision = match row.get::<_, &str>(1) {
        "auto_propose" => TriageDecision::AutoPropose,
        "escalate" => TriageDecision::Escalate,
        "discard" => TriageDecision::Discard,
        other => return Err(corrupt(format!("decision {other:?}"))),
    };
    let triage = TriageOutcome {
        decision,
        eligible: row.get(2),
        calibration_slice: row.get(3),
        score: row.get(4),
        auto_propensity: row.get(5),
    };
    let version: String = row.get(7);
    // calibration_rate is NOT NULL, so a NULL is a version with no row.
    let Some(rate) = row.get::<_, Option<f64>>(9) else {
        return Err(corrupt(format!(
            "calibration row {branch:?} cites policy {version:?}, which is not recorded"
        )));
    };
    let threshold = match row.get::<_, Option<f32>>(8) {
        None => AutoThreshold::Never,
        Some(threshold) => AutoThreshold::AtLeast(threshold),
    };
    let policy = TriagePolicy::new(threshold, rate).map_err(|error| PgError::CorruptRow {
        table: "triage_policy",
        reason: format!("policy {version:?}: {error}"),
    })?;
    policy.explains(&triage).map_err(|error| {
        corrupt(format!(
            "calibration row {branch:?} is not one policy {version:?} can have produced: {error}"
        ))
    })?;
    // explains admits a slice row only if it is eligible, so this holds.
    let sample = triage
        .adjudicate(row.get(6))
        .ok_or_else(|| corrupt(format!("calibration row {branch:?} is not eligible")))?;
    Ok((BranchId(branch), sample))
}
