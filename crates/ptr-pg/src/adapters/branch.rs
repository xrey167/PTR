//! Sealed agent branches, their triage log and their observed outcomes, in the
//! work schema. A stored branch is a candidate; nothing here merges it.

use std::collections::BTreeMap;

use ptr_branch::{
    BranchError, BranchId, BranchOp, InputsDigest, RangeDigest, SealedBranch, SealedBranchParts,
    TriageDecision, TriageOutcome, ValueDigest,
};
use ptr_semdb::{SemanticPayload, SemanticValue};
use ptr_types::{CommitIndex, Generation, PrincipalId, Revision, TypeId};

use super::{check_text, database, digest_from, to_i64, to_u64, PgSubstrate};
use crate::error::PgError;

/// What later happened to a triaged branch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchOutcome {
    /// Its merge plan committed at this index.
    Merged(CommitIndex),
    /// A later commit at this index reverted it.
    Reverted(CommitIndex),
    /// Certification against a newer snapshot refused it.
    Conflicted,
    Discarded,
    /// A person judged that auto-proposing it would have been harmful.
    AdjudicatedHarmful,
    AdjudicatedHarmless,
}

impl BranchOutcome {
    fn name(self) -> &'static str {
        match self {
            Self::Merged(_) => "merged",
            Self::Reverted(_) => "reverted",
            Self::Conflicted => "conflicted",
            Self::Discarded => "discarded",
            Self::AdjudicatedHarmful => "adjudicated_harmful",
            Self::AdjudicatedHarmless => "adjudicated_harmless",
        }
    }

    fn commit_index(self) -> Option<u64> {
        match self {
            Self::Merged(index) | Self::Reverted(index) => Some(index.0),
            _ => None,
        }
    }
}

impl PgSubstrate {
    /// Store a sealed branch with every declared dependency and staged op,
    /// including the base value and the base input set of every touched key.
    /// A branch id is stored once; storing it again is refused by the
    /// primary key.
    ///
    /// Only a `SealedBranch` can be stored, and one exists only once
    /// `SealedBranch::from_parts` (or sealing, which goes through it) has
    /// checked every sealing invariant: no operation on a key reserved to
    /// ingress, no `Put` or `Remove` of an unread key, a base value and an
    /// input set for exactly the keys operations touch. They are rechecked
    /// here before any row is written, and a branch that breaks one is
    /// refused as [`PgError::InvalidBranch`], as is a string PostgreSQL
    /// `text` cannot hold ([`PgError::InvalidText`]).
    pub async fn store_branch(&mut self, branch: &SealedBranch) -> Result<(), PgError> {
        let work = self.schemas.work.clone();
        branch.recheck().map_err(|error| PgError::InvalidBranch {
            branch: branch.id().0.clone(),
            error,
        })?;
        check_branch_text(branch)?;
        let id = branch.id().0.as_str();
        let transaction = self.client.transaction().await.map_err(database)?;
        transaction
            .execute(
                &format!(
                    "INSERT INTO {work}.branch (id, author, base_revision) VALUES ($1, $2, $3)"
                ),
                &[
                    &id,
                    &branch.author().0,
                    &to_i64(branch.base_revision().0, "base_revision")?,
                ],
            )
            .await
            .map_err(database)?;
        for (key, digest) in branch.reads() {
            transaction
                .execute(
                    &format!(
                        "INSERT INTO {work}.branch_read (branch, key, digest) VALUES ($1, $2, $3)"
                    ),
                    &[&id, key, &digest.as_bytes().to_vec()],
                )
                .await
                .map_err(database)?;
        }
        for (prefix, digest) in branch.scans() {
            transaction
                .execute(
                    &format!(
                        "INSERT INTO {work}.branch_scan (branch, prefix, digest) VALUES ($1, $2, $3)"
                    ),
                    &[&id, prefix, &digest.as_bytes().to_vec()],
                )
                .await
                .map_err(database)?;
        }
        for (target, generation) in branch.relied() {
            transaction
                .execute(
                    &format!(
                        "INSERT INTO {work}.branch_relied (branch, target, generation) \
                         VALUES ($1, $2, $3)"
                    ),
                    &[&id, target, &to_i64(generation.0, "generation")?],
                )
                .await
                .map_err(database)?;
        }
        // A sealed branch records a base value and an input set for exactly
        // the keys its operations touch, so the two maps zip key by key.
        for ((key, digest), inputs) in branch
            .touched_base()
            .iter()
            .zip(branch.touched_inputs().values())
        {
            transaction
                .execute(
                    &format!(
                        "INSERT INTO {work}.branch_touched (branch, key, base_digest, inputs_digest) \
                         VALUES ($1, $2, $3, $4)"
                    ),
                    &[
                        &id,
                        key,
                        &digest.as_bytes().to_vec(),
                        &inputs.as_bytes().to_vec(),
                    ],
                )
                .await
                .map_err(database)?;
        }
        for (ordinal, op) in branch.ops().iter().enumerate() {
            let row = OpRow::from_op(op);
            let ordinal = i32::try_from(ordinal).map_err(|_| PgError::OutOfRange {
                field: "branch_op.ordinal",
            })?;
            transaction
                .execute(
                    &format!(
                        "INSERT INTO {work}.branch_op \
                         (branch, ordinal, kind, key, value_kind, value_text, value_type, \
                          value_source, value_bytes, amount, member) \
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)"
                    ),
                    &[
                        &id,
                        &ordinal,
                        &row.kind,
                        &row.key,
                        &row.value_kind,
                        &row.value_text,
                        &row.value_type,
                        &row.value_source,
                        &row.value_bytes,
                        &row.amount,
                        &row.member,
                    ],
                )
                .await
                .map_err(database)?;
        }
        transaction.commit().await.map_err(database)
    }

    /// Load a stored branch exactly as it was sealed.
    ///
    /// The rows are rebuilt through `SealedBranch::from_parts`, so what comes
    /// back passes every sealing invariant a freshly sealed branch does; rows
    /// changed after they were stored come back as an error, never as a
    /// branch certification could plan from.
    ///
    /// # Errors
    /// Refuses a branch stored before the input sets of touched keys were
    /// recorded as [`PgError::BranchWithoutInputSets`]: it cannot be
    /// certified and must be re-run. A digest that is not 32 bytes, or a
    /// key, prefix or target stored twice, is a [`PgError::CorruptRow`]; two
    /// generations relied on for one target, or rows that break a sealing
    /// invariant (an operation on a reserved key, an overwrite of an unread
    /// key, a touched key without its base value or input set, a base value
    /// or input set no operation needs), are a [`PgError::CorruptBranch`]
    /// naming the `BranchError`.
    pub async fn load_branch(&self, id: &BranchId) -> Result<Option<SealedBranch>, PgError> {
        let work = &self.schemas.work;
        let Some(header) = self
            .client
            .query_opt(
                &format!("SELECT author, base_revision FROM {work}.branch WHERE id = $1"),
                &[&id.0],
            )
            .await
            .map_err(database)?
        else {
            return Ok(None);
        };

        let mut reads = BTreeMap::new();
        for row in self
            .client
            .query(
                &format!("SELECT key, digest FROM {work}.branch_read WHERE branch = $1"),
                &[&id.0],
            )
            .await
            .map_err(database)?
        {
            let key: String = row.get(0);
            let digest = ValueDigest::from_bytes(digest_from(row.get(1), "branch_read")?);
            if reads.insert(key, digest).is_some() {
                return Err(stored_twice("branch_read", "a key"));
            }
        }
        let mut scans = BTreeMap::new();
        for row in self
            .client
            .query(
                &format!("SELECT prefix, digest FROM {work}.branch_scan WHERE branch = $1"),
                &[&id.0],
            )
            .await
            .map_err(database)?
        {
            let prefix: String = row.get(0);
            let digest = RangeDigest::from_bytes(digest_from(row.get(1), "branch_scan")?);
            if scans.insert(prefix, digest).is_some() {
                return Err(stored_twice("branch_scan", "a prefix"));
            }
        }
        let mut relied = BTreeMap::new();
        for row in self
            .client
            .query(
                &format!(
                    "SELECT target, generation FROM {work}.branch_relied WHERE branch = $1 \
                     ORDER BY target, generation"
                ),
                &[&id.0],
            )
            .await
            .map_err(database)?
        {
            let target: String = row.get(0);
            let declared = Generation(to_u64(row.get(1), "branch_relied")?);
            match relied.insert(target.clone(), declared) {
                None => {}
                Some(earlier) if earlier == declared => {
                    return Err(stored_twice("branch_relied", "a target"));
                }
                // A branch relies on at most one generation of a target, as
                // `Branch::rely_on` enforces; rows naming two are refused
                // rather than collapsed to either.
                Some(earlier) => {
                    return Err(PgError::CorruptBranch {
                        branch: id.0.clone(),
                        error: BranchError::ConflictingReliance {
                            target,
                            relied: earlier,
                            declared,
                        },
                    });
                }
            }
        }
        let mut touched_base = BTreeMap::new();
        let mut touched_inputs = BTreeMap::new();
        for row in self
            .client
            .query(
                &format!(
                    "SELECT key, base_digest, inputs_digest FROM {work}.branch_touched \
                     WHERE branch = $1 ORDER BY key"
                ),
                &[&id.0],
            )
            .await
            .map_err(database)?
        {
            let key: String = row.get(0);
            let Some(inputs) = row.get::<_, Option<Vec<u8>>>(2) else {
                return Err(PgError::BranchWithoutInputSets {
                    branch: id.0.clone(),
                    key,
                });
            };
            let inputs = InputsDigest::from_bytes(digest_from(inputs, "branch_touched")?);
            let base = ValueDigest::from_bytes(digest_from(row.get(1), "branch_touched")?);
            if touched_inputs.insert(key.clone(), inputs).is_some()
                || touched_base.insert(key, base).is_some()
            {
                return Err(stored_twice("branch_touched", "a key"));
            }
        }
        let ops = self
            .client
            .query(
                &format!(
                    "SELECT kind, key, value_kind, value_text, value_type, value_source, \
                            value_bytes, amount, member \
                     FROM {work}.branch_op WHERE branch = $1 ORDER BY ordinal"
                ),
                &[&id.0],
            )
            .await
            .map_err(database)?
            .into_iter()
            .map(|row| {
                OpRow {
                    kind: row.get(0),
                    key: row.get(1),
                    value_kind: row.get(2),
                    value_text: row.get(3),
                    value_type: row.get(4),
                    value_source: row.get(5),
                    value_bytes: row.get(6),
                    amount: row.get(7),
                    member: row.get(8),
                }
                .into_op()
            })
            .collect::<Result<Vec<_>, _>>()?;

        SealedBranch::from_parts(SealedBranchParts {
            id: id.clone(),
            author: PrincipalId(header.get(0)),
            base_revision: Revision(to_u64(header.get(1), "branch")?),
            reads,
            scans,
            relied,
            touched_base,
            touched_inputs,
            ops,
        })
        .map(Some)
        .map_err(|error| PgError::CorruptBranch {
            branch: id.0.clone(),
            error,
        })
    }

    /// Log a branch's triage with what off-policy evaluation needs later: the
    /// score, the propensity of auto-proposing and the policy version.
    ///
    /// The version must name a recorded policy (a foreign key refuses any
    /// other), but the triage is stored as given: that its decision and
    /// propensity are the ones that policy produces for its eligibility,
    /// slice flag and score is the caller's obligation.
    pub async fn record_triage(
        &self,
        branch: &BranchId,
        triage: &TriageOutcome,
        policy_version: &str,
    ) -> Result<(), PgError> {
        check_text("branch_triage.branch", &branch.0)?;
        check_text("branch_triage.policy_version", policy_version)?;
        let work = &self.schemas.work;
        let decision = match triage.decision {
            TriageDecision::AutoPropose => "auto_propose",
            TriageDecision::Escalate => "escalate",
            TriageDecision::Discard => "discard",
        };
        self.client
            .execute(
                &format!(
                    "INSERT INTO {work}.branch_triage \
                     (branch, decision, eligible, calibration_slice, score, auto_propensity, \
                      policy_version) \
                     VALUES ($1, $2, $3, $4, $5, $6, $7)"
                ),
                &[
                    &branch.0,
                    &decision,
                    &triage.eligible,
                    &triage.calibration_slice,
                    &triage.score,
                    &triage.auto_propensity,
                    &policy_version,
                ],
            )
            .await
            .map_err(database)?;
        Ok(())
    }

    /// Append an observed outcome. Each outcome kind is recorded once per
    /// branch and never rewritten.
    pub async fn record_outcome(
        &self,
        branch: &BranchId,
        outcome: BranchOutcome,
    ) -> Result<(), PgError> {
        check_text("branch_outcome.branch", &branch.0)?;
        let work = &self.schemas.work;
        let commit = outcome
            .commit_index()
            .map(|index| to_i64(index, "commit_index"))
            .transpose()?;
        self.client
            .execute(
                &format!(
                    "INSERT INTO {work}.branch_outcome (branch, outcome, commit_index) \
                     VALUES ($1, $2, $3)"
                ),
                &[&branch.0, &outcome.name(), &commit],
            )
            .await
            .map_err(database)?;
        Ok(())
    }
}

/// A row a primary key keeps unique was found twice for one branch.
fn stored_twice(table: &'static str, what: &str) -> PgError {
    PgError::CorruptRow {
        table,
        reason: format!("{what} is stored twice for one branch"),
    }
}

/// Refuse a branch with a string PostgreSQL `text` cannot hold, before any row
/// is written.
fn check_branch_text(branch: &SealedBranch) -> Result<(), PgError> {
    check_text("branch.id", &branch.id().0)?;
    check_text("branch.author", &branch.author().0)?;
    for key in branch.reads().keys() {
        check_text("branch_read.key", key)?;
    }
    for prefix in branch.scans().keys() {
        check_text("branch_scan.prefix", prefix)?;
    }
    for target in branch.relied().keys() {
        check_text("branch_relied.target", target)?;
    }
    for key in branch.touched_base().keys() {
        check_text("branch_touched.key", key)?;
    }
    for op in branch.ops() {
        check_text("branch_op.key", op.key())?;
        match op {
            BranchOp::Put {
                value: SemanticValue::Text(text),
                ..
            } => check_text("branch_op.value_text", text)?,
            BranchOp::Put {
                value: SemanticValue::Payload(payload),
                ..
            } => {
                check_text("branch_op.value_type", &payload.type_id.0)?;
                check_text("branch_op.value_source", &payload.source)?;
            }
            BranchOp::SetInsert { member, .. } | BranchOp::SetRemove { member, .. } => {
                check_text("branch_op.member", member)?;
            }
            BranchOp::Remove { .. } | BranchOp::Add { .. } => {}
        }
    }
    Ok(())
}

/// One `branch_op` row.
struct OpRow {
    kind: String,
    key: String,
    value_kind: Option<String>,
    value_text: Option<String>,
    value_type: Option<String>,
    value_source: Option<String>,
    value_bytes: Option<Vec<u8>>,
    amount: Option<i64>,
    member: Option<String>,
}

impl OpRow {
    fn empty(kind: &str, key: &str) -> Self {
        Self {
            kind: kind.to_owned(),
            key: key.to_owned(),
            value_kind: None,
            value_text: None,
            value_type: None,
            value_source: None,
            value_bytes: None,
            amount: None,
            member: None,
        }
    }

    fn from_op(op: &BranchOp) -> Self {
        match op {
            BranchOp::Put { key, value } => {
                let mut row = Self::empty("put", key);
                match value {
                    SemanticValue::Text(text) => {
                        row.value_kind = Some("text".into());
                        row.value_text = Some(text.clone());
                    }
                    SemanticValue::Payload(payload) => {
                        row.value_kind = Some("payload".into());
                        row.value_type = Some(payload.type_id.0.clone());
                        row.value_source = Some(payload.source.clone());
                        row.value_bytes = Some(payload.bytes.clone());
                    }
                }
                row
            }
            BranchOp::Remove { key } => Self::empty("remove", key),
            BranchOp::Add { key, amount } => {
                let mut row = Self::empty("add", key);
                row.amount = Some(*amount);
                row
            }
            BranchOp::SetInsert { key, member } => {
                let mut row = Self::empty("set_insert", key);
                row.member = Some(member.clone());
                row
            }
            BranchOp::SetRemove { key, member } => {
                let mut row = Self::empty("set_remove", key);
                row.member = Some(member.clone());
                row
            }
        }
    }

    fn into_op(self) -> Result<BranchOp, PgError> {
        let corrupt = |reason: &str| PgError::CorruptRow {
            table: "branch_op",
            reason: reason.to_owned(),
        };
        let key = self.key;
        match self.kind.as_str() {
            "put" => {
                let value = match self.value_kind.as_deref() {
                    Some("text") => SemanticValue::Text(
                        self.value_text
                            .ok_or_else(|| corrupt("text put without text"))?,
                    ),
                    Some("payload") => SemanticValue::Payload(SemanticPayload {
                        type_id: TypeId(
                            self.value_type
                                .ok_or_else(|| corrupt("payload without type"))?,
                        ),
                        source: self
                            .value_source
                            .ok_or_else(|| corrupt("payload without source"))?,
                        bytes: self
                            .value_bytes
                            .ok_or_else(|| corrupt("payload without bytes"))?,
                    }),
                    _ => return Err(corrupt("put without a value kind")),
                };
                Ok(BranchOp::Put { key, value })
            }
            "remove" => Ok(BranchOp::Remove { key }),
            "add" => Ok(BranchOp::Add {
                key,
                amount: self.amount.ok_or_else(|| corrupt("add without amount"))?,
            }),
            "set_insert" => Ok(BranchOp::SetInsert {
                key,
                member: self
                    .member
                    .ok_or_else(|| corrupt("set insert without member"))?,
            }),
            "set_remove" => Ok(BranchOp::SetRemove {
                key,
                member: self
                    .member
                    .ok_or_else(|| corrupt("set remove without member"))?,
            }),
            other => Err(corrupt(&format!("unknown op kind {other:?}"))),
        }
    }
}
