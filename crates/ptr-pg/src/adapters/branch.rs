//! Sealed agent branches, their triage log and their observed outcomes, in the
//! work schema. A stored branch is a candidate; nothing here merges it.

use std::collections::BTreeMap;

use ptr_branch::{
    BranchId, BranchOp, RangeDigest, SealedBranch, TriageDecision, TriageOutcome, ValueDigest,
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
    /// Store a sealed branch with every declared dependency and staged op.
    /// A branch id is stored once; storing it again is refused by the
    /// primary key.
    pub async fn store_branch(&mut self, branch: &SealedBranch) -> Result<(), PgError> {
        let work = self.schemas.work.clone();
        check_branch_text(branch)?;
        let id = branch.id.0.as_str();
        let transaction = self.client.transaction().await.map_err(database)?;
        transaction
            .execute(
                &format!(
                    "INSERT INTO {work}.branch (id, author, base_revision) VALUES ($1, $2, $3)"
                ),
                &[
                    &id,
                    &branch.author.0,
                    &to_i64(branch.base_revision.0, "base_revision")?,
                ],
            )
            .await
            .map_err(database)?;
        for (key, digest) in &branch.reads {
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
        for (prefix, digest) in &branch.scans {
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
        for (target, generation) in &branch.relied {
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
        for (key, digest) in &branch.touched_base {
            transaction
                .execute(
                    &format!(
                        "INSERT INTO {work}.branch_touched (branch, key, base_digest) \
                         VALUES ($1, $2, $3)"
                    ),
                    &[&id, key, &digest.as_bytes().to_vec()],
                )
                .await
                .map_err(database)?;
        }
        for (ordinal, op) in branch.ops.iter().enumerate() {
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
            reads.insert(
                row.get::<_, String>(0),
                ValueDigest::from_bytes(digest_from(row.get(1), "branch_read")?),
            );
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
            scans.insert(
                row.get::<_, String>(0),
                RangeDigest::from_bytes(digest_from(row.get(1), "branch_scan")?),
            );
        }
        let mut relied = BTreeMap::new();
        for row in self
            .client
            .query(
                &format!("SELECT target, generation FROM {work}.branch_relied WHERE branch = $1"),
                &[&id.0],
            )
            .await
            .map_err(database)?
        {
            relied.insert(
                row.get::<_, String>(0),
                Generation(to_u64(row.get(1), "branch_relied")?),
            );
        }
        let mut touched_base = BTreeMap::new();
        for row in self
            .client
            .query(
                &format!("SELECT key, base_digest FROM {work}.branch_touched WHERE branch = $1"),
                &[&id.0],
            )
            .await
            .map_err(database)?
        {
            touched_base.insert(
                row.get::<_, String>(0),
                ValueDigest::from_bytes(digest_from(row.get(1), "branch_touched")?),
            );
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

        Ok(Some(SealedBranch {
            id: id.clone(),
            author: PrincipalId(header.get(0)),
            base_revision: Revision(to_u64(header.get(1), "branch")?),
            reads,
            scans,
            relied,
            touched_base,
            ops,
        }))
    }

    /// Log a branch's triage with what off-policy evaluation needs later: the
    /// score, the propensity of auto-proposing and the policy version.
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

/// Refuse a branch with a string PostgreSQL `text` cannot hold, before any row
/// is written.
fn check_branch_text(branch: &SealedBranch) -> Result<(), PgError> {
    check_text("branch.id", &branch.id.0)?;
    check_text("branch.author", &branch.author.0)?;
    for key in branch.reads.keys() {
        check_text("branch_read.key", key)?;
    }
    for prefix in branch.scans.keys() {
        check_text("branch_scan.prefix", prefix)?;
    }
    for target in branch.relied.keys() {
        check_text("branch_relied.target", target)?;
    }
    for key in branch.touched_base.keys() {
        check_text("branch_touched.key", key)?;
    }
    for op in &branch.ops {
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
