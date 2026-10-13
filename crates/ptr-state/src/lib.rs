use ptr_ledger::{
    CommittedEvent, ExpertAuthorityClass, ExpertAuthorityKind, ExpertFenceAction,
    ExpertFenceResume, InputInvalidationKind, KvTaintChange, LedgerEvent,
};
use ptr_pods::{ExpertCommitRecord, ExpertState, OutcomeKind, RetiredReason};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplyOutcome {
    Applied,
    Duplicate,
    OutOfOrder,
    Gap,
}

#[derive(Clone, Debug, Default)]
pub struct MaterializedState {
    pub values: BTreeMap<String, String>,
    pub last_applied: u64,
}

/// Classify an incoming commit index against the last applied one.
///
/// `None` means `incoming` is exactly the next index and may be applied. Every
/// backend (the in-memory reference, Turso, the Postgres substrate) decides
/// with this one function, so they cannot disagree about which index is next.
pub fn classify_next(last_applied: u64, incoming: u64) -> Option<ApplyOutcome> {
    if incoming == last_applied {
        Some(ApplyOutcome::Duplicate)
    } else if incoming < last_applied {
        Some(ApplyOutcome::OutOfOrder)
    } else if incoming != last_applied.saturating_add(1) {
        Some(ApplyOutcome::Gap)
    } else {
        None
    }
}

impl MaterializedState {
    /// Apply the next commit's projection entries and advance `last_applied`.
    /// Duplicate, older, or skipped indices return the corresponding
    /// `ApplyOutcome` without changing state.
    pub fn try_apply(&mut self, committed: &CommittedEvent) -> ApplyOutcome {
        let index = committed.index.0;
        if let Some(refusal) = classify_next(self.last_applied, index) {
            return refusal;
        }

        for (key, value) in projection_entries(committed) {
            self.values.insert(key, value);
        }
        self.last_applied = index;
        ApplyOutcome::Applied
    }

    pub fn apply(&mut self, committed: &CommittedEvent) {
        let _ = self.try_apply(committed);
    }
}

/// The materialized key that records that a history holds a semantic record
/// with an attributed origin (ledger tag 12): every such record projects it,
/// with the value `1`, and a record without an origin does not.
///
/// ptr-runtime replays a record without an origin only while the marker is
/// absent, and refuses a compacted snapshot in the lifecycle layout from
/// before attributed records (PTRLC001) that carries it: no history that
/// layout describes could have set it.
pub const ATTESTED_MARKER: &str = "semdb:attested";

/// The prefix of the materialized keys that record a merged branch, one key
/// per branch id ([`merged_branch_key`]). A PTRLC001 compacted snapshot that
/// carries one is refused, as for [`ATTESTED_MARKER`].
pub const MERGED_BRANCH_PREFIX: &str = "branch-merge:";

/// The materialized key a merge of `branch` projects:
/// `branch-merge:<byte length>:<id>`. The length is part of the key so that no
/// branch id, whatever bytes it holds, names another branch's key.
pub fn merged_branch_key(branch: &str) -> String {
    format!("{MERGED_BRANCH_PREFIX}{}:{branch}", branch.len())
}

/// The branch id a [`merged_branch_key`] names, or `None` for a key that is
/// not exactly one: the prefix, a decimal byte length without a leading zero,
/// `:`, and an id of exactly that many bytes.
pub fn merged_branch_of(key: &str) -> Option<&str> {
    let (length, branch) = key.strip_prefix(MERGED_BRANCH_PREFIX)?.split_once(':')?;
    let canonical = !length.is_empty()
        && length.bytes().all(|byte| byte.is_ascii_digit())
        && (length == "0" || !length.starts_with('0'));
    (canonical && length.parse::<usize>().ok()? == branch.len()).then_some(branch)
}

/// The value a merge projects at its branch's [`merged_branch_key`]: the
/// index it was committed at, `:`, and the plan digest in lowercase
/// hexadecimal.
pub fn merged_branch_entry(index: u64, plan: &[u8; 32]) -> String {
    let mut entry = format!("{index}:");
    for byte in plan {
        entry.push_str(&format!("{byte:02x}"));
    }
    entry
}

/// The commit index and plan digest of a [`merged_branch_entry`], or `None`
/// for a value that is not exactly one: a decimal index without a sign or a
/// leading zero, `:`, and 64 lowercase hexadecimal digits.
pub fn parse_merged_branch_entry(entry: &str) -> Option<(u64, [u8; 32])> {
    let (index, hex) = entry.split_once(':')?;
    let canonical_index = !index.is_empty()
        && index.bytes().all(|byte| byte.is_ascii_digit())
        && (index == "0" || !index.starts_with('0'));
    if !canonical_index || hex.len() != 64 {
        return None;
    }
    let index = index.parse().ok()?;
    let mut plan = [0; 32];
    for (byte, pair) in plan.iter_mut().zip(hex.as_bytes().chunks(2)) {
        let digit = |c: u8| match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            _ => None,
        };
        *byte = digit(pair[0])? << 4 | digit(pair[1])?;
    }
    Some((index, plan))
}

/// The key/value entries one committed event projects to.
///
/// This is the single definition of what an event means as current state; the
/// in-memory reference, the Turso adapter and any other backend (the Postgres
/// substrate in `ptr-pg`) apply exactly these entries, so two backends cannot
/// disagree about the projection of the same history.
pub fn projection_entries(committed: &CommittedEvent) -> Vec<(String, String)> {
    let index = committed.index.0;
    match &committed.event {
        // Lifecycle materialization records the position, not a second copy of
        // semantic payloads. Their authoritative replay belongs to ptr-semdb.
        LedgerEvent::SemanticDeltaCommitted {
            revision, origin, ..
        } => {
            let mut entries = vec![("semdb:revision".into(), revision.0.to_string())];
            if *origin != ptr_ledger::SemanticOrigin::Legacy {
                entries.push((ATTESTED_MARKER.into(), "1".into()));
            }
            if let ptr_ledger::SemanticOrigin::Merge(merge) = origin {
                entries.push((
                    merged_branch_key(&merge.branch),
                    merged_branch_entry(index, &merge.plan),
                ));
            }
            entries
        }
        LedgerEvent::CapsuleCommitted {
            project,
            capsule,
            generation,
        } => vec![
            (
                format!("capsule:{capsule}:generation"),
                generation.0.to_string(),
            ),
            (format!("capsule:{capsule}:project"), project.to_string()),
        ],
        LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
            vec![(format!("capsule:{capsule}:generation"), new.0.to_string())]
        }
        LedgerEvent::Revoked {
            subject,
            generation,
        } => vec![(format!("revoked:{subject}"), generation.0.to_string())],
        LedgerEvent::HardConstraintCommitted { key, generation } => {
            vec![(format!("constraint:{key}"), generation.0.to_string())]
        }
        LedgerEvent::VerifierAttested { subject, passed } => vec![(
            format!("verifier:{subject}"),
            if *passed { "pass" } else { "fail" }.to_owned(),
        )],
        LedgerEvent::ProcedurePromoted { id, generation } => vec![(
            format!("procedure:{id}:generation"),
            generation.0.to_string(),
        )],
        LedgerEvent::ProcedureRevoked { id, generation } => {
            vec![(format!("revoked:procedure:{id}"), generation.0.to_string())]
        }
        LedgerEvent::SnapshotCommitted { revision, covers } => vec![
            ("snapshot:last_revision".into(), revision.to_string()),
            ("snapshot:covers_commit".into(), covers.0.to_string()),
        ],
        // An attempt is addressed by its own commit position, so a settlement
        // overwrites the same key rather than adding a second row that a reader
        // would have to reconcile.
        LedgerEvent::EffectAttempted {
            target, operation, ..
        } => vec![
            (format!("effect:{index}:state"), "attempted".to_owned()),
            (format!("effect:{index}:target"), target.clone()),
            (format!("effect:{index}:operation"), operation.clone()),
        ],
        LedgerEvent::EffectSettled { attempt, .. } => {
            vec![(format!("effect:{}:state", attempt.0), "settled".to_owned())]
        }
        LedgerEvent::EffectReconciled {
            attempt, applied, ..
        } => vec![
            (
                format!("effect:{}:state", attempt.0),
                "reconciled".to_owned(),
            ),
            (format!("effect:{}:applied", attempt.0), applied.to_string()),
        ],
        LedgerEvent::ScopeLifecycle(_) => Vec::new(),
        LedgerEvent::ProtectedStateCommitted {
            domain,
            logical_id,
            generation,
            revision,
            ciphertext_digest,
            ..
        } => vec![
            (
                format!("protected:{domain}:{logical_id}:generation"),
                generation.0.to_string(),
            ),
            (
                format!("protected:{domain}:{logical_id}:revision"),
                revision.0.to_string(),
            ),
            (
                format!("protected:{domain}:{logical_id}:ciphertext"),
                hex_digest(ciphertext_digest),
            ),
        ],
        LedgerEvent::MeshTunnelLifecycle(event) => vec![
            (
                format!("mesh:{}:{}:revision", event.network_id, event.peer_id),
                event.revision.0.to_string(),
            ),
            (
                format!("mesh:{}:{}:state", event.network_id, event.peer_id),
                format!("{:?}", event.kind),
            ),
        ],
        LedgerEvent::ExecutionManifestAdmitted {
            manifest_digest, ..
        } => vec![(
            format!("execution-manifest:{}:state", hex_digest(manifest_digest)),
            "admitted".to_owned(),
        )],
        LedgerEvent::ExecutionManifestRevoked {
            manifest_digest, ..
        } => vec![(
            format!("execution-manifest:{}:state", hex_digest(manifest_digest)),
            "revoked".to_owned(),
        )],
        LedgerEvent::ExpertStructuralCommitted {
            project, record, ..
        } => expert_entries(project, record),
        LedgerEvent::ExpertFenceCommitted {
            project,
            pod,
            action,
            resume,
            generation,
            epoch,
            ..
        } => {
            // A readmission restores the state the fence left, which the event records; a
            // retirement ends in the terminal `Retired(InputRevoked)`.
            let state = match (action, resume) {
                (ExpertFenceAction::Fence | ExpertFenceAction::Escalate, _) => ExpertState::Fenced,
                (ExpertFenceAction::Retire, _) => ExpertState::Retired(RetiredReason::InputRevoked),
                (ExpertFenceAction::Readmit, ExpertFenceResume::Serving) => ExpertState::Serving,
                (ExpertFenceAction::Readmit, ExpertFenceResume::Frozen) => ExpertState::Frozen,
            };
            vec![(
                expert_key(&project.0, &pod.0),
                expert_entry(generation.0, state, *epoch),
            )]
        }
        // The compound invalidation of one input. The event names the input and commits to the
        // fenced set by count and digest only, so this stateless projection cannot say which
        // `expert:` keys moved: the runtime, which owns the dependency index that derives the
        // set, writes those. Here the input itself is recorded under `expert-input:` and a
        // revocation also under `revoked:`, as a `Revoked` record does.
        LedgerEvent::InputInvalidated {
            input,
            generation,
            kind,
            fence_count,
            fence_digest,
            ..
        } => {
            let mut entries = Vec::new();
            let value = match kind {
                InputInvalidationKind::Revoked => {
                    entries.push((format!("revoked:{input}"), generation.0.to_string()));
                    expert_input_entry_revoked(*fence_count, fence_digest)
                }
                InputInvalidationKind::Superseded {
                    new_generation,
                    new_digest,
                    ..
                } => expert_input_entry_superseded(
                    new_generation.0,
                    new_digest,
                    *fence_count,
                    fence_digest,
                ),
            };
            entries.push((expert_input_key(input, generation.0), value));
            entries
        }
        // An authority change. Like the compound invalidation it commits to the fenced experts
        // by count and digest only, so the `expert:` keys that moved are the runtime's to write;
        // here the authority itself is recorded under `expert-authority:`.
        LedgerEvent::ExpertAuthorityChanged {
            kind,
            id_digest,
            epoch,
            class,
            fence_count,
            fence_digest,
        } => vec![(
            expert_authority_key(*kind),
            expert_authority_entry(*epoch, id_digest, *class, *fence_count, fence_digest),
        )],
        // One entry per subject: `kv-taint:<kind>:<hex>` is `approx:<report>:<recipe>` while
        // tainted and `cleared:<recompute>` once a cold exact recomputation cleared it.
        LedgerEvent::KvTaintCommitted { subjects, change } => {
            let value = match change {
                KvTaintChange::Record {
                    report_digest,
                    recipe_digest,
                } => format!(
                    "approx:{}:{}",
                    hex_digest(report_digest),
                    hex_digest(recipe_digest)
                ),
                KvTaintChange::Clear {
                    recompute_digest, ..
                } => format!("cleared:{}", hex_digest(recompute_digest)),
            };
            subjects
                .iter()
                .map(|(kind, digest)| {
                    (
                        format!("{KV_TAINT_PREFIX}{}:{}", kind.name(), hex_digest(digest)),
                        value.clone(),
                    )
                })
                .collect()
        }
        LedgerEvent::PodEvidenceCommitted { .. }
        | LedgerEvent::PodOutputAdmitted { .. }
        | LedgerEvent::PodHypothesisCommitted { .. }
        | LedgerEvent::PolicyBundleActivated { .. }
        | LedgerEvent::PolicyBundleRevoked { .. }
        | LedgerEvent::SessionRevoked { .. }
        | LedgerEvent::TierBackendLifecycle { .. }
        | LedgerEvent::TierObjectCommitted { .. }
        | LedgerEvent::TierReplicaLifecycle { .. } => Vec::new(),
        // The head of a stable pod identity: `pod-head:<identity>` holds the canonical revision
        // address of the current head. The ledger codec already refused a malformed address, so
        // the fallback to an empty value is unreachable for a committed event.
        LedgerEvent::PodHeadAdvanced { identity, new, .. } => vec![(
            pod_head_key(&identity.0),
            // An address that does not render is projected as a value no reader parses, so a
            // restore of this state refuses it (`PodHeadError::MalformedEntry`) instead of
            // holding an empty or invented head.
            new.canonical_uri()
                .unwrap_or_else(|_| POD_HEAD_UNRENDERABLE.to_owned()),
        )],
    }
}

/// The projected value of a head whose address cannot be rendered. The ledger codec refuses such
/// an address, so a committed event never produces it.
pub const POD_HEAD_UNRENDERABLE: &str = "unrenderable";

/// The prefix of every pod head key ([`pod_head_key`]).
pub const POD_HEAD_PREFIX: &str = "pod-head:";

/// The materialized key of one pod identity's head: `pod-head:<identity>`. Its value is the
/// canonical `ptr://` revision address of the head.
pub fn pod_head_key(identity: &str) -> String {
    format!("{POD_HEAD_PREFIX}{identity}")
}

/// The prefix of every approximate-KV taint key: `kv-taint:<kind>:<hex digest>`.
pub const KV_TAINT_PREFIX: &str = "kv-taint:";

/// The prefix of every expert key ([`expert_key`]).
pub const EXPERT_PREFIX: &str = "expert:";

/// The materialized key of one expert: `expert:<project>/<pod>`. Project and pod ids are
/// restricted to `[A-Za-z0-9._:-]`, so no id contains the `/` and no two experts share a key.
pub fn expert_key(project: &str, pod: &str) -> String {
    format!("{EXPERT_PREFIX}{project}/{pod}")
}

/// The value of an expert key: `<generation>:<state>:<state_version>` with a decimal generation
/// and state version and a state name from [`expert_state_name`].
pub fn expert_entry(generation: u64, state: ExpertState, state_version: u64) -> String {
    format!("{generation}:{}:{state_version}", expert_state_name(state))
}

/// Stable lowercase name of an expert state in a materialized value. Retirement reasons are
/// joined with `-` so the value stays three `:`-separated fields.
pub fn expert_state_name(state: ExpertState) -> &'static str {
    match state {
        ExpertState::Candidate => "candidate",
        ExpertState::Gated => "gated",
        ExpertState::Serving => "serving",
        ExpertState::Frozen => "frozen",
        ExpertState::Fenced => "fenced",
        ExpertState::Superseded => "superseded",
        ExpertState::Retired(RetiredReason::Pruned) => "retired-pruned",
        ExpertState::Retired(RetiredReason::Superseded) => "retired-superseded",
        ExpertState::Retired(RetiredReason::GateFailed) => "retired-gate-failed",
        ExpertState::Retired(RetiredReason::InputRevoked) => "retired-input-revoked",
    }
}

/// The prefix of every expert input key ([`expert_input_key`]).
pub const EXPERT_INPUT_PREFIX: &str = "expert-input:";

/// The prefix of every expert dependency commitment key ([`expert_inputs_key`]).
pub const EXPERT_INPUTS_PREFIX: &str = "expert-inputs:";

/// The materialized key of one invalidated input generation:
/// `expert-input:<identity byte length>:<identity>:<generation>`. The length is part of the key
/// so that no identity, whatever bytes it holds, names another input's key.
pub fn expert_input_key(identity: &str, generation: u64) -> String {
    format!(
        "{EXPERT_INPUT_PREFIX}{}:{identity}:{generation}",
        identity.len()
    )
}

/// The value of a revoked input key: `revoked:<fence count>:<hex fence digest>`.
pub fn expert_input_entry_revoked(fence_count: u32, fence_digest: &[u8; 32]) -> String {
    format!("revoked:{fence_count}:{}", hex_digest(fence_digest))
}

/// The value of a superseded input key:
/// `superseded:<new generation>:<hex new digest>:<fence count>:<hex fence digest>`.
pub fn expert_input_entry_superseded(
    new_generation: u64,
    new_digest: &[u8; 32],
    fence_count: u32,
    fence_digest: &[u8; 32],
) -> String {
    format!(
        "superseded:{new_generation}:{}:{fence_count}:{}",
        hex_digest(new_digest),
        hex_digest(fence_digest)
    )
}

/// The identity and generation an [`expert_input_key`] names, or `None` for a key that is not
/// exactly one: the prefix, a decimal byte length without a leading zero, `:`, an identity of
/// exactly that many bytes, `:` and a decimal generation without a sign or a leading zero.
pub fn parse_expert_input_key(key: &str) -> Option<(&str, u64)> {
    let rest = key.strip_prefix(EXPERT_INPUT_PREFIX)?;
    let (length, rest) = rest.split_once(':')?;
    let canonical = |digits: &str| {
        !digits.is_empty()
            && digits.bytes().all(|byte| byte.is_ascii_digit())
            && (digits == "0" || !digits.starts_with('0'))
    };
    if !canonical(length) {
        return None;
    }
    let length: usize = length.parse().ok()?;
    let identity = rest.get(..length)?;
    let generation = rest.get(length..)?.strip_prefix(':')?;
    if !canonical(generation) {
        return None;
    }
    Some((identity, generation.parse().ok()?))
}

/// What an [`expert_input_key`] value says happened to the input generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertInputEntry {
    /// `revoked:...`: the generation was revoked.
    Revoked,
    /// `superseded:...`: the identity moved to `new_generation`.
    Superseded { new_generation: u64 },
}

/// Parse a value written by [`expert_input_entry_revoked`] or [`expert_input_entry_superseded`],
/// or `None` for anything else.
pub fn parse_expert_input_entry(value: &str) -> Option<ExpertInputEntry> {
    let mut fields = value.split(':');
    match fields.next()? {
        "revoked" => {
            let count = fields.next()?;
            let digest = fields.next()?;
            (fields.next().is_none()
                && count.parse::<u32>().is_ok()
                && digest.len() == 64
                && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .then_some(ExpertInputEntry::Revoked)
        }
        "superseded" => {
            let new_generation = fields.next()?.parse().ok()?;
            let new_digest = fields.next()?;
            let count = fields.next()?;
            let digest = fields.next()?;
            (fields.next().is_none()
                && new_digest.len() == 64
                && count.parse::<u32>().is_ok()
                && digest.len() == 64)
                .then_some(ExpertInputEntry::Superseded { new_generation })
        }
        _ => None,
    }
}

/// The generation an [`expert_inputs_entry`] value names, or `None` for a value that is not
/// `<generation>:<training>:<evaluation>:<64 hex digits>`.
pub fn parse_expert_inputs_generation(value: &str) -> Option<u64> {
    let mut fields = value.split(':');
    let generation = fields.next()?.parse().ok()?;
    let training = fields.next()?;
    let evaluation = fields.next()?;
    let digest = fields.next()?;
    (fields.next().is_none()
        && training.parse::<usize>().is_ok()
        && evaluation.parse::<usize>().is_ok()
        && digest.len() == 64
        && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
    .then_some(generation)
}

/// The materialized key of the dependency commitment of one expert: `expert-inputs:<project>/<pod>`.
pub fn expert_inputs_key(project: &str, pod: &str) -> String {
    format!("{EXPERT_INPUTS_PREFIX}{project}/{pod}")
}

/// The value of a dependency commitment key:
/// `<generation>:<training count>:<evaluation count>:<hex block digest>`.
pub fn expert_inputs_entry(
    generation: u64,
    training: usize,
    evaluation: usize,
    block_digest: &[u8; 32],
) -> String {
    format!(
        "{generation}:{training}:{evaluation}:{}",
        hex_digest(block_digest)
    )
}

/// The prefix of every expert authority key ([`expert_authority_key`]).
pub const EXPERT_AUTHORITY_PREFIX: &str = "expert-authority:";

/// Stable lowercase name of an authority kind in a materialized key.
pub fn expert_authority_kind_name(kind: ExpertAuthorityKind) -> &'static str {
    match kind {
        ExpertAuthorityKind::GateConfig => "gate-config",
        ExpertAuthorityKind::Evaluator => "evaluator",
    }
}

/// Stable lowercase name of an authority change class in a materialized value.
pub fn expert_authority_class_name(class: ExpertAuthorityClass) -> &'static str {
    match class {
        ExpertAuthorityClass::Revoke => "revoke",
        ExpertAuthorityClass::TighteningRotation => "tightening",
        ExpertAuthorityClass::NonTightening => "non-tightening",
    }
}

/// The materialized key of one authority kind: `expert-authority:<kind name>`.
pub fn expert_authority_key(kind: ExpertAuthorityKind) -> String {
    format!(
        "{EXPERT_AUTHORITY_PREFIX}{}",
        expert_authority_kind_name(kind)
    )
}

/// The kind an [`expert_authority_key`] names, or `None` for a key that is not exactly one.
pub fn parse_expert_authority_key(key: &str) -> Option<ExpertAuthorityKind> {
    let name = key.strip_prefix(EXPERT_AUTHORITY_PREFIX)?;
    [
        ExpertAuthorityKind::GateConfig,
        ExpertAuthorityKind::Evaluator,
    ]
    .into_iter()
    .find(|kind| expert_authority_kind_name(*kind) == name)
}

/// The value of an authority key:
/// `<epoch>:<hex id digest>:<class name>:<fence count>:<hex fence digest>`, the state of the
/// latest change of that kind.
pub fn expert_authority_entry(
    epoch: u64,
    id_digest: &[u8; 32],
    class: ExpertAuthorityClass,
    fence_count: u32,
    fence_digest: &[u8; 32],
) -> String {
    format!(
        "{epoch}:{}:{}:{fence_count}:{}",
        hex_digest(id_digest),
        expert_authority_class_name(class),
        hex_digest(fence_digest)
    )
}

/// The epoch and the hex id digest an [`expert_authority_entry`] value names, or `None` for a
/// value that is not exactly one.
pub fn parse_expert_authority_entry(value: &str) -> Option<(u64, &str)> {
    let mut fields = value.split(':');
    let epoch = fields.next()?;
    let id = fields.next()?;
    let class = fields.next()?;
    let count = fields.next()?;
    let digest = fields.next()?;
    let hex = |text: &str| text.len() == 64 && text.bytes().all(|b| b.is_ascii_hexdigit());
    let known_class = [
        ExpertAuthorityClass::Revoke,
        ExpertAuthorityClass::TighteningRotation,
        ExpertAuthorityClass::NonTightening,
    ]
    .into_iter()
    .any(|candidate| expert_authority_class_name(candidate) == class);
    (fields.next().is_none()
        && !epoch.starts_with('0')
        && hex(id)
        && known_class
        && count.parse::<u32>().is_ok()
        && hex(digest))
    .then(|| epoch.parse().ok().map(|epoch| (epoch, id)))
    .flatten()
}

/// The entries an expert structural record projects: one per expert the operation leaves in a
/// new state, derived from the record by [`ptr_pods::StructuralOp::outcomes`] so the projection
/// and the runtime registry cannot disagree. A record that does not decode projects nothing: the
/// runtime refuses it before it is committed, and this function has no way to report an error.
fn expert_entries(project: &ptr_types::ProjectId, record: &[u8]) -> Vec<(String, String)> {
    let Ok(record) = ExpertCommitRecord::decode_canonical(record) else {
        return Vec::new();
    };
    let mut entries = Vec::new();
    for outcome in record.proposal().op().outcomes() {
        entries.push((
            expert_key(&project.0, &outcome.pod.0),
            expert_entry(outcome.generation.0, outcome.state, outcome.state_version),
        ));
        // A version 3 record commits to the dependency block of every expert it creates.
        if let (OutcomeKind::Result, Some(inputs)) = (outcome.kind, record.inputs()) {
            entries.push((
                expert_inputs_key(&project.0, &outcome.pod.0),
                expert_inputs_entry(
                    outcome.generation.0,
                    inputs.training().len(),
                    inputs.evaluation().len(),
                    &inputs.canonical_digest(),
                ),
            ));
        }
    }
    entries
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(feature = "turso-backend")]
pub struct TursoMaterializedState {
    _database: turso::Database,
    connection: turso::Connection,
    last_applied: u64,
}

#[cfg(feature = "turso-backend")]
impl TursoMaterializedState {
    pub async fn open(path: &str) -> Result<Self, String> {
        let database = turso::Builder::new_local(path)
            .build()
            .await
            .map_err(|error| error.to_string())?;
        let connection = database.connect().map_err(|error| error.to_string())?;
        connection
            .execute_batch(
                "
                CREATE TABLE IF NOT EXISTS ptr_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    commit_index INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ptr_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_applied INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO ptr_meta(id, last_applied) VALUES (1, 0);
                ",
            )
            .await
            .map_err(|error| error.to_string())?;

        let mut rows = connection
            .query("SELECT last_applied FROM ptr_meta WHERE id = 1", ())
            .await
            .map_err(|error| error.to_string())?;
        let last_applied = match rows.next().await.map_err(|error| error.to_string())? {
            Some(row) => {
                let value: i64 = row.get(0).map_err(|error| error.to_string())?;
                u64::try_from(value)
                    .map_err(|_| "negative last_applied in Turso state".to_owned())?
            }
            None => 0,
        };

        Ok(Self {
            _database: database,
            connection,
            last_applied,
        })
    }

    pub fn last_applied(&self) -> u64 {
        self.last_applied
    }

    pub async fn get(&self, key: &str) -> Result<Option<String>, String> {
        let mut rows = self
            .connection
            .query("SELECT value FROM ptr_state WHERE key = ?1", (key,))
            .await
            .map_err(|error| error.to_string())?;
        match rows.next().await.map_err(|error| error.to_string())? {
            Some(row) => row
                .get::<String>(0)
                .map(Some)
                .map_err(|error| error.to_string()),
            None => Ok(None),
        }
    }

    /// Persist the next commit's projection entries and watermark in one
    /// transaction, then advance the local watermark. Duplicate, older, or
    /// skipped indices return an outcome without writing.
    ///
    /// # Errors
    /// Returns database transaction, statement, or commit errors as strings;
    /// the local watermark advances only after a successful commit.
    pub async fn try_apply(&mut self, committed: &CommittedEvent) -> Result<ApplyOutcome, String> {
        let index = committed.index.0;
        if let Some(refusal) = classify_next(self.last_applied, index) {
            return Ok(refusal);
        }

        let tx = self
            .connection
            .transaction()
            .await
            .map_err(|error| error.to_string())?;

        for (key, value) in projection_entries(committed) {
            tx.execute(
                "
                INSERT INTO ptr_state(key, value, commit_index)
                VALUES (?1, ?2, ?3)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    commit_index = excluded.commit_index
                ",
                (key, value, index as i64),
            )
            .await
            .map_err(|error| error.to_string())?;
        }

        tx.execute(
            "UPDATE ptr_meta SET last_applied = ?1 WHERE id = 1",
            (index as i64,),
        )
        .await
        .map_err(|error| error.to_string())?;
        tx.commit().await.map_err(|error| error.to_string())?;
        self.last_applied = index;
        Ok(ApplyOutcome::Applied)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_kv_taint_event_projects_one_entry_per_subject() {
        use ptr_ledger::KvTaintSubjectKind;
        use ptr_types::CommitIndex;
        let project = |change| {
            projection_entries(&CommittedEvent {
                index: CommitIndex(1),
                event: LedgerEvent::KvTaintCommitted {
                    subjects: vec![
                        (KvTaintSubjectKind::Action, [1; 32]),
                        (KvTaintSubjectKind::Request, [2; 32]),
                    ],
                    change,
                },
            })
        };
        let record = project(KvTaintChange::Record {
            report_digest: [3; 32],
            recipe_digest: [4; 32],
        });
        let approx = format!("approx:{}:{}", "03".repeat(32), "04".repeat(32));
        assert_eq!(
            record,
            vec![
                (
                    format!("kv-taint:action:{}", "01".repeat(32)),
                    approx.clone()
                ),
                (format!("kv-taint:request:{}", "02".repeat(32)), approx),
            ]
        );
        let clear = project(KvTaintChange::Clear {
            recipe_digest: [4; 32],
            recompute_digest: [5; 32],
        });
        let cleared = format!("cleared:{}", "05".repeat(32));
        assert_eq!(
            clear,
            vec![
                (
                    format!("kv-taint:action:{}", "01".repeat(32)),
                    cleared.clone()
                ),
                (format!("kv-taint:request:{}", "02".repeat(32)), cleared),
            ]
        );
    }

    #[test]
    fn a_fence_event_projects_fenced_at_its_epoch_and_a_readmission_serving_at_the_next() {
        use ptr_ledger::{ExpertFenceCause, ExpertFenceRevocation};
        use ptr_types::{CommitIndex, Generation, PodId, ProjectId};
        let project_event = |event| {
            projection_entries(&CommittedEvent {
                index: CommitIndex(1),
                event,
            })
        };
        let event = |action, resume, expected, epoch| LedgerEvent::ExpertFenceCommitted {
            project: ProjectId::from("p"),
            pod: PodId::from("a"),
            action,
            cause: ExpertFenceCause::AuthorityQuarantine,
            resume,
            generation: Generation(2),
            expected_state_version: expected,
            epoch,
            revocation: None,
            evidence: None,
        };
        assert_eq!(
            project_event(event(
                ExpertFenceAction::Fence,
                ExpertFenceResume::Serving,
                1,
                2
            )),
            vec![("expert:p/a".to_owned(), "2:fenced:2".to_owned())]
        );
        assert_eq!(
            project_event(event(
                ExpertFenceAction::Readmit,
                ExpertFenceResume::Serving,
                2,
                3
            )),
            vec![("expert:p/a".to_owned(), "2:serving:3".to_owned())]
        );
        // A readmission restores the state the fence left: a frozen expert stays frozen.
        assert_eq!(
            project_event(event(
                ExpertFenceAction::Readmit,
                ExpertFenceResume::Frozen,
                2,
                3
            )),
            vec![("expert:p/a".to_owned(), "2:frozen:3".to_owned())]
        );
        // An escalation is a fence: fenced at the new epoch.
        let escalation = LedgerEvent::ExpertFenceCommitted {
            project: ProjectId::from("p"),
            pod: PodId::from("a"),
            action: ExpertFenceAction::Escalate,
            cause: ExpertFenceCause::InputRevoked,
            resume: ExpertFenceResume::Frozen,
            generation: Generation(2),
            expected_state_version: 2,
            epoch: 3,
            revocation: Some(ExpertFenceRevocation {
                subject: "k".to_owned(),
                generation: Generation(1),
            }),
            evidence: None,
        };
        assert_eq!(
            project_event(escalation),
            vec![("expert:p/a".to_owned(), "2:fenced:3".to_owned())]
        );
        // A retirement of the revoked, fenced expert is terminal: the retired state at the next
        // epoch, whatever state the fence left.
        for resume in [ExpertFenceResume::Serving, ExpertFenceResume::Frozen] {
            let retirement = LedgerEvent::ExpertFenceCommitted {
                project: ProjectId::from("p"),
                pod: PodId::from("a"),
                action: ExpertFenceAction::Retire,
                cause: ExpertFenceCause::InputRevoked,
                resume,
                generation: Generation(2),
                expected_state_version: 3,
                epoch: 4,
                revocation: Some(ExpertFenceRevocation {
                    subject: "k".to_owned(),
                    generation: Generation(1),
                }),
                evidence: None,
            };
            assert_eq!(
                project_event(retirement),
                vec![(
                    "expert:p/a".to_owned(),
                    "2:retired-input-revoked:4".to_owned()
                )]
            );
        }
    }

    #[test]
    fn expert_entries_use_a_stable_three_field_encoding() {
        use ptr_pods::{ExpertState, RetiredReason};
        assert_eq!(expert_key("p", "a"), "expert:p/a");
        assert_eq!(expert_entry(2, ExpertState::Serving, 1), "2:serving:1");
        assert_eq!(
            expert_entry(1, ExpertState::Retired(RetiredReason::GateFailed), 9),
            "1:retired-gate-failed:9"
        );
        let names = [
            ExpertState::Candidate,
            ExpertState::Gated,
            ExpertState::Serving,
            ExpertState::Frozen,
            ExpertState::Fenced,
            ExpertState::Superseded,
            ExpertState::Retired(RetiredReason::Pruned),
            ExpertState::Retired(RetiredReason::Superseded),
            ExpertState::Retired(RetiredReason::GateFailed),
            ExpertState::Retired(RetiredReason::InputRevoked),
        ]
        .map(expert_state_name);
        let distinct: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(distinct.len(), names.len());
        assert!(names.iter().all(|name| !name.contains(':')));
    }

    use super::*;

    #[test]
    fn only_the_exact_next_index_is_applicable() {
        assert_eq!(classify_next(4, 5), None);
        assert_eq!(classify_next(4, 4), Some(ApplyOutcome::Duplicate));
        assert_eq!(classify_next(4, 2), Some(ApplyOutcome::OutOfOrder));
        assert_eq!(classify_next(4, 7), Some(ApplyOutcome::Gap));
        assert_eq!(classify_next(0, 1), None);
        assert_eq!(
            classify_next(u64::MAX, u64::MAX),
            Some(ApplyOutcome::Duplicate)
        );
    }

    #[test]
    fn an_input_invalidation_projects_the_input_and_never_guesses_the_fenced_experts() {
        use ptr_ledger::InputInvalidationKind;
        use ptr_types::{CommitIndex, Generation};
        let project = |kind| {
            projection_entries(&CommittedEvent {
                index: CommitIndex(4),
                event: LedgerEvent::InputInvalidated {
                    input: "dataset:a:b".to_owned(),
                    generation: Generation(3),
                    binding_digest: [1; 32],
                    kind,
                    fence_count: 12,
                    fence_digest: [2; 32],
                },
            })
        };
        // A revocation also lands under `revoked:`, as a `Revoked` record does.
        let revoked = project(InputInvalidationKind::Revoked);
        let value = format!("revoked:12:{}", "02".repeat(32));
        assert_eq!(
            revoked,
            vec![
                ("revoked:dataset:a:b".to_owned(), "3".to_owned()),
                ("expert-input:11:dataset:a:b:3".to_owned(), value.clone()),
            ]
        );
        assert_eq!(
            parse_expert_input_key("expert-input:11:dataset:a:b:3"),
            Some(("dataset:a:b", 3))
        );
        assert_eq!(
            parse_expert_input_entry(&value),
            Some(ExpertInputEntry::Revoked)
        );
        // A supersession records the move and no revocation.
        let superseded = project(InputInvalidationKind::Superseded {
            old_generation: Generation(3),
            new_generation: Generation(4),
            new_digest: [5; 32],
            replacement_ref: [6; 32],
        });
        assert_eq!(superseded.len(), 1);
        assert_eq!(
            superseded[0].1,
            format!("superseded:4:{}:12:{}", "05".repeat(32), "02".repeat(32))
        );
        assert_eq!(
            parse_expert_input_entry(&superseded[0].1),
            Some(ExpertInputEntry::Superseded { new_generation: 4 })
        );
        // The projection names no `expert:` key: the set is derived from the runtime's index.
        assert!(revoked
            .iter()
            .chain(&superseded)
            .all(|(key, _)| !key.starts_with(EXPERT_PREFIX)));
    }

    #[test]
    fn the_input_key_and_value_parsers_accept_exactly_what_the_writers_write() {
        for identity in ["a", "x:y", "with space", "multi\u{e9}byte", "7:3"] {
            for generation in [1u64, 10, u64::MAX] {
                let key = expert_input_key(identity, generation);
                assert_eq!(parse_expert_input_key(&key), Some((identity, generation)));
            }
        }
        for bad in [
            "expert-input:",
            "expert-input:1:a",
            "expert-input:1:a:",
            "expert-input:01:a:1",
            "expert-input:2:a:1",
            "expert-input:1:a:01",
            "expert-input:1:a:-1",
            "expert-input:x:a:1",
            "expert:1:a:1",
        ] {
            assert_eq!(parse_expert_input_key(bad), None, "{bad}");
        }
        for bad in [
            "",
            "revoked",
            "revoked:1",
            "revoked:1:abc",
            "revoked:x:00",
            "superseded:1",
            "unknown:1:2",
        ] {
            assert_eq!(parse_expert_input_entry(bad), None, "{bad}");
        }
        let digest = "ab".repeat(32);
        assert_eq!(
            parse_expert_input_entry(&format!("revoked:0:{digest}:extra")),
            None
        );
        assert_eq!(
            parse_expert_inputs_generation(&expert_inputs_entry(3, 2, 1, &[0xab; 32])),
            Some(3)
        );
        assert_eq!(parse_expert_inputs_generation("3:2:1:zz"), None);
        assert_eq!(parse_expert_inputs_generation("3:2:1"), None);
    }

    #[test]
    fn a_version_3_record_commits_to_its_dependency_block_per_created_expert() {
        use ptr_pods::{
            BaseModelDigest, BaseModelId, BaseModelRef, ExpertInputRef, ExpertInputs, ExpertLevel,
            ExpertPodSpec, ExpertProposal, ManifestDigest, ProposerRef, RationaleDigest,
            StructuralOp,
        };
        use ptr_types::{CommitIndex, Generation, PodId, ProjectId};
        let project = ProjectId::from("p");
        let base = BaseModelRef::new(
            BaseModelId::new("b").unwrap(),
            BaseModelDigest::new([2; 32]).unwrap(),
            1,
        )
        .unwrap();
        let spec = ExpertPodSpec::new(
            project.clone(),
            PodId::from("a"),
            ManifestDigest::new([1; 32]).unwrap(),
            ExpertLevel::Segment,
            base,
            Generation(1),
        )
        .unwrap();
        let proposal = ExpertProposal::new(
            project.clone(),
            ProposerRef::new(project.clone(), PodId::from("meta"), Generation(1)).unwrap(),
            CommitIndex(1),
            StructuralOp::Birth { result: spec },
            RationaleDigest::new([3; 32]).unwrap(),
        )
        .unwrap();
        let block = ExpertInputs::new(
            vec![ExpertInputRef::new("d", Generation(1), [4; 32]).unwrap()],
            vec![ExpertInputRef::new("e", Generation(2), [5; 32]).unwrap()],
        )
        .unwrap();
        let project_record = |record: ExpertCommitRecord| {
            projection_entries(&CommittedEvent {
                index: CommitIndex(2),
                event: LedgerEvent::ExpertStructuralCommitted {
                    project: project.clone(),
                    op_kind: ptr_ledger::ExpertOpKind::Birth,
                    proposal_id: [0; 32],
                    operation_id: [0; 32],
                    evidence_digest: [7; 32],
                    record: record.encode_canonical(),
                },
            })
        };
        let v3 = ExpertCommitRecord::new_with_inputs(
            proposal.clone(),
            [7; 32],
            1,
            [8; 32],
            b"report".to_vec(),
            block.clone(),
        )
        .unwrap();
        let entries = project_record(v3);
        assert_eq!(
            entries,
            vec![
                (
                    "expert:p/a".to_owned(),
                    expert_entry(1, ExpertState::Serving, 1)
                ),
                (
                    "expert-inputs:p/a".to_owned(),
                    expert_inputs_entry(1, 1, 1, &block.canonical_digest())
                ),
            ]
        );
        // Versions 1 and 2 project exactly what they always did.
        let v2 = ExpertCommitRecord::new_with_evidence(
            proposal,
            [7; 32],
            1,
            [8; 32],
            b"report".to_vec(),
        )
        .unwrap();
        assert_eq!(project_record(v2).len(), 1);
    }

    #[test]
    fn an_authority_change_projects_one_entry_per_kind_and_the_value_parses_back() {
        use ptr_types::CommitIndex;
        let project = |kind, class| {
            projection_entries(&CommittedEvent {
                index: CommitIndex(5),
                event: LedgerEvent::ExpertAuthorityChanged {
                    kind,
                    id_digest: [0xab; 32],
                    epoch: 7,
                    class,
                    fence_count: 3,
                    fence_digest: [0xcd; 32],
                },
            })
        };
        for (kind, name) in [
            (ExpertAuthorityKind::GateConfig, "gate-config"),
            (ExpertAuthorityKind::Evaluator, "evaluator"),
        ] {
            let entries = project(kind, ExpertAuthorityClass::Revoke);
            let key = format!("expert-authority:{name}");
            let value = format!("7:{}:revoke:3:{}", "ab".repeat(32), "cd".repeat(32));
            assert_eq!(entries, vec![(key.clone(), value.clone())]);
            assert_eq!(expert_authority_key(kind), key);
            assert_eq!(parse_expert_authority_key(&key), Some(kind));
            assert_eq!(
                parse_expert_authority_entry(&value),
                Some((7, "ab".repeat(32).as_str()))
            );
        }
        for class in [
            ExpertAuthorityClass::TighteningRotation,
            ExpertAuthorityClass::NonTightening,
        ] {
            let entries = project(ExpertAuthorityKind::GateConfig, class);
            assert!(parse_expert_authority_entry(&entries[0].1).is_some());
        }
        for bad in [
            "expert-authority:",
            "expert-authority:gate",
            "expert-authority:gate-config:x",
            "expert:gate-config",
        ] {
            assert_eq!(parse_expert_authority_key(bad), None, "{bad}");
        }
        let digest = "ab".repeat(32);
        for bad in [
            String::new(),
            format!("0:{digest}:revoke:3:{digest}"),
            format!("07:{digest}:revoke:3:{digest}"),
            format!("7:{digest}:unknown:3:{digest}"),
            format!("7:{digest}:revoke:x:{digest}"),
            format!("7:short:revoke:3:{digest}"),
            format!("7:{digest}:revoke:3:{digest}:extra"),
        ] {
            assert_eq!(parse_expert_authority_entry(&bad), None, "{bad}");
        }
    }
}
