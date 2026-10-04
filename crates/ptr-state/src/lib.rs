use ptr_ledger::{CommittedEvent, LedgerEvent};
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

/// The projection key prefix for peer membership (D5 in issue #23).
///
/// One key per peer, `peer:<node id>`, whose value is either
/// [`member_entry`] or [`WITHDRAWN_MEMBER`]. A single key rather than one per
/// field because a node id may itself contain `:`, and a value split on a
/// control character — which no identifier may contain — has exactly one
/// reading.
pub const MEMBER_PREFIX: &str = "peer:";

/// The value a withdrawn peer's key holds.
pub const WITHDRAWN_MEMBER: &str = "withdrawn";

/// The projection key the group's committed codec ceiling is recorded under.
pub const CODEC_CEILING_KEY: &str = "ledger:codec-ceiling";

/// The projection key for one peer's membership.
pub fn member_key(peer: &str) -> String {
    format!("{MEMBER_PREFIX}{peer}")
}

/// The peer a [`member_key`] names, or `None` for any other key.
pub fn member_of(key: &str) -> Option<&str> {
    key.strip_prefix(MEMBER_PREFIX)
        .filter(|peer| !peer.is_empty())
}

/// The value an admitted peer's key holds: `admitted`, principal and grant set
/// name, separated by line feeds.
pub fn member_entry(principal: &str, grant_set: &str) -> String {
    format!("admitted\n{principal}\n{grant_set}")
}

/// The principal and grant set of a [`member_entry`], `Some(None)` for
/// [`WITHDRAWN_MEMBER`], and `None` for a value that is exactly neither.
pub fn parse_member_entry(entry: &str) -> Option<Option<(&str, &str)>> {
    if entry == WITHDRAWN_MEMBER {
        return Some(None);
    }
    let mut fields = entry.split('\n');
    let (Some("admitted"), Some(principal), Some(grant_set), None) =
        (fields.next(), fields.next(), fields.next(), fields.next())
    else {
        return None;
    };
    if principal.is_empty() || grant_set.is_empty() {
        return None;
    }
    Some(Some((principal, grant_set)))
}

/// The projection key prefix for Pod scopes (the Pod-access follow-up to D5 in
/// issue #23).
///
/// One key per peer, `pod-scope:<node id>`, whose value is either
/// [`pod_scope_entry`] or [`WITHDRAWN_POD_SCOPE`]. A family of its own rather
/// than a field of [`MEMBER_PREFIX`]'s value: execution membership and Pod scope
/// are admitted and withdrawn independently, and one key holding both would make
/// every record of one rewrite the other.
pub const POD_SCOPE_PREFIX: &str = "pod-scope:";

/// The value a withdrawn Pod scope's key holds.
pub const WITHDRAWN_POD_SCOPE: &str = "withdrawn";

/// The projection key for one peer's Pod scope.
pub fn pod_scope_key(peer: &str) -> String {
    format!("{POD_SCOPE_PREFIX}{peer}")
}

/// The peer a [`pod_scope_key`] names, or `None` for any other key.
pub fn pod_scope_of(key: &str) -> Option<&str> {
    key.strip_prefix(POD_SCOPE_PREFIX)
        .filter(|peer| !peer.is_empty())
}

/// The value an admitted Pod scope's key holds: `admitted` and the project,
/// then one line per pair, its capability and payload type separated by a tab.
///
/// The scope is the whole record (a Pod scope is data, so nothing about it is
/// left for a host to supply), and the separators are control characters no
/// field of a committed scope may contain ([`ptr_ledger::valid_pod_scope_field`]),
/// so the value has exactly one reading.
pub fn pod_scope_entry(
    project: &ptr_types::ProjectId,
    entries: &[(ptr_types::CapabilityId, ptr_types::TypeId)],
) -> String {
    let mut value = format!("admitted\n{project}");
    for (capability, input) in entries {
        value.push('\n');
        value.push_str(&capability.0);
        value.push('\t');
        value.push_str(&input.0);
    }
    value
}

/// A Pod scope as its projection describes it: the project and the pairs.
pub type ProjectedPodScope = (
    ptr_types::ProjectId,
    Vec<(ptr_types::CapabilityId, ptr_types::TypeId)>,
);

/// The scope of a [`pod_scope_entry`], `Some(None)` for [`WITHDRAWN_POD_SCOPE`],
/// and `None` for any value no committed record projects.
///
/// Strict, not merely structural: the scope read back must also be canonical
/// by [`ptr_ledger::check_pod_scope`], the rule the codec holds the record to.
/// A value with pairs out of order, a duplicated pair, a pair too many or an
/// empty field is therefore not a scope at all, rather than one that differs
/// from the history it claims to summarise.
pub fn parse_pod_scope_entry(entry: &str) -> Option<Option<ProjectedPodScope>> {
    if entry == WITHDRAWN_POD_SCOPE {
        return Some(None);
    }
    let mut lines = entry.split('\n');
    let (Some("admitted"), Some(project)) = (lines.next(), lines.next()) else {
        return None;
    };
    let mut entries = Vec::new();
    for line in lines {
        let (capability, input) = line.split_once('\t')?;
        entries.push((
            ptr_types::CapabilityId::from(capability),
            ptr_types::TypeId::from(input),
        ));
    }
    let project = ptr_types::ProjectId::from(project);
    ptr_ledger::check_pod_scope(&project, &entries).ok()?;
    Some(Some((project, entries)))
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
        LedgerEvent::PeerAdmitted {
            peer,
            principal,
            grant_set,
        } => vec![(member_key(&peer.0), member_entry(principal, grant_set))],
        LedgerEvent::PeerWithdrawn { peer } => {
            vec![(member_key(&peer.0), WITHDRAWN_MEMBER.to_owned())]
        }
        LedgerEvent::CodecCeilingRaised { version } => {
            vec![(CODEC_CEILING_KEY.to_owned(), version.to_string())]
        }
        LedgerEvent::PodScopeAdmitted {
            peer,
            project,
            entries,
        } => vec![(pod_scope_key(&peer.0), pod_scope_entry(project, entries))],
        LedgerEvent::PodScopeWithdrawn { peer } => {
            vec![(pod_scope_key(&peer.0), WITHDRAWN_POD_SCOPE.to_owned())]
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
    }
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
}
