"""Temporary review-branch-only integration; removed before the final PR."""
from pathlib import Path
import json
import re

R = Path.cwd()
def change(p, a, b, n=1):
    f = R / p
    s = f.read_text()
    assert s.count(a) == n, (p, a, s.count(a))
    f.write_text(s.replace(a, b))

change('crates/ptr-ledger/src/lib.rs', 'Generation, ProjectId};', 'Generation, ProjectId, Revision};')
change('crates/ptr-ledger/src/lib.rs', 'pub enum LedgerEvent {', '''pub enum LedgerEvent {
    /// Opaque versioned semantic transaction. ptr-runtime validates its schema
    /// and both revisions before append and again during replay.
    SemanticDeltaCommitted {
        base_revision: Revision,
        revision: Revision,
        encoded_delta: Vec<u8>,
    },''')
change('crates/ptr-ledger/src/lib.rs', '    match event {\n', '''    match event {
        LedgerEvent::SemanticDeltaCommitted { base_revision, revision, encoded_delta } => {
            out.push(8);
            put_u64(&mut out, base_revision.0);
            put_u64(&mut out, revision.0);
            put_bytes(&mut out, encoded_delta);
        }
''')
change('crates/ptr-ledger/src/lib.rs', '    let event = match tag {', '''    let event = match tag {
        8 => LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            encoded_delta: cursor.bytes()?.to_vec(),
        },''')
change('crates/ptr-ledger/src/lib.rs', 'fn put_string(out: &mut Vec<u8>, value: &str) {\n    let bytes = value.as_bytes();', '''fn put_string(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {''')
change('crates/ptr-ledger/src/lib.rs', '''    fn string(&mut self) -> io::Result<String> {
        let length = self.u32()? as usize;
        let end = self.offset.saturating_add(length);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 string"))
    }''', '''    fn bytes(&mut self) -> io::Result<&'a [u8]> {
        let length = self.u32()? as usize;
        let end = self.offset.checked_add(length).ok_or_else(truncated)?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn string(&mut self) -> io::Result<String> {
        String::from_utf8(self.bytes()?.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 string"))
    }''')
change('crates/ptr-state/src/lib.rs', '    match event {', '''    match event {
        // Lifecycle materialization records the position, not a second copy of
        // semantic payloads. Their authoritative replay belongs to ptr-semdb.
        LedgerEvent::SemanticDeltaCommitted { revision, .. } => {
            vec![("semdb:revision".into(), revision.0.to_string())]
        }''')
change('crates/ptr-runtime/src/lib.rs', 'pub mod execution;', 'pub mod execution;\npub mod semantic;')
change('crates/ptr-runtime/src/lib.rs', 'use ptr_semdb::{SemanticDelta, SemanticHost, SemanticSnapshot};', 'use ptr_semdb::{PreparedDelta, SemanticError, SemanticHost, SemanticSnapshot};')
change('crates/ptr-runtime/src/lib.rs', 'pub enum RuntimeError {', 'pub enum RuntimeError {\n    Semantic(SemanticError),')
change('crates/ptr-runtime/src/lib.rs', '''            runtime.validate_lifecycle_event(&committed.event)?;
            runtime.apply_committed(committed);''', '''            runtime.validate_lifecycle_event(&committed.event)?;
            let semantic = runtime.prepare_semantic_event(&committed.event)?;
            runtime.apply_committed(committed, semantic)?;''')
change('crates/ptr-runtime/src/lib.rs', '''            runtime.validate_lifecycle_event(&expected.event)?;
            let actual''', '''            runtime.validate_lifecycle_event(&expected.event)?;
            let semantic = runtime.prepare_semantic_event(&expected.event)?;
            let actual''')
change('crates/ptr-runtime/src/lib.rs', '            runtime.apply_committed(&committed);', '            runtime.apply_committed(&committed, semantic)?;')
p = R / 'crates/ptr-runtime/src/lib.rs'
s = p.read_text()
a = s.index('    pub fn ingest_text(&mut self, request: RequestId, text: impl Into<String>) -> Revision {')
b = s.index('\n    pub fn run_model_once', a)
p.write_text(s[:a] + s[b:])
change('crates/ptr-runtime/src/lib.rs', 'let revision = self.ingest_text(request_id.clone(), raw_text.clone());', 'let revision = self.ingest_text(request_id.clone(), raw_text.clone())?;', 2)
old = '''            let mut delta = SemanticDelta::default();
            delta.upserts.insert(
                format!("request:{request_id}:pod:{}:output_type", pod.manifest().id),
                output.type_id.to_string(),
            );'''
change('crates/ptr-runtime/src/lib.rs', old + '\n            self.semdb.apply_delta(delta);', '            self.promote_pod_output(&request_id, &pod.manifest().id, &output)?;')
change('crates/ptr-runtime/src/lib.rs', old + '\n            let (revision, _) = self.semdb.apply_delta(delta);', '            let revision = self.promote_pod_output(&request_id, &pod.manifest().id, &output)?;')
change('crates/ptr-runtime/src/lib.rs', '''        self.validate_lifecycle_event(&event)?;
        self.execution.begin_commit();''', '''        self.validate_lifecycle_event(&event)?;
        let semantic = self.prepare_semantic_event(&event)?;
        self.append_prepared(event, semantic)
    }

    fn append_prepared(&mut self, event: LedgerEvent, semantic: Option<PreparedDelta>) -> Result<CommitIndex, RuntimeError> {
        if self.execution.is_fenced() { return Err(RuntimeError::ExecutionFenced); }
        self.execution.begin_commit();''')
change('crates/ptr-runtime/src/lib.rs', '        self.apply_committed(&committed);', '        self.apply_committed(&committed, semantic)?;')
change('crates/ptr-runtime/src/lib.rs', '''            LedgerEvent::Revoked { .. }
            | LedgerEvent::ProcedureRevoked { .. }''', '''            LedgerEvent::SemanticDeltaCommitted { .. }
            | LedgerEvent::Revoked { .. }
            | LedgerEvent::ProcedureRevoked { .. }''')
change('crates/ptr-runtime/src/lib.rs', '''    fn apply_committed(&mut self, committed: &CommittedEvent) {
        match &committed.event {''', '''    fn apply_committed(&mut self, committed: &CommittedEvent, semantic: Option<PreparedDelta>) -> Result<(), RuntimeError> {
        match &committed.event {
            LedgerEvent::SemanticDeltaCommitted { .. } => {
                self.semdb.apply_prepared(semantic.ok_or(RuntimeError::Semantic(SemanticError::InvalidEncoding))?)
                    .map_err(RuntimeError::Semantic)?;
            }''')
change('crates/ptr-runtime/src/lib.rs', '''        self.emit(RuntimeEvent::CommitApplied(committed.index));
    }''', '''        self.emit(RuntimeEvent::CommitApplied(committed.index));
        Ok(())
    }''')
# Only migrate old callers, never rewrite already-fallible new tests.
for relative in ['crates/ptr-runtime/tests/runtime.rs', 'crates/ptr-runtime/tests/execution_authority.rs', 'crates/ptr-runtime/tests/common/mod.rs']:
    p = R / relative
    s = re.sub(r'(\.ingest_text\([^;\n]+\))\s*;', r'\1.unwrap();', p.read_text())
    p.write_text(s)
change('crates/ptr-runtime/tests/runtime.rs', 'assert_eq!(runtime.events().len(), 2);', '''assert_eq!(runtime.events().len(), 3);
    assert!(matches!(runtime.events()[0].event, ptr_events::RuntimeEvent::CommitApplied(_)));''')
change('crates/ptr-semdb/tests/smoke.rs', 'h.apply_delta(d);', 'h.apply_delta(d).unwrap();')
change('bins/ptr-bench/src/main.rs', '''    for i in 0..128 {
        host.dependencies_mut()
            .depends_on(format!("derived:{i}"), format!("input:{i}"));
    }''', '''    let mut setup = SemanticDelta::default();
    for i in 0..128 {
        setup.dependencies.insert(format!("derived:{i}"), [format!("input:{i}")].into());
    }
    host.apply_delta(setup).expect("valid benchmark graph");''')
change('bins/ptr-bench/src/main.rs', 'i.to_string());', 'i.to_string().into());')
change('bins/ptr-bench/src/main.rs', 'host.apply_delta(delta);', 'host.apply_delta(delta).expect("valid benchmark update");')
p = R / 'crates/ptr-ledger/tests/file_ledger.rs'
s = p.read_text().replace('Generation, ProjectId};', 'Generation, ProjectId, Revision};')
s = s.replace('let expected = vec![', '''let expected = vec![
        LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(2), revision: Revision(3), encoded_delta: vec![0, 1, 255],
        },''')
p.write_text(s)
p = R / 'crates/ptr-runtime/tests/execution_authority.rs'
s = p.read_text()
a = s.index('fn semantic_change_rejects')
b = s.index('\n#[test]', a)
old = s[a:b]
new = old.replace('''    assert!(matches!(
        runtime.execute_prepared(&session, permit),
        Err(ExecutionError::AuthorizationDenied(
            AuthorizationDenial::StaleRevision { .. }
        ))
    ));''', '''    assert_eq!(runtime.execute_prepared(&session, permit), Err(ExecutionError::StalePermit));
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("p"), &action, TTL),
        Err(ExecutionError::AuthorizationDenied(AuthorizationDenial::StaleRevision { .. }))
    ));''')
assert new != old
s = s[:a] + new + s[b:]
s += '''
#[test]
fn idempotent_semantic_ingestion_does_not_invalidate_valid_permits() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime.prepare_execution(&session, &ProjectId::from("p"), &action, TTL).unwrap();
    let before = runtime.committed_events().len();
    runtime.ingest_text(RequestId::from("request"), "ground state").unwrap();
    assert_eq!(runtime.committed_events().len(), before);
    runtime.execute_prepared(&session, permit).unwrap();
    assert_eq!(probe.executions(), 1);
}
'''
p.write_text(s)
updates = {
    'ptr-semdb': [
        'Text and typed binary/source values share a revisioned semantic state',
        'Bounded canonical PTRSD001 delta codec and staged atomic publication',
        'Transactional dependency replacement and transitive removal of stale derived values',
        'Host-bound snapshots and prepared changes reject foreign or stale admission'],
    'ptr-ledger': ['SemanticDeltaCommitted tag 8 preserves base/result revisions and opaque transaction bytes; legacy event tags are unchanged'],
    'ptr-runtime': [
        'Ordered semantic journal publication before acknowledgment or model resume',
        'Semantic payload/dependency/revision reconstruction with schema and transition validation during replay',
        'Complete typed Pod bytes and source identity are revision-significant',
        'Fallible ingestion, optimistic base revision and canonical no-op handling'],
    'ptr-state': ['Semantic transaction position and revision materialization without duplicated payload ownership'],
}
for name, items in updates.items():
    p = R / 'crates' / name / 'component.toml'
    s = p.read_text().replace('last_reviewed = "2026-09-18"', 'last_reviewed = "2026-09-19"')
    s = s.replace('implemented = [', 'implemented = [\n' + ''.join('  '+json.dumps(i)+',\n' for i in items), 1)
    s = s.replace('Durable SemDB payload/revision reconstruction and network-authenticated/scoped Pod integration', 'Network-authenticated/scoped Pod integration and neural checkpoint admission')
    s = s.replace('Reconstruct actual semantic payloads and dependency revisions durably before adding persistent execution authority', 'Add authenticated framing/snapshots and checkpoint admission before persistent execution authority')
    s = s.replace('Typed ground/derived query keys instead of String→String state', 'Typed query keys beyond the String-key/SemanticValue map')
    s = s.replace('Replace string map with typed key/value/query interfaces', 'Add typed query keys and automatic dependency capture over SemanticValue')
    s = s.replace('Unit test proving local dependency invalidation', 'Integration regression preserving local dependency closure semantics')
    s = s.replace('local_change_invalidates_only_dependency_closure unit test', 'local_change_invalidates_only_dependency_closure integration test')
    p.write_text(s)
    p = R / 'crates' / name / 'README.md'
    p.write_text(p.read_text() + '''

## P0.2 semantic journal integration

[Durable semantic-state contract](../../docs/architecture/22-durable-semantic-state.md)
records the new code/codec, ownership and replay boundaries. Publication follows
successful journal append. Typed Pod bytes and source identity participate in
semantic revisions. Logical removals do not erase log history; neural checkpoints
and authenticated framing remain separate gates. Execution evidence is in the PR.
''')
p = R / 'docs/PRIORITIES.md'
s = p.read_text().replace('checksummed framing, real snapshots, SemDB payload replay and cluster durability remain open', 'semantic payload/dependency replay implemented in the reference path; checksummed framing, real snapshots, neural checkpoint admission and cluster durability remain open')
p.write_text(s + '\n\nP0.2: [durable semantic transactions](architecture/22-durable-semantic-state.md) close the in-memory-only ingestion and type-only Pod-output data path. Record integrity and explicit checkpoint/snapshot admission remain separate requirements.\n')
p = R / 'STATUS.md'
p.write_text(p.read_text() + '\n\nP0.2 adds canonical journaled semantic transactions, typed binary Pod results, transactional dependencies and validated revision reconstruction. Invalidated current derivations are evicted; prior log records and immutable snapshots remain. See `docs/architecture/22-durable-semantic-state.md`. Neural checkpoints, secure erasure, authenticated framing and cluster composition remain open.\n')
p = R / 'docs/SECURITY_P0_REVIEW_20260919.md'
p.write_text(p.read_text() + '\n\n## P0.2 follow-up: semantic data path\n\nThe lifecycle-only-replay finding is superseded for newly journaled data by [durable semantic transactions](architecture/22-durable-semantic-state.md). Runtime replay restores typed contents, source identity, dependencies and revisions. It cannot recover legacy unjournaled contents, erase historical log bytes, or admit neural checkpoints. Other release gates remain open.\n')
p = R / 'docs/architecture/21-scoped-execution.md'
p.write_text(p.read_text() + '\n\nP0.2 follow-up: [semantic payload/dependency/revision reconstruction](22-durable-semantic-state.md) is now implemented. Real semantic commits invalidate pending permits; validated no-ops do not. Execution authority remains process-local.\n')
(R / 'bins/ptr-bench/README.md').write_text('''# ptr-bench

Reference microbenchmarks and lifecycle probes. Run `cargo run -p ptr-bench -- all`.
These probes do not establish model quality or production readiness.

Dependency setup now uses SemanticDelta.dependencies and timed updates use the
fallible typed-value/staged invalidation API. Historical timings of the old
in-place string map are not equivalent baselines. Measure the exact revision.
See [semantic durability](../../docs/architecture/22-durable-semantic-state.md).
''')
