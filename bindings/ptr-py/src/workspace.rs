//! The bounded, verifier-gated workspace of `ptr-pods` as a pure data structure.
//!
//! The workspace judges every proposal with a verifier fixed at construction (an
//! arithmetic task or a JSON schema) at a fixed required level. A proposal carries its
//! author, its text and a bid; it has no field for a report, a level, a threshold or a
//! verifier, and the host clamps the bid. Python can submit, run a cycle and read the
//! broadcast; it cannot change how proposals are judged.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use ptr_pods::{
    Bid, CycleHost, PayloadRef, Proposal, ProposalInit, RefusalKind, SlotContent, SubmitError,
    SubscribeRequest, SubscriberId, UnrestrictedAuthors, VerifierIdentity, Workspace,
    WorkspaceConfig, WorkspaceConfigInit, WorkspaceSnapshot,
};
use ptr_types::{Digest, Generation, PodId, Probability, TypeId, VerificationLevel};
use ptr_verifier::{
    ArithmeticVerifier, JsonSchemaVerifier, VerificationReport, VerificationStatus, Verifier,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sha2::{Digest as _, Sha256};

use crate::convert::{hex, PtrWorkspaceCapacityError, PtrWorkspaceError};

/// Longest candidate text, in bytes.
const MAX_TEXT_BYTES: usize = 64 * 1024;

enum Task {
    Arithmetic(ArithmeticVerifier),
    Json(JsonSchemaVerifier),
}

/// The host verifier: looks the proposal's text up by its content digest and runs the
/// task's verifier on it. A proposal whose text is unknown is judged `Unknown`.
struct TextVerifier<'a> {
    task: &'a Task,
    texts: &'a BTreeMap<Digest, Arc<String>>,
}

impl Verifier<Proposal> for TextVerifier<'_> {
    fn verify(&self, proposal: &Proposal) -> VerificationReport {
        match self.texts.get(proposal.content()) {
            Some(text) => match self.task {
                Task::Arithmetic(verifier) => verifier.verify(text),
                Task::Json(verifier) => verifier.verify(text),
            },
            None => VerificationReport {
                status: VerificationStatus::Unknown,
                level: VerificationLevel::Unverified,
                score: Probability::new(0.0).expect("zero is a probability"),
                findings: Vec::new(),
            },
        }
    }
}

struct Inner {
    workspace: Workspace,
    texts: BTreeMap<Digest, Arc<String>>,
    subscribers: BTreeMap<u64, SubscriberId>,
}

fn workspace_error(error: impl std::fmt::Display) -> PyErr {
    PtrWorkspaceError::new_err(error.to_string())
}

fn refusal_name(kind: RefusalKind) -> &'static str {
    match kind {
        RefusalKind::Stale => "stale",
        RefusalKind::AuthorRevoked => "author_revoked",
        RefusalKind::NotPassed => "not_passed",
        RefusalKind::HardFinding => "hard_finding",
        RefusalKind::BelowRequiredLevel => "below_required_level",
        RefusalKind::Duplicate => "duplicate",
        RefusalKind::AuthorSlotCap => "author_slot_cap",
        RefusalKind::Outbid => "outbid",
        _ => "other",
    }
}

fn slot_dict<'py>(
    py: Python<'py>,
    slot: &SlotContent,
    texts: &BTreeMap<Digest, Arc<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("slot", slot.slot().get())?;
    out.set_item("author", &slot.author().0)?;
    out.set_item("proposal", hex(slot.proposal().as_bytes()))?;
    out.set_item("content", hex(slot.content()))?;
    out.set_item(
        "candidate",
        texts.get(slot.content()).map(|text| text.as_str()),
    )?;
    out.set_item("bid", slot.bid().get())?;
    out.set_item("admitted_at", slot.admitted_at().get())?;
    out.set_item(
        "authority",
        format!("{:?}", slot.authority()).to_lowercase(),
    )?;
    Ok(out)
}

fn snapshot_dict<'py>(
    py: Python<'py>,
    snapshot: &WorkspaceSnapshot,
    texts: &BTreeMap<Digest, Arc<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("revision", snapshot.revision().get())?;
    out.set_item("digest", hex(snapshot.digest()))?;
    out.set_item("occupied", snapshot.occupied())?;
    out.set_item(
        "authority",
        format!("{:?}", snapshot.authority()).to_lowercase(),
    )?;
    out.set_item("verifier", snapshot.verifier().name())?;
    let slots = PyList::empty(py);
    for slot in snapshot.slots().iter().flatten() {
        slots.append(slot_dict(py, slot, texts)?)?;
    }
    out.set_item("slots", slots)?;
    out.set_item(
        "admitted",
        snapshot
            .admitted()
            .iter()
            .map(|id| hex(id.as_bytes()))
            .collect::<Vec<_>>(),
    )?;
    let refusals = PyList::empty(py);
    for refusal in snapshot.refusals() {
        let item = PyDict::new(py);
        item.set_item("proposal", hex(refusal.proposal().as_bytes()))?;
        item.set_item("author", &refusal.author().0)?;
        item.set_item("reason", refusal_name(refusal.reason().kind()))?;
        let findings = PyList::empty(py);
        for finding in refusal.findings() {
            let entry = PyDict::new(py);
            entry.set_item("code", finding.code())?;
            entry.set_item("severity", if finding.hard() { "hard" } else { "soft" })?;
            findings.append(entry)?;
        }
        item.set_item("findings", findings)?;
        refusals.append(item)?;
    }
    out.set_item("refusals", refusals)?;
    Ok(out)
}

/// A bounded workspace that admits verified proposals and broadcasts snapshots.
///
/// `BoundedWorkspace(kind, spec, *, slots=4, tie_break_seed=0)`: `kind` is
/// `"arithmetic"` (`spec` is the task expression) or `"json_schema"` (`spec` is the
/// schema). Every proposal's text is judged by that verifier at the `sample_verified`
/// level; a slot is taken only by a `pass` with no hard finding at that level, never by a
/// score. At most `slots` proposals are admitted per cycle; the rest get a refusal with
/// the reason and the finding codes. Slot content is provisional: nothing here commits.
/// There is no setter for the verifier, the level or any bound.
#[pyclass(frozen, name = "BoundedWorkspace", module = "ptr_py")]
pub(crate) struct PyWorkspace {
    task: Task,
    identity: VerifierIdentity,
    inner: Mutex<Inner>,
}

impl PyWorkspace {
    fn inner(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[pymethods]
impl PyWorkspace {
    #[new]
    #[pyo3(signature = (kind, spec, *, slots=4, tie_break_seed=0))]
    fn new(kind: &str, spec: &str, slots: usize, tie_break_seed: u64) -> PyResult<Self> {
        let task = match kind {
            "arithmetic" => Task::Arithmetic(ArithmeticVerifier::new(spec)),
            "json_schema" => Task::Json(JsonSchemaVerifier::new(spec)),
            other => {
                return Err(workspace_error(format!(
                    "unknown workspace kind {other:?}; expected arithmetic or json_schema"
                )))
            }
        };
        let mut init = WorkspaceConfigInit::conservative(slots);
        init.tie_break_seed = tie_break_seed;
        let config = WorkspaceConfig::new(init).map_err(workspace_error)?;
        let digest: Digest = Sha256::digest(spec.as_bytes()).into();
        let identity = VerifierIdentity::new(&format!("ptr-py-{kind}"), digest)
            .ok_or_else(|| workspace_error("the verifier identity is invalid"))?;
        Ok(Self {
            task,
            identity,
            inner: Mutex::new(Inner {
                workspace: Workspace::new(config),
                texts: BTreeMap::new(),
                subscribers: BTreeMap::new(),
            }),
        })
    }

    /// Place a proposal in the pending area; returns its id (hex). It occupies no slot
    /// until a cycle judges it. `bid` is only a request: the host clamps it. `author` is an
    /// unauthenticated label: the per-author quota is per label, not per principal, and
    /// every author is accepted at generation 1.
    ///
    /// Raises PtrWorkspaceError for a blank author or empty text, a full pending area, a
    /// full per-author quota or a duplicate proposal.
    #[pyo3(signature = (author, candidate, *, bid=1))]
    fn submit(&self, author: &str, candidate: &str, bid: u32) -> PyResult<String> {
        if candidate.len() > MAX_TEXT_BYTES {
            return Err(PtrWorkspaceCapacityError::new_err(format!(
                "candidate text is {} bytes; the limit is {MAX_TEXT_BYTES}",
                candidate.len()
            )));
        }
        let digest: Digest = Sha256::digest(candidate.as_bytes()).into();
        // The number of held texts needs no cap of its own: a text is held only while its
        // proposal is pending (at most the pending cap of 64) or its content occupies a slot
        // (at most the slot count), and every cycle drops the rest.
        let mut inner = self.inner();
        let proposal = Proposal::new(ProposalInit {
            author: PodId::from(author),
            author_generation: Generation(1),
            content: digest,
            payload: PayloadRef::new(TypeId::from("text"), digest),
            computed_against: inner.workspace.revision(),
            bid: Bid::new(bid),
        })
        .map_err(workspace_error)?;
        let id = inner
            .workspace
            .submit(proposal)
            .map_err(|error| match error {
                SubmitError::PendingFull | SubmitError::AuthorPendingFull => {
                    PtrWorkspaceCapacityError::new_err(error.to_string())
                }
                _ => workspace_error(error),
            })?;
        inner.texts.insert(digest, Arc::new(candidate.to_owned()));
        Ok(hex(id.as_bytes()))
    }

    /// Subscribe a Pod to broadcasts; returns the subscriber id.
    ///
    /// Raises PtrWorkspaceError when the subscriber limit is reached or the Pod is
    /// already subscribed.
    fn subscribe(&self, pod: &str) -> PyResult<u64> {
        let mut inner = self.inner();
        let id = inner
            .workspace
            .subscribe(SubscribeRequest {
                pod: PodId::from(pod),
                generation: Generation(1),
            })
            .map_err(workspace_error)?;
        inner.subscribers.insert(id.get(), id);
        Ok(id.get())
    }

    /// End a subscription: its mailbox is dropped and its place counts against the subscriber
    /// limit no longer, so the same Pod can subscribe again.
    ///
    /// Raises PtrWorkspaceError for an unknown subscriber.
    fn unsubscribe(&self, subscriber: u64) -> PyResult<()> {
        let mut inner = self.inner();
        let id = *inner
            .subscribers
            .get(&subscriber)
            .ok_or_else(|| workspace_error(format!("unknown subscriber {subscriber}")))?;
        inner.workspace.unsubscribe(id).map_err(workspace_error)?;
        inner.subscribers.remove(&subscriber);
        Ok(())
    }

    /// Run one cycle: judge every pending proposal with the fixed verifier, admit up to
    /// `slots`, and broadcast one snapshot. Returns the snapshot as a dict: `revision`,
    /// `digest`, `occupied`, `slots` (with `author`, `candidate`, `bid`), `admitted`,
    /// `refusals` (with `reason` and finding `code`/`severity`; messages stay with the
    /// host) and `delivered`/`overflowed` subscriber ids.
    ///
    /// Raises PtrWorkspaceError when the revision is exhausted.
    fn run_cycle<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (report, texts) = py.detach(|| {
            let mut inner = self.inner();
            // The verifier reads the texts while the cycle borrows the workspace mutably.
            let texts = inner.texts.clone();
            let verifier = TextVerifier {
                task: &self.task,
                texts: &texts,
            };
            let authors = UnrestrictedAuthors;
            let host = CycleHost {
                verifier: &verifier,
                identity: &self.identity,
                authors: &authors,
            };
            let report = inner.workspace.run_cycle(host).map_err(workspace_error)?;
            // Keep only the texts a slot still holds: pending proposals are consumed by a
            // cycle, so everything else can never be read or verified again.
            let held: Vec<Digest> = report
                .snapshot
                .slots()
                .iter()
                .flatten()
                .map(|slot| *slot.content())
                .collect();
            inner.texts.retain(|digest, _| held.contains(digest));
            Ok::<_, PyErr>((report, texts))
        })?;
        let out = snapshot_dict(py, &report.snapshot, &texts)?;
        out.set_item(
            "delivered",
            report
                .delivered
                .iter()
                .map(|id| id.get())
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "overflowed",
            report
                .overflowed
                .iter()
                .map(|id| id.get())
                .collect::<Vec<_>>(),
        )?;
        Ok(out)
    }

    /// Take the oldest snapshot waiting in a subscriber's mailbox, or None.
    ///
    /// Candidate texts are read from the workspace's current text map, which keeps only the
    /// texts of occupied slots: a stale snapshot whose slot content has since been retired
    /// shows `candidate: None` for it (the content digest is still there).
    ///
    /// Raises PtrWorkspaceError for an unknown subscriber.
    fn receive<'py>(
        &self,
        py: Python<'py>,
        subscriber: u64,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        // Plain Rust values are collected under the lock; the Python dicts are built after it
        // is released.
        let (snapshot, texts) = {
            let mut inner = self.inner();
            let id = *inner
                .subscribers
                .get(&subscriber)
                .ok_or_else(|| workspace_error(format!("unknown subscriber {subscriber}")))?;
            let snapshot = inner.workspace.receive(id).map_err(workspace_error)?;
            (snapshot, inner.texts.clone())
        };
        match snapshot {
            Some(snapshot) => Ok(Some(snapshot_dict(py, &snapshot, &texts)?)),
            None => Ok(None),
        }
    }

    /// The current workspace revision (cycles run).
    #[getter]
    fn revision(&self) -> u64 {
        self.inner().workspace.revision().get()
    }

    /// Candidate texts currently held (pending proposals plus occupied slots).
    #[getter]
    fn held_texts(&self) -> usize {
        self.inner().texts.len()
    }

    /// Occupied slots.
    #[getter]
    fn occupied(&self) -> usize {
        self.inner().workspace.occupied()
    }

    /// Proposals waiting for the next cycle.
    #[getter]
    fn pending(&self) -> usize {
        self.inner().workspace.pending_len()
    }

    /// Slot capacity.
    #[getter]
    fn slots(&self) -> usize {
        self.inner().workspace.config().slots()
    }
}
