//! The read-back of a finished request: `GET /v1/vcw/traces/{id}`.
//!
//! The service keeps the trace of the last [`MAX_RETAINED_TRACES`] requests in memory and renders
//! it, together with a summary of the ledger events the request committed, as JSON. The events are
//! read from the runtime's ledger once, when the request finishes (the runtime lock is held then
//! anyway), and kept as kind, index, size and digest; **a query never takes the runtime lock**, so
//! it cannot wait behind a long request and needs no admission slot. The trace carries digests, finding codes, Pod ids, the routing
//! support and propensity, and the budget; it carries no candidate payload and no finding message.
//! Nothing here is authoritative: the ledger is, and the trace digest is bound into the committed
//! admission of an accepted answer.

use std::collections::VecDeque;

use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_runtime::vcw::{ClassSummary, StepOrigin, TraceStep, VcwTrace};
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};

/// Most traces the service retains; the oldest is dropped first.
pub const MAX_RETAINED_TRACES: usize = 64;
/// Most ledger events one report lists.
pub const MAX_REPORTED_EVENTS: usize = 64;

/// One retained request.
pub(super) struct Retained {
    id: String,
    outcome: &'static str,
    backend: String,
    trace: Value,
    /// `None` while the request runs: the report then has outcome `"running"` and `ledger: null`.
    /// The trace itself is `null` until the loop has finished.
    committed: Option<LedgerSummary>,
}

/// The ledger events one request committed: the first index, how many, and a bounded summary.
pub(super) struct LedgerSummary {
    first: u64,
    count: u64,
    events: Vec<Value>,
}

impl LedgerSummary {
    /// Summarizes `events`, the slice of the ledger from the request's first event on. This is
    /// the only work done while the runtime lock is held: at most [`MAX_REPORTED_EVENTS`]
    /// encodings and digests, and no rendering of an event.
    pub(super) fn of(events: &[CommittedEvent]) -> Self {
        Self {
            first: events.first().map_or(0, |event| event.index.0),
            count: events.len() as u64,
            events: summarize(events),
        }
    }
}

/// A bounded store of finished requests.
#[derive(Default)]
pub(super) struct TraceStore {
    items: VecDeque<Retained>,
}

impl TraceStore {
    /// Registers the request `id` as running before its loop starts, with a `null` trace, so a
    /// query during the loop answers `running` instead of 404.
    pub(super) fn start(&mut self, id: &str, backend: &str) {
        self.push(Retained {
            id: id.to_owned(),
            outcome: "running",
            backend: backend.to_owned(),
            trace: Value::Null,
            committed: None,
        });
    }

    /// Stores the trace of the finished loop of `id`: into its running entry if it is still
    /// retained, else as a new entry.
    pub(super) fn insert(&mut self, id: &str, backend: &str, trace: &VcwTrace) {
        if let Some(item) = self
            .items
            .iter_mut()
            .rev()
            .find(|item| item.id == id && item.trace.is_null())
        {
            item.trace = trace_json(trace);
            return;
        }
        self.push(Retained {
            id: id.to_owned(),
            outcome: "running",
            backend: backend.to_owned(),
            trace: trace_json(trace),
            committed: None,
        });
    }

    fn push(&mut self, item: Retained) {
        if self.items.len() >= MAX_RETAINED_TRACES {
            self.items.pop_front();
        }
        self.items.push_back(item);
    }

    /// Records how the request `id` ended and the ledger range it committed, if it is still
    /// retained.
    pub(super) fn finish(&mut self, id: &str, outcome: &'static str, ledger: LedgerSummary) {
        if let Some(item) = self.items.iter_mut().rev().find(|item| item.id == id) {
            item.outcome = outcome;
            item.committed = Some(ledger);
        }
    }

    /// The report of `id`, or `None` if it is not retained. Never touches the runtime.
    pub(super) fn report(&self, id: &str) -> Option<Value> {
        let item = self.items.iter().rev().find(|item| item.id == id)?;
        let ledger = item.committed.as_ref().map(|summary| {
            json!({
                "first_index": summary.first,
                "count": summary.count,
                "events": summary.events,
            })
        });
        Some(json!({
            "id": item.id,
            "outcome": item.outcome,
            "backend": item.backend,
            "trace": item.trace,
            "ledger": ledger,
        }))
    }
}

/// Kind, index, encoded size and a digest of each event; never the event's content, which can
/// carry raw payload bytes. Size and digest come from the event's canonical encoding
/// ([`ptr_ledger::encoded_event`], the bytes a log record frames), not from a rendering of it.
fn summarize(events: &[CommittedEvent]) -> Vec<Value> {
    events
        .iter()
        .take(MAX_REPORTED_EVENTS)
        .map(|event| {
            let encoded = ptr_ledger::encoded_event(&event.event);
            let digest = Sha256::digest(&encoded);
            json!({
                "index": event.index.0,
                "kind": event_kind(&event.event),
                "bytes": encoded.len(),
                "digest": hex(&digest[..8]),
            })
        })
        .collect()
}

/// The variant name of an event. Only the name is rendered: the writer refuses the first
/// character after it, so a derived `Debug` stops there and the fields are never formatted.
fn event_kind(event: &LedgerEvent) -> String {
    #[derive(Default)]
    struct Name(String);
    impl std::fmt::Write for Name {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            for c in text.chars() {
                if !(c.is_ascii_alphanumeric() || c == '_') {
                    return Err(std::fmt::Error);
                }
                self.0.push(c);
            }
            Ok(())
        }
    }
    let mut name = Name::default();
    // An error here is the early stop, not a failure.
    let _ = std::fmt::write(&mut name, format_args!("{event:?}"));
    name.0
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn origin(origin: StepOrigin) -> &'static str {
    match origin {
        StepOrigin::Initial => "initial",
        StepOrigin::Repaired => "repaired",
        StepOrigin::Regenerated => "regenerated",
        StepOrigin::Escalated => "escalated",
    }
}

fn step_json(step: &TraceStep) -> Value {
    let routes: Vec<Value> = step
        .routes
        .iter()
        .map(|route| {
            json!({
                "class": format!("{:?}", route.class()),
                "pod": route.pod().0,
                "generation": route.pod_generation().0,
                "propensity": route.propensity().get(),
                "support": route
                    .support()
                    .iter()
                    .map(|(pod, width)| json!({ "pod": pod.0, "width": width }))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let candidate = step.candidate.as_ref().map(|candidate| {
        json!({
            "digest": hex(&candidate.payload_digest),
            "status": format!("{:?}", candidate.report.status),
            "level": format!("{:?}", candidate.report.level),
            "hard_codes": candidate.report.hard_codes,
            "hard_total": candidate.report.hard_total,
            "class": match &candidate.class {
                ClassSummary::Accept => json!("accept"),
                ClassSummary::Repair(codes) => json!({
                    "repair": codes.iter().map(|code| code.as_str()).collect::<Vec<_>>(),
                }),
                ClassSummary::Fallback(reason) => json!({ "fallback": format!("{reason:?}") }),
            },
            "workspace": format!("{:?}", candidate.workspace),
        })
    });
    json!({
        "round": step.round,
        "origin": origin(step.origin),
        "authors": step
            .authors
            .iter()
            .map(|author| json!({ "pod": author.pod.0, "generation": author.generation.0 }))
            .collect::<Vec<_>>(),
        "routes": routes,
        "unroutable": step.unroutable.iter().map(|code| code.as_str()).collect::<Vec<_>>(),
        "candidate": candidate,
        "decision": format!("{:?}", step.decision),
        "budget_after": {
            "rounds": step.budget_after.rounds,
            "verifier_calls": step.budget_after.verifier_calls,
            "pod_calls": step.budget_after.pod_calls,
        },
    })
}

fn trace_json(trace: &VcwTrace) -> Value {
    let spent = trace.spent();
    let budget = trace.budget();
    json!({
        "digest": hex(trace.digest()),
        "outcome": format!("{:?}", trace.outcome()),
        "budget": {
            "verifier_calls": budget.verifier_calls.get(),
            "rounds": budget.rounds.get(),
        },
        "spent": {
            "rounds": spent.rounds,
            "verifier_calls": spent.verifier_calls,
            "pod_calls": spent.pod_calls,
        },
        "steps": trace.steps().iter().map(step_json).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::{CapsuleId, CommitIndex, Generation, ProjectId};

    fn raw(text: &str) -> CommittedEvent {
        CommittedEvent {
            index: CommitIndex(7),
            event: LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from(text),
                generation: Generation(1),
            },
        }
    }

    #[test]
    fn event_kind_is_the_variant_name_without_rendering_the_fields() {
        assert_eq!(event_kind(&raw("x").event), "CapsuleCommitted");
        // A payload that would dominate a full rendering does not change the kind.
        assert_eq!(
            event_kind(&raw(&"y".repeat(1 << 20)).event),
            "CapsuleCommitted"
        );
    }

    #[test]
    fn size_and_digest_come_from_the_canonical_encoding_not_the_debug_text() {
        let event = raw("secret-payload");
        let summary = summarize(std::slice::from_ref(&event));
        let encoded = ptr_ledger::encoded_event(&event.event);
        assert_eq!(summary[0]["bytes"], encoded.len());
        assert_ne!(summary[0]["bytes"], format!("{:?}", event.event).len());
        let digest = Sha256::digest(&encoded);
        assert_eq!(summary[0]["digest"], hex(&digest[..8]));
        assert_eq!(summary[0]["index"], 7);
        assert!(!summary[0].to_string().contains("secret-payload"));
    }

    #[test]
    fn at_most_the_reported_events_are_summarized_but_all_are_counted() {
        let events: Vec<CommittedEvent> = (0..MAX_REPORTED_EVENTS + 5)
            .map(|i| raw(&i.to_string()))
            .collect();
        let summary = LedgerSummary::of(&events);
        assert_eq!(summary.count, (MAX_REPORTED_EVENTS + 5) as u64);
        assert_eq!(summary.events.len(), MAX_REPORTED_EVENTS);
    }

    #[test]
    fn a_started_request_reads_back_as_running_before_its_loop_has_a_trace() {
        let mut store = TraceStore::default();
        store.start("live", "simulated");
        let report = store.report("live").unwrap();
        assert_eq!(report["outcome"], "running");
        assert!(report["trace"].is_null());
        assert!(report["ledger"].is_null());
        assert_eq!(store.items.len(), 1);
        for i in 0..MAX_RETAINED_TRACES + 3 {
            store.start(&format!("r{i}"), "simulated");
        }
        assert_eq!(store.items.len(), MAX_RETAINED_TRACES);
        assert!(store.report("live").is_none());
    }

    #[test]
    fn a_request_that_has_not_finished_reports_running_and_a_null_ledger() {
        let mut store = TraceStore::default();
        store.items.push_back(Retained {
            id: "live".to_owned(),
            outcome: "running",
            backend: "simulated".to_owned(),
            trace: json!({ "digest": "00" }),
            committed: None,
        });
        let report = store.report("live").unwrap();
        assert_eq!(report["outcome"], "running");
        assert!(report["ledger"].is_null());
        assert_eq!(report["trace"]["digest"], "00");
        // Finishing it fills the ledger and the outcome.
        store.finish("live", "verified", LedgerSummary::of(&[raw("a")]));
        let report = store.report("live").unwrap();
        assert_eq!(report["outcome"], "verified");
        assert_eq!(report["ledger"]["count"], 1);
        assert_eq!(report["ledger"]["first_index"], 7);
    }
}
