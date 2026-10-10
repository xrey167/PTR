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

use ptr_ledger::CommittedEvent;
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
    /// First ledger index the request committed, and how many events.
    committed: Option<(u64, u64, Vec<Value>)>,
}

/// A bounded store of finished requests.
#[derive(Default)]
pub(super) struct TraceStore {
    items: VecDeque<Retained>,
}

impl TraceStore {
    pub(super) fn insert(&mut self, id: &str, backend: &str, trace: &VcwTrace) {
        if self.items.len() >= MAX_RETAINED_TRACES {
            self.items.pop_front();
        }
        self.items.push_back(Retained {
            id: id.to_owned(),
            outcome: "running",
            backend: backend.to_owned(),
            trace: trace_json(trace),
            committed: None,
        });
    }

    /// Records how the request `id` ended and the ledger range it committed (`events` is the
    /// slice of the ledger from the request's first event on), if it is still retained.
    pub(super) fn finish(&mut self, id: &str, outcome: &'static str, events: &[CommittedEvent]) {
        if let Some(item) = self.items.iter_mut().rev().find(|item| item.id == id) {
            item.outcome = outcome;
            let first = events.first().map_or(0, |event| event.index.0);
            item.committed = Some((first, events.len() as u64, summarize(events)));
        }
    }

    /// The report of `id`, or `None` if it is not retained. Never touches the runtime.
    pub(super) fn report(&self, id: &str) -> Option<Value> {
        let item = self.items.iter().rev().find(|item| item.id == id)?;
        let ledger = item.committed.as_ref().map(|(first, count, events)| {
            json!({ "first_index": first, "count": count, "events": events })
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

/// Kind, index, rendered size and a digest of each event; never the event's content, which can
/// carry raw payload bytes.
fn summarize(events: &[CommittedEvent]) -> Vec<Value> {
    events
        .iter()
        .take(MAX_REPORTED_EVENTS)
        .map(|event| {
            let text = format!("{:?}", event.event);
            let digest = Sha256::digest(text.as_bytes());
            json!({
                "index": event.index.0,
                "kind": event_kind(&text),
                "bytes": text.len(),
                "digest": hex(&digest[..8]),
            })
        })
        .collect()
}

/// The variant name of a `Debug` rendering of an enum value.
fn event_kind(debug: &str) -> &str {
    let end = debug
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(debug.len());
    &debug[..end]
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

    #[test]
    fn event_kind_is_the_variant_name() {
        assert_eq!(
            event_kind("RawEvidenceCommitted { a: 1 }"),
            "RawEvidenceCommitted"
        );
        assert_eq!(event_kind("Unit"), "Unit");
        assert_eq!(event_kind(""), "");
    }
}
