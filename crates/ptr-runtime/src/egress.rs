//! Egress refusals as committed history, coalesced so a flood cannot become a flood of records.
//!
//! A Pod's link says where it may send. When a send is refused (an unlisted host or port, a
//! topic no filter matches or none given, a lease that spent its frame budget) the refusal
//! happened before anything left the host, so no effect needs reconciling. It still matters to
//! whoever audits the run, and a log line is not evidence. The runtime commits it as a
//! [`LedgerEvent::EgressRefused`] so the refusal replays with the rest of the history.
//!
//! **Bounded (Invariant 8).** A Pod that keeps being refused must not be able to grow the
//! ledger without limit, so refusals are counted, not appended one by one. Within a window of
//! [`EGRESS_WINDOW_MILLIS`] the runtime keeps, per Pod revision, one counter per distinct
//! `(reason, target)` and at most [`MAX_EGRESS_KEYS_PER_POD`] distinct pairs; a further pair is
//! folded into the counter of its reason with the target [`OVERFLOW_TARGET`]. When the window
//! rolls over, or on [`PtrRuntime::flush_egress_refusals`], each counter becomes one event
//! carrying its `count`. So a window commits at most [`MAX_EGRESS_EVENTS_PER_POD_WINDOW`]
//! events per Pod however many refusals there were, and the counts add up to the refusals.
//! At most [`MAX_TRACKED_EGRESS_PODS`] Pods are tracked at once; a new Pod beyond that flushes
//! the table first.
//!
//! A counter is memory until it is flushed, so a crash loses the refusals of the open window.
//! That is the price of the bound, and it is acceptable here because a refusal had no effect:
//! the audit trail is complete for every closed window.
//!
//! The record carries the trace, the Pod revision that tried, the redacted target (a
//! `host:port` or a topic digest, see `ptr_pods::redact_target` and `redact_topic`) and the
//! reason. It carries no payload, no resolved address and no credential (Invariant 13).
use crate::{PtrRuntime, RuntimeError};
use ptr_ledger::LedgerEvent;
use ptr_pods::{NativeProtocolResponse, ProtocolError};
use ptr_types::{CommitIndex, EgressRefusalReason, PodRevisionAddress, Timestamp, TraceId};
use std::collections::BTreeMap;

/// The length of a coalescing window, in milliseconds.
pub const EGRESS_WINDOW_MILLIS: u64 = 60_000;

/// The most distinct `(reason, target)` pairs counted separately per Pod and window.
pub const MAX_EGRESS_KEYS_PER_POD: usize = 8;

/// The target a refusal is counted under once a Pod has used up its distinct pairs.
pub const OVERFLOW_TARGET: &str = "*";

/// The most events one Pod can have committed for one window: the distinct pairs plus one
/// overflow counter per reason.
pub const MAX_EGRESS_EVENTS_PER_POD_WINDOW: usize =
    MAX_EGRESS_KEYS_PER_POD + EgressRefusalReason::ALL.len();

/// The most Pods whose open window is tracked at once.
pub const MAX_TRACKED_EGRESS_PODS: usize = 1024;

/// One refusal to count.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EgressRefusalRecord {
    /// The trace of the request whose send was refused.
    pub trace_id: TraceId,
    /// The Pod revision that tried to send.
    pub source: PodRevisionAddress,
    /// The redacted `host:port` or topic digest. Never a payload.
    pub target: String,
    /// Which rule refused it.
    pub reason: EgressRefusalReason,
    /// When it was refused, in milliseconds since the Unix epoch; decides the window.
    pub observed_at: Timestamp,
}

/// What recording a refusal did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EgressRefusalOutcome {
    /// Counted in the open window; nothing was appended yet.
    Counted,
    /// A window closed (or the table was full): these events were appended.
    Committed(Vec<CommitIndex>),
}

#[derive(Clone, Debug)]
struct Counter {
    trace_id: TraceId,
    count: u32,
}

#[derive(Clone, Debug)]
struct PodWindow {
    window: u64,
    counters: BTreeMap<(u8, String), Counter>,
}

/// The open windows of every Pod that has been refused recently.
#[derive(Clone, Debug, Default)]
pub(crate) struct EgressRefusalWindows {
    pods: BTreeMap<PodRevisionAddress, PodWindow>,
}

/// A closed counter on its way to the ledger, with the window it was counted in so it can be
/// put back where it came from if the append fails.
type Closed = (u64, LedgerEvent);

fn event_of(
    window: u64,
    source: &PodRevisionAddress,
    key: &(u8, String),
    counter: &Counter,
) -> Option<Closed> {
    Some((
        window,
        LedgerEvent::EgressRefused {
            trace_id: counter.trace_id.clone(),
            source: source.clone(),
            target: key.1.clone(),
            count: counter.count,
            reason: EgressRefusalReason::from_code(key.0)?,
        },
    ))
}

impl EgressRefusalWindows {
    fn take_pod(&mut self, source: &PodRevisionAddress) -> Vec<Closed> {
        self.pods
            .remove(source)
            .map(|pod| {
                pod.counters
                    .iter()
                    .filter_map(|(key, counter)| event_of(pod.window, source, key, counter))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn take_all(&mut self) -> Vec<Closed> {
        std::mem::take(&mut self.pods)
            .into_iter()
            .flat_map(|(source, pod)| {
                pod.counters
                    .iter()
                    .filter_map(|(key, counter)| event_of(pod.window, &source, key, counter))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Put a closed counter back into the window it came from, without the capacity eviction
    /// and window rollover [`Self::absorb`] does: it is returned to the table it was taken from,
    /// so the table cannot be larger than it was, and nothing it holds can be flushed away
    /// unrecorded. Counts add; the overflow rule still applies to the distinct pairs.
    fn restore(&mut self, window: u64, event: LedgerEvent) {
        let LedgerEvent::EgressRefused {
            trace_id,
            source,
            target,
            count,
            reason,
        } = event
        else {
            return;
        };
        let pod = self.pods.entry(source).or_insert_with(|| PodWindow {
            window,
            counters: BTreeMap::new(),
        });
        let counter = pod
            .counters
            .entry((reason.code(), target))
            .or_insert_with(|| Counter { trace_id, count: 0 });
        counter.count = counter.count.saturating_add(count);
    }

    /// Count `count` refusals; return the events of whatever window this closed.
    fn absorb(
        &mut self,
        source: &PodRevisionAddress,
        trace_id: &TraceId,
        reason: EgressRefusalReason,
        target: &str,
        count: u32,
        window: u64,
    ) -> Vec<Closed> {
        let mut closed = Vec::new();
        if !self.pods.contains_key(source) && self.pods.len() >= MAX_TRACKED_EGRESS_PODS {
            closed.extend(self.take_all());
        }
        if self
            .pods
            .get(source)
            .is_some_and(|pod| pod.window != window)
        {
            closed.extend(self.take_pod(source));
        }
        let pod = self
            .pods
            .entry(source.clone())
            .or_insert_with(|| PodWindow {
                window,
                counters: BTreeMap::new(),
            });
        let mut key = (reason.code(), target.to_owned());
        if !pod.counters.contains_key(&key) {
            let distinct = pod
                .counters
                .keys()
                .filter(|(_, target)| target != OVERFLOW_TARGET)
                .count();
            if target != OVERFLOW_TARGET && distinct >= MAX_EGRESS_KEYS_PER_POD {
                key.1 = OVERFLOW_TARGET.to_owned();
            }
        }
        let counter = pod.counters.entry(key).or_insert_with(|| Counter {
            trace_id: trace_id.clone(),
            count: 0,
        });
        counter.count = counter.count.saturating_add(count);
        closed
    }
}

impl PtrRuntime {
    /// Count an egress refusal. It is appended as a [`LedgerEvent::EgressRefused`] when its
    /// window closes (the next refusal of the same Pod in a later window, or
    /// [`PtrRuntime::flush_egress_refusals`]), together with the other refusals the window
    /// counted; see the module documentation for the bound.
    ///
    /// The record is validated with the same rules replay applies before it is counted, so a
    /// record this refuses is one the ledger would refuse to read back.
    ///
    /// # Errors
    /// [`RuntimeError::Ledger`] when the record is malformed (empty or over-long trace, invalid
    /// source address, a target that is empty, over its bound or carries whitespace), or the
    /// runtime's own errors when an append of a closed window fails because the ledger is
    /// uncertain or the append is refused; the events not yet appended stay counted.
    pub fn record_egress_refusal(
        &mut self,
        record: EgressRefusalRecord,
    ) -> Result<EgressRefusalOutcome, RuntimeError> {
        let probe = LedgerEvent::EgressRefused {
            trace_id: record.trace_id.clone(),
            source: record.source.clone(),
            target: record.target.clone(),
            count: 1,
            reason: record.reason,
        };
        // The same shape check the codec applies, run before anything is counted.
        ptr_ledger::check_encodable(&probe)
            .map_err(|error| RuntimeError::Ledger(error.to_string()))?;
        let closed = self.egress_refusals.absorb(
            &record.source,
            &record.trace_id,
            record.reason,
            &record.target,
            1,
            record.observed_at.0 / EGRESS_WINDOW_MILLIS,
        );
        if closed.is_empty() {
            return Ok(EgressRefusalOutcome::Counted);
        }
        self.commit_egress_events(closed)
            .map(EgressRefusalOutcome::Committed)
    }

    /// Append the counters of every open window now, for a shutdown, a test or an audit that
    /// must not wait for the window to close.
    pub fn flush_egress_refusals(&mut self) -> Result<Vec<CommitIndex>, RuntimeError> {
        let events = self.egress_refusals.take_all();
        self.commit_egress_events(events)
    }

    /// The runtime-side dispatch hook for a native protocol call made on behalf of the Pod
    /// revision `source`: a call refused by its link's egress policy or rate limit
    /// ([`ProtocolError::EgressRefused`]) is counted for the ledger, any other outcome is not
    /// an egress refusal and is left alone (`Ok(None)`). Whoever runs a `ptr_pods`
    /// native executor for a Pod passes its result here.
    pub fn record_native_protocol_outcome(
        &mut self,
        source: PodRevisionAddress,
        trace_id: TraceId,
        observed_at: Timestamp,
        outcome: &Result<NativeProtocolResponse, ProtocolError>,
    ) -> Result<Option<EgressRefusalOutcome>, RuntimeError> {
        match outcome {
            Err(ProtocolError::EgressRefused { target, reason }) => self
                .record_egress_refusal(EgressRefusalRecord {
                    trace_id,
                    source,
                    target: target.clone(),
                    reason: *reason,
                    observed_at,
                })
                .map(Some),
            _ => Ok(None),
        }
    }

    /// Append events in order; on the first failure the events not appended are counted again
    /// rather than lost.
    fn commit_egress_events(
        &mut self,
        events: Vec<Closed>,
    ) -> Result<Vec<CommitIndex>, RuntimeError> {
        let mut committed = Vec::with_capacity(events.len());
        let mut pending = events.into_iter();
        while let Some((window, event)) = pending.next() {
            match self.commit_settlement(event.clone()) {
                Ok(index) => committed.push(index),
                Err(error) => {
                    for (window, event) in std::iter::once((window, event)).chain(pending) {
                        self.egress_refusals.restore(window, event);
                    }
                    return Err(error);
                }
            }
        }
        Ok(committed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_config::PtrConfig;
    use ptr_types::{Generation, NamespaceId, PodAddress, PodId, ProjectId};

    fn pod(n: usize) -> PodRevisionAddress {
        PodRevisionAddress {
            address: PodAddress::new(
                ProjectId::from("p"),
                NamespaceId::from("ns"),
                PodId::from(format!("pod{n}").as_str()),
            )
            .unwrap(),
            semantic_revision: [1; 32],
            generation: Generation(1),
        }
    }

    fn total(windows: &EgressRefusalWindows) -> u64 {
        windows
            .pods
            .values()
            .flat_map(|pod| pod.counters.values())
            .map(|counter| u64::from(counter.count))
            .sum()
    }

    fn record(n: usize, at: u64) -> EgressRefusalRecord {
        EgressRefusalRecord {
            trace_id: TraceId::from("t"),
            source: pod(n),
            target: "h:1".into(),
            reason: EgressRefusalReason::HostNotAllowed,
            observed_at: Timestamp(at),
        }
    }

    #[test]
    fn a_failed_flush_at_the_pod_capacity_loses_no_counter() {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        for n in 0..MAX_TRACKED_EGRESS_PODS {
            runtime.record_egress_refusal(record(n, 1)).unwrap();
        }
        assert_eq!(runtime.egress_refusals.pods.len(), MAX_TRACKED_EGRESS_PODS);
        // The append fails: the ledger is uncertain about a commit in flight.
        runtime.mark_commit_uncertain_for_test();
        // The 1025th Pod forces a flush of the full table, which fails.
        let refused = runtime.record_egress_refusal(record(MAX_TRACKED_EGRESS_PODS, 2));
        assert!(refused.is_err());
        assert_eq!(
            total(&runtime.egress_refusals),
            MAX_TRACKED_EGRESS_PODS as u64 + 1,
            "every counter of the table that failed to flush is still counted, and so is the \
             refusal that triggered the flush"
        );
        assert_eq!(
            runtime.egress_refusals.pods.len(),
            MAX_TRACKED_EGRESS_PODS + 1,
            "the table is larger than its capacity only while the ledger refuses appends"
        );
    }

    #[test]
    fn restore_returns_a_closed_counter_to_its_window_without_evicting_anything() {
        let mut windows = EgressRefusalWindows::default();
        let source = pod(0);
        let trace = TraceId::from("t");
        let reason = EgressRefusalReason::RateLimited;
        windows.absorb(&source, &trace, reason, "h:1", 2, 7);
        let taken = windows.take_all();
        assert!(windows.pods.is_empty());
        for (window, event) in taken {
            windows.restore(window, event);
        }
        assert_eq!(total(&windows), 2);
        assert_eq!(windows.pods[&source].window, 7);
    }
}
