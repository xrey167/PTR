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
//! hashed `host:` or `topic:` digest, see `ptr_pods::redact_endpoint` and `redact_topic`) and the
//! reason. It carries no payload, no resolved address and no credential (Invariant 13).
use crate::{PtrRuntime, RuntimeError};
use ptr_ledger::{EgressRefusalSource, LedgerEvent};
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
    /// The redacted `host:` or `topic:` digest. Never a payload.
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
    /// Counting stopped at `u32::MAX` because more refusals came in than a counter can hold
    /// (only reachable while the ledger refuses appends, since a counter about to overflow
    /// closes its window first). Travels to the event so the audit says "at least".
    saturated: bool,
}

impl Counter {
    fn new(trace_id: TraceId) -> Self {
        Self {
            trace_id,
            count: 0,
            saturated: false,
        }
    }

    /// Add `count` refusals; going past `u32::MAX` pins the count there and says so.
    fn merge(&mut self, count: u32, saturated: bool) {
        match self.count.checked_add(count) {
            Some(total) => self.count = total,
            None => {
                self.count = u32::MAX;
                self.saturated = true;
            }
        }
        self.saturated |= saturated;
    }

    fn would_overflow(&self, count: u32) -> bool {
        self.count.checked_add(count).is_none()
    }
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
    /// Refusals of Pods that did not fit the table while an append was failing, folded per
    /// reason (so at most one counter per reason, however many Pods). Committed with the next
    /// flush as events whose source is [`EgressRefusalSource::Overflow`] and whose target is
    /// [`OVERFLOW_TARGET`].
    overflow: BTreeMap<u8, Counter>,
    overflow_window: u64,
}

/// A closed counter on its way to the ledger, with the window it was counted in so it can be
/// put back where it came from if the append fails.
type Closed = (u64, LedgerEvent);

fn event_of(
    window: u64,
    source: EgressRefusalSource,
    key: &(u8, String),
    counter: &Counter,
) -> Option<Closed> {
    Some((
        window,
        LedgerEvent::EgressRefused {
            trace_id: counter.trace_id.clone(),
            source,
            target: key.1.clone(),
            count: counter.count,
            saturated: counter.saturated,
            reason: EgressRefusalReason::from_code(key.0)?,
        },
    ))
}

/// The key a refusal is counted under: its own `(reason, target)` while the Pod has room for
/// another exact target, the reason's overflow target once it has used up
/// [`MAX_EGRESS_KEYS_PER_POD`]. Used by `absorb` and `restore` alike, so no path grows a Pod's
/// counters past the bound plus one overflow key per reason.
fn bounded_key(
    counters: &BTreeMap<(u8, String), Counter>,
    reason: u8,
    target: &str,
) -> (u8, String) {
    let key = (reason, target.to_owned());
    if counters.contains_key(&key) || target == OVERFLOW_TARGET {
        return key;
    }
    let distinct = counters
        .keys()
        .filter(|(_, target)| target != OVERFLOW_TARGET)
        .count();
    if distinct >= MAX_EGRESS_KEYS_PER_POD {
        (reason, OVERFLOW_TARGET.to_owned())
    } else {
        key
    }
}

impl EgressRefusalWindows {
    fn take_pod(&mut self, source: &PodRevisionAddress) -> Vec<Closed> {
        self.pods
            .remove(source)
            .map(|pod| {
                pod.counters
                    .iter()
                    .filter_map(|(key, counter)| {
                        event_of(
                            pod.window,
                            EgressRefusalSource::Pod(source.clone()),
                            key,
                            counter,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn take_all(&mut self) -> Vec<Closed> {
        let mut closed: Vec<Closed> = std::mem::take(&mut self.pods)
            .into_iter()
            .flat_map(|(source, pod)| {
                pod.counters
                    .iter()
                    .filter_map(|(key, counter)| {
                        event_of(
                            pod.window,
                            EgressRefusalSource::Pod(source.clone()),
                            key,
                            counter,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let overflow_window = self.overflow_window;
        closed.extend(
            std::mem::take(&mut self.overflow)
                .iter()
                .filter_map(|(code, counter)| {
                    event_of(
                        overflow_window,
                        EgressRefusalSource::Overflow,
                        &(*code, OVERFLOW_TARGET.to_owned()),
                        counter,
                    )
                }),
        );
        closed
    }

    /// How many refusals are counted, in the table and in the overflow.
    #[cfg(test)]
    fn total(&self) -> u64 {
        self.pods
            .values()
            .flat_map(|pod| pod.counters.values())
            .chain(self.overflow.values())
            .map(|counter| u64::from(counter.count))
            .sum()
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
            saturated,
            reason,
        } = event
        else {
            return;
        };
        // The table never grows past its capacity here: a Pod that is not in it already, when
        // it is full (or an event that already stands for the overflow), folds into the
        // per-reason overflow counters, of which there are at most one per reason.
        let source = match source {
            EgressRefusalSource::Pod(source)
                if self.pods.contains_key(&source) || self.pods.len() < MAX_TRACKED_EGRESS_PODS =>
            {
                source
            }
            EgressRefusalSource::Pod(_) | EgressRefusalSource::Overflow => {
                self.overflow_window = window;
                self.overflow
                    .entry(reason.code())
                    .or_insert_with(|| Counter::new(trace_id))
                    .merge(count, saturated);
                return;
            }
        };
        let pod = self.pods.entry(source).or_insert_with(|| PodWindow {
            window,
            counters: BTreeMap::new(),
        });
        // The same fold as `absorb`: a Pod never holds more than MAX_EGRESS_KEYS_PER_POD exact
        // targets, however many failed rollovers put counters back.
        let key = bounded_key(&pod.counters, reason.code(), &target);
        pod.counters
            .entry(key)
            .or_insert_with(|| Counter::new(trace_id))
            .merge(count, saturated);
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
        // A counter that this refusal would push past u32::MAX closes its window first, so the
        // events carry exact counts for as long as the ledger takes appends.
        let would_overflow = self.pods.get(source).is_some_and(|pod| {
            let key = bounded_key(&pod.counters, reason.code(), target);
            pod.counters
                .get(&key)
                .is_some_and(|counter| counter.would_overflow(count))
        });
        if would_overflow {
            closed.extend(self.take_pod(source));
        }
        let pod = self
            .pods
            .entry(source.clone())
            .or_insert_with(|| PodWindow {
                window,
                counters: BTreeMap::new(),
            });
        let key = bounded_key(&pod.counters, reason.code(), target);
        pod.counters
            .entry(key)
            .or_insert_with(|| Counter::new(trace_id.clone()))
            .merge(count, false);
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
            source: EgressRefusalSource::Pod(record.source.clone()),
            target: record.target.clone(),
            count: 1,
            saturated: false,
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

    /// Append events in order; on the first failure (the ledger refuses, or the runtime is
    /// fenced) the events not appended are counted again rather than lost: they go back into
    /// the bounded table, which is the pending list, and are committed by a later flush once the
    /// fence is lifted. The append respects the execution fence, unlike a settlement.
    fn commit_egress_events(
        &mut self,
        events: Vec<Closed>,
    ) -> Result<Vec<CommitIndex>, RuntimeError> {
        let mut committed = Vec::with_capacity(events.len());
        let mut pending = events.into_iter();
        while let Some((window, event)) = pending.next() {
            match self.commit_gated(event.clone()) {
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
        windows.total()
    }

    fn record(n: usize, at: u64) -> EgressRefusalRecord {
        EgressRefusalRecord {
            trace_id: TraceId::from("t"),
            source: pod(n),
            target: "host:0123456789abcdef:1:1".into(),
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
        assert!(
            runtime.egress_refusals.pods.len() <= MAX_TRACKED_EGRESS_PODS,
            "the table never exceeds its capacity, even while the ledger refuses appends"
        );
    }

    #[test]
    fn ten_thousand_pods_refused_while_every_append_fails_stay_within_the_bound_and_are_all_counted(
    ) {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime.mark_commit_uncertain_for_test();
        for n in 0..10_000 {
            let _ = runtime.record_egress_refusal(record(n, 1));
            assert!(runtime.egress_refusals.pods.len() <= MAX_TRACKED_EGRESS_PODS);
            assert!(runtime.egress_refusals.overflow.len() <= EgressRefusalReason::ALL.len());
        }
        assert_eq!(total(&runtime.egress_refusals), 10_000);
    }

    #[test]
    fn overflow_counters_come_back_as_events_with_their_counts_once_the_ledger_works() {
        let mut windows = EgressRefusalWindows::default();
        for n in 0..MAX_TRACKED_EGRESS_PODS + 5 {
            let taken = windows.absorb(
                &pod(n),
                &TraceId::from("t"),
                EgressRefusalReason::HostNotAllowed,
                "host:0123456789abcdef:1:1",
                1,
                3,
            );
            // Nothing closes until the table is full; the 1025th Pod flushes it.
            for (window, event) in taken {
                windows.restore(window, event);
            }
        }
        assert!(windows.pods.len() <= MAX_TRACKED_EGRESS_PODS);
        assert_eq!(windows.total(), (MAX_TRACKED_EGRESS_PODS + 5) as u64);
        let events = windows.take_all();
        let overflow: Vec<_> = events
            .iter()
            .filter(|(_, event)| {
                matches!(event, LedgerEvent::EgressRefused { source: EgressRefusalSource::Overflow, target, .. }
                    if target == OVERFLOW_TARGET)
            })
            .collect();
        assert!(!overflow.is_empty());
        for (_, event) in &events {
            ptr_ledger::check_encodable(event).unwrap();
        }
    }

    #[test]
    fn failed_rollovers_of_one_pod_with_a_new_target_each_keep_its_keys_bounded() {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime.mark_commit_uncertain_for_test();
        let rollovers = 5 * MAX_EGRESS_KEYS_PER_POD;
        for n in 0..rollovers {
            let mut refused = record(0, n as u64 * EGRESS_WINDOW_MILLIS);
            refused.target = format!("host:{n:016x}:1:1");
            let _ = runtime.record_egress_refusal(refused);
            let keys = runtime.egress_refusals.pods[&pod(0)].counters.len();
            assert!(
                keys <= MAX_EGRESS_KEYS_PER_POD + EgressRefusalReason::ALL.len(),
                "{keys} keys after {n} failed rollovers"
            );
        }
        assert_eq!(total(&runtime.egress_refusals), rollovers as u64);
    }

    fn unsettled_effect() -> crate::execution::UnsettledEffect {
        crate::execution::UnsettledEffect {
            attempt: CommitIndex(1),
            key: None,
            target: "t".into(),
            operation: "op".into(),
            effect: ptr_types::Effect::External,
            identity: crate::execution::ActionIdentity {
                project: ProjectId::from("p"),
                principal: "principal".into(),
                revision: ptr_types::Revision(1),
                generation: Generation(1),
                action_digest: [0; 32],
            },
        }
    }

    fn egress_events(runtime: &PtrRuntime) -> Vec<LedgerEvent> {
        runtime
            .committed_events()
            .iter()
            .filter(|committed| matches!(committed.event, LedgerEvent::EgressRefused { .. }))
            .map(|committed| committed.event.clone())
            .collect()
    }

    #[test]
    fn a_fenced_runtime_commits_no_refusal_and_loses_none_until_the_fence_is_lifted() {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        for n in 0..3 {
            runtime.record_egress_refusal(record(n, 1)).unwrap();
        }
        // An effect attempt awaits its settlement: the execution fence is up.
        runtime.execution.record_attempt(unsettled_effect());
        assert!(runtime.execution.is_fenced());
        let refused = runtime.flush_egress_refusals();
        assert!(
            matches!(refused, Err(RuntimeError::ExecutionFenced)),
            "{refused:?}"
        );
        assert!(egress_events(&runtime).is_empty(), "nothing is committed");
        assert_eq!(total(&runtime.egress_refusals), 3, "and nothing is lost");
        // A refusal recorded meanwhile is counted too.
        let _ = runtime.record_egress_refusal(record(7, 2));
        assert_eq!(total(&runtime.egress_refusals), 4);
        // The fence is lifted by settling the attempt.
        runtime
            .execution
            .settle(CommitIndex(1), crate::execution::Settlement::NotApplied);
        assert!(!runtime.execution.is_fenced());
        let committed = runtime.flush_egress_refusals().unwrap();
        assert_eq!(committed.len(), 4);
        assert_eq!(egress_events(&runtime).len(), 4);
        assert_eq!(total(&runtime.egress_refusals), 0);
    }

    #[test]
    fn a_counter_that_would_pass_u32_max_closes_its_window_or_says_it_saturated() {
        // Ledger working: the counter closes its window early and the events stay exact.
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime.record_egress_refusal(record(0, 1)).unwrap();
        runtime
            .egress_refusals
            .pods
            .get_mut(&pod(0))
            .unwrap()
            .counters
            .values_mut()
            .for_each(|counter| counter.count = u32::MAX - 1);
        runtime.record_egress_refusal(record(0, 1)).unwrap();
        let outcome = runtime.record_egress_refusal(record(0, 1)).unwrap();
        assert!(matches!(outcome, EgressRefusalOutcome::Committed(ref i) if i.len() == 1));
        let events = egress_events(&runtime);
        assert!(matches!(
            events.as_slice(),
            [LedgerEvent::EgressRefused {
                count: u32::MAX,
                saturated: false,
                ..
            }]
        ));
        assert_eq!(total(&runtime.egress_refusals), 1);

        // Ledger away: the merge saturates, and the event says so once the ledger is back.
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime.record_egress_refusal(record(0, 1)).unwrap();
        runtime
            .egress_refusals
            .pods
            .get_mut(&pod(0))
            .unwrap()
            .counters
            .values_mut()
            .for_each(|counter| counter.count = u32::MAX - 1);
        runtime.mark_commit_uncertain_for_test();
        let _ = runtime.record_egress_refusal(record(0, 1));
        let _ = runtime.record_egress_refusal(record(0, 1));
        assert_eq!(total(&runtime.egress_refusals), u64::from(u32::MAX));
        runtime.execution.complete_commit();
        runtime.flush_egress_refusals().unwrap();
        let events = egress_events(&runtime);
        assert!(
            matches!(
                events.as_slice(),
                [LedgerEvent::EgressRefused {
                    count: u32::MAX,
                    saturated: true,
                    ..
                }]
            ),
            "{events:?}"
        );
    }

    #[test]
    fn restore_returns_a_closed_counter_to_its_window_without_evicting_anything() {
        let mut windows = EgressRefusalWindows::default();
        let source = pod(0);
        let trace = TraceId::from("t");
        let reason = EgressRefusalReason::RateLimited;
        windows.absorb(&source, &trace, reason, "host:0123456789abcdef:1:1", 2, 7);
        let taken = windows.take_all();
        assert!(windows.pods.is_empty());
        for (window, event) in taken {
            windows.restore(window, event);
        }
        assert_eq!(total(&windows), 2);
        assert_eq!(windows.pods[&source].window, 7);
    }
}
