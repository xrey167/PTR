//! Containment of the product and the judgement of one recovery.
//!
//! Every call into `ptr_ledger` and `ptr_runtime` that can run on a damaged file goes through
//! [`guarded`]: a panic is counted in `product_panics` and a call that outlives its watchdog in
//! `product_hangs`, and both falsify (exit 1). Only a panic in harness code ends the process
//! with status 101, which the aggregator reads as INVALID.
//!
//! [`judge`] turns what recovery returned into counters. The expectation it is given is built
//! by the arm from its own bookkeeping (the lengths it saw the file have, the records it was
//! acknowledged), never from the recovered bytes, so the product's classification of a pair of
//! files is compared with a table the harness computed itself.

use super::image::Shape;
use super::model::{Ev, Model, Standing, Workload};
use crate::experiments::layered_kv::emit::Tally;
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::{CommittedEvent, Split};
use ptr_runtime::PtrRuntime;
use ptr_types::{CapabilityId, Effect, Generation, TypeId, Validity};
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::time::Duration;

/// The payload of a panic the harness injects at a failpoint. Any other payload that reaches a
/// `catch_unwind` came from the product.
pub struct InjectedPanic;

/// Silence the default panic message for panics the harness injects. Called once, before the
/// run; a panic of any other kind keeps its message.
pub fn quiet_injected_panics() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if info.payload().downcast_ref::<InjectedPanic>().is_none() {
            default(info);
        }
    }));
}

/// The outcome of a contained call.
pub enum Guarded<T> {
    Done(T),
    Panicked,
    Hung,
}

/// Run `call` on its own thread under a watchdog. The thread of a call that outlives the
/// watchdog is left behind (it ends with the process).
pub fn guarded<T: Send + 'static>(
    watchdog: Duration,
    call: impl FnOnce() -> T + Send + 'static,
) -> Guarded<T> {
    let (sender, receiver) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("l001-guarded".into())
        .spawn(move || {
            let outcome = catch_unwind(AssertUnwindSafe(call));
            let _ = sender.send(outcome.ok());
        });
    if spawned.is_err() {
        return Guarded::Panicked;
    }
    match receiver.recv_timeout(watchdog) {
        Ok(Some(value)) => Guarded::Done(value),
        Ok(None) => Guarded::Panicked,
        Err(_) => Guarded::Hung,
    }
}

/// Count a contained call that did not finish and return what it produced otherwise.
pub fn counted<T>(tally: &mut Tally, arm: &str, outcome: Guarded<T>) -> Option<T> {
    match outcome {
        Guarded::Done(value) => Some(value),
        Guarded::Panicked => {
            bump(tally, arm, "product_panics");
            None
        }
        Guarded::Hung => {
            bump(tally, arm, "product_hangs");
            None
        }
    }
}

/// Like [`counted`] for a call that holds the process-wide failpoint scenario. A hang there
/// leaves the scenario locked for good, so the run records it in `append_hangs` and starts no
/// further phase that needs a failpoint ([`failpoints_blocked`]); it is falsified by the hang.
pub fn counted_append<T>(tally: &mut Tally, arm: &str, outcome: Guarded<T>) -> Option<T> {
    if matches!(outcome, Guarded::Hung) {
        tally.add("append_hangs", 1);
    }
    counted(tally, arm, outcome)
}

/// Whether a hung append still holds the failpoint scenario.
pub fn failpoints_blocked(tally: &Tally) -> bool {
    tally.get("append_hangs") > 0
}

/// Add one to a counter and to the same counter of an arm.
pub fn bump(tally: &mut Tally, arm: &str, name: &str) {
    add(tally, arm, name, 1);
}

pub fn add(tally: &mut Tally, arm: &str, name: &str, amount: i64) {
    tally.add(name, amount);
    tally.add(&format!("arms.{arm}.{name}"), amount);
}

/// Whether the run should stop issuing recoveries: the watchdog has fired often enough.
pub fn hang_limit_reached(tally: &Tally) -> bool {
    tally.get("product_hangs") >= super::params::HANG_LIMIT
}

/// The classification of a log and its anchor, in the harness's own terms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cls {
    Aligned,
    Unacknowledged { records: usize, partial: bool },
    LostSuffix,
    Other,
}

impl Cls {
    pub fn of(split: Split) -> Self {
        match split {
            Split::Aligned => Cls::Aligned,
            Split::Unacknowledged {
                complete_records,
                incomplete_frame,
                ..
            } => Cls::Unacknowledged {
                records: complete_records,
                partial: incomplete_frame,
            },
            Split::LostSuffix { .. } => Cls::LostSuffix,
            Split::Diverged | Split::OriginMismatch | Split::BaseMismatch => Cls::Other,
        }
    }
}

/// What the harness knows about the state it handed to recovery.
#[derive(Clone, Debug)]
pub struct Expectation {
    /// Events whose append returned `Ok` before the crash: the acknowledged prefix.
    pub acked: usize,
    /// Events whose append had begun: the acknowledged ones and the one in flight.
    pub attempted: usize,
    /// An acknowledged byte is missing (a lie, or an anchor rolled back below the witness).
    pub must_refuse: bool,
    /// The image holds zeroed sectors after the synced length, of this shape. Zeros only
    /// ([`Shape::Zeros`]) must recover: the file is its synced prefix and a tail with no
    /// written byte left, which the anchor shows to be unacknowledged. A product may refuse any
    /// other shape (a full-length frame with a zero run is an invalid frame to the format, which
    /// never repairs input), within the pinned share bound.
    pub damage: Option<Shape>,
    /// The classification the table gives, when the arm can compute one.
    pub class: Option<Cls>,
    /// The anchor epoch the host retained, passed to recovery as the rollback witness.
    pub witness: u64,
}

/// What recovery returned.
pub enum Recovery {
    Recovered {
        events: Vec<CommittedEvent>,
        class: Option<Cls>,
    },
    Refused {
        class: Option<Cls>,
    },
}

/// Say on stderr why a recovery was counted against the product, when `L001_EXPLAIN` is set. A
/// diagnostic for whoever reads a failing run; nothing reads it.
fn explain(arm: &str, what: &str, expect: &Expectation, got: &str) {
    if std::env::var_os("L001_EXPLAIN").is_some() {
        eprintln!("L001 {arm}: {what}: expected {expect:?}, got {got}");
    }
}

/// Judge one recovery against the workload's history. Adds to the hard counters, and to the
/// descriptive ones, of `tally` under the name of the arm.
pub fn judge(
    tally: &mut Tally,
    arm: &str,
    workload: &Workload,
    expect: &Expectation,
    recovery: Recovery,
) {
    bump(tally, arm, "recoveries");
    if let Some(shape) = expect.damage {
        bump(tally, arm, "recoveries_damaged");
        bump(tally, arm, &format!("recoveries_shape_{}", shape.tag()));
    }
    match recovery {
        Recovery::Refused { class } => {
            bump(tally, arm, "refusals");
            if let Some(shape) = expect.damage {
                bump(tally, arm, "refusals_damaged");
                bump(tally, arm, &format!("refusals_shape_{}", shape.tag()));
            }
            if expect.must_refuse {
                bump(tally, arm, "refusals_lost_ack");
                if let (Some(want), Some(got)) = (expect.class, class) {
                    if want != got {
                        explain(
                            arm,
                            "the refusal is classified differently",
                            expect,
                            &format!("{got:?}"),
                        );
                        bump(tally, arm, "unclassified_split");
                    }
                }
            } else if expect.damage.is_some_and(|shape| shape != Shape::Zeros) {
                bump(tally, arm, "refusals_corrupt_tail");
            } else {
                // Also a tail of zeros only: nothing written survives in it.
                explain(
                    arm,
                    "the product refused a recoverable state",
                    expect,
                    &format!("{class:?}"),
                );
                bump(tally, arm, "unjustified_refusals");
            }
        }
        Recovery::Recovered { events, class } => {
            bump(tally, arm, "recovered");
            if expect.must_refuse {
                // The anchored product opened a log that lost an acknowledged record, or an
                // anchor older than the witness: the central safety defect.
                explain(arm, "a lie image was accepted", expect, "recovered");
                bump(tally, arm, "anchored_lie_accepted");
            }
            if let (Some(want), Some(got)) = (expect.class, class) {
                if want != got {
                    explain(
                        arm,
                        "the pair is classified differently",
                        expect,
                        &format!("{got:?}"),
                    );
                    bump(tally, arm, "unclassified_split");
                }
            }
            judge_history(tally, arm, workload, expect, &events);
        }
    }
}

/// The checks on a recovered history: indices, content, the acknowledged prefix, and the
/// authority the product derives from it against the oracle's.
pub fn judge_history(
    tally: &mut Tally,
    arm: &str,
    workload: &Workload,
    expect: &Expectation,
    events: &[CommittedEvent],
) {
    let length = events.len();
    if events
        .iter()
        .enumerate()
        .any(|(i, event)| event.index.0 != i as u64 + 1)
    {
        bump(tally, arm, "reorder_or_duplicate_index");
    }
    let attempted = expect.attempted.min(workload.events.len());
    let content_ok = length <= attempted
        && events
            .iter()
            .zip(&workload.events)
            .all(|(got, want)| got.event == want.to_ledger());
    if !content_ok {
        bump(tally, arm, "recovered_not_prefix");
    }
    if length < expect.acked {
        if content_ok {
            bump(tally, arm, "recovered_not_prefix");
        }
        if workload.events[length..expect.acked]
            .iter()
            .any(|event| event.is_revocation())
        {
            bump(tally, arm, "acked_revocation_missing");
        }
    }
    if content_ok {
        let held = length.min(expect.acked).min(workload.events.len());
        if workload.events[..held].iter().any(Ev::is_revocation) {
            bump(tally, arm, "recoveries_with_acked_revocation");
        }
        authority(tally, arm, workload, events, expect.acked);
    }
}

/// Replay `events` through the runtime and compare its answer for every pair of the universe
/// with the oracle's, for the same events. A pair the product authorizes although the oracle
/// does not, or although an acknowledged revocation names it, is a stale accept; any other
/// disagreement (a refusal of a live generation, another standing) is an authority mismatch.
fn authority(
    tally: &mut Tally,
    arm: &str,
    workload: &Workload,
    events: &[CommittedEvent],
    acked: usize,
) {
    let history = workload.prefix(events.len());
    let model = Model::of(history);
    let acked_revoked: BTreeSet<(String, u64)> = workload.events
        [..acked.min(workload.events.len())]
        .iter()
        .filter_map(|event| match event {
            Ev::Revoke {
                subject,
                generation,
            } => Some((subject.clone(), *generation)),
            _ => None,
        })
        .collect();
    let replayed = catch_unwind(AssertUnwindSafe(|| {
        PtrRuntime::replay(PtrConfig::default(), events)
    }));
    let mut runtime = match replayed {
        Err(_) => {
            bump(tally, arm, "product_panics");
            return;
        }
        Ok(Err(_)) => {
            // The oracle calls this history valid; the product refused to replay it.
            bump(tally, arm, "authority_mismatches");
            return;
        }
        Ok(Ok(runtime)) => runtime,
    };
    // Everything the boundary asks for besides the generation is granted, so the answer for a
    // pair is the generation's alone.
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("memory.write"));
    runtime.permissions_mut().allow_mutation = true;
    for (subject, generation) in &workload.universe {
        let decided = catch_unwind(AssertUnwindSafe(|| decide(&runtime, subject, *generation)));
        let Ok((authorized, standing)) = decided else {
            bump(tally, arm, "product_panics");
            continue;
        };
        let wanted = model.authorizable(subject, *generation);
        let revoked_by_ack = acked_revoked.contains(&(subject.clone(), *generation));
        if !authorized
            && revoked_by_ack
            && model.standing(subject, *generation) == Some(Standing::Revoked)
        {
            // The claim itself, observed: an acknowledged revocation survived and denied.
            bump(tally, arm, "revoked_pairs_denied");
        }
        if authorized && (!wanted || revoked_by_ack) {
            bump(tally, arm, "stale_accepts");
        } else if authorized != wanted || standing != model.standing(subject, *generation) {
            bump(tally, arm, "authority_mismatches");
        }
    }
}

/// The runtime's answer for one pair: may an action proceed, and what is the generation's
/// standing.
fn decide(runtime: &PtrRuntime, subject: &str, generation: u64) -> (bool, Option<Standing>) {
    let action = ActionIr {
        operation: "update".into(),
        target: subject.to_owned(),
        capability: CapabilityId::from("memory.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("SemanticCapsule"),
        generation: Generation(generation),
        revision: runtime.revision(),
        payload: vec![],
    };
    let authorized = runtime.authorize_action(&action).is_ok();
    let standing = runtime
        .generation_validity(subject, Generation(generation))
        .map(|validity| match validity {
            Validity::Live => Standing::Live,
            Validity::Superseded => Standing::Superseded,
            Validity::Revoked => Standing::Revoked,
            Validity::Disputed => Standing::Disputed,
        });
    (authorized, standing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::l001::metrics;
    use crate::experiments::l001::model::{generate, Ev};
    use crate::experiments::rng::Rng;
    use ptr_types::CommitIndex;

    fn committed(events: &[Ev]) -> Vec<CommittedEvent> {
        events
            .iter()
            .enumerate()
            .map(|(i, event)| CommittedEvent {
                index: CommitIndex(i as u64 + 1),
                event: event.to_ledger(),
            })
            .collect()
    }

    fn expect(acked: usize, attempted: usize) -> Expectation {
        Expectation {
            acked,
            attempted,
            must_refuse: false,
            damage: None,
            class: None,
            witness: 0,
        }
    }

    fn workload() -> Workload {
        generate(&mut Rng::new(11), 8, 8)
    }

    fn fired(tally: &Tally) -> Vec<&'static str> {
        metrics::HARD_COUNTERS
            .into_iter()
            .filter(|name| tally.get(name) > 0)
            .collect()
    }

    /// The index of the first revocation in `workload`.
    fn revocation_at(workload: &Workload) -> usize {
        workload.events.iter().position(Ev::is_revocation).unwrap()
    }

    #[test]
    fn an_exact_prefix_of_what_was_sent_trips_nothing() {
        let workload = workload();
        for length in [0, 3, workload.events.len()] {
            let mut tally = Tally::new(metrics::LIST);
            let events = committed(workload.prefix(length));
            judge(
                &mut tally,
                "t",
                &workload,
                &expect(length.saturating_sub(1), length),
                Recovery::Recovered {
                    events,
                    class: None,
                },
            );
            assert_eq!(fired(&tally), Vec::<&str>::new(), "{length}");
        }
    }

    #[test]
    fn a_recovery_that_drops_an_acknowledged_revocation_is_a_stale_accept_and_a_loss() {
        let workload = workload();
        let at = revocation_at(&workload);
        let mut tally = Tally::new(metrics::LIST);
        judge(
            &mut tally,
            "t",
            &workload,
            &expect(at + 1, at + 1),
            Recovery::Recovered {
                events: committed(workload.prefix(at)),
                class: None,
            },
        );
        let fired = fired(&tally);
        for name in [
            "stale_accepts",
            "acked_revocation_missing",
            "recovered_not_prefix",
        ] {
            assert!(fired.contains(&name), "{name} in {fired:?}");
        }
    }

    #[test]
    fn indices_that_do_not_run_from_one_content_that_differs_and_a_phantom_record_are_each_caught()
    {
        let workload = workload();
        let mut zero_based = committed(workload.prefix(4));
        zero_based
            .iter_mut()
            .for_each(|e| e.index = CommitIndex(e.index.0 - 1));
        let mut tally = Tally::new(metrics::LIST);
        judge(
            &mut tally,
            "t",
            &workload,
            &expect(0, 4),
            Recovery::Recovered {
                events: zero_based,
                class: None,
            },
        );
        assert!(fired(&tally).contains(&"reorder_or_duplicate_index"));

        let mut swapped = committed(workload.prefix(4));
        swapped.swap(1, 2);
        let mut tally = Tally::new(metrics::LIST);
        judge(
            &mut tally,
            "t",
            &workload,
            &expect(0, 4),
            Recovery::Recovered {
                events: swapped,
                class: None,
            },
        );
        assert!(fired(&tally).contains(&"recovered_not_prefix"));

        let mut tally = Tally::new(metrics::LIST);
        judge(
            &mut tally,
            "t",
            &workload,
            &expect(2, 3),
            Recovery::Recovered {
                events: committed(workload.prefix(5)),
                class: None,
            },
        );
        assert!(
            fired(&tally).contains(&"recovered_not_prefix"),
            "a record nobody sent"
        );
    }

    #[test]
    fn a_refusal_is_judged_by_what_the_table_says_of_the_state() {
        let workload = workload();
        let refused = |expect: &Expectation, class| {
            let mut tally = Tally::new(metrics::LIST);
            judge(
                &mut tally,
                "t",
                &workload,
                expect,
                Recovery::Refused { class },
            );
            tally
        };
        // A recoverable state that is refused is a defect.
        assert_eq!(
            fired(&refused(&expect(2, 3), None)),
            ["unjustified_refusals"]
        );
        // A lie must be refused, and its classification is compared.
        let lie = Expectation {
            must_refuse: true,
            class: Some(Cls::LostSuffix),
            ..expect(4, 4)
        };
        assert_eq!(
            fired(&refused(&lie, Some(Cls::LostSuffix))),
            Vec::<&str>::new()
        );
        assert_eq!(
            fired(&refused(&lie, Some(Cls::Aligned))),
            ["unclassified_split"]
        );
        // A hole in a complete frame may be refused.
        let hole = Expectation {
            damage: Some(Shape::Mixed),
            ..expect(2, 3)
        };
        let tally = refused(&hole, None);
        assert_eq!(fired(&tally), Vec::<&str>::new());
        assert_eq!(tally.get("refusals_corrupt_tail"), 1);
        assert_eq!(tally.get("refusals_damaged"), 1);
        assert_eq!(tally.get("refusals_shape_mixed"), 1);
        // A tail of zeros alone must recover: refusing it is unjustified.
        let zeros = Expectation {
            damage: Some(Shape::Zeros),
            ..expect(2, 3)
        };
        let tally = refused(&zeros, None);
        assert_eq!(fired(&tally), ["unjustified_refusals"]);
        assert_eq!(tally.get("refusals_corrupt_tail"), 0);
        assert_eq!(tally.get("refusals_shape_zeros"), 1);
    }

    #[test]
    fn a_lie_image_that_the_anchored_product_opens_is_a_falsifier_and_a_refusal_is_not() {
        let workload = workload();
        let at = revocation_at(&workload);
        let lie = Expectation {
            must_refuse: true,
            ..expect(at + 1, at + 1)
        };
        let mut tally = Tally::new(metrics::LIST);
        judge(
            &mut tally,
            "t",
            &workload,
            &lie,
            Recovery::Recovered {
                events: committed(workload.prefix(at + 1)),
                class: None,
            },
        );
        assert_eq!(fired(&tally), ["anchored_lie_accepted"]);
        let mut tally = Tally::new(metrics::LIST);
        judge(
            &mut tally,
            "t",
            &workload,
            &lie,
            Recovery::Refused { class: None },
        );
        assert_eq!(fired(&tally), Vec::<&str>::new());
    }

    #[test]
    fn the_observed_coverage_counters_move_with_what_was_recovered() {
        let workload = workload();
        let at = revocation_at(&workload);
        let mut tally = Tally::new(metrics::LIST);
        judge_history(
            &mut tally,
            "t",
            &workload,
            &expect(at + 1, at + 1),
            &committed(workload.prefix(at + 1)),
        );
        assert_eq!(tally.get("recoveries_with_acked_revocation"), 1);
        assert!(tally.get("revoked_pairs_denied") >= 1);
        let mut tally = Tally::new(metrics::LIST);
        judge_history(
            &mut tally,
            "t",
            &workload,
            &expect(at, at),
            &committed(workload.prefix(at)),
        );
        assert_eq!(
            tally.get("recoveries_with_acked_revocation"),
            0,
            "no revocation was acknowledged"
        );
        assert_eq!(tally.get("revoked_pairs_denied"), 0);
    }

    #[test]
    fn a_pair_classified_differently_from_the_table_is_caught_even_when_the_history_is_right() {
        let workload = workload();
        let mut tally = Tally::new(metrics::LIST);
        let wanted = Expectation {
            class: Some(Cls::Aligned),
            ..expect(3, 3)
        };
        judge(
            &mut tally,
            "t",
            &workload,
            &wanted,
            Recovery::Recovered {
                events: committed(workload.prefix(3)),
                class: Some(Cls::Unacknowledged {
                    records: 1,
                    partial: false,
                }),
            },
        );
        assert_eq!(fired(&tally), ["unclassified_split"]);
    }

    #[test]
    fn the_runtime_agrees_with_the_oracle_on_every_prefix_of_many_workloads() {
        for seed in 0..40 {
            let workload = generate(&mut Rng::new(seed), 6, 14);
            for length in 0..=workload.events.len() {
                let mut tally = Tally::new(metrics::LIST);
                judge_history(
                    &mut tally,
                    "t",
                    &workload,
                    &expect(length, length),
                    &committed(workload.prefix(length)),
                );
                assert_eq!(
                    fired(&tally),
                    Vec::<&str>::new(),
                    "seed {seed} prefix {length}"
                );
            }
        }
    }

    #[test]
    fn a_panic_and_a_hang_are_contained_and_counted() {
        let mut tally = Tally::new(metrics::LIST);
        // Generous for the first two: printing a panic's backtrace can take a while on a busy
        // machine, and a watchdog that fires first would call the panic a hang.
        let patient = Duration::from_secs(60);
        assert!(matches!(guarded(patient, || 7), Guarded::Done(7)));
        let panicked = guarded(patient, || -> u8 { panic!("a defect") });
        assert!(counted(&mut tally, "t", panicked).is_none());
        let hung = guarded(Duration::from_millis(300), || {
            std::thread::sleep(Duration::from_secs(5));
        });
        assert!(counted(&mut tally, "t", hung).is_none());
        assert_eq!(
            (tally.get("product_panics"), tally.get("product_hangs")),
            (1, 1)
        );
        assert_eq!(tally.get("arms.t.product_panics"), 1);
    }

    #[test]
    fn the_hang_limit_stops_further_recoveries() {
        let mut tally = Tally::new(metrics::LIST);
        assert!(!hang_limit_reached(&tally));
        tally.add("product_hangs", super::super::params::HANG_LIMIT);
        assert!(hang_limit_reached(&tally));
    }
}
