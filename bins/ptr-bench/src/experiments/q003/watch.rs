//! What a long run reports about itself, and the whole-run watchdog.
//!
//! Q003-v2's seed 29 ran 13 hours without an outcome and nothing in its record could say whether
//! it was stuck or slow: the drivers' pipe reads looked alike, there was no heartbeat, and the only
//! bound was a per-request timeout (one hour) that a slow cross-encoder never reached. Here the
//! harness keeps a [`Monitor`] (the phase, the request in flight, the slowest request, the
//! children), prints a heartbeat from it at a fixed interval, and a [`Watchdog`] ends the run with
//! exit status 2 and the typed reason `run-deadline` when the whole run exceeds its pinned budget.
//! A stop is always the harness's (status 2), never a finding about the product.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The typed reason a run ended with status 2 because of a driver or a limit. The tag is what
/// the operator, the tests and the mutation checker read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    /// The driver process could not be started.
    SpawnFailed,
    /// A request line could not be sent (the driver's input is closed, or it is not read).
    WriteFailed,
    /// A request was not answered within the per-request limit.
    RequestTimeout,
    /// The driver ended: its output closed, or it exited with a status.
    DriverExited,
    /// The driver answered `{"ok": false, ...}`.
    DriverRefused,
    /// A reply that is not one JSON object per line, or has the wrong shape.
    BadReply,
    /// A request the harness built that does not fit the one-line framing.
    BadRequest,
    /// The thread reading a driver's output stopped without a reply.
    ReaderStopped,
    /// The whole run exceeded its budget.
    RunDeadline,
}

impl StopReason {
    pub fn tag(self) -> &'static str {
        match self {
            StopReason::SpawnFailed => "spawn-failed",
            StopReason::WriteFailed => "write-failed",
            StopReason::RequestTimeout => "request-timeout",
            StopReason::DriverExited => "driver-exited",
            StopReason::DriverRefused => "driver-refused",
            StopReason::BadReply => "bad-reply",
            StopReason::BadRequest => "bad-request",
            StopReason::ReaderStopped => "reader-stopped",
            StopReason::RunDeadline => "run-deadline",
        }
    }
}

/// A harness stop: a reason and the detail an operator needs. It is exit status 2.
#[derive(Debug, Eq, PartialEq)]
pub struct Stop {
    pub reason: StopReason,
    pub detail: String,
}

impl Stop {
    pub fn new(reason: StopReason, detail: impl Into<String>) -> Self {
        Self {
            reason,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for Stop {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "harness stop [{}]: {}",
            self.reason.tag(),
            self.detail
        )
    }
}

impl From<Stop> for String {
    fn from(stop: Stop) -> String {
        stop.to_string()
    }
}

struct Flight {
    arm: String,
    op: String,
    since: Instant,
}

#[derive(Default)]
struct State {
    phase: String,
    flights: BTreeMap<String, Flight>,
    requests: u64,
    slowest: Option<(String, String, Duration)>,
    children: BTreeMap<u32, String>,
}

/// The run's self-description: shared by the bridges, the runner and the watchdog.
pub struct Monitor {
    started: Instant,
    state: Mutex<State>,
}

/// The monitor of this process.
pub fn monitor() -> &'static Monitor {
    static MONITOR: OnceLock<Monitor> = OnceLock::new();
    MONITOR.get_or_init(Monitor::new)
}

impl Default for Monitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Monitor {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            state: Mutex::new(State::default()),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        // A poisoned lock only means another thread panicked; the numbers are still readable.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn phase(&self, text: impl Into<String>) {
        self.state().phase = text.into();
    }

    pub fn request_begin(&self, arm: &str, op: &str) {
        self.state().flights.insert(
            arm.to_string(),
            Flight {
                arm: arm.to_string(),
                op: op.to_string(),
                since: Instant::now(),
            },
        );
    }

    /// The request of `arm` ended (answered or not); its duration joins the slowest-request record.
    pub fn request_end(&self, arm: &str) {
        let mut state = self.state();
        if let Some(flight) = state.flights.remove(arm) {
            let took = flight.since.elapsed();
            state.requests += 1;
            let slower = state
                .slowest
                .as_ref()
                .is_none_or(|(_, _, longest)| took > *longest);
            if slower {
                state.slowest = Some((flight.arm, flight.op, took));
            }
        }
    }

    pub fn child_started(&self, arm: &str, pid: u32) {
        self.state().children.insert(pid, arm.to_string());
    }

    pub fn child_gone(&self, pid: u32) {
        self.state().children.remove(&pid);
    }

    pub fn children(&self) -> Vec<u32> {
        self.state().children.keys().copied().collect()
    }

    /// One line: how long the run has gone, where it is, what is in flight and the slowest request.
    pub fn describe(&self) -> String {
        let state = self.state();
        let mut text = format!(
            "elapsed {} s, phase {:?}, {} requests answered",
            self.started.elapsed().as_secs(),
            state.phase,
            state.requests
        );
        for flight in state.flights.values() {
            text.push_str(&format!(
                ", {} {} in flight for {} s",
                flight.arm,
                flight.op,
                flight.since.elapsed().as_secs()
            ));
        }
        if let Some((arm, op, took)) = &state.slowest {
            text.push_str(&format!(
                ", slowest request {arm} {op} {} s",
                took.as_secs()
            ));
        }
        text
    }
}

/// Ends the run when it exceeds `limit`, and prints a heartbeat every `heartbeat`. Dropping it
/// (the run ended in time) stops the thread.
pub struct Watchdog {
    done: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Watchdog {
    /// Start watching `monitor`. At the deadline `on_expiry` is called once with the typed stop,
    /// from the watchdog's thread; the production action is [`stop_the_process`].
    pub fn start(
        monitor: &'static Monitor,
        limit: Duration,
        heartbeat: Duration,
        on_expiry: impl FnOnce(Stop) + Send + 'static,
    ) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        let handle = std::thread::spawn(move || {
            let began = Instant::now();
            let mut last_heartbeat = Instant::now();
            let tick = Duration::from_millis(50).min(limit).min(heartbeat);
            loop {
                if flag.load(Ordering::SeqCst) {
                    return;
                }
                if began.elapsed() >= limit {
                    on_expiry(Stop::new(
                        StopReason::RunDeadline,
                        format!(
                            "the run exceeded its budget of {} s; {}",
                            limit.as_secs(),
                            monitor.describe()
                        ),
                    ));
                    return;
                }
                if last_heartbeat.elapsed() >= heartbeat {
                    eprintln!(
                        "{}: heartbeat: {}",
                        super::params::BENCHMARK,
                        monitor.describe()
                    );
                    last_heartbeat = Instant::now();
                }
                std::thread::sleep(tick);
            }
        });
        Self {
            done,
            handle: Some(handle),
        }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The production expiry: name the stop, kill the drivers this process started, exit with 2.
pub fn stop_the_process(stop: Stop) {
    eprintln!("{}: {stop}", super::params::BENCHMARK);
    for pid in monitor().children() {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status();
    }
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn leaked() -> &'static Monitor {
        Box::leak(Box::new(Monitor::new()))
    }

    #[test]
    fn the_watchdog_ends_a_run_that_exceeds_its_budget_with_the_typed_reason() {
        let monitor = leaked();
        monitor.phase("checkpoint 2: query 100/700");
        monitor.request_begin("strong", "search");
        let (sender, receiver) = mpsc::channel();
        let _watchdog = Watchdog::start(
            monitor,
            Duration::from_millis(200),
            Duration::from_secs(3600),
            move |stop| {
                let _ = sender.send(stop);
            },
        );
        let stop = receiver
            .recv_timeout(Duration::from_secs(20))
            .expect("the watchdog fires");
        assert_eq!(stop.reason, StopReason::RunDeadline);
        assert!(
            stop.detail.contains("exceeded its budget"),
            "{}",
            stop.detail
        );
        assert!(stop.detail.contains("checkpoint 2: query 100/700"));
        assert!(stop.detail.contains("strong search in flight"));
        assert!(stop
            .to_string()
            .starts_with("harness stop [run-deadline]: "));
    }

    #[test]
    fn a_run_that_ends_in_time_is_not_stopped() {
        let (sender, receiver) = mpsc::channel();
        let watchdog = Watchdog::start(
            leaked(),
            Duration::from_millis(400),
            Duration::from_secs(3600),
            move |stop| {
                let _ = sender.send(stop);
            },
        );
        drop(watchdog);
        std::thread::sleep(Duration::from_millis(700));
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn the_monitor_keeps_the_slowest_request_and_the_children() {
        let monitor = Monitor::new();
        monitor.request_begin("bm25", "search");
        monitor.request_end("bm25");
        monitor.request_begin("strong", "search");
        std::thread::sleep(Duration::from_millis(30));
        monitor.request_end("strong");
        let text = monitor.describe();
        assert!(text.contains("2 requests answered"), "{text}");
        assert!(text.contains("slowest request strong search"), "{text}");
        monitor.child_started("strong", 4242);
        assert_eq!(monitor.children(), vec![4242]);
        monitor.child_gone(4242);
        assert!(monitor.children().is_empty());
    }

    #[test]
    fn every_reason_has_its_own_tag() {
        let all = [
            StopReason::SpawnFailed,
            StopReason::WriteFailed,
            StopReason::RequestTimeout,
            StopReason::DriverExited,
            StopReason::DriverRefused,
            StopReason::BadReply,
            StopReason::BadRequest,
            StopReason::ReaderStopped,
            StopReason::RunDeadline,
        ];
        let tags: std::collections::BTreeSet<_> = all.iter().map(|reason| reason.tag()).collect();
        assert_eq!(tags.len(), all.len());
    }
}
