//! The worker supervisor through the public API (doc 42 stage S1): real child processes,
//! real cgroups, real user and network namespaces. Linux only, feature `worker-supervisor`.
//!
//! The child is this very test binary, started with `--exact supervisor_child_helper` and an
//! environment variable that names what it should do. The helper does nothing in an ordinary
//! run, so it passes as a test of its own.
//!
//! Tests that need root (a user id of their own, a cgroup-v2 group, a network namespace) detect
//! what the host provides and skip with a printed reason; under `CI=true` they fail instead,
//! unless `PTR_WORKER_UNPRIVILEGED_CI=1` says the runner is unprivileged on purpose.
#![cfg(all(feature = "worker-supervisor", target_os = "linux"))]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ptr_fabric::{
    Channel, ChannelError, ConfinePolicy, ConfinementProbe, LostCause, ReapFailure, ShmExport,
    SpawnRequest, Supervisor, SupervisorConfig, SupervisorError, SupervisorEvent,
};

const MODE_VAR: &str = "PTR_SUP_CHILD";

// --- the child ----------------------------------------------------------------------------

/// What the spawned copy of this binary does; an ordinary test run returns at once.
#[test]
fn supervisor_child_helper() {
    let Ok(mode) = std::env::var(MODE_VAR) else {
        return;
    };
    // SAFETY: the supervisor put the channel at descriptor 3 and nothing else owns it.
    let channel = unsafe { Channel::from_inherited_fd(3) }.expect("inherited channel");
    match mode.as_str() {
        "idle" => {
            // Wait until the supervisor closes its end or kills this process.
            let _ = channel.recv();
        }
        "send-fd-then-sleep" => {
            // A descendant that is not confined: the cgroup kill must reach it too.
            // Never waited for on purpose: it must outlive this process until the cgroup kill.
            #[allow(clippy::zombie_processes)]
            let grandchild = std::process::Command::new("/bin/sleep")
                .arg("120")
                .spawn()
                .expect("grandchild");
            channel
                .send(grandchild.id().to_string().as_bytes(), &[])
                .expect("pid frame");
            let export = ShmExport::create(&[7u8; 10_000]).expect("export");
            channel
                .send(b"queued descriptor", &[export.fd()])
                .expect("fd frame");
            let _ = channel.recv();
        }
        other => panic!("unknown mode {other}"),
    }
}

// --- helpers ------------------------------------------------------------------------------

/// The test binary copied to a directory of this process's own, because the worker runs under a
/// user id of its own and the target directory may not be readable by it.
///
/// The directory has a random name and mode 0711 (the worker's user can enter it but not list
/// it); the copy is written under a temporary name, synced and renamed, so no exec ever sees a
/// file that is still open for writing. It runs under `SPAWN` (see `spawn_in`).
fn worker_program() -> PathBuf {
    static COPY: OnceLock<PathBuf> = OnceLock::new();
    COPY.get_or_init(|| {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        let source = std::env::current_exe().expect("current exe");
        remove_stale_copies();
        let mut random = [0u8; 8];
        ptr_fabric::fill_random(&mut random).expect("random");
        let directory = std::env::temp_dir().join(format!(
            "ptr-fabric-supervisor-test-{}-{:016x}",
            std::process::id(),
            u64::from_le_bytes(random)
        ));
        std::fs::DirBuilder::new()
            .mode(0o711)
            .create(&directory)
            .expect("a fresh test dir");
        let partial = directory.join("partial");
        std::fs::copy(&source, &partial).expect("copy the test binary");
        std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o755)).expect("mode");
        std::fs::File::open(&partial)
            .and_then(|file| file.sync_all())
            .expect("sync");
        let target = directory.join("worker");
        std::fs::rename(&partial, &target).expect("publish the copy");
        target
    })
    .clone()
}

/// Directories earlier runs of this test left behind (a directory is named after the pid of the
/// process that made it): remove those whose process is gone. Errors are ignored; a directory
/// made by another user stays.
fn remove_stale_copies() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("ptr-fabric-supervisor-test-") else {
            continue;
        };
        let Some(pid) = rest
            .split('-')
            .next()
            .and_then(|pid| pid.parse::<u32>().ok())
        else {
            continue;
        };
        if !Path::new(&format!("/proc/{pid}")).exists() {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Held while a worker is spawned: another thread's fork between the copy and the exec would
/// hand the copy's write descriptor to a child that has not exec'd yet, and the exec would fail
/// with "Text file busy". Every fork of these tests goes through `spawn_in`.
static SPAWN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Spawn under the lock. `ETXTBSY` from a fork this test file does not control (the probe forks
/// too) is retried here, in the test helper only; the library never swallows it.
fn spawn_in(
    supervisor: &mut Supervisor,
    request: &SpawnRequest,
) -> Result<ptr_fabric::SpawnedWorker, SupervisorError> {
    let _guard = SPAWN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut request = request.clone();
    if request.program.as_os_str().is_empty() {
        request.program = worker_program();
    }
    for attempt in 0..50 {
        match supervisor.spawn(&request) {
            Err(SupervisorError::Spawn(message))
                if message.contains("Text file busy") && attempt < 49 =>
            {
                std::thread::sleep(Duration::from_millis(20));
            }
            other => return other,
        }
    }
    unreachable!("the loop returns on its last attempt")
}

fn request(mode: &str) -> SpawnRequest {
    let mut request = SpawnRequest::new("")
        .arg("--exact")
        .arg("supervisor_child_helper")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(MODE_VAR, mode);
    request.channel_fd_flag = None;
    request
}

/// Whether this host gives a worker all three mechanisms the supervisor applies. Skips with a
/// printed reason where it does not; fails under `CI=true` unless the runner is declared
/// unprivileged.
fn privileged(test: &str) -> bool {
    let probe = ConfinementProbe::detect();
    let missing: Vec<String> = [
        ("own user id", &probe.own_uid),
        ("cgroup v2", &probe.cgroup_v2),
        ("network namespace", &probe.net_namespace),
    ]
    .into_iter()
    .filter_map(|(name, availability)| match availability {
        ptr_fabric::Availability::Available => None,
        ptr_fabric::Availability::Unavailable(why) => Some(format!("{name}: {why}")),
    })
    .collect();
    if missing.is_empty() {
        return true;
    }
    let reason = missing.join("; ");
    let strict = std::env::var("CI").as_deref() == Ok("true")
        && std::env::var("PTR_WORKER_UNPRIVILEGED_CI").as_deref() != Ok("1");
    assert!(
        !strict,
        "{test} needs root under CI=true but the host lacks: {reason}"
    );
    eprintln!("SKIPPED {test}: {reason}");
    false
}

fn config() -> SupervisorConfig {
    SupervisorConfig {
        max_workers: 4,
        reap_timeout: Duration::from_secs(10),
        ..SupervisorConfig::default()
    }
}

fn process_exists(pid: i32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

fn wait_for<T>(what: &str, mut check: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(value) = check() {
            return value;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

// --- tests --------------------------------------------------------------------------------

#[test]
fn spawn_applies_what_the_kernel_then_shows() {
    if !privileged("spawn_applies_what_the_kernel_then_shows") {
        return;
    }
    let mut supervisor = Supervisor::new(config()).expect("supervisor");
    let worker = spawn_in(&mut supervisor, &request("idle")).expect("spawn");
    let observation = &worker.observation;
    assert!(observation.own_uid, "{observation:?}");
    assert!(observation.cgroup_v2, "{observation:?}");
    assert!(observation.net_namespace, "{observation:?}");
    assert!(observation.no_new_privs, "{observation:?}");
    assert!(observation.capabilities_dropped, "{observation:?}");
    assert!(observation.bounding_set_empty, "{observation:?}");
    // The worker has not installed a filter (this helper never does), so the confinement is
    // everything but seccomp, and the supervisor does not claim more.
    assert!(!observation.seccomp_filter);
    assert_eq!(
        worker.confinement,
        ptr_fabric::Confinement::Partial {
            missing: vec![ptr_fabric::Mechanism::Seccomp]
        }
    );
    assert_eq!(
        supervisor
            .confirm_seccomp(worker.session, &[])
            .expect("confirm"),
        worker.confinement
    );
    // The assigned user id is the one the kernel shows, and it is not root.
    let uid = worker.uid.expect("a uid of its own");
    assert!(uid >= SupervisorConfig::default().uid_base);
    let status = std::fs::read_to_string(format!("/proc/{}/status", worker.pid)).expect("status");
    assert!(
        status.contains(&format!("Uid:\t{uid}\t{uid}\t{uid}\t{uid}")),
        "{status}"
    );
    let directory = worker.cgroup.clone().expect("a cgroup");
    assert!(directory.is_dir());
    let procs = std::fs::read_to_string(directory.join("cgroup.procs")).expect("procs");
    assert!(procs.lines().any(|line| line == worker.pid.to_string()));
    let reaped = supervisor.reap(worker.session).expect("reap");
    assert!(reaped.cgroup_removed());
    assert!(!directory.exists());
    assert!(!process_exists(worker.pid as i32));
}

#[test]
fn reap_kills_the_whole_cgroup_removes_it_and_drains_the_queued_descriptor() {
    if !privileged("reap_kills_the_whole_cgroup_removes_it_and_drains_the_queued_descriptor") {
        return;
    }
    let mut supervisor = Supervisor::new(config()).expect("supervisor");
    let worker = spawn_in(&mut supervisor, &request("send-fd-then-sleep")).expect("spawn");
    let channel = supervisor.channel(worker.session).expect("channel");
    assert!(channel
        .wait_readable(Some(Duration::from_secs(15)))
        .expect("wait"));
    let pid_frame = channel.recv().expect("pid frame");
    let grandchild: i32 = String::from_utf8(pid_frame.body)
        .expect("utf8")
        .parse()
        .expect("pid");
    assert!(process_exists(grandchild));
    // The descriptor frame is queued but the supervisor never reads it: only the drain after
    // the reap may close it.
    assert!(channel
        .wait_readable(Some(Duration::from_secs(15)))
        .expect("wait"));
    let directory = worker.cgroup.clone().expect("a cgroup");
    assert!(std::fs::read_to_string(directory.join("cgroup.procs"))
        .expect("procs")
        .lines()
        .any(|line| line == grandchild.to_string()));

    let reaped = supervisor.reap(worker.session).expect("reap");
    assert_eq!(reaped.session(), worker.session);
    assert_eq!(
        reaped.drain_report().frames,
        1,
        "{:?}",
        reaped.drain_report()
    );
    assert_eq!(
        reaped.drain_report().descriptors,
        1,
        "{:?}",
        reaped.drain_report()
    );
    assert!(reaped.cgroup_removed());
    assert!(!directory.exists());
    wait_for("the grandchild to be gone", || {
        (!process_exists(grandchild)).then_some(())
    });
    assert!(!process_exists(worker.pid as i32));
    assert!(!supervisor.is_registered(worker.session));
    assert!(matches!(
        supervisor.reap(worker.session),
        Err(SupervisorError::UnknownSession(_))
    ));
}

#[test]
fn a_crashed_worker_is_detected_then_reaped() {
    let mut supervisor = Supervisor::new(SupervisorConfig {
        policy: ConfinePolicy::BEST_EFFORT,
        ..config()
    })
    .expect("supervisor");
    let worker = spawn_in(&mut supervisor, &request("idle")).expect("spawn");
    assert!(supervisor.poll().is_empty(), "a live worker is not lost");
    supervisor.kill_process(worker.session).expect("kill");
    let event = wait_for("the loss to be reported", || supervisor.poll().pop());
    match event {
        SupervisorEvent::Lost { session, cause } => {
            assert_eq!(session, worker.session);
            assert!(
                matches!(
                    cause,
                    LostCause::ChildExited(info) if info.signal == Some(9)
                ) || cause == LostCause::ChannelHungUp
            );
        }
    }
    // Reported once.
    assert!(supervisor.poll().is_empty());
    // The worker's end is closed: the channel says so, which is the second way to detect it.
    let channel = supervisor.channel(worker.session).expect("channel");
    assert!(channel.peer_hung_up());
    assert_eq!(channel.recv().unwrap_err(), ChannelError::PeerClosed);
    let reaped = supervisor.reap(worker.session).expect("reap");
    assert_eq!(reaped.exit().signal, Some(9));
    assert!(!supervisor.is_registered(worker.session));
}

#[test]
fn the_number_of_workers_is_bounded_and_sessions_are_fresh() {
    let mut supervisor = Supervisor::new(SupervisorConfig {
        max_workers: 2,
        policy: ConfinePolicy::NONE,
        ..config()
    })
    .expect("supervisor");
    let first = spawn_in(&mut supervisor, &request("idle")).expect("first");
    let second = spawn_in(&mut supervisor, &request("idle")).expect("second");
    assert_ne!(first.session, second.session);
    assert!(matches!(
        spawn_in(&mut supervisor, &request("idle")),
        Err(SupervisorError::TooManyWorkers { max: 2 })
    ));
    // Without the supervisor's own mechanisms, capabilities are still dropped and the
    // confinement honestly says nothing is applied.
    assert!(first.observation.capabilities_dropped);
    assert!(first.observation.no_new_privs);
    assert_eq!(first.confinement, ptr_fabric::Confinement::None);
    assert!(first.cgroup.is_none() && first.uid.is_none());
    supervisor.reap(first.session).expect("reap first");
    // A slot is free again, and the new worker has a session nobody had before.
    let third = spawn_in(&mut supervisor, &request("idle")).expect("third");
    assert!(third.session != first.session && third.session != second.session);
}

#[test]
fn a_program_that_cannot_start_is_a_typed_error_and_leaves_nothing_behind() {
    // A group of its own to create worker groups in, so concurrent tests do not interfere.
    let parent = cgroup_parent().and_then(|own| {
        let private = own.join(format!("ptr-test-parent-{}", std::process::id()));
        std::fs::create_dir(&private).ok().map(|()| private)
    });
    let mut supervisor = Supervisor::new(SupervisorConfig {
        cgroup_parent: parent.clone(),
        ..config()
    })
    .expect("supervisor");
    let error = spawn_in(
        &mut supervisor,
        &SpawnRequest::new("/nonexistent/ptr-worker"),
    )
    .unwrap_err();
    assert!(matches!(error, SupervisorError::Spawn(_)), "{error}");
    assert!(supervisor.sessions().is_empty());
    if let Some(parent) = parent {
        let leftovers: Vec<_> = std::fs::read_dir(&parent)
            .expect("parent")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        std::fs::remove_dir(&parent).expect("remove the private parent group");
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }
}

#[test]
fn a_required_mechanism_the_host_lacks_refuses_the_spawn() {
    let probe = ConfinementProbe::detect();
    if probe.available() == ptr_fabric::AppliedMechanisms::ALL {
        eprintln!("SKIPPED: this host provides every mechanism, so nothing is refused");
        return;
    }
    let mut supervisor = Supervisor::new(SupervisorConfig {
        policy: ConfinePolicy::REQUIRE_ALL,
        ..config()
    })
    .expect("supervisor");
    assert!(matches!(
        spawn_in(&mut supervisor, &request("idle")),
        Err(SupervisorError::Unavailable { .. })
    ));
    assert!(supervisor.sessions().is_empty());
}

#[test]
fn dropping_the_supervisor_kills_its_workers() {
    let pid = {
        let mut supervisor = Supervisor::new(config()).expect("supervisor");
        let worker = spawn_in(&mut supervisor, &request("idle")).expect("spawn");
        assert!(process_exists(worker.pid as i32));
        worker.pid as i32
    };
    wait_for("the worker to be gone", || {
        (!process_exists(pid)).then_some(())
    });
}

#[test]
fn reap_reports_a_step_that_cannot_complete_and_keeps_the_worker() {
    // A zero timeout cannot wait for anything: if the child is still running when the reap
    // looks, the failure is typed and the worker stays registered for a retry.
    let mut supervisor = Supervisor::new(SupervisorConfig {
        policy: ConfinePolicy::NONE,
        reap_timeout: Duration::ZERO,
        ..config()
    })
    .expect("supervisor");
    let worker = spawn_in(&mut supervisor, &request("idle")).expect("spawn");
    match supervisor.reap(worker.session) {
        Ok(_) => {} // the kill won the race with the first wait
        Err(SupervisorError::Reap(ReapFailure::ChildNotExited)) => {
            assert!(supervisor.is_registered(worker.session));
        }
        Err(other) => panic!("unexpected {other}"),
    }
}

fn cgroup_parent() -> Option<PathBuf> {
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    let mount = mountinfo.lines().find_map(|line| {
        let (head, tail) = line.split_once(" - ")?;
        (tail.split_whitespace().next()? == "cgroup2")
            .then(|| head.split_whitespace().nth(4).map(str::to_owned))?
    })?;
    let own = std::fs::read_to_string("/proc/self/cgroup")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("0::").map(str::to_owned))?;
    Some(Path::new(&mount).join(own.trim_start_matches('/')))
}

#[test]
fn two_supervisors_with_disjoint_uid_ranges_never_share_an_id() {
    if !privileged("two_supervisors_with_disjoint_uid_ranges") {
        return;
    }
    let range = |base| SupervisorConfig {
        uid_base: base,
        uid_span: 8,
        max_workers: 4,
        ..config()
    };
    let mut first = Supervisor::new(range(2_100_000_000)).expect("first");
    let mut second = Supervisor::new(range(2_100_000_100)).expect("second");
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..3 {
        for (supervisor, base) in [(&mut first, 2_100_000_000u32), (&mut second, 2_100_000_100)] {
            let worker = spawn_in(supervisor, &request("idle")).expect("spawn");
            let uid = worker.uid.expect("a uid");
            assert!((base..base + 8).contains(&uid), "{uid} outside its range");
            seen.insert(uid);
            let again = supervisor.uid_of(worker.session).expect("session");
            assert_eq!(again, Some(uid));
        }
    }
    // The ids live at one time within one supervisor are all different.
    let live: std::collections::BTreeSet<u32> = first
        .sessions()
        .into_iter()
        .filter_map(|session| first.uid_of(session).ok().flatten())
        .collect();
    assert_eq!(live.len(), first.sessions().len());
}

#[test]
fn an_unsafe_uid_range_is_refused_at_start() {
    for base in [0u32, u32::MAX - 2] {
        let error = Supervisor::new(SupervisorConfig {
            uid_base: base,
            uid_span: 16,
            ..config()
        })
        .err()
        .expect("refused");
        assert!(matches!(error, SupervisorError::Config(_)), "{error}");
    }
    let error = Supervisor::new(SupervisorConfig {
        uid_span: 2,
        max_workers: 4,
        ..config()
    })
    .err()
    .expect("refused");
    assert!(matches!(error, SupervisorError::Config(_)), "{error}");
}

#[test]
fn a_channel_descriptor_below_three_is_refused() {
    let mut supervisor = Supervisor::new(config()).expect("supervisor");
    let mut bad = request("idle");
    bad.channel_fd = 2;
    bad.program = worker_program();
    assert!(matches!(
        supervisor.spawn(&bad),
        Err(SupervisorError::Config(_))
    ));
}
