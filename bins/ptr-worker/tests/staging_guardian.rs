//! The supervisor-owned CUDA staging lifecycle with real processes (doc 42 rules 3, 3a, 3b): a
//! real guardian process (`ptr-worker guardian --backend fake`, the CPU double of the CUDA
//! backend), real worker processes under a `PodHost`, the staging pool and the export registry.
//! Tensors between workers travel by `Bytes` in these tests (the CUDA carriers are not in the
//! host yet); what is exercised is the staging logic: fresh buffer per handle while the importer
//! lives, reuse only after it was reaped, free only after both domains were, export records
//! consumed once, the reap order between workers and guardian, and the crash of the guardian.
//!
//! The descriptors of the staging buffers are real: the guardian exports a `memfd`, the test
//! receives it through `SCM_RIGHTS` and reads and writes it like a producer and a consumer would.
//! The GPU form of the same lifecycle is `tests/staging_cuda.rs` of ptr-fabric (ignored here).
//!
//! Needs no privilege: where the host gives a cgroup, the guardian and the workers get one each.
#![cfg(all(feature = "data-plane", target_os = "linux"))]

use std::os::fd::OwnedFd;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;
#[cfg(feature = "fault-injection")]
use std::time::Instant;

use ptr_fabric::{
    host_page_size, AcquireRequest, ChargeBook, ConfinePolicy, ExportRecord, GuardianHostConfig,
    ImportClaim, Mechanism, PoolLimits, PoolRefusal, RegistryRefusal, SupervisorConfig,
    SupervisorError,
};
use ptr_runtime::WorkerSession;
use ptr_worker::host::{HostConfig, HostError, Placement, PodHost};
use ptr_worker::staging_host::{
    GuardianLoss, StagingError, StagingEvent, StagingHost, StagingHostConfig,
};

const OPERATION: u64 = 7;
const LANE: u64 = 20;

// --- environment -------------------------------------------------------------------------

fn worker_program() -> PathBuf {
    static COPY: OnceLock<PathBuf> = OnceLock::new();
    COPY.get_or_init(|| {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        let source = PathBuf::from(env!("CARGO_BIN_EXE_ptr-worker"));
        remove_stale_copies();
        let mut random = [0u8; 8];
        ptr_fabric::fill_random(&mut random).expect("random");
        let directory = std::env::temp_dir().join(format!(
            "ptr-worker-staging-test-{}-{:016x}",
            std::process::id(),
            u64::from_le_bytes(random)
        ));
        std::fs::DirBuilder::new()
            .mode(0o711)
            .create(&directory)
            .expect("a fresh test dir");
        let partial = directory.join("partial");
        std::fs::copy(&source, &partial).expect("copy the worker binary");
        std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o755)).expect("mode");
        std::fs::File::open(&partial)
            .and_then(|file| file.sync_all())
            .expect("sync");
        let target = directory.join("ptr-worker");
        std::fs::rename(&partial, &target).expect("publish the copy");
        target
    })
    .clone()
}

fn remove_stale_copies() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("ptr-worker-staging-test-") else {
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

/// Every test runs alone: a fork while another test just wrote the worker copy would hand the
/// copy's write descriptor to a child that has not exec'd yet ("Text file busy").
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn own_cgroup() -> Option<PathBuf> {
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

/// A cgroup of the test's own to create the guardian and worker groups in.
struct TestGroup(Option<PathBuf>);

impl TestGroup {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let number = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let group = own_cgroup().and_then(|own| {
            let directory = own.join(format!("ptr-stg-test-{}-{number}", std::process::id()));
            std::fs::create_dir(&directory).ok().map(|()| directory)
        });
        Self(group)
    }

    fn path(&self) -> Option<&Path> {
        self.0.as_deref()
    }
}

impl Drop for TestGroup {
    fn drop(&mut self) {
        if let Some(directory) = &self.0 {
            let _ = std::fs::remove_dir(directory);
        }
    }
}

fn supervisor_config(group: &TestGroup, max_workers: usize) -> SupervisorConfig {
    SupervisorConfig {
        max_workers,
        policy: ConfinePolicy::BEST_EFFORT,
        cgroup_parent: group.path().map(Path::to_path_buf),
        reap_timeout: Duration::from_secs(15),
        ..SupervisorConfig::default()
    }
}

struct World {
    // Dropped in this order: the host ends its workers first, then the staging host (and with
    // it the guardian) goes, then the group. A guardian must outlive the workers it backs.
    host: PodHost,
    staging: StagingHost,
    book: ChargeBook,
    _group: TestGroup,
}

fn world(limits: PoolLimits, max_export_records: usize) -> World {
    world_with_timeout(limits, max_export_records, Duration::from_secs(10))
}

fn world_with_timeout(
    limits: PoolLimits,
    max_export_records: usize,
    request_timeout: Duration,
) -> World {
    let group = TestGroup::new();
    let mut host_config = HostConfig::new(worker_program());
    host_config.supervisor = supervisor_config(&group, 8);
    host_config.key_timeout = Duration::from_secs(5);
    host_config.call_timeout = Duration::from_secs(15);
    let host = PodHost::new(host_config).expect("host");
    let mut guardian = GuardianHostConfig::new(worker_program())
        .arg("--backend")
        .arg("fake");
    guardian.supervisor = SupervisorConfig {
        policy: guardian.supervisor.policy,
        ..supervisor_config(&group, 1)
    };
    guardian.request_timeout = request_timeout;
    let book = ChargeBook::new();
    let staging = spawn_retrying(|| {
        StagingHost::spawn(
            &StagingHostConfig {
                guardian: guardian.clone(),
                limits: limits.clone(),
                max_export_records,
                // The tests also run on hosts without a cgroup or a namespace to give.
                require_isolation: false,
            },
            Box::new(book.clone()),
        )
    });
    World {
        host,
        staging,
        book,
        _group: group,
    }
}

/// A fork outside this file's control can make an exec fail with `ETXTBSY`; retry only here.
fn spawn_retrying<T>(mut attempt: impl FnMut() -> Result<T, StagingError>) -> T {
    for tries in 0..50 {
        match attempt() {
            Err(StagingError::Guardian(SupervisorError::Spawn(message)))
                if message.contains("Text file busy") && tries < 49 =>
            {
                std::thread::sleep(Duration::from_millis(20));
            }
            other => return other.expect("staging host"),
        }
    }
    unreachable!("the loop returns on its last attempt")
}

fn placement(principal: &str, stage: &str) -> Placement {
    Placement {
        principal: principal.to_owned(),
        stage_identity: stage.to_owned(),
        seccomp: true,
        extra_args: Vec::new(),
        device: None,
        staging: None,
        profile: ptr_fabric::SeccompProfile::Cpu,
    }
}

impl World {
    fn worker(&mut self, principal: &str) -> WorkerSession {
        for tries in 0..50 {
            match self.host.spawn_worker(&placement(principal, principal)) {
                Err(HostError::Supervisor(SupervisorError::Spawn(message)))
                    if message.contains("Text file busy") && tries < 49 =>
                {
                    std::thread::sleep(Duration::from_millis(20));
                }
                other => return other.expect("worker"),
            }
        }
        unreachable!("the loop returns on its last attempt")
    }

    fn request(
        &self,
        producer: WorkerSession,
        consumer: WorkerSession,
        len: u64,
    ) -> AcquireRequest {
        AcquireRequest {
            producer_session: producer.0,
            consumer_session: consumer.0,
            consumer_lane: LANE,
            operation: OPERATION,
            device: 0,
            len,
        }
    }

    /// Buffers the guardian process holds right now, counted from its descriptor table.
    fn guardian_buffers(&self) -> usize {
        let pid = self.staging.guardian().pid();
        std::fs::read_dir(format!("/proc/{pid}/fd"))
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter_map(|entry| std::fs::read_link(entry.path()).ok())
                    .filter(|target| target.to_string_lossy().contains("ptr-fabric-fake-staging"))
                    .count()
            })
            .unwrap_or(0)
    }
}

fn read_all(file: &std::fs::File, len: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; usize::try_from(len).expect("fits")];
    file.read_exact_at(&mut bytes, 0).expect("read the buffer");
    bytes
}

fn file_of(fd: OwnedFd) -> std::fs::File {
    std::fs::File::from(fd)
}

fn index_of(world: &World, wanted: impl Fn(&StagingEvent) -> bool) -> Option<usize> {
    world
        .staging
        .events()
        .iter()
        .position(|logged| wanted(&logged.event))
}

// --- tests ---------------------------------------------------------------------------------

#[test]
fn the_guardian_is_a_trusted_process_in_a_domain_of_its_own() {
    let _serial = serial();
    let mut world = world(PoolLimits::small_host(), 16);
    let worker = world.worker("producer-pod");
    let guardian = world.staging.guardian().spawned().clone();
    // Not seccomp-confined: it is trusted code that needs the driver's calls. Everything the
    // supervisor applies is still observed for it.
    let confinement = &world.staging.guardian().spawned().confinement;
    assert!(
        confinement.missing().contains(&Mechanism::Seccomp),
        "the guardian installs no seccomp filter: {confinement:?}"
    );
    assert!(guardian.observation.capabilities_dropped);
    assert!(guardian.observation.no_new_privs);
    if let Some(cgroup) = &guardian.cgroup {
        let workers_cgroup = world
            .host
            .supervisor()
            .cgroup_of(ptr_fabric::SessionId(worker.0))
            .expect("worker session");
        assert_ne!(
            Some(cgroup.as_path()),
            workers_cgroup,
            "a domain of its own"
        );
    }
    eprintln!("GUARDIAN: {guardian:?}");
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn buffers_are_fresh_while_the_consumer_lives_reused_after_it_is_reaped_and_freed_after_both() {
    let _serial = serial();
    let mut world = world(PoolLimits::small_host(), 16);
    let page = host_page_size();
    let producer = world.worker("producer-pod");
    let consumer = world.worker("consumer-pod-1");
    let len = 3000u64;

    // Two handles to a live consumer: two fresh buffers, each charged as allocated (a page) to
    // both sessions.
    let first = world
        .staging
        .issue_export(&world.request(producer, consumer, len), 10_000, 0)
        .expect("first");
    let second = world
        .staging
        .issue_export(&world.request(producer, consumer, len), 10_000, 0)
        .expect("second");
    assert!(!first.lease.reused && !second.lease.reused);
    assert_ne!(first.lease.slot, second.lease.slot);
    assert_eq!(first.lease.footprint, page);
    assert_eq!(world.guardian_buffers(), 2);
    assert_eq!(
        (
            world.book.charged(producer.0),
            world.book.charged(consumer.0)
        ),
        (2 * page, 2 * page)
    );

    // The producer writes through its descriptor; the consumer's descriptor shows the same
    // bytes: the buffer is shared memory held by the guardian.
    let producer_view = file_of(
        world
            .staging
            .export_descriptor(first.lease.slot, &first.lease.record)
            .expect("producer fd"),
    );
    let consumer_view = file_of(
        world
            .staging
            .export_descriptor(first.lease.slot, &first.lease.record)
            .expect("consumer fd"),
    );
    producer_view.write_all_at(&[0xAB; 3000], 0).expect("write");
    assert_eq!(read_all(&consumer_view, 3000), vec![0xAB; 3000]);

    // Admission: the record is consumed once; a claim by another session, operation or size
    // finds nothing, and neither does a locator that was never issued.
    let claim = ImportClaim {
        producer_session: producer.0,
        consumer_session: consumer.0,
        operation: OPERATION,
        device: 0,
        size: page,
    };
    for other in [
        ImportClaim {
            consumer_session: producer.0,
            ..claim
        },
        ImportClaim {
            producer_session: consumer.0,
            ..claim
        },
        ImportClaim {
            operation: OPERATION + 1,
            ..claim
        },
        ImportClaim {
            size: 2 * page,
            ..claim
        },
    ] {
        assert!(matches!(
            world.staging.admit_import(&other, 0),
            Err(StagingError::Registry(RegistryRefusal::NoRecord))
        ));
    }
    assert_eq!(
        world.staging.admit_import(&claim, 0).expect("first").slot,
        first.lease.slot
    );
    assert_eq!(
        world.staging.admit_import(&claim, 0).expect("second").slot,
        second.lease.slot
    );
    assert!(matches!(
        world.staging.admit_import(&claim, 0),
        Err(StagingError::Registry(RegistryRefusal::NoRecord))
    ));
    // A descriptor is not minted for a record the pool did not issue.
    let forged = ExportRecord {
        consumer_session: producer.0,
        ..first.lease.record
    };
    assert!(matches!(
        world.staging.export_descriptor(first.lease.slot, &forged),
        Err(StagingError::Pool(PoolRefusal::RecordMismatch { .. }))
    ));

    // The consumer is reaped: both buffers become reusable, none is freed, nothing is
    // recredited (the producer may still hold its import).
    world
        .staging
        .reap_worker(&mut world.host, consumer)
        .expect("reap consumer");
    assert!(matches!(
        world.staging.events().last().map(|logged| &logged.event),
        Some(StagingEvent::WorkerReaped {
            freed: 0,
            reusable: 2,
            ..
        })
    ));
    assert_eq!(world.guardian_buffers(), 2);
    assert_eq!(world.book.charged(producer.0), 2 * page);

    // A new consumer on the same lane gets the old buffers back, zeroed over the whole
    // footprint: the producer's 0xAB bytes are gone from the descriptor the old consumer held.
    let replacement = world.worker("consumer-pod-2");
    let third = world
        .staging
        .issue_export(&world.request(producer, replacement, len), 10_000, 0)
        .expect("third");
    assert!(third.lease.reused);
    assert_eq!(third.lease.slot, first.lease.slot);
    assert_eq!(read_all(&consumer_view, page), vec![0u8; page as usize]);
    let fourth = world
        .staging
        .issue_export(&world.request(producer, replacement, len), 10_000, 0)
        .expect("fourth");
    assert!(fourth.lease.reused);
    // A third handle to a live consumer gets a fresh one again.
    let fifth = world
        .staging
        .issue_export(&world.request(producer, replacement, len), 10_000, 0)
        .expect("fifth");
    assert!(!fifth.lease.reused);
    assert_eq!(world.guardian_buffers(), 3);
    // The charge moved from the reaped consumer to the new one.
    assert_eq!(world.book.charged(consumer.0), 0);
    assert_eq!(world.book.charged(replacement.0), 3 * page);

    // The producer is reaped while the consumer lives: nothing is freed.
    world
        .staging
        .reap_worker(&mut world.host, producer)
        .expect("reap producer");
    assert_eq!(world.guardian_buffers(), 3);
    assert_eq!(world.book.charged(replacement.0), 3 * page);
    // Only when the consumer is reaped as well do the buffers go, and with them every charge.
    world
        .staging
        .reap_worker(&mut world.host, replacement)
        .expect("reap replacement");
    assert_eq!(world.guardian_buffers(), 0);
    assert_eq!(world.book.total(), 0);
    assert_eq!(world.staging.pool().slot_count(), 0);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn shutdown_ends_every_worker_domain_before_the_guardian() {
    let _serial = serial();
    let mut world = world(PoolLimits::small_host(), 16);
    let producer = world.worker("producer-pod");
    let consumer = world.worker("consumer-pod");
    let other = world.worker("other-pod");
    world
        .staging
        .issue_export(&world.request(producer, consumer, 100), 10_000, 0)
        .expect("export");
    let guardian_pid = world.staging.guardian().pid();
    assert!(Path::new(&format!("/proc/{guardian_pid}")).exists());

    world.staging.shutdown(&mut world.host).expect("shutdown");

    assert!(world.host.workers().is_empty());
    assert!(world.staging.is_ended());
    let reaped: Vec<usize> = world
        .staging
        .events()
        .iter()
        .enumerate()
        .filter(|(_, logged)| matches!(logged.event, StagingEvent::WorkerReaped { .. }))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(reaped.len(), 3, "{:?}", world.staging.events());
    let guardian_reaped = index_of(&world, |event| {
        matches!(event, StagingEvent::GuardianReaped)
    })
    .expect("the guardian was reaped");
    assert!(
        reaped.iter().all(|index| *index < guardian_reaped),
        "every worker before the guardian: {:?}",
        world.staging.events()
    );
    // By timestamp too, and the guardian process existed at each worker's reap.
    let events = world.staging.events();
    let guardian_time = events[guardian_reaped].at;
    for index in &reaped {
        assert!(events[*index].at <= guardian_time);
        assert!(matches!(
            events[*index].event,
            StagingEvent::WorkerReaped {
                guardian_alive: true,
                ..
            }
        ));
    }
    let _ = (other, consumer);
    // The guardian process is gone, and nothing is charged.
    assert!(!Path::new(&format!("/proc/{guardian_pid}")).exists());
    assert_eq!(world.book.total(), 0);
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_dead_guardian_ends_every_worker_it_backed_and_nothing_else() {
    let _serial = serial();
    let mut world = world(PoolLimits::small_host(), 16);
    let producer = world.worker("producer-pod");
    let consumer = world.worker("consumer-pod");
    let bystander = world.worker("bystander-pod");
    world
        .staging
        .issue_export(&world.request(producer, consumer, 100), 10_000, 0)
        .expect("export");
    assert!(world.book.total() > 0);
    assert!(world
        .staging
        .watch(&mut world.host)
        .expect("watch")
        .is_empty());

    world
        .staging
        .kill_guardian_process()
        .expect("kill the guardian");
    let deadline = Instant::now() + Duration::from_secs(20);
    let events = loop {
        let events = world.staging.watch(&mut world.host).expect("watch");
        if !events.is_empty() {
            break events;
        }
        assert!(
            Instant::now() < deadline,
            "the loss of the guardian was not noticed"
        );
        std::thread::sleep(Duration::from_millis(5));
    };

    let StagingEvent::GuardianLost { terminated, .. } = &events[0] else {
        panic!("the first event is the loss: {events:?}");
    };
    let mut terminated = terminated.clone();
    terminated.sort_unstable();
    let mut expected = vec![producer.0, consumer.0];
    expected.sort_unstable();
    assert_eq!(terminated, expected, "{events:?}");
    assert!(matches!(
        events[1],
        StagingEvent::BuffersAbandoned {
            buffers: 1,
            records: 1,
            ..
        }
    ));
    assert_eq!(events.last(), Some(&StagingEvent::GuardianReaped));
    // The workers it backed are gone, the bystander runs, and nothing stays charged or issued.
    assert_eq!(world.host.workers(), vec![bystander]);
    assert_eq!(world.book.total(), 0);
    assert_eq!(world.staging.pool().slot_count(), 0);
    assert!(world.staging.registry().is_empty());
    assert!(world.staging.is_ended());
    assert!(matches!(
        world
            .staging
            .issue_export(&world.request(producer, consumer, 100), 10_000, 0),
        Err(StagingError::GuardianGone)
    ));
    assert_eq!(
        world
            .host
            .call_pod(bystander, "sum", b"1 2 3")
            .expect("the bystander still answers"),
        ptr_worker::wire::WorkerReply::Ok(b"6".to_vec())
    );
    world.staging.shutdown(&mut world.host).expect("shutdown");
    assert!(world.host.workers().is_empty());
}

#[test]
fn a_full_producer_quota_is_recovered_with_the_oldest_exports_first() {
    let _serial = serial();
    let page = host_page_size();
    let mut limits = PoolLimits::small_host();
    limits.max_producer_bytes = 2 * page;
    let mut world = world(limits, 16);
    let producer = world.worker("producer-pod");
    let first = world.worker("consumer-pod-1");
    let second = world.worker("consumer-pod-2");
    let third = world.worker("consumer-pod-3");
    // Different lanes, so each consumer has buffers of its own.
    let mut request = world.request(producer, first, 100);
    world
        .staging
        .issue_export(&request, 10_000, 0)
        .expect("to the first");
    request.consumer_session = second.0;
    request.consumer_lane = LANE + 1;
    world
        .staging
        .issue_export(&request, 10_000, 0)
        .expect("to the second");
    request.consumer_session = third.0;
    request.consumer_lane = LANE + 2;
    let refusal = world
        .staging
        .issue_export(&request, 10_000, 0)
        .expect_err("the producer is full");
    let StagingError::Pool(PoolRefusal::ProducerQuota {
        needed_release,
        producer_session,
        ..
    }) = refusal
    else {
        panic!("expected the producer quota, got {refusal:?}");
    };
    assert_eq!((producer_session, needed_release), (producer.0, page));
    // The refusal leased nothing.
    assert_eq!(world.guardian_buffers(), 2);

    let plan = world
        .staging
        .pool()
        .recovery_plan(producer.0, needed_release);
    assert_eq!(plan.consumers, vec![first.0], "the oldest export first");
    assert!(plan.covers);
    world
        .staging
        .recycle(&mut world.host, &plan)
        .expect("recycle");
    // The first consumer's buffer went with the pair; the second consumer was left running and
    // its buffer waits for it.
    assert_eq!(world.guardian_buffers(), 1);
    assert!(world.host.workers().contains(&second));
    assert!(world.host.workers().contains(&third));
    assert!(!world.host.workers().contains(&producer));
    assert_eq!(world.book.charged(first.0), 0);

    let replacement = world.worker("producer-pod-2");
    let mut request = world.request(replacement, third, 100);
    request.consumer_lane = LANE + 2;
    world
        .staging
        .issue_export(&request, 10_000, 0)
        .expect("a new producer has room");
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn the_export_registry_is_bounded_and_nothing_leaks_on_a_refusal() {
    let _serial = serial();
    let mut world = world(PoolLimits::small_host(), 1);
    let producer = world.worker("producer-pod");
    let consumer = world.worker("consumer-pod");
    world
        .staging
        .issue_export(&world.request(producer, consumer, 100), 10_000, 0)
        .expect("one record");
    let refusal = world
        .staging
        .issue_export(&world.request(producer, consumer, 100), 10_000, 0)
        .expect_err("the registry is full");
    assert!(matches!(
        refusal,
        StagingError::Registry(RegistryRefusal::Full { max: 1 })
    ));
    // No second buffer was leased for the refused handle.
    assert_eq!(world.guardian_buffers(), 1);
    assert_eq!(world.staging.pool().slot_count(), 1);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_failed_reap_after_a_guardian_loss_is_resumed_by_the_next_watch() {
    let _serial = serial();
    let mut world = world(PoolLimits::small_host(), 16);
    let producer = world.worker("producer-pod");
    let consumer = world.worker("consumer-pod");
    let bystander = world.worker("bystander-pod");
    world
        .staging
        .issue_export(&world.request(producer, consumer, 100), 10_000, 0)
        .expect("export");
    world.staging.kill_guardian_process().expect("kill");
    // The first worker reap that the loss needs fails once.
    world.staging.inject_reap_failures(1);
    let deadline = Instant::now() + Duration::from_secs(20);
    let error = loop {
        match world.staging.watch(&mut world.host) {
            Err(error) => break error,
            Ok(events) => assert!(events.is_empty(), "the loss finished without the fault"),
        }
        assert!(Instant::now() < deadline, "the loss was not noticed");
        std::thread::sleep(Duration::from_millis(5));
    };
    assert!(matches!(error, StagingError::Host(_)), "{error:?}");
    // The supervisor reports a loss once; the host remembers it. Nothing is half-done: the
    // backed workers still run, the guardian is not reaped, no new buffer is leased.
    assert!(world.staging.loss_pending());
    assert!(!world.staging.is_ended());
    assert!(world.host.workers().contains(&producer) && world.host.workers().contains(&consumer));
    assert!(matches!(
        world
            .staging
            .issue_export(&world.request(producer, consumer, 100), 10_000, 0),
        Err(StagingError::GuardianGone)
    ));
    // The next watch finishes the job.
    let events = world.staging.watch(&mut world.host).expect("resumed");
    let StagingEvent::GuardianLost { terminated, .. } = &events[0] else {
        panic!("the first event is the loss: {events:?}");
    };
    let mut terminated = terminated.clone();
    terminated.sort_unstable();
    let mut expected = vec![producer.0, consumer.0];
    expected.sort_unstable();
    assert_eq!(terminated, expected);
    assert_eq!(events.last(), Some(&StagingEvent::GuardianReaped));
    assert_eq!(world.host.workers(), vec![bystander]);
    assert!(!world.staging.loss_pending() && world.staging.is_ended());
    assert_eq!(world.book.total(), 0);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_shutdown_whose_worker_reap_fails_leaves_the_guardian_alive_and_resumes() {
    let _serial = serial();
    let mut world = world(PoolLimits::small_host(), 16);
    let producer = world.worker("producer-pod");
    let consumer = world.worker("consumer-pod");
    world
        .staging
        .issue_export(&world.request(producer, consumer, 100), 10_000, 0)
        .expect("export");
    let guardian_pid = world.staging.guardian().pid();
    world.staging.inject_reap_failures(1);
    assert!(world.staging.shutdown(&mut world.host).is_err());
    assert!(!world.staging.is_ended());
    assert!(Path::new(&format!("/proc/{guardian_pid}")).exists());
    assert_eq!(
        world.host.workers().len(),
        2,
        "the failed reap ended nobody"
    );
    // Again, now without the fault: workers first, guardian last.
    world.staging.shutdown(&mut world.host).expect("shutdown");
    assert!(world.staging.is_ended() && world.host.workers().is_empty());
    let last = world.staging.events().last().expect("events");
    assert_eq!(last.event, StagingEvent::GuardianReaped);
    assert!(!Path::new(&format!("/proc/{guardian_pid}")).exists());
}

#[test]
fn a_guardian_that_stops_answering_is_killed_and_handled_as_lost() {
    let _serial = serial();
    let mut world = world_with_timeout(PoolLimits::small_host(), 16, Duration::from_millis(300));
    let producer = world.worker("producer-pod");
    let consumer = world.worker("consumer-pod");
    world
        .staging
        .issue_export(&world.request(producer, consumer, 100), 10_000, 0)
        .expect("export");
    assert!(world
        .staging
        .watch(&mut world.host)
        .expect("watch")
        .is_empty());
    let pid = world.staging.guardian().pid();
    let status = std::process::Command::new("kill")
        .args(["-STOP", &pid.to_string()])
        .status()
        .expect("kill -STOP");
    assert!(status.success());
    // A request of the pool that times out breaks the stub (the ping of `watch` is the other
    // way, tested below).
    let refused = world
        .staging
        .issue_export(&world.request(producer, consumer, 100), 10_000, 0)
        .expect_err("the guardian does not answer");
    assert!(
        matches!(refused, StagingError::Pool(PoolRefusal::Backend(_))),
        "{refused:?}"
    );
    assert!(world.staging.pool().backend().is_broken());
    // Memory may still be held: nothing was recredited by the failed call.
    assert!(world.book.total() > 0);

    let events = world.staging.watch(&mut world.host).expect("the loss path");
    let StagingEvent::GuardianLost { cause, terminated } = &events[0] else {
        panic!("the first event is the loss: {events:?}");
    };
    assert_eq!(*cause, GuardianLoss::Unresponsive);
    assert_eq!(terminated.len(), 2);
    assert_eq!(events.last(), Some(&StagingEvent::GuardianReaped));
    assert!(world.host.workers().is_empty());
    assert_eq!(world.book.total(), 0);
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
}

#[test]
fn watch_pings_the_guardian_and_finds_a_stopped_one_without_a_failing_request() {
    let _serial = serial();
    let mut world = world_with_timeout(PoolLimits::small_host(), 16, Duration::from_millis(300));
    let producer = world.worker("producer-pod");
    let consumer = world.worker("consumer-pod");
    world
        .staging
        .issue_export(&world.request(producer, consumer, 100), 10_000, 0)
        .expect("export");
    // A live guardian answers the ping and nothing happens.
    assert!(world
        .staging
        .watch(&mut world.host)
        .expect("watch")
        .is_empty());
    assert!(!world.staging.pool().backend().is_broken());
    let pid = world.staging.guardian().pid();
    let status = std::process::Command::new("kill")
        .args(["-STOP", &pid.to_string()])
        .status()
        .expect("kill -STOP");
    assert!(status.success());
    // No request of the pool failed first: the ping of this very call did not get its answer
    // within the deadline, so the guardian is killed and the loss handled in the same call.
    let events = world.staging.watch(&mut world.host).expect("the loss path");
    let StagingEvent::GuardianLost { cause, terminated } = &events[0] else {
        panic!("the first event is the loss: {events:?}");
    };
    assert_eq!(*cause, GuardianLoss::Unresponsive);
    assert_eq!(terminated.len(), 2);
    assert_eq!(events.last(), Some(&StagingEvent::GuardianReaped));
    assert!(world.host.workers().is_empty());
    assert_eq!(world.book.total(), 0);
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
}

fn fd_targets(pid: u32) -> Vec<String> {
    std::fs::read_dir(format!("/proc/{pid}/fd"))
        .expect("the descriptor table")
        .filter_map(Result::ok)
        .filter_map(|entry| std::fs::read_link(entry.path()).ok())
        .map(|target| target.to_string_lossy().into_owned())
        .collect()
}

#[test]
fn the_descriptor_tables_hold_only_what_each_process_may_have() {
    let _serial = serial();
    let mut world = world(PoolLimits::small_host(), 16);
    let producer = world.worker("producer-pod");
    let consumer = world.worker("consumer-pod");
    for _ in 0..2 {
        world
            .staging
            .issue_export(&world.request(producer, consumer, 100), 10_000, 0)
            .expect("export");
    }
    // A worker has its standard streams on /dev/null and its one channel, no staging memfd and
    // no other socket: the guardian's channel is not reachable from a worker.
    for worker in [producer, consumer] {
        let pid = world.host.pid(worker).expect("pid");
        let targets = fd_targets(pid);
        let sockets = targets.iter().filter(|t| t.starts_with("socket:")).count();
        let others: Vec<_> = targets
            .iter()
            .filter(|t| !t.starts_with("socket:") && t.as_str() != "/dev/null")
            .collect();
        assert_eq!(
            sockets, 1,
            "a worker holds only its own channel: {targets:?}"
        );
        assert!(
            others.is_empty(),
            "unexpected descriptors in a worker: {targets:?}"
        );
    }
    // The guardian: the null streams, the one channel to the supervisor, and one memfd per
    // buffer. Nothing else (no listening socket, no file, no second socket).
    let guardian = fd_targets(world.staging.guardian().pid());
    let sockets = guardian.iter().filter(|t| t.starts_with("socket:")).count();
    let memfds = guardian
        .iter()
        .filter(|t| {
            t.starts_with("/memfd:ptr-fabric-fake-staging")
                || t.contains("memfd:ptr-fabric-fake-staging")
        })
        .count();
    let rest: Vec<_> = guardian
        .iter()
        .filter(|t| !t.starts_with("socket:") && !t.contains("memfd:") && t.as_str() != "/dev/null")
        .collect();
    assert_eq!(sockets, 1, "{guardian:?}");
    assert_eq!(memfds, 2, "{guardian:?}");
    assert!(
        rest.is_empty(),
        "unexpected descriptors in the guardian: {guardian:?}"
    );
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn two_producers_to_one_consumer_are_admitted_each_by_its_own_record() {
    let _serial = serial();
    let mut world = world(PoolLimits::small_host(), 16);
    let first = world.worker("producer-pod-1");
    let second = world.worker("producer-pod-2");
    let consumer = world.worker("consumer-pod");
    let page = host_page_size();
    // Fan-in: same consumer, operation, device and size from two producers.
    let a = world
        .staging
        .issue_export(&world.request(first, consumer, 100), 10_000, 0)
        .expect("a");
    let b = world
        .staging
        .issue_export(&world.request(second, consumer, 100), 10_000, 0)
        .expect("b");
    let claim = |producer: WorkerSession| ImportClaim {
        producer_session: producer.0,
        consumer_session: consumer.0,
        operation: OPERATION,
        device: 0,
        size: page,
    };
    assert_eq!(
        world
            .staging
            .admit_import(&claim(second), 0)
            .expect("b")
            .slot,
        b.lease.slot
    );
    assert_eq!(
        world
            .staging
            .admit_import(&claim(first), 0)
            .expect("a")
            .slot,
        a.lease.slot
    );
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

// --- the same, with the CUDA backend in the guardian (GPU only) -----------------------------

/// A guardian process that holds real VMM staging allocations: the supervisor leases a buffer
/// through the pool, the guardian fills and exports it, and this process imports and copies it
/// like a consumer would. Needs a build with `cuda-ipc` or `cuda-ipc-core` and an NVIDIA GPU.
#[cfg(feature = "cuda-ipc-core")]
#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn the_guardian_holds_vmm_allocations_on_a_gpu() {
    use ptr_fabric::{import_and_copy, CudaDriver, ImportExpectation};

    let _serial = serial();
    let group = TestGroup::new();
    let mut host_config = HostConfig::new(worker_program());
    host_config.supervisor = supervisor_config(&group, 8);
    let mut host = PodHost::new(host_config).expect("host");
    let mut guardian = GuardianHostConfig::new(worker_program())
        .arg("--backend")
        .arg("cuda");
    guardian.supervisor = SupervisorConfig {
        policy: guardian.supervisor.policy,
        ..supervisor_config(&group, 1)
    };
    let book = ChargeBook::new();
    let mut staging = StagingHost::spawn(
        &StagingHostConfig {
            guardian,
            limits: PoolLimits::small_host(),
            max_export_records: 16,
            require_isolation: false,
        },
        Box::new(book.clone()),
    )
    .expect("staging host");
    let driver = CudaDriver::open().expect("a CUDA driver in the test process");
    let producer = host
        .spawn_worker(&placement("producer-pod", "producer-pod"))
        .expect("producer");
    let consumer = host
        .spawn_worker(&placement("consumer-pod", "consumer-pod"))
        .expect("consumer");
    let tensor = vec![0x42u8; 30_000];
    let request = AcquireRequest {
        producer_session: producer.0,
        consumer_session: consumer.0,
        consumer_lane: LANE,
        operation: OPERATION,
        device: 0,
        len: tensor.len() as u64,
    };
    let issued = staging.issue_export(&request, 10_000, 0).expect("issue");
    staging.fill(issued.lease.slot, &tensor).expect("fill");
    let fd = staging
        .export_descriptor(issued.lease.slot, &issued.lease.record)
        .expect("export");
    let admitted = staging
        .admit_import(
            &ImportClaim {
                producer_session: producer.0,
                consumer_session: consumer.0,
                operation: OPERATION,
                device: 0,
                size: issued.lease.footprint,
            },
            0,
        )
        .expect("admitted");
    let expected = ImportExpectation {
        byte_len: tensor.len() as u64,
        content_digest: ptr_protocol::content_digest(&tensor),
        consumer_session: consumer.0,
        operation: OPERATION,
        source_device: 0,
        consumer_device: 0,
        locator_alloc_size: issued.lease.footprint,
    };
    let copy = import_and_copy(&driver, fd, &admitted.record, &expected).expect("import");
    assert_eq!(copy, tensor);
    staging.shutdown(&mut host).expect("shutdown");
    assert_eq!(book.total(), 0);
}
