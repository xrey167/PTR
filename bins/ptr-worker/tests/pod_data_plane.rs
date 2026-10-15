//! The Pod data plane of stage S1 with real processes: a supervisor (the `PodHost`), real
//! `ptr-worker` children under a user id, a cgroup, a network namespace and a seccomp
//! allowlist of their own, handles admitted by the runtime's `HandleTable`, tensors by sealed
//! `memfd` or by bytes. Linux only; needs the `data-plane` and `fault-injection` features.
//!
//! Tests that need root (own uid, cgroup v2, network namespace) detect what the host provides
//! and skip with a printed reason; under `CI=true` they fail instead, unless
//! `PTR_WORKER_UNPRIVILEGED_CI=1` says the runner is unprivileged on purpose. Every test that
//! moves a tensor prints which carrier ran (`PATH: HostShm` or `PATH: Bytes`).
#![cfg(all(
    feature = "data-plane",
    feature = "fault-injection",
    target_os = "linux"
))]

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ptr_fabric::{
    Availability, ConfinePolicy, ConfinementProbe, LostCause, Mechanism, SessionId,
    SupervisorConfig, SupervisorError,
};
use ptr_protocol::activation::{BoundaryDtype, ChainElement, UpstreamChain};
use ptr_protocol::{content_digest, Carrier, CarrierLocator, TensorHandle};
use ptr_runtime::{
    HandleRejection, OperationCheck, OperationFault, OperationLiveness, StageLiveness,
    WorkerSession,
};
use ptr_types::{Generation, ProjectId, TypeId, Validity};
use ptr_worker::host::{
    AdmitRequest, HostConfig, HostError, HostEvent, Placement, PodHost, ProduceSpec, Transfer,
};
use ptr_worker::wire::WorkerReply;

const OPERATION: u64 = 31;
const PAYLOAD: &str = "ptr.split.activation.v1";
const PRODUCER_STAGE: &str = "stage";
const CONSUMER_STAGE: &str = "consumer-stage";
const PAGE: u64 = 4096;
/// Three pages: the footprint is exactly this, so the charges can be checked to the byte.
const TENSOR_LEN: usize = 3 * PAGE as usize;
const NOW_MS: u64 = 1_000;

// --- the host's environment, faked ------------------------------------------------------

struct Live;

impl StageLiveness for Live {
    fn validity(&self, target: &str, generation: Generation) -> Option<Validity> {
        (matches!(target, "stage" | "model") && generation.0 == 1).then_some(Validity::Live)
    }

    fn manifest_digest(&self, target: &str, generation: Generation) -> Option<[u8; 32]> {
        (matches!(target, "stage" | "model") && generation.0 == 1).then_some([7; 32])
    }

    fn placement_epoch(&self, stage: &str) -> Option<u64> {
        (stage == "stage").then_some(5)
    }

    fn placement_epoch_in(&self, project: &ProjectId, stage: &str) -> Option<u64> {
        (project.0 == "p")
            .then(|| self.placement_epoch(stage))
            .flatten()
    }

    fn fence_epoch(&self, _expert: &str) -> Option<u64> {
        None
    }
}

/// The authority's operation record: every operation is live unless its producer session was
/// declared dead.
#[derive(Default)]
struct Operations {
    dead_producers: RefCell<BTreeSet<WorkerSession>>,
}

impl OperationLiveness for Operations {
    fn operation_admits(&self, check: OperationCheck<'_>) -> Result<(), OperationFault> {
        if self
            .dead_producers
            .borrow()
            .contains(&check.producer_session)
        {
            return Err(OperationFault::ProducerSessionMismatch {
                operation: check.parent_operation_id,
            });
        }
        Ok(())
    }
}

fn element(name: &str) -> ChainElement {
    ChainElement {
        identity: name.to_owned(),
        generation: 1,
        digest: [7; 32],
        fence_epoch: 0,
    }
}

fn tensor() -> Vec<u8> {
    (0..TENSOR_LEN)
        .map(|index| ((index * 7) % 251) as u8)
        .collect()
}

fn template(stage: &str, nonce: u64) -> TensorHandle {
    TensorHandle {
        carrier: Carrier::Bytes,
        locator: CarrierLocator::Bytes { inline: vec![0] },
        dtype: BoundaryDtype::F32,
        shape: vec![(TENSOR_LEN / 4) as u64],
        byte_len: TENSOR_LEN as u64,
        upstream_chain: UpstreamChain {
            stage: element(stage),
            base_model: element("model"),
            adapters: vec![],
            experts: vec![],
        },
        placement_epoch: 5,
        signed_absolute_deadline_ms: 5_000 + nonce,
        handle_nonce: nonce,
        parent_operation_id: OPERATION,
        consumer: CONSUMER_STAGE.to_owned(),
        payload_type: TypeId::from(PAYLOAD),
        content_digest: [0; 32],
        signature: [0; 64],
    }
}

// --- privilege and paths -----------------------------------------------------------------

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
        Availability::Available => None,
        Availability::Unavailable(why) => Some(format!("{name}: {why}")),
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

/// The worker binary copied to a directory of this process's own: the worker runs under a user
/// id of its own, and the target directory may not be readable by it.
///
/// The directory has a random name and mode 0711 (the worker's user can enter it but not list
/// it); the copy is written under a temporary name, synced and renamed, so no exec ever sees a
/// file that is still open for writing.
fn worker_program() -> PathBuf {
    static COPY: OnceLock<PathBuf> = OnceLock::new();
    COPY.get_or_init(|| {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        let source = PathBuf::from(env!("CARGO_BIN_EXE_ptr-worker"));
        remove_stale_copies();
        let mut random = [0u8; 8];
        ptr_fabric::fill_random(&mut random).expect("random");
        let directory = std::env::temp_dir().join(format!(
            "ptr-worker-data-plane-test-{}-{:016x}",
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

/// Directories earlier runs of this test left behind (a directory is named after the pid of the
/// process that made it): remove those whose process is gone. Errors are ignored; a directory
/// made by another user stays.
fn remove_stale_copies() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("ptr-worker-data-plane-test-") else {
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

/// Every test runs alone: the tests count descriptors, and a fork by one test while another has
/// just written the worker copy would hand the copy's write descriptor to a child that has not
/// exec'd yet ("Text file busy").
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn table(world: &World) -> &ptr_runtime::HandleTable {
    world.host.table(CONSUMER_STAGE).expect("table")
}

/// Spawn a worker; `ETXTBSY` from a fork outside this file's control is retried here, in the
/// test helper only. The library never swallows it.
fn spawn(host: &mut PodHost, placement: &Placement) -> Result<WorkerSession, HostError> {
    for attempt in 0..50 {
        match host.spawn_worker(placement) {
            Err(HostError::Supervisor(SupervisorError::Spawn(message)))
                if message.contains("Text file busy") && attempt < 49 =>
            {
                std::thread::sleep(Duration::from_millis(20));
            }
            other => return other,
        }
    }
    unreachable!("the loop returns on its last attempt")
}

fn open_descriptors() -> usize {
    std::fs::read_dir("/proc/self/fd").expect("proc fd").count()
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

/// A cgroup of the test's own to create worker groups in, so tests do not see each other's.
struct TestGroup(Option<PathBuf>);

impl TestGroup {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let number = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Only where this process may create groups at all (root, or a delegated group).
        let group = own_cgroup().and_then(|own| {
            let directory = own.join(format!("ptr-pod-test-{}-{number}", std::process::id()));
            std::fs::create_dir(&directory).ok().map(|()| directory)
        });
        Self(group)
    }

    fn path(&self) -> Option<&Path> {
        self.0.as_deref()
    }

    /// The worker groups still present under this one.
    fn leftovers(&self) -> Vec<PathBuf> {
        let Some(directory) = &self.0 else {
            return Vec::new();
        };
        std::fs::read_dir(directory)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path.is_dir())
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Drop for TestGroup {
    fn drop(&mut self) {
        if let Some(directory) = &self.0 {
            let _ = std::fs::remove_dir(directory);
        }
    }
}

fn wait_for<T>(what: &str, mut check: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(value) = check() {
            return value;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

// --- the world ---------------------------------------------------------------------------

struct World {
    host: PodHost,
    producer: WorkerSession,
    consumer: WorkerSession,
    operations: Operations,
    project: ProjectId,
    // Dropped last: the supervisor inside `host` kills the workers first, then the group goes.
    group: TestGroup,
}

fn host_config(group: &TestGroup, policy: ConfinePolicy) -> HostConfig {
    let mut config = HostConfig::new(worker_program());
    config.supervisor = SupervisorConfig {
        max_workers: 8,
        policy,
        cgroup_parent: group.path().map(Path::to_path_buf),
        reap_timeout: Duration::from_secs(15),
        ..SupervisorConfig::default()
    };
    config.key_timeout = Duration::from_secs(3);
    config.call_timeout = Duration::from_secs(15);
    config
}

fn placement(principal: &str, stage: &str) -> Placement {
    Placement {
        principal: principal.to_owned(),
        stage_identity: stage.to_owned(),
        seccomp: true,
        extra_args: Vec::new(),
    }
}

fn world(policy: ConfinePolicy) -> World {
    world_with(policy, &[])
}

/// A world whose producer worker is started with `producer_args` (for example `--fault ...`).
fn world_with(policy: ConfinePolicy, producer_args: &[&str]) -> World {
    let group = TestGroup::new();
    let mut host = PodHost::new(host_config(&group, policy)).expect("host");
    host.add_consumer_stage(CONSUMER_STAGE).expect("stage");
    let mut producer_placement = placement("producer-pod", PRODUCER_STAGE);
    producer_placement.extra_args = producer_args.iter().map(Into::into).collect();
    let producer = spawn(&mut host, &producer_placement).expect("producer");
    let consumer = spawn(&mut host, &placement("consumer-pod", CONSUMER_STAGE)).expect("consumer");
    World {
        host,
        producer,
        consumer,
        operations: Operations::default(),
        project: ProjectId::from("p"),
        group,
    }
}

impl World {
    fn produce(&mut self, nonce: u64) -> ptr_worker::host::Produced {
        let pending = self.start_produce(nonce);
        self.host.finish_produce(pending).expect("produced")
    }

    fn start_produce(&mut self, nonce: u64) -> ptr_worker::host::PendingProduce {
        self.host
            .start_produce(
                self.producer,
                self.consumer,
                ProduceSpec {
                    template: template(PRODUCER_STAGE, nonce),
                    tensor: tensor(),
                },
            )
            .expect("start produce")
    }

    fn admit(&mut self, produced: ptr_worker::host::Produced) -> Result<Transfer, HostError> {
        self.host.admit(AdmitRequest {
            produced,
            consumer: self.consumer,
            consumer_stage: CONSUMER_STAGE,
            now_ms: NOW_MS,
            operation: OPERATION,
            project: &self.project,
            live: &Live,
            operations: &self.operations,
        })
    }

    fn charged(&self) -> u64 {
        self.host.quota().pair_bytes(self.producer, self.consumer)
    }

    fn maps_of(&self, session: WorkerSession) -> String {
        let pid = self.host.pid(session).expect("pid");
        std::fs::read_to_string(format!("/proc/{pid}/maps")).unwrap_or_default()
    }
}

fn report_path(carrier: Carrier) {
    eprintln!("PATH: {carrier:?}");
}

// --- tests --------------------------------------------------------------------------------

#[test]
fn pods_run_over_the_channel_under_the_confinement_the_kernel_shows() {
    let _serial = serial();
    let group = TestGroup::new();
    let mut host = PodHost::new(host_config(&group, ConfinePolicy::BEST_EFFORT)).expect("host");
    let session = spawn(&mut host, &placement("pod", "stage")).expect("spawn");
    assert_eq!(
        host.call_pod(session, "sum", b"1 2 3").expect("call"),
        WorkerReply::Ok(b"6".to_vec())
    );
    assert_eq!(
        host.call_pod(session, "crc32", b"123456789").expect("call"),
        WorkerReply::Ok(vec![0xcb, 0xf4, 0x39, 0x26])
    );
    // The seccomp filter is in force (the host asked the kernel), whatever else is missing.
    let confinement = host.confinement(session).expect("confinement");
    assert!(
        !confinement.missing().contains(&Mechanism::Seccomp),
        "{confinement:?}"
    );
    eprintln!("CONFINEMENT: {confinement:?}");
    let outcome = host.reap(session).expect("reap");
    assert!(host.workers().is_empty());
    assert_eq!(outcome.released_pairs, 0);
}

#[test]
fn a_confined_worker_gets_eperm_for_socket_fork_and_the_rest() {
    let _serial = serial();
    let group = TestGroup::new();
    let mut host = PodHost::new(host_config(&group, ConfinePolicy::BEST_EFFORT)).expect("host");
    let session = spawn(&mut host, &placement("pod", "stage")).expect("spawn");
    let calls = host.probe(session).expect("probe");
    assert!(calls.len() >= 6, "{calls:?}");
    for (name, errno) in &calls {
        assert_eq!(
            *errno, 1,
            "{name} must fail with EPERM, got errno {errno}: {calls:?}"
        );
    }
    let names: Vec<&str> = calls.iter().map(|(name, _)| name.as_str()).collect();
    for expected in ["socket", "socketpair", "fork", "open", "io_uring_setup"] {
        assert!(names.contains(&expected), "{names:?}");
    }
    host.reap(session).expect("reap");
}

#[test]
fn a_handle_travels_by_the_best_carrier_and_consumed_frees_only_what_it_may() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let carrier = world
        .host
        .carrier_between(world.producer, world.consumer)
        .expect("carrier");
    let full = privileged_or_note("a_handle_travels_by_the_best_carrier");
    assert_eq!(
        carrier == Carrier::HostShm,
        full,
        "carrier {carrier:?} on a host with full={full}"
    );
    report_path(carrier);

    let produced = world.produce(1);
    assert_eq!(produced.handle.carrier, carrier);
    assert_eq!(produced.region.is_some(), carrier == Carrier::HostShm);
    let expected_digest = content_digest(&tensor());
    assert_eq!(produced.handle.content_digest, expected_digest);

    let transfer = world.admit(produced).expect("admitted");
    assert_eq!(
        transfer.footprint_bytes(),
        if carrier == Carrier::HostShm {
            3 * PAGE
        } else {
            TENSOR_LEN as u64
        }
    );
    // Charged from admission on, against both workers of the pair.
    assert_eq!(world.charged(), 3 * PAGE);
    assert_eq!(world.host.quota().host_bytes(), 3 * PAGE);

    let outcome = world
        .host
        .consume(transfer, NOW_MS + 1, &Live, &world.operations)
        .expect("consumed");
    assert_eq!(outcome.digest, expected_digest);
    assert_eq!(outcome.mapped_len, TENSOR_LEN as u64);
    if carrier == Carrier::HostShm {
        // `Consumed` freed nothing: the bytes are shared, so only the teardown releases them.
        assert!(outcome.reclaimed.is_none());
        assert_eq!(world.charged(), 3 * PAGE);
        assert_eq!(
            world
                .host
                .table(CONSUMER_STAGE)
                .and_then(|t| t.inflight_bytes("producer-pod")),
            Some(3 * PAGE)
        );
        // The consumer unmapped before it answered.
        assert!(
            !world.maps_of(world.consumer).contains("memfd:"),
            "{}",
            world.maps_of(world.consumer)
        );
        let outcomes = world.host.recycle(world.producer).expect("recycle");
        assert_eq!(
            outcomes.len(),
            2,
            "the producer and its counterpart are reaped"
        );
        let released: u64 = outcomes.iter().map(|outcome| outcome.released_bytes).sum();
        assert_eq!(released, 3 * PAGE);
        assert_eq!(
            outcomes
                .iter()
                .map(|outcome| outcome.released_pairs)
                .sum::<usize>(),
            1
        );
        assert!(world.group.leftovers().is_empty());
    } else {
        // A copy shares nothing: the notice frees the budget at once.
        assert!(outcome.reclaimed.is_some());
        assert_eq!(world.charged(), 0);
    }
    assert_eq!(world.host.quota().host_bytes(), 0);
    assert!(world.host.quota().tracked_pairs().is_empty());
}

/// Whether this host can run the shared carrier; says so when it cannot, without failing.
fn privileged_or_note(test: &str) -> bool {
    let probe = ConfinementProbe::detect();
    let full = probe.available() == ptr_fabric::AppliedMechanisms::ALL;
    if !full {
        eprintln!("NOTE {test}: no full confinement on this host, so the Bytes path runs");
        let strict = std::env::var("CI").as_deref() == Ok("true")
            && std::env::var("PTR_WORKER_UNPRIVILEGED_CI").as_deref() != Ok("1");
        assert!(!strict, "{test} needs root under CI=true");
    }
    full
}

#[test]
fn unconfined_workers_exchange_tensors_by_bytes_only() {
    let _serial = serial();
    // No uid, cgroup or namespace: the confinement is not Full, so no shared carrier.
    let mut world = world(ConfinePolicy::NONE);
    let carrier = world
        .host
        .carrier_between(world.producer, world.consumer)
        .expect("carrier");
    assert_eq!(carrier, Carrier::Bytes);
    report_path(carrier);
    let produced = world.produce(1);
    assert!(produced.region.is_none());
    let transfer = world.admit(produced).expect("admitted");
    let outcome = world
        .host
        .consume(transfer, NOW_MS + 1, &Live, &world.operations)
        .expect("consumed");
    assert_eq!(outcome.digest, content_digest(&tensor()));
    assert!(
        outcome.reclaimed.is_some(),
        "a copy is released by the notice"
    );
    assert_eq!(world.host.quota().host_bytes(), 0);
}

#[test]
fn a_producer_crash_mid_flight_fences_refuses_the_open_handle_and_releases_only_after_the_reap() {
    let _serial = serial();
    if !privileged("a_producer_crash_mid_flight") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    assert_eq!(
        world
            .host
            .carrier_between(world.producer, world.consumer)
            .expect("carrier"),
        Carrier::HostShm
    );
    report_path(Carrier::HostShm);
    let producer_group = world
        .host
        .supervisor()
        .cgroup_of(SessionId(world.producer.0))
        .expect("session")
        .expect("a cgroup")
        .to_path_buf();
    assert!(producer_group.is_dir());

    // Two exports in flight: the first is read and admitted, the second reply (with its
    // descriptor) stays queued on the channel, unread.
    let first = world.start_produce(1);
    let _second_in_flight = world.start_produce(2);
    let produced = world.host.finish_produce(first).expect("first");
    assert!(world
        .host
        .reply_waiting(world.producer, Duration::from_secs(15))
        .expect("wait"));
    let transfer = world.admit(produced).expect("admitted");
    assert_eq!(world.charged(), 3 * PAGE);

    // The producer dies before the consumer uses the handle.
    world
        .host
        .supervisor_mut()
        .kill_process(SessionId(world.producer.0))
        .expect("kill");
    let event = wait_for("the loss to be reported", || world.host.poll().pop());
    match event {
        HostEvent::Lost { session, cause } => {
            assert_eq!(session, world.producer);
            assert!(
                matches!(
                    cause,
                    LostCause::ChildExited(info) if info.signal == Some(9)
                ) || cause == LostCause::ChannelHungUp
            );
        }
    }
    assert!(world.host.is_fenced(world.producer));
    // The authority fences the producer's session too; the operation refuses it from now on.
    world
        .operations
        .dead_producers
        .borrow_mut()
        .insert(world.producer);

    // The open handle is refused: the descriptor is not forwarded to anyone.
    let refused = world
        .host
        .consume(transfer, NOW_MS + 1, &Live, &world.operations)
        .unwrap_err();
    assert!(
        matches!(
            refused,
            HostError::Rejected(HandleRejection::Abandoned | HandleRejection::UnknownTicket)
        ),
        "{refused:?}"
    );
    // Fencing recredits nothing: the buffer may still be referenced.
    assert_eq!(world.charged(), 3 * PAGE);
    assert!(
        producer_group.is_dir(),
        "the group is still there until the reap"
    );

    // Reap: group killed, child waited, group removed, old channel drained. The second reply's
    // descriptor was still queued and is closed by the drain.
    let first_reap = world.host.reap(world.producer).expect("reap the producer");
    assert!(first_reap.reaped.as_ref().expect("reaped").cgroup_removed());
    assert!(!producer_group.exists());
    assert!(
        first_reap
            .reaped
            .as_ref()
            .expect("reaped")
            .drain_report()
            .descriptors
            >= 1,
        "{:?}",
        first_reap
            .reaped
            .as_ref()
            .map(|reaped| reaped.drain_report())
    );
    // The consumer is still alive and may hold a reference: still charged, honestly.
    assert_eq!(first_reap.released_pairs, 0);
    assert_eq!(world.charged(), 3 * PAGE);

    // Recycling the pair reaps the consumer; only then is the ledger recredited.
    let outcomes = world.host.recycle(world.producer).expect("recycle");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].released_pairs, 1);
    assert_eq!(outcomes[0].released_bytes, 3 * PAGE);
    assert_eq!(world.host.quota().host_bytes(), 0);
    assert!(world.host.quota().tracked_pairs().is_empty());
    assert_eq!(
        world
            .host
            .table(CONSUMER_STAGE)
            .and_then(|table| table.inflight_bytes("producer-pod"))
            .unwrap_or(0),
        0
    );
    assert!(world.group.leftovers().is_empty());
}

#[test]
fn a_consumer_that_lies_about_consumed_keeps_its_mapping_and_the_bytes_stay_charged_until_reap() {
    let _serial = serial();
    if !privileged("a_consumer_that_lies_about_consumed") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    report_path(Carrier::HostShm);
    let produced = world.produce(1);
    let transfer = world.admit(produced).expect("admitted");
    let outcome = world
        .host
        .consume_retaining(transfer, NOW_MS + 1, &Live, &world.operations)
        .expect("consumed");
    // The consumer answered `Consumed` and verified the bytes...
    assert_eq!(outcome.digest, content_digest(&tensor()));
    assert!(outcome.reclaimed.is_none());
    // ...but still holds the mapping: the kernel shows it in the consumer's address space.
    let maps = world.maps_of(world.consumer);
    assert!(
        maps.contains("memfd:"),
        "the lying consumer must still map the region: {maps}"
    );
    // So the bytes stay charged: `Consumed` frees nothing.
    assert_eq!(world.charged(), 3 * PAGE);
    assert_eq!(
        world
            .host
            .table(CONSUMER_STAGE)
            .and_then(|t| t.inflight_bytes("producer-pod")),
        Some(3 * PAGE)
    );
    // Reaping the producer alone does not release them: the consumer may still read.
    let producer_reap = world.host.reap(world.producer).expect("reap producer");
    assert_eq!(producer_reap.released_pairs, 0);
    assert_eq!(world.charged(), 3 * PAGE);
    // Reaping the consumer, the holder of the mapping, completes the teardown.
    let consumer_reap = world.host.reap(world.consumer).expect("reap consumer");
    assert_eq!(consumer_reap.released_pairs, 1);
    assert_eq!(consumer_reap.released_bytes, 3 * PAGE);
    assert_eq!(world.host.quota().host_bytes(), 0);
}

#[test]
fn a_worker_cannot_sign_for_a_stage_it_was_not_given() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::NONE);
    let pending = world
        .host
        .start_produce(
            world.producer,
            world.consumer,
            ProduceSpec {
                template: template("another-stage", 1),
                tensor: tensor(),
            },
        )
        .expect("sent");
    match world.host.finish_produce(pending) {
        Err(HostError::Worker(code)) => assert_eq!(code, "stage-not-bound"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(world.host.quota().host_bytes(), 0);
}

#[test]
fn a_bad_challenge_signature_is_refused_and_the_worker_is_reaped() {
    let _serial = serial();
    let group = TestGroup::new();
    let mut host = PodHost::new(host_config(&group, ConfinePolicy::BEST_EFFORT)).expect("host");
    host.add_consumer_stage(CONSUMER_STAGE).expect("stage");
    let mut bad = placement("pod", "stage");
    bad.extra_args = vec!["--fault".into(), "bad-signature".into()];
    let error = spawn(&mut host, &bad).unwrap_err();
    assert!(
        matches!(error, HostError::KeyChallengeBadSignature),
        "{error:?}"
    );
    assert!(host.workers().is_empty());
    assert!(
        host.supervisor().sessions().is_empty(),
        "the worker was reaped"
    );
    assert!(group.leftovers().is_empty(), "its cgroup is gone");
    // Nothing was registered: a good worker of the same principal still spawns.
    spawn(&mut host, &placement("pod", "stage")).expect("a good worker");
}

#[test]
fn a_worker_that_never_answers_the_challenge_times_out_and_is_reaped() {
    let _serial = serial();
    let group = TestGroup::new();
    let mut config = host_config(&group, ConfinePolicy::BEST_EFFORT);
    config.key_timeout = Duration::from_millis(500);
    let mut host = PodHost::new(config).expect("host");
    let mut silent = placement("pod", "stage");
    silent.extra_args = vec!["--fault".into(), "silent".into()];
    let error = spawn(&mut host, &silent).unwrap_err();
    assert!(matches!(error, HostError::KeyChallengeTimeout), "{error:?}");
    assert!(host.supervisor().sessions().is_empty());
    assert!(group.leftovers().is_empty());
}

#[test]
fn a_fenced_consumer_gets_no_descriptor_and_the_handle_is_given_up() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::NONE);
    let produced = world.produce(1);
    let transfer = world.admit(produced).expect("admitted");
    world.host.fence(world.consumer);
    let error = world
        .host
        .consume(transfer, NOW_MS + 1, &Live, &world.operations)
        .unwrap_err();
    assert!(matches!(error, HostError::WorkerFenced(session) if session == world.consumer));
    // Nothing was consumed, and the producer is not fenced by it.
    assert!(!world.host.is_fenced(world.producer));
}

#[test]
fn admission_refuses_what_it_should_before_any_descriptor_moves() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::NONE);
    // A consumer stage nobody added.
    let produced = world.produce(1);
    let error = world
        .host
        .admit(AdmitRequest {
            produced,
            consumer: world.consumer,
            consumer_stage: "nobody",
            now_ms: NOW_MS,
            operation: OPERATION,
            project: &world.project,
            live: &Live,
            operations: &world.operations,
        })
        .unwrap_err();
    assert!(matches!(error, HostError::UnknownStage(_)), "{error:?}");
    // The wrong operation.
    let produced = world.produce(2);
    let error = world
        .host
        .admit(AdmitRequest {
            produced,
            consumer: world.consumer,
            consumer_stage: CONSUMER_STAGE,
            now_ms: NOW_MS,
            operation: OPERATION + 1,
            project: &world.project,
            live: &Live,
            operations: &world.operations,
        })
        .unwrap_err();
    assert!(
        matches!(
            error,
            HostError::Rejected(HandleRejection::WrongOperation { .. })
        ),
        "{error:?}"
    );
    assert_eq!(world.host.quota().host_bytes(), 0);
}

// --- the worker may not choose what the host decides ---------------------------------------

/// A producer started with `fault` answers a produce call dishonestly. The host must refuse,
/// close the descriptor, fence and reap the producer, and admit nothing.
fn a_dishonest_producer_is_refused(fault: &str, what: &str) {
    let mut world = world_with(ConfinePolicy::BEST_EFFORT, &["--fault", fault]);
    let before = open_descriptors();
    let pending = world.start_produce(1);
    let error = world.host.finish_produce(pending).unwrap_err();
    match error {
        HostError::Violation {
            worker,
            what: found,
        } => {
            assert_eq!(worker, world.producer);
            assert_eq!(found, what, "fault {fault}");
        }
        other => panic!("fault {fault}: expected a violation, got {other:?}"),
    }
    // Fenced and reaped: only the consumer is left, the producer's channel and the offered
    // descriptor are closed (one descriptor fewer than before, not the same or more).
    assert_eq!(world.host.workers(), vec![world.consumer], "fault {fault}");
    assert!(!world
        .host
        .supervisor()
        .is_registered(SessionId(world.producer.0)));
    assert!(
        open_descriptors() < before,
        "fault {fault}: {} descriptors before, {} after",
        before,
        open_descriptors()
    );
    assert_eq!(world.host.quota().host_bytes(), 0);
    assert!(
        world.group.leftovers().len() <= 1,
        "only the consumer's group is left"
    );
}

#[test]
fn a_producer_that_swaps_the_carrier_is_refused() {
    let _serial = serial();
    if privileged("a_producer_that_swaps_the_carrier") {
        a_dishonest_producer_is_refused("carrier-swap", "carrier-swapped");
    }
}

#[test]
fn a_producer_that_changes_a_field_the_authority_fixed_is_refused() {
    let _serial = serial();
    // On the Bytes path too: no privilege is needed to tamper.
    let mut world = world_with(ConfinePolicy::NONE, &["--fault", "tamper-field"]);
    let pending = world.start_produce(1);
    let error = world.host.finish_produce(pending).unwrap_err();
    assert!(
        matches!(
            error,
            HostError::Violation {
                what: "template-tampered",
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(world.host.workers(), vec![world.consumer]);
}

#[test]
fn a_producer_that_hands_over_a_region_that_is_not_a_sealed_exact_memfd_is_refused() {
    let _serial = serial();
    if privileged("a_producer_that_hands_over_a_bad_region") {
        a_dishonest_producer_is_refused("region-unsealed", "bad-region");
        a_dishonest_producer_is_refused("region-wrong-size", "bad-region");
        a_dishonest_producer_is_refused("region-not-memfd", "bad-region");
    }
}

// --- sessions, expiry and fencing without polling ------------------------------------------

#[test]
fn fencing_an_old_session_leaves_the_respawned_worker_registered() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::NONE);
    let old = world.producer;
    // The producer is respawned under a new session: same principal, same stage.
    let new = spawn(&mut world.host, &placement("producer-pod", PRODUCER_STAGE)).expect("respawn");
    assert_ne!(old, new);
    // Reaping the old session must not deactivate the new one.
    world.host.reap(old).expect("reap the old session");
    world.producer = new;
    let produced = world.produce(7);
    world
        .admit(produced)
        .expect("the respawned producer is still registered");
}

#[test]
fn abandoned_and_unused_handles_do_not_accumulate() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::NONE);
    let mut transfers = Vec::new();
    for nonce in 1..=5 {
        let produced = world.produce(nonce);
        transfers.push(world.admit(produced).expect("admitted"));
    }
    assert_eq!(table(&world).open_handles(), 5);
    assert_eq!(world.host.quota().host_bytes(), 5 * 3 * PAGE);
    // The consumer goes away: three handles are given up on the consume path, two are simply
    // never used.
    world.host.fence(world.consumer);
    for transfer in transfers.drain(..3) {
        let error = world
            .host
            .consume(transfer, NOW_MS + 1, &Live, &world.operations)
            .unwrap_err();
        assert!(matches!(error, HostError::WorkerFenced(_)), "{error:?}");
    }
    assert_eq!(
        table(&world).open_handles(),
        2,
        "the three given up are closed"
    );
    // Nothing is reclaimed before the deadline plus the skew...
    assert!(world.host.expire(NOW_MS + 1).is_empty());
    // ...and after it every claim comes back, the table is empty and the ledger is zero.
    let notices = world.host.expire(100_000);
    assert_eq!(notices.len(), 5);
    assert_eq!(table(&world).open_handles(), 0);
    assert_eq!(table(&world).slots_held("producer-pod").unwrap_or(0), 0);
    assert_eq!(world.host.quota().host_bytes(), 0);
    drop(transfers);
}

#[test]
fn a_killed_producer_is_fenced_by_admit_and_consume_without_anybody_polling() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::NONE);
    let produced = world.produce(1);
    let transfer = world.admit(produced).expect("admitted");
    let second = world.produce(2);
    world
        .host
        .supervisor_mut()
        .kill_process(SessionId(world.producer.0))
        .expect("kill");
    // Wait for the kernel to show the death on the channel, but never call `poll`, and never
    // tell the authority's operation record: the host must notice by itself.
    let channel_gone = || {
        world
            .host
            .supervisor()
            .channel(SessionId(world.producer.0))
            .expect("channel")
            .peer_hung_up()
    };
    wait_for("the producer's channel to hang up", || {
        channel_gone().then_some(())
    });
    assert!(
        !world.host.is_fenced(world.producer),
        "nobody has looked yet"
    );
    // admit looks first and refuses the second handle.
    let error = world.admit(second).unwrap_err();
    assert!(
        matches!(error, HostError::WorkerFenced(session) if session == world.producer),
        "{error:?}"
    );
    // consume of the first, already admitted, handle is refused with exactly this rejection.
    let error = world
        .host
        .consume(transfer, NOW_MS + 1, &Live, &world.operations)
        .unwrap_err();
    assert!(
        matches!(error, HostError::Rejected(HandleRejection::Abandoned)),
        "{error:?}"
    );
    assert!(world.operations.dead_producers.borrow().is_empty());
}
