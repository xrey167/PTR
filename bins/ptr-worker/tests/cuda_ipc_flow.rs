//! The `CudaIpc` carrier end to end between two real worker processes (doc 42 rules 1, 3, 3a and
//! 3b, stage S2 on a CPU): a `PodHost` with a `StagingHost` attached, a real guardian process
//! (`--backend fake`, memfd-backed), real workers started with `--staging fake`, the export
//! registry, the pool and the runtime's `HandleTable` with the host-wide ledger.
//!
//! What moves is real: the host leases a staging buffer, the producer worker writes the tensor
//! through the descriptor it is given, the signed handle carries a `CudaIpc` locator the host
//! chose, admission claims the supervisor's export record before the table sees the handle, and
//! the consumer worker copies the tensor out of a fresh descriptor and verifies the digest.
//! Nothing here has run on a GPU: the GPU form is `ptr-fabric`'s `staging_cuda` (ignored).
//!
//! `CudaIpc` needs full confinement (`carrier_permitted_for`), so the tests that move a handle by
//! it need root and skip with a printed reason where the host cannot give it (and fail under
//! `CI=true` unless `PTR_WORKER_UNPRIVILEGED_CI=1`). The fallback tests run everywhere.
#![cfg(all(
    feature = "data-plane",
    feature = "fault-injection",
    target_os = "linux"
))]

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::os::fd::OwnedFd;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ptr_fabric::{
    host_page_size, Availability, ChargeBook, ConfinePolicy, ConfinementProbe, ExportRecord,
    GuardianHostConfig, PoolLimits, PoolRefusal, RegistryRefusal, SeccompProfile, SessionId,
    SupervisorConfig, SupervisorError, MAX_FILL_BYTES,
};
use ptr_protocol::activation::{BoundaryDtype, ChainElement, UpstreamChain};
use ptr_protocol::{content_digest, Carrier, CarrierLocator, CudaExport, TensorHandle};
use ptr_runtime::{
    HandleRejection, OperationCheck, OperationFault, OperationLiveness, StageLiveness,
    WorkerSession,
};
use ptr_types::{Generation, ProjectId, TypeId, Validity};
use ptr_worker::guardian::BackendChoice;
use ptr_worker::host::{
    AdmitRequest, HostConfig, HostError, HostEvent, PendingProduce, Placement, PodHost,
    ProduceSpec, Produced, StagedContext, Transfer,
};
use ptr_worker::staging_host::{StagingError, StagingHost, StagingHostConfig};

const OPERATION: u64 = 31;
const PAYLOAD: &str = "ptr.split.activation.v1";
const PRODUCER_STAGE: &str = "stage";
const CONSUMER_STAGE: &str = "consumer-stage";
const NOW_MS: u64 = 1_000;
const DEVICE: u32 = 0;

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

/// The authority's operation record: every operation is live unless its producer was declared
/// dead.
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

/// A tensor of `len` bytes (a multiple of 4: it is shaped as `f32`).
fn tensor(len: usize, seed: usize) -> Vec<u8> {
    assert_eq!(len % 4, 0);
    (0..len)
        .map(|index| (((index + seed) * 7) % 251) as u8)
        .collect()
}

fn template(nonce: u64, len: usize) -> TensorHandle {
    TensorHandle {
        carrier: Carrier::Bytes,
        locator: CarrierLocator::Bytes { inline: vec![0] },
        dtype: BoundaryDtype::F32,
        shape: vec![(len / 4) as u64],
        byte_len: len as u64,
        upstream_chain: UpstreamChain {
            stage: element(PRODUCER_STAGE),
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

// --- privilege, paths, processes ----------------------------------------------------------

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

/// The worker binary copied to a directory of this process's own (mode 0711, random name,
/// written under a temporary name, synced and renamed), because the worker runs under a user id
/// of its own and the target directory may not be readable by it.
fn worker_program() -> PathBuf {
    static COPY: OnceLock<PathBuf> = OnceLock::new();
    COPY.get_or_init(|| {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        let source = PathBuf::from(env!("CARGO_BIN_EXE_ptr-worker"));
        remove_stale_copies();
        let mut random = [0u8; 8];
        ptr_fabric::fill_random(&mut random).expect("random");
        let directory = std::env::temp_dir().join(format!(
            "ptr-worker-cuda-ipc-test-{}-{:016x}",
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
        let Some(rest) = name.strip_prefix("ptr-worker-cuda-ipc-test-") else {
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
/// copy's write descriptor to a child that has not exec'd yet ("Text file busy"), and the tests
/// count descriptors.
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
            let directory = own.join(format!("ptr-ipc-test-{}-{number}", std::process::id()));
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

fn supervisor_config(
    group: &TestGroup,
    max_workers: usize,
    policy: ConfinePolicy,
) -> SupervisorConfig {
    SupervisorConfig {
        max_workers,
        policy,
        cgroup_parent: group.path().map(Path::to_path_buf),
        reap_timeout: Duration::from_secs(15),
        ..SupervisorConfig::default()
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

fn fd_targets(pid: u32) -> Vec<String> {
    std::fs::read_dir(format!("/proc/{pid}/fd"))
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter_map(|entry| std::fs::read_link(entry.path()).ok())
                .map(|target| target.to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

fn file_of(fd: OwnedFd) -> std::fs::File {
    std::fs::File::from(fd)
}

fn read_all(file: &std::fs::File, len: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; usize::try_from(len).expect("fits")];
    file.read_exact_at(&mut bytes, 0).expect("read the buffer");
    bytes
}

// --- the world ---------------------------------------------------------------------------

struct World {
    // Dropped in this order: the host ends its workers first, then the staging host (and with
    // it the guardian) goes, then the group. A guardian must outlive the workers it backs.
    host: PodHost,
    staging: StagingHost,
    book: ChargeBook,
    operations: Operations,
    project: ProjectId,
    _group: TestGroup,
}

fn world(policy: ConfinePolicy) -> World {
    let group = TestGroup::new();
    let mut host_config = HostConfig::new(worker_program());
    host_config.supervisor = supervisor_config(&group, 8, policy);
    host_config.key_timeout = Duration::from_secs(5);
    host_config.call_timeout = Duration::from_secs(15);
    let mut host = PodHost::new(host_config).expect("host");
    host.add_consumer_stage(CONSUMER_STAGE).expect("stage");
    let mut guardian = GuardianHostConfig::new(worker_program())
        .arg("--backend")
        .arg(BackendChoice::Fake.name());
    guardian.supervisor = SupervisorConfig {
        policy: guardian.supervisor.policy,
        ..supervisor_config(&group, 1, ConfinePolicy::BEST_EFFORT)
    };
    let book = ChargeBook::new();
    let staging = spawn_staging(&guardian, &book);
    World {
        host,
        staging,
        book,
        operations: Operations::default(),
        project: ProjectId::from("p"),
        _group: group,
    }
}

fn spawn_staging(guardian: &GuardianHostConfig, book: &ChargeBook) -> StagingHost {
    for tries in 0..50 {
        let result = StagingHost::spawn(
            &StagingHostConfig {
                guardian: guardian.clone(),
                limits: PoolLimits::small_host(),
                max_export_records: 16,
                // The tests also run on hosts without a cgroup or a namespace to give.
                require_isolation: false,
            },
            Box::new(book.clone()),
        );
        match result {
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

/// What a worker is started as.
struct Spec<'a> {
    principal: &'a str,
    stage: &'a str,
    device: Option<u32>,
    /// Serve the staging import of the guardian.
    staging: bool,
    profile: SeccompProfile,
    extra_args: &'a [&'a str],
}

impl<'a> Spec<'a> {
    fn producer(extra_args: &'a [&'a str]) -> Self {
        Self {
            principal: "producer-pod",
            stage: PRODUCER_STAGE,
            device: Some(DEVICE),
            staging: true,
            profile: SeccompProfile::Cpu,
            extra_args,
        }
    }

    fn consumer(principal: &'a str) -> Self {
        Self {
            principal,
            stage: CONSUMER_STAGE,
            device: Some(DEVICE),
            staging: true,
            profile: SeccompProfile::Cpu,
            extra_args: &[],
        }
    }
}

impl World {
    fn spawn(&mut self, spec: &Spec<'_>) -> WorkerSession {
        let placement = Placement {
            principal: spec.principal.to_owned(),
            stage_identity: spec.stage.to_owned(),
            seccomp: true,
            extra_args: spec.extra_args.iter().map(Into::into).collect(),
            device: spec.device,
            staging: spec.staging.then(|| self.staging.import()).flatten(),
            profile: spec.profile,
        };
        for tries in 0..50 {
            match self.host.spawn_worker(&placement) {
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

    /// Producer and consumer of the usual kind.
    fn pair(&mut self) -> (WorkerSession, WorkerSession) {
        let producer = self.spawn(&Spec::producer(&[]));
        let consumer = self.spawn(&Spec::consumer("consumer-pod"));
        (producer, consumer)
    }

    fn start_produce(
        &mut self,
        producer: WorkerSession,
        consumer: WorkerSession,
        nonce: u64,
        tensor: Vec<u8>,
    ) -> Result<PendingProduce, HostError> {
        let len = tensor.len();
        self.host.start_produce_staged(
            StagedContext {
                staging: &mut self.staging,
                now_ms: NOW_MS,
            },
            producer,
            consumer,
            ProduceSpec {
                template: template(nonce, len),
                tensor,
            },
        )
    }

    fn produce(
        &mut self,
        producer: WorkerSession,
        consumer: WorkerSession,
        nonce: u64,
        tensor: Vec<u8>,
    ) -> Produced {
        let pending = self
            .start_produce(producer, consumer, nonce, tensor)
            .expect("start produce");
        self.host.finish_produce(pending).expect("produced")
    }

    fn admit(
        &mut self,
        produced: Produced,
        consumer: WorkerSession,
    ) -> Result<Transfer, HostError> {
        self.host.admit_staged(
            &mut self.staging,
            AdmitRequest {
                produced,
                consumer,
                consumer_stage: CONSUMER_STAGE,
                now_ms: NOW_MS,
                operation: OPERATION,
                project: &self.project,
                live: &Live,
                operations: &self.operations,
            },
        )
    }

    fn consume(
        &mut self,
        transfer: Transfer,
    ) -> Result<ptr_worker::host::ConsumeOutcome, HostError> {
        self.host.consume_staged(
            &mut self.staging,
            transfer,
            NOW_MS + 1,
            &Live,
            &self.operations,
        )
    }

    fn reap(&mut self, session: WorkerSession) {
        self.staging
            .reap_worker(&mut self.host, session)
            .expect("reap");
    }

    /// Buffers the guardian process holds right now, counted from its descriptor table.
    fn guardian_buffers(&self) -> usize {
        fd_targets(self.staging.guardian().pid())
            .iter()
            .filter(|target| target.contains("ptr-fabric-fake-staging"))
            .count()
    }

    /// The host-wide ledger as `(producer, consumer, pair, host)` bytes.
    fn ledger(&self, producer: WorkerSession, consumer: WorkerSession) -> (u64, u64, u64, u64) {
        let quota = self.host.quota();
        (
            quota.producer_bytes(producer),
            quota.consumer_bytes(consumer),
            quota.pair_bytes(producer, consumer),
            quota.host_bytes(),
        )
    }

    /// A worker holds no staging buffer or tensor source open or mapped.
    fn assert_holds_nothing(&self, session: WorkerSession) {
        let pid = self.host.pid(session).expect("pid");
        for target in fd_targets(pid) {
            assert!(
                !target.contains("ptr-fabric-fake-staging")
                    && !target.contains("ptr-fabric-tensor"),
                "worker {session:?} still holds {target}"
            );
        }
        let maps = std::fs::read_to_string(format!("/proc/{pid}/maps")).unwrap_or_default();
        assert!(
            !maps.contains("ptr-fabric-fake-staging") && !maps.contains("ptr-fabric-tensor"),
            "worker {session:?} still maps a buffer:\n{maps}"
        );
    }
}

fn staging_refusal(error: &HostError) -> Option<&StagingError> {
    match error {
        HostError::Staging(inner) => Some(inner),
        _ => None,
    }
}

fn is_no_record(error: &HostError) -> bool {
    matches!(
        staging_refusal(error),
        Some(StagingError::Registry(RegistryRefusal::NoRecord))
    )
}

fn page() -> u64 {
    host_page_size()
}

// --- the transfer itself ------------------------------------------------------------------

/// One handle from a producer to a consumer through `CudaIpc`, checked at every step.
fn transfer_through_cuda_ipc(len: usize) {
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    assert_eq!(
        world
            .host
            .carrier_between_staged(producer, consumer, &world.staging)
            .expect("carrier"),
        Carrier::CudaIpc
    );
    eprintln!("PATH: CudaIpc");
    let data = tensor(len, 0);
    let expected_digest = content_digest(&data);
    let footprint = (len as u64).div_ceil(page()) * page();

    let produced = world.produce(producer, consumer, 1, data.clone());
    // The host chose the carrier and the locator; the worker signed exactly them.
    assert_eq!(produced.carrier, Carrier::CudaIpc);
    assert_eq!(produced.handle.carrier, Carrier::CudaIpc);
    assert!(produced.region.is_none(), "no descriptor comes back");
    assert_eq!(produced.handle.content_digest, expected_digest);
    assert_eq!(
        produced.handle.locator,
        CarrierLocator::CudaIpc {
            device: DEVICE,
            export: CudaExport::Vmm {
                fd_slot: 0,
                alloc_size: footprint
            },
            event_handle: [0; 64],
        }
    );
    let lease = produced.staged.expect("a lease").lease;
    assert_eq!(lease.footprint, footprint);
    assert!(!lease.reused);
    // The producer wrote the tensor through its descriptor: the buffer holds it, and zeros
    // after it, and the producer kept nothing open.
    let through = file_of(
        world
            .staging
            .export_descriptor(lease.slot, &lease.record)
            .expect("a descriptor"),
    );
    let buffer = read_all(&through, footprint);
    assert_eq!(&buffer[..len], &data[..]);
    assert!(buffer[len..].iter().all(|byte| *byte == 0));
    world.assert_holds_nothing(producer);

    let transfer = world.admit(produced, consumer).expect("admitted");
    assert_eq!(transfer.footprint_bytes(), footprint);
    let outcome = world.consume(transfer).expect("consumed");
    assert_eq!(outcome.digest, expected_digest);
    assert_eq!(outcome.mapped_len, len as u64);
    assert!(outcome.reclaimed.is_none(), "Consumed frees nothing");
    world.assert_holds_nothing(consumer);
    assert_eq!(world.guardian_buffers(), 1);

    world.staging.shutdown(&mut world.host).expect("shutdown");
    assert_eq!(world.book.total(), 0);
    assert_eq!(world.host.quota().host_bytes(), 0);
}

#[test]
fn a_cudaipc_handle_flows_between_two_real_workers_with_the_digest_checked() {
    let _serial = serial();
    if !privileged("a_cudaipc_handle_flows_between_two_real_workers") {
        return;
    }
    // Ten pages less a little: the padding is part of the footprint and stays zero.
    transfer_through_cuda_ipc(9 * page() as usize + 2_000);
}

#[test]
fn a_tensor_above_the_guardian_fill_limit_flows_because_the_producer_writes_through_the_buffer() {
    let _serial = serial();
    if !privileged("a_tensor_above_the_guardian_fill_limit_flows") {
        return;
    }
    let len = 1 << 20;
    assert!(
        len > MAX_FILL_BYTES,
        "the guardian could fill {MAX_FILL_BYTES} bytes at most in one call"
    );
    transfer_through_cuda_ipc(len);
}

// --- charging -----------------------------------------------------------------------------

/// Every staging byte is charged once against the producer, the consumer and the host, and given
/// back only after both workers were reaped, in either order.
fn exact_once_accounting(reap_producer_first: bool) {
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    assert_eq!(world.ledger(producer, consumer), (0, 0, 0, 0));
    assert_eq!(world.book.total(), 0);

    // 10 000 bytes in a 12 288 byte buffer: the charge is the allocated footprint.
    let produced = world.produce(producer, consumer, 1, tensor(10_000, 0));
    let footprint = produced.staged.expect("lease").lease.footprint;
    assert_eq!(footprint, 3 * page());
    // Leased, not admitted: the pool's book holds it for both sessions, the ledger nothing yet.
    assert_eq!(
        (
            world.book.charged(producer.0),
            world.book.charged(consumer.0)
        ),
        (footprint, footprint)
    );
    assert_eq!(world.ledger(producer, consumer), (0, 0, 0, 0));

    // Admission charges the ledger once: producer, consumer, pair and host each hold exactly
    // one footprint, not two (the host-wide figure is not the sum of the two sessions).
    let transfer = world.admit(produced, consumer).expect("admitted");
    assert_eq!(transfer.footprint_bytes(), footprint);
    assert_eq!(
        world.ledger(producer, consumer),
        (footprint, footprint, footprint, footprint)
    );
    assert_eq!(
        world
            .host
            .table(CONSUMER_STAGE)
            .and_then(|table| table.inflight_bytes("producer-pod")),
        Some(footprint)
    );
    // The pool's book did not change: each of the two books is charged once per buffer.
    assert_eq!(world.book.total(), 2 * footprint);

    // `Consumed` frees nothing.
    let outcome = world.consume(transfer).expect("consumed");
    assert!(outcome.reclaimed.is_none());
    assert_eq!(
        world.ledger(producer, consumer),
        (footprint, footprint, footprint, footprint)
    );
    assert_eq!(world.book.total(), 2 * footprint);
    assert_eq!(world.guardian_buffers(), 1);

    let (first, second) = if reap_producer_first {
        (producer, consumer)
    } else {
        (consumer, producer)
    };
    // One side reaped: the other may still hold the buffer, so nothing is given back anywhere.
    world.reap(first);
    assert_eq!(
        world.ledger(producer, consumer),
        (footprint, footprint, footprint, footprint)
    );
    assert_eq!(world.book.total(), 2 * footprint);
    assert_eq!(world.guardian_buffers(), 1);
    assert_eq!(world.host.quota().tracked_pairs().len(), 1);

    // Both reaped: the pair's charge and the pool's are released, and the guardian freed the
    // buffer.
    world.reap(second);
    assert_eq!(world.ledger(producer, consumer), (0, 0, 0, 0));
    assert!(world.host.quota().tracked_pairs().is_empty());
    assert_eq!(world.book.total(), 0);
    assert_eq!(world.guardian_buffers(), 0);
    assert_eq!(world.staging.pool().slot_count(), 0);
    assert_eq!(world.host.quota().usage().release_faults, 0);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn staging_bytes_are_charged_once_and_released_only_after_both_reaps_producer_first() {
    let _serial = serial();
    if !privileged("staging_bytes_are_charged_once_producer_first") {
        return;
    }
    exact_once_accounting(true);
}

#[test]
fn staging_bytes_are_charged_once_and_released_only_after_both_reaps_consumer_first() {
    let _serial = serial();
    if !privileged("staging_bytes_are_charged_once_consumer_first") {
        return;
    }
    exact_once_accounting(false);
}

#[test]
fn a_handle_the_ledger_refuses_leaves_its_buffer_charged_to_the_pool_until_the_reaps() {
    let _serial = serial();
    if !privileged("a_handle_the_ledger_refuses") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    // The default host quota is 16 MiB per session: a 17 MiB footprint fits the pool (1 GiB per
    // session) but not the ledger, so admission is refused by the table.
    let produced = world.produce(producer, consumer, 1, tensor(15 << 20, 0));
    let footprint = produced.staged.expect("lease").lease.footprint;
    let second = world.produce(producer, consumer, 2, tensor(2 << 20, 1));
    assert!(world.admit(produced, consumer).is_ok());
    let refused = world.admit(second, consumer).unwrap_err();
    assert!(
        matches!(refused, HostError::Rejected(HandleRejection::HostQuota(_))),
        "the host-wide ledger refuses: {refused:?}"
    );
    // The first handle is charged once; the refused one left the ledger untouched, and its
    // record is spent, so it cannot be admitted again.
    assert_eq!(world.host.quota().pair_bytes(producer, consumer), footprint);
    // The pool still holds both buffers for both sessions until the reaps.
    assert_eq!(world.staging.pool().slot_count(), 2);
    assert!(world.book.total() > 2 * footprint);
    world.staging.shutdown(&mut world.host).expect("shutdown");
    assert_eq!(world.book.total(), 0);
    assert_eq!(world.host.quota().host_bytes(), 0);
}

// --- refusals -----------------------------------------------------------------------------

#[test]
fn a_signed_locator_with_another_size_or_device_than_the_hosts_is_a_violation() {
    let _serial = serial();
    if !privileged("a_signed_locator_with_another_size_or_device") {
        return;
    }
    for fault in ["staging-wrong-size", "staging-wrong-device"] {
        let mut world = world(ConfinePolicy::BEST_EFFORT);
        let producer = world.spawn(&Spec::producer(&["--fault", fault]));
        let consumer = world.spawn(&Spec::consumer("consumer-pod"));
        let pending = world
            .start_produce(producer, consumer, 1, tensor(8_000, 0))
            .expect("start");
        let error = world.host.finish_produce(pending).unwrap_err();
        assert!(
            matches!(
                error,
                HostError::Violation {
                    what: "template-tampered",
                    ..
                }
            ),
            "{fault}: {error:?}"
        );
        // The host fenced and reaped the producer on its own: the pool and the registry learn
        // of it from the host, not from a call of the staging host.
        assert!(!world.host.is_registered(producer));
        if fault == "staging-wrong-size" {
            assert_eq!(world.staging.sync_reaped(&world.host), vec![producer.0]);
        } else {
            // `watch` runs the same sync (a direct host reap leaves the pool stale until then).
            assert!(world
                .staging
                .watch(&mut world.host)
                .expect("watch")
                .is_empty());
        }
        assert!(world.staging.registry().is_empty());
        assert_eq!(world.ledger(producer, consumer), (0, 0, 0, 0));
        // The buffer waits for the consumer's reap (rule 3b), then goes.
        assert_eq!(world.guardian_buffers(), 1);
        world.reap(consumer);
        assert_eq!(world.guardian_buffers(), 0);
        assert_eq!(world.book.total(), 0);
        world.staging.shutdown(&mut world.host).expect("shutdown");
    }
}

#[test]
fn a_forged_locator_finds_no_record_and_the_table_is_never_reached() {
    let _serial = serial();
    if !privileged("a_forged_locator_finds_no_record") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    let produced = world.produce(producer, consumer, 1, tensor(8_000, 0));
    let issued = produced.staged;
    let honest = produced.handle.clone();
    for what in ["alloc_size", "device"] {
        let mut handle = honest.clone();
        if let CarrierLocator::CudaIpc {
            device,
            export: CudaExport::Vmm { alloc_size, .. },
            ..
        } = &mut handle.locator
        {
            if what == "alloc_size" {
                *alloc_size += 4096;
            } else {
                *device += 1;
            }
        }
        let forged = Produced {
            producer: produced.producer,
            carrier: Carrier::CudaIpc,
            encoded: handle.encode().expect("a well-formed handle"),
            handle,
            region: None,
            staged: issued,
        };
        let error = world.admit(forged, consumer).unwrap_err();
        assert!(is_no_record(&error), "{what}: {error:?}");
        // Neither the ledger nor the table saw it.
        assert_eq!(world.ledger(producer, consumer), (0, 0, 0, 0));
        assert_eq!(
            world
                .host
                .table(CONSUMER_STAGE)
                .and_then(|table| table.inflight_bytes("producer-pod")),
            Some(0)
        );
    }
    // The refusals consumed nothing: the honest handle is still admitted.
    let transfer = world.admit(produced, consumer).expect("the honest handle");
    world.consume(transfer).expect("consumed");
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn a_claim_for_another_consumer_or_producer_is_refused_and_gets_no_descriptor() {
    let _serial = serial();
    if !privileged("a_claim_for_another_consumer_or_producer") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    let other = world.spawn(&Spec::consumer("other-pod"));
    let produced = world.produce(producer, consumer, 1, tensor(8_000, 0));
    let lease = produced.staged.expect("lease").lease;

    // Presented by another consumer session.
    let by_other = replay_of(&produced);
    let error = world.admit(by_other, other).unwrap_err();
    assert!(is_no_record(&error), "{error:?}");
    // Claimed as the work of another producer session.
    let as_other = Produced {
        producer: other,
        carrier: produced.carrier,
        handle: produced.handle.clone(),
        encoded: produced.encoded.clone(),
        region: None,
        staged: produced.staged,
    };
    let error = world.admit(as_other, consumer).unwrap_err();
    assert!(is_no_record(&error), "{error:?}");
    // The pool gives no descriptor of the buffer to a session the record does not name.
    let for_other = ExportRecord {
        consumer_session: other.0,
        ..lease.record
    };
    let refused = world
        .staging
        .export_descriptor(lease.slot, &for_other)
        .unwrap_err();
    assert!(
        matches!(
            refused,
            StagingError::Pool(PoolRefusal::RecordMismatch {
                field: "consumer_session"
            })
        ),
        "{refused:?}"
    );
    assert_eq!(world.ledger(producer, consumer), (0, 0, 0, 0));
    // The addressed consumer is unaffected.
    let transfer = world.admit(produced, consumer).expect("admitted");
    let outcome = world.consume(transfer).expect("consumed");
    assert_eq!(outcome.digest, content_digest(&tensor(8_000, 0)));
    world.assert_holds_nothing(other);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn a_handle_admitted_once_cannot_be_admitted_again() {
    let _serial = serial();
    if !privileged("a_handle_admitted_once_cannot_be_admitted_again") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    let produced = world.produce(producer, consumer, 1, tensor(8_000, 0));
    let again = replay_of(&produced);
    let transfer = world.admit(produced, consumer).expect("admitted");
    // While it is open: the record is spent.
    let error = world.admit(replay_of(&again), consumer).unwrap_err();
    assert!(is_no_record(&error), "{error:?}");
    world.consume(transfer).expect("consumed");
    // After it was consumed: the same.
    let error = world.admit(again, consumer).unwrap_err();
    assert!(is_no_record(&error), "{error:?}");
    // The ledger holds the one admission.
    let footprint = 2 * page();
    assert_eq!(world.host.quota().pair_bytes(producer, consumer), footprint);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

fn replay_of(produced: &Produced) -> Produced {
    Produced {
        producer: produced.producer,
        carrier: produced.carrier,
        handle: produced.handle.clone(),
        encoded: produced.encoded.clone(),
        region: None,
        staged: produced.staged,
    }
}

#[test]
fn the_unstaged_calls_refuse_a_cudaipc_handle() {
    let _serial = serial();
    if !privileged("the_unstaged_calls_refuse_a_cudaipc_handle") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    let produced = world.produce(producer, consumer, 1, tensor(8_000, 0));
    let error = world
        .host
        .admit(AdmitRequest {
            produced,
            consumer,
            consumer_stage: CONSUMER_STAGE,
            now_ms: NOW_MS,
            operation: OPERATION,
            project: &world.project,
            live: &Live,
            operations: &world.operations,
        })
        .unwrap_err();
    assert!(matches!(error, HostError::StagingRequired), "{error:?}");
    // A transfer admitted through the staged call cannot be consumed without it either, and
    // the open handle is given up rather than left open.
    let produced = world.produce(producer, consumer, 2, tensor(20_000, 0));
    let transfer = world.admit(produced, consumer).expect("admitted");
    let error = world
        .host
        .consume(transfer, NOW_MS + 1, &Live, &world.operations)
        .unwrap_err();
    assert!(matches!(error, HostError::StagingRequired), "{error:?}");
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn a_consumer_refuses_a_buffer_whose_bytes_are_not_the_signed_digest() {
    let _serial = serial();
    if !privileged("a_consumer_refuses_a_buffer_whose_bytes") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let producer = world.spawn(&Spec::producer(&["--fault", "staging-corrupt"]));
    let consumer = world.spawn(&Spec::consumer("consumer-pod"));
    let produced = world.produce(producer, consumer, 1, tensor(8_000, 0));
    let transfer = world.admit(produced, consumer).expect("admitted");
    let error = world.consume(transfer).unwrap_err();
    assert!(
        matches!(&error, HostError::Worker(code) if code == "staging-digest-mismatch"),
        "{error:?}"
    );
    // The consumer kept nothing and is not blamed for the producer's bytes.
    world.assert_holds_nothing(consumer);
    assert!(!world.host.is_fenced(consumer));
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

// --- reuse and teardown -------------------------------------------------------------------

#[test]
fn a_buffer_is_reused_only_after_its_consumer_was_reaped_and_only_zeroed() {
    let _serial = serial();
    if !privileged("a_buffer_is_reused_only_after_its_consumer_was_reaped") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    // Both tensors need ten pages: the footprint is the same, so reuse is possible.
    let (long, short) = (tensor(40_000, 0), tensor(37_000, 3));
    let footprint = 10 * page();

    let first = world.produce(producer, consumer, 1, long.clone());
    let lease1 = first.staged.expect("lease").lease;
    assert_eq!(lease1.footprint, footprint);
    let old = file_of(
        world
            .staging
            .export_descriptor(lease1.slot, &lease1.record)
            .expect("a descriptor"),
    );
    let transfer = world.admit(first, consumer).expect("admitted");
    world.consume(transfer).expect("consumed");
    assert_eq!(&read_all(&old, footprint)[..40_000], &long[..]);

    // The consumer lives and said `Consumed`: the next handle gets a fresh buffer.
    let second = world.produce(producer, consumer, 2, long.clone());
    let lease2 = second.staged.expect("lease").lease;
    assert!(!lease2.reused);
    assert_ne!(lease2.slot, lease1.slot);
    assert_eq!(world.staging.pool().slot_count(), 2);

    // The consumer is reaped. Nothing is released (the producer lives), both buffers become
    // reusable, and the ledger still holds the admitted pair.
    world.reap(consumer);
    for lease in [lease1, lease2] {
        let info = world.staging.pool().slot(lease.slot).expect("slot");
        assert!(info.consumer_reaped && !info.producer_reaped);
    }
    assert_eq!(world.staging.pool().slot_count(), 2);
    assert_eq!(world.guardian_buffers(), 2);
    assert_eq!(world.host.quota().pair_bytes(producer, consumer), footprint);

    // A new consumer session on the same stage (the same lane) gets the oldest buffer back,
    // and the shorter tensor leaves none of the longer one's tail behind.
    let consumer_b = world.spawn(&Spec::consumer("consumer-pod-b"));
    let third = world.produce(producer, consumer_b, 3, short.clone());
    let lease3 = third.staged.expect("lease").lease;
    assert!(lease3.reused);
    assert_eq!(lease3.slot, lease1.slot);
    let reused = read_all(&old, footprint);
    assert_eq!(
        &reused[..37_000],
        &short[..],
        "the new tensor, through the old descriptor"
    );
    assert!(
        reused[37_000..].iter().all(|byte| *byte == 0),
        "the tail of the previous tensor was scrubbed"
    );
    // The charge moved with the buffer: the old consumer keeps only the other buffer's.
    assert_eq!(world.book.charged(producer.0), 2 * footprint);
    assert_eq!(world.book.charged(consumer.0), footprint);
    assert_eq!(world.book.charged(consumer_b.0), footprint);
    let transfer = world.admit(third, consumer_b).expect("admitted");
    let outcome = world.consume(transfer).expect("consumed");
    assert_eq!(outcome.digest, content_digest(&short));

    world.staging.shutdown(&mut world.host).expect("shutdown");
    assert_eq!(world.book.total(), 0);
    assert_eq!(world.host.quota().host_bytes(), 0);
    assert_eq!(world.guardian_buffers(), 0);
}

#[test]
fn a_buffer_reused_for_a_new_consumer_is_charged_to_the_ledger_again_until_the_producer_is_reaped()
{
    let _serial = serial();
    if !privileged("a_buffer_reused_for_a_new_consumer_is_charged_again") {
        return;
    }
    // The documented limit of the one-charge design: the ledger charges pairs and cannot move
    // a charge between them, so a reused buffer counts once per admitted pair. It over-counts,
    // never under-counts, and everything is released with the producer's reap.
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    let footprint = 2 * page();
    let first = world.produce(producer, consumer, 1, tensor(8_000, 0));
    let slot = first.staged.expect("lease").lease.slot;
    let transfer = world.admit(first, consumer).expect("admitted");
    world.consume(transfer).expect("consumed");
    world.reap(consumer);
    let consumer_b = world.spawn(&Spec::consumer("consumer-pod-b"));
    let second = world.produce(producer, consumer_b, 2, tensor(8_000, 1));
    assert_eq!(second.staged.expect("lease").lease.slot, slot);
    let transfer = world.admit(second, consumer_b).expect("admitted");
    world.consume(transfer).expect("consumed");
    // One buffer in the pool; two pair charges in the ledger.
    assert_eq!(world.staging.pool().slot_count(), 1);
    assert_eq!(world.host.quota().producer_bytes(producer), 2 * footprint);
    assert_eq!(world.host.quota().host_bytes(), 2 * footprint);
    assert_eq!(world.book.charged(producer.0), footprint);
    world.reap(producer);
    world.reap(consumer_b);
    assert_eq!(world.host.quota().host_bytes(), 0);
    assert!(world.host.quota().tracked_pairs().is_empty());
    assert_eq!(world.book.total(), 0);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn a_producer_that_crashes_after_producing_is_fenced_its_handle_refused_and_the_buffer_freed_after_both_reaps(
) {
    let _serial = serial();
    if !privileged("a_producer_that_crashes_after_producing") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    let produced = world.produce(producer, consumer, 1, tensor(8_000, 0));
    let footprint = produced.staged.expect("lease").lease.footprint;

    world
        .host
        .supervisor_mut()
        .kill_process(SessionId(producer.0))
        .expect("kill");
    let event = wait_for("the loss to be reported", || world.host.poll().pop());
    match event {
        HostEvent::Lost { session, .. } => assert_eq!(session, producer),
    }
    // The handle of a fenced producer is refused before the registry or the table is touched.
    let error = world.admit(produced, consumer).unwrap_err();
    assert!(
        matches!(error, HostError::WorkerFenced(s) if s == producer),
        "{error:?}"
    );
    assert_eq!(world.ledger(producer, consumer), (0, 0, 0, 0));
    assert_eq!(
        world.staging.registry().len(),
        1,
        "the record is still there"
    );

    // Reaping the producer drops its records but not the buffer: the consumer may import it.
    world.reap(producer);
    assert!(world.staging.registry().is_empty());
    assert_eq!(world.guardian_buffers(), 1);
    assert_eq!(world.book.total(), 2 * footprint);
    world.reap(consumer);
    assert_eq!(world.guardian_buffers(), 0);
    assert_eq!(world.book.total(), 0);
    assert_eq!(world.staging.pool().slot_count(), 0);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn a_producer_killed_while_the_produce_call_is_in_flight_leaks_nothing() {
    let _serial = serial();
    if !privileged("a_producer_killed_while_the_produce_call_is_in_flight") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    let pending = world
        .start_produce(producer, consumer, 1, tensor(1 << 20, 0))
        .expect("start");
    world
        .host
        .supervisor_mut()
        .kill_process(SessionId(producer.0))
        .expect("kill");
    // Whether the worker answered before it died is a race; either way nothing may leak.
    let finished = world.host.finish_produce(pending);
    wait_for("the loss to be reported", || {
        (!world.host.is_registered(producer) || world.host.is_fenced(producer))
            .then_some(())
            .or_else(|| world.host.poll().pop().map(|_| ()))
    });
    if let Ok(produced) = finished {
        let error = world.admit(produced, consumer).unwrap_err();
        assert!(matches!(error, HostError::WorkerFenced(_)), "{error:?}");
    }
    assert_eq!(world.ledger(producer, consumer), (0, 0, 0, 0));
    world.reap(producer);
    world.reap(consumer);
    assert_eq!(world.book.total(), 0);
    assert_eq!(world.guardian_buffers(), 0);
    assert_eq!(world.host.quota().host_bytes(), 0);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

// --- fallback -----------------------------------------------------------------------------

/// Whatever `carrier_between` answers is what a staged host answers too when `CudaIpc` is not
/// allowed, and the tensor flows by it with nothing leased.
fn assert_falls_back(world: &mut World, producer: WorkerSession, consumer: WorkerSession) {
    let baseline = world
        .host
        .carrier_between(producer, consumer)
        .expect("carrier");
    assert_ne!(baseline, Carrier::CudaIpc);
    assert_eq!(
        world
            .host
            .carrier_between_staged(producer, consumer, &world.staging)
            .expect("carrier"),
        baseline
    );
    eprintln!("PATH: {baseline:?}");
    let data = tensor(8_000, 0);
    let produced = world.produce(producer, consumer, 1, data.clone());
    assert_eq!(produced.carrier, baseline);
    assert!(produced.staged.is_none());
    let transfer = world.admit(produced, consumer).expect("admitted");
    let outcome = world.consume(transfer).expect("consumed");
    assert_eq!(outcome.digest, content_digest(&data));
    // Nothing was leased or recorded.
    assert_eq!(world.staging.pool().slot_count(), 0);
    assert!(world.staging.registry().is_empty());
    assert_eq!(world.book.total(), 0);
    assert_eq!(world.guardian_buffers(), 0);
}

#[test]
fn without_full_confinement_the_carrier_is_bytes_and_nothing_is_leased() {
    let _serial = serial();
    // No uid, cgroup or namespace: the confinement is not Full, so no shared carrier at all.
    let mut world = world(ConfinePolicy::NONE);
    let (producer, consumer) = world.pair();
    assert_eq!(
        world
            .host
            .carrier_between_staged(producer, consumer, &world.staging)
            .expect("carrier"),
        Carrier::Bytes
    );
    assert_falls_back(&mut world, producer, consumer);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn workers_on_different_devices_or_without_a_device_or_import_fall_back() {
    let _serial = serial();
    // On a root host the fallback is `HostShm`, on an unprivileged one `Bytes`: whatever the
    // unstaged host would have chosen.
    for (producer_spec, consumer_spec) in [
        // Two devices.
        (
            Spec::producer(&[]),
            Spec {
                device: Some(DEVICE + 1),
                ..Spec::consumer("consumer-pod")
            },
        ),
        // The placement names no device for the consumer.
        (
            Spec::producer(&[]),
            Spec {
                device: None,
                ..Spec::consumer("consumer-pod")
            },
        ),
        // The producer was not started with the staging import.
        (
            Spec {
                staging: false,
                ..Spec::producer(&[])
            },
            Spec::consumer("consumer-pod"),
        ),
    ] {
        let mut world = world(ConfinePolicy::BEST_EFFORT);
        let producer = world.spawn(&producer_spec);
        let consumer = world.spawn(&consumer_spec);
        assert_falls_back(&mut world, producer, consumer);
        world.staging.shutdown(&mut world.host).expect("shutdown");
    }
}

#[test]
fn a_host_without_a_usable_staging_host_never_chooses_cudaipc() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    // The unstaged entry points never choose it, whatever the placement says.
    let baseline = world
        .host
        .carrier_between(producer, consumer)
        .expect("carrier");
    assert_ne!(baseline, Carrier::CudaIpc);
    let data = tensor(8_000, 0);
    let pending = world
        .host
        .start_produce(
            producer,
            consumer,
            ProduceSpec {
                template: template(1, data.len()),
                tensor: data.clone(),
            },
        )
        .expect("start");
    assert_eq!(pending.carrier(), baseline);
    let produced = world.host.finish_produce(pending).expect("produced");
    assert!(produced.staged.is_none());
    // A staging host that was ended is not usable: it is not chosen for a carrier.
    assert!(world.staging.is_usable());
    world.staging.shutdown(&mut world.host).expect("shutdown");
    assert!(world.staging.is_ended() && !world.staging.is_usable());
}

#[test]
fn a_lease_for_a_session_the_host_never_had_is_refused_and_never_freed_early() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::NONE);
    let (producer, consumer) = world.pair();
    let request = |p: u64, c: u64| ptr_fabric::AcquireRequest {
        producer_session: p,
        consumer_session: c,
        consumer_lane: 1,
        operation: OPERATION,
        device: DEVICE,
        len: 100,
    };
    // The checked lease refuses sessions the host does not have, and leases nothing.
    let refused = world
        .staging
        .issue_export_for(&world.host, &request(9_001, 9_002), 10_000, 0)
        .unwrap_err();
    assert!(
        matches!(refused, StagingError::Host(HostError::UnknownWorker(_))),
        "{refused:?}"
    );
    assert_eq!(world.staging.pool().slot_count(), 0);
    let ok = world
        .staging
        .issue_export_for(&world.host, &request(producer.0, consumer.0), 10_000, 0)
        .expect("registered sessions");
    // The raw lease does not check; a buffer of sessions the host never had is not freed by
    // the sync (it is not known to be reaped), only by the shutdown that ends everything.
    world
        .staging
        .issue_export(&request(9_001, 9_002), 10_000, 0)
        .expect("raw lease");
    assert_eq!(world.staging.pool().slot_count(), 2);
    assert!(world.staging.sync_reaped(&world.host).is_empty());
    assert_eq!(world.staging.pool().slot_count(), 2);
    assert!(world.staging.pool().slot(ok.lease.slot).is_some());
    world.staging.shutdown(&mut world.host).expect("shutdown");
    assert_eq!(world.book.total(), 0);
}

#[test]
fn a_cuda_import_is_never_chosen_while_its_producer_cannot_write() {
    let _serial = serial();
    if !privileged("a_cuda_import_is_never_chosen") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    // A host without a GPU cannot start a CUDA guardian: pretend the import is CUDA.
    world.staging.override_import(Some(BackendChoice::Cuda));
    let (producer, consumer) = world.pair();
    let baseline = world
        .host
        .carrier_between(producer, consumer)
        .expect("carrier");
    assert_ne!(baseline, Carrier::CudaIpc);
    assert_eq!(
        world
            .host
            .carrier_between_staged(producer, consumer, &world.staging)
            .expect("carrier"),
        baseline,
        "the CUDA producer refuses every staged produce, so no lease may be made for it"
    );
    assert_falls_back(&mut world, producer, consumer);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn two_live_leases_of_one_key_are_admitted_in_lease_order_or_refused() {
    let _serial = serial();
    if !privileged("two_live_leases_of_one_key") {
        return;
    }
    // In order: both are admitted, each with its own buffer.
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    let first = world.produce(producer, consumer, 1, tensor(8_000, 0));
    let second = world.produce(producer, consumer, 2, tensor(8_000, 1));
    assert_ne!(
        first.staged.expect("lease").lease.slot,
        second.staged.expect("lease").lease.slot
    );
    let t1 = world.admit(first, consumer).expect("first");
    let t2 = world.admit(second, consumer).expect("second");
    assert_eq!(
        world.consume(t1).expect("1").digest,
        content_digest(&tensor(8_000, 0))
    );
    assert_eq!(
        world.consume(t2).expect("2").digest,
        content_digest(&tensor(8_000, 1))
    );
    world.staging.shutdown(&mut world.host).expect("shutdown");

    // Out of order: the registry hands out the older record, which is not this handle's.
    let mut world = self::world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    let _first = world.produce(producer, consumer, 1, tensor(8_000, 0));
    let second = world.produce(producer, consumer, 2, tensor(8_000, 1));
    let error = world.admit(second, consumer).unwrap_err();
    assert!(
        matches!(error, HostError::StagedExportMismatch),
        "{error:?}"
    );
    assert_eq!(world.ledger(producer, consumer), (0, 0, 0, 0));
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn a_guardian_that_died_is_not_usable_and_the_carrier_falls_back_before_any_watch() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    assert!(world.staging.is_usable());
    world.staging.kill_guardian_process().expect("kill");
    wait_for("the guardian to be gone", || {
        (!world.staging.is_usable()).then_some(())
    });
    let baseline = world
        .host
        .carrier_between(producer, consumer)
        .expect("carrier");
    assert_eq!(
        world
            .host
            .carrier_between_staged(producer, consumer, &world.staging)
            .expect("carrier"),
        baseline
    );
    // The loss path then ends the workers it backed and the guardian.
    world.staging.watch(&mut world.host).expect("watch");
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

// --- the seccomp profile ------------------------------------------------------------------

impl World {
    /// Like `spawn`, but the refusal is the result (the Text-file-busy retry is the same).
    fn try_spawn(&mut self, spec: &Spec<'_>) -> Result<WorkerSession, HostError> {
        let placement = Placement {
            principal: spec.principal.to_owned(),
            stage_identity: spec.stage.to_owned(),
            seccomp: true,
            extra_args: spec.extra_args.iter().map(Into::into).collect(),
            device: spec.device,
            staging: spec.staging.then(|| self.staging.import()).flatten(),
            profile: spec.profile,
        };
        self.spawn_placement(&placement)
    }

    fn spawn_placement(&mut self, placement: &Placement) -> Result<WorkerSession, HostError> {
        for tries in 0..50 {
            match self.host.spawn_worker(placement) {
                Err(HostError::Supervisor(SupervisorError::Spawn(message)))
                    if message.contains("Text file busy") && tries < 49 =>
                {
                    std::thread::sleep(Duration::from_millis(20));
                }
                other => return other,
            }
        }
        unreachable!("the loop returns on its last attempt")
    }
}

#[test]
fn a_cpu_worker_is_classified_cpu_and_a_probe_that_shows_the_other_filter_is_refused() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let (producer, consumer) = world.pair();
    for session in [producer, consumer] {
        assert_eq!(
            world.host.profile(session).expect("profile"),
            SeccompProfile::Cpu
        );
        assert_eq!(
            world.host.profile_weakening(session).expect("weakening"),
            None
        );
    }
    // A worker whose probe answer says the wide ioctl reached the kernel is classified as the
    // GPU profile; the host asked for the CPU one, so it is refused and reaped.
    let before = world.host.workers().len();
    let error = world
        .try_spawn(&Spec {
            principal: "liar-pod",
            ..Spec::producer(&["--fault", "profile-mismatch"])
        })
        .unwrap_err();
    assert!(
        matches!(
            error,
            HostError::ProfileMismatch {
                requested: SeccompProfile::Cpu,
                observed: SeccompProfile::Gpu
            }
        ),
        "{error:?}"
    );
    assert_eq!(world.host.workers().len(), before, "the liar was reaped");
    assert_eq!(world.host.supervisor().sessions().len(), before);
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

#[test]
fn a_gpu_placement_without_a_driver_or_a_device_fails_typed_and_leaves_nothing() {
    let _serial = serial();
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    // Misplaced: nothing is spawned.
    let error = world
        .try_spawn(&Spec {
            device: None,
            profile: SeccompProfile::Gpu,
            ..Spec::producer(&[])
        })
        .unwrap_err();
    assert!(
        matches!(error, HostError::ProfileMisplaced(_)),
        "no device: {error:?}"
    );
    let unconfined = Placement {
        principal: "gpu-pod".to_owned(),
        stage_identity: PRODUCER_STAGE.to_owned(),
        seccomp: false,
        extra_args: Vec::new(),
        device: Some(DEVICE),
        staging: None,
        profile: SeccompProfile::Gpu,
    };
    let error = world.spawn_placement(&unconfined).unwrap_err();
    assert!(
        matches!(error, HostError::ProfileMisplaced(_)),
        "no seccomp: {error:?}"
    );
    assert!(world.host.workers().is_empty());
    assert!(world.host.supervisor().sessions().is_empty());

    #[cfg(feature = "cuda-ipc-core")]
    if ptr_fabric::CudaDriver::open().is_ok() {
        eprintln!("SKIPPED the no-driver arm: this host has a CUDA driver");
        world.staging.shutdown(&mut world.host).expect("shutdown");
        return;
    }
    // A host without a driver: the worker says so instead of signing, and is reaped.
    let error = world
        .try_spawn(&Spec {
            profile: SeccompProfile::Gpu,
            ..Spec::producer(&[])
        })
        .unwrap_err();
    let expected = if cfg!(feature = "cuda-ipc-core") {
        "gpu-no-driver"
    } else {
        "gpu-not-built"
    };
    assert!(
        matches!(&error, HostError::WorkerStartup(code) if code == expected),
        "{error:?}"
    );
    assert!(
        world.host.workers().is_empty(),
        "nothing is left registered"
    );
    assert!(world.host.supervisor().sessions().is_empty());
    world.staging.shutdown(&mut world.host).expect("shutdown");
}

/// A worker that initialises CUDA, installs the GPU seccomp filter and is classified `Gpu`.
#[cfg(feature = "cuda-ipc-core")]
#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md section 12"]
fn a_gpu_profile_worker_runs_its_filter_and_is_classified_gpu_on_a_gpu() {
    let _serial = serial();
    if !privileged("a_gpu_profile_worker_runs_its_filter") {
        return;
    }
    let mut world = world(ConfinePolicy::BEST_EFFORT);
    let gpu = world.spawn(&Spec {
        principal: "gpu-pod",
        profile: SeccompProfile::Gpu,
        ..Spec::producer(&[])
    });
    assert_eq!(
        world.host.profile(gpu).expect("profile"),
        SeccompProfile::Gpu
    );
    assert!(world
        .host
        .profile_weakening(gpu)
        .expect("weakening")
        .is_some());
    // Still a confined worker that serves calls, and the shared carriers stay available to it.
    assert_eq!(
        world.host.call_pod(gpu, "sum", b"1 2 3").expect("call"),
        ptr_worker::wire::WorkerReply::Ok(b"6".to_vec())
    );
    world.staging.shutdown(&mut world.host).expect("shutdown");
}
