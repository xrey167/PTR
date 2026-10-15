//! The worker side of the data plane (doc 42 stage S1): `ptr-worker serve --channel-fd <n>`.
//!
//! The worker inherits one end of a `SOCK_SEQPACKET` channel from its supervisor and serves the
//! same [`ProtocolSession`] as the stdio form over it: one encoded `ControlFrame` per frame,
//! with descriptors attached to frames (a produced `memfd` rides on the reply that carries its
//! handle, a received one on the call that asks to consume it).
//!
//! In order:
//!
//! 1. **seccomp first.** With `confined` the worker installs the allowlist of
//!    `ptr_fabric::install_worker_seccomp` before it does anything else; a filter that denies
//!    `execve` cannot go on before the exec, so this is the earliest the worker can do it.
//! 2. **The key.** The first frame is the supervisor's: the session id, the one stage identity
//!    the worker may serve and the secret key bound to it. The worker never chooses either.
//! 3. **The challenge.** The supervisor sends a random nonce; the worker signs
//!    `domain | session | nonce` with the key and answers. Only then does the supervisor
//!    register the worker's key for its stage.
//! 4. **The control session**, as responder. Pod calls (`worker.call.v1`) run the built-in Pods;
//!    data calls (`worker.data.v1`) produce a handle (sealed `memfd` or inline bytes) signed with
//!    the worker's key, or consume an admitted handle: map it read-only, verify the digest,
//!    unmap, answer. A call is served as soon as it is queued, so the descriptors that came
//!    with it belong to exactly that call.
//!
//! **The GPU profile.** With `gpu_device` (`--gpu-profile <device>`, only together with
//! `--confined`, builds with `cuda-ipc-core`) the worker does not install the CPU allowlist: it
//! initialises CUDA on that device first (`prepare_gpu_worker`, because the filter refuses
//! `open`) and then installs the weaker GPU filter (`install_gpu_filter`), keeping the
//! `GpuWorker` that is the only way to import afterwards. A worker that cannot do that (no
//! driver, no such device, a build without the feature) answers the host's challenge with a
//! startup refusal naming why (`gpu-no-driver`, `gpu-no-such-device`, `gpu-prepare-failed`,
//! `gpu-filter-failed`, `gpu-not-built`) and ends, so the host gets a typed error. Every
//! confined worker answers `probe` with `probe_profile_calls`, from which the host classifies
//! the filter it actually runs.
//!
//! Staged calls (`CudaIpc`, doc 42 rules 3a and 3b) need `--staging fake|cuda`, the import the
//! guardian's backend pairs with. A staged produce receives two descriptors, the supervisor's
//! staging buffer and a sealed `memfd` holding the tensor, writes the tensor through the first
//! (no further than `byte_len`), and signs a handle whose locator is the one the supervisor put
//! in the template, never one it chose. A staged consume receives the staging buffer and the
//! supervisor's export record, copies `byte_len` bytes into memory of its own, verifies the
//! digest on that copy and answers; it keeps nothing mapped. A worker started without
//! `--staging` refuses both with `staging-not-served`.
//!
//! The worker holds no ledger handle and no admission grant: every handle it produces is
//! admitted by the supervisor's process, and a handle it is given is mapped only after that
//! admission.

use std::os::fd::{AsFd, BorrowedFd, OwnedFd};

use ed25519_dalek::{Signer, SigningKey};
use ptr_fabric::{
    export_bytes, import_bytes, install_worker_seccomp, probe_forbidden_calls, probe_profile_calls,
    Channel, ChannelError, ConfineError, FabricError, Frame, ShmExpectation, ShmExport, ShmMapping,
};
use ptr_pods::PodRegistry;
use ptr_protocol::control::{decode, encode, CodecError, ControlFrame};
use ptr_protocol::session::{
    ProtocolSession, SessionConfig, SessionError, SessionEvent, SessionState,
};
use ptr_protocol::TypedPayload;
use ptr_protocol::{
    content_digest, handle_signing_bytes, Carrier, CarrierLocator, CudaExport, TensorHandle,
};
use ptr_types::CapabilityId;

use crate::data_wire::{
    challenge_message, decode_challenge, encode_response, encode_startup_refusal, wipe, DataReply,
    DataRequest, DataWireError, KeyFrame, WireExportRecord, DATA_TYPE,
};
use crate::guardian::BackendChoice;
use crate::pods::{builtin_registry, dispatch, BUILTIN};
use crate::serve::{worker_offer, ServeLimits, WORKER_NODE_ID};
use crate::staging_import::{read_copy, write_through};

/// Most produced regions the worker keeps open (the producer keeps its descriptors until the
/// supervisor reaps it, rule 3b), and most mappings it retains on request.
const MAX_EXPORTS: usize = 64;
const MAX_RETAINED: usize = 16;

/// What the worker does wrong on purpose, for the tests of the supervisor's refusals. Built
/// only with the `fault-injection` feature; a release worker has no way to select one.
#[cfg(feature = "fault-injection")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fault {
    /// Answer the challenge with a signature of another key.
    BadSignature,
    /// Never answer the challenge.
    Silent,
    /// Sign a handle of the other carrier than the one asked for, with a writable, unsealed
    /// memfd attached when it asked for `HostShm`.
    CarrierSwap,
    /// Change an authority-fixed field (the nonce) of the handle before signing it.
    TamperField,
    /// Attach a memfd of the right size that is not sealed.
    RegionUnsealed,
    /// Attach a sealed memfd of another size.
    RegionWrongSize,
    /// Attach a descriptor that is not a memfd at all.
    RegionNotMemfd,
    /// Sign a `CudaIpc` handle whose locator names a larger allocation than the supervisor's.
    StagingWrongSize,
    /// Sign a `CudaIpc` handle whose locator names another device than the supervisor's.
    StagingWrongDevice,
    /// Write other bytes into the staging buffer than the tensor whose digest is signed.
    StagingCorrupt,
    /// Answer `probe` as the other profile's filter would (the host asked for the CPU profile
    /// and is told the filter is the GPU one).
    ProfileMismatch,
}

#[cfg(feature = "fault-injection")]
impl Fault {
    /// The name `--fault` takes.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "bad-signature" => Self::BadSignature,
            "silent" => Self::Silent,
            "carrier-swap" => Self::CarrierSwap,
            "tamper-field" => Self::TamperField,
            "region-unsealed" => Self::RegionUnsealed,
            "region-wrong-size" => Self::RegionWrongSize,
            "region-not-memfd" => Self::RegionNotMemfd,
            "staging-wrong-size" => Self::StagingWrongSize,
            "staging-wrong-device" => Self::StagingWrongDevice,
            "staging-corrupt" => Self::StagingCorrupt,
            "profile-mismatch" => Self::ProfileMismatch,
            _ => return None,
        })
    }
}

/// How the worker is started.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChannelServeConfig {
    /// Install the seccomp allowlist first.
    pub confined: bool,
    /// Install the GPU-profile filter for this device instead of the CPU allowlist
    /// (`--gpu-profile`); needs `confined`.
    pub gpu_device: Option<u32>,
    /// The staging import this worker serves (`--staging`); `None` refuses staged calls.
    pub staging: Option<BackendChoice>,
    pub limits: ServeLimits,
    #[cfg(feature = "fault-injection")]
    pub fault: Option<Fault>,
}

/// Why serving ended abnormally.
#[derive(Debug)]
pub enum ChannelServeError {
    Channel(ChannelError),
    Confine(ConfineError),
    Wire(DataWireError),
    Codec(CodecError),
    Session(SessionError),
    BadConfig(SessionError),
    Registry(String),
    /// The supervisor closed the channel (`during_handshake`: before the control Hello).
    PeerVanished {
        during_handshake: bool,
    },
    /// A frame arrived that the protocol does not allow at this point.
    Protocol(&'static str),
    /// The worker could not become what it was asked to be and told the supervisor so; `code`
    /// is the startup code the supervisor received.
    Startup {
        code: &'static str,
    },
}

impl ChannelServeError {
    /// The startup code, if this is a refusal to start.
    pub fn startup_code(&self) -> Option<&'static str> {
        match self {
            Self::Startup { code } => Some(code),
            _ => None,
        }
    }
}

impl std::fmt::Display for ChannelServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ChannelServeError {}

impl From<ChannelError> for ChannelServeError {
    fn from(error: ChannelError) -> Self {
        Self::Channel(error)
    }
}

/// What a clean session did.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChannelServeSummary {
    pub peer_node: Option<String>,
    pub calls_executed: u64,
    pub duplicates_served: u64,
}

/// Serve one session over `channel` until the supervisor says Bye (`Ok`) or fails.
pub fn serve_channel(
    channel: Channel,
    config: ChannelServeConfig,
) -> Result<ChannelServeSummary, ChannelServeError> {
    let gpu = confine(&channel, &config)?;
    let registry =
        builtin_registry().map_err(|error| ChannelServeError::Registry(error.to_string()))?;
    let (key, stage_identity, session) = accept_key(&channel, &config)?;
    let mut plane = DataPlane {
        key,
        stage_identity,
        session,
        confined: config.confined,
        gpu,
        staging: config.staging,
        exports: Vec::new(),
        retained: Vec::new(),
        #[cfg(feature = "fault-injection")]
        fault: config.fault,
    };
    run_session(&channel, &registry, &mut plane, config)
}

/// What a GPU-profile worker keeps: the proof that CUDA was prepared before the filter.
#[cfg(feature = "cuda-ipc-core")]
type GpuHandle = ptr_fabric::GpuWorker;
/// A build without `cuda-ipc-core` has no GPU worker; this is never constructed.
#[cfg(not(feature = "cuda-ipc-core"))]
#[allow(dead_code)]
struct GpuHandle;

/// Step 1: the seccomp filter of the worker's profile. The CPU allowlist is installed before
/// anything else. The GPU profile prepares CUDA first (the filter refuses `open`), then installs
/// its filter; if it cannot, the supervisor is told why in place of the signature.
fn confine(
    channel: &Channel,
    config: &ChannelServeConfig,
) -> Result<Option<GpuHandle>, ChannelServeError> {
    match (config.confined, config.gpu_device) {
        (false, None) => Ok(None),
        (true, None) => {
            install_worker_seccomp().map_err(ChannelServeError::Confine)?;
            Ok(None)
        }
        (false, Some(_)) => refuse_startup(channel, "gpu-needs-confined"),
        (true, Some(device)) => gpu_confine(channel, device),
    }
}

#[cfg(feature = "cuda-ipc-core")]
fn gpu_confine(channel: &Channel, device: u32) -> Result<Option<GpuHandle>, ChannelServeError> {
    use ptr_fabric::{prepare_gpu_worker, CudaError};

    let prepared = match prepare_gpu_worker(device) {
        Ok(prepared) => prepared,
        Err(CudaError::NoDriver { .. }) => return refuse_startup(channel, "gpu-no-driver"),
        Err(CudaError::NoSuchDevice { .. }) => {
            return refuse_startup(channel, "gpu-no-such-device")
        }
        Err(_) => return refuse_startup(channel, "gpu-prepare-failed"),
    };
    match prepared.install_gpu_filter() {
        Ok(worker) => Ok(Some(worker)),
        Err(_) => refuse_startup(channel, "gpu-filter-failed"),
    }
}

#[cfg(not(feature = "cuda-ipc-core"))]
fn gpu_confine(channel: &Channel, _device: u32) -> Result<Option<GpuHandle>, ChannelServeError> {
    refuse_startup(channel, "gpu-not-built")
}

/// Read the key frame and the challenge (without using the secret), answer with a startup
/// refusal instead of the signature, and end. The supervisor then reaps this process.
fn refuse_startup<T>(channel: &Channel, code: &'static str) -> Result<T, ChannelServeError> {
    let mut frame = recv_frame(channel, true)?;
    wipe(&mut frame.body);
    recv_frame(channel, true)?;
    channel.send(&encode_startup_refusal(code), &[])?;
    Err(ChannelServeError::Startup { code })
}

/// Steps 2 and 3: take the key frame, answer the challenge.
fn accept_key(
    channel: &Channel,
    config: &ChannelServeConfig,
) -> Result<(SigningKey, String, u64), ChannelServeError> {
    let mut frame = recv_frame(channel, true)?;
    if !frame.descriptors.is_empty() {
        wipe(&mut frame.body);
        return Err(ChannelServeError::Protocol("descriptors on the key frame"));
    }
    let decoded = KeyFrame::decode(&frame.body);
    // The secret is in the key frame now; do not leave the receive buffer's copy around.
    wipe(&mut frame.body);
    let key_frame = decoded.map_err(ChannelServeError::Wire)?;
    let key = SigningKey::from_bytes(&key_frame.secret);
    let challenge = recv_frame(channel, true)?;
    let nonce = decode_challenge(&challenge.body).map_err(ChannelServeError::Wire)?;
    let message = challenge_message(key_frame.session, &key_frame.stage_identity, &nonce);
    let signature = key.sign(&message).to_bytes();
    #[cfg(feature = "fault-injection")]
    let signature = match config.fault {
        Some(Fault::Silent) => {
            // Wait until the supervisor gives up and closes the channel or kills this process.
            while channel.recv().is_ok() {}
            return Err(ChannelServeError::PeerVanished {
                during_handshake: true,
            });
        }
        Some(Fault::BadSignature) => SigningKey::from_bytes(&[0x5A; 32])
            .sign(&message)
            .to_bytes(),
        _ => signature,
    };
    #[cfg(not(feature = "fault-injection"))]
    let _ = config;
    channel.send(&encode_response(&signature), &[])?;
    Ok((key, key_frame.stage_identity.clone(), key_frame.session))
}

fn recv_frame(channel: &Channel, during_handshake: bool) -> Result<Frame, ChannelServeError> {
    match channel.recv() {
        Ok(frame) => Ok(frame),
        Err(ChannelError::PeerClosed) => Err(ChannelServeError::PeerVanished { during_handshake }),
        Err(error) => Err(ChannelServeError::Channel(error)),
    }
}

fn send_control(
    channel: &Channel,
    frame: &ControlFrame,
    descriptors: &[BorrowedFd<'_>],
) -> Result<(), ChannelServeError> {
    let body = encode(frame).map_err(ChannelServeError::Codec)?;
    channel.send(&body, descriptors)?;
    Ok(())
}

/// Step 4.
fn run_session(
    channel: &Channel,
    registry: &PodRegistry,
    plane: &mut DataPlane,
    config: ChannelServeConfig,
) -> Result<ChannelServeSummary, ChannelServeError> {
    let nonce = u64::from(std::process::id());
    let mut session_config = SessionConfig::new(WORKER_NODE_ID, worker_offer(), nonce);
    session_config.capabilities = BUILTIN
        .iter()
        .map(|pod| CapabilityId::from(pod.name))
        .collect();
    session_config.window = config.limits.window;
    session_config.completed_capacity = config.limits.completed_capacity;
    let mut session = ProtocolSession::new(session_config).map_err(ChannelServeError::BadConfig)?;
    let mut summary = ChannelServeSummary::default();

    while session.state() != SessionState::Closed {
        let frame = recv_frame(channel, session.state() == SessionState::Idle)?;
        let Frame { body, descriptors } = frame;
        let control = decode(&body).map_err(ChannelServeError::Codec)?;
        // Descriptors belong to a call; on any other frame they are closed unused.
        let mut held = if matches!(control, ControlFrame::Call { .. }) {
            descriptors
        } else {
            drop(descriptors);
            Vec::new()
        };
        match session.on_frame(control) {
            Ok(handled) => {
                for out in &handled.send {
                    send_control(channel, out, &[])?;
                }
                if matches!(handled.event, SessionEvent::DuplicateServed { .. }) {
                    summary.duplicates_served += 1;
                }
            }
            Err(error) => {
                if let Some(bye) = session.take_farewell() {
                    let _ = send_control(channel, &bye, &[]);
                }
                return Err(ChannelServeError::Session(error));
            }
        }
        // Serve at once: the descriptors in `held` belong to the call just queued.
        while session.queued_calls() > 0 && session.state() != SessionState::Closed {
            let mut produced: Vec<OwnedFd> = Vec::new();
            let mut handler = |payload: &TypedPayload| -> TypedPayload {
                handle_payload(registry, plane, payload, &mut held, &mut produced)
            };
            let Some(reply) = session
                .serve_next(&mut handler)
                .map_err(ChannelServeError::Session)?
            else {
                break;
            };
            summary.calls_executed += 1;
            let borrowed: Vec<BorrowedFd<'_>> = produced.iter().map(AsFd::as_fd).collect();
            send_control(channel, &reply, &borrowed)?;
        }
    }
    summary.peer_node = session.peer_node_id().map(str::to_owned);
    Ok(summary)
}

fn handle_payload(
    registry: &PodRegistry,
    plane: &mut DataPlane,
    payload: &TypedPayload,
    held: &mut Vec<OwnedFd>,
    produced: &mut Vec<OwnedFd>,
) -> TypedPayload {
    if payload.type_id.0 != DATA_TYPE {
        return dispatch(registry, payload).to_payload();
    }
    let reply = match DataRequest::from_payload(payload) {
        Ok(request) => plane.handle(request, held, produced),
        Err(error) => DataReply::Refused {
            code: format!("bad-data-call-{error}"),
        },
    };
    reply.to_payload()
}

/// What the worker keeps between calls.
struct DataPlane {
    key: SigningKey,
    stage_identity: String,
    /// The session the supervisor gave this worker (the consumer session of its export records).
    session: u64,
    /// The seccomp filter is installed: `probe` answers with the profile probe.
    confined: bool,
    /// The GPU worker of a GPU-profile worker: the only way to import after its filter.
    gpu: Option<GpuHandle>,
    staging: Option<BackendChoice>,
    /// The regions it produced: the producer holds them until its domain is reaped.
    exports: Vec<ShmExport>,
    /// Mappings kept after the answer, on request (a fault-injection hook).
    retained: Vec<ShmMapping>,
    #[cfg(feature = "fault-injection")]
    fault: Option<Fault>,
}

fn refused(code: &str) -> DataReply {
    DataReply::Refused {
        code: code.to_owned(),
    }
}

fn fabric_code(error: &FabricError) -> String {
    format!("fabric-{error}")
}

impl DataPlane {
    fn handle(
        &mut self,
        request: DataRequest,
        held: &mut Vec<OwnedFd>,
        produced: &mut Vec<OwnedFd>,
    ) -> DataReply {
        match request {
            DataRequest::Produce { template, tensor } => self.produce(&template, &tensor, produced),
            DataRequest::Consume { handle, retain } => self.consume(&handle, retain, held),
            DataRequest::ProduceStaged { template } => self.produce_staged(&template, held),
            DataRequest::ConsumeStaged { handle, record } => {
                self.consume_staged(&handle, &record, held)
            }
            DataRequest::Probe => DataReply::Probed {
                calls: self.probe_report(),
            },
        }
    }

    /// The probe a confined worker sends is the profile probe (the forbidden calls plus the wide
    /// `ioctl` the host classifies the filter by); an unconfined one sends the plain list.
    fn probe_report(&self) -> Vec<(String, i32)> {
        let report = if self.confined {
            probe_profile_calls()
        } else {
            probe_forbidden_calls()
        };
        #[allow(unused_mut)]
        let mut calls: Vec<(String, i32)> = report
            .into_iter()
            .map(|call| (call.name.to_owned(), call.errno))
            .collect();
        #[cfg(feature = "fault-injection")]
        if self.fault == Some(Fault::ProfileMismatch) {
            // Tell the host the wide `ioctl` reached the kernel (the GPU filter) whatever this
            // worker's filter is.
            for (name, errno) in &mut calls {
                if name == ptr_fabric::PROFILE_PROBE_CALL {
                    *errno = libc::EBADF;
                }
            }
        }
        calls
    }

    /// Build the tensor's carrier, fill in the digest and locator of the template, sign it.
    fn produce(
        &mut self,
        template: &[u8],
        tensor: &[u8],
        produced: &mut Vec<OwnedFd>,
    ) -> DataReply {
        let mut handle = match TensorHandle::decode(template) {
            Ok(handle) => handle,
            Err(error) => return refused(&format!("bad-template-{error:?}")),
        };
        if handle.byte_len != tensor.len() as u64 {
            return refused("tensor-length-mismatch");
        }
        // The worker signs only for the stage the supervisor bound it to.
        if handle.upstream_chain.stage.identity != self.stage_identity {
            return refused("stage-not-bound");
        }
        handle.content_digest = content_digest(tensor);
        match handle.carrier {
            Carrier::HostShm => {
                if self.exports.len() >= MAX_EXPORTS {
                    return refused("too-many-exports");
                }
                let export = match ShmExport::create(tensor) {
                    Ok(export) => export,
                    Err(error) => return refused(&fabric_code(&error)),
                };
                handle.locator = export.locator(0);
                let descriptor = match export.fd().try_clone_to_owned() {
                    Ok(descriptor) => descriptor,
                    Err(_) => return refused("descriptor-clone-failed"),
                };
                self.exports.push(export);
                produced.push(descriptor);
            }
            Carrier::Bytes => match export_bytes(tensor) {
                Ok(locator) => handle.locator = locator,
                Err(error) => return refused(&fabric_code(&error)),
            },
            _ => return refused("carrier-not-served"),
        }
        #[cfg(feature = "fault-injection")]
        self.apply_fault(&mut handle, tensor, produced);
        handle.signature = self.key.sign(&handle_signing_bytes(&handle)).to_bytes();
        match handle.encode() {
            Ok(encoded) => DataReply::Produced { handle: encoded },
            Err(error) => {
                produced.clear();
                refused(&format!("encode-{error:?}"))
            }
        }
    }

    /// Use an admitted handle: map (or read the inline bytes), verify the digest, answer. The
    /// mapping is dropped (unmapped) before the answer unless `retain` asks to keep it.
    fn consume(&mut self, encoded: &[u8], retain: bool, held: &mut Vec<OwnedFd>) -> DataReply {
        let handle = match TensorHandle::decode(encoded) {
            Ok(handle) => handle,
            Err(error) => return refused(&format!("bad-handle-{error:?}")),
        };
        match (&handle.carrier, &handle.locator) {
            (Carrier::HostShm, CarrierLocator::HostShm { .. }) => {
                if held.len() != 1 {
                    return refused("expected-one-descriptor");
                }
                let descriptor = held.remove(0);
                let expectation = match ShmExpectation::from_handle(&handle) {
                    Ok(expectation) => expectation,
                    Err(error) => return refused(&fabric_code(&error)),
                };
                let mapping = match ShmMapping::map(descriptor, &expectation) {
                    Ok(mapping) => mapping,
                    Err(error) => return refused(&fabric_code(&error)),
                };
                let digest = content_digest(mapping.as_bytes());
                let mapped_len = mapping.len() as u64;
                // Keeping a mapping after the answer is a lie a release worker never tells: only
                // the fault-injection build honours the request.
                if cfg!(feature = "fault-injection") && retain && self.retained.len() < MAX_RETAINED
                {
                    self.retained.push(mapping);
                }
                DataReply::Consumed { digest, mapped_len }
            }
            (Carrier::Bytes, CarrierLocator::Bytes { .. }) => match import_bytes(&handle) {
                Ok(bytes) => DataReply::Consumed {
                    digest: content_digest(bytes),
                    mapped_len: bytes.len() as u64,
                },
                Err(error) => refused(&fabric_code(&error)),
            },
            _ => refused("carrier-not-served"),
        }
    }
}

impl DataPlane {
    /// A staged produce (doc 42 rule 3a): the tensor is read from a sealed `memfd` the
    /// supervisor made (verified against the digest the supervisor put in the template), written
    /// through the staging buffer, and the handle is signed with exactly the locator the
    /// supervisor expects.
    fn produce_staged(&mut self, template: &[u8], held: &mut Vec<OwnedFd>) -> DataReply {
        let Some(import) = self.staging else {
            return refused("staging-not-served");
        };
        let mut handle = match TensorHandle::decode(template) {
            Ok(handle) => handle,
            Err(error) => return refused(&format!("bad-template-{error:?}")),
        };
        if handle.upstream_chain.stage.identity != self.stage_identity {
            return refused("stage-not-bound");
        }
        let alloc_size = match (&handle.carrier, &handle.locator) {
            (
                Carrier::CudaIpc,
                CarrierLocator::CudaIpc {
                    export: CudaExport::Vmm { alloc_size, .. },
                    ..
                },
            ) => *alloc_size,
            _ => return refused("carrier-not-served"),
        };
        if held.len() != 2 {
            return refused("expected-two-descriptors");
        }
        let staging = held.remove(0);
        let source = held.remove(0);
        // The tensor: a sealed memfd of exactly `byte_len` bytes whose digest is the template's.
        let expectation = ShmExpectation {
            byte_len: handle.byte_len,
            offset: 0,
            content_digest: handle.content_digest,
        };
        let mapping = match ShmMapping::map(source, &expectation) {
            Ok(mapping) => mapping,
            Err(error) => return refused(&fabric_code(&error)),
        };
        let tensor = mapping.as_bytes();
        let digest = content_digest(tensor);
        match import {
            BackendChoice::Fake => {
                #[cfg(feature = "fault-injection")]
                let corrupted: Option<Vec<u8>> = (self.fault == Some(Fault::StagingCorrupt))
                    .then(|| tensor.iter().map(|byte| !byte).collect());
                #[cfg(feature = "fault-injection")]
                let written: &[u8] = corrupted.as_deref().unwrap_or(tensor);
                #[cfg(not(feature = "fault-injection"))]
                let written: &[u8] = tensor;
                if let Err(error) = write_through(staging, alloc_size, written) {
                    return refused(&error.code());
                }
            }
            // `ptr-fabric` has no writable import of a VMM descriptor yet: the producer side of
            // the CUDA staging path is the guardian's fill (at most about 64 KiB per fill).
            BackendChoice::Cuda => {
                drop(staging);
                return refused("cuda-producer-write-not-built");
            }
        }
        handle.content_digest = digest;
        #[cfg(feature = "fault-injection")]
        self.apply_staging_fault(&mut handle);
        handle.signature = self.key.sign(&handle_signing_bytes(&handle)).to_bytes();
        match handle.encode() {
            Ok(encoded) => DataReply::Produced { handle: encoded },
            Err(error) => refused(&format!("encode-{error:?}")),
        }
    }

    /// A staged consume (doc 42 rule 3a): check the supervisor's export record against the
    /// handle and against this worker's own session, copy `byte_len` bytes out of the staging
    /// buffer into memory of the worker's own, unmap, verify the digest on the copy.
    fn consume_staged(
        &mut self,
        encoded: &[u8],
        record: &WireExportRecord,
        held: &mut Vec<OwnedFd>,
    ) -> DataReply {
        let Some(import) = self.staging else {
            return refused("staging-not-served");
        };
        let handle = match TensorHandle::decode(encoded) {
            Ok(handle) => handle,
            Err(error) => return refused(&format!("bad-handle-{error:?}")),
        };
        let (device, alloc_size) = match (&handle.carrier, &handle.locator) {
            (
                Carrier::CudaIpc,
                CarrierLocator::CudaIpc {
                    device,
                    export: CudaExport::Vmm { alloc_size, .. },
                    ..
                },
            ) => (*device, *alloc_size),
            _ => return refused("carrier-not-served"),
        };
        if held.len() != 1 {
            return refused("expected-one-descriptor");
        }
        // The record is the supervisor's: it must name this worker and this handle.
        if record.consumer_session != self.session {
            return refused("record-consumer-mismatch");
        }
        if record.operation != handle.parent_operation_id {
            return refused("record-operation-mismatch");
        }
        if record.device != device {
            return refused("record-device-mismatch");
        }
        if record.size != alloc_size {
            return refused("record-size-mismatch");
        }
        let staging = held.remove(0);
        let copy = match import {
            BackendChoice::Fake => match read_copy(staging, alloc_size, handle.byte_len) {
                Ok(copy) => copy,
                Err(error) => return refused(&error.code()),
            },
            BackendChoice::Cuda => {
                match cuda_copy(&handle, record, staging, self.session, self.gpu.as_ref()) {
                    Ok(copy) => copy,
                    Err(code) => return refused(&code),
                }
            }
        };
        let digest = content_digest(&copy);
        if digest != handle.content_digest {
            return refused("staging-digest-mismatch");
        }
        DataReply::Consumed {
            digest,
            mapped_len: copy.len() as u64,
        }
    }
}

/// The CUDA import of a consumer (`ptr_fabric::import_and_copy`): record against handle, import,
/// map exactly the recorded size, copy `byte_len` bytes, release, verify the digest. Has not run
/// on a GPU.
#[cfg(feature = "cuda-ipc-core")]
fn cuda_copy(
    handle: &TensorHandle,
    record: &WireExportRecord,
    staging: OwnedFd,
    session: u64,
    gpu: Option<&GpuHandle>,
) -> Result<Vec<u8>, String> {
    use ptr_fabric::{import_and_copy, CudaDriver, ExportRecord, ImportExpectation};

    let CarrierLocator::CudaIpc { device, .. } = &handle.locator else {
        return Err("carrier-not-served".to_owned());
    };
    let expected = ImportExpectation::from_handle(handle, session, *device)
        .map_err(|_| "cuda-bad-expectation".to_owned())?;
    let export = ExportRecord {
        producer_session: record.producer_session,
        consumer_session: record.consumer_session,
        operation: record.operation,
        device: record.device,
        size: record.size,
    };
    // A GPU-profile worker imports through the worker the filter was installed with (a driver
    // opened now would have to `open` device nodes, which the filter refuses); another worker
    // opens the driver itself.
    match gpu {
        Some(worker) => worker
            .import_and_copy(staging, &export, &expected)
            .map_err(|_| "cuda-import-failed".to_owned()),
        None => {
            let driver = CudaDriver::open().map_err(|_| "cuda-no-driver".to_owned())?;
            import_and_copy(&driver, staging, &export, &expected)
                .map_err(|_| "cuda-import-failed".to_owned())
        }
    }
}

#[cfg(not(feature = "cuda-ipc-core"))]
fn cuda_copy(
    _handle: &TensorHandle,
    _record: &WireExportRecord,
    _staging: OwnedFd,
    _session: u64,
    _gpu: Option<&GpuHandle>,
) -> Result<Vec<u8>, String> {
    Err("cuda-not-built".to_owned())
}

#[cfg(feature = "fault-injection")]
impl DataPlane {
    /// Sign a `CudaIpc` locator other than the supervisor's (after the honest handle is built).
    fn apply_staging_fault(&mut self, handle: &mut TensorHandle) {
        let CarrierLocator::CudaIpc { device, export, .. } = &mut handle.locator else {
            return;
        };
        match self.fault {
            Some(Fault::StagingWrongSize) => {
                if let CudaExport::Vmm { alloc_size, .. } = export {
                    *alloc_size = alloc_size.saturating_add(4096);
                }
            }
            Some(Fault::StagingWrongDevice) => *device = device.wrapping_add(1),
            _ => {}
        }
    }
}

/// A memfd of `len` bytes that is writable, resizable and carries no seal.
#[cfg(feature = "fault-injection")]
fn unsealed_memfd(len: usize) -> Option<OwnedFd> {
    use std::os::fd::FromRawFd;
    // SAFETY: `memfd_create` takes a NUL-terminated name and flags; a negative result is an
    // error, otherwise the descriptor is new and owned by nothing else, so `from_raw_fd` is
    // sound. `ftruncate` takes that descriptor and a length.
    unsafe {
        let raw = libc::memfd_create(c"fault".as_ptr(), libc::MFD_CLOEXEC);
        if raw < 0 {
            return None;
        }
        let fd = OwnedFd::from_raw_fd(raw);
        if libc::ftruncate(raw, len as libc::off_t) != 0 {
            return None;
        }
        Some(fd)
    }
}

#[cfg(feature = "fault-injection")]
impl DataPlane {
    /// Misbehave as the selected fault says, after the honest handle is built and before it is
    /// signed (so the signature is valid and only the supervisor's own checks can refuse it).
    fn apply_fault(
        &mut self,
        handle: &mut TensorHandle,
        tensor: &[u8],
        produced: &mut Vec<OwnedFd>,
    ) {
        let Some(fault) = self.fault else {
            return;
        };
        match fault {
            Fault::BadSignature
            | Fault::Silent
            | Fault::StagingWrongSize
            | Fault::StagingWrongDevice
            | Fault::StagingCorrupt
            | Fault::ProfileMismatch => {}
            Fault::TamperField => handle.handle_nonce = handle.handle_nonce.wrapping_add(1000),
            Fault::CarrierSwap => {
                if handle.carrier == Carrier::HostShm {
                    // A Bytes handle, with a writable memfd smuggled along.
                    produced.clear();
                    produced.extend(unsealed_memfd(tensor.len()));
                    handle.carrier = Carrier::Bytes;
                    if let Ok(locator) = export_bytes(tensor) {
                        handle.locator = locator;
                    }
                } else if let Ok(export) = ShmExport::create(tensor) {
                    handle.carrier = Carrier::HostShm;
                    handle.locator = export.locator(0);
                    produced.extend(export.fd().try_clone_to_owned().ok());
                    self.exports.push(export);
                }
            }
            Fault::RegionUnsealed | Fault::RegionWrongSize | Fault::RegionNotMemfd => {
                if handle.carrier != Carrier::HostShm {
                    return;
                }
                produced.clear();
                match fault {
                    Fault::RegionUnsealed => produced.extend(unsealed_memfd(tensor.len())),
                    Fault::RegionWrongSize => {
                        if let Ok(export) = ShmExport::create(&tensor[..tensor.len() / 2]) {
                            produced.extend(export.fd().try_clone_to_owned().ok());
                            self.exports.push(export);
                        }
                    }
                    _ => produced.extend(std::io::stdin().as_fd().try_clone_to_owned().ok()),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Signature;
    use ptr_protocol::activation::{BoundaryDtype, ChainElement, UpstreamChain};
    use ptr_types::TypeId;
    use std::fs::File;
    use std::os::fd::FromRawFd;
    use std::os::unix::fs::FileExt;

    const STAGE: &str = "stage";
    const SESSION: u64 = 42;
    const OPERATION: u64 = 7;
    const ALLOC: u64 = 8192;

    fn plane(staging: Option<BackendChoice>) -> DataPlane {
        DataPlane {
            key: SigningKey::from_bytes(&[9; 32]),
            stage_identity: STAGE.to_owned(),
            session: SESSION,
            confined: false,
            gpu: None,
            staging,
            exports: Vec::new(),
            retained: Vec::new(),
            #[cfg(feature = "fault-injection")]
            fault: None,
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
        (0..5000usize).map(|n| ((n * 7) % 251) as u8).collect()
    }

    fn handle(carrier: Carrier, alloc_size: u64, digest: [u8; 32]) -> TensorHandle {
        let locator = match carrier {
            Carrier::CudaIpc => CarrierLocator::CudaIpc {
                device: 0,
                export: CudaExport::Vmm {
                    fd_slot: 0,
                    alloc_size,
                },
                event_handle: [0; 64],
            },
            _ => CarrierLocator::Bytes { inline: tensor() },
        };
        TensorHandle {
            carrier,
            locator,
            dtype: BoundaryDtype::F32,
            shape: vec![1250],
            byte_len: 5000,
            upstream_chain: UpstreamChain {
                stage: element(STAGE),
                base_model: element("model"),
                adapters: vec![],
                experts: vec![],
            },
            placement_epoch: 5,
            signed_absolute_deadline_ms: 9_000,
            handle_nonce: 1,
            parent_operation_id: OPERATION,
            consumer: "consumer-stage".to_owned(),
            payload_type: TypeId::from("ptr.split.activation.v1"),
            content_digest: digest,
            signature: [0; 64],
        }
    }

    fn memfd(len: u64) -> (OwnedFd, File) {
        // SAFETY: the name is NUL terminated; a negative result is an error, otherwise the
        // descriptor is new and owned by nothing else.
        let raw = unsafe { libc::memfd_create(c"channel-serve-test".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(raw >= 0, "memfd_create");
        // SAFETY: `raw` is a fresh descriptor that nothing else owns.
        let file = File::from(unsafe { OwnedFd::from_raw_fd(raw) });
        file.set_len(len).expect("size");
        let shared = file.try_clone().expect("dup");
        (OwnedFd::from(file), shared)
    }

    fn source(bytes: &[u8]) -> OwnedFd {
        ShmExport::create(bytes).expect("a sealed source").into_fd()
    }

    fn record() -> WireExportRecord {
        WireExportRecord {
            producer_session: 1,
            consumer_session: SESSION,
            operation: OPERATION,
            device: 0,
            size: ALLOC,
        }
    }

    fn code(reply: &DataReply) -> &str {
        match reply {
            DataReply::Refused { code } => code,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// A buffer holding `data`, as a producer would have left it.
    fn filled(data: &[u8]) -> (OwnedFd, File) {
        let (buffer, shared) = memfd(ALLOC);
        shared.write_all_at(data, 0).expect("write");
        (buffer, shared)
    }

    #[test]
    fn a_staged_produce_writes_through_the_buffer_and_signs_the_hosts_locator() {
        let mut plane = plane(Some(BackendChoice::Fake));
        let data = tensor();
        let template = handle(Carrier::CudaIpc, ALLOC, content_digest(&data));
        let (buffer, shared) = memfd(ALLOC);
        let mut held = vec![buffer, source(&data)];
        let reply = plane.produce_staged(&template.encode().expect("template"), &mut held);
        let DataReply::Produced { handle: encoded } = reply else {
            panic!("{reply:?}");
        };
        assert!(held.is_empty(), "both descriptors were consumed");
        let signed = TensorHandle::decode(&encoded).expect("a handle");
        assert_eq!(
            signed.locator, template.locator,
            "the host's locator, unchanged"
        );
        let mut unsigned = signed.clone();
        unsigned.signature = [0; 64];
        assert_eq!(unsigned, template);
        plane
            .key
            .verifying_key()
            .verify_strict(
                &handle_signing_bytes(&signed),
                &Signature::from_bytes(&signed.signature),
            )
            .expect("signed by the worker's key");
        let mut written = vec![0u8; ALLOC as usize];
        shared.read_exact_at(&mut written, 0).expect("read");
        assert_eq!(&written[..5000], &data[..]);
        assert!(written[5000..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn a_staged_produce_refuses_what_it_should_not_serve_with_a_typed_code() {
        let data = tensor();
        let good = handle(Carrier::CudaIpc, ALLOC, content_digest(&data))
            .encode()
            .expect("template");
        // No import selected.
        let mut held = vec![memfd(ALLOC).0, source(&data)];
        assert_eq!(
            code(&plane(None).produce_staged(&good, &mut held)),
            "staging-not-served"
        );
        let mut served = plane(Some(BackendChoice::Fake));
        // The wrong number of descriptors.
        let mut held = vec![memfd(ALLOC).0];
        assert_eq!(
            code(&served.produce_staged(&good, &mut held)),
            "expected-two-descriptors"
        );
        // A template that is not a handle, one of another carrier, one for another stage.
        let mut held = vec![memfd(ALLOC).0, source(&data)];
        assert!(code(&served.produce_staged(b"nope", &mut held)).starts_with("bad-template"));
        let bytes = handle(Carrier::Bytes, ALLOC, content_digest(&data))
            .encode()
            .expect("template");
        assert_eq!(
            code(&served.produce_staged(&bytes, &mut held)),
            "carrier-not-served"
        );
        let mut other = handle(Carrier::CudaIpc, ALLOC, content_digest(&data));
        other.upstream_chain.stage.identity = "elsewhere".to_owned();
        assert_eq!(
            code(&served.produce_staged(&other.encode().expect("template"), &mut held)),
            "stage-not-bound"
        );
        // A tensor source that is not the committed bytes (the digest of the template differs).
        let mut held = vec![memfd(ALLOC).0, source(&data)];
        let wrong = handle(Carrier::CudaIpc, ALLOC, [1; 32])
            .encode()
            .expect("template");
        assert!(code(&served.produce_staged(&wrong, &mut held)).starts_with("fabric-"));
        // A staging buffer that is not the recorded size.
        let mut held = vec![memfd(ALLOC * 2).0, source(&data)];
        assert_eq!(
            code(&served.produce_staged(&good, &mut held)),
            "staging-size-mismatch"
        );
        // The CUDA producer has no writable import yet.
        let mut cuda = plane(Some(BackendChoice::Cuda));
        let mut held = vec![memfd(ALLOC).0, source(&data)];
        assert_eq!(
            code(&cuda.produce_staged(&good, &mut held)),
            "cuda-producer-write-not-built"
        );
    }

    #[test]
    fn a_staged_consume_copies_out_and_verifies_the_digest() {
        let mut served = plane(Some(BackendChoice::Fake));
        let data = tensor();
        let encoded = handle(Carrier::CudaIpc, ALLOC, content_digest(&data))
            .encode()
            .expect("handle");
        let (buffer, _shared) = filled(&data);
        let mut held = vec![buffer];
        assert_eq!(
            served.consume_staged(&encoded, &record(), &mut held),
            DataReply::Consumed {
                digest: content_digest(&data),
                mapped_len: 5000
            }
        );
        // Bytes in the buffer that are not the committed ones are refused.
        let inverted: Vec<u8> = data.iter().map(|byte| !byte).collect();
        let (buffer, _shared) = filled(&inverted);
        let mut held = vec![buffer];
        assert_eq!(
            code(&served.consume_staged(&encoded, &record(), &mut held)),
            "staging-digest-mismatch"
        );
    }

    #[test]
    fn a_staged_consume_refuses_a_record_that_does_not_name_this_worker_and_handle() {
        let mut served = plane(Some(BackendChoice::Fake));
        let data = tensor();
        let encoded = handle(Carrier::CudaIpc, ALLOC, content_digest(&data))
            .encode()
            .expect("handle");
        let cases = [
            (
                WireExportRecord {
                    consumer_session: SESSION + 1,
                    ..record()
                },
                "record-consumer-mismatch",
            ),
            (
                WireExportRecord {
                    operation: OPERATION + 1,
                    ..record()
                },
                "record-operation-mismatch",
            ),
            (
                WireExportRecord {
                    device: 1,
                    ..record()
                },
                "record-device-mismatch",
            ),
            (
                WireExportRecord {
                    size: ALLOC + 4096,
                    ..record()
                },
                "record-size-mismatch",
            ),
        ];
        for (wrong, expected) in cases {
            let (buffer, _shared) = filled(&data);
            let mut held = vec![buffer];
            assert_eq!(
                code(&served.consume_staged(&encoded, &wrong, &mut held)),
                expected
            );
        }
        // No descriptor, two descriptors, no import, a handle of another carrier or locator.
        assert_eq!(
            code(&served.consume_staged(&encoded, &record(), &mut Vec::new())),
            "expected-one-descriptor"
        );
        let mut two = vec![filled(&data).0, filled(&data).0];
        assert_eq!(
            code(&served.consume_staged(&encoded, &record(), &mut two)),
            "expected-one-descriptor"
        );
        let mut held = vec![filled(&data).0];
        assert_eq!(
            code(&plane(None).consume_staged(&encoded, &record(), &mut held)),
            "staging-not-served"
        );
        let bytes = handle(Carrier::Bytes, ALLOC, content_digest(&data))
            .encode()
            .expect("handle");
        let mut held = vec![filled(&data).0];
        assert_eq!(
            code(&served.consume_staged(&bytes, &record(), &mut held)),
            "carrier-not-served"
        );
        assert!(
            code(&served.consume_staged(b"nope", &record(), &mut held)).starts_with("bad-handle")
        );
        // The CUDA import without a driver (or without the feature) is a typed refusal.
        let mut cuda = plane(Some(BackendChoice::Cuda));
        let mut held = vec![filled(&data).0];
        assert!(code(&cuda.consume_staged(&encoded, &record(), &mut held)).starts_with("cuda-"));
    }
}
