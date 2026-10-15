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
//! The worker holds no ledger handle and no admission grant: every handle it produces is
//! admitted by the supervisor's process, and a handle it is given is mapped only after that
//! admission.

use std::os::fd::{AsFd, BorrowedFd, OwnedFd};

use ed25519_dalek::{Signer, SigningKey};
use ptr_fabric::{
    export_bytes, import_bytes, install_worker_seccomp, probe_forbidden_calls, Channel,
    ChannelError, ConfineError, FabricError, Frame, ShmExpectation, ShmExport, ShmMapping,
};
use ptr_pods::PodRegistry;
use ptr_protocol::control::{decode, encode, CodecError, ControlFrame};
use ptr_protocol::session::{
    ProtocolSession, SessionConfig, SessionError, SessionEvent, SessionState,
};
use ptr_protocol::TypedPayload;
use ptr_protocol::{content_digest, handle_signing_bytes, Carrier, CarrierLocator, TensorHandle};
use ptr_types::CapabilityId;

use crate::data_wire::{
    challenge_message, decode_challenge, encode_response, wipe, DataReply, DataRequest,
    DataWireError, KeyFrame, DATA_TYPE,
};
use crate::pods::{builtin_registry, dispatch, BUILTIN};
use crate::serve::{worker_offer, ServeLimits, WORKER_NODE_ID};

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
            _ => return None,
        })
    }
}

/// How the worker is started.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChannelServeConfig {
    /// Install the seccomp allowlist first.
    pub confined: bool,
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
    if config.confined {
        install_worker_seccomp().map_err(ChannelServeError::Confine)?;
    }
    let registry =
        builtin_registry().map_err(|error| ChannelServeError::Registry(error.to_string()))?;
    let (key, stage_identity) = accept_key(&channel, &config)?;
    let mut plane = DataPlane {
        key,
        stage_identity,
        exports: Vec::new(),
        retained: Vec::new(),
        #[cfg(feature = "fault-injection")]
        fault: config.fault,
    };
    run_session(&channel, &registry, &mut plane, config)
}

/// Steps 2 and 3: take the key frame, answer the challenge.
fn accept_key(
    channel: &Channel,
    config: &ChannelServeConfig,
) -> Result<(SigningKey, String), ChannelServeError> {
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
    Ok((key, key_frame.stage_identity.clone()))
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
            DataRequest::Probe => DataReply::Probed {
                calls: probe_forbidden_calls()
                    .into_iter()
                    .map(|call| (call.name.to_owned(), call.errno))
                    .collect(),
            },
        }
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
            Fault::BadSignature | Fault::Silent => {}
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
