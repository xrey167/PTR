//! The loopback wire: one real `PodHost` and several real `PodClient` endpoints over iroh.
//!
//! Every payload is one PodWire request on its own authenticated connection (`PodClient::request`
//! opens one connection per request, so "one chunk per request" is also "one chunk per
//! connection"). The host serves it with `serve_once`: the host's access policy decides on the
//! authenticated peer, the intake Pod keeps the payload in a one-slot mailbox and answers with the
//! SHA-256 of what it received, and the receiving `FrameAssembler` then takes the payload under the
//! principal that `serve_once` reports, which is the peer id the connection authenticated and
//! nothing the payload or the harness says.
//!
//! The assembler's clock is virtual (`now_ms`): PodWire v1 carries no deadline, so the only clock
//! is the one the caller hands the assembler, and a virtual one keeps every verdict independent of
//! how long a connection took. Only the bounded timeout of one exchange is wall time.

use super::{Counters, Res};
use ed25519_dalek::{SigningKey, VerifyingKey};
use ptr_ledger::integrity::sha256;
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_podwire::{
    encode_request, AssemblyPolicy, FrameAssembler, Offered, PeerBook, PodAccessPolicy, PodClient,
    PodHost, PodOutcome, PodRequest, PodScope, RefusalCode, Rejection,
};
use ptr_protocol::activation::{ACTIVATION_CHUNK_TYPE, SPLIT_POLICY_TYPE};
use ptr_protocol::TypedPayload;
use ptr_types::{
    CapabilityId, Effect, NodeId, PodId, Probability, ProjectId, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

/// The capability the intake Pod serves.
pub const CAPABILITY: &str = "split-activation";
/// The project of the intake Pod.
const PROJECT: &str = "split-pod-b";
/// The output type of the intake Pod: the digest of what it received.
const ACK_TYPE: &str = "SplitAck";
/// Bound on one exchange (connect, send, serve, answer). Wall time, never a verdict.
const EXCHANGE_TIMEOUT: Duration = Duration::from_secs(60);
/// The bound the clients wait for an answer.
const CLIENT_TIMEOUT: Duration = Duration::from_secs(30);

/// Type ids the intake Pod's manifest accepts besides the allowlist. The manifest is wide on
/// purpose: it is the host's access policy, keyed by the authenticated peer, that has to keep
/// them out for a strict peer, so the policy and not the manifest is what is under test.
pub const EXTRA_ACCEPTED: [&str; 4] = [
    "ptr.kv.page.v1",
    "ptr.kv.snapshot.v1",
    "ptr.split.activation-chunk.v2",
    "ptr.tensor.blob.v1",
];

/// A peer of the host: its own endpoint and the key its runtime signs chunks with.
pub struct Peer {
    pub client: PodClient,
    /// The peer id the connection authenticates (the iroh endpoint id).
    pub id: String,
    /// The Ed25519 key of the runtime behind this endpoint.
    pub key: SigningKey,
    /// The stage identity this peer's runtime may speak for.
    pub stage: &'static str,
}

/// Which scope the host's policy gives a peer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Standing {
    /// The allowlisted types only.
    Strict,
    /// The allowlisted types and the extra accepted types: a deliberately misconfigured scope.
    Lax,
    /// Not admitted at all.
    Stranger,
}

struct Intake {
    manifest: PodManifest,
    mailbox: Arc<Mutex<Option<TypedPayload>>>,
}

impl DynPod for Intake {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        let digest = sha256(&input.bytes);
        let mut slot = self
            .mailbox
            .lock()
            .map_err(|_| "the mailbox is poisoned".to_string())?;
        if slot.is_some() {
            return Err("the mailbox still holds a payload".to_string());
        }
        *slot = Some(input);
        Ok(TypedPayload {
            type_id: TypeId::from(ACK_TYPE),
            bytes: digest.to_vec(),
        })
    }
}

struct Pass;

impl Verifier<TypedPayload> for Pass {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).expect("1.0 is a probability"),
            findings: vec![],
        }
    }
}

/// What the host and the assembler did with one payload.
#[derive(Debug)]
pub struct Exchange {
    /// The refusal the host's access decision answered with; `None` when the Pod ran.
    pub host_refusal: Option<RefusalCode>,
    /// The assembler's verdict, when the payload reached it.
    pub verdict: Option<Result<Offered, Rejection>>,
}

/// What came of one send.
#[derive(Debug)]
pub enum Sent {
    /// PodWire's own codec refused the request before anything left (`codec_refusals`).
    Codec,
    /// The request travelled and was answered.
    Exchange(Exchange),
}

impl Sent {
    /// The assembler's rejection, if the payload reached it and was refused.
    pub fn rejection(&self) -> Option<&Rejection> {
        match self {
            Self::Exchange(Exchange {
                verdict: Some(Err(rejection)),
                ..
            }) => Some(rejection),
            _ => None,
        }
    }

    /// Whether the assembler took the payload.
    pub fn admitted(&self) -> bool {
        matches!(
            self,
            Self::Exchange(Exchange {
                verdict: Some(Ok(_)),
                ..
            })
        )
    }

    /// Whether the intake Pod ran (the host's access decision let the payload in).
    pub fn reached_pod(&self) -> bool {
        matches!(
            self,
            Self::Exchange(Exchange {
                host_refusal: None,
                ..
            })
        )
    }

    pub fn host_refusal(&self) -> Option<RefusalCode> {
        match self {
            Self::Exchange(exchange) => exchange.host_refusal,
            Self::Codec => None,
        }
    }
}

/// The host, the address book, the receiving assembler and the clocks.
pub struct Net {
    rt: Runtime,
    host: PodHost,
    book: PeerBook,
    host_id: String,
    mailbox: Arc<Mutex<Option<TypedPayload>>>,
    pub assembler: FrameAssembler,
    /// The assembler's virtual clock.
    pub now_ms: u64,
    next_request: u64,
    connections: u64,
}

fn scope(strict: bool) -> PodScope {
    let mut scope = PodScope::new(ProjectId::from(PROJECT));
    let cap = CapabilityId::from(CAPABILITY);
    for ty in [ACTIVATION_CHUNK_TYPE, SPLIT_POLICY_TYPE] {
        scope = scope.allow(cap.clone(), TypeId::from(ty));
    }
    if !strict {
        for ty in EXTRA_ACCEPTED {
            scope = scope.allow(cap.clone(), TypeId::from(ty));
        }
    }
    scope
}

impl Net {
    /// A runtime, the host with its intake Pod, and an empty assembler (see [`Self::reset`]).
    pub fn new(rt: Runtime, peers: &[(&Peer, Standing)]) -> Res<Self> {
        let mailbox = Arc::new(Mutex::new(None));
        let mut accepts: Vec<TypeId> = [ACTIVATION_CHUNK_TYPE, SPLIT_POLICY_TYPE]
            .into_iter()
            .chain(EXTRA_ACCEPTED)
            .map(TypeId::from)
            .collect();
        accepts.sort();
        let pod = Arc::new(Intake {
            manifest: PodManifest {
                project: ProjectId::from(PROJECT),
                id: PodId::from("split-intake"),
                capabilities: vec![CapabilityId::from(CAPABILITY)],
                accepts,
                produces: vec![TypeId::from(ACK_TYPE)],
                effects: vec![Effect::Pure],
                protocol_version: 1,
            },
            mailbox: Arc::clone(&mailbox),
        });
        let mut registry = PodRegistry::default();
        registry.register(pod);
        let mut policy = PodAccessPolicy::new();
        for (peer, standing) in peers {
            match standing {
                Standing::Strict => policy
                    .admit(NodeId(peer.id.clone()), scope(true))
                    .map_err(|e| format!("admitting a strict peer: {e}"))?,
                Standing::Lax => policy
                    .admit(NodeId(peer.id.clone()), scope(false))
                    .map_err(|e| format!("admitting a lax peer: {e}"))?,
                Standing::Stranger => {}
            }
        }
        let host = rt
            .block_on(PodHost::bind(registry, policy, Pass))
            .map_err(|e| format!("binding the host: {e}"))?;
        let host_id = host.identity().public_key;
        let mut book = PeerBook::new();
        book.record(host.address());
        let assembler = FrameAssembler::new(super::cases::limits())
            .map_err(|violation| format!("the assembly policy: {violation:?}"))?;
        Ok(Self {
            rt,
            host,
            book,
            host_id,
            mailbox,
            assembler,
            now_ms: 1_000,
            next_request: 0,
            connections: 0,
        })
    }

    /// A fresh receiver under `limits` with `principals` registered: the verifying key of each
    /// peer's runtime and the one stage it may speak for. Time restarts at the virtual origin.
    pub fn reset(
        &mut self,
        limits: AssemblyPolicy,
        principals: &[(&str, VerifyingKey, &str)],
    ) -> Res<()> {
        let mut assembler = FrameAssembler::new(limits)
            .map_err(|violation| format!("the assembly policy: {violation:?}"))?;
        for (principal, key, stage) in principals {
            assembler.register_principal(principal, *key, stage);
        }
        self.assembler = assembler;
        self.now_ms = 1_000;
        if self
            .mailbox
            .lock()
            .map_err(|_| "the mailbox is poisoned".to_string())?
            .take()
            .is_some()
        {
            return Err("the mailbox held a payload between cases".to_string());
        }
        Ok(())
    }

    /// Send `payload` from `from`: over a fresh authenticated connection, through the host's
    /// access decision, the intake Pod and the assembler.
    pub fn send(
        &mut self,
        counters: &mut Counters,
        from: &Peer,
        payload: &TypedPayload,
    ) -> Res<Sent> {
        self.send_as(counters, from, None, payload)
    }

    /// As [`Self::send`], but offers the payload to the assembler under `claimed` instead of the
    /// principal the connection authenticated: the wrong binding, for a control only.
    pub fn send_as(
        &mut self,
        counters: &mut Counters,
        from: &Peer,
        claimed: Option<&str>,
        payload: &TypedPayload,
    ) -> Res<Sent> {
        self.next_request += 1;
        let request = PodRequest {
            addressed_to: self.host_id.clone(),
            request_id: self.next_request,
            capability: CapabilityId::from(CAPABILITY),
            expect_protocol: 1,
            payload: payload.clone(),
        };
        if encode_request(&request).is_err() {
            counters.add("codec_refusals", 1);
            return Ok(Sent::Codec);
        }
        let address = self
            .book
            .locate(&NodeId(self.host_id.clone()))
            .map_err(|e| format!("locating the host: {e:?}"))?;
        let host = &self.host;
        let client = &from.client;
        let outcome = self.rt.block_on(async {
            tokio::time::timeout(EXCHANGE_TIMEOUT, async {
                tokio::join!(host.serve_once(), client.request(address, &request))
            })
            .await
        });
        let (served, answered) = outcome.map_err(|_| "an exchange timed out".to_string())?;
        let served = served.map_err(|e| format!("the host failed to serve: {e}"))?;
        let answered = answered.map_err(|e| format!("the client failed to ask: {e}"))?;
        self.connections += 1;
        counters.add("requests", 1);
        if served.peer.public_key != from.id {
            return Err("the host authenticated another peer than the sender".to_string());
        }
        if served.request_id != Some(self.next_request) {
            return Err("the host served another request id".to_string());
        }
        let taken = self
            .mailbox
            .lock()
            .map_err(|_| "the mailbox is poisoned".to_string())?
            .take();
        match answered.outcome {
            PodOutcome::Refused { code } => {
                if taken.is_some() {
                    return Err("a refused request left a payload in the mailbox".to_string());
                }
                Ok(Sent::Exchange(Exchange {
                    host_refusal: Some(code),
                    verdict: None,
                }))
            }
            PodOutcome::Answered { output } => {
                let received = taken.ok_or("an answered request left the mailbox empty")?;
                if received != *payload {
                    counters.add("mailbox_payload_mismatches", 1);
                }
                if output.bytes != sha256(&payload.bytes).to_vec() {
                    counters.add("ack_digest_mismatches", 1);
                }
                let principal = claimed.unwrap_or(&served.peer.public_key).to_owned();
                let verdict =
                    self.assembler
                        .offer(self.now_ms, &principal, self.connections, &received);
                Ok(Sent::Exchange(Exchange {
                    host_refusal: None,
                    verdict: Some(verdict),
                }))
            }
        }
    }

    /// Close every endpoint.
    pub fn close(&self, peers: &[&Peer]) {
        self.rt.block_on(async {
            for peer in peers {
                peer.client.close().await;
            }
            self.host.close().await;
        });
    }
}

/// Bind one client endpoint on `rt`.
pub fn bind_peer(rt: &Runtime, key: SigningKey, stage: &'static str) -> Res<Peer> {
    let mut client = rt
        .block_on(PodClient::bind())
        .map_err(|e| format!("binding a client: {e}"))?;
    client.with_request_timeout(CLIENT_TIMEOUT);
    let id = client.identity().public_key;
    Ok(Peer {
        client,
        id,
        key,
        stage,
    })
}

/// The key of a runtime that is not the upstream one, from a fixed seed.
pub fn other_key(tag: u8) -> SigningKey {
    SigningKey::from_bytes(&[tag; 32])
}
