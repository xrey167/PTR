//! One member of a group, carrying raft messages over `ALPN_RAFT`.
use crate::frame::{decode_batch, encode_batch, FrameError};
use ptr_ledger::{
    decode_message, encode_message, CommittedEvent, LedgerEvent, RaftMessage, RaftNode,
};
use ptr_net::{IrohTransport, NodeIdentity, ALPN_RAFT};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

/// Whether a refusal means "that member is away" rather than "something is wrong".
fn is_unreachable(error: &ClusterError) -> bool {
    matches!(
        error,
        ClusterError::Transport(_) | ClusterError::NoAddress { .. }
    )
}

/// How long one exchange waits for a peer before calling it unreachable.
///
/// A default rather than a rule: the right deadline depends on the deployment, and
/// what matters here is that there *is* one. Without it a member that stopped
/// answering holds up every round it appears in, and a single dead host stalls a
/// group that still has a majority.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Bound on how many delivery rounds one call will drive before giving up.
///
/// A group that never goes quiet is a defect, and a caller that loops forever
/// hides it. This turns it into an error with a name.
const MAX_ROUNDS: usize = 64;

/// Where a member can be reached, and which key proves it is that member.
///
/// Both halves are needed and neither is optional. The address says where to send;
/// the key is what an accepted connection is checked against, and without it an
/// address is just a hint that anyone could answer.
#[derive(Clone, Debug)]
pub struct MemberAddress {
    pub id: u64,
    pub identity: NodeIdentity,
    pub address: ptr_net::EndpointAddr,
}

/// Members a call could not reach, and why.
///
/// Unreachability is ordinary: a member is restarting, or its host is gone. A
/// leader whose proposal failed because one follower was away would be unable to
/// commit anything while any member is down, which is the opposite of what a
/// majority is for. So a transport failure toward one peer is reported rather than
/// raised, and the caller decides whether it matters.
///
/// Refusals that are *not* about reachability — a forged sender, a malformed frame,
/// a node that would not step — do abort the call. They mean something is wrong,
/// not that somebody is away.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Unreached(pub Vec<(u64, ClusterError)>);

impl Unreached {
    /// Whether every peer answered.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The members that did not answer, each once and in id order.
    ///
    /// One call can try a member in several rounds, so the attempts in `0` may name
    /// it more than once — that is a record of what was attempted, whereas this is
    /// the answer to "who is away".
    pub fn ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.0.iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

/// Why a cluster operation was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClusterError {
    /// The transport refused or failed.
    Transport(String),
    /// The frame was malformed.
    Frame(FrameError),
    /// A message inside a well-formed frame is not a raft message.
    Undecodable { position: usize, reason: String },
    /// The connection authenticated a key that belongs to no member of this group.
    UnknownPeer { key: String },
    /// The messages claim to come from a member other than the one the connection
    /// authenticated. This is the case the whole layer exists for.
    ForgedSender { authenticated: u64, claimed: u64 },
    /// A message addressed to a member other than this one.
    Misaddressed { to: u64, expected: u64 },
    /// The node refused to step or propose.
    Node(String),
    /// This member does not know how to reach a peer it must send to.
    NoAddress { id: u64 },
    /// Delivery did not settle within the bound.
    NotQuiescent,
}

impl ClusterError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Transport(_) => "PTR_CLUSTER_TRANSPORT",
            Self::Frame(_) => "PTR_CLUSTER_FRAME",
            Self::Undecodable { .. } => "PTR_CLUSTER_UNDECODABLE",
            Self::UnknownPeer { .. } => "PTR_CLUSTER_UNKNOWN_PEER",
            Self::ForgedSender { .. } => "PTR_CLUSTER_FORGED_SENDER",
            Self::Misaddressed { .. } => "PTR_CLUSTER_MISADDRESSED",
            Self::Node(_) => "PTR_CLUSTER_NODE",
            Self::NoAddress { .. } => "PTR_CLUSTER_NO_ADDRESS",
            Self::NotQuiescent => "PTR_CLUSTER_NOT_QUIESCENT",
        }
    }
}

impl From<FrameError> for ClusterError {
    /// Carry a framing refusal through unchanged.
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

impl fmt::Display for ClusterError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for ClusterError {}

/// A raft member with an authenticated endpoint.
///
/// Not a daemon: nothing here ticks or retries on its own. A caller decides when to
/// accept, when to send and when to give up, which is also what keeps the tests
/// free of sleeps.
pub struct ClusterMember {
    id: u64,
    node: Mutex<RaftNode>,
    transport: IrohTransport,
    peers: Vec<MemberAddress>,
    request_timeout: Duration,
}

impl ClusterMember {
    /// Bind an endpoint for `ALPN_RAFT` and open this member's durable state.
    pub async fn bind(dir: &Path, id: u64, voters: &[u64]) -> Result<Self, ClusterError> {
        let transport = IrohTransport::bind(&[ALPN_RAFT])
            .await
            .map_err(ClusterError::Transport)?;
        let node = RaftNode::open(dir, id, voters).map_err(ClusterError::Node)?;
        Ok(Self {
            id,
            node: Mutex::new(node),
            transport,
            peers: Vec::new(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        })
    }

    /// This member's id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The authenticated identity other members must expect from this one.
    pub fn identity(&self) -> NodeIdentity {
        self.transport.identity()
    }

    /// Where other members should send.
    pub fn address(&self) -> ptr_net::EndpointAddr {
        self.transport.direct_addr()
    }

    /// How long to wait for one peer before treating it as unreachable.
    pub fn with_request_timeout(&mut self, timeout: Duration) {
        self.request_timeout = timeout;
    }

    /// Admit a peer by id, key and address.
    pub fn admit(&mut self, peer: MemberAddress) {
        self.peers.retain(|known| known.id != peer.id);
        self.peers.push(peer);
    }

    /// Whether this member currently believes it leads.
    pub fn is_leader(&self) -> bool {
        self.locked().is_leader()
    }

    /// The term this member is in.
    pub fn term(&self) -> u64 {
        self.locked().term()
    }

    /// The events this member has applied, in commit order.
    pub fn committed_events(&self) -> Vec<CommittedEvent> {
        self.locked().committed_events().to_vec()
    }

    /// The raft log position this member considers committed.
    pub fn raft_committed(&self) -> u64 {
        self.locked().raft_committed()
    }

    /// Record this member's application state at a committed position.
    ///
    /// The payload is the application's — for PTR, a PTRCS002 compacted snapshot.
    /// This layer carries it and has no opinion about what is inside.
    pub fn record_snapshot(&self, index: u64, state: Vec<u8>) -> Result<(), ClusterError> {
        self.locked()
            .record_snapshot(index, state)
            .map_err(ClusterError::Node)
    }

    /// Take the snapshot a leader installed on this member, if one arrived.
    ///
    /// What arrives is the elected leader's. Because the sender is the
    /// authenticated connection, this rules out a *stranger* supplying a
    /// consistent pair — it does not rule out the leader itself. A deployment
    /// that retains anchors elsewhere should use
    /// [`ClusterMember::take_installed_snapshot_matching`] instead.
    pub fn take_installed_snapshot(&self) -> Option<ptr_ledger::InstalledSnapshot> {
        self.locked().take_installed_snapshot()
    }

    /// Take an installed snapshot only if it matches an anchor retained out of
    /// band.
    ///
    /// A mismatch refuses and leaves the snapshot installed, so the member stays
    /// blocked rather than continuing as though nothing had arrived.
    pub fn take_installed_snapshot_matching(
        &self,
        retained: ptr_ledger::RetainedSnapshotAnchor,
    ) -> Result<ptr_ledger::InstalledSnapshot, ptr_ledger::SnapshotAnchorMismatch> {
        self.locked().take_installed_snapshot_matching(retained)
    }

    /// Propose a change to who votes in this group, and drive it to quiescence.
    ///
    /// One voter at a time, and only from the leader — both for the reason
    /// [`ptr_ledger::MembershipChange`] gives: consecutive configurations have to
    /// overlap in a majority, or the group can split into two that each believe
    /// they can commit.
    ///
    /// A member being added has to be admitted first, or its peers have nowhere
    /// to send to it.
    pub async fn propose_membership(
        &self,
        change: ptr_ledger::MembershipChange,
    ) -> Result<Unreached, ClusterError> {
        let messages = self
            .locked()
            .propose_membership(change)
            .map_err(ClusterError::Node)?;
        self.drive(messages).await
    }

    /// The voters this member currently believes the group has, ascending.
    pub fn voters(&self) -> Vec<u64> {
        self.locked().voters()
    }

    /// Account for an installed snapshot: how many events its payload covers.
    pub fn resume_with_applied(&self, applied_events: u64) {
        self.locked().resume_with_applied(applied_events);
    }

    /// Stand for election and drive the result to quiescence.
    pub async fn campaign(&self) -> Result<Unreached, ClusterError> {
        let messages = self.locked().campaign().map_err(ClusterError::Node)?;
        self.drive(messages).await
    }

    /// Advance this member's logical clock once and deliver what it produces.
    pub async fn tick(&self) -> Result<Unreached, ClusterError> {
        let messages = self.locked().tick().map_err(ClusterError::Node)?;
        self.drive(messages).await
    }

    /// Propose an event and drive the result to quiescence.
    ///
    /// Whether the proposal was *decided* is not this call's answer: that is what
    /// [`ClusterMember::committed_events`] says once a majority has replied. A peer
    /// in the returned [`Unreached`] simply did not answer.
    pub async fn propose(&self, event: LedgerEvent) -> Result<Unreached, ClusterError> {
        let messages = self.locked().propose(event).map_err(ClusterError::Node)?;
        self.drive(messages).await
    }

    /// Accept one connection, step what it carries, and answer with what that
    /// produced for the sender.
    ///
    /// Returns the messages the step produced for **other** members, which this
    /// member's own loop must deliver: driving them from inside the response would
    /// open a connection to a peer while that peer waits on this one.
    pub async fn serve_once(&self) -> Result<Vec<RaftMessage>, ClusterError> {
        let incoming = self
            .transport
            .accept_once(crate::MAX_FRAME_BYTES)
            .await
            .map_err(ClusterError::Transport)?;

        // The sender is the connection, not the payload. Everything below depends
        // on this line: a peer may write any id into a raft message, and a receiver
        // that trusts it accepts votes and appends from a member that never sent
        // them.
        let authenticated = self.authenticate(&incoming.peer);
        let outcome = authenticated.and_then(|from| self.accept(from, &incoming.payload));

        match outcome {
            Ok((reply, leftover)) => {
                let frame = encode_batch(&reply)?;
                incoming
                    .respond(&frame)
                    .await
                    .map_err(ClusterError::Transport)?;
                Ok(leftover)
            }
            Err(error) => {
                // Answer with an empty batch rather than leaving the peer hanging.
                // The refusal is this member's business; a stalled connection would
                // be the peer's problem instead.
                let empty = encode_batch(&[])?;
                let _ = incoming.respond(&empty).await;
                Err(error)
            }
        }
    }

    /// Map an authenticated key onto an admitted member id.
    fn authenticate(&self, peer: &NodeIdentity) -> Result<u64, ClusterError> {
        self.peers
            .iter()
            .find(|known| known.identity.public_key == peer.public_key)
            .map(|known| known.id)
            .ok_or_else(|| ClusterError::UnknownPeer {
                key: peer.public_key.clone(),
            })
    }

    /// Decode, check and step one frame from an authenticated member.
    ///
    /// Returns `(reply, leftover)`: the encoded messages for the sender, and the
    /// ones for other members.
    fn accept(
        &self,
        from: u64,
        payload: &[u8],
    ) -> Result<(Vec<Vec<u8>>, Vec<RaftMessage>), ClusterError> {
        let encoded = decode_batch(payload)?;
        let mut messages = Vec::with_capacity(encoded.len());
        for (position, bytes) in encoded.iter().enumerate() {
            let message = decode_message(bytes)
                .map_err(|reason| ClusterError::Undecodable { position, reason })?;
            if message.from != from {
                return Err(ClusterError::ForgedSender {
                    authenticated: from,
                    claimed: message.from,
                });
            }
            if message.to != self.id {
                return Err(ClusterError::Misaddressed {
                    to: message.to,
                    expected: self.id,
                });
            }
            messages.push(message);
        }

        // The whole batch is checked before any of it is stepped: a caller that
        // stepped the first message and then refused the second would have applied
        // part of a frame it rejected.
        let mut produced = Vec::new();
        {
            let mut node = self.locked();
            for message in messages {
                produced.extend(node.step(message).map_err(ClusterError::Node)?);
            }
        }
        let (reply, leftover): (Vec<RaftMessage>, Vec<RaftMessage>) =
            produced.into_iter().partition(|message| message.to == from);
        let encoded_reply = reply
            .iter()
            .map(encode_message)
            .collect::<Result<Vec<Vec<u8>>, String>>()
            .map_err(ClusterError::Node)?;
        Ok((encoded_reply, leftover))
    }

    /// Send messages, step the replies, and repeat until nothing is in flight.
    ///
    /// A peer that cannot be reached is recorded and skipped; the rest of the round
    /// proceeds. A leader that abandoned a proposal because one follower was down
    /// could not commit while any member is down.
    async fn drive(&self, initial: Vec<RaftMessage>) -> Result<Unreached, ClusterError> {
        let mut unreached = Unreached::default();
        let mut queue = initial;
        for _ in 0..MAX_ROUNDS {
            if queue.is_empty() {
                return Ok(unreached);
            }
            let mut grouped: HashMap<u64, Vec<Vec<u8>>> = HashMap::new();
            for message in &queue {
                let encoded = encode_message(message).map_err(ClusterError::Node)?;
                grouped.entry(message.to).or_default().push(encoded);
            }
            // Sorted, so one call's sequence of connections is the same every time
            // it is replayed with the same inputs.
            let mut destinations: Vec<u64> = grouped.keys().copied().collect();
            destinations.sort_unstable();

            let mut next = Vec::new();
            for destination in destinations {
                let batch = grouped
                    .remove(&destination)
                    .expect("destination is present");
                let replies = match self.exchange(destination, &batch).await {
                    Ok(replies) => replies,
                    Err(error) if is_unreachable(&error) => {
                        unreached.0.push((destination, error));
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                let mut node = self.locked();
                for message in replies {
                    next.extend(node.step(message).map_err(ClusterError::Node)?);
                }
            }
            queue = next;
        }
        Err(ClusterError::NotQuiescent)
    }

    /// One request/response with a peer, returning the messages it answered with.
    async fn exchange(&self, to: u64, batch: &[Vec<u8>]) -> Result<Vec<RaftMessage>, ClusterError> {
        let peer = self
            .peers
            .iter()
            .find(|known| known.id == to)
            .ok_or(ClusterError::NoAddress { id: to })?;
        let frame = encode_batch(batch)?;
        let response = tokio::time::timeout(
            self.request_timeout,
            self.transport.request(
                peer.address.clone(),
                ALPN_RAFT,
                &frame,
                crate::MAX_FRAME_BYTES,
            ),
        )
        .await
        .map_err(|_| ClusterError::Transport("the peer did not answer in time".to_owned()))?
        .map_err(ClusterError::Transport)?;

        let encoded = decode_batch(&response)?;
        let mut messages = Vec::with_capacity(encoded.len());
        for (position, bytes) in encoded.iter().enumerate() {
            let message = decode_message(bytes)
                .map_err(|reason| ClusterError::Undecodable { position, reason })?;
            // A reply is checked exactly as a request is: the peer answering is the
            // peer that was asked, and the answer is addressed here.
            if message.from != to {
                return Err(ClusterError::ForgedSender {
                    authenticated: to,
                    claimed: message.from,
                });
            }
            if message.to != self.id {
                return Err(ClusterError::Misaddressed {
                    to: message.to,
                    expected: self.id,
                });
            }
            messages.push(message);
        }
        Ok(messages)
    }

    /// Borrow the node.
    ///
    /// Panics if a previous holder panicked: a node whose write sequence was
    /// interrupted is not a node to keep stepping.
    fn locked(&self) -> std::sync::MutexGuard<'_, RaftNode> {
        self.node.lock().expect("raft node lock is not poisoned")
    }

    /// Close the endpoint.
    pub async fn close(&self) {
        self.transport.close().await;
    }
}
