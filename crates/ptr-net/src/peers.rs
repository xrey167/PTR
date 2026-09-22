//! Where a peer can be reached, on the deployment's authority rather than anyone
//! else's.
//!
//! An address is not an identity. A [`NodeId`] is a public key and says *who*; an
//! address says *where*, changes when a host moves, and — on its own — is a hint
//! that anyone could answer. `ptr-cluster` already says this in `MemberAddress`'s own
//! doc comment and pairs the two. The two request/response wires did not: a caller
//! handed them a bare address, so nothing recorded whose authority that address came
//! from.
//!
//! This module is that authority, and it is shaped so the mistakes it exists to
//! prevent are unrepresentable rather than checked:
//!
//! - **An address cannot be filed under another peer's name.** [`PeerBook::record`]
//!   takes the address alone and derives the key from the id the address itself
//!   carries. There is no `(peer, address)` pair to transpose, so an attacker's
//!   address recorded by mistake lands under the *attacker's* id and a requester
//!   asking for the honest peer still does not find it.
//! - **A requester cannot dial an address it was handed.** The only type either wire
//!   client accepts is [`PeerAddress`], which has no public constructor and can be
//!   obtained only from [`PeerBook::locate`]. An address read out of a payload, or
//!   learned from a peer, cannot be turned into one.
//! - **An unrecorded peer is refused before a connection is attempted**, so the
//!   refusal costs nothing and reveals nothing.
//!
//! What remains, and is not closed here: an address that is *wrong for the right id*
//! — stale, or crafted to point the honest id at somebody else's socket. The book
//! cannot tell, and neither can anything else at this layer. What happens instead is
//! that the dial fails, because [`crate::IrohTransport::request`] refuses a
//! connection whose authenticated key is not the one asked for. So a wrong address
//! costs a failed request rather than a request served by the wrong node. Finding
//! the *right* address for an id is discovery, which this module deliberately does
//! not do.
use crate::EndpointAddr;
use ptr_types::NodeId;
use std::collections::BTreeMap;
use std::fmt;

/// An address a requester is entitled to dial.
///
/// There is no public constructor: one of these comes from [`PeerBook::locate`] and
/// from nowhere else. That is the same device `VerifiedDispatch` uses in the runtime,
/// applied to the question *where did this address come from* — and the answer is
/// always "the book the deployment installed".
///
/// ```compile_fail
/// use ptr_net::PeerAddress;
/// use ptr_types::NodeId;
/// let _ = PeerAddress {
///     peer: NodeId::from("somebody"),
///     address: unimplemented!(),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct PeerAddress {
    peer: NodeId,
    address: EndpointAddr,
}

impl PeerAddress {
    /// Which peer this address is for.
    ///
    /// A wire client checks the request it is about to send against this, so a
    /// request composed for one host and dialled at another fails locally instead of
    /// spending a round trip to be refused.
    pub fn peer(&self) -> &NodeId {
        &self.peer
    }

    /// The address, for the transport.
    pub fn into_address(self) -> EndpointAddr {
        self.address
    }
}

/// Why an address could not be resolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BookError {
    /// No entry for this peer. A requester reaches nothing the deployment has not
    /// written down.
    Unknown { peer: NodeId },
}

impl BookError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Unknown { .. } => "PTR_NET_UNKNOWN_PEER",
        }
    }
}

impl fmt::Display for BookError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for BookError {}

/// Where each peer the deployment knows about can be reached.
///
/// Installed by the host. Nothing here reads bytes, so there is no path by which a
/// peer, a payload or the network can add an entry — the absence of such a method is
/// the mechanism.
#[derive(Clone, Debug, Default)]
pub struct PeerBook {
    entries: BTreeMap<NodeId, EndpointAddr>,
}

impl PeerBook {
    /// An empty book: a requester using it reaches nobody.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record where a peer can be reached, returning the address it replaces.
    ///
    /// **Replaces**, where [`crate::NodeIdentity`]-keyed *authority* tables refuse a
    /// second entry. The difference is deliberate and it is about what the entry
    /// means: an address is a fact about where a node is, and a node legitimately
    /// moves, so refusing an update would leave a deployment unable to follow one.
    /// An admission grant is a decision, and a decision that changed silently would
    /// be authority nobody took.
    ///
    /// The key is the id the address carries, so this cannot file an address under a
    /// peer it does not belong to.
    pub fn record(&mut self, address: EndpointAddr) -> Option<EndpointAddr> {
        let peer = NodeId(address.id.to_string());
        self.entries.insert(peer, address)
    }

    /// Forget a peer. A requester cannot reach it from the next lookup onward,
    /// because the address is read from this book at every lookup.
    pub fn forget(&mut self, peer: &NodeId) -> bool {
        self.entries.remove(peer).is_some()
    }

    /// Where to dial a peer, or a refusal.
    pub fn locate(&self, peer: &NodeId) -> Result<PeerAddress, BookError> {
        let address = self
            .entries
            .get(peer)
            .cloned()
            .ok_or_else(|| BookError::Unknown { peer: peer.clone() })?;
        Ok(PeerAddress {
            peer: peer.clone(),
            address,
        })
    }

    /// Whether this peer has an address recorded.
    pub fn knows(&self, peer: &NodeId) -> bool {
        self.entries.contains_key(peer)
    }

    /// How many peers are recorded.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the book is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
