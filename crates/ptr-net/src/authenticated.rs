//! A peer identity that only a transport can produce.
//!
//! [`NodeId`] is a string and anyone can write one. Every layer that admits a peer
//! used to take one and document that the caller must pass what the transport
//! proved, never what a payload claimed — a rule kept in one place and unwritten
//! everywhere else. [`AuthenticatedPeer`] makes it a type: it has no public
//! constructor, so the only way to hold one is to have been handed it by this
//! crate's transport, which builds it from the key the QUIC handshake verified.
//!
//! That is the decision recorded as D1 in issue #23: `ptr-runtime` depends on this
//! crate (its base, which has no transport dependency) so that `admit_peer` can
//! demand an `AuthenticatedPeer`, and a composition written anywhere cannot pass an
//! identity the transport did not prove. It does not compile.
//!
//! ```compile_fail
//! // A struct literal does not compile: the field is private.
//! let forged = ptr_net::AuthenticatedPeer { id: ptr_types::NodeId::from("anyone") };
//! ```
//!
//! ```compile_fail
//! // Nor is there a conversion from a bare id.
//! let forged: ptr_net::AuthenticatedPeer = ptr_types::NodeId::from("anyone").into();
//! ```
//!
//! **Tests** need peers without a network. The `test-support` feature adds
//! [`AuthenticatedPeer::assume_authenticated`], and `scripts/check_repo.py` refuses
//! any manifest that enables that feature outside `[dev-dependencies]`, so a
//! production build of any package in this repository cannot reach it.

use ptr_types::NodeId;

/// A peer key a transport authenticated.
///
/// Holding one is evidence that a connection proved possession of this key's
/// secret half. It says nothing about what the peer may do: authenticating a peer
/// is not authorizing an action, and every authority table keyed by it is still the
/// host's policy.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AuthenticatedPeer {
    id: NodeId,
}

impl AuthenticatedPeer {
    /// Built only where a transport has verified the key. Crate-private on purpose.
    #[cfg_attr(not(feature = "iroh-backend"), allow(dead_code))]
    pub(crate) fn from_transport(id: NodeId) -> Self {
        Self { id }
    }

    /// Treat a bare id as authenticated, for tests that have no network.
    ///
    /// Only with the `test-support` feature, which `scripts/check_repo.py` allows
    /// in `[dev-dependencies]` alone.
    #[cfg(feature = "test-support")]
    pub fn assume_authenticated(id: NodeId) -> Self {
        Self { id }
    }

    /// The authenticated key, as the id every policy table is keyed by.
    pub fn id(&self) -> &NodeId {
        &self.id
    }

    /// The authenticated key in its textual form.
    pub fn public_key(&self) -> &str {
        &self.id.0
    }
}

impl std::fmt::Display for AuthenticatedPeer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.id.0)
    }
}
