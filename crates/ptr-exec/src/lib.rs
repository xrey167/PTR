//! Bounded isolate execution semantics. Tokio/Compio are transport substrates, not the semantic scheduler contract.

mod isolate;
mod mailbox;

pub use isolate::{Effect, Isolate, ReplayTrace};
pub use mailbox::{Mailbox, MailboxSender, Receive, SendError};
