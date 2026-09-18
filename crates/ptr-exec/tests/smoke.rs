use ptr_exec::{Mailbox,SendError};
#[test] fn bounded_mailbox_returns_full_value(){ let m=Mailbox::bounded(1); m.try_send(1).unwrap(); assert_eq!(m.try_send(2),Err(SendError::Full(2))); }
