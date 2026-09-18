use ptr_net::{ALPN_RAFT,ALPN_PODWIRE};
#[test] fn alpns_are_distinct(){ assert_ne!(ALPN_RAFT,ALPN_PODWIRE); }
