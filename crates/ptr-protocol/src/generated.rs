pub mod podwire {
    include!(concat!(env!("OUT_DIR"), "/ptr.podwire.v1.rs"));
}
pub mod model {
    include!(concat!(env!("OUT_DIR"), "/ptr.model.v1.rs"));
}
pub mod events {
    include!(concat!(env!("OUT_DIR"), "/ptr.events.v1.rs"));
}
pub mod raft {
    include!(concat!(env!("OUT_DIR"), "/ptr.raft.v1.rs"));
}
