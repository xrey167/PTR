fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    std::env::set_var("PROTOC", protoc);
    let protos = [
        "../../proto/podwire.proto",
        "../../proto/model_events.proto",
        "../../proto/events.proto",
        "../../proto/raft_transport.proto",
    ];
    prost_build::Config::new()
        .compile_protos(&protos, &["../../proto"])
        .expect("compile PTR protobuf schemas");
    for proto in protos {
        println!("cargo:rerun-if-changed={proto}");
    }
}
