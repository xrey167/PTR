# Network and Protocol

PTR separates semantic protocol from encoding. PodWire is the semantic contract. Protobuf/prost is a candidate for versioned network control frames, rkyv for trusted local hot data, and DLPack for GPU tensors. Iroh/QUIC is the primary candidate for authenticated node connectivity and separate ALPN protocols.
