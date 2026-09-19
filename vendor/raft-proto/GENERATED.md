# Prost generation provenance

Generated from the unmodified raft-proto 0.7.0 eraftpb.proto by protobuf-build
0.15.1 / prost 0.11.9 during diagnostic run 35416816312. Original output hashes:

```json
{
  "mod.rs": "7cda43dea9a215159360b2eb338d488e911a677833a6dc4025074d1dbf12e225",
  "wrapper_eraftpb.rs": "8f09ba45b86cf26ea195af1f231ae32da97784c7c823d85294d986383c29c100",
  "eraftpb.rs": "e304c57de1767b77241e1e880c3cccbefc3cdc71df96d22f2519012f268ae36b"
}
```

The checked-in eraftpb.rs is unchanged. The accessor file retains field accessors
and replaces default-ref lookup with the PTR prost codec trait. All legacy
protobuf2 Message/Clear implementations (including unimplemented reflection/parser
methods) are removed. src/codec.rs delegates encoding/decoding to prost itself.
Build generation is disabled explicitly; the original schema/build source is
retained as provenance. This fork supports prost-codec only, matching PTR's
selected adapter. No protobuf-codec compatibility is claimed.
