use ptr_storage::{InMemoryStore,ObjectStore}; use ptr_types::ArtifactId;
#[test] fn in_memory_store_roundtrips(){ let mut s=InMemoryStore::default(); let id=ArtifactId::from("a"); s.put(id.clone(),vec![1,2]); assert_eq!(s.get(&id),Some(&[1,2][..])); }
