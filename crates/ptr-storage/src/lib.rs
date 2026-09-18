use ptr_types::ArtifactId;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactRef {
    pub id: ArtifactId,
    pub locations: Vec<String>,
}

pub trait ObjectStore {
    fn put(&mut self, id: ArtifactId, bytes: Vec<u8>);
    fn get(&self, id: &ArtifactId) -> Option<&[u8]>;
}

#[derive(Default)]
pub struct InMemoryStore {
    values: BTreeMap<ArtifactId, Vec<u8>>,
}
impl ObjectStore for InMemoryStore {
    fn put(&mut self, id: ArtifactId, bytes: Vec<u8>) {
        self.values.insert(id, bytes);
    }
    fn get(&self, id: &ArtifactId) -> Option<&[u8]> {
        self.values.get(id).map(Vec::as_slice)
    }
}
