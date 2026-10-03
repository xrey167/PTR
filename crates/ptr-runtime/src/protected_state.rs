use ptr_storage::{
    IntegrityReport, ProtectedHandle, ProtectedRecord, ProtectedStateError, ProtectedStateStore,
};

/// Runtime-owned boundary for protected persistence.  The storage backend is
/// deliberately injected so runtime admission and recovery do not depend on
/// a particular key provider or filesystem implementation.
pub struct ProtectedStateCoordinator<S> {
    store: S,
}

impl<S: ProtectedStateStore> ProtectedStateCoordinator<S> {
    pub fn new(store: S) -> Self {
        Self { store }
    }

    pub fn seal(
        &mut self,
        record: ProtectedRecord,
    ) -> Result<ProtectedHandle, ProtectedStateError> {
        self.store.seal(record)
    }

    pub fn open(&self, handle: &ProtectedHandle) -> Result<Vec<u8>, ProtectedStateError> {
        self.store.open(handle)
    }

    pub fn verify(&self, handle: &ProtectedHandle) -> Result<IntegrityReport, ProtectedStateError> {
        self.store.verify(handle)
    }

    pub fn rotate_key(
        &mut self,
        key_id: ptr_types::KeyId,
        key: [u8; 32],
    ) -> Result<(), ProtectedStateError> {
        self.store.rotate_key(key_id, key)
    }

    pub fn into_inner(self) -> S {
        self.store
    }
}
