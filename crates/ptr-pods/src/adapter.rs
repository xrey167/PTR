use crate::{DynPod, NeuralPodDescriptor, NeuralPodExecutor, PodManifest, TypedPayload};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterError {
    InvalidDescriptor,
    ManifestInputMismatch,
    ManifestOutputMismatch,
    ManifestHashMismatch,
}

pub struct NeuralPodAdapter<E> {
    manifest: PodManifest,
    descriptor: NeuralPodDescriptor,
    executor: E,
}

impl<E> NeuralPodAdapter<E>
where
    E: NeuralPodExecutor,
{
    pub fn new(
        manifest: PodManifest,
        descriptor: NeuralPodDescriptor,
        executor: E,
    ) -> Result<Self, AdapterError> {
        descriptor
            .validate()
            .map_err(|_| AdapterError::InvalidDescriptor)?;
        if !manifest.accepts.contains(&descriptor.input_schema) {
            return Err(AdapterError::ManifestInputMismatch);
        }
        if !manifest.produces.contains(&descriptor.output_schema) {
            return Err(AdapterError::ManifestOutputMismatch);
        }
        if manifest.digest() != descriptor.manifest_hash {
            return Err(AdapterError::ManifestHashMismatch);
        }
        Ok(Self {
            manifest,
            descriptor,
            executor,
        })
    }

    pub fn descriptor(&self) -> &NeuralPodDescriptor {
        &self.descriptor
    }
}

impl<E> DynPod for NeuralPodAdapter<E>
where
    E: NeuralPodExecutor + Send + Sync,
    E::Error: std::fmt::Debug,
{
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        if input.type_id != self.descriptor.input_schema {
            return Err(format!("input type {} is not admitted", input.type_id));
        }
        let mut lease = self
            .executor
            .activate(&self.descriptor)
            .map_err(|error| format!("activation failed: {error:?}"))?;
        let output = match self.executor.infer(&mut lease, input) {
            Ok(output) => output,
            Err(error) => {
                let recovery = self.executor.abort(lease);
                return Err(match recovery {
                    Ok(()) => format!("inference failed: {error:?}"),
                    Err(recovery) => {
                        format!("inference failed: {error:?}; lease recovery failed: {recovery:?}")
                    }
                });
            }
        };
        self.executor
            .release(lease)
            .map_err(|error| format!("release failed: {error:?}"))?;
        if !self.manifest.produces.contains(&output.type_id) {
            return Err(format!("output type {} is not admitted", output.type_id));
        }
        Ok(output)
    }
}
