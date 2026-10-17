//! The guardian process (doc 42 rule 3a): `ptr-worker guardian --channel-fd <n> --backend
//! fake|cuda`.
//!
//! The guardian holds the staging buffers of the CUDA carriers on behalf of the supervisor, so
//! that no worker owns an exported allocation and a crashing producer cannot free one. It holds
//! no model, serves no Pod and reads nothing but the channel its supervisor gave it: the
//! protocol (`ptr_fabric::serve_guardian`) has allocate, fill, export, scrub and release, and a
//! request that arrives with a descriptor is refused.
//!
//! It is **not** seccomp-confined, unlike a worker: it is the supervisor's own code from the same
//! image, and it needs the driver calls the worker allowlist forbids. It still runs in a cgroup
//! of its own and a network namespace with no interface (see
//! `ptr_fabric::GuardianHostConfig`), with no capabilities and `PR_SET_NO_NEW_PRIVS`.
//!
//! It exits when the supervisor closes its end of the channel, freeing every buffer first, and
//! the supervisor ends it only after every worker domain is reaped.
//!
//! `--backend fake` is the CPU double (`ptr_fabric::FakeBackend`, memfd-backed) that the tests
//! and a host without a GPU use; `--backend cuda` needs a build with feature `cuda-ipc-core`
//! (or `cuda-ipc`) and a driver, and has not run on a GPU.

use std::fmt;

use ptr_fabric::{
    serve_guardian, Channel, ChannelError, FakeBackend, GuardianError, GuardianLimits,
    GuardianSummary,
};

/// Which staging backend the guardian holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendChoice {
    /// The CPU double.
    Fake,
    /// The CUDA driver (VMM staging allocations).
    Cuda,
}

impl BackendChoice {
    /// The name `--backend` takes.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "fake" => Some(Self::Fake),
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }

    /// The name `--backend` takes, and `--staging` for the workers that import its buffers.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fake => "fake",
            Self::Cuda => "cuda",
        }
    }
}

/// Why the guardian could not run or stopped abnormally.
#[derive(Debug)]
pub enum GuardianRunError {
    /// The inherited descriptor is not a usable channel.
    Channel(ChannelError),
    /// The CUDA backend was asked for and is not available (not compiled in, or no driver).
    NoCudaBackend(String),
    /// The serve loop failed.
    Serve(GuardianError),
}

impl fmt::Display for GuardianRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Channel(error) => write!(formatter, "channel: {error}"),
            Self::NoCudaBackend(reason) => write!(formatter, "no CUDA backend: {reason}"),
            Self::Serve(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for GuardianRunError {}

/// Take over the inherited channel at `fd` and serve the supervisor on it until it closes.
pub fn run_guardian(fd: i32, backend: BackendChoice) -> Result<GuardianSummary, GuardianRunError> {
    // SAFETY: the supervisor started this process with the channel at descriptor `fd` and
    // nothing else in the process owns it; the channel takes ownership.
    let channel = unsafe { Channel::from_inherited_fd(fd) }.map_err(GuardianRunError::Channel)?;
    serve_on(&channel, backend)
}

/// Serve on an already open channel. The backend is created before the first request, so a
/// driver that is missing is reported before the supervisor sends anything.
pub fn serve_on(
    channel: &Channel,
    backend: BackendChoice,
) -> Result<GuardianSummary, GuardianRunError> {
    let limits = GuardianLimits::default();
    match backend {
        BackendChoice::Fake => {
            serve_guardian(channel, FakeBackend::new(), limits).map_err(GuardianRunError::Serve)
        }
        BackendChoice::Cuda => serve_cuda(channel, limits),
    }
}

#[cfg(feature = "cuda-ipc-core")]
fn serve_cuda(
    channel: &Channel,
    limits: GuardianLimits,
) -> Result<GuardianSummary, GuardianRunError> {
    let backend = ptr_fabric::CudaBackend::open()
        .map_err(|error| GuardianRunError::NoCudaBackend(error.to_string()))?;
    serve_guardian(channel, backend, limits).map_err(GuardianRunError::Serve)
}

#[cfg(not(feature = "cuda-ipc-core"))]
fn serve_cuda(
    _channel: &Channel,
    _limits: GuardianLimits,
) -> Result<GuardianSummary, GuardianRunError> {
    Err(GuardianRunError::NoCudaBackend(
        "this build has no CUDA staging backend (feature cuda-ipc or cuda-ipc-core)".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_parse() {
        assert_eq!(BackendChoice::parse("fake"), Some(BackendChoice::Fake));
        assert_eq!(BackendChoice::parse("cuda"), Some(BackendChoice::Cuda));
        assert_eq!(BackendChoice::parse("gpu"), None);
        // The workers' `--staging` takes the same names.
        for choice in [BackendChoice::Fake, BackendChoice::Cuda] {
            assert_eq!(BackendChoice::parse(choice.name()), Some(choice));
        }
    }

    #[cfg(not(feature = "cuda-ipc-core"))]
    #[test]
    fn the_cuda_backend_is_refused_in_a_build_without_it() {
        let (_ours, theirs) = Channel::pair().expect("pair");
        assert!(matches!(
            serve_on(&theirs, BackendChoice::Cuda),
            Err(GuardianRunError::NoCudaBackend(_))
        ));
    }
}
