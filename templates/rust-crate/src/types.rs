use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendState {
    Ready,
    Degraded,
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecuteRequest<T> {
    pub key: String,
    pub input: T,
    pub preferred_backend: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct ExecuteRequestRef<'request, T> {
    pub key: &'request str,
    pub input: &'request T,
    pub preferred_backend: Option<&'request str>,
}

impl<T> ExecuteRequest<T> {
    pub fn as_ref(&self) -> ExecuteRequestRef<'_, T> {
        ExecuteRequestRef {
            key: &self.key,
            input: &self.input,
            preferred_backend: self.preferred_backend.as_deref(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecuteResult<T> {
    pub value: T,
    pub backend: String,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BackendRegistry<T> {
    entries: HashMap<String, T>,
}

impl<T> BackendRegistry<T> {
    pub(crate) fn insert(&mut self, name: impl Into<String>, backend: T) -> Option<T> {
        self.entries.insert(name.into(), backend)
    }

    pub(crate) fn get(&self, name: &str) -> Option<&T> {
        self.entries.get(name)
    }
}
