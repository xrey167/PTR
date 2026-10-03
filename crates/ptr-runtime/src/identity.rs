use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use ptr_types::{AuthenticationLevel, IdentityContext, IdentityProvider, SessionId, Timestamp};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "oidc-http")]
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OidcAlgorithm {
    EdDsa,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OidcError {
    InvalidToken,
    InvalidEncoding,
    UnsupportedAlgorithm,
    UnknownKey(String),
    InvalidSignature,
    MissingClaim(&'static str),
    InvalidClaim(&'static str),
    IssuerMismatch,
    AudienceMismatch,
    NotYetValid,
    Expired,
    TokenReplay,
    RefreshFailed(String),
    JwksIssuerMismatch,
    JwksAudienceMismatch,
    InvalidJwk,
    ForwardAuthDenied,
}

#[derive(Clone, Default)]
pub struct JwksStore {
    keys: BTreeMap<String, VerifyingKey>,
    revision: u64,
}

impl JwksStore {
    pub fn insert(&mut self, kid: impl Into<String>, key: VerifyingKey) {
        self.keys.insert(kid.into(), key);
        self.revision = self.revision.saturating_add(1);
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn get(&self, kid: &str) -> Option<&VerifyingKey> {
        self.keys.get(kid)
    }

    fn from_document(document: &Value) -> Result<Self, OidcError> {
        let keys = document
            .get("keys")
            .and_then(Value::as_array)
            .ok_or(OidcError::InvalidJwk)?;
        let mut store = Self::default();
        for item in keys {
            if item.get("kty").and_then(Value::as_str) != Some("OKP")
                || item.get("crv").and_then(Value::as_str) != Some("Ed25519")
            {
                continue;
            }
            let kid = item
                .get("kid")
                .and_then(Value::as_str)
                .ok_or(OidcError::InvalidJwk)?;
            let x = item
                .get("x")
                .and_then(Value::as_str)
                .ok_or(OidcError::InvalidJwk)?;
            let bytes: [u8; 32] = URL_SAFE_NO_PAD
                .decode(x)
                .map_err(|_| OidcError::InvalidJwk)?
                .try_into()
                .map_err(|_| OidcError::InvalidJwk)?;
            let key = VerifyingKey::from_bytes(&bytes).map_err(|_| OidcError::InvalidJwk)?;
            store.insert(kid, key);
        }
        if store.keys.is_empty() {
            return Err(OidcError::InvalidJwk);
        }
        Ok(store)
    }
}

pub trait JwksRefresher: Send + Sync {
    fn refresh(&self) -> Result<JwksDocument, OidcError>;
}

pub struct JwksDocument {
    pub issuer: String,
    pub audience: String,
    pub store: JwksStore,
}

pub type JwksFetchFn = Arc<dyn Fn(&str) -> Result<Vec<u8>, OidcError> + Send + Sync>;

pub struct HttpJwksRefresher {
    url: String,
    issuer: String,
    audience: String,
    fetch: JwksFetchFn,
}

/// Production HTTPS JWKS fetcher. The injected `HttpJwksRefresher` remains
/// available for deterministic tests; this adapter is opt-in so applications
/// choose when a blocking HTTP client is acceptable at their identity boundary.
#[cfg(feature = "oidc-http")]
pub struct ReqwestJwksRefresher {
    url: String,
    issuer: String,
    audience: String,
    client: reqwest::blocking::Client,
}

#[cfg(feature = "oidc-http")]
impl ReqwestJwksRefresher {
    pub fn new(
        url: impl Into<String>,
        issuer: impl Into<String>,
        audience: impl Into<String>,
    ) -> Result<Self, OidcError> {
        let url = url.into();
        if !url.starts_with("https://") {
            return Err(OidcError::InvalidEncoding);
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|error| OidcError::RefreshFailed(error.to_string()))?;
        Ok(Self {
            url,
            issuer: issuer.into(),
            audience: audience.into(),
            client,
        })
    }
}

#[cfg(feature = "oidc-http")]
impl JwksRefresher for ReqwestJwksRefresher {
    fn refresh(&self) -> Result<JwksDocument, OidcError> {
        const MAX_JWKS_BYTES: usize = 1024 * 1024;
        let response = self
            .client
            .get(&self.url)
            .send()
            .map_err(|error| OidcError::RefreshFailed(error.to_string()))?;
        if !response.status().is_success() {
            return Err(OidcError::RefreshFailed(format!(
                "JWKS endpoint returned {}",
                response.status()
            )));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_JWKS_BYTES as u64)
        {
            return Err(OidcError::RefreshFailed(
                "JWKS document exceeds limit".into(),
            ));
        }
        let mut bytes = Vec::new();
        response
            .take((MAX_JWKS_BYTES as u64).saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|error| OidcError::RefreshFailed(error.to_string()))?;
        if bytes.len() > MAX_JWKS_BYTES {
            return Err(OidcError::RefreshFailed(
                "JWKS document exceeds limit".into(),
            ));
        }
        let document: Value = serde_json::from_slice(&bytes)
            .map_err(|error| OidcError::RefreshFailed(error.to_string()))?;
        Ok(JwksDocument {
            issuer: self.issuer.clone(),
            audience: self.audience.clone(),
            store: JwksStore::from_document(&document)?,
        })
    }
}

impl HttpJwksRefresher {
    pub fn new(
        url: impl Into<String>,
        issuer: impl Into<String>,
        audience: impl Into<String>,
        fetch: JwksFetchFn,
    ) -> Result<Self, OidcError> {
        let url = url.into();
        if !(url.starts_with("https://")
            || url.starts_with("http://127.0.0.1/")
            || url.starts_with("http://localhost/"))
        {
            return Err(OidcError::InvalidEncoding);
        }
        Ok(Self {
            url,
            issuer: issuer.into(),
            audience: audience.into(),
            fetch,
        })
    }
}

impl JwksRefresher for HttpJwksRefresher {
    fn refresh(&self) -> Result<JwksDocument, OidcError> {
        const MAX_JWKS_BYTES: usize = 1024 * 1024;
        let bytes = (self.fetch)(&self.url)?;
        if bytes.len() > MAX_JWKS_BYTES {
            return Err(OidcError::RefreshFailed(
                "JWKS document exceeds limit".into(),
            ));
        }
        let document: Value = serde_json::from_slice(&bytes)
            .map_err(|error| OidcError::RefreshFailed(error.to_string()))?;
        Ok(JwksDocument {
            issuer: self.issuer.clone(),
            audience: self.audience.clone(),
            store: JwksStore::from_document(&document)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForwardAuthHeaders {
    pub subject: String,
    pub session_id: String,
    pub groups: Vec<String>,
    pub issued_at: Timestamp,
    pub expires_at: Timestamp,
    pub authentication_level: AuthenticationLevel,
}

pub struct AutheliaForwardAuthAdapter {
    pub issuer: String,
}

impl AutheliaForwardAuthAdapter {
    pub fn new(issuer: impl Into<String>) -> Self {
        Self {
            issuer: issuer.into(),
        }
    }

    pub fn authenticate_headers(
        &self,
        headers: &ForwardAuthHeaders,
        now: Timestamp,
    ) -> Result<IdentityContext, OidcError> {
        if headers.subject.is_empty() || headers.session_id.is_empty() {
            return Err(OidcError::ForwardAuthDenied);
        }
        if headers.issued_at > now || headers.expires_at <= now {
            return Err(OidcError::Expired);
        }
        let mut groups = headers.groups.clone();
        groups.sort();
        groups.dedup();
        if groups.iter().any(|group| group.is_empty()) {
            return Err(OidcError::ForwardAuthDenied);
        }
        let mut context = IdentityContext {
            subject: headers.subject.clone(),
            issuer: self.issuer.clone(),
            groups,
            authentication_level: headers.authentication_level,
            session_id: SessionId(headers.session_id.clone()),
            issued_at: headers.issued_at,
            expires_at: headers.expires_at,
            identity_digest: [0; 32],
        };
        context.identity_digest = context.canonical_digest();
        Ok(context)
    }
}

impl IdentityProvider for AutheliaForwardAuthAdapter {
    type Error = OidcError;

    fn authenticate(&self, credential: &[u8]) -> Result<IdentityContext, Self::Error> {
        let value: Value =
            serde_json::from_slice(credential).map_err(|_| OidcError::InvalidToken)?;
        let groups = value
            .get("groups")
            .and_then(Value::as_array)
            .ok_or(OidcError::MissingClaim("groups"))?
            .iter()
            .map(|group| {
                group
                    .as_str()
                    .map(str::to_owned)
                    .ok_or(OidcError::InvalidClaim("groups"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let level = match value.get("acr").and_then(Value::as_str) {
            Some("mfa") => AuthenticationLevel::MultiFactor,
            Some("hardware") => AuthenticationLevel::HardwareBound,
            Some("password") => AuthenticationLevel::Password,
            _ => return Err(OidcError::ForwardAuthDenied),
        };
        self.authenticate_headers(
            &ForwardAuthHeaders {
                subject: required_string(&value, "subject")?,
                session_id: required_string(&value, "session_id")?,
                groups,
                issued_at: required_timestamp(&value, "issued_at")?,
                expires_at: required_timestamp(&value, "expires_at")?,
                authentication_level: level,
            },
            // The verification time always comes from the server clock. A
            // credential must never be able to choose the instant at which its
            // own expiry is evaluated.
            current_timestamp()?,
        )
    }
}

pub struct OidcIdentityAdapter {
    pub issuer: String,
    pub audience: String,
    pub jwks: JwksStore,
    pub allowed_algorithms: Vec<OidcAlgorithm>,
    used_jti: Mutex<BTreeSet<String>>,
    refresh: Mutex<RefreshState>,
    refresh_cooldown: Duration,
}

/// Default minimum spacing between JWKS fetches triggered by unknown key ids.
pub const DEFAULT_JWKS_REFRESH_COOLDOWN: Duration = Duration::from_secs(30);

/// Keys obtained through a refresh, kept only for one cooldown window. The same
/// window also rate-limits fetches, so unauthenticated callers presenting
/// unknown `kid` values cannot turn into unbounded requests against the IdP.
#[derive(Default)]
struct RefreshState {
    fetched: Option<(Instant, Arc<JwksStore>)>,
    last_attempt: Option<Instant>,
}

impl OidcIdentityAdapter {
    pub fn new(issuer: impl Into<String>, audience: impl Into<String>, jwks: JwksStore) -> Self {
        Self {
            issuer: issuer.into(),
            audience: audience.into(),
            jwks,
            allowed_algorithms: vec![OidcAlgorithm::EdDsa],
            used_jti: Mutex::new(BTreeSet::new()),
            refresh: Mutex::new(RefreshState::default()),
            refresh_cooldown: DEFAULT_JWKS_REFRESH_COOLDOWN,
        }
    }

    pub fn with_refresh_cooldown(mut self, cooldown: Duration) -> Self {
        self.refresh_cooldown = cooldown;
        self
    }

    pub fn authenticate_at(
        &self,
        credential: &[u8],
        now: Timestamp,
    ) -> Result<IdentityContext, OidcError> {
        self.authenticate_at_with_store(credential, now, &self.jwks)
    }

    pub fn authenticate_at_with_refresh(
        &self,
        credential: &[u8],
        now: Timestamp,
        refresher: &dyn JwksRefresher,
    ) -> Result<IdentityContext, OidcError> {
        let kid = token_kid(credential)?;
        if self.jwks.get(&kid).is_some() {
            return self.authenticate_at(credential, now);
        }
        let store = {
            // The lock is held across the fetch on purpose: concurrent callers
            // with unknown key ids queue here and then observe the cooldown
            // instead of each issuing their own request.
            let mut state = self
                .refresh
                .lock()
                .map_err(|_| OidcError::RefreshFailed("refresh state poisoned".into()))?;
            let started = Instant::now();
            let cached = state.fetched.as_ref().and_then(|(fetched_at, store)| {
                (started.duration_since(*fetched_at) < self.refresh_cooldown
                    && store.get(&kid).is_some())
                .then(|| Arc::clone(store))
            });
            match cached {
                Some(store) => store,
                None => {
                    if state.last_attempt.is_some_and(|attempt| {
                        started.duration_since(attempt) < self.refresh_cooldown
                    }) {
                        return Err(OidcError::UnknownKey(kid));
                    }
                    // Failed fetches count as attempts, so an unreachable or
                    // hostile IdP endpoint is not hammered either.
                    state.last_attempt = Some(started);
                    let document = refresher.refresh()?;
                    if document.issuer != self.issuer {
                        return Err(OidcError::JwksIssuerMismatch);
                    }
                    if document.audience != self.audience {
                        return Err(OidcError::JwksAudienceMismatch);
                    }
                    let store = Arc::new(document.store);
                    state.fetched = Some((started, Arc::clone(&store)));
                    store
                }
            }
        };
        self.authenticate_at_with_store(credential, now, &store)
    }

    fn authenticate_at_with_store(
        &self,
        credential: &[u8],
        now: Timestamp,
        jwks: &JwksStore,
    ) -> Result<IdentityContext, OidcError> {
        let token = std::str::from_utf8(credential).map_err(|_| OidcError::InvalidToken)?;
        let mut parts = token.split('.');
        let encoded_header = parts.next().ok_or(OidcError::InvalidToken)?;
        let encoded_claims = parts.next().ok_or(OidcError::InvalidToken)?;
        let encoded_signature = parts.next().ok_or(OidcError::InvalidToken)?;
        if parts.next().is_some() {
            return Err(OidcError::InvalidToken);
        }
        let header = decode_json(encoded_header)?;
        let claims = decode_json(encoded_claims)?;
        match header.get("alg").and_then(Value::as_str) {
            Some("EdDSA") if self.allowed_algorithms.contains(&OidcAlgorithm::EdDsa) => {}
            _ => return Err(OidcError::UnsupportedAlgorithm),
        }
        let kid = header
            .get("kid")
            .and_then(Value::as_str)
            .ok_or(OidcError::MissingClaim("kid"))?;
        let key = jwks
            .get(kid)
            .ok_or_else(|| OidcError::UnknownKey(kid.to_owned()))?;
        let signature_bytes = URL_SAFE_NO_PAD
            .decode(encoded_signature)
            .map_err(|_| OidcError::InvalidEncoding)?;
        let signature: [u8; 64] = signature_bytes
            .as_slice()
            .try_into()
            .map_err(|_| OidcError::InvalidSignature)?;
        let signing_input = format!("{encoded_header}.{encoded_claims}");
        key.verify(signing_input.as_bytes(), &Signature::from_bytes(&signature))
            .map_err(|_| OidcError::InvalidSignature)?;

        let issuer = required_string(&claims, "iss")?;
        if issuer != self.issuer {
            return Err(OidcError::IssuerMismatch);
        }
        if !audience_matches(claims.get("aud"), &self.audience) {
            return Err(OidcError::AudienceMismatch);
        }
        let subject = required_string(&claims, "sub")?;
        let session = required_string(&claims, "sid")?;
        let issued_at = required_timestamp(&claims, "iat")?;
        let expires_at = required_timestamp(&claims, "exp")?;
        if issued_at > now {
            return Err(OidcError::NotYetValid);
        }
        if expires_at <= now {
            return Err(OidcError::Expired);
        }
        if let Some(jti) = claims.get("jti") {
            let jti = jti
                .as_str()
                .filter(|value| !value.is_empty())
                .ok_or(OidcError::InvalidClaim("jti"))?;
            let mut used = self.used_jti.lock().map_err(|_| OidcError::InvalidToken)?;
            if !used.insert(jti.to_owned()) {
                return Err(OidcError::TokenReplay);
            }
        }
        let groups = canonical_groups(claims.get("groups"))?;
        let authentication_level = authentication_level(&claims);
        let mut context = IdentityContext {
            subject,
            issuer,
            groups,
            authentication_level,
            session_id: SessionId(session),
            issued_at,
            expires_at,
            identity_digest: [0; 32],
        };
        context.identity_digest = context.canonical_digest();
        Ok(context)
    }
}

impl IdentityProvider for OidcIdentityAdapter {
    type Error = OidcError;

    fn authenticate(&self, credential: &[u8]) -> Result<IdentityContext, Self::Error> {
        self.authenticate_at(credential, current_timestamp()?)
    }
}

fn current_timestamp() -> Result<Timestamp, OidcError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| Timestamp(elapsed.as_secs()))
        .map_err(|_| OidcError::InvalidClaim("system_time"))
}

fn decode_json(encoded: &str) -> Result<Value, OidcError> {
    let bytes = URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| OidcError::InvalidEncoding)?;
    serde_json::from_slice(&bytes).map_err(|_| OidcError::InvalidEncoding)
}

fn token_kid(credential: &[u8]) -> Result<String, OidcError> {
    let token = std::str::from_utf8(credential).map_err(|_| OidcError::InvalidToken)?;
    let header = token
        .split('.')
        .next()
        .ok_or(OidcError::InvalidToken)
        .and_then(decode_json)?;
    header
        .get("kid")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or(OidcError::MissingClaim("kid"))
}

fn required_string(claims: &Value, name: &'static str) -> Result<String, OidcError> {
    let value = claims
        .get(name)
        .and_then(Value::as_str)
        .ok_or(OidcError::MissingClaim(name))?;
    if value.is_empty() {
        return Err(OidcError::InvalidClaim(name));
    }
    Ok(value.to_owned())
}

fn required_timestamp(claims: &Value, name: &'static str) -> Result<Timestamp, OidcError> {
    let value = claims
        .get(name)
        .and_then(Value::as_u64)
        .ok_or(OidcError::MissingClaim(name))?;
    Ok(Timestamp(value))
}

fn audience_matches(value: Option<&Value>, expected: &str) -> bool {
    match value {
        Some(Value::String(value)) => value == expected,
        Some(Value::Array(values)) => values.iter().any(|value| value.as_str() == Some(expected)),
        _ => false,
    }
}

fn canonical_groups(value: Option<&Value>) -> Result<Vec<String>, OidcError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let values = value.as_array().ok_or(OidcError::InvalidClaim("groups"))?;
    let mut groups = values
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|group| !group.is_empty())
                .map(str::to_owned)
                .ok_or(OidcError::InvalidClaim("groups"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    groups.sort();
    groups.dedup();
    Ok(groups)
}

fn authentication_level(claims: &Value) -> AuthenticationLevel {
    match claims.get("acr").and_then(Value::as_str) {
        Some("hw" | "hardware" | "hardware-bound") => AuthenticationLevel::HardwareBound,
        Some("mfa" | "multi_factor" | "multi-factor") => AuthenticationLevel::MultiFactor,
        Some("password") => AuthenticationLevel::Password,
        _ => AuthenticationLevel::Password,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use serde_json::json;

    fn token(key: &SigningKey, claims: Value) -> String {
        token_with_kid(key, claims, "k1")
    }

    fn token_with_kid(key: &SigningKey, claims: Value, kid: &str) -> String {
        let header = URL_SAFE_NO_PAD.encode(
            serde_json::json!({"alg":"EdDSA", "kid":kid})
                .to_string()
                .as_bytes(),
        );
        let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
        let input = format!("{header}.{payload}");
        let signature = URL_SAFE_NO_PAD.encode(key.sign(input.as_bytes()).to_bytes());
        format!("{input}.{signature}")
    }

    fn adapter(key: &SigningKey) -> OidcIdentityAdapter {
        let mut jwks = JwksStore::default();
        jwks.insert("k1", key.verifying_key());
        OidcIdentityAdapter::new("https://issuer", "ptr", jwks)
    }

    fn claims() -> Value {
        json!({
            "iss": "https://issuer", "aud": ["other", "ptr"], "sub": "user",
            "sid": "session", "iat": 10, "exp": 100, "groups": ["z", "a", "a"],
            "acr": "mfa"
        })
    }

    #[test]
    fn local_jwks_verifies_and_canonicalizes_identity() {
        let key = SigningKey::from_bytes(&[31; 32]);
        let identity = adapter(&key)
            .authenticate_at(token(&key, claims()).as_bytes(), Timestamp(50))
            .unwrap();
        assert_eq!(identity.groups, vec!["a", "z"]);
        assert_eq!(
            identity.authentication_level,
            AuthenticationLevel::MultiFactor
        );
        assert!(identity.has_valid_digest());
    }

    #[test]
    fn issuer_audience_expiry_fail_closed() {
        let key = SigningKey::from_bytes(&[32; 32]);
        let adapter = adapter(&key);
        assert_eq!(
            adapter.authenticate_at(
                token(
                    &key,
                    json!({"iss":"wrong", "aud":"ptr", "sub":"u", "sid":"s", "iat":1, "exp":100})
                )
                .as_bytes(),
                Timestamp(50)
            ),
            Err(OidcError::IssuerMismatch)
        );
        assert_eq!(
            adapter.authenticate_at(
                token(&key, json!({"iss":"https://issuer", "aud":"ptr", "sub":"u", "sid":"s", "iat":1, "exp":50})).as_bytes(),
                Timestamp(50)
            ),
            Err(OidcError::Expired)
        );
    }

    #[test]
    fn jti_replay_is_rejected() {
        let key = SigningKey::from_bytes(&[33; 32]);
        let adapter = adapter(&key);
        let token = token(
            &key,
            json!({"iss":"https://issuer", "aud":"ptr", "sub":"u", "sid":"s", "iat":1, "exp":100, "jti":"once"}),
        );
        assert!(adapter
            .authenticate_at(token.as_bytes(), Timestamp(50))
            .is_ok());
        assert_eq!(
            adapter.authenticate_at(token.as_bytes(), Timestamp(50)),
            Err(OidcError::TokenReplay)
        );
    }

    struct StaticRefresher {
        document: JwksDocument,
    }

    impl JwksRefresher for StaticRefresher {
        fn refresh(&self) -> Result<JwksDocument, OidcError> {
            Ok(JwksDocument {
                issuer: self.document.issuer.clone(),
                audience: self.document.audience.clone(),
                store: self.document.store.clone(),
            })
        }
    }

    #[test]
    fn unknown_kid_refreshes_once_without_stale_fallback() {
        let old_key = SigningKey::from_bytes(&[34; 32]);
        let new_key = SigningKey::from_bytes(&[35; 32]);
        let adapter = adapter(&old_key);
        let mut new_store = JwksStore::default();
        new_store.insert("k2", new_key.verifying_key());
        let refresher = StaticRefresher {
            document: JwksDocument {
                issuer: "https://issuer".into(),
                audience: "ptr".into(),
                store: new_store,
            },
        };
        let token = token_with_kid(&new_key, claims(), "k2");
        assert!(adapter
            .authenticate_at_with_refresh(token.as_bytes(), Timestamp(50), &refresher)
            .is_ok());
    }

    #[test]
    fn http_refresh_adapter_validates_url_and_jwks_document() {
        let key = SigningKey::from_bytes(&[36; 32]);
        let jwk = URL_SAFE_NO_PAD.encode(key.verifying_key().to_bytes());
        let body =
            format!(r#"{{"keys":[{{"kty":"OKP","crv":"Ed25519","kid":"k1","x":"{jwk}"}}]}}"#)
                .into_bytes();
        let refresher = HttpJwksRefresher::new(
            "https://issuer/jwks",
            "https://issuer",
            "ptr",
            Arc::new(move |_| Ok(body.clone())),
        )
        .unwrap();
        let document = refresher.refresh().unwrap();
        assert_eq!(document.issuer, "https://issuer");
        assert_eq!(document.store.revision(), 1);
        assert!(HttpJwksRefresher::new(
            "http://example.invalid/jwks",
            "https://issuer",
            "ptr",
            Arc::new(|_| Ok(Vec::new())),
        )
        .is_err());
    }

    #[test]
    fn http_refresh_adapter_rejects_oversized_documents() {
        let refresher = HttpJwksRefresher::new(
            "https://issuer/jwks",
            "https://issuer",
            "ptr",
            Arc::new(|_| Ok(vec![b'x'; 1024 * 1024 + 1])),
        )
        .unwrap();
        assert!(matches!(
            refresher.refresh(),
            Err(OidcError::RefreshFailed(message))
                if message == "JWKS document exceeds limit"
        ));
    }

    #[test]
    fn forward_auth_headers_are_typed_and_expiring() {
        let adapter = AutheliaForwardAuthAdapter::new("https://authelia");
        let context = adapter
            .authenticate_headers(
                &ForwardAuthHeaders {
                    subject: "user".into(),
                    session_id: "session".into(),
                    groups: vec!["z".into(), "a".into()],
                    issued_at: Timestamp(1),
                    expires_at: Timestamp(100),
                    authentication_level: AuthenticationLevel::MultiFactor,
                },
                Timestamp(50),
            )
            .unwrap();
        assert_eq!(context.groups, vec!["a", "z"]);
        assert!(context.has_valid_digest());
    }

    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingRefresher {
        document: JwksDocument,
        calls: AtomicUsize,
    }

    impl JwksRefresher for CountingRefresher {
        fn refresh(&self) -> Result<JwksDocument, OidcError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(JwksDocument {
                issuer: self.document.issuer.clone(),
                audience: self.document.audience.clone(),
                store: self.document.store.clone(),
            })
        }
    }

    #[test]
    fn unknown_kid_refreshes_are_rate_limited_and_cached() {
        let old_key = SigningKey::from_bytes(&[37; 32]);
        let new_key = SigningKey::from_bytes(&[38; 32]);
        let adapter = adapter(&old_key).with_refresh_cooldown(Duration::from_secs(3600));
        let mut new_store = JwksStore::default();
        new_store.insert("k2", new_key.verifying_key());
        let refresher = CountingRefresher {
            document: JwksDocument {
                issuer: "https://issuer".into(),
                audience: "ptr".into(),
                store: new_store,
            },
            calls: AtomicUsize::new(0),
        };
        // A flood of unknown key ids results in a single fetch.
        for attempt in 0..5 {
            let unknown = token_with_kid(&new_key, claims(), &format!("unknown-{attempt}"));
            assert!(matches!(
                adapter.authenticate_at_with_refresh(unknown.as_bytes(), Timestamp(50), &refresher),
                Err(OidcError::UnknownKey(_))
            ));
        }
        assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
        // The fetched key is cached, so a legitimate rotated key needs no new fetch.
        let valid = token_with_kid(&new_key, claims(), "k2");
        assert!(adapter
            .authenticate_at_with_refresh(valid.as_bytes(), Timestamp(50), &refresher)
            .is_ok());
        assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn forward_auth_credential_cannot_choose_its_own_clock() {
        let adapter = AutheliaForwardAuthAdapter::new("https://authelia");
        let credential = json!({
            "subject": "user", "session_id": "session", "groups": ["a"], "acr": "mfa",
            "issued_at": 1, "expires_at": 100, "now": 50
        });
        assert_eq!(
            adapter.authenticate(credential.to_string().as_bytes()),
            Err(OidcError::Expired)
        );
    }
}
