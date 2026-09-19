//! Deterministic version-1 delta encoding; not an authenticated journal format.
use super::{SemanticDelta, SemanticError, SemanticPayload, SemanticValue};
use ptr_types::TypeId;

pub const MAX_DELTA_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_DELTA_ITEMS: usize = 16_384;
pub const MAX_KEY_BYTES: usize = 4096;
const MAGIC: &[u8; 8] = b"PTRSD001";

fn key(value: &str) -> Result<(), SemanticError> {
    if value.is_empty() || value.len() > MAX_KEY_BYTES { Err(SemanticError::InvalidKey) }
    else { Ok(()) }
}
struct Writer(Vec<u8>);
impl Writer {
    fn raw(&mut self, value: &[u8]) -> Result<(), SemanticError> {
        if value.len() > MAX_DELTA_BYTES.saturating_sub(self.0.len()) { return Err(SemanticError::LimitExceeded); }
        self.0.extend_from_slice(value);
        Ok(())
    }
    fn count(&mut self, value: usize) -> Result<(), SemanticError> {
        if value > MAX_DELTA_ITEMS { return Err(SemanticError::LimitExceeded); }
        self.raw(&(value as u32).to_le_bytes())
    }
    fn bytes(&mut self, value: &[u8]) -> Result<(), SemanticError> {
        let size = u32::try_from(value.len()).map_err(|_| SemanticError::LimitExceeded)?;
        self.raw(&size.to_le_bytes())?;
        self.raw(value)
    }
    fn key(&mut self, value: &str) -> Result<(), SemanticError> { key(value)?; self.bytes(value.as_bytes()) }
}
impl SemanticDelta {
    pub fn encode(&self) -> Result<Vec<u8>, SemanticError> {
        if self.removals.iter().any(|key| self.upserts.contains_key(key) || self.dependencies.contains_key(key)) {
            return Err(SemanticError::ConflictingOperation);
        }
        let mut out = Writer(Vec::new());
        out.raw(MAGIC)?;
        out.count(self.upserts.len())?;
        for (name, value) in &self.upserts {
            out.key(name)?;
            match value {
                SemanticValue::Text(text) => { out.raw(&[0])?; out.bytes(text.as_bytes())?; }
                SemanticValue::Payload(payload) => {
                    out.raw(&[1])?;
                    out.key(&payload.type_id.0)?;
                    out.key(&payload.source)?;
                    out.bytes(&payload.bytes)?;
                }
            }
        }
        out.count(self.removals.len())?;
        for name in &self.removals { out.key(name)?; }
        out.count(self.dependencies.len())?;
        for (name, inputs) in &self.dependencies {
            out.key(name)?;
            out.count(inputs.len())?;
            for input in inputs { out.key(input)?; }
        }
        Ok(out.0)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, SemanticError> {
        if bytes.len() > MAX_DELTA_BYTES { return Err(SemanticError::LimitExceeded); }
        let mut reader = Reader { bytes, offset: 0 };
        if reader.take(8)? != MAGIC { return Err(SemanticError::InvalidEncoding); }
        let mut delta = Self::default();
        for _ in 0..reader.count()? {
            let name = reader.key()?;
            let value = match reader.take(1)?[0] {
                0 => SemanticValue::Text(reader.string()?),
                1 => SemanticValue::Payload(SemanticPayload {
                    type_id: TypeId(reader.key()?), source: reader.key()?, bytes: reader.bytes()?.to_vec(),
                }),
                _ => return Err(SemanticError::InvalidEncoding),
            };
            if delta.upserts.insert(name, value).is_some() { return Err(SemanticError::InvalidEncoding); }
        }
        for _ in 0..reader.count()? {
            if !delta.removals.insert(reader.key()?) { return Err(SemanticError::InvalidEncoding); }
        }
        for _ in 0..reader.count()? {
            let name = reader.key()?;
            let mut inputs = std::collections::BTreeSet::new();
            for _ in 0..reader.count()? {
                if !inputs.insert(reader.key()?) { return Err(SemanticError::InvalidEncoding); }
            }
            if delta.dependencies.insert(name, inputs).is_some() { return Err(SemanticError::InvalidEncoding); }
        }
        // Require the unique sorted representation, rejecting trailing bytes and
        // alternate orderings. Never allocate from an untrusted count directly.
        if reader.offset != bytes.len() || delta.encode()? != bytes { return Err(SemanticError::InvalidEncoding); }
        Ok(delta)
    }
}
struct Reader<'a> { bytes: &'a [u8], offset: usize }
impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], SemanticError> {
        let end = self.offset.checked_add(length).ok_or(SemanticError::InvalidEncoding)?;
        let value = self.bytes.get(self.offset..end).ok_or(SemanticError::InvalidEncoding)?;
        self.offset = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<usize, SemanticError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes")) as usize)
    }
    fn count(&mut self) -> Result<usize, SemanticError> {
        let count = self.u32()?;
        if count > MAX_DELTA_ITEMS || count > (self.bytes.len() - self.offset) / 4 { return Err(SemanticError::LimitExceeded); }
        Ok(count)
    }
    fn bytes(&mut self) -> Result<&'a [u8], SemanticError> { let n = self.u32()?; self.take(n) }
    fn string(&mut self) -> Result<String, SemanticError> {
        std::str::from_utf8(self.bytes()?).map(str::to_owned).map_err(|_| SemanticError::InvalidEncoding)
    }
    fn key(&mut self) -> Result<String, SemanticError> { let value = self.string()?; key(&value)?; Ok(value) }
}
