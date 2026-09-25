use crate::error::PgError;

/// Oldest server the substrate supports (`server_version_num`).
pub const MINIMUM_SERVER: u32 = 160_000;
/// pgvector version that introduced `halfvec`.
pub const MINIMUM_PGVECTOR: (u32, u32, u32) = (0, 7, 0);

/// Lexical ranking the substrate can use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LexicalBackend {
    /// Built-in full-text search (`tsvector`/`ts_rank_cd`). Always available;
    /// not BM25, and the baseline every BM25 candidate is measured against.
    BuiltinFullText,
    /// Tiger Data `pg_textsearch` BM25 (PostgreSQL licence).
    PgTextsearch,
    /// ParadeDB `pg_search` BM25 (AGPL-3.0; physical replication is not in the
    /// community edition).
    ParadeDb,
}

/// What one server offers, parsed from catalog queries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Capabilities {
    pub server_version_num: u32,
    pub pgvector: Option<(u32, u32, u32)>,
    pub vectorchord: Option<String>,
    pub pg_textsearch: Option<String>,
    pub pg_search: Option<String>,
}

impl Capabilities {
    /// Build from `server_version_num` and `(name, version)` pairs of
    /// installed extensions.
    pub fn from_catalog<'a, I>(server_version_num: u32, installed: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut capabilities = Self {
            server_version_num,
            pgvector: None,
            vectorchord: None,
            pg_textsearch: None,
            pg_search: None,
        };
        for (name, version) in installed {
            match name {
                "vector" => capabilities.pgvector = parse_version(version),
                "vchord" => capabilities.vectorchord = Some(version.to_owned()),
                "pg_textsearch" => capabilities.pg_textsearch = Some(version.to_owned()),
                "pg_search" => capabilities.pg_search = Some(version.to_owned()),
                _ => {}
            }
        }
        capabilities
    }

    /// Refuse a server the substrate cannot run on.
    pub fn check_supported(&self) -> Result<(), PgError> {
        if self.server_version_num < MINIMUM_SERVER {
            return Err(PgError::UnsupportedServer {
                found: self.server_version_num,
                minimum: MINIMUM_SERVER,
            });
        }
        match self.pgvector {
            Some(version) if version >= MINIMUM_PGVECTOR => Ok(()),
            _ => Err(PgError::MissingExtension { name: "vector" }),
        }
    }

    /// Whether pgvector supports iterative index scans (0.8.0 and later),
    /// which filtered approximate search needs to return complete results.
    pub fn iterative_scans(&self) -> bool {
        self.pgvector.is_some_and(|version| version >= (0, 8, 0))
    }

    /// Lexical backends present on this server, in the order of preference:
    /// a permissively licensed BM25 index first, the built-in baseline last.
    /// Which one a deployment uses is a configuration decision backed by the
    /// lexical-search evaluation, not something this probe decides.
    pub fn lexical_backends(&self) -> Vec<LexicalBackend> {
        let mut backends = Vec::new();
        if self.pg_textsearch.is_some() {
            backends.push(LexicalBackend::PgTextsearch);
        }
        if self.pg_search.is_some() {
            backends.push(LexicalBackend::ParadeDb);
        }
        backends.push(LexicalBackend::BuiltinFullText);
        backends
    }
}

/// Parse `major.minor.patch`; a missing patch reads as `0`.
pub fn parse_version(version: &str) -> Option<(u32, u32, u32)> {
    let mut parts = version.split('.').map(|part| part.parse::<u32>().ok());
    let major = parts.next()??;
    let minor = parts.next()??;
    let patch = match parts.next() {
        Some(patch) => patch?,
        None => 0,
    };
    Some((major, minor, patch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_parse_with_and_without_patch() {
        assert_eq!(parse_version("0.8.6"), Some((0, 8, 6)));
        assert_eq!(parse_version("1.4"), Some((1, 4, 0)));
        assert_eq!(parse_version("x.1"), None);
    }

    #[test]
    fn an_old_server_or_missing_pgvector_is_refused() {
        let old = Capabilities::from_catalog(150_008, [("vector", "0.8.6")]);
        assert!(matches!(
            old.check_supported(),
            Err(PgError::UnsupportedServer { .. })
        ));
        let bare = Capabilities::from_catalog(180_006, []);
        assert_eq!(
            bare.check_supported(),
            Err(PgError::MissingExtension { name: "vector" })
        );
        let no_halfvec = Capabilities::from_catalog(180_006, [("vector", "0.6.0")]);
        assert!(no_halfvec.check_supported().is_err());
    }

    #[test]
    fn a_permissive_bm25_index_is_preferred_and_the_baseline_is_always_present() {
        let caps = Capabilities::from_catalog(
            180_006,
            [
                ("vector", "0.8.6"),
                ("pg_search", "0.25.10"),
                ("pg_textsearch", "1.4.0"),
            ],
        );
        assert_eq!(
            caps.lexical_backends(),
            vec![
                LexicalBackend::PgTextsearch,
                LexicalBackend::ParadeDb,
                LexicalBackend::BuiltinFullText
            ]
        );
        assert!(caps.iterative_scans());
        let plain = Capabilities::from_catalog(180_006, [("vector", "0.7.4")]);
        assert_eq!(
            plain.lexical_backends(),
            vec![LexicalBackend::BuiltinFullText]
        );
        assert!(!plain.iterative_scans());
    }
}
