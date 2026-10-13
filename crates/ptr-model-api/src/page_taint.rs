//! Taint and lineage carried with cached KV pages and catalog entries.
//!
//! [`TaintedPage`] is metadata only (it holds digests, not K/V bytes). Its taint is a
//! [`KvReuseTaint`]; [`TaintedPage::join`] is the least upper bound: taints join, and the
//! part sets union, so it is commutative, associative and idempotent. Frame fields of the
//! activation-frame / E008 transport are out of scope here (they belong to ptr-protocol).

use crate::{KvReuseReport, KvReuseTaint};
use ptr_types::{Digest, Generation};
use std::collections::BTreeSet;
use std::fmt;

/// Upper bound on the number of parts one (joined) page may reference (Invariant 8).
pub const MAX_PAGE_PARTS: usize = 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PageTaintError {
    EmptyPageId,
    /// A join would exceed [`MAX_PAGE_PARTS`].
    TooManyParts {
        parts: usize,
    },
    /// Exact content was required but the page is tainted by Approximate reuse.
    Tainted {
        report_digest: Digest,
        page_ids: Vec<String>,
    },
}

impl fmt::Display for PageTaintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPageId => write!(f, "page id is empty"),
            Self::TooManyParts { parts } => {
                write!(
                    f,
                    "joined page would have {parts} parts (max {MAX_PAGE_PARTS})"
                )
            }
            Self::Tainted { page_ids, .. } => {
                write!(
                    f,
                    "content is tainted by Approximate KV reuse (pages: {page_ids:?})"
                )
            }
        }
    }
}

impl std::error::Error for PageTaintError {}

/// One cached page that contributed to a [`TaintedPage`].
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PageRef {
    pub page_id: String,
    /// `SegmentedModelRequest::lineage_digest` of the prefix the page was computed under.
    pub lineage_digest: Digest,
    /// Digest of the page's K/V payload, computed by the caller.
    pub payload_digest: Digest,
}

/// A cached page (or the join of several) with its taint and the lineage of each part.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaintedPage {
    parts: BTreeSet<PageRef>,
    taint: KvReuseTaint,
}

impl TaintedPage {
    pub fn new(
        page_id: impl Into<String>,
        lineage_digest: Digest,
        payload_digest: Digest,
        taint: KvReuseTaint,
    ) -> Result<Self, PageTaintError> {
        let page_id = page_id.into();
        if page_id.is_empty() {
            return Err(PageTaintError::EmptyPageId);
        }
        let mut parts = BTreeSet::new();
        parts.insert(PageRef {
            page_id,
            lineage_digest,
            payload_digest,
        });
        Ok(Self { parts, taint })
    }

    /// A page produced by a run with the given reuse report (`None` = no reuse = Exact).
    pub fn from_report(
        page_id: impl Into<String>,
        lineage_digest: Digest,
        payload_digest: Digest,
        report: Option<&KvReuseReport>,
    ) -> Result<Self, PageTaintError> {
        let taint = report.map_or(KvReuseTaint::Exact, KvReuseReport::taint);
        Self::new(page_id, lineage_digest, payload_digest, taint)
    }

    pub fn taint(&self) -> KvReuseTaint {
        self.taint
    }

    pub fn is_exact(&self) -> bool {
        !self.taint.is_approximate()
    }

    pub fn parts(&self) -> &BTreeSet<PageRef> {
        &self.parts
    }

    /// Least upper bound of two pages: joined taint, unioned parts. Fails only when the
    /// union would exceed [`MAX_PAGE_PARTS`].
    pub fn join(&self, other: &Self) -> Result<Self, PageTaintError> {
        let parts: BTreeSet<PageRef> = self.parts.union(&other.parts).cloned().collect();
        if parts.len() > MAX_PAGE_PARTS {
            return Err(PageTaintError::TooManyParts { parts: parts.len() });
        }
        Ok(Self {
            parts,
            taint: self.taint.join(other.taint),
        })
    }

    /// Refuse tainted content. The error names every part, since the taint of a join cannot
    /// be attributed to a single part.
    pub fn require_exact(&self) -> Result<&Self, PageTaintError> {
        match self.taint {
            KvReuseTaint::Exact => Ok(self),
            KvReuseTaint::Approximate { report_digest } => Err(PageTaintError::Tainted {
                report_digest,
                page_ids: self.parts.iter().map(|p| p.page_id.clone()).collect(),
            }),
        }
    }
}

/// A catalog entry: a key at a generation, backed by a [`TaintedPage`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaintedCatalogEntry {
    key: String,
    generation: Generation,
    page: TaintedPage,
}

impl TaintedCatalogEntry {
    pub fn new(
        key: impl Into<String>,
        generation: Generation,
        page: TaintedPage,
    ) -> Result<Self, PageTaintError> {
        let key = key.into();
        if key.is_empty() {
            return Err(PageTaintError::EmptyPageId);
        }
        Ok(Self {
            key,
            generation,
            page,
        })
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn page(&self) -> &TaintedPage {
        &self.page
    }

    pub fn require_exact(&self) -> Result<&TaintedPage, PageTaintError> {
        self.page.require_exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(n: u8) -> Digest {
        [n; 32]
    }

    #[test]
    fn empty_ids_are_rejected() {
        assert_eq!(
            TaintedPage::new("", d(1), d(2), KvReuseTaint::Exact),
            Err(PageTaintError::EmptyPageId)
        );
        let page = TaintedPage::new("p", d(1), d(2), KvReuseTaint::Exact).unwrap();
        assert_eq!(
            TaintedCatalogEntry::new("", Generation(1), page),
            Err(PageTaintError::EmptyPageId)
        );
    }

    #[test]
    fn join_is_bounded() {
        let mut acc = TaintedPage::new("p0", d(0), d(0), KvReuseTaint::Exact).unwrap();
        let mut failed = None;
        for i in 1..=MAX_PAGE_PARTS {
            let id = format!("p{i}");
            let next = TaintedPage::new(id, d(1), d(1), KvReuseTaint::Exact).unwrap();
            match acc.join(&next) {
                Ok(joined) => acc = joined,
                Err(error) => {
                    failed = Some(error);
                    break;
                }
            }
        }
        assert_eq!(
            failed,
            Some(PageTaintError::TooManyParts {
                parts: MAX_PAGE_PARTS + 1
            })
        );
    }
}
