-- Eligibility metadata of a search document: who may see it, when it holds and
-- what it states. A query applies these as conditions of the scan itself, never
-- after the LIMIT: a filter applied to rows an approximate index already chose
-- leaves a result short of documents that exist.
--
-- principals: the principals and groups that may see the document; empty means
--   the whole project. A query's audience (its principal plus its groups) must
--   share a member with a non-empty list.
-- valid_from / valid_until: half-open, valid_from <= now < valid_until; NULL is
--   unbounded on that side. Milliseconds since the Unix epoch.
-- semantic_role / predicate / tags: the metadata a query can ask for.
--
-- Existing rows become project-wide, always valid and unlabelled: what they
-- were before this migration.

ALTER TABLE {{derived}}.search_document
    ADD COLUMN principals text[] NOT NULL DEFAULT '{}',
    ADD COLUMN valid_from bigint CHECK (valid_from IS NULL OR valid_from >= 0),
    ADD COLUMN valid_until bigint CHECK (valid_until IS NULL OR valid_until >= 0),
    ADD COLUMN semantic_role text,
    ADD COLUMN predicate text,
    ADD COLUMN tags text[] NOT NULL DEFAULT '{}',
    ADD CONSTRAINT search_document_validity_ordered
        CHECK (valid_from IS NULL OR valid_until IS NULL OR valid_from < valid_until);

CREATE INDEX search_document_principals ON {{derived}}.search_document USING gin (principals);
CREATE INDEX search_document_tags ON {{derived}}.search_document USING gin (tags);
CREATE INDEX search_document_validity ON {{derived}}.search_document (valid_from, valid_until);
