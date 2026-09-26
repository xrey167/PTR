-- Derived search cache. Capsule content is not in the ledger, so these rows
-- are not a ledger projection: they are computed outside any projection
-- transaction (embeddings included), keyed by content digest, and dropped with
-- the projection on rebuild. Only live generations are ever indexed, and the
-- projector deletes a generation's row in the same transaction that
-- supersedes or revokes it, so the vector and text indexes never hold
-- non-live rows. Every hit must still be validated against the lifecycle
-- authority before it is used.

CREATE TABLE {{derived}}.embedding_space (
    id text PRIMARY KEY CHECK (id ~ '^[a-z][a-z0-9_]{0,39}$'),
    model text NOT NULL,
    revision text NOT NULL,
    dims integer NOT NULL CHECK (dims BETWEEN 1 AND 4000),
    metric text NOT NULL CHECK (metric = 'cosine')
);

CREATE TABLE {{derived}}.search_document (
    capsule text NOT NULL,
    generation bigint NOT NULL CHECK (generation >= 0),
    project text NOT NULL,
    content_digest bytea NOT NULL CHECK (length(content_digest) = 32),
    body text NOT NULL,
    -- 'simple' keeps the baseline language-neutral; a language configuration
    -- or a BM25 index is an evaluated variant, not a default.
    lexeme tsvector GENERATED ALWAYS AS (to_tsvector('simple'::regconfig, body)) STORED,
    space text REFERENCES {{derived}}.embedding_space (id),
    -- Half precision: halfvec(768) is 1,544 bytes and stays inline. Quantised
    -- forms belong in expression indexes, never in the stored value.
    embedding halfvec,
    -- The projection watermark when the row was written.
    indexed_at_commit bigint NOT NULL CHECK (indexed_at_commit >= 0),
    PRIMARY KEY (capsule, generation),
    CHECK ((space IS NULL) = (embedding IS NULL))
);

CREATE INDEX search_document_lexeme ON {{derived}}.search_document USING gin (lexeme);
CREATE INDEX search_document_project ON {{derived}}.search_document (project);
