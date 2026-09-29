-- Ledger projection: everything here is a function of the committed ledger
-- prefix recorded in projection_watermark, and a rebuild drops this schema and
-- replays. The projector is the only writer. It verifies that every record it
-- applies chains from the anchor it last stored, so a projection can never
-- silently hold a history the ledger does not.

CREATE TABLE {{projection}}.projection_watermark (
    id smallint PRIMARY KEY CHECK (id = 1),
    last_applied bigint NOT NULL CHECK (last_applied >= 0),
    -- Chain digest of the last applied record; NULL is the empty-log anchor.
    last_anchor bytea CHECK (last_anchor IS NULL OR length(last_anchor) = 32),
    CHECK ((last_applied = 0) = (last_anchor IS NULL))
);
INSERT INTO {{projection}}.projection_watermark (id, last_applied, last_anchor)
    VALUES (1, 0, NULL);

-- The anchor of every applied record, so a re-delivered record can be told
-- apart from a different record at the same index.
CREATE TABLE {{projection}}.applied_commit (
    commit_index bigint PRIMARY KEY CHECK (commit_index > 0),
    anchor bytea NOT NULL CHECK (length(anchor) = 32)
);

-- Exactly the key/value entries ptr-state defines for each committed event,
-- for the materialized-state slot. Lifecycle is not read from here.
CREATE TABLE {{projection}}.state_entry (
    key text PRIMARY KEY,
    value text NOT NULL,
    commit_index bigint NOT NULL CHECK (commit_index > 0)
);

-- Live generation per lifecycle target, keyed exactly as the runtime keys it:
-- a capsule id, constraint:<key> or procedure:<id>. Every lifecycle change of
-- a target updates its row (commit_index at least), which is the row lock that
-- orders the projector against derived-cache writers.
CREATE TABLE {{projection}}.live_generation (
    target text PRIMARY KEY,
    generation bigint NOT NULL CHECK (generation >= 0),
    -- The project of a capsule; NULL for constraints and procedures.
    project text,
    commit_index bigint NOT NULL CHECK (commit_index > 0)
);

-- The revocation set, never overwritten: one row per revoked (subject,
-- generation), including tombstones committed before the generation is live.
CREATE TABLE {{projection}}.tombstone (
    subject text NOT NULL,
    generation bigint NOT NULL CHECK (generation >= 0),
    commit_index bigint NOT NULL CHECK (commit_index > 0),
    PRIMARY KEY (subject, generation)
);

-- Which commit published each semantic revision, so a reader bound to a
-- revision can tell whether the projection has reached it.
CREATE TABLE {{projection}}.semantic_revision (
    revision bigint PRIMARY KEY CHECK (revision > 0),
    base_revision bigint NOT NULL CHECK (base_revision >= 0),
    commit_index bigint NOT NULL UNIQUE CHECK (commit_index > 0)
);

-- Append-only log of what each commit projected, keyed by (commit_index,
-- ordinal) and written in the same transaction as the projection. The
-- projector applies commit n + 1 only after n has committed, so a consumer
-- reading commit_index > offset (up to the watermark) never skips a row.
CREATE TABLE {{projection}}.projection_event (
    commit_index bigint NOT NULL CHECK (commit_index > 0),
    ordinal smallint NOT NULL CHECK (ordinal >= 0),
    topic text NOT NULL,
    subject text NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (commit_index, ordinal)
);

CREATE FUNCTION {{projection}}.refuse_rewrite() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'append-only table %.% cannot be updated or deleted',
        TG_TABLE_SCHEMA, TG_TABLE_NAME
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$;

CREATE TRIGGER projection_event_append_only
    BEFORE UPDATE OR DELETE ON {{projection}}.projection_event
    FOR EACH ROW EXECUTE FUNCTION {{projection}}.refuse_rewrite();

CREATE TRIGGER applied_commit_append_only
    BEFORE UPDATE OR DELETE ON {{projection}}.applied_commit
    FOR EACH ROW EXECUTE FUNCTION {{projection}}.refuse_rewrite();

CREATE TRIGGER tombstone_append_only
    BEFORE UPDATE OR DELETE ON {{projection}}.tombstone
    FOR EACH ROW EXECUTE FUNCTION {{projection}}.refuse_rewrite();

-- Consumers of the projection event log. A consumer may update derived state
-- or wake the runtime; it may not perform effects.
CREATE TABLE {{projection}}.event_consumer (
    consumer text PRIMARY KEY,
    committed bigint NOT NULL DEFAULT 0 CHECK (committed >= 0)
);
