-- Working state: non-authoritative, not derived from the ledger, never
-- dropped by a projection rebuild. Branches are candidates, never state:
-- merging goes through certification, verification and the runtime's
-- revision-checked commit.

CREATE FUNCTION {{work}}.refuse_rewrite() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'append-only table %.% cannot be updated or deleted',
        TG_TABLE_SCHEMA, TG_TABLE_NAME
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$;

CREATE TABLE {{work}}.branch (
    id text PRIMARY KEY,
    author text NOT NULL,
    base_revision bigint NOT NULL CHECK (base_revision >= 0),
    sealed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE {{work}}.branch_read (
    branch text NOT NULL REFERENCES {{work}}.branch (id) ON DELETE CASCADE,
    key text NOT NULL,
    digest bytea NOT NULL CHECK (length(digest) = 32),
    PRIMARY KEY (branch, key)
);

CREATE TABLE {{work}}.branch_scan (
    branch text NOT NULL REFERENCES {{work}}.branch (id) ON DELETE CASCADE,
    prefix text NOT NULL,
    digest bytea NOT NULL CHECK (length(digest) = 32),
    PRIMARY KEY (branch, prefix)
);

CREATE TABLE {{work}}.branch_relied (
    branch text NOT NULL REFERENCES {{work}}.branch (id) ON DELETE CASCADE,
    target text NOT NULL,
    generation bigint NOT NULL CHECK (generation >= 0),
    PRIMARY KEY (branch, target)
);

CREATE TABLE {{work}}.branch_touched (
    branch text NOT NULL REFERENCES {{work}}.branch (id) ON DELETE CASCADE,
    key text NOT NULL,
    base_digest bytea NOT NULL CHECK (length(base_digest) = 32),
    PRIMARY KEY (branch, key)
);

CREATE TABLE {{work}}.branch_op (
    branch text NOT NULL REFERENCES {{work}}.branch (id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    kind text NOT NULL CHECK (kind IN ('put', 'remove', 'add', 'set_insert', 'set_remove')),
    key text NOT NULL,
    value_kind text CHECK (value_kind IN ('text', 'payload')),
    value_text text,
    value_type text,
    value_source text,
    value_bytes bytea,
    amount bigint,
    member text,
    PRIMARY KEY (branch, ordinal),
    CHECK ((kind = 'put') = (value_kind IS NOT NULL)),
    CHECK ((kind = 'add') = (amount IS NOT NULL)),
    CHECK ((kind IN ('set_insert', 'set_remove')) = (member IS NOT NULL)),
    CHECK (value_kind IS DISTINCT FROM 'text' OR value_text IS NOT NULL),
    CHECK (value_kind IS DISTINCT FROM 'payload'
        OR (value_type IS NOT NULL AND value_source IS NOT NULL AND value_bytes IS NOT NULL))
);

-- One logged triage per branch, with what off-policy evaluation needs.
CREATE TABLE {{work}}.branch_triage (
    branch text PRIMARY KEY REFERENCES {{work}}.branch (id) ON DELETE CASCADE,
    decision text NOT NULL CHECK (decision IN ('auto_propose', 'escalate', 'discard')),
    eligible boolean NOT NULL,
    calibration_slice boolean NOT NULL,
    score real NOT NULL CHECK (score >= 0 AND score <= 1),
    auto_propensity double precision NOT NULL
        CHECK (auto_propensity >= 0 AND auto_propensity <= 1),
    policy_version text NOT NULL,
    decided_at timestamptz NOT NULL DEFAULT now(),
    -- A calibration-slice branch is an eligible branch sent to a person.
    CHECK (NOT calibration_slice OR (eligible AND decision = 'escalate')),
    -- Ineligible branches were decided by verification alone.
    CHECK (eligible OR auto_propensity = 0)
);

-- Outcomes are appended, never rewritten; an adjudication is a person's
-- verdict on the branch itself, independent of what later merged.
CREATE TABLE {{work}}.branch_outcome (
    branch text NOT NULL REFERENCES {{work}}.branch (id) ON DELETE CASCADE,
    outcome text NOT NULL CHECK (outcome IN (
        'merged', 'reverted', 'conflicted', 'discarded',
        'adjudicated_harmful', 'adjudicated_harmless')),
    commit_index bigint CHECK (commit_index > 0),
    observed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (branch, outcome),
    CHECK ((outcome IN ('merged', 'reverted')) = (commit_index IS NOT NULL))
);

CREATE TRIGGER branch_outcome_append_only
    BEFORE UPDATE ON {{work}}.branch_outcome
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
