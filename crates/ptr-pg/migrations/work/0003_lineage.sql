-- Adapter lineage catalog and replay pool. Rows describe sealed artifacts in
-- object storage. Which adapter may serve is not decided by these rows: until
-- adapter promotion is a committed ledger event and serving is admitted
-- through checkpoint binding, this catalog is a working record.

CREATE TABLE {{work}}.adapter (
    id text PRIMARY KEY,
    domain text NOT NULL,
    base_model text NOT NULL,
    base_revision text NOT NULL,
    origin text NOT NULL CHECK (origin IN ('trained', 'consolidated')),
    parent text REFERENCES {{work}}.adapter (id),
    rank integer NOT NULL CHECK (rank > 0),
    artifact text NOT NULL,
    artifact_sha256 bytea NOT NULL CHECK (length(artifact_sha256) = 32),
    data_fingerprint bytea NOT NULL CHECK (length(data_fingerprint) = 32),
    status text NOT NULL CHECK (status IN ('candidate', 'gated', 'serving', 'retired')),
    CHECK (origin = 'trained' OR parent IS NULL),
    CHECK (parent IS DISTINCT FROM id)
);

CREATE TABLE {{work}}.adapter_source (
    consolidated text NOT NULL REFERENCES {{work}}.adapter (id),
    source text NOT NULL REFERENCES {{work}}.adapter (id),
    PRIMARY KEY (consolidated, source),
    CHECK (consolidated <> source)
);

-- The data manifest: every raw input an adapter was trained on. Erasure
-- propagation starts here.
CREATE TABLE {{work}}.adapter_input (
    adapter text NOT NULL REFERENCES {{work}}.adapter (id) ON DELETE CASCADE,
    input text NOT NULL,
    PRIMARY KEY (adapter, input)
);
CREATE INDEX adapter_input_by_input ON {{work}}.adapter_input (input);

CREATE TABLE {{work}}.replay_sample (
    id text PRIMARY KEY,
    stratum text NOT NULL,
    -- A held-out sample can never be pooled: the column admits one value.
    split text NOT NULL CHECK (split = 'train'),
    stability double precision NOT NULL CHECK (stability > 0),
    difficulty double precision NOT NULL CHECK (difficulty BETWEEN 1 AND 10),
    last_probe_model_time double precision NOT NULL,
    lapses integer NOT NULL DEFAULT 0 CHECK (lapses >= 0)
);

-- Probe history on the training clock, append-only.
CREATE TABLE {{work}}.replay_probe (
    sample text NOT NULL REFERENCES {{work}}.replay_sample (id) ON DELETE CASCADE,
    model_time double precision NOT NULL,
    loss double precision NOT NULL,
    PRIMARY KEY (sample, model_time)
);

CREATE TRIGGER replay_probe_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.replay_probe
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
