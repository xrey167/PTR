-- Weak supervision store. Votes, gold labels and snapshots are kept apart so a
-- predicted label can never be read back as gold.

CREATE TABLE {{work}}.label_schema (
    id text PRIMARY KEY,
    classes text[] NOT NULL CHECK (cardinality(classes) >= 2)
);

CREATE TABLE {{work}}.labeling_function (
    name text PRIMARY KEY,
    kind text NOT NULL CHECK (kind IN ('verifier', 'heuristic', 'model', 'agent'))
);

CREATE TABLE {{work}}.label_item (
    label_schema text NOT NULL REFERENCES {{work}}.label_schema (id),
    item text NOT NULL,
    PRIMARY KEY (label_schema, item)
);

-- A vote is a class estimate or, from a verifier only, a veto; a missing row
-- is an abstention.
CREATE TABLE {{work}}.label_vote (
    label_schema text NOT NULL,
    item text NOT NULL,
    function text NOT NULL REFERENCES {{work}}.labeling_function (name),
    vote_kind text NOT NULL CHECK (vote_kind IN ('class', 'veto')),
    class integer NOT NULL CHECK (class >= 0),
    PRIMARY KEY (label_schema, item, function),
    FOREIGN KEY (label_schema, item) REFERENCES {{work}}.label_item (label_schema, item)
);

CREATE TABLE {{work}}.gold_label (
    label_schema text NOT NULL,
    item text NOT NULL,
    class integer NOT NULL CHECK (class >= 0),
    source text NOT NULL CHECK (source IN ('oracle', 'human')),
    annotator text,
    sampling text NOT NULL CHECK (sampling IN ('uniform', 'active')),
    PRIMARY KEY (label_schema, item),
    FOREIGN KEY (label_schema, item) REFERENCES {{work}}.label_item (label_schema, item),
    CHECK ((source = 'human') = (annotator IS NOT NULL))
);

-- A label snapshot is a dataset: registered with a digest and a card, not
-- referenced by a mutable URI.
CREATE TABLE {{work}}.label_snapshot (
    id text PRIMARY KEY,
    label_schema text NOT NULL REFERENCES {{work}}.label_schema (id),
    dataset_sha256 bytea NOT NULL CHECK (length(dataset_sha256) = 32),
    card text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
