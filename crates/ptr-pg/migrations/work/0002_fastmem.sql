-- Fast-weight working memories. The write journal is the record; checkpoints
-- are folds of it bound to the digest of the writes they include. Keys and
-- values are stored exactly as the writer composed them (little-endian f32
-- bit patterns), before normalisation, so a restore re-admits the same bits
-- and refolds to the same state. Journal rows are derived from semantic
-- inputs and are as sensitive as those inputs.

CREATE TABLE {{work}}.fastmem_memory (
    id text PRIMARY KEY,
    principal text NOT NULL,
    thread text NOT NULL,
    heads integer NOT NULL CHECK (heads BETWEEN 1 AND 64),
    key_dim integer NOT NULL CHECK (key_dim BETWEEN 1 AND 1024),
    value_dim integer NOT NULL CHECK (value_dim BETWEEN 1 AND 1024),
    checkpoint_interval integer NOT NULL CHECK (checkpoint_interval BETWEEN 1 AND 1000000),
    max_writes integer NOT NULL CHECK (max_writes BETWEEN 1 AND 65536),
    -- Digests of the sealed key projection and value codebook; a memory's
    -- vectors mean nothing under any other.
    projection_digest bytea NOT NULL CHECK (length(projection_digest) = 32),
    codebook_seed bigint NOT NULL,
    UNIQUE (principal, thread)
);

CREATE TABLE {{work}}.fastmem_write (
    memory text NOT NULL REFERENCES {{work}}.fastmem_memory (id) ON DELETE CASCADE,
    seq bigint NOT NULL CHECK (seq > 0),
    source_key text NOT NULL,
    source_generation bigint NOT NULL CHECK (source_generation >= 0),
    input_digest bytea NOT NULL CHECK (length(input_digest) = 32),
    key_cells bytea NOT NULL,
    value_cells bytea NOT NULL,
    beta real NOT NULL CHECK (beta > 0 AND beta <= 1),
    decay_kind text NOT NULL CHECK (decay_kind IN ('none', 'scalar', 'per_channel')),
    decay_cells bytea,
    PRIMARY KEY (memory, seq),
    CHECK ((decay_kind = 'none') = (decay_cells IS NULL))
);

-- Revocation looks writes up by the input they came from.
CREATE INDEX fastmem_write_source
    ON {{work}}.fastmem_write (source_key, source_generation);

CREATE TABLE {{work}}.fastmem_checkpoint (
    memory text NOT NULL REFERENCES {{work}}.fastmem_memory (id) ON DELETE CASCADE,
    applied_seq bigint NOT NULL CHECK (applied_seq > 0),
    -- Digest of the ordered writes this state folds.
    binding_digest bytea NOT NULL CHECK (length(binding_digest) = 32),
    -- PTRFW001 bytes: header, f32 cells, SHA-256. Stored out of line and
    -- uncompressed: f32 noise does not compress.
    state bytea NOT NULL,
    PRIMARY KEY (memory, applied_seq)
);
ALTER TABLE {{work}}.fastmem_checkpoint ALTER COLUMN state SET STORAGE EXTERNAL;
