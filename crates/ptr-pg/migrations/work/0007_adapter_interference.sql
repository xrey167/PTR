-- Interference evidence per adapter and layer, as ptr-lineage measured it when
-- the adapter was a candidate: the largest principal-angle overlap with any
-- earlier adapter on the output and input side, their chance levels, and the
-- earlier adapter it overlapped most. Promotion and consolidation decisions can
-- then be audited. Rows are never rewritten.

CREATE TABLE {{work}}.adapter_interference (
    adapter text NOT NULL REFERENCES {{work}}.adapter (id),
    layer text NOT NULL,
    output_overlap double precision NOT NULL
        CHECK (output_overlap >= 0 AND output_overlap <= 1),
    input_overlap double precision NOT NULL
        CHECK (input_overlap >= 0 AND input_overlap <= 1),
    output_chance double precision NOT NULL
        CHECK (output_chance >= 0 AND output_chance <= 1),
    input_chance double precision NOT NULL
        CHECK (input_chance >= 0 AND input_chance <= 1),
    worst text REFERENCES {{work}}.adapter (id),
    PRIMARY KEY (adapter, layer),
    CHECK (worst IS DISTINCT FROM adapter)
);

CREATE TRIGGER adapter_interference_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.adapter_interference
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
