-- Interference evidence per adapter and layer, as ptr-lineage measured it when
-- the adapter was a candidate: the largest principal-angle overlap with any
-- earlier adapter on the output and input side, their chance levels, and the
-- earlier adapter it overlapped most. Promotion and consolidation decisions can
-- then be audited. Rows are never rewritten.

-- One report per adapter. Its header is written in the transaction that
-- writes its layers, so a second report for the adapter (identical,
-- overlapping or disjoint, concurrent or not) is refused by the primary key
-- rather than merged into the first.
CREATE TABLE {{work}}.adapter_interference_report (
    adapter text PRIMARY KEY REFERENCES {{work}}.adapter (id),
    -- How many layers the report has: exactly its adapter_interference rows,
    -- checked when the report commits.
    layer_count integer NOT NULL CHECK (layer_count > 0),
    recorded_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE {{work}}.adapter_interference (
    adapter text NOT NULL REFERENCES {{work}}.adapter_interference_report (adapter),
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

CREATE TRIGGER adapter_interference_report_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.adapter_interference_report
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();

CREATE TRIGGER adapter_interference_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.adapter_interference
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();

-- At commit, a report written or given a layer in the transaction must have
-- exactly layer_count layer rows. A report therefore commits with all its
-- layers, and since layer rows are never deleted, a layer appended by any
-- later transaction is refused: a stored report never grows.
CREATE FUNCTION {{work}}.check_interference_layer_count() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    recorded integer;
    stored bigint;
BEGIN
    SELECT layer_count INTO recorded
        FROM {{work}}.adapter_interference_report WHERE adapter = NEW.adapter;
    SELECT count(*) INTO stored
        FROM {{work}}.adapter_interference WHERE adapter = NEW.adapter;
    IF recorded IS DISTINCT FROM stored THEN
        RAISE EXCEPTION 'interference report of % has % layers but was recorded with %',
            NEW.adapter, stored, recorded
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER adapter_interference_report_layer_count
    AFTER INSERT ON {{work}}.adapter_interference_report
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_interference_layer_count();

CREATE CONSTRAINT TRIGGER adapter_interference_layer_count
    AFTER INSERT ON {{work}}.adapter_interference
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_interference_layer_count();
