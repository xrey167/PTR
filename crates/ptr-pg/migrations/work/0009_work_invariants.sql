-- Invariants the Rust constructors enforce that the work schema did not, so a
-- row written around them (raw SQL, another client, a defect) is refused by
-- the database rather than found by a loader later, or never.
--
-- A CHECK added to a table that may already hold rows written around those
-- constructors is NOT VALID: every row inserted or updated from this version
-- on is checked, and a schema holding an older violating row still upgrades
-- (validating it would make that impossible); loaders keep refusing such
-- rows. Triggers apply to rows written from this version on.
--
-- A row whose check spans rows of other tables is checked either when it is
-- written, against rows these triggers make immutable, or when its
-- transaction commits (a deferred constraint trigger), so a record written
-- in one transaction may insert its rows in any order. Each deferred check
-- looks at one row and its counterparts by key, never at a whole record.

-- Whether a double precision value is finite. PostgreSQL orders NaN above
-- every number, so `value > 0` holds for NaN and for Infinity.
CREATE FUNCTION {{work}}.is_finite(value double precision) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT value NOT IN ('NaN'::double precision, 'Infinity'::double precision,
                         '-Infinity'::double precision)
$$;

-- ---------------------------------------------------------------------------
-- Sealed branches (work migrations 1 and 8): SealedBranch::from_parts.

-- No operation writes a namespace reserved to ingress
-- (ptr_branch::RESERVED_PREFIXES), a set operation names a non-empty member,
-- and an operation carries exactly the value columns of its kind.
ALTER TABLE {{work}}.branch_op
    ADD CONSTRAINT branch_op_key_not_reserved
        CHECK (NOT starts_with(key, 'request:') AND NOT starts_with(key, 'pod-output:'))
        NOT VALID,
    ADD CONSTRAINT branch_op_member_not_empty
        CHECK (member IS NULL OR member <> '') NOT VALID,
    ADD CONSTRAINT branch_op_value_columns CHECK (
        CASE value_kind
            WHEN 'text' THEN value_type IS NULL AND value_source IS NULL
                AND value_bytes IS NULL
            WHEN 'payload' THEN value_text IS NULL
            ELSE value_text IS NULL AND value_type IS NULL AND value_source IS NULL
                AND value_bytes IS NULL
        END) NOT VALID;

CREATE INDEX branch_op_by_key ON {{work}}.branch_op (branch, key);

-- A stored branch is never rewritten: its header and every row describing it
-- stay as sealed, and go only when the whole branch is deleted.
CREATE TRIGGER branch_never_rewritten
    BEFORE UPDATE ON {{work}}.branch
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_read_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.branch_read
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_scan_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.branch_scan
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_relied_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.branch_relied
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_touched_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.branch_touched
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_op_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.branch_op
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();

-- At commit: a Put or Remove names a key the branch read, and every operated
-- key has a recorded base value and input set.
CREATE FUNCTION {{work}}.check_branch_op() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM {{work}}.branch WHERE id = NEW.branch) THEN
        RETURN NULL;
    END IF;
    IF NEW.kind IN ('put', 'remove') AND NOT EXISTS (
        SELECT 1 FROM {{work}}.branch_read WHERE branch = NEW.branch AND key = NEW.key
    ) THEN
        RAISE EXCEPTION 'branch % overwrites %, which it never read', NEW.branch, NEW.key
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM {{work}}.branch_touched
        WHERE branch = NEW.branch AND key = NEW.key AND inputs_digest IS NOT NULL
    ) THEN
        RAISE EXCEPTION 'branch % operates on % without its base value and input set',
            NEW.branch, NEW.key
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER branch_op_sealed
    AFTER INSERT ON {{work}}.branch_op
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_op();

-- At commit: a base value is recorded only for a key an operation touches,
-- and equals the digest of the same key's read.
CREATE FUNCTION {{work}}.check_branch_touched() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM {{work}}.branch WHERE id = NEW.branch) THEN
        RETURN NULL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM {{work}}.branch_op WHERE branch = NEW.branch AND key = NEW.key
    ) THEN
        RAISE EXCEPTION 'branch % records a base value for %, which no operation touches',
            NEW.branch, NEW.key
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF EXISTS (
        SELECT 1 FROM {{work}}.branch_read
        WHERE branch = NEW.branch AND key = NEW.key AND digest <> NEW.base_digest
    ) THEN
        RAISE EXCEPTION 'branch % records a base value for % that differs from its read',
            NEW.branch, NEW.key
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER branch_touched_sealed
    AFTER INSERT ON {{work}}.branch_touched
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_touched();

-- At commit: a read of a touched key has the digest recorded as its base, so
-- the check holds whichever of the two rows is written last.
CREATE FUNCTION {{work}}.check_branch_read() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM {{work}}.branch_touched
        WHERE branch = NEW.branch AND key = NEW.key AND base_digest <> NEW.digest
    ) THEN
        RAISE EXCEPTION 'branch % read % with a digest other than its recorded base value',
            NEW.branch, NEW.key
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER branch_read_sealed
    AFTER INSERT ON {{work}}.branch_read
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_read();

-- The rules of TriagePolicy::triage every policy shares (explains): a branch
-- verification decided is never auto-proposed, an eligible one is never
-- discarded, outside the calibration slice an eligible branch is
-- auto-proposed exactly when its propensity is positive (one minus a rate
-- below one when the threshold admits it, zero otherwise), and a slice,
-- which needs a positive rate, has propensity below one.
ALTER TABLE {{work}}.branch_triage
    ADD CONSTRAINT branch_triage_verification_never_auto_proposes
        CHECK (eligible OR decision <> 'auto_propose') NOT VALID,
    ADD CONSTRAINT branch_triage_eligible_never_discarded
        CHECK (NOT eligible OR decision <> 'discard') NOT VALID,
    ADD CONSTRAINT branch_triage_auto_proposed_exactly_when_admitted
        CHECK (NOT eligible OR calibration_slice
            OR (decision = 'auto_propose') = (auto_propensity > 0)) NOT VALID,
    ADD CONSTRAINT branch_triage_slice_propensity
        CHECK (NOT calibration_slice OR auto_propensity < 1) NOT VALID;

-- A revert is a later commit undoing the branch's merge
-- (PgSubstrate::record_outcome): it needs the merge, at a smaller commit
-- index. The merge row is never rewritten, and a branch is merged once.
CREATE FUNCTION {{work}}.check_branch_revert() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.outcome = 'reverted' AND NOT EXISTS (
        SELECT 1 FROM {{work}}.branch_outcome
        WHERE branch = NEW.branch AND outcome = 'merged' AND commit_index < NEW.commit_index
    ) THEN
        RAISE EXCEPTION 'branch % is reverted at % without an earlier merge',
            NEW.branch, NEW.commit_index
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER branch_outcome_revert_follows_merge
    BEFORE INSERT ON {{work}}.branch_outcome
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_revert();

-- ---------------------------------------------------------------------------
-- Fast-weight memories (work migration 2): ptr_fastmem::check_config and
-- write admission.

-- The state fits the neural-state payload bound: heads * key_dim * value_dim
-- is at most MAX_STATE_CELLS (16 Mi cells).
ALTER TABLE {{work}}.fastmem_memory
    ADD CONSTRAINT fastmem_memory_state_cells
        CHECK (heads::bigint * key_dim * value_dim <= 16777216) NOT VALID;

-- A registration, and the configuration its journal was written under, is
-- never rewritten; a write journal row is never rewritten either (revocation
-- deletes one).
CREATE TRIGGER fastmem_memory_never_rewritten
    BEFORE UPDATE ON {{work}}.fastmem_memory
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER fastmem_write_never_rewritten
    BEFORE UPDATE ON {{work}}.fastmem_write
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();

-- A write has the shape its memory's configuration admits: a key of
-- heads * key_dim and a value of heads * value_dim f32 cells, one scalar
-- decay factor or one per key cell.
CREATE FUNCTION {{work}}.check_fastmem_write() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    key_len bigint;
    value_len bigint;
BEGIN
    SELECT heads::bigint * key_dim, heads::bigint * value_dim INTO key_len, value_len
        FROM {{work}}.fastmem_memory WHERE id = NEW.memory;
    IF NOT FOUND THEN
        RETURN NEW;
    END IF;
    IF length(NEW.key_cells) <> 4 * key_len
        OR length(NEW.value_cells) <> 4 * value_len
        OR (NEW.decay_kind = 'scalar' AND length(NEW.decay_cells) <> 4)
        OR (NEW.decay_kind = 'per_channel' AND length(NEW.decay_cells) <> 4 * key_len) THEN
        RAISE EXCEPTION 'write % of memory % does not have the shape its configuration admits',
            NEW.seq, NEW.memory
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER fastmem_write_shape
    BEFORE INSERT ON {{work}}.fastmem_write
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_fastmem_write();

-- ---------------------------------------------------------------------------
-- Adapter lineage and replay pool (work migrations 3 and 7):
-- ptr_lineage::Lineage::register and the replay pool.

-- An adapter is registered as a candidate and names only adapters of its own
-- base: a lineage has one base model.
CREATE FUNCTION {{work}}.check_adapter_registration() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    parent_model text;
    parent_revision text;
BEGIN
    IF NEW.status <> 'candidate' THEN
        RAISE EXCEPTION 'adapter % is registered as %, not as a candidate', NEW.id, NEW.status
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF NEW.parent IS NOT NULL THEN
        SELECT base_model, base_revision INTO parent_model, parent_revision
            FROM {{work}}.adapter WHERE id = NEW.parent;
        IF FOUND AND (parent_model <> NEW.base_model OR parent_revision <> NEW.base_revision)
        THEN
            RAISE EXCEPTION 'adapter % continues % from another base model', NEW.id, NEW.parent
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER adapter_registration
    BEFORE INSERT ON {{work}}.adapter
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_adapter_registration();

-- A registered adapter changes only its status, along its lifecycle:
-- candidate to gated, gated to serving, anything but retired to retired.
CREATE FUNCTION {{work}}.check_adapter_transition() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.id, NEW.domain, NEW.base_model, NEW.base_revision, NEW.origin, NEW.parent,
        NEW.rank, NEW.artifact, NEW.artifact_sha256, NEW.data_fingerprint)
        IS DISTINCT FROM
       (OLD.id, OLD.domain, OLD.base_model, OLD.base_revision, OLD.origin, OLD.parent,
        OLD.rank, OLD.artifact, OLD.artifact_sha256, OLD.data_fingerprint) THEN
        RAISE EXCEPTION 'adapter % can change only its status', OLD.id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF NEW.status <> OLD.status AND NOT (
        (OLD.status = 'candidate' AND NEW.status = 'gated')
        OR (OLD.status = 'gated' AND NEW.status = 'serving')
        OR (OLD.status <> 'retired' AND NEW.status = 'retired')
    ) THEN
        RAISE EXCEPTION 'adapter % cannot move from % to %', OLD.id, OLD.status, NEW.status
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER adapter_transition
    BEFORE UPDATE ON {{work}}.adapter
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_adapter_transition();

-- Only a consolidated adapter has sources, and each is on its base. The
-- sources and the data manifest are part of the registered record: never
-- rewritten, and removed only with their adapter, so the edges erasure
-- follows (Lineage::affected_by) cannot be lost.
CREATE FUNCTION {{work}}.check_adapter_source() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    target_origin text;
    target_model text;
    target_revision text;
BEGIN
    SELECT origin, base_model, base_revision INTO target_origin, target_model, target_revision
        FROM {{work}}.adapter WHERE id = NEW.consolidated;
    IF NOT FOUND THEN
        RETURN NEW;
    END IF;
    IF target_origin <> 'consolidated' THEN
        RAISE EXCEPTION 'adapter % was trained, so it has no consolidation source',
            NEW.consolidated
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF EXISTS (
        SELECT 1 FROM {{work}}.adapter
        WHERE id = NEW.source
          AND (base_model <> target_model OR base_revision <> target_revision)
    ) THEN
        RAISE EXCEPTION 'adapter % consolidates % from another base model',
            NEW.consolidated, NEW.source
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER adapter_source_endpoints
    BEFORE INSERT ON {{work}}.adapter_source
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_adapter_source();
CREATE TRIGGER adapter_source_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.adapter_source
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER adapter_input_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.adapter_input
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();

-- At commit: a consolidated adapter names at least one source. Its origin is
-- never rewritten and its sources never removed, so this holds from then on.
CREATE FUNCTION {{work}}.check_consolidation_sources() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.origin = 'consolidated'
        AND EXISTS (SELECT 1 FROM {{work}}.adapter WHERE id = NEW.id)
        AND NOT EXISTS (SELECT 1 FROM {{work}}.adapter_source WHERE consolidated = NEW.id)
    THEN
        RAISE EXCEPTION 'consolidated adapter % names no source', NEW.id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER adapter_consolidation_sources
    AFTER INSERT ON {{work}}.adapter
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_consolidation_sources();

-- Memory states and probes are finite: the pool refuses a nonfinite model
-- time, loss, stability or difficulty.
ALTER TABLE {{work}}.replay_sample
    ADD CONSTRAINT replay_sample_finite
        CHECK ({{work}}.is_finite(stability) AND {{work}}.is_finite(difficulty)
            AND {{work}}.is_finite(last_probe_model_time)) NOT VALID;
ALTER TABLE {{work}}.replay_probe
    ADD CONSTRAINT replay_probe_finite
        CHECK ({{work}}.is_finite(model_time) AND {{work}}.is_finite(loss)) NOT VALID;

-- The training clock is monotone: a sample's last probe never moves back,
-- and a probe never precedes the sample's last probe or a probe already
-- recorded for it.
CREATE FUNCTION {{work}}.check_replay_clock() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.last_probe_model_time < OLD.last_probe_model_time
        OR NEW.id <> OLD.id OR NEW.split <> OLD.split THEN
        RAISE EXCEPTION 'replay sample % cannot move its last probe back or change identity',
            OLD.id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER replay_sample_clock
    BEFORE UPDATE ON {{work}}.replay_sample
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_replay_clock();

CREATE FUNCTION {{work}}.check_replay_probe() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM {{work}}.replay_sample
        WHERE id = NEW.sample AND last_probe_model_time > NEW.model_time
    ) OR EXISTS (
        SELECT 1 FROM {{work}}.replay_probe
        WHERE sample = NEW.sample AND model_time > NEW.model_time
    ) THEN
        RAISE EXCEPTION 'a probe of % at model time % precedes one already recorded',
            NEW.sample, NEW.model_time
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER replay_probe_clock
    BEFORE INSERT ON {{work}}.replay_probe
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_replay_probe();

-- ---------------------------------------------------------------------------
-- Weak supervision (work migrations 4 and 6): LabelSchema::new and
-- VoteMatrix::new.

-- A schema has at least two classes, all distinct, non-empty and not NULL,
-- in a one-dimensional array indexed from one, so class `c` of a vote is
-- element `c + 1`.
CREATE FUNCTION {{work}}.label_classes_valid(classes text[]) RETURNS boolean
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT coalesce(
        array_ndims(classes) = 1
            AND array_lower(classes, 1) = 1
            AND cardinality(classes) >= 2
            AND array_position(classes, NULL) IS NULL
            AND array_position(classes, '') IS NULL
            AND cardinality(classes) = (SELECT count(DISTINCT class) FROM unnest(classes) AS class),
        false)
$$;

ALTER TABLE {{work}}.label_schema
    ADD CONSTRAINT label_schema_classes_valid
        CHECK ({{work}}.label_classes_valid(classes)) NOT VALID;

-- A function has a non-empty name, and an adapter it names has a non-empty
-- id.
ALTER TABLE {{work}}.labeling_function
    ADD CONSTRAINT labeling_function_name_not_empty CHECK (name <> '') NOT VALID,
    ADD CONSTRAINT labeling_function_adapter_not_empty
        CHECK (adapter IS NULL OR adapter <> '') NOT VALID;

-- Schemas and functions are never rewritten: a vote's class index and kind
-- are checked against them when it is written.
CREATE TRIGGER label_schema_never_rewritten
    BEFORE UPDATE ON {{work}}.label_schema
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER labeling_function_never_rewritten
    BEFORE UPDATE ON {{work}}.labeling_function
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();

-- A veto comes only from a verifier and a class vote only from any other
-- kind, and a vote or a gold label names a class of its schema.
CREATE FUNCTION {{work}}.check_label_vote() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    function_kind text;
    classes integer;
BEGIN
    SELECT kind INTO function_kind FROM {{work}}.labeling_function WHERE name = NEW.function;
    IF FOUND AND (NEW.vote_kind = 'veto') <> (function_kind = 'verifier') THEN
        RAISE EXCEPTION 'labeling function % of kind % cannot cast a % vote',
            NEW.function, function_kind, NEW.vote_kind
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    SELECT cardinality(s.classes) INTO classes
        FROM {{work}}.label_schema s WHERE s.id = NEW.label_schema;
    IF FOUND AND NEW.class >= classes THEN
        RAISE EXCEPTION 'class % is outside schema %, which has % classes',
            NEW.class, NEW.label_schema, classes
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER label_vote_kind_and_class
    BEFORE INSERT OR UPDATE ON {{work}}.label_vote
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_label_vote();

CREATE FUNCTION {{work}}.check_gold_label() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    classes integer;
BEGIN
    SELECT cardinality(s.classes) INTO classes
        FROM {{work}}.label_schema s WHERE s.id = NEW.label_schema;
    IF FOUND AND NEW.class >= classes THEN
        RAISE EXCEPTION 'gold class % is outside schema %, which has % classes',
            NEW.class, NEW.label_schema, classes
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER gold_label_class
    BEFORE INSERT OR UPDATE ON {{work}}.gold_label
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_gold_label();
