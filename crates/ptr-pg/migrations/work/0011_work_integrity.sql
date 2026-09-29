-- Invariants work migration 9 left to the loaders, or checked only when the
-- rows they depend on happened to be written first. As in migration 9, a
-- CHECK added to a table that may already hold rows written around the Rust
-- constructors is NOT VALID, triggers apply to rows written from this version
-- on, and the loaders keep refusing older rows.

-- ---------------------------------------------------------------------------
-- Sealed branches: SealedBranch::from_parts.

-- A set operation records whether its member was in the set at its key in the
-- branch's base (BranchOp::SetInsert and SetRemove carry `in_base`), and no
-- other operation records it. Rows stored before this version have no value
-- and none can be derived for them (a wrong `false` on an insert of a member
-- the base held would let certification undo a concurrent removal), so the
-- check is NOT VALID and load_branch refuses such a row
-- (PgError::BranchWithoutSetBase): the branch must be re-run.
ALTER TABLE {{work}}.branch_op
    ADD COLUMN member_in_base boolean,
    ADD CONSTRAINT branch_op_member_in_base
        CHECK ((kind IN ('set_insert', 'set_remove')) = (member_in_base IS NOT NULL))
        NOT VALID;

-- At commit: every set operation on one member of one key records the same
-- base presence, since all of them describe one base. The other rule
-- from_parts applies to it (no member present in a base without the key) and
-- its rule that a Remove names a key whose input set is empty compare a
-- digest with the digest of absence or of the empty set, which the database
-- does not recompute: load_branch refuses rows breaking them.
CREATE FUNCTION {{work}}.check_branch_op_member() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.member_in_base IS NOT NULL AND EXISTS (
        SELECT 1 FROM {{work}}.branch_op
        WHERE branch = NEW.branch AND key = NEW.key AND member = NEW.member
          AND kind IN ('set_insert', 'set_remove')
          AND member_in_base IS DISTINCT FROM NEW.member_in_base
    ) THEN
        RAISE EXCEPTION 'branch % records two base presences for member % of %',
            NEW.branch, NEW.member, NEW.key
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER branch_op_one_base_presence
    AFTER INSERT ON {{work}}.branch_op
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_op_member();

-- Whether a row whose xmin is `inserted` was written by the current
-- transaction or one of its subtransactions (a savepoint, released or still
-- open). Callers ask only about a row they can see: a row another transaction
-- is still writing is invisible, so a visible row whose writer is still in
-- progress is this transaction's own.
--
-- An xmin is 32 bits. The transaction's own writers are numbered from its
-- top-level id upwards, so the 64-bit id is rebuilt as the first id at or
-- after the top-level one with the same low 32 bits, and asked for its
-- status. A row committed at least 2^32 transactions earlier whose xmin
-- equals, modulo 2^32, the id of a transaction still in progress is taken for
-- this transaction's: no row younger than that can be.
CREATE FUNCTION {{work}}.written_by_this_transaction(inserted xid) RETURNS boolean
    LANGUAGE plpgsql VOLATILE AS $$
DECLARE
    top xid8 := pg_current_xact_id();
    top_id numeric := top::text::numeric;
    low numeric := inserted::text::numeric;
    top_low numeric := top_id % 4294967296;
    candidate numeric;
BEGIN
    IF inserted = top::xid THEN
        RETURN true;
    END IF;
    candidate := top_id - top_low + low
        + CASE WHEN low < top_low THEN 4294967296 ELSE 0 END;
    -- No transaction lives through 2^31 transaction ids.
    IF candidate - top_id >= 2147483648 THEN
        RETURN false;
    END IF;
    BEGIN
        RETURN pg_xact_status(candidate::text::xid8) IS NOT DISTINCT FROM 'in progress';
    EXCEPTION WHEN invalid_parameter_value THEN
        -- An id not yet assigned: no transaction has written it.
        RETURN false;
    END;
END;
$$;

-- A stored branch never grows: a read, scan, reliance, touched key or
-- operation is accepted only in the transaction that writes its branch
-- header (in the header's statement or a later one, in any savepoint), and
-- refused in any later transaction, whatever the deferred checks would make
-- of it. The check runs after the row, after the table's CHECKs and the
-- header's foreign key (whose internal trigger sorts first), so a row those
-- refuse keeps their refusal.
--
-- This binds a branch's rows to one transaction, not to the author who
-- sealed it: a writer with the work schema's privileges can still delete a
-- whole branch and write different rows under its id in one transaction, as
-- store_branch writes a branch. Nothing in the database tells those rows from
-- sealed ones.
CREATE FUNCTION {{work}}.check_branch_row_sealed_once() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    header_xmin xid;
BEGIN
    SELECT xmin INTO header_xmin FROM {{work}}.branch WHERE id = NEW.branch;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'a row of %.% names branch %, which is not stored',
            TG_TABLE_SCHEMA, TG_TABLE_NAME, NEW.branch
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF NOT {{work}}.written_by_this_transaction(header_xmin) THEN
        RAISE EXCEPTION 'branch % was stored by an earlier transaction and gains no row in %.%',
            NEW.branch, TG_TABLE_SCHEMA, TG_TABLE_NAME
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER branch_read_sealed_once
    AFTER INSERT ON {{work}}.branch_read
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_row_sealed_once();
CREATE TRIGGER branch_scan_sealed_once
    AFTER INSERT ON {{work}}.branch_scan
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_row_sealed_once();
CREATE TRIGGER branch_relied_sealed_once
    AFTER INSERT ON {{work}}.branch_relied
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_row_sealed_once();
CREATE TRIGGER branch_touched_sealed_once
    AFTER INSERT ON {{work}}.branch_touched
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_row_sealed_once();
CREATE TRIGGER branch_op_sealed_once
    AFTER INSERT ON {{work}}.branch_op
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_row_sealed_once();

-- ---------------------------------------------------------------------------
-- Fast-weight memories: FastMemory::restore and validate_write.

-- A registration is never changed. An UPDATE that leaves every column as it
-- was is let through: every journal append makes one to its memory's row
-- (below), which is what orders appends at every isolation level.
CREATE FUNCTION {{work}}.refuse_change() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'table %.% cannot change a row', TG_TABLE_SCHEMA, TG_TABLE_NAME
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER fastmem_memory_never_rewritten ON {{work}}.fastmem_memory;
CREATE TRIGGER fastmem_memory_never_changed
    BEFORE UPDATE ON {{work}}.fastmem_memory
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_change();

-- The journal takes no sequence number at or above the largest bigint, as a
-- FastMemory takes none at or above its limit: restore_memory restores with
-- the limit i64::MAX (FastMemory::with_sequence_limit), so a journal ending
-- at i64::MAX - 1 restores to a memory that refuses its next write rather
-- than one numbering it where no row can go. The sequence space is exhausted
-- after i64::MAX - 1, however few rows the journal holds.
ALTER TABLE {{work}}.fastmem_write
    ADD CONSTRAINT fastmem_write_seq_below_limit
        CHECK (seq < 9223372036854775807) NOT VALID;

-- The IEEE-754 binary32 bit pattern of each four-byte little-endian cell of
-- `cells` (how ptr-pg stores an f32), read as a non-negative integer, with the
-- cell's position from zero. With the sign bit (2^31) cleared, magnitudes
-- order as these integers do, the infinities and NaN above every finite
-- value.
CREATE FUNCTION {{work}}.f32_cells(cells bytea)
    RETURNS TABLE (cell integer, bits bigint)
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT i,
           get_byte(cells, 4 * i)::bigint
           | (get_byte(cells, 4 * i + 1)::bigint << 8)
           | (get_byte(cells, 4 * i + 2)::bigint << 16)
           | (get_byte(cells, 4 * i + 3)::bigint << 24)
    FROM generate_series(0, length(cells) / 4 - 1) AS i
$$;

-- Every value cell is finite and at most MAX_VALUE_MAGNITUDE (2^24, bits
-- 0x4B800000) in magnitude.
CREATE FUNCTION {{work}}.fastmem_value_cells_valid(cells bytea) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT coalesce(bool_and((bits & 2147483647) <= 1266679808), true)
    FROM {{work}}.f32_cells(cells)
$$;

-- Every key cell is finite (below 0x7F800000 in magnitude) and no head of
-- `key_dim` cells is all zero (0.0 or -0.0; a subnormal cell is not zero).
CREATE FUNCTION {{work}}.fastmem_key_cells_valid(cells bytea, key_dim integer) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT coalesce(bool_and(finite AND nonzero), true)
    FROM (
        SELECT bool_and((bits & 2147483647) < 2139095040) AS finite,
               bool_or((bits & 2147483647) <> 0) AS nonzero
        FROM {{work}}.f32_cells(cells)
        GROUP BY cell / key_dim
    ) AS head
$$;

-- Every decay factor lies in (0, 1]: a positive pattern (sign bit clear) of
-- at most 1.0 (0x3F800000) and not zero; subnormal factors are positive.
CREATE FUNCTION {{work}}.fastmem_decay_cells_valid(cells bytea) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT coalesce(bool_and(bits > 0 AND bits <= 1065353216), true)
    FROM {{work}}.f32_cells(cells)
$$;

-- A write names a registered memory and satisfies validate_write for its
-- configuration: the lengths migration 9 checked, finite key cells with no
-- all-zero head, finite value cells within MAX_VALUE_MAGNITUDE, and decay
-- factors in (0, 1]. The strength's range is the column's CHECK. The memory
-- must be written before the write, in an earlier statement or earlier in
-- the same one.
CREATE OR REPLACE FUNCTION {{work}}.check_fastmem_write() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    head_dim integer;
    key_len bigint;
    value_len bigint;
BEGIN
    SELECT key_dim, heads::bigint * key_dim, heads::bigint * value_dim
        INTO head_dim, key_len, value_len
        FROM {{work}}.fastmem_memory WHERE id = NEW.memory;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'write % names memory %, which is not registered', NEW.seq, NEW.memory
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF length(NEW.key_cells) <> 4 * key_len
        OR length(NEW.value_cells) <> 4 * value_len
        OR (NEW.decay_kind = 'scalar' AND length(NEW.decay_cells) <> 4)
        OR (NEW.decay_kind = 'per_channel' AND length(NEW.decay_cells) <> 4 * key_len) THEN
        RAISE EXCEPTION 'write % of memory % does not have the shape its configuration admits',
            NEW.seq, NEW.memory
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF NOT {{work}}.fastmem_key_cells_valid(NEW.key_cells, head_dim)
        OR NOT {{work}}.fastmem_value_cells_valid(NEW.value_cells)
        OR (NEW.decay_cells IS NOT NULL
            AND NOT {{work}}.fastmem_decay_cells_valid(NEW.decay_cells)) THEN
        RAISE EXCEPTION 'write % of memory % holds a cell ptr_fastmem::validate_write refuses',
            NEW.seq, NEW.memory
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

-- A journal holds at most its memory's max_writes rows, the bound
-- FastMemory::restore checks row by row. The memory's row is updated without
-- change first, which takes a lock append_write's FOR UPDATE and every other
-- append wait for; the journal is then counted in a statement issued after
-- that lock was granted. At READ COMMITTED that count includes every append
-- committed before, and at REPEATABLE READ or SERIALIZABLE an append whose
-- snapshot predates one that committed meanwhile fails with a serialization
-- error at the update, rather than counting a journal it cannot see.
-- Revocation only deletes writes, which frees room, so a revocation in flight
-- can make this refuse a write that would have fit, never admit one that does
-- not.
CREATE FUNCTION {{work}}.check_fastmem_capacity() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    capacity integer;
    stored bigint;
BEGIN
    UPDATE {{work}}.fastmem_memory SET max_writes = max_writes WHERE id = NEW.memory
        RETURNING max_writes INTO capacity;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'write % names memory %, which is not registered', NEW.seq, NEW.memory
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    SELECT count(*) INTO stored FROM {{work}}.fastmem_write WHERE memory = NEW.memory;
    IF stored >= capacity THEN
        RAISE EXCEPTION 'the journal of memory % holds its %-write limit', NEW.memory, capacity
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

-- Named to fire after fastmem_write_shape, so a malformed row is refused
-- before any lock is taken.
CREATE TRIGGER fastmem_write_within_capacity
    BEFORE INSERT ON {{work}}.fastmem_write
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_fastmem_capacity();

-- ---------------------------------------------------------------------------
-- Adapter lineage, replay pool and weak supervision: cross-table checks made
-- when a row is written refuse it when the row they check against is
-- missing, instead of leaving it to the foreign key. A foreign key is
-- checked at the end of its statement, so it accepted a row whose parent
-- comes later in the same statement (a later VALUES row, a data-modifying
-- CTE) although these checks, finding no parent, had skipped it. A parent
-- must now be written before its child, in an earlier statement or earlier
-- in the same one.

CREATE OR REPLACE FUNCTION {{work}}.check_adapter_registration() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    parent_model text;
    parent_revision text;
BEGIN
    IF NEW.status <> 'candidate' THEN
        RAISE EXCEPTION 'adapter % is registered as %, not as a candidate', NEW.id, NEW.status
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    -- The table's CHECKs refuse a parent for a consolidated adapter and a
    -- parent naming the adapter itself, after this trigger: the lookup is
    -- left to them there, so they keep their refusal.
    IF NEW.parent IS NOT NULL AND NEW.origin = 'trained' AND NEW.parent <> NEW.id THEN
        SELECT base_model, base_revision INTO parent_model, parent_revision
            FROM {{work}}.adapter WHERE id = NEW.parent;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'adapter % continues %, which is not registered', NEW.id, NEW.parent
                USING ERRCODE = 'foreign_key_violation';
        END IF;
        IF parent_model <> NEW.base_model OR parent_revision <> NEW.base_revision THEN
            RAISE EXCEPTION 'adapter % continues % from another base model', NEW.id, NEW.parent
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION {{work}}.check_adapter_source() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    target_origin text;
    target_model text;
    target_revision text;
    source_model text;
    source_revision text;
BEGIN
    SELECT origin, base_model, base_revision INTO target_origin, target_model, target_revision
        FROM {{work}}.adapter WHERE id = NEW.consolidated;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'a source names adapter %, which is not registered', NEW.consolidated
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF target_origin <> 'consolidated' THEN
        RAISE EXCEPTION 'adapter % was trained, so it has no consolidation source',
            NEW.consolidated
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    SELECT base_model, base_revision INTO source_model, source_revision
        FROM {{work}}.adapter WHERE id = NEW.source;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'adapter % consolidates %, which is not registered',
            NEW.consolidated, NEW.source
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF source_model <> target_model OR source_revision <> target_revision THEN
        RAISE EXCEPTION 'adapter % consolidates % from another base model',
            NEW.consolidated, NEW.source
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION {{work}}.check_replay_probe() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    last_probe double precision;
BEGIN
    SELECT last_probe_model_time INTO last_probe
        FROM {{work}}.replay_sample WHERE id = NEW.sample;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'a probe names replay sample %, which is not recorded', NEW.sample
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF last_probe > NEW.model_time OR EXISTS (
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

CREATE OR REPLACE FUNCTION {{work}}.check_label_vote() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    function_kind text;
    classes integer;
BEGIN
    SELECT kind INTO function_kind FROM {{work}}.labeling_function WHERE name = NEW.function;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'a vote names labeling function %, which is not recorded', NEW.function
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF (NEW.vote_kind = 'veto') <> (function_kind = 'verifier') THEN
        RAISE EXCEPTION 'labeling function % of kind % cannot cast a % vote',
            NEW.function, function_kind, NEW.vote_kind
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    SELECT cardinality(s.classes) INTO classes
        FROM {{work}}.label_schema s WHERE s.id = NEW.label_schema;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'a vote names label schema %, which is not recorded', NEW.label_schema
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF NEW.class >= classes THEN
        RAISE EXCEPTION 'class % is outside schema %, which has % classes',
            NEW.class, NEW.label_schema, classes
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION {{work}}.check_gold_label() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    classes integer;
BEGIN
    SELECT cardinality(s.classes) INTO classes
        FROM {{work}}.label_schema s WHERE s.id = NEW.label_schema;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'a gold label names label schema %, which is not recorded',
            NEW.label_schema
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF NEW.class >= classes THEN
        RAISE EXCEPTION 'gold class % is outside schema %, which has % classes',
            NEW.class, NEW.label_schema, classes
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

-- ---------------------------------------------------------------------------
-- Removal.

CREATE FUNCTION {{work}}.refuse_removal() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'rows of %.% are never removed by %', TG_TABLE_SCHEMA, TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$;

-- A registered adapter is retired, never deleted, as ptr_lineage::Lineage
-- keeps it: deleting one would cascade its data manifest away (the edges
-- Lineage::affected_by follows for erasure) and free its id for other
-- weights. No substrate path deletes an adapter.
CREATE TRIGGER adapter_never_removed
    BEFORE DELETE ON {{work}}.adapter
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_removal();

-- TRUNCATE fires no row trigger, so every table whose rows are never deleted
-- on their own refuses it too (a whole branch or sample still goes by
-- DELETE, and TRUNCATE ... CASCADE of a parent reaches these tables and is
-- refused with them).
CREATE TRIGGER branch_read_never_truncated BEFORE TRUNCATE ON {{work}}.branch_read
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER branch_scan_never_truncated BEFORE TRUNCATE ON {{work}}.branch_scan
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER branch_relied_never_truncated BEFORE TRUNCATE ON {{work}}.branch_relied
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER branch_touched_never_truncated BEFORE TRUNCATE ON {{work}}.branch_touched
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER branch_op_never_truncated BEFORE TRUNCATE ON {{work}}.branch_op
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER branch_triage_never_truncated BEFORE TRUNCATE ON {{work}}.branch_triage
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER branch_outcome_never_truncated BEFORE TRUNCATE ON {{work}}.branch_outcome
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER replay_probe_never_truncated BEFORE TRUNCATE ON {{work}}.replay_probe
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER triage_policy_never_truncated BEFORE TRUNCATE ON {{work}}.triage_policy
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER triage_policy_sample_never_truncated
    BEFORE TRUNCATE ON {{work}}.triage_policy_sample
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER adapter_interference_report_never_truncated
    BEFORE TRUNCATE ON {{work}}.adapter_interference_report
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER adapter_interference_never_truncated
    BEFORE TRUNCATE ON {{work}}.adapter_interference
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER adapter_source_never_truncated BEFORE TRUNCATE ON {{work}}.adapter_source
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER adapter_input_never_truncated BEFORE TRUNCATE ON {{work}}.adapter_input
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
CREATE TRIGGER adapter_never_truncated BEFORE TRUNCATE ON {{work}}.adapter
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
