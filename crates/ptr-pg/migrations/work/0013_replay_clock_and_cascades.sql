-- Three rules of earlier work migrations that held only for writers that did
-- not race or nest their writes or undo them within one statement, and the
-- name resolution every work function relied on.
--
-- * The training clock (migrations 9 and 11). A sample update was compared
--   only with the row it replaced: with a probe at model time 12 stored for
--   a sample whose row still said 10, moving the row to 11 passed, and the
--   sample then claimed a last probe earlier than a probe of its own
--   append-only history. A probe and a sample update of one sample were
--   each checked against what the other had not committed yet, so both could
--   commit and leave the same state.
-- * Removal (migration 1, refuse_rewrite). A DELETE was let through whenever
--   pg_trigger_depth() > 1, which was meant for the delete cascading from a
--   deleted branch or sample but holds for a DELETE issued by any trigger
--   function. A trigger the writer created, on any table it may create one
--   on (a temporary one included), removed rows no parent's deletion ever
--   takes (triage policies and their calibration sets, interference reports,
--   adapter sources, and data manifests, whose adapter is never deleted) and
--   single rows of a branch or sample that stayed stored: an adjudication
--   could be removed and written again with the opposite verdict under a
--   policy calibrated on it.
-- * Parents written again (migrations 4 and 5). A policy's calibration rows
--   kept their branches, label items and snapshots their label schema, and
--   votes their labeling function through NO ACTION foreign keys, which
--   PostgreSQL checks at the end of the statement and passes when a row with
--   the old key is stored again by then. One statement could delete a
--   calibration branch, then each of its rows (the removal triggers below let
--   them go, the branch being gone), and write the branch again under its id:
--   the adjudication was gone, and the opposite verdict could be written
--   under the policy. Schemas and functions refuse only UPDATE (migration 9),
--   so one statement could delete a schema and write it again with other
--   classes, or a function with another kind: every stored vote and gold
--   label kept a class index that now named another class or none, and a
--   verifier held class votes.
--
-- * Name resolution (every work migration). No function of the work schema
--   fixed its search_path. A PL/pgSQL body, and a SQL-language body that is
--   not BEGIN ATOMIC, is parsed when it runs, under the search_path of the
--   session that fired it, so its operators (=, <, >, <>) and unqualified
--   functions resolved through whatever that session put first. A writer
--   able to create an operator in some schema of its own could put it
--   first and have it decide every comparison these functions make with
--   that operator, and so the checks resting on them, the CHECKs that call
--   is_finite and the other helpers included. Without a fixed path, a text
--   = that never holds lets the removal triggers below delete an
--   adjudication of a stored branch, float < and > let the clock checks
--   pass a clock moved back, and a float <> that always holds makes
--   is_finite accept NaN.
--
-- The first two are triggers: they apply to writes from this version on, and
-- no stored row is rewritten or checked. The third replaces four foreign
-- keys, each of which checks the stored rows once, against parents the key
-- it replaces already required them to name. The fourth changes only how the
-- functions resolve names; every table and function they name is qualified
-- with the work schema, and everything else they use is in pg_catalog.

-- ---------------------------------------------------------------------------
-- The training clock.

-- An update that changes a sample moves its last probe only forward, keeps
-- its identity, and leaves the last probe at or after every probe stored for
-- the sample, whether or not it moves the clock. An update that leaves the
-- row as it was passes unchecked: every probe makes one to its sample's row
-- at the end of its statement (renew_replay_sample, below), which with the
-- lock each probe takes before it is checked is what orders probes and
-- sample updates at every isolation level.
--
-- The UPDATE holds the row's lock when this runs, and every probe takes that
-- lock before it is checked. At READ COMMITTED the probes read here are
-- every probe committed before, and none of the sample commits until this
-- update does; at REPEATABLE READ or SERIALIZABLE an update whose snapshot
-- predates a probe committed meanwhile fails with a serialization error
-- instead, since that probe's own update of the row came after the snapshot.
CREATE OR REPLACE FUNCTION {{work}}.check_replay_clock() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    newest double precision;
BEGIN
    IF NEW IS NOT DISTINCT FROM OLD THEN
        RETURN NEW;
    END IF;
    IF NEW.last_probe_model_time < OLD.last_probe_model_time
        OR NEW.id <> OLD.id OR NEW.split <> OLD.split THEN
        RAISE EXCEPTION 'replay sample % cannot move its last probe back or change identity',
            OLD.id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    SELECT max(model_time) INTO newest FROM {{work}}.replay_probe WHERE sample = OLD.id;
    IF newest > NEW.last_probe_model_time THEN
        RAISE EXCEPTION 'replay sample % cannot put its last probe at %, before its probe at %',
            OLD.id, NEW.last_probe_model_time, newest
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

-- A probe is checked after it locks its sample's row FOR NO KEY UPDATE, the
-- lock every update of the sample and every other probe of it waits for; the
-- stored probes are then read in a statement issued after that lock was
-- granted. At READ COMMITTED the row read is its latest version and the
-- probes read include every probe committed before; at REPEATABLE READ or
-- SERIALIZABLE a probe whose snapshot predates a sample update or another
-- probe committed meanwhile fails with a serialization error at the lock,
-- since each of those left a newer version of the row (a probe through
-- renew_replay_sample, below). A probe whose model time the table refuses
-- (NULL or not finite) is left to the table's constraints, without the lock.
--
-- The lock changes no row. A no-change update here would leave the row
-- changed by a trigger of the current command, and PostgreSQL then refuses
-- the command's own later update or delete of that row (SQLSTATE 27000), so
-- a statement that records a probe and then updates its sample could not be
-- written. With the lock it is checked as two statements in turn would be:
-- the probe against the row as it found it, the update against every probe
-- stored, the statement's own included.
CREATE OR REPLACE FUNCTION {{work}}.check_replay_probe() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    last_probe double precision;
BEGIN
    IF NEW.model_time IS NULL OR NOT {{work}}.is_finite(NEW.model_time) THEN
        RETURN NEW;
    END IF;
    SELECT last_probe_model_time INTO last_probe
        FROM {{work}}.replay_sample WHERE id = NEW.sample
        FOR NO KEY UPDATE;
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

-- At the end of the statement that wrote it, every probe updates its
-- sample's row without change, which check_replay_clock lets through. The
-- row then has a version written by the probe's transaction, which no
-- snapshot taken before that transaction commits sees, and a writer at
-- REPEATABLE READ or SERIALIZABLE holding such a snapshot fails with a
-- serialization error when it locks or updates the row. The lock
-- check_replay_probe takes would not do that alone: at those levels a row
-- that a committed transaction only locked is locked and updated again
-- without error.
--
-- Running after the statement's own writes, the update finds the row as the
-- statement left it, so it does not refuse a statement that records a probe
-- and then updates the sample. A statement that records a probe and then
-- deletes its sample is refused by the probe's foreign key (SQLSTATE 23503),
-- whose check fires before this trigger (AFTER triggers of one event fire in
-- name order, and the key's internal trigger is named RI_...); a sample this
-- function finds gone is refused with the same code. The update writes the
-- sample's row again, and PostgreSQL checks every CHECK of a row an UPDATE
-- writes, NOT VALID ones included: a probe of a sample stored before
-- migration 9 with a stability, difficulty or clock that is not finite is
-- refused by replay_sample_finite.
CREATE FUNCTION {{work}}.renew_replay_sample() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    UPDATE {{work}}.replay_sample SET lapses = lapses WHERE id = NEW.sample;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'a probe names replay sample %, which is not recorded', NEW.sample
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER replay_probe_renews_sample
    AFTER INSERT ON {{work}}.replay_probe
    FOR EACH ROW EXECUTE FUNCTION {{work}}.renew_replay_sample();

-- ---------------------------------------------------------------------------
-- Removal only with the parent.

-- Refuses every UPDATE and every DELETE, whoever issues it. Tables whose rows
-- go with a parent row delete them through the triggers below instead.
CREATE OR REPLACE FUNCTION {{work}}.refuse_rewrite() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    RAISE EXCEPTION 'append-only table %.% cannot be updated or deleted',
        TG_TABLE_SCHEMA, TG_TABLE_NAME
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$;

-- A row of a branch is deleted only once its branch is. The delete that
-- cascades from the branch's row runs after that row is deleted in the same
-- transaction, so the branch is gone for it; any other delete, issued by a
-- statement or by a trigger function, finds the branch still stored and is
-- refused.
CREATE FUNCTION {{work}}.refuse_removal_from_stored_branch() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM {{work}}.branch WHERE id = OLD.branch) THEN
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'a row of %.% goes only with branch %, which is stored',
        TG_TABLE_SCHEMA, TG_TABLE_NAME, OLD.branch
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$;

-- A probe is deleted only once its sample is, as a branch's rows are with
-- their branch.
CREATE FUNCTION {{work}}.refuse_removal_from_stored_sample() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM {{work}}.replay_sample WHERE id = OLD.sample) THEN
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'a probe of replay sample % goes only with the sample, which is stored',
        OLD.sample
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$;

-- The tables whose rows cascade from a parent: rewriting stays refused, and
-- deletion goes through the parent's check. Every other table refuse_rewrite
-- guards against DELETE (triage_policy, triage_policy_sample,
-- adapter_interference_report, adapter_interference, adapter_source and
-- adapter_input, whose adapter is never deleted) now refuses every one.
DROP TRIGGER branch_read_append_only ON {{work}}.branch_read;
CREATE TRIGGER branch_read_append_only
    BEFORE UPDATE ON {{work}}.branch_read
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_read_removed_with_branch
    BEFORE DELETE ON {{work}}.branch_read
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_removal_from_stored_branch();

DROP TRIGGER branch_scan_append_only ON {{work}}.branch_scan;
CREATE TRIGGER branch_scan_append_only
    BEFORE UPDATE ON {{work}}.branch_scan
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_scan_removed_with_branch
    BEFORE DELETE ON {{work}}.branch_scan
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_removal_from_stored_branch();

DROP TRIGGER branch_relied_append_only ON {{work}}.branch_relied;
CREATE TRIGGER branch_relied_append_only
    BEFORE UPDATE ON {{work}}.branch_relied
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_relied_removed_with_branch
    BEFORE DELETE ON {{work}}.branch_relied
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_removal_from_stored_branch();

DROP TRIGGER branch_touched_append_only ON {{work}}.branch_touched;
CREATE TRIGGER branch_touched_append_only
    BEFORE UPDATE ON {{work}}.branch_touched
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_touched_removed_with_branch
    BEFORE DELETE ON {{work}}.branch_touched
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_removal_from_stored_branch();

DROP TRIGGER branch_op_append_only ON {{work}}.branch_op;
CREATE TRIGGER branch_op_append_only
    BEFORE UPDATE ON {{work}}.branch_op
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_op_removed_with_branch
    BEFORE DELETE ON {{work}}.branch_op
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_removal_from_stored_branch();

DROP TRIGGER branch_triage_append_only ON {{work}}.branch_triage;
CREATE TRIGGER branch_triage_append_only
    BEFORE UPDATE ON {{work}}.branch_triage
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_triage_removed_with_branch
    BEFORE DELETE ON {{work}}.branch_triage
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_removal_from_stored_branch();

DROP TRIGGER branch_outcome_append_only ON {{work}}.branch_outcome;
CREATE TRIGGER branch_outcome_append_only
    BEFORE UPDATE ON {{work}}.branch_outcome
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_outcome_removed_with_branch
    BEFORE DELETE ON {{work}}.branch_outcome
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_removal_from_stored_branch();

DROP TRIGGER replay_probe_append_only ON {{work}}.replay_probe;
CREATE TRIGGER replay_probe_append_only
    BEFORE UPDATE ON {{work}}.replay_probe
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER replay_probe_removed_with_sample
    BEFORE DELETE ON {{work}}.replay_probe
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_removal_from_stored_sample();

-- ---------------------------------------------------------------------------
-- Parents written again.

-- RESTRICT refuses the delete of a parent any row names, whatever the
-- statement writes after it: unlike NO ACTION it does not look for the key
-- stored again at the end of the statement, and it cannot be deferred. It
-- looks for the rows naming the parent at the end of the statement, so a
-- statement that also deletes every such row may still delete the parent.
--
-- Calibration rows are never deleted (refuse_rewrite above) or truncated
-- (migration 11), so a branch a recorded policy was calibrated on is never
-- deleted, and its triage and adjudication, which go only with it, stay. A
-- branch no calibration row names is deleted with its rows as before.
ALTER TABLE {{work}}.triage_policy_sample
    DROP CONSTRAINT triage_policy_sample_branch_fkey,
    ADD CONSTRAINT triage_policy_sample_branch_fkey
        FOREIGN KEY (branch) REFERENCES {{work}}.branch (id) ON DELETE RESTRICT;

-- A label schema an item or a snapshot names, and a labeling function a vote
-- names, is not deleted, so it is not written again under its key while those
-- rows stay: a vote or gold label (which names its schema through its item)
-- keeps the class indices and kind it was checked against when written. A
-- statement that deletes the parent together with every row naming it,
-- votes and gold labels included, is not refused, and one nothing names is
-- deleted, and may be written again, as before. Votes, gold labels, items
-- and snapshots themselves are not append-only.
ALTER TABLE {{work}}.label_item
    DROP CONSTRAINT label_item_label_schema_fkey,
    ADD CONSTRAINT label_item_label_schema_fkey
        FOREIGN KEY (label_schema) REFERENCES {{work}}.label_schema (id) ON DELETE RESTRICT;
ALTER TABLE {{work}}.label_snapshot
    DROP CONSTRAINT label_snapshot_label_schema_fkey,
    ADD CONSTRAINT label_snapshot_label_schema_fkey
        FOREIGN KEY (label_schema) REFERENCES {{work}}.label_schema (id) ON DELETE RESTRICT;
ALTER TABLE {{work}}.label_vote
    DROP CONSTRAINT label_vote_function_fkey,
    ADD CONSTRAINT label_vote_function_fkey
        FOREIGN KEY (function) REFERENCES {{work}}.labeling_function (name) ON DELETE RESTRICT;

-- ---------------------------------------------------------------------------
-- Name resolution.

-- Every other function of the work schema resolves names under the same
-- fixed path as the ones above: pg_catalog first, so no operator, function or
-- type the caller's path offers is consulted, and pg_temp last and explicit,
-- so no temporary relation is found before a catalog one (PostgreSQL never
-- looks up operators or functions in pg_temp). The setting is part of each
-- function and applies whenever it runs, whoever fires it. It does not
-- survive a later CREATE OR REPLACE FUNCTION that omits it, so a function
-- replaced by a later migration must state it again.
ALTER FUNCTION {{work}}.check_adapter_registration() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_adapter_source() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_adapter_transition() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_branch_op() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_branch_op_member() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_branch_read() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_branch_revert() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_branch_row_sealed_once() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_branch_touched() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_branch_triage_policy() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_calibration_growth() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_calibration_size() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_consolidation_sources() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_fastmem_capacity() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_fastmem_memory_change() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_fastmem_write() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_gold_label() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_interference_growth() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_interference_layer_count() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.check_label_vote() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.refuse_change() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.refuse_removal() SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.written_by_this_transaction(xid) SET search_path = pg_catalog, pg_temp;
-- The SQL-language helpers the CHECKs call. A function with a SET clause is
-- no longer inlined into the expression that calls it; each CHECK calls it
-- as a function instead, with the same result.
ALTER FUNCTION {{work}}.is_finite(double precision) SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.label_classes_valid(text[]) SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.f32_cells(bytea) SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.fastmem_value_cells_valid(bytea) SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.fastmem_key_cells_valid(bytea, integer) SET search_path = pg_catalog, pg_temp;
ALTER FUNCTION {{work}}.fastmem_decay_cells_valid(bytea) SET search_path = pg_catalog, pg_temp;
