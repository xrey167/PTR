-- An adapter's record is complete when the transaction that registers it
-- commits.
--
-- Work migration 9 made an adapter's consolidation sources and its data
-- manifest (adapter_source, adapter_input) part of its registered record,
-- never rewritten, and migration 13 made them never deleted; nothing refused
-- an INSERT. Any later transaction could append to either: an input to the
-- manifest the trainer declared, where erasure propagation starts
-- (Lineage::affected_by), or a source to a registered consolidation, so
-- that two registered consolidations could be made each other's source, a
-- cycle Lineage::register cannot construct, since every adapter a record
-- names is registered before it.
--
-- From this version on the database stamps each adapter with the transaction
-- that registers it (adapter.registered_xact), whatever the INSERT gives,
-- and no update changes the stamp. A source or a manifest entry is accepted
-- only in that transaction, in the adapter's statement or a later one, in
-- any savepoint, as a stored branch's rows are only in the transaction of its
-- header (work migration 11). The branch check reads the header's xmin,
-- which is the sealing transaction because nothing updates a header. An
-- adapter's row is updated along its lifecycle, and check_adapter_transition
-- lets an update that leaves every column as it was through, so its xmin
-- names whichever transaction wrote the row last: a transaction that updated
-- it, without changing anything, would have passed for the one that
-- registered it.
--
-- Nor does the stamp alone name the transaction: a transaction id means
-- something only in the cluster that assigned it, and a copy of the work
-- schema that is not physical keeps the stamp as data. A restore does (pg_dump
-- writes a table's rows before its triggers), and so does a logical-
-- replication subscriber (whose apply worker runs no ordinary trigger). In a
-- cluster whose counter is behind a copied stamp, the transaction later
-- given that id would have passed for the registering one, free to extend
-- the adapter's manifest or close a cycle through it. So an adapter is
-- registered by this transaction only when its stamp is this transaction's id
-- and its row was written by this transaction (its xmin, in any savepoint, as
-- migration 11 reads a header's). The registering transaction wrote the row
-- it stamped, and a copy is written by the transaction that stores it, which
-- has committed before any other sees the row; this holds in whichever
-- cluster the copy goes to, one sharing the stamping cluster's system
-- identifier (a promoted standby, a clone of a base backup) included. The one
-- transaction whose id a copied stamp repeats could still give the row its
-- own xmin by updating it, so check_adapter_transition refuses that update:
-- an adapter whose stamp is this transaction's id, on a row this transaction
-- did not write, does not change in this transaction (it does in the next).
-- A physical copy (a standby, a base backup, pg_upgrade) carries the
-- transaction counter with the rows, so its stamps name only transactions it
-- has already run.
--
-- Registered in one transaction, adapters could still name each other: a
-- source is also refused when it descends from its consolidation, through
-- parents and sources, so the lineage stays acyclic as Lineage::register
-- keeps it. Its order within the transaction is not kept: a consolidation
-- may name a source registered after it in the same transaction, which no
-- other transaction can tell from the order Lineage::register uses.
--
-- The checks apply to rows written from this version on; no stored row is
-- rewritten or checked. An adapter registered before it carries no stamp and
-- was registered by an earlier transaction, so it gains no source or
-- manifest entry, and a cycle or a row appended before it stays.
--
-- The work schema's other foreign keys, every one but the two this migration
-- seals (adapter_source.consolidated and adapter_input.adapter), and why each
-- is as it is. An entry names a parent and the children whose keys to it it
-- covers, a child followed by its key's columns wherever it has another key
-- to the same parent:
--   * branch -> branch_read, branch_scan, branch_relied, branch_touched,
--     branch_op: sealed in the transaction of the header (migration 11),
--     whose xmin stays that transaction's, since a header is never updated.
--   * branch -> branch_triage: one row per branch (its key), logged by
--     record_triage once the branch is stored, never rewritten: appended
--     once, after the branch, by design.
--   * branch -> branch_outcome: appended over the branch's life (a merge, a
--     revert after it, one adjudication), never rewritten: appendable by
--     design.
--   * branch -> triage_policy_sample: a stored branch is sampled by each
--     later policy calibrated on it, so it gains one calibration row per such
--     policy, never rewritten; the rows are counted per policy (below), not
--     per branch: appendable by design.
--   * triage_policy -> triage_policy_sample: counted, exactly
--     calibration_size rows when the policy commits and never more
--     (migration 5).
--   * triage_policy -> branch_triage (policy_version): every branch triaged
--     under a committed policy cites it, for as long as the policy is in
--     force, and a triage row is never rewritten: appendable by design.
--   * adapter -> adapter_interference_report: one per adapter (its key),
--     stored by record_interference for an adapter already registered,
--     whatever its status: appended once, after the adapter, by design.
--   * adapter_interference_report -> adapter_interference: counted, exactly
--     layer_count rows when the report commits and never more (migration 7).
--   * adapter -> adapter_source (source): a consolidation registered later
--     may name a registered adapter, whatever its status, as a source, so
--     the adapter gains a row here in any later transaction; the row is
--     sealed with its consolidation (above) and never rewritten, and erasure
--     propagation follows it from the source to the consolidation
--     (Lineage::affected_by): appendable by design.
--   * adapter -> adapter (parent), adapter_interference (worst),
--     labeling_function (adapter): a column of the child's own row, which
--     never changes and names only an adapter registered before it
--     (migrations 7, 9 and 11): a later adapter may continue a registered
--     one, a layer of a later report name one as its worst overlap, and a
--     later model labeling function name one as the adapter that produced
--     its votes, at any time.
--   * fastmem_memory -> fastmem_write: the journal, appendable by design
--     within max_writes and above the sequence high-water mark (migrations 11
--     and 12), and deleted by revocation.
--   * fastmem_memory -> fastmem_checkpoint: folds of the journal taken as it
--     grows, each bound to the digest of the writes it includes and deleted
--     with them: appendable by design.
--   * replay_sample -> replay_probe: the probe history on the training clock,
--     appendable by design, in clock order (migrations 9, 13 and 14).
--   * label_schema -> label_item, label_snapshot: the weak-supervision store
--     grows as items arrive, and its rows are not append-only (migration 13);
--     a schema is never rewritten, so what a row was checked against stays.
--     A snapshot is one row naming its dataset by digest, with no member
--     rows.
--   * label_item -> label_vote, gold_label: functions vote and annotators
--     label as the store grows; not append-only (migration 13).
--   * labeling_function -> label_vote: as for an item; a function is never
--     rewritten, so the kind a vote was checked against stays.

ALTER TABLE {{work}}.adapter ADD COLUMN registered_xact xid8;

-- The stamp: the transaction that writes the row, its top-level id in any
-- savepoint. A registration rolled back with its savepoint leaves no row to
-- stamp.
CREATE FUNCTION {{work}}.stamp_adapter_registration() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    NEW.registered_xact := pg_current_xact_id();
    RETURN NEW;
END;
$$;

CREATE TRIGGER adapter_registration_stamped
    BEFORE INSERT ON {{work}}.adapter
    FOR EACH ROW EXECUTE FUNCTION {{work}}.stamp_adapter_registration();

-- Whether this transaction registered the adapter whose row carries the
-- stamp `registered` and the xmin `written`: the stamp is this transaction's
-- id, and this transaction, in any savepoint, wrote the row. A stamp copied
-- from another cluster may repeat the id; the row it was copied onto was
-- written by the transaction that stored the copy.
CREATE FUNCTION {{work}}.registered_by_this_transaction(registered xid8, written xid)
    RETURNS boolean
    LANGUAGE plpgsql VOLATILE SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    IF registered IS DISTINCT FROM pg_current_xact_id() THEN
        RETURN false;
    END IF;
    RETURN {{work}}.written_by_this_transaction(written);
END;
$$;

-- A registered adapter changes only its status, along its lifecycle:
-- candidate to gated, gated to serving, anything but retired to retired.
-- Every other column is compared, the stamp included, so an update can
-- neither move a registration into its own transaction nor out of one. An
-- adapter whose stamp is this transaction's id but whose row this transaction
-- did not write carries a stamp copied from another cluster: an update would
-- give its row this transaction's xmin and pass it for registered here, so it
-- does not change in this transaction.
CREATE OR REPLACE FUNCTION {{work}}.check_adapter_transition() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    written xid;
BEGIN
    IF to_jsonb(NEW) - 'status' IS DISTINCT FROM to_jsonb(OLD) - 'status' THEN
        RAISE EXCEPTION 'adapter % can change only its status', OLD.id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF OLD.registered_xact = pg_current_xact_id() THEN
        SELECT xmin INTO written FROM {{work}}.adapter WHERE id = OLD.id;
        IF NOT {{work}}.registered_by_this_transaction(OLD.registered_xact, written) THEN
            RAISE EXCEPTION 'adapter % is stamped with the id of this transaction, which did not register it, and does not change in it',
                OLD.id
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;
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

-- A source or a manifest entry is accepted only in the transaction that
-- registered its adapter (named by the column the trigger's argument gives),
-- and refused in any later one, whatever else that transaction writes, and in
-- a transaction whose id a copied stamp repeats. The check runs after the
-- row, after the table's CHECKs and the adapter's foreign key (whose internal
-- trigger sorts first), so a row those refuse keeps their refusal.
--
-- As for a branch, this binds the rows to one transaction, not to the
-- trainer: a writer with the work schema's privileges can still register an
-- adapter under a new id with whatever sources and manifest it likes.
CREATE FUNCTION {{work}}.check_adapter_row_sealed_once() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    named text := to_jsonb(NEW) ->> TG_ARGV[0];
    registered xid8;
    written xid;
BEGIN
    SELECT registered_xact, xmin INTO registered, written
        FROM {{work}}.adapter WHERE id = named;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'a row of %.% names adapter %, which is not registered',
            TG_TABLE_SCHEMA, TG_TABLE_NAME, named
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF NOT {{work}}.registered_by_this_transaction(registered, written) THEN
        RAISE EXCEPTION 'adapter % was registered by an earlier transaction and gains no row in %.%',
            named, TG_TABLE_SCHEMA, TG_TABLE_NAME
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER adapter_source_sealed_once
    AFTER INSERT ON {{work}}.adapter_source
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_adapter_row_sealed_once('consolidated');
CREATE TRIGGER adapter_input_sealed_once
    AFTER INSERT ON {{work}}.adapter_input
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_adapter_row_sealed_once('adapter');

-- A source does not descend from its consolidation: the consolidation is not
-- among the adapters the source continues or consolidates, directly or
-- through others. The consolidation was registered by this transaction (the
-- check above: adapter_source_sealed_once sorts, and so fires, before
-- adapter_source_without_cycle), and an adapter another transaction
-- registered, in this cluster or in the one a copy came from, descends only
-- from adapters stored before it, never from one this transaction registers,
-- so the walk follows only adapters this transaction registered, as the
-- check above tells them. It runs after the statement's rows are written, so
-- two sources one statement writes that close a cycle each see the other.
CREATE FUNCTION {{work}}.check_adapter_source_acyclic() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    IF EXISTS (
        WITH RECURSIVE ancestor (id) AS (
            SELECT a.id FROM {{work}}.adapter a
                WHERE a.id = NEW.source
                  AND {{work}}.registered_by_this_transaction(a.registered_xact, a.xmin)
            UNION
            SELECT a.id
                FROM ancestor
                CROSS JOIN LATERAL (
                    SELECT p.parent FROM {{work}}.adapter p
                        WHERE p.id = ancestor.id AND p.parent IS NOT NULL
                    UNION ALL
                    SELECT e.source FROM {{work}}.adapter_source e
                        WHERE e.consolidated = ancestor.id
                ) AS named (id)
                JOIN {{work}}.adapter a ON a.id = named.id
                WHERE {{work}}.registered_by_this_transaction(a.registered_xact, a.xmin)
        )
        SELECT 1 FROM ancestor WHERE ancestor.id = NEW.consolidated
    ) THEN
        RAISE EXCEPTION 'adapter % consolidates %, which descends from it',
            NEW.consolidated, NEW.source
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER adapter_source_without_cycle
    AFTER INSERT ON {{work}}.adapter_source
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_adapter_source_acyclic();
