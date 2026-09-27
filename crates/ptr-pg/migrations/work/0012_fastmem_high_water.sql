-- Fast-weight memories: the journal's sequence high-water mark
-- (FastMemory::with_sequence_high_water).
--
-- A FastMemory keeps every sequence number it took: revoking a write leaves
-- its next number where it was. Revocation deletes journal rows, so the
-- largest number a journal holds could fall, and an append checked against
-- it, or a memory restored from it, took the numbers of revoked writes
-- again: a journal ending at i64::MAX - 1 whose last write was revoked took
-- further writes, one of them at a number already journaled and handed out.
-- Each memory now keeps the largest sequence number ever journaled for it,
-- which never falls; a write's number must exceed it, and restore_memory
-- numbers the next write above it.
--
-- A memory registered before this version starts from the largest number
-- its journal holds, and a journal holding a row at i64::MAX, which
-- migration 11 refuses and the loaders refuse as corrupt, from i64::MAX - 1,
-- the largest mark, so its registration takes no further write. The
-- numbers of writes revoked before this version were not kept, so a
-- journal whose last writes were revoked before it can take those numbers
-- once more.
--
-- A registration whose state exceeds MAX_STATE_CELLS, which migration 9's
-- NOT VALID check leaves in place and every loader refuses as corrupt, keeps
-- mark 0: rewriting its row would check that constraint and fail the
-- upgrade. So does a memory with no journal, whose mark is 0 anyway. No
-- write or restore reads the mark of a registration the loaders refuse.
ALTER TABLE {{work}}.fastmem_memory
    ADD COLUMN last_seq bigint NOT NULL DEFAULT 0,
    ADD CONSTRAINT fastmem_memory_last_seq_below_limit
        CHECK (last_seq >= 0 AND last_seq < 9223372036854775807);

-- A registration is never changed, except that its high-water mark rises.
-- A mark raised without a write stands for writes journaled and revoked
-- since, which the mark would equally have reached; lowering it would hand
-- revoked numbers out again.
CREATE FUNCTION {{work}}.check_fastmem_memory_change() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF to_jsonb(NEW) - 'last_seq' IS DISTINCT FROM to_jsonb(OLD) - 'last_seq' THEN
        RAISE EXCEPTION 'table %.% cannot change a row', TG_TABLE_SCHEMA, TG_TABLE_NAME
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF NEW.last_seq < OLD.last_seq THEN
        RAISE EXCEPTION 'the sequence high-water mark of memory % falls from % to %',
            OLD.id, OLD.last_seq, NEW.last_seq
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER fastmem_memory_never_changed ON {{work}}.fastmem_memory;
CREATE TRIGGER fastmem_memory_never_changed
    BEFORE UPDATE ON {{work}}.fastmem_memory
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_fastmem_memory_change();

-- PostgreSQL checks every CHECK constraint of a row an UPDATE writes, NOT
-- VALID ones included, so only rows that constraint admits are rewritten.
UPDATE {{work}}.fastmem_memory AS memory
    SET last_seq = least(
        (SELECT max(seq) FROM {{work}}.fastmem_write AS write
         WHERE write.memory = memory.id),
        9223372036854775806)
    WHERE memory.heads::bigint * memory.key_dim * memory.value_dim <= 16777216
      AND EXISTS (SELECT 1 FROM {{work}}.fastmem_write AS write
                  WHERE write.memory = memory.id);

-- A write's sequence number exceeds its memory's high-water mark, which then
-- becomes that number. The mark is read by the no-change update that orders
-- appends (migration 11), so at READ COMMITTED it includes every append
-- committed before, and at REPEATABLE READ or SERIALIZABLE an append whose
-- snapshot predates a committed one fails with a serialization error there.
-- A row whose number the column's CHECKs refuse (zero or less, or i64::MAX)
-- is left to them, and a refused row's statement undoes its mark.
CREATE OR REPLACE FUNCTION {{work}}.check_fastmem_capacity() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    capacity integer;
    high_water bigint;
    stored bigint;
BEGIN
    IF NEW.seq IS NULL OR NEW.seq <= 0 OR NEW.seq >= 9223372036854775807 THEN
        RETURN NEW;
    END IF;
    UPDATE {{work}}.fastmem_memory SET max_writes = max_writes WHERE id = NEW.memory
        RETURNING max_writes, last_seq INTO capacity, high_water;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'write % names memory %, which is not registered', NEW.seq, NEW.memory
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF NEW.seq <= high_water THEN
        RAISE EXCEPTION 'write % of memory % does not follow %, the largest sequence number its journal has held',
            NEW.seq, NEW.memory, high_water
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    SELECT count(*) INTO stored FROM {{work}}.fastmem_write WHERE memory = NEW.memory;
    IF stored >= capacity THEN
        RAISE EXCEPTION 'the journal of memory % holds its %-write limit', NEW.memory, capacity
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    UPDATE {{work}}.fastmem_memory SET last_seq = NEW.seq WHERE id = NEW.memory;
    RETURN NEW;
END;
$$;
