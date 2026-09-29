-- The training clock, checked when a probe's transaction commits.
--
-- Migration 13 had every probe update its sample's row without change when
-- its transaction committed (renew_replay_sample), and check_replay_clock
-- let an update that left the row as it was through unchecked. Neither
-- compared the sample with the probe: a transaction that recorded a probe at
-- model time 12 for a sample whose last probe was 10, and never moved the
-- sample, committed, and the sample went on claiming a last probe earlier
-- than a probe of its own append-only history, so a schedule or a memory
-- state rebuilt from replay_sample read a stale elapsed time.
--
-- From this version on a probe's transaction commits only once its sample's
-- last probe is at or after every probe stored for the sample, a deferred
-- check that replaces the no-change update, and every update of a sample is
-- checked, one that leaves the row as it was included.
--
-- The check applies to probes written from this version on, and no stored
-- row is rewritten or checked. A sample a version 13 transaction left behind
-- its probe keeps its row until its next write, which catches it up: an
-- update of it is refused unless it reaches its newest probe, and a new
-- probe of it must follow that one and commits only with the sample at or
-- after it.

-- ---------------------------------------------------------------------------
-- Every update is checked.

-- An update of a sample moves its last probe only forward, keeps its
-- identity, and leaves the last probe at or after every probe stored for the
-- sample, an update that leaves the row as it was included: no function of
-- this schema makes one any more, and one a writer makes is checked as any
-- other. The ordering is migration 13's: the UPDATE holds the row's lock
-- when this runs, and every probe takes that lock before it is checked, so
-- at READ COMMITTED the probes read here are every probe committed before,
-- and at REPEATABLE READ or SERIALIZABLE an update whose snapshot predates a
-- change of the row fails with a serialization error instead.
CREATE OR REPLACE FUNCTION {{work}}.check_replay_clock() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    newest double precision;
BEGIN
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

-- ---------------------------------------------------------------------------
-- The sample reaches its probes by commit.

-- When the transaction that wrote a probe commits, its sample's last probe
-- is at or after every probe stored for the sample. Until then the probe may
-- lead: the sample may be updated to reach it by a later statement of the
-- transaction, or by the same statement however it nests the two, a probe
-- written by a function that the UPDATE of its sample calls included (the
-- check writes no row, so it leaves none changed by a trigger of the
-- current command, which PostgreSQL refuses a later write of with SQLSTATE
-- 27000). Each write is still checked as it is made: the probe against the
-- row it finds (check_replay_probe), and each update against every probe
-- stored, the transaction's own included (check_replay_clock).
--
-- The check reads the row the probe's transaction left: that transaction
-- holds the row's lock from the probe on (check_replay_probe takes it FOR NO
-- KEY UPDATE), so no other transaction changes the row before it commits. A
-- probe whose sample a later statement of its transaction deletes went with
-- it (the key cascades), and there is nothing left to check; a sample
-- written again under the same id is checked against its own probes. A
-- writer that sets the trigger IMMEDIATE (SET CONSTRAINTS) has the check
-- run at the end of each statement instead, nested ones included, and so
-- has a probe refused that a later statement would have caught up with; no
-- SET CONSTRAINTS setting skips it, since every event runs once before
-- commit and no later write moves a last probe back or removes a probe of a
-- stored sample.
--
-- It also keeps the order migration 13's no-change update gave probes and
-- sample updates of one sample at every isolation level. A probe past its
-- sample's last probe commits only with an update of the row by its own
-- transaction, which leaves a version of the row no snapshot taken before
-- that commit sees, so a writer at REPEATABLE READ or SERIALIZABLE holding
-- such a snapshot fails with a serialization error when it locks or updates
-- the row, as check_replay_probe and every update do. A probe at the last
-- probe itself leaves the row only locked, which at those levels a later
-- writer locks and updates again without error, and needs no more: that
-- writer sees the sample at or after the probe's model time, or fails when
-- it locks the row, so a probe it writes is refused unless it follows the
-- probe (one at the same model time by the primary key), and an update it
-- makes cannot move the last probe back before it.
--
-- Migration 13's no-change update wrote the sample's row again, and
-- PostgreSQL checks every CHECK of a row an UPDATE writes, NOT VALID ones
-- included, so a probe of a sample stored before migration 9 with a
-- stability, difficulty or last probe that is not finite was refused by
-- replay_sample_finite when its transaction committed. The check keeps
-- refusing it, with the same SQLSTATE (23514) and constraint name, whether
-- or not the transaction updates the sample (an update of it is refused by
-- the constraint itself).
CREATE FUNCTION {{work}}.check_replay_clock_reached() RETURNS trigger
    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    last_probe double precision;
    finite boolean;
    newest double precision;
BEGIN
    SELECT last_probe_model_time,
           {{work}}.is_finite(stability) AND {{work}}.is_finite(difficulty)
               AND {{work}}.is_finite(last_probe_model_time)
        INTO last_probe, finite
        FROM {{work}}.replay_sample WHERE id = NEW.sample;
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;
    IF NOT finite THEN
        RAISE EXCEPTION 'replay sample % has a memory state or last probe that is not finite',
            NEW.sample
            USING ERRCODE = 'check_violation', CONSTRAINT = 'replay_sample_finite';
    END IF;
    SELECT max(model_time) INTO newest FROM {{work}}.replay_probe WHERE sample = NEW.sample;
    IF newest > last_probe THEN
        RAISE EXCEPTION 'replay sample % keeps its last probe at %, before its probe at %',
            NEW.sample, last_probe, newest
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

DROP TRIGGER replay_probe_renews_sample ON {{work}}.replay_probe;
DROP FUNCTION {{work}}.renew_replay_sample();

CREATE CONSTRAINT TRIGGER replay_probe_reached_by_sample
    AFTER INSERT ON {{work}}.replay_probe
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_replay_clock_reached();
