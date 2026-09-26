-- Triage policies as they were in force, and exactly which adjudicated
-- calibration-slice branches each was calibrated on. A policy's harm rate may
-- only be estimated on adjudications it did not see (F003), which is checkable
-- only when the calibration set is recorded. Rows are never rewritten, and a
-- calibration set is complete when its policy commits.

CREATE TABLE {{work}}.triage_policy (
    version text PRIMARY KEY CHECK (version <> ''),
    rule text NOT NULL
        CHECK (rule IN ('manual', 'conformal_risk_control', 'learn_then_test')),
    -- NULL: nothing is auto-proposed.
    threshold real CHECK (threshold >= 0 AND threshold <= 1),
    calibration_rate double precision NOT NULL
        CHECK (calibration_rate >= 0 AND calibration_rate < 1),
    alpha double precision CHECK (alpha > 0 AND alpha < 1),
    delta double precision CHECK (delta > 0 AND delta < 1),
    -- How many branches the threshold was calibrated on: exactly the policy's
    -- triage_policy_sample rows, checked when the policy commits.
    calibration_size integer NOT NULL CHECK (calibration_size >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    -- A calibrated rule names its risk level; learn-then-test also its confidence.
    CHECK ((rule = 'manual') = (alpha IS NULL)),
    CHECK ((rule = 'learn_then_test') = (delta IS NOT NULL)),
    -- A manual threshold was calibrated on nothing, a calibrated one on something.
    CHECK ((rule = 'manual') = (calibration_size = 0))
);

CREATE TABLE {{work}}.triage_policy_sample (
    policy_version text NOT NULL REFERENCES {{work}}.triage_policy (version),
    -- A branch a recorded policy was calibrated on cannot be deleted: its
    -- adjudication is the evidence the threshold rests on.
    branch text NOT NULL REFERENCES {{work}}.branch (id),
    PRIMARY KEY (policy_version, branch)
);

CREATE TRIGGER triage_policy_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.triage_policy
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();

CREATE TRIGGER triage_policy_sample_append_only
    BEFORE UPDATE OR DELETE ON {{work}}.triage_policy_sample
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();

-- A policy commits with exactly calibration_size sample rows, and its set
-- never grows. Two checks together guarantee it, and each counts a set once
-- rather than once per sample, so recording a set of N branches in one
-- statement reads O(N) sample rows, not O(N^2):
--   * at commit, a policy recorded in the transaction must have exactly
--     calibration_size samples, counted once for the policy row;
--   * at the end of every statement that adds sample rows, no policy it adds
--     to may have more samples than its calibration_size, counted once per
--     policy the statement names.
-- Sample rows are never deleted and a committed set already has its size, so
-- a sample appended by any later transaction is refused by the second check.
CREATE FUNCTION {{work}}.check_calibration_size() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    stored bigint;
BEGIN
    SELECT count(*) INTO stored
        FROM {{work}}.triage_policy_sample WHERE policy_version = NEW.version;
    IF stored <> NEW.calibration_size THEN
        RAISE EXCEPTION 'triage policy % has % calibration samples but was recorded with %',
            NEW.version, stored, NEW.calibration_size
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER triage_policy_calibration_size
    AFTER INSERT ON {{work}}.triage_policy
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_calibration_size();

CREATE FUNCTION {{work}}.check_calibration_growth() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    policy text;
    recorded integer;
    stored bigint;
BEGIN
    FOR policy, recorded IN
        SELECT version, calibration_size FROM {{work}}.triage_policy
            WHERE version IN (SELECT policy_version FROM added)
    LOOP
        SELECT count(*) INTO stored
            FROM {{work}}.triage_policy_sample WHERE policy_version = policy;
        IF stored > recorded THEN
            RAISE EXCEPTION 'triage policy % has % calibration samples but was recorded with %',
                policy, stored, recorded
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;
    END LOOP;
    RETURN NULL;
END;
$$;

CREATE TRIGGER triage_policy_sample_calibration_size
    AFTER INSERT ON {{work}}.triage_policy_sample
    REFERENCING NEW TABLE AS added
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.check_calibration_growth();

-- A triage row logged from this version on names a recorded policy. The key
-- checks only that the version exists, not that the row's decision and
-- propensity follow from that policy's threshold and calibration rate. It is
-- NOT VALID: rows logged before work version 5 keep citing versions no table
-- recorded, and validating them would make every schema holding one
-- impossible to upgrade.
ALTER TABLE {{work}}.branch_triage
    ADD CONSTRAINT branch_triage_policy_version_fkey
    FOREIGN KEY (policy_version) REFERENCES {{work}}.triage_policy (version)
    NOT VALID;
