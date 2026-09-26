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

-- At commit, a policy recorded or given a calibration sample in the
-- transaction must have exactly calibration_size sample rows. A policy
-- therefore commits with its whole calibration set, and since sample rows are
-- never deleted, a sample appended by any later transaction is refused: the
-- set a policy was recorded with never grows.
CREATE FUNCTION {{work}}.check_calibration_size() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    policy text;
    recorded integer;
    stored bigint;
BEGIN
    IF TG_TABLE_NAME = 'triage_policy' THEN
        policy := NEW.version;
    ELSE
        policy := NEW.policy_version;
    END IF;
    SELECT calibration_size INTO recorded
        FROM {{work}}.triage_policy WHERE version = policy;
    SELECT count(*) INTO stored
        FROM {{work}}.triage_policy_sample WHERE policy_version = policy;
    IF recorded IS DISTINCT FROM stored THEN
        RAISE EXCEPTION 'triage policy % has % calibration samples but was recorded with %',
            policy, stored, recorded
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER triage_policy_calibration_size
    AFTER INSERT ON {{work}}.triage_policy
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_calibration_size();

CREATE CONSTRAINT TRIGGER triage_policy_sample_calibration_size
    AFTER INSERT ON {{work}}.triage_policy_sample
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_calibration_size();

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
