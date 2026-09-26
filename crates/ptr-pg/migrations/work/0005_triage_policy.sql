-- Triage policies as they were in force, and exactly which adjudicated
-- calibration-slice branches each was calibrated on. A policy's harm rate may
-- only be estimated on adjudications it did not see (F003), which is checkable
-- only when the calibration set is recorded. Rows are never rewritten.

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
    created_at timestamptz NOT NULL DEFAULT now(),
    -- A calibrated rule names its risk level; learn-then-test also its confidence.
    CHECK ((rule = 'manual') = (alpha IS NULL)),
    CHECK ((rule = 'learn_then_test') = (delta IS NOT NULL))
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

-- Every logged triage names a recorded policy, so its propensity and the
-- threshold it applied can be recovered.
ALTER TABLE {{work}}.branch_triage
    ADD CONSTRAINT branch_triage_policy_version_fkey
    FOREIGN KEY (policy_version) REFERENCES {{work}}.triage_policy (version);
