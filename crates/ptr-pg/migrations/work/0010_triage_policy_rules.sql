-- A triage row is one the policy it cites can have produced, and a policy's
-- calibration rate is one ptr_branch::TriagePolicy::new admits. Migration 9
-- gave triage rows only the rules every policy shares, so a row written
-- around record_triage could still cite a policy that never logs it: a slice
-- row of a policy with calibration rate zero, whose adjudication then became
-- a calibration sample, or an admitted score escalated with propensity zero,
-- counted under that policy's version by every triage metric.
--
-- As in migration 9, a CHECK added to a table that may already hold rows
-- written around the Rust constructors is NOT VALID, and the trigger applies
-- to rows written from this version on; loaders keep refusing older rows.

-- A positive calibration rate lowers the propensity the policy logs for an
-- admitted score, 1 - calibration_rate, below one: for a positive rate of at
-- most 2^-54 it rounds to one in double precision, exactly as in Rust, and
-- the policy's slice rows would be logged with propensity one, as if
-- escalating them were impossible.
ALTER TABLE {{work}}.triage_policy
    ADD CONSTRAINT triage_policy_rate_lowers_propensity
        CHECK (calibration_rate = 0 OR 1::double precision - calibration_rate < 1) NOT VALID;

-- The rules of TriagePolicy::explains for the cited policy, which
-- record_triage applies before it writes. For an eligible branch the
-- auto-propose propensity is 1 - calibration_rate when the policy's threshold
-- admits the score and zero otherwise, compared exactly (both sides compute
-- it in IEEE double precision, the score and threshold compare as the same
-- single-precision values Rust compares, and every column keeps its value
-- bit for bit); a calibration-slice branch needs a positive rate; outside the
-- slice a branch is auto-proposed exactly when the threshold admits its
-- score. A branch verification decided keeps the table's own CHECKs, which
-- are the same for every policy.
--
-- A recorded policy is never rewritten, so a row checked when it is written
-- stays explained. The trigger runs after the row: after the table's CHECKs
-- and after the policy's foreign key (whose internal trigger sorts first),
-- so a row those refuse keeps their refusal, and a cited policy the check
-- cannot see is refused rather than skipped.
CREATE FUNCTION {{work}}.check_branch_triage_policy() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    policy_threshold real;
    rate double precision;
    admitted boolean;
    propensity double precision;
BEGIN
    SELECT threshold, calibration_rate INTO policy_threshold, rate
        FROM {{work}}.triage_policy WHERE version = NEW.policy_version;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'the triage of branch % cites policy %, which is not recorded',
            NEW.branch, NEW.policy_version
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF NOT NEW.eligible THEN
        RETURN NULL;
    END IF;
    -- A NULL threshold auto-proposes nothing.
    admitted := policy_threshold IS NOT NULL AND NEW.score >= policy_threshold;
    propensity := CASE WHEN admitted THEN 1::double precision - rate ELSE 0 END;
    IF NEW.auto_propensity <> propensity THEN
        RAISE EXCEPTION 'branch % logs propensity %, which policy % does not log for score %',
            NEW.branch, NEW.auto_propensity, NEW.policy_version, NEW.score
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF NEW.calibration_slice AND rate = 0 THEN
        RAISE EXCEPTION 'branch % is in the calibration slice of policy %, whose rate is zero',
            NEW.branch, NEW.policy_version
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF NOT NEW.calibration_slice AND (NEW.decision = 'auto_propose') <> admitted THEN
        RAISE EXCEPTION 'branch % is triaged % outside the slice, against policy %''s threshold',
            NEW.branch, NEW.decision, NEW.policy_version
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER branch_triage_explained_by_policy
    AFTER INSERT ON {{work}}.branch_triage
    FOR EACH ROW EXECUTE FUNCTION {{work}}.check_branch_triage_policy();
