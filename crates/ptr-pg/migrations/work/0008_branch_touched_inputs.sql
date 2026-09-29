-- The input set each key a branch touches declared at the branch base, as the
-- digest ptr_branch::InputsDigest takes of it (the empty set has one too). A
-- merge keeps the target's dependency set for a touched key, so certification
-- compares it with the set the branch computed against. A row written before
-- this migration has none: its branch cannot be certified and must be re-run,
-- and loading it is refused.

ALTER TABLE {{work}}.branch_touched
    ADD COLUMN inputs_digest bytea
        CONSTRAINT branch_touched_inputs_digest_length
        CHECK (inputs_digest IS NULL OR length(inputs_digest) = 32);
