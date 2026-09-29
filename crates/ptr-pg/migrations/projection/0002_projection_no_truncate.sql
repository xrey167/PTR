-- TRUNCATE fires no row trigger, so the append-only tables refuse it as
-- well: emptying the revocation set would make revoked generations
-- admissible again (is_admissible, source_admission, search, and the journal
-- and cache writers that check it) until a rebuild, and emptying the applied
-- anchors or the event log would lose what re-delivery and consumers are
-- checked against. A rebuild drops the whole schema instead.

CREATE TRIGGER tombstone_never_truncated BEFORE TRUNCATE ON {{projection}}.tombstone
    FOR EACH STATEMENT EXECUTE FUNCTION {{projection}}.refuse_rewrite();
CREATE TRIGGER applied_commit_never_truncated BEFORE TRUNCATE ON {{projection}}.applied_commit
    FOR EACH STATEMENT EXECUTE FUNCTION {{projection}}.refuse_rewrite();
CREATE TRIGGER projection_event_never_truncated
    BEFORE TRUNCATE ON {{projection}}.projection_event
    FOR EACH STATEMENT EXECUTE FUNCTION {{projection}}.refuse_rewrite();
