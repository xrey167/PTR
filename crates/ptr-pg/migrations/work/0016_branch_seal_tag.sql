-- Keyed tags on stored branches.
--
-- Work migration 11 binds a branch's rows to the transaction that writes its
-- header: nothing changes, removes or appends one row of a stored branch.
-- It does not bind them to whoever sealed the branch. A writer with the work
-- schema's privileges can delete a whole branch and write other rows under
-- its id in one transaction, as store_branch writes a branch, or write rows
-- with the triggers off, and those rows load like sealed ones whenever they
-- keep every sealing invariant.
--
-- From this version on a substrate holding a branch seal key
-- (PgSubstrate::with_branch_seal_key) tags each branch it stores: a row of
-- branch_seal holding an HMAC-SHA-256 under that key of the domain
-- "ptr-pg/branch-seal/v1" and the sealed branch's digest
-- (ptr_branch::SealedBranch::seal_digest), which covers the id, the author,
-- the base revision and every row. load_branch under the key returns a
-- branch only if one of its tags is the key's tag of exactly what its rows
-- rebuild into: it refuses a branch with no tag (PgError::BranchWithoutSealTag)
-- and one none of whose tags verifies (PgError::BranchSealMismatch). The key
-- is the host's and is never stored here, so a writer of this schema who
-- does not hold it cannot make rows it wrote load under it.
--
-- The tags are a table of their own, not a column of the header, so the
-- header stays as sealed: work migration 11 binds a branch's rows to the
-- transaction that wrote its header (its xmin), which an update of the header
-- would move. A branch stored before this version, by a substrate without a
-- key, or under another key gains a tag in any later transaction
-- (PgSubstrate::seal_stored_branch, once the host vouches for the sealed
-- branch its rows hold), without being deleted: a branch a recorded policy
-- was calibrated on cannot be (migration 13), and deleting any other takes its
-- triage and outcomes with it. A branch may carry several tags, one per key
-- that tagged it, so a key is rotated by tagging every branch under the new
-- one. A tag is never rewritten, and goes only with its branch, as the
-- branch's other rows do; a tag appended by a writer without the key
-- verifies under no key and changes nothing a key accepts.
--
-- No function is added: the tags are computed and checked by the substrate,
-- which alone holds the key, and the triggers below use the functions of
-- migrations 1, 11 and 13.
--
-- The foreign key this migration adds, and why it is as it is:
--   * branch -> branch_seal: tags appended to a stored branch in its own
--     transaction or any later one, each never rewritten and going only with
--     the branch; a tag names no row it covers, so the branch's sealed rows
--     stay bound to its header's transaction: appendable by design.
CREATE TABLE {{work}}.branch_seal (
    branch text NOT NULL REFERENCES {{work}}.branch (id) ON DELETE CASCADE,
    tag bytea NOT NULL CHECK (octet_length(tag) = 32),
    tagged_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (branch, tag)
);

CREATE TRIGGER branch_seal_append_only
    BEFORE UPDATE ON {{work}}.branch_seal
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_rewrite();
CREATE TRIGGER branch_seal_removed_with_branch
    BEFORE DELETE ON {{work}}.branch_seal
    FOR EACH ROW EXECUTE FUNCTION {{work}}.refuse_removal_from_stored_branch();
CREATE TRIGGER branch_seal_never_truncated BEFORE TRUNCATE ON {{work}}.branch_seal
    FOR EACH STATEMENT EXECUTE FUNCTION {{work}}.refuse_removal();
