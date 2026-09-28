-- A keyed tag on each stored branch.
--
-- Work migration 11 binds a branch's rows to the transaction that writes its
-- header: nothing changes, removes or appends one row of a stored branch.
-- It does not bind them to whoever sealed the branch. A writer with the work
-- schema's privileges can delete a whole branch and write other rows under
-- its id in one transaction, as store_branch writes a branch, or write rows
-- with the triggers off, and those rows load like sealed ones whenever they
-- keep every sealing invariant.
--
-- From this version on store_branch, when the substrate holds a branch seal
-- key (PgSubstrate::with_branch_seal_key), writes the header's seal_tag: an
-- HMAC-SHA-256 under that key of the domain "ptr-pg/branch-seal/v1" and the
-- sealed branch's digest (ptr_branch::SealedBranch::seal_digest), which
-- covers the id, the author, the base revision and every row. load_branch
-- under the key refuses a branch whose rows do not carry the tag of exactly
-- what they rebuild into (PgError::BranchSealMismatch) and one with no tag
-- (PgError::BranchWithoutSealTag), such as a branch stored before this
-- version or by a substrate without a key. The key is the host's and is
-- never stored here, so a writer of this schema who does not hold it cannot
-- make rows it did not seal load under it.
--
-- The column holds a whole tag or nothing. No function is added: the tag is
-- computed and checked by the substrate, which alone holds the key.
ALTER TABLE {{work}}.branch
    ADD COLUMN seal_tag bytea,
    ADD CONSTRAINT branch_seal_tag_length
        CHECK (seal_tag IS NULL OR octet_length(seal_tag) = 32);
