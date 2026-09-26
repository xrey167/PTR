use std::fmt;

use ptr_branch::BranchError;

/// Every refusal of the Postgres substrate, with no driver type in it: a
/// database error crosses this boundary as its SQLSTATE and message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PgError {
    /// A schema or embedding-space identifier is not a safe SQL identifier:
    /// not `[a-z][a-z0-9_]{0,39}`, or a keyword PostgreSQL reserves. A
    /// schema prefix is also refused when its names would begin `pg_`,
    /// which PostgreSQL reserves for system schemas.
    InvalidIdentifier { value: String },
    /// A schema set is not the three schemas `SchemaSet::with_prefix` makes of
    /// one prefix. Its names could be another instance's, which a rebuild
    /// would drop and the projector write under a lock that instance never
    /// takes, so it is refused before any session opens.
    InvalidSchemaSet {
        projection: String,
        derived: String,
        work: String,
    },
    /// The environment variable that should hold the connection string is
    /// unset or empty. The string itself is never part of a config file.
    MissingDsn { variable: String },
    /// The server is older than the substrate supports.
    UnsupportedServer { found: u32, minimum: u32 },
    /// A required extension is not installed and cannot be created.
    MissingExtension { name: &'static str },
    /// An applied migration's checksum differs from the catalog's: someone
    /// edited a migration after it ran.
    MigrationDrift { version: u32 },
    /// The database has a migration this build does not know: it was migrated
    /// by a newer build.
    UnknownMigration { version: u32 },
    /// The migration catalog itself is malformed.
    InvalidCatalog { message: &'static str },
    /// The session holding the migration lock ended (it was terminated, or
    /// its connection failed) before a schema change committed, so the
    /// change was rolled back rather than committed without the lock. The
    /// changes committed before it were made under the lock; a retry takes
    /// the lock afresh and applies what is still pending.
    MigrationLockLost,
    /// A value read back from the database is not a valid domain value.
    CorruptRow { table: &'static str, reason: String },
    /// A record does not chain from the projection's stored anchor, or a
    /// re-delivered record differs from the one applied at its index: the
    /// projection holds a history the ledger does not (a discarded tail, a
    /// restored backup, a foreign log). It must be dropped and rebuilt.
    ForeignHistory { index: u64 },
    /// A record handed to the projector cannot be chained: its index differs
    /// from the anchor it came with, or the canonical encoder refused it.
    InvalidRecord { index: u64, reason: String },
    /// The ledger's anchor is behind the projection: the projection is ahead of
    /// the history it claims to project.
    ProjectionAhead { projection: u64, ledger: u64 },
    /// A read asked for at least `fence` but the projection has applied only
    /// up to `watermark`.
    ProjectionBehind { watermark: u64, fence: u64 },
    /// A derived row was offered for a generation that is not live.
    NotLive { target: String, generation: u64 },
    /// A value does not fit the signed 64-bit column it is stored in.
    OutOfRange { field: &'static str },
    /// The connection string names a non-loopback host; this build has no TLS
    /// connector and refuses rather than sending credentials in the clear.
    TlsRequired { host: String },
    /// An embedding does not match its space's dimension.
    DimensionMismatch { expected: usize, actual: usize },
    /// An embedding value is not finite or does not fit half precision.
    InvalidEmbedding { reason: &'static str },
    /// An embedding space is already registered with a different model,
    /// revision or dimension. A space's vectors are comparable only with each
    /// other, so a space is never redefined in place.
    SpaceConflict { space: String },
    /// A triage policy record was refused: a calibration branch is not an
    /// adjudicated calibration-slice branch, or the record's rule, rerun on
    /// the stored adjudications of its calibration branches, chooses another
    /// threshold, so the policy would claim a guarantee its samples do not
    /// support.
    InvalidPolicy {
        version: String,
        reason: &'static str,
    },
    /// A fast-memory registration was refused: its configuration is outside
    /// the ranges `ptr_fastmem::check_config` supports, so no write could
    /// ever be appended to it.
    InvalidMemory {
        memory: String,
        reason: &'static str,
    },
    /// A fast-memory journal append was refused: the sequence number does not
    /// follow the journal, the journal is full, the memory does not exist, or
    /// the request fails the memory configuration and admission rules.
    InvalidWrite {
        memory: String,
        reason: &'static str,
    },
    /// A fast-memory checkpoint was refused: its binding digest does not
    /// match the stored journal prefix up to its applied write, that write is
    /// not journaled, or its shape differs from the memory's. Its state cells
    /// are not refolded.
    InvalidCheckpoint {
        memory: String,
        reason: &'static str,
    },
    /// A sealed branch was refused before any row was written because it
    /// breaks the sealing invariant `error` names
    /// (`ptr_branch::SealedBranch::from_parts` lists them). Every
    /// `SealedBranch` passed them when it was built, so `store_branch`
    /// rechecking them refuses only a defect in how one was built.
    InvalidBranch { branch: String, error: BranchError },
    /// A stored branch's rows are not a branch sealing could have produced,
    /// so they were changed after they were stored: rebuilt through
    /// `ptr_branch::SealedBranch::from_parts` they break the sealing
    /// invariant `error` names (an operation on a key reserved to ingress, a
    /// `Put` or `Remove` of a key the branch did not read, a touched key
    /// without its base value or input set, ...), or they rely on two
    /// generations of one target (`ConflictingReliance`). No branch is
    /// returned, so nothing certifies a plan from the rows.
    CorruptBranch { branch: String, error: BranchError },
    /// An interference report was refused before any row was written: it
    /// names another adapter than the one it is recorded for, it
    /// has no layer (storing it would record no evidence while claiming the
    /// adapter's one report), or more layers than the table can count.
    InvalidInterference {
        adapter: String,
        reason: &'static str,
    },
    /// A stored branch was sealed before the input set of each touched key was
    /// recorded (its `branch_touched` row has no `inputs_digest`).
    /// Certification could not tell whether a touched derived key was rewired
    /// to other inputs, so the branch cannot be certified and must be re-run
    /// on a current snapshot.
    BranchWithoutInputSets { branch: String, key: String },
    /// A string contains NUL, which a PostgreSQL `text` value cannot hold.
    InvalidText { field: &'static str },
    /// The database refused a statement.
    Database { sqlstate: String, message: String },
    /// The connection failed or closed.
    Connection { message: String },
}

impl PgError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidIdentifier { .. } => "PTR_PG_INVALID_IDENTIFIER",
            Self::InvalidSchemaSet { .. } => "PTR_PG_INVALID_SCHEMA_SET",
            Self::MissingDsn { .. } => "PTR_PG_MISSING_DSN",
            Self::UnsupportedServer { .. } => "PTR_PG_UNSUPPORTED_SERVER",
            Self::MissingExtension { .. } => "PTR_PG_MISSING_EXTENSION",
            Self::MigrationDrift { .. } => "PTR_PG_MIGRATION_DRIFT",
            Self::UnknownMigration { .. } => "PTR_PG_UNKNOWN_MIGRATION",
            Self::InvalidCatalog { .. } => "PTR_PG_INVALID_CATALOG",
            Self::MigrationLockLost => "PTR_PG_MIGRATION_LOCK_LOST",
            Self::CorruptRow { .. } => "PTR_PG_CORRUPT_ROW",
            Self::ForeignHistory { .. } => "PTR_PG_FOREIGN_HISTORY",
            Self::InvalidRecord { .. } => "PTR_PG_INVALID_RECORD",
            Self::ProjectionAhead { .. } => "PTR_PG_PROJECTION_AHEAD",
            Self::ProjectionBehind { .. } => "PTR_PG_PROJECTION_BEHIND",
            Self::NotLive { .. } => "PTR_PG_NOT_LIVE",
            Self::OutOfRange { .. } => "PTR_PG_OUT_OF_RANGE",
            Self::TlsRequired { .. } => "PTR_PG_TLS_REQUIRED",
            Self::DimensionMismatch { .. } => "PTR_PG_DIMENSION_MISMATCH",
            Self::InvalidEmbedding { .. } => "PTR_PG_INVALID_EMBEDDING",
            Self::SpaceConflict { .. } => "PTR_PG_SPACE_CONFLICT",
            Self::InvalidPolicy { .. } => "PTR_PG_INVALID_POLICY",
            Self::InvalidMemory { .. } => "PTR_PG_INVALID_MEMORY",
            Self::InvalidWrite { .. } => "PTR_PG_INVALID_WRITE",
            Self::InvalidCheckpoint { .. } => "PTR_PG_INVALID_CHECKPOINT",
            Self::InvalidBranch { .. } => "PTR_PG_INVALID_BRANCH",
            Self::CorruptBranch { .. } => "PTR_PG_CORRUPT_BRANCH",
            Self::InvalidInterference { .. } => "PTR_PG_INVALID_INTERFERENCE",
            Self::BranchWithoutInputSets { .. } => "PTR_PG_BRANCH_WITHOUT_INPUT_SETS",
            Self::InvalidText { .. } => "PTR_PG_INVALID_TEXT",
            Self::Database { .. } => "PTR_PG_DATABASE",
            Self::Connection { .. } => "PTR_PG_CONNECTION",
        }
    }
}

impl fmt::Display for PgError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { value } => {
                write!(formatter, "{value:?} is not a valid identifier")
            }
            Self::InvalidSchemaSet {
                projection,
                derived,
                work,
            } => write!(
                formatter,
                "schemas {projection:?}, {derived:?} and {work:?} are not the projection, \
                 derived and work schemas of one prefix"
            ),
            Self::MissingDsn { variable } => {
                write!(formatter, "environment variable {variable} holds no connection string")
            }
            Self::UnsupportedServer { found, minimum } => write!(
                formatter,
                "server_version_num {found} is below the supported minimum {minimum}"
            ),
            Self::MissingExtension { name } => write!(formatter, "extension {name} is missing"),
            Self::MigrationDrift { version } => {
                write!(formatter, "applied migration {version} differs from the catalog")
            }
            Self::UnknownMigration { version } => {
                write!(formatter, "database has migration {version} this build does not know")
            }
            Self::InvalidCatalog { message } => write!(formatter, "invalid migration catalog: {message}"),
            Self::MigrationLockLost => write!(
                formatter,
                "the migration lock's session ended before a schema change committed; \
                 the change was rolled back"
            ),
            Self::CorruptRow { table, reason } => write!(formatter, "corrupt row in {table}: {reason}"),
            Self::ForeignHistory { index } => {
                write!(formatter, "record {index} is not part of the projected history")
            }
            Self::InvalidRecord { index, reason } => {
                write!(formatter, "record {index} cannot be chained: {reason}")
            }
            Self::ProjectionAhead { projection, ledger } => write!(
                formatter,
                "projection at {projection} is ahead of the ledger at {ledger}"
            ),
            Self::ProjectionBehind { watermark, fence } => write!(
                formatter,
                "projection at {watermark} has not reached the required commit {fence}"
            ),
            Self::NotLive { target, generation } => {
                write!(formatter, "{target:?} generation {generation} is not live")
            }
            Self::OutOfRange { field } => write!(formatter, "{field} exceeds the signed 64-bit range"),
            Self::TlsRequired { host } => write!(
                formatter,
                "{host:?} is not a loopback host and this build has no TLS connector"
            ),
            Self::DimensionMismatch { expected, actual } => {
                write!(formatter, "embedding has {actual} dimensions, space has {expected}")
            }
            Self::InvalidEmbedding { reason } => write!(formatter, "invalid embedding: {reason}"),
            Self::SpaceConflict { space } => write!(
                formatter,
                "embedding space {space:?} is registered with a different model, revision or dimension"
            ),
            Self::InvalidPolicy { version, reason } => {
                write!(formatter, "triage policy {version:?} refused: {reason}")
            }
            Self::InvalidMemory { memory, reason } => {
                write!(formatter, "fast-memory registration of {memory:?} refused: {reason}")
            }
            Self::InvalidWrite { memory, reason } => {
                write!(formatter, "fast-memory write to {memory:?} refused: {reason}")
            }
            Self::InvalidCheckpoint { memory, reason } => {
                write!(formatter, "fast-memory checkpoint of {memory:?} refused: {reason}")
            }
            Self::InvalidBranch { branch, error } => {
                write!(formatter, "branch {branch:?} refused: {error}")
            }
            Self::CorruptBranch { branch, error } => {
                write!(formatter, "stored branch {branch:?} is corrupt: {error}")
            }
            Self::InvalidInterference { adapter, reason } => {
                write!(formatter, "interference report for {adapter:?} refused: {reason}")
            }
            Self::BranchWithoutInputSets { branch, key } => write!(
                formatter,
                "branch {branch:?} was sealed before the input set of touched key {key:?} \
                 was recorded; it cannot be certified and must be re-run"
            ),
            Self::InvalidText { field } => write!(
                formatter,
                "{field} contains NUL, which PostgreSQL text cannot store"
            ),
            Self::Database { sqlstate, message } => write!(formatter, "[{sqlstate}] {message}"),
            Self::Connection { message } => write!(formatter, "connection: {message}"),
        }
    }
}

impl std::error::Error for PgError {}
