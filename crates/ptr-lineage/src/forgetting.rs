use crate::error::LineageError;
use crate::lineage::AdapterId;
use crate::scale::{self, Wide};
use sha2::{Digest, Sha256};

/// `values[i][j]`: score on task `j` after training stage `i`, for `T` stages
/// and `T` tasks (task `j` is the one introduced at stage `j`).
///
/// Every summary relies on the matrix being square, non-empty and finite, so
/// the field is private and [`AccuracyMatrix::new`], which checks that, is
/// the only way to build one:
///
/// ```compile_fail
/// use ptr_lineage::AccuracyMatrix;
/// let ragged = AccuracyMatrix { values: vec![vec![0.9, 0.1], vec![0.8]] };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct AccuracyMatrix {
    values: Vec<Vec<f64>>,
}

impl AccuracyMatrix {
    /// Build a square stage-by-task score matrix. Scores need only be finite;
    /// they are not restricted to probabilities.
    ///
    /// No intermediate result of a summary overflows or underflows: every
    /// difference of scores and every sum is rounded as `f64` rounds it but
    /// with an exponent range `f64` does not bound, and every mean is that
    /// sum divided by the count and rounded once. The average accuracy is
    /// therefore always finite, backward transfer and forgetting are
    /// infinite only when their exact value exceeds `f64::MAX` (a score of
    /// `f64::MAX` forgotten down to `-f64::MAX`), never NaN, and a summary
    /// is zero only when its exact value is at most half the smallest
    /// subnormal, `5e-324`, in magnitude. A task's forgetting, a difference
    /// of two scores, is exact below the normal range: a task scored
    /// `5e-324` and finally zero is forgotten by `5e-324`, not by zero. A
    /// mean of such changes is rounded like any other: that one forgetting
    /// over three tasks is a third of `5e-324`, and the average forgetting
    /// and backward transfer round it to zero, as the `f64` quotient does.
    ///
    /// # Errors
    /// Rejects an empty matrix, rows of the wrong length, or nonfinite scores.
    pub fn new(values: Vec<Vec<f64>>) -> Result<Self, LineageError> {
        let tasks = values.len();
        if tasks == 0 {
            return Err(LineageError::Empty {
                field: "accuracy matrix",
            });
        }
        for row in &values {
            if row.len() != tasks {
                return Err(LineageError::ShapeMismatch {
                    field: "accuracy matrix row",
                    expected: tasks,
                    actual: row.len(),
                });
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(LineageError::NonFinite {
                    field: "accuracy matrix",
                });
            }
        }
        Ok(Self { values })
    }

    fn last(&self) -> &[f64] {
        self.values.last().expect("non-empty matrix")
    }

    /// Mean final score over all tasks.
    pub fn average_accuracy(&self) -> f64 {
        scale::mean(self.last())
    }

    /// Backward transfer (Lopez-Paz and Ranzato): mean change on every earlier
    /// task between just after learning it and the end. Negative is
    /// forgetting. Zero for a single task.
    pub fn backward_transfer(&self) -> f64 {
        let tasks = self.values.len();
        if tasks < 2 {
            return 0.0;
        }
        let last = self.last();
        let changes: Vec<Wide> = (0..tasks - 1)
            .map(|j| scale::difference(last[j], self.values[j][j]))
            .collect();
        scale::wide_mean(&changes)
    }

    /// Forgetting of each earlier task (Chaudhry et al.): its best score at any
    /// earlier stage minus its final score. Never negative.
    pub fn task_forgetting(&self) -> Vec<f64> {
        self.wide_forgetting()
            .into_iter()
            .map(Wide::to_f64)
            .collect()
    }

    /// Mean forgetting over earlier tasks, or zero for a single task.
    pub fn average_forgetting(&self) -> f64 {
        let forgetting = self.wide_forgetting();
        if forgetting.is_empty() {
            0.0
        } else {
            scale::wide_mean(&forgetting)
        }
    }

    /// Each earlier task's forgetting, `max(0, best - last)`, as a
    /// [`scale::difference`], which neither overflows nor underflows.
    fn wide_forgetting(&self) -> Vec<Wide> {
        let tasks = self.values.len();
        let last = self.last();
        (0..tasks.saturating_sub(1))
            .map(|j| {
                let best = (j..tasks - 1)
                    .map(|stage| self.values[stage][j])
                    .fold(f64::NEG_INFINITY, f64::max);
                let forgetting = scale::difference(best, last[j]);
                if forgetting.signum() > 0.0 {
                    forgetting
                } else {
                    Wide::ZERO
                }
            })
            .collect()
    }
}

/// Thresholds an adapter must meet before it may serve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForgettingGate {
    pub max_average_forgetting: f64,
    pub max_task_forgetting: f64,
    pub min_backward_transfer: f64,
    /// Largest allowed drop on the public regression suite, which catches the
    /// forgetting of general ability that task-local suites cannot see.
    pub max_public_regression: f64,
}

/// One threshold an adapter missed.
#[derive(Clone, Debug, PartialEq)]
pub enum GateViolation {
    AverageForgetting {
        observed: f64,
        limit: f64,
    },
    TaskForgetting {
        task: usize,
        observed: f64,
        limit: f64,
    },
    BackwardTransfer {
        observed: f64,
        limit: f64,
    },
    PublicRegression {
        observed: f64,
        limit: f64,
    },
    /// A promotion was evaluated without a serving score for the public suite. A candidate
    /// is never compared with a missing baseline as if the baseline were zero: the absence
    /// is itself the failure, so nothing without a baseline can pass.
    MissingBaseline,
}

/// The outcome of a gate. Only [`ForgettingGate::evaluate`] produces one,
/// and [`crate::Lineage::gate`] relies on that: the fields are private, so a
/// report cannot be built or changed by hand.
///
/// ```compile_fail
/// use ptr_lineage::{AdapterId, GateReport};
/// let forged = GateReport {
///     adapter: AdapterId::from("a1"),
///     violations: Vec::new(),
///     gate_config_digest: [0; 32],
/// };
/// ```
///
/// It binds the verdict to the adapter id passed to
/// [`ForgettingGate::evaluate`] and to nothing else: the gate cannot tell
/// which adapter the accuracy matrix and public-suite scores were measured
/// on, and the report names no lineage or base model, so passing scores
/// measured on the adapter it names is the caller's obligation.
#[derive(Clone, Debug, PartialEq)]
pub struct GateReport {
    adapter: AdapterId,
    violations: Vec<GateViolation>,
    /// Digest of the gate configuration that produced this verdict, stored by
    /// [`ForgettingGate::evaluate`] so a report cannot be bound to another gate.
    gate_config_digest: [u8; 32],
    /// Whether [`ForgettingGate::evaluate_promotion`] produced this report. It is not part of
    /// the canonical encoding or the digest, so a decoded report never has it: a promotion is
    /// justified only by a report evaluated here, in this process.
    promotion_evaluated: bool,
}

impl GateReport {
    /// Whether this report came from [`ForgettingGate::evaluate_promotion`], the one path that
    /// can say a baseline is missing. A report from [`ForgettingGate::evaluate`] (whose serving
    /// score cannot be absent) or one decoded from bytes is not, and cannot justify a promotion.
    pub fn promotion_evaluated(&self) -> bool {
        self.promotion_evaluated
    }

    /// The adapter named when this report was evaluated.
    pub fn adapter(&self) -> &AdapterId {
        &self.adapter
    }

    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn violations(&self) -> &[GateViolation] {
        &self.violations
    }

    /// Digest of the gate configuration that evaluated this report.
    pub fn gate_config_digest(&self) -> [u8; 32] {
        self.gate_config_digest
    }
}

/// The exact artifact and context a gate verdict was measured on, supplied by
/// the trusted evaluator that ran the measurement. All fields are digests
/// the caller computed; the gate records them and does not check them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateSubject {
    pub proposal_id: [u8; 32],
    pub operation_id: [u8; 32],
    pub artifact_digest: [u8; 32],
    pub base_model_digest: [u8; 32],
    pub eval_manifest_digest: [u8; 32],
}

impl GateSubject {
    pub fn new(
        proposal_id: [u8; 32],
        operation_id: [u8; 32],
        artifact_digest: [u8; 32],
        base_model_digest: [u8; 32],
        eval_manifest_digest: [u8; 32],
    ) -> Self {
        Self {
            proposal_id,
            operation_id,
            artifact_digest,
            base_model_digest,
            eval_manifest_digest,
        }
    }
}

const GATE_CONFIG_DOMAIN: &[u8] = b"ptr.lineage.gate-config.v1";
const BOUND_REPORT_DOMAIN: &[u8] = b"ptr.lineage.bound-gate-report.v1";

/// A [`GateReport`] bound to the proposal, operation, artifact, base model,
/// evaluation manifest and gate configuration it was produced under. Only
/// [`GateReport::bind`] produces one; the fields are private, so a bound
/// report cannot be built or changed by hand.
///
/// ```compile_fail
/// use ptr_lineage::{BoundGateReport, GateSubject};
/// let z = [0u8; 32];
/// let forged = BoundGateReport {
///     subject: GateSubject::new(z, z, z, z, z),
///     gate_config_digest: z,
/// };
/// ```
///
/// The binding is only as trustworthy as the party that supplied the
/// [`GateSubject`]: it must be the trusted evaluator, never the proposer's
/// sandbox. The digest names what was claimed, it does not prove the
/// measurements were taken on that artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundGateReport {
    report: GateReport,
    subject: GateSubject,
    gate_config_digest: [u8; 32],
}

impl BoundGateReport {
    pub fn report(&self) -> &GateReport {
        &self.report
    }

    pub fn subject(&self) -> &GateSubject {
        &self.subject
    }

    pub fn gate_config_digest(&self) -> [u8; 32] {
        self.gate_config_digest
    }

    pub fn passed(&self) -> bool {
        self.report.passed()
    }

    /// See [`GateReport::promotion_evaluated`].
    pub fn promotion_evaluated(&self) -> bool {
        self.report.promotion_evaluated()
    }

    /// SHA-256 over the domain tag, the subject, the gate configuration
    /// digest, the adapter id and the violations, each length-prefixed or
    /// fixed-width so no two reports share an encoding.
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(BOUND_REPORT_DOMAIN);
        for d in [
            &self.subject.proposal_id,
            &self.subject.operation_id,
            &self.subject.artifact_digest,
            &self.subject.base_model_digest,
            &self.subject.eval_manifest_digest,
            &self.gate_config_digest,
        ] {
            h.update(d);
        }
        let adapter = self.report.adapter.0.as_bytes();
        h.update((adapter.len() as u64).to_le_bytes());
        h.update(adapter);
        h.update((self.report.violations.len() as u64).to_le_bytes());
        for v in &self.report.violations {
            match v {
                GateViolation::AverageForgetting { observed, limit } => {
                    h.update([0u8]);
                    h.update(canonical_bits(*observed));
                    h.update(canonical_bits(*limit));
                }
                GateViolation::TaskForgetting {
                    task,
                    observed,
                    limit,
                } => {
                    h.update([1u8]);
                    h.update((*task as u64).to_le_bytes());
                    h.update(canonical_bits(*observed));
                    h.update(canonical_bits(*limit));
                }
                GateViolation::BackwardTransfer { observed, limit } => {
                    h.update([2u8]);
                    h.update(canonical_bits(*observed));
                    h.update(canonical_bits(*limit));
                }
                GateViolation::PublicRegression { observed, limit } => {
                    h.update([3u8]);
                    h.update(canonical_bits(*observed));
                    h.update(canonical_bits(*limit));
                }
                GateViolation::MissingBaseline => h.update([4u8]),
            }
        }
        h.finalize().into()
    }
}

/// Leading magic and version of a [`BoundGateReport`] encoding.
const BOUND_REPORT_MAGIC: &[u8; 8] = b"PTRBGR01";

/// The largest encoding [`BoundGateReport::decode_canonical`] reads (1 MiB, the same bound as an
/// expert commit record, which embeds one).
pub const MAX_BOUND_GATE_REPORT_BYTES: usize = 1 << 20;

fn malformed<T>(message: &'static str) -> Result<T, LineageError> {
    Err(LineageError::InvalidParameter {
        field: "bound gate report",
        message,
    })
}

struct ReportReader<'a> {
    bytes: &'a [u8],
}

impl<'a> ReportReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], LineageError> {
        if self.bytes.len() < n {
            return malformed("truncated");
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, LineageError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, LineageError> {
        let raw = self.take(4)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn u64(&mut self) -> Result<u64, LineageError> {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(raw))
    }

    fn f64(&mut self) -> Result<f64, LineageError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn array32(&mut self) -> Result<[u8; 32], LineageError> {
        let mut raw = [0u8; 32];
        raw.copy_from_slice(self.take(32)?);
        Ok(raw)
    }
}

impl BoundGateReport {
    /// The canonical bytes of this report.
    ///
    /// All integers are little-endian. `report = magic "PTRBGR01" | proposal_id[32] |
    /// operation_id[32] | artifact_digest[32] | base_model_digest[32] | eval_manifest_digest[32]
    /// | gate_config_digest[32] | adapter (u32 length, UTF-8) | violation count u32 |
    /// violations | passed u8 | digest[32]`. A violation is a tag byte (0 AverageForgetting,
    /// 1 TaskForgetting, 2 BackwardTransfer, 3 PublicRegression, 4 MissingBaseline, which has
    /// no fields), for TaskForgetting the task as
    /// `u64`, then `observed` and `limit` as the little-endian bits of an `f64`, `-0.0` as
    /// `0.0` and every NaN as one pattern (the bits [`BoundGateReport::digest`] hashes). `passed`
    /// is 1 exactly when there is no violation, and `digest` is [`BoundGateReport::digest`].
    pub fn encode_canonical(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(BOUND_REPORT_MAGIC);
        for d in [
            &self.subject.proposal_id,
            &self.subject.operation_id,
            &self.subject.artifact_digest,
            &self.subject.base_model_digest,
            &self.subject.eval_manifest_digest,
            &self.gate_config_digest,
        ] {
            out.extend_from_slice(d);
        }
        let adapter = self.report.adapter.0.as_bytes();
        out.extend_from_slice(&(adapter.len() as u32).to_le_bytes());
        out.extend_from_slice(adapter);
        out.extend_from_slice(&(self.report.violations.len() as u32).to_le_bytes());
        for v in &self.report.violations {
            match v {
                GateViolation::AverageForgetting { observed, limit } => {
                    out.push(0);
                    out.extend_from_slice(&canonical_bits(*observed));
                    out.extend_from_slice(&canonical_bits(*limit));
                }
                GateViolation::TaskForgetting {
                    task,
                    observed,
                    limit,
                } => {
                    out.push(1);
                    out.extend_from_slice(&(*task as u64).to_le_bytes());
                    out.extend_from_slice(&canonical_bits(*observed));
                    out.extend_from_slice(&canonical_bits(*limit));
                }
                GateViolation::BackwardTransfer { observed, limit } => {
                    out.push(2);
                    out.extend_from_slice(&canonical_bits(*observed));
                    out.extend_from_slice(&canonical_bits(*limit));
                }
                GateViolation::PublicRegression { observed, limit } => {
                    out.push(3);
                    out.extend_from_slice(&canonical_bits(*observed));
                    out.extend_from_slice(&canonical_bits(*limit));
                }
                GateViolation::MissingBaseline => out.push(4),
            }
        }
        out.push(u8::from(self.passed()));
        out.extend_from_slice(&self.digest());
        out
    }

    /// Decode a report and re-derive its digest.
    ///
    /// The result is the report the bytes describe, not proof that a trusted evaluator produced
    /// it: the digest is an unkeyed hash. What it gives a replaying reader is that the evidence a
    /// committed record embeds is exactly the evidence whose digest the record (and ledger event)
    /// carry.
    ///
    /// # Errors
    /// [`LineageError::InvalidParameter`] for input over [`MAX_BOUND_GATE_REPORT_BYTES`], a wrong
    /// magic, truncated or trailing bytes, an unknown violation tag, a `passed` byte that is not
    /// 1 exactly when there is no violation, a stored digest that is not the digest of the
    /// decoded fields, and any byte string that does not re-encode to itself (for instance a
    /// `-0.0` or a non-canonical NaN).
    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, LineageError> {
        if bytes.len() > MAX_BOUND_GATE_REPORT_BYTES {
            return malformed("exceeds the size limit");
        }
        let mut r = ReportReader { bytes };
        if r.take(BOUND_REPORT_MAGIC.len())? != BOUND_REPORT_MAGIC {
            return malformed("unknown magic");
        }
        let subject = GateSubject::new(
            r.array32()?,
            r.array32()?,
            r.array32()?,
            r.array32()?,
            r.array32()?,
        );
        let gate_config_digest = r.array32()?;
        let adapter_len = r.u32()? as usize;
        let adapter = match std::str::from_utf8(r.take(adapter_len)?) {
            Ok(adapter) => adapter.to_owned(),
            Err(_) => return malformed("adapter is not UTF-8"),
        };
        let count = r.u32()? as usize;
        // Every violation takes at least one byte (a missing baseline is its tag alone), so a
        // count the input cannot hold is refused before anything is allocated for it.
        if count > r.bytes.len() {
            return malformed("violation count exceeds the input");
        }
        let mut violations = Vec::with_capacity(count);
        for _ in 0..count {
            violations.push(match r.u8()? {
                0 => GateViolation::AverageForgetting {
                    observed: r.f64()?,
                    limit: r.f64()?,
                },
                1 => {
                    let Ok(task) = usize::try_from(r.u64()?) else {
                        return malformed("task index does not fit");
                    };
                    GateViolation::TaskForgetting {
                        task,
                        observed: r.f64()?,
                        limit: r.f64()?,
                    }
                }
                2 => GateViolation::BackwardTransfer {
                    observed: r.f64()?,
                    limit: r.f64()?,
                },
                3 => GateViolation::PublicRegression {
                    observed: r.f64()?,
                    limit: r.f64()?,
                },
                4 => GateViolation::MissingBaseline,
                _ => return malformed("unknown violation tag"),
            });
        }
        let passed = r.u8()?;
        let stored_digest = r.array32()?;
        if !r.bytes.is_empty() {
            return malformed("trailing bytes");
        }
        let report = GateReport {
            adapter: AdapterId(adapter),
            violations,
            gate_config_digest,
            promotion_evaluated: false,
        }
        .bind(subject);
        if passed != u8::from(report.passed()) {
            return malformed("passed flag disagrees with the violations");
        }
        if report.digest() != stored_digest {
            return malformed("stored digest is not the digest of the fields");
        }
        if report.encode_canonical() != bytes {
            return malformed("non-canonical encoding");
        }
        Ok(report)
    }
}

impl GateReport {
    /// Bind this verdict to the subject it was measured on. The gate
    /// configuration digest is the one [`ForgettingGate::evaluate`] stored in
    /// the report; the caller cannot supply another.
    pub fn bind(self, subject: GateSubject) -> BoundGateReport {
        BoundGateReport {
            gate_config_digest: self.gate_config_digest,
            report: self,
            subject,
        }
    }
}

/// The bits a threshold or measurement is hashed by: `-0.0` hashes as `0.0` and
/// every NaN as one pattern, so equal values never digest differently.
fn canonical_bits(value: f64) -> [u8; 8] {
    let value = if value == 0.0 {
        0.0
    } else if value.is_nan() {
        f64::NAN
    } else {
        value
    };
    value.to_bits().to_le_bytes()
}

/// Public-suite scores of the serving model and of the candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublicSuite {
    pub serving: f64,
    pub candidate: f64,
}

/// Public-suite scores for a promotion, where the serving score may be absent.
///
/// [`PublicSuite`] always holds a serving score, so a caller that has none must invent one,
/// and an invented zero turns "no baseline" into "any candidate is at least as good". This
/// type lets the absence be said: [`ForgettingGate::evaluate_promotion`] reports
/// [`GateViolation::MissingBaseline`] for it and passes nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionSuite {
    /// Score of the model that serves now (the base model for a first adapter) on the
    /// public suite, or `None` when it was not measured.
    pub serving: Option<f64>,
    pub candidate: f64,
}

impl ForgettingGate {
    /// Canonical digest of the four thresholds: a domain tag and each
    /// threshold as the little-endian bits of its `f64` in field order, with
    /// `-0.0` hashed as `0.0`.
    ///
    /// # Errors
    /// Returns `LineageError::InvalidParameter` for a nonfinite threshold,
    /// the same refusal as [`ForgettingGate::evaluate`].
    pub fn config_digest(&self) -> Result<[u8; 32], LineageError> {
        self.check()?;
        let mut h = Sha256::new();
        h.update(GATE_CONFIG_DOMAIN);
        for t in [
            self.max_average_forgetting,
            self.max_task_forgetting,
            self.min_backward_transfer,
            self.max_public_regression,
        ] {
            h.update(canonical_bits(t));
        }
        Ok(h.finalize().into())
    }

    fn check(&self) -> Result<(), LineageError> {
        let thresholds = [
            ("max_average_forgetting", self.max_average_forgetting),
            ("max_task_forgetting", self.max_task_forgetting),
            ("min_backward_transfer", self.min_backward_transfer),
            ("max_public_regression", self.max_public_regression),
        ];
        for (field, value) in thresholds {
            if !value.is_finite() {
                return Err(LineageError::InvalidParameter {
                    field,
                    message: "must be finite",
                });
            }
        }
        Ok(())
    }

    /// Report every exceeded forgetting or regression limit and every missed
    /// backward-transfer minimum. Equality passes each threshold. A drop in
    /// public score that overflows to positive infinity exceeds every limit
    /// and is reported as a public-regression violation.
    ///
    /// # Errors
    /// Returns `LineageError::InvalidParameter` for a nonfinite threshold and
    /// `LineageError::NonFinite` for a nonfinite public-suite score, before
    /// anything is evaluated: every comparison with NaN is false, so a NaN
    /// limit would otherwise report no violation and pass, and an infinite
    /// score would pass as an unbounded improvement.
    pub fn evaluate(
        &self,
        adapter: &AdapterId,
        matrix: &AccuracyMatrix,
        public: PublicSuite,
    ) -> Result<GateReport, LineageError> {
        self.evaluate_scores(
            adapter,
            matrix,
            Some(public.serving),
            public.candidate,
            false,
        )
    }

    /// Like [`ForgettingGate::evaluate`] for a promotion, whose serving baseline may be
    /// missing: an absent serving score is the violation
    /// [`GateViolation::MissingBaseline`] (in place of a public-regression comparison), so
    /// the report cannot pass.
    ///
    /// # Errors
    /// The refusals of [`ForgettingGate::evaluate`], for whichever scores are present.
    pub fn evaluate_promotion(
        &self,
        adapter: &AdapterId,
        matrix: &AccuracyMatrix,
        suite: PromotionSuite,
    ) -> Result<GateReport, LineageError> {
        self.evaluate_scores(adapter, matrix, suite.serving, suite.candidate, true)
    }

    fn evaluate_scores(
        &self,
        adapter: &AdapterId,
        matrix: &AccuracyMatrix,
        serving: Option<f64>,
        candidate: f64,
        promotion_evaluated: bool,
    ) -> Result<GateReport, LineageError> {
        self.check()?;
        let gate_config_digest = self.config_digest()?;
        if !(serving.is_none_or(f64::is_finite) && candidate.is_finite()) {
            return Err(LineageError::NonFinite {
                field: "public suite score",
            });
        }
        let mut violations = Vec::new();
        let average = matrix.average_forgetting();
        if average > self.max_average_forgetting {
            violations.push(GateViolation::AverageForgetting {
                observed: average,
                limit: self.max_average_forgetting,
            });
        }
        for (task, observed) in matrix.task_forgetting().into_iter().enumerate() {
            if observed > self.max_task_forgetting {
                violations.push(GateViolation::TaskForgetting {
                    task,
                    observed,
                    limit: self.max_task_forgetting,
                });
            }
        }
        let transfer = matrix.backward_transfer();
        if transfer < self.min_backward_transfer {
            violations.push(GateViolation::BackwardTransfer {
                observed: transfer,
                limit: self.min_backward_transfer,
            });
        }
        match serving {
            None => violations.push(GateViolation::MissingBaseline),
            Some(serving) => {
                let regression = serving - candidate;
                if regression.is_nan() || regression > self.max_public_regression {
                    violations.push(GateViolation::PublicRegression {
                        observed: regression,
                        limit: self.max_public_regression,
                    });
                }
            }
        }
        Ok(GateReport {
            adapter: adapter.clone(),
            violations,
            gate_config_digest,
            promotion_evaluated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix() -> AccuracyMatrix {
        AccuracyMatrix::new(vec![
            vec![0.9, 0.1, 0.1],
            vec![0.8, 0.85, 0.1],
            vec![0.7, 0.8, 0.9],
        ])
        .unwrap()
    }

    #[test]
    fn backward_transfer_and_forgetting_match_their_definitions() {
        let m = matrix();
        // BWT = ((0.7 - 0.9) + (0.8 - 0.85)) / 2 = -0.125
        assert!((m.backward_transfer() + 0.125).abs() < 1e-12);
        // f_0 = max(0.9, 0.8) - 0.7 = 0.2; f_1 = 0.85 - 0.8 = 0.05
        let forgetting = m.task_forgetting();
        assert!((forgetting[0] - 0.2).abs() < 1e-12);
        assert!((forgetting[1] - 0.05).abs() < 1e-12);
        assert!((m.average_accuracy() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn a_gate_reports_every_violated_threshold() {
        let gate = ForgettingGate {
            max_average_forgetting: 0.1,
            max_task_forgetting: 0.15,
            min_backward_transfer: -0.1,
            max_public_regression: 0.01,
        };
        let report = gate
            .evaluate(
                &AdapterId::from("candidate"),
                &matrix(),
                PublicSuite {
                    serving: 0.5,
                    candidate: 0.45,
                },
            )
            .unwrap();
        assert!(!report.passed());
        assert_eq!(report.violations().len(), 4);
    }

    fn lenient_gate() -> ForgettingGate {
        ForgettingGate {
            max_average_forgetting: 1.0,
            max_task_forgetting: 1.0,
            min_backward_transfer: -1.0,
            max_public_regression: 1.0,
        }
    }

    #[test]
    fn a_nonfinite_public_score_is_refused_rather_than_passed() {
        let scores = [
            (0.5, f64::NAN),
            (f64::NAN, 0.5),
            (0.5, f64::INFINITY),
            (f64::NEG_INFINITY, 0.5),
            (f64::INFINITY, f64::INFINITY),
        ];
        for (serving, candidate) in scores {
            assert_eq!(
                lenient_gate().evaluate(
                    &AdapterId::from("candidate"),
                    &matrix(),
                    PublicSuite { serving, candidate },
                ),
                Err(LineageError::NonFinite {
                    field: "public suite score"
                }),
                "serving {serving} candidate {candidate}"
            );
        }
    }

    #[test]
    fn a_nonfinite_threshold_is_refused_before_evaluation() {
        let public = PublicSuite {
            serving: 0.9,
            candidate: 0.0,
        };
        for limit in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let gates = [
                (
                    "max_average_forgetting",
                    ForgettingGate {
                        max_average_forgetting: limit,
                        ..lenient_gate()
                    },
                ),
                (
                    "max_task_forgetting",
                    ForgettingGate {
                        max_task_forgetting: limit,
                        ..lenient_gate()
                    },
                ),
                (
                    "min_backward_transfer",
                    ForgettingGate {
                        min_backward_transfer: limit,
                        ..lenient_gate()
                    },
                ),
                (
                    "max_public_regression",
                    ForgettingGate {
                        max_public_regression: limit,
                        ..lenient_gate()
                    },
                ),
            ];
            for (field, gate) in gates {
                assert_eq!(
                    gate.evaluate(&AdapterId::from("candidate"), &matrix(), public),
                    Err(LineageError::InvalidParameter {
                        field,
                        message: "must be finite",
                    }),
                    "{field} = {limit}"
                );
            }
        }
        // All-NaN limits compare false everywhere; they must not yield a
        // passing report for a candidate that regressed badly.
        let nan_gate = ForgettingGate {
            max_average_forgetting: f64::NAN,
            max_task_forgetting: f64::NAN,
            min_backward_transfer: f64::NAN,
            max_public_regression: f64::NAN,
        };
        assert!(nan_gate
            .evaluate(&AdapterId::from("candidate"), &matrix(), public)
            .is_err());
    }

    #[test]
    fn a_ragged_matrix_is_refused() {
        assert!(AccuracyMatrix::new(vec![vec![0.9, 0.1], vec![0.8]]).is_err());
    }

    fn gate() -> ForgettingGate {
        ForgettingGate {
            max_average_forgetting: 0.5,
            max_task_forgetting: 0.5,
            min_backward_transfer: -0.5,
            max_public_regression: 0.02,
        }
    }

    fn subject() -> GateSubject {
        GateSubject::new([1; 32], [2; 32], [3; 32], [4; 32], [5; 32])
    }

    fn bound(gate: &ForgettingGate, subject: GateSubject, public: PublicSuite) -> BoundGateReport {
        gate.evaluate(&AdapterId::from("a1"), &matrix(), public)
            .unwrap()
            .bind(subject)
    }

    fn passing() -> PublicSuite {
        PublicSuite {
            serving: 0.8,
            candidate: 0.8,
        }
    }

    fn hex(d: [u8; 32]) -> String {
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn bound_report_digest_is_deterministic_and_pinned() {
        let a = bound(&gate(), subject(), passing());
        let b = bound(&gate(), subject(), passing());
        assert_eq!(a.digest(), b.digest());
        assert_eq!(
            hex(a.digest()),
            "7fee8d89e60dd392982bb95e31b17603dc978138132a9e239139c5c8a913dbb7"
        );
        assert_eq!(
            hex(gate().config_digest().unwrap()),
            "838c0828f4ab702d4bd7b0a8d2067597e5b38c68786e8cb53ded2c43743d7dcd"
        );
        assert_eq!(a.subject(), &subject());
        assert_eq!(a.gate_config_digest(), gate().config_digest().unwrap());
        assert_eq!(a.passed(), a.report().passed());
    }

    #[test]
    fn bound_report_digest_changes_with_every_subject_field() {
        let base = bound(&gate(), subject(), passing()).digest();
        let s = subject();
        let variants = [
            GateSubject {
                proposal_id: [9; 32],
                ..s
            },
            GateSubject {
                operation_id: [9; 32],
                ..s
            },
            GateSubject {
                artifact_digest: [9; 32],
                ..s
            },
            GateSubject {
                base_model_digest: [9; 32],
                ..s
            },
            GateSubject {
                eval_manifest_digest: [9; 32],
                ..s
            },
        ];
        for v in variants {
            assert_ne!(bound(&gate(), v, passing()).digest(), base);
        }
    }

    #[test]
    fn bound_report_digest_changes_with_every_threshold() {
        let base = bound(&gate(), subject(), passing());
        let g = gate();
        let variants = [
            ForgettingGate {
                max_average_forgetting: 0.51,
                ..g
            },
            ForgettingGate {
                max_task_forgetting: 0.51,
                ..g
            },
            ForgettingGate {
                min_backward_transfer: -0.51,
                ..g
            },
            ForgettingGate {
                max_public_regression: 0.03,
                ..g
            },
        ];
        for v in variants {
            assert_ne!(v.config_digest().unwrap(), g.config_digest().unwrap());
            assert_ne!(bound(&v, subject(), passing()).digest(), base.digest());
        }
    }

    #[test]
    fn bound_report_digest_changes_with_the_verdict_and_adapter() {
        let pass = bound(&gate(), subject(), passing());
        let fail = bound(
            &gate(),
            subject(),
            PublicSuite {
                serving: 0.9,
                candidate: 0.5,
            },
        );
        assert!(pass.passed() && !fail.passed());
        assert_ne!(pass.digest(), fail.digest());
        let other = gate()
            .evaluate(&AdapterId::from("a2"), &matrix(), passing())
            .unwrap()
            .bind(subject());
        assert_ne!(pass.digest(), other.digest());
    }

    #[test]
    fn a_report_carries_the_digest_of_the_gate_that_evaluated_it() {
        let strict = gate();
        let lax = ForgettingGate {
            max_average_forgetting: 1e9,
            max_task_forgetting: 1e9,
            min_backward_transfer: -1e9,
            max_public_regression: 1e9,
        };
        let report = lax
            .evaluate(&AdapterId::from("a1"), &matrix(), passing())
            .unwrap();
        assert_eq!(report.gate_config_digest(), lax.config_digest().unwrap());
        assert_ne!(report.gate_config_digest(), strict.config_digest().unwrap());
        // `bind` takes no gate: the lax digest is what ends up bound.
        let bound = report.bind(subject());
        assert_eq!(bound.gate_config_digest(), lax.config_digest().unwrap());
        assert_ne!(bound.gate_config_digest(), strict.config_digest().unwrap());
    }

    #[test]
    fn nonfinite_thresholds_have_no_digest_and_negative_zero_equals_zero() {
        let g = gate();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let broken = ForgettingGate {
                max_task_forgetting: bad,
                ..g
            };
            assert!(broken.config_digest().is_err());
        }
        let zero = ForgettingGate {
            min_backward_transfer: 0.0,
            ..g
        };
        let negative_zero = ForgettingGate {
            min_backward_transfer: -0.0,
            ..g
        };
        assert_eq!(zero.config_digest(), negative_zero.config_digest());
    }
}
