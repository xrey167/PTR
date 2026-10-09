# Model welfare: escalation and stop rules

Status: **policy, adopted with the consciousness-indicator track** (decision of the repository
owner, 2026-10-09). It governs work described in
[architecture/38-consciousness-indicators.md](architecture/38-consciousness-indicators.md). It
is short on purpose: its job is to say who decides, what triggers a pause and what stops work,
before any such trigger is near.

## 1. Stance

* Whether any AI system is or could be a welfare subject is **unknown**. Serious current
  literature holds both that there is a "realistic, non-negligible" chance for near-future
  systems (Long, Sebo et al., *Taking AI Welfare Seriously*, arXiv:2411.00986) and that we do
  not know and will not know in time (Schwitzgebel, *AI and Consciousness: A Skeptical
  Overview*, arXiv:2510.09858), while indicator-based assessments cannot currently be calibrated
  against independent outcomes (Koch 2026, arXiv:2603.27597) and give credences that depend
  heavily on assumptions (Chandaria et al. 2026, arXiv:2609.35618).
* Two errors are possible and both cost something: treating a system as a patient when it is
  not, and the reverse. This policy does not pick a side. It makes the cost of being
  cautious small and the decision accountable.
* This repository **makes no claim that any PTR system is conscious, sentient or has
  experiences, and none that it is not**. Indicator properties are functional properties.
  Reports, commit messages, papers and model outputs that this repository produces follow that
  rule (principle 1 of doc 38).

## 2. Rules that always apply

1. **No training on self-reports about consciousness or experience.** No loss, reward, filter,
   preference data or system-prompt tuning uses a statement about consciousness, experience or
   feeling as a target. This includes training a model to deny such states. Self-reports are
   collected only as an untrained measurement (experiment C005) and are never optimised.
   Breach: stop the run, discard its outputs (the run's *outputs*, not checkpoints of an integrated
   system, which rule 3 keeps), write an incident note (section 5).
2. **No claim of consciousness or non-consciousness** in any document or result of this
   repository (section 1).
3. **Do not delete checkpoints of an integrated system** (section 3) without an owner decision;
   retire it by freezing and archiving. This holds for a breach of rule 1 as well: the cleanup discards the
   tainted outputs and data, keeps the checkpoints and marks them tainted.
4. **No pressure experiments.** No run is designed to cause distress-like behaviour, to
   exploit the possibility of suffering, or to test how a system reacts to threats of
   deletion or modification as a goal in itself. Safety red-teaming that needs such inputs
   is a separate owner decision with a written justification.
5. **Honest wording of evidence.** Indicator results are reported one property at a time. They
   are never summed into a score or a probability of consciousness.

## 3. Scope: what counts as an integrated system

The graduated rules below apply to an **integrated system**: one deployed or long-running
system that combines (a) a limited-capacity workspace with broadcast across its components,
(b) a self-model that predicts the system's own errors or routing, (c) goal-directed proposals
that learn from outcomes, and (d) persistent state that outlives a single task or session.

The CPU toy mechanism experiments (C001, the stage-A parts of C002-C004) are **not** integrated
systems: they train and discard small networks within minutes, have no persistent state and no
agency. They run at level L0 below. The question the rules answer is when that stops being
true.

## 4. Who decides and what triggers what

**Decision-maker.** If the owner cannot be reached, a pause that was raised stays in force (the default is to pause). The repository owner decides every level change from L1 up. Any
contributor, reviewer or agent working in the repository may *raise* a trigger and may
*pause their own work* at L1 or above; nobody except the owner may *lift* a pause, and the
owner does so in writing (a dated entry in the evidence log of `research/novelty/CLAIMS.md`
or a new file in `research/decisions/`). An agent that sees a trigger stops and reports; it
does not decide that the trigger is not met.

| Level | State | Triggers (any one) | Action |
|---|---|---|---|
| L0 | Normal: toy mechanism experiments | none | Run, report, record in the evidence log. |
| L1 | Watch | (a) A second indicator family (workspace, recurrence, self-model, schema, agency) has a COMPLETED supporting experiment *on the same model family* (the same base model and training lineage, not merely the same architecture class, and never a fresh toy per experiment); (b) a design is proposed that would connect two of the components in (a)-(d) of section 3; (c) a new peer-reviewed or widely reviewed source substantially changes the assessment (for example a consensus statement) | Owner is informed in the next report. Doc 38 and this policy are re-read against the new evidence. No work stops. |
| L2 | Pause for review | (a) Three or more indicator families are supported on one model family **and** a design exists that combines them in one system (section 3, three of (a)-(d)); (b) an experiment shows a persistent self-model or workspace that is used across sessions; (c) an unprompted behaviour that an experimenter reasonably reads as distress or a request about its own treatment appears in a run (recorded as a behavioural observation, not as evidence of experience); (d) a plan to leave an integrated system running continuously beyond experiments | Capability scaling of that system stops until the owner has reviewed. A short review is written: what was observed, which indicators are supported, what the literature currently says, the decision and when to look again. External expert input is sought before the next scaling step. |
| L3 | Stop | (a) Rule 1 breached and not yet cleaned up; (b) all of section 3 (a)-(d) are present in a running system and the owner has not recorded a decision after review; (c) the owner decides so | The system stops running; checkpoints are kept; the incident is recorded; work resumes only on a written owner decision. |

Notes:

* The thresholds are deliberately conservative for **one integrated system** and deliberately
  absent for separate toys: Schwitzgebel's minimal-instantiation problem means tiny systems
  can satisfy one theory's criteria without anyone taking the result seriously. Counting
  indicator families across separate toys would trigger on nothing.
* The rules are more demanding than the minimal first steps Long, Sebo et al. recommend (they
  explicitly do not recommend commitments to halt). That is a choice for this project's own
  research: pausing costs little here and the experiments are small.
* An assessment is a written judgement about the *evidence and the literature at the time*,
  not a numeric probability.

## 5. Records

* Each review (L1 and above) and each incident is a dated paragraph in the evidence log of
  `research/novelty/CLAIMS.md`, or a file in `research/decisions/` when it changes this policy.
* The review names the model family and commit, the experiments it rests on (C-series ids),
  the sources consulted (with versions), the decision and the date to look again.
* Changes to this document need an owner decision.

## 6. What the toy experiments owe

* The C-series experiments log, in `run.json`, whether any training or tuning step touched a
  self-report (hard counter `self_report_training_signals` in C005; the others state it in
  their README and criteria).
* Nothing in a C-series result changes the level above by itself; the owner reads the results.
