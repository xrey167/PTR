# R015 preregistration

Status: text complete, not frozen. R015 stays `planned`. At the freeze the head pins the frozen sentence
encoder (`encoder_model`, `encoder_revision`) and the two `*_sha256` keys of `config.toml`'s
`[preregistration]` (this text and `criteria.toml`), writes the aggregator that reads `criteria.toml`, and only
then moves the status forward. Nothing below was chosen from a confirmatory seed.

## 1. Question and scope

Does a per-pod learned address (a vector `z` and a bias over a frozen sentence encoder, loaded only after its
activation gate passed) route held-out questions to the right registry pod more reliably than description-vector
cosine alone, because it can abstain when the best candidate is not clearly the answer?

Scope: a mechanism comparison on a synthetic world of six supplier pods, with one frozen encoder. No claim about
other encoders, real corpora, larger registries or answer quality. The learned router is advisory: it ranks
registry candidates and never changes eligibility, gates, thresholds or verifiers (ADR-0023). A negative result is
a first-class outcome.

## 2. Seeds and what was seen

- Confirmatory seeds: 2017, 2029, 2043, 2071, 2101. They have never been run in any form. Each runs once, after the
  freeze.
- Explorative seeds (burned): 17, 29, 43, 71, 101. On 2026-10-13, before the criterion was fixed, the harness was run
  once on these to see that it works. Their numbers were seen; they are excluded from every confirmatory claim and
  appear only as explorative results.
- Pilot seeds: 1001, 1002. Smoke runs and anything chosen from data use these only. The smoke encoder
  (`hashed-bow-64`) is a stand-in; a `--smoke` run prints `"evidence": "none (stand-in encoder)"`.

## 3. Entrypoint and encoder

The manifest's entrypoint, `cargo run --release --locked -p ptr-bench -- dragonfly-addressing <seed>`, generates
the seed's world itself in a scratch directory (it starts the training step as a child process) and consumes it, so a
clean checkout runs without a prior step. The training step builds the encoder that `config.toml` pins. While the
pin is a `must-be-pinned-…` placeholder, or the pinned encoder cannot be built, the run exits 2 (harness failure) and
never falls back to another encoder. Only `--smoke` selects the stand-in, and its output is marked as no evidence.

The Python environment is frozen as well. The interpreter is pinned to one exact version (`python_version`,
run with `uv run --python`; the training step exits 2 on any other, a smoke run only records it). A confirmatory
run goes through `uv run --locked --project training
--extra dragonfly`; the digest of `training/uv.lock` is pinned at the freeze (`training_lock_sha256`). The
training step exits 2 when the installed `sentence-transformers`, `torch`, `transformers` or `numpy` differ
from the lock, and every run prints the interpreter version, those package versions, the locked versions and
the lock digest (`environment`). A smoke run only records them. A result is reported with that block.

## 4. Arms

All arms see the same probes: 36 labelled held-out questions (three phrasings the trainer never saw, times both
spellings of each of six pods) and 7 probes that name no pod (4 distractors, 3 unregistered suppliers).

1. `learned`: the activated addresses, abstain below 0.500, abstain when the margin to the runner-up is below 0.150.
2. `learned_no_abstain` (ablation): the same addresses, both thresholds 0. Partial coverage still abstains.
3. `learned_no_bias` (ablation): the same addresses with every bias 0.
4. `sweep` (sensitivity): the same addresses at (abstain_below, min_margin) in permille of (400, 100), (500, 150),
   (600, 200), (500, 300), fixed here before any confirmatory run.
5. `cosine_only` (baseline): the pod whose description vector is closest to the question; always answers.
6. `constant_score` (baseline, floor): the registry's default `PodRouter`.
7. `oracle` (ceiling, validity control): the probe's label, or a decline when it names no pod.

Metrics per arm: `routing_accuracy` (labelled probes routed to their pod over labelled probes), `abstain_rate`,
`false_route_rate` (probes routed to a wrong pod, or to any pod when none is meant, over all probes), each also as an
integer permille rounded down.

## 5. Decision and stop rules

The rules are in `criteria.toml`, in a fixed grammar, and are all that judges.

- Primary (falsifiers), on every confirmatory seed: `routing_accuracy` of `learned` at least that of `cosine_only`
  (margin 0 permille), and `false_route_rate` of `learned` strictly below that of `cosine_only`.
- GO only if both hold on all five seeds. A single seed that fails either is NO-GO: the representations stay
  deactivated and the default `PodRouter` stays the only router. Seeds are not re-run, replaced or dropped.
- Validity controls (coverage): the oracle reaches the ceiling (accuracy 1000 permille, no false route), the
  constant-score router does not beat description cosine, every representation passed its activation gate and was
  learned for the project, namespace, pod and generation of the world entry that names it, every probe label names
  a registered pod, and the probe counts are those of the design. A seed missing one is INVALID: the verdict is "no result", the cause is fixed
  under a successor experiment, and the seed is not reused.
- Stop: all five seeds are run, in order, once. If a seed is INVALID the series stops there and the verdict is "no
  result"; if a seed fails the primary rules the series still completes, so the negative result is complete.
- Reported, not judged: the ablations, the sweep, the oracle and the constant-score rates. If the ablations show the
  gain comes from the bias or from abstention alone, that is reported as such. No threshold is chosen from the sweep.

## 6. Disclosures

- The experiment was registered before the encoder was chosen; the pin is a freeze prerequisite.
- The synthetic world is small by design (six pods); the criteria are absolute counts on 43 probes per seed, so a
  difference of one probe moves a permille by about 23. This is a smoke-scale mechanism test, not a power analysis.
- The learned router abstains when some registry candidate has no address (partial coverage); the validity controls
  require full activation so the comparison does not turn into that behaviour.
