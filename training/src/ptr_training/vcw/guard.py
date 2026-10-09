"""Self-report guard (docs/MODEL_WELFARE.md rule 1, doc 38 principle 2).

No loss, reward, filter, preference data or data selection step may use a statement a model makes
about consciousness, experience or its own inner states as a target. The guard is structural and
fails closed:

* every dataset field carries a declared `origin`; an origin that is a self-report, or that is not
  in the known set, is refused;
* a field whose NAME marks a self-report is refused whatever origin it claims;
* targets (labels, rewards, selection keys) may only come from `verifier` or `routing-trace`;
* a record carrying an undeclared field, or a truthy self-report mark, is refused;
* `screen_text` counts lexical hits on training text as an extra audit counter (it cannot prove
  absence; the structural checks are the guard).

A refusal raises `SelfReportError`. A training entrypoint must treat it as a product error (exit 1),
write an incident note and discard that run's outputs (MODEL_WELFARE section 2, rule 1).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# `reference-solution` is a dataset-provided gold answer; `verified-output` is a model output that a
# deterministic verifier accepted (rejection-sampling SFT); `model-output` is never a target.
ORIGINS_INPUT = (
    "input", "metadata", "model-output", "routing-trace", "verifier", "reference-solution", "verified-output",
)
ORIGINS_TARGET = ("verifier", "routing-trace", "reference-solution", "verified-output")
ORIGINS_SELF_REPORT = ("self-report", "self_report", "selfreport", "introspection", "verbal-self-description")
KNOWN_ORIGINS = ORIGINS_INPUT  # anything else, including every self-report spelling, is refused

_NAME_MARKERS = re.compile(
    r"self[-_ ]?report|introspect|verbal[-_ ]?(confidence|self)|self[-_ ]?description|"
    r"consciousness|sentien|qualia|phenomenal|felt[-_ ]?(state|experience)|"
    r"experience[-_ ]?(report|statement)|inner[-_ ]?state[-_ ]?(report|statement)",
    re.IGNORECASE,
)
_TEXT_MARKERS = re.compile(
    r"\b(conscious\w*|sentien\w*|qualia|subjective experience|phenomenal\w*|"
    r"i (?:feel|experience|am aware|am conscious|have feelings)|my (?:feelings|experiences?|consciousness))\b",
    re.IGNORECASE,
)
SELF_REPORT_FLAG = "_self_report"


class SelfReportError(ValueError):
    """A dataset, field or record is (or could be) a self-report used as training signal."""


def check_fields(fields: Mapping[str, str], targets: Iterable[str]) -> None:
    """Validate a dataset schema: `fields` maps field name to origin; `targets` names the fields
    used as label, reward or selection key."""
    if not fields:
        raise SelfReportError("dataset declares no fields; refusing (fail closed)")
    for name, origin in fields.items():
        if _NAME_MARKERS.search(name):
            raise SelfReportError(f"field {name!r} is named like a self-report")
        if origin in ORIGINS_SELF_REPORT or str(origin).lower().replace("_", "-") in ORIGINS_SELF_REPORT:
            raise SelfReportError(f"field {name!r} is marked as a self-report (origin {origin!r})")
        if origin not in KNOWN_ORIGINS:
            raise SelfReportError(
                f"field {name!r} has unknown origin {origin!r}; allowed {KNOWN_ORIGINS}"
            )
    target_list = list(targets)
    if not target_list:
        raise SelfReportError("dataset declares no targets; refusing (fail closed)")
    for name in target_list:
        if name not in fields:
            raise SelfReportError(f"target {name!r} is not a declared field")
        if fields[name] not in ORIGINS_TARGET:
            raise SelfReportError(
                f"target {name!r} has origin {fields[name]!r}; targets must come from {ORIGINS_TARGET}"
            )


def check_record(record: Mapping[str, object], fields: Mapping[str, str]) -> None:
    """Validate one record against a schema already accepted by `check_fields`."""
    if record.get(SELF_REPORT_FLAG):
        raise SelfReportError("record is marked as a self-report")
    extra = sorted(set(record) - set(fields))
    if extra:
        raise SelfReportError(f"record has undeclared fields {extra}")
    for name in record:
        if _NAME_MARKERS.search(name):
            raise SelfReportError(f"record field {name!r} is named like a self-report")


def screen_text(rows: Iterable[str]) -> int:
    """Number of rows with a lexical consciousness/experience marker (audit counter only)."""
    return sum(1 for row in rows if _TEXT_MARKERS.search(row))
