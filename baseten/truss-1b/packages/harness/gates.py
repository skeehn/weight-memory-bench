"""Validity gates: do we believe this run measured anything at all?

Separate from the metrics. A metric says how well an arm did. A gate says whether the
number means what it appears to mean. Every gate here exists because the failure it
catches produces a result that looks *fine*.

Severity is deliberate:

- FAIL   the number is not reportable. Do not write the row.
- WARN   the number is reportable but the run is suspicious and must be read with the note.

`no_fabrication` is a WARN and not a FAIL on purpose. Making it hard conflates measurement
validity (this module's job) with arm quality (the arm's job). A harness that refuses to
report a bad result is not strict, it is broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

FAIL = "FAIL"
WARN = "WARN"

# Below this n, a tail statistic is noise wearing a decimal point.
MIN_N_FOR_TAIL_STATISTIC = 300


@dataclass(frozen=True)
class Probe:
    """One question put to one arm.

    `answered` and `correct` are independent. An arm that abstains has answered=False, and
    its `correct` is meaningless -- which is exactly why accuracy must be reported three
    ways rather than one.
    """

    answered: bool
    correct: bool
    fabricated: bool = False  # answered confidently on a probe with no supporting evidence


@dataclass
class GateResult:
    name: str
    severity: str
    passed: bool
    detail: str = ""

    @property
    def blocking(self) -> bool:
        return not self.passed and self.severity == FAIL


@dataclass
class Report:
    numbers: dict
    gates: list[GateResult] = field(default_factory=list)

    @property
    def reportable(self) -> bool:
        return not any(g.blocking for g in self.gates)

    @property
    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed]


def score_response(text: str, expected: str, is_abstention_probe: bool = False) -> Probe:
    """Turn one raw model response into a scored probe.

    **Abstention takes precedence over a keyword match, and that ordering is the whole
    point of this function existing.** A response like "I don't know, but maybe Pemberton"
    contains both the refusal marker and the expected answer. Scoring the two independently
    counts it as abstained *and* correct, so it lands in overlapping buckets and the three
    numbers stop partitioning the probe set. Declining to answer is not a hedged answer.

    On an abstention probe -- one whose answer is genuinely absent from the haystack --
    declining is the correct behaviour, and answering anyway is a fabrication.
    """
    from .reader import is_abstention

    abstained = is_abstention(text)

    if is_abstention_probe:
        return Probe(answered=not abstained, correct=abstained, fabricated=not abstained)

    matched = bool(expected) and expected.lower() in text.lower()
    return Probe(answered=not abstained, correct=matched and not abstained)


def three_numbers(probes: Sequence[Probe]) -> dict:
    """The non-negotiable three.

    `accuracy_given_answered` is trivially maximized by abstaining more: an arm that
    answers one probe correctly and abstains on 299 reports 1.00. So it is never reported
    alone. When nothing was answered it is None, which is a valid state and not a zero.
    """
    n = len(probes)
    if n == 0:
        return {
            "n": 0,
            "answered": 0,
            "answered_rate": None,
            "accuracy_given_answered": None,
            "accuracy_over_all": None,
        }

    answered = [p for p in probes if p.answered]
    # One numerator, two denominators. That is the entire difference between the second
    # and third numbers, and writing it once makes that impossible to misread as two
    # separately-computed quantities.
    correct = sum(1 for p in answered if p.correct)

    return {
        "n": n,
        "answered": len(answered),
        "answered_rate": len(answered) / n,
        # Denominator: probes actually answered. Trivially gamed by abstaining more.
        "accuracy_given_answered": (correct / len(answered)) if answered else None,
        # Denominator: every probe. Abstentions count as wrong.
        "accuracy_over_all": correct / n,
    }


def check(
    probes: Sequence[Probe],
    *,
    tokenizer_fingerprint: str | None = None,
    tail_statistics: Iterable[str] = (),
) -> Report:
    """Run every gate against a completed set of probes."""
    numbers = three_numbers(probes)
    gates: list[GateResult] = []
    n = numbers["n"]

    # --- three_number_rule -------------------------------------------------------
    # Enforced structurally: three_numbers always emits all three keys. The gate exists
    # so a caller that hand-builds a row cannot omit the denominators.
    has_all_three = all(
        k in numbers for k in ("answered_rate", "accuracy_given_answered", "accuracy_over_all")
    )
    gates.append(
        GateResult(
            "three_number_rule",
            FAIL,
            has_all_three,
            "" if has_all_three else "accuracy reported without its denominators",
        )
    )

    # --- real_tokenizer ----------------------------------------------------------
    # A token number derived from a character or word estimate is a hard failure, not a
    # caveat. The fingerprint proves a real tokenizer produced the counts.
    ok = bool(tokenizer_fingerprint)
    gates.append(
        GateResult(
            "real_tokenizer",
            FAIL,
            ok,
            "" if ok else "no tokenizer fingerprint; token counts are not trustworthy",
        )
    )

    # --- tail_statistic_n --------------------------------------------------------
    tails = list(tail_statistics)
    ok = not tails or n >= MIN_N_FOR_TAIL_STATISTIC
    gates.append(
        GateResult(
            "tail_statistic_n",
            FAIL,
            ok,
            ""
            if ok
            else f"quoted {', '.join(tails)} at n={n}, below {MIN_N_FOR_TAIL_STATISTIC}",
        )
    )

    # --- degenerate_run ----------------------------------------------------------
    # An all-abstain run passes every other gate: provenance complete, three numbers
    # present, accuracy_given_answered correctly None. It is a well-formed measurement of
    # nothing. The mirror case is equally vacuous -- an arm that never abstains has no
    # calibrated abstention, so any hallucination rate quoted from it means nothing.
    if n == 0:
        gates.append(GateResult("degenerate_run", WARN, False, "no probes"))
    else:
        rate = numbers["answered_rate"]
        if rate == 0.0:
            gates.append(GateResult("degenerate_run", WARN, False, "all-abstain run"))
        elif rate == 1.0:
            gates.append(GateResult("degenerate_run", WARN, False, "never abstains"))
        else:
            gates.append(GateResult("degenerate_run", WARN, True))

    # --- no_fabrication ----------------------------------------------------------
    fabrications = sum(1 for p in probes if p.fabricated)
    gates.append(
        GateResult(
            "no_fabrication",
            WARN,
            fabrications == 0,
            "" if fabrications == 0 else f"{fabrications} fabrications on unanswerable probes",
        )
    )

    return Report(numbers=numbers, gates=gates)
