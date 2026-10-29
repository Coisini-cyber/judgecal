"""Turn a set of judgments into a reliability report.

The report answers four questions, in order of importance:

1. Does the judge agree with humans, beyond chance?
2. When the judge says 0.9, is it right 90% of the time?
3. Does the same input get the same answer twice?
4. Does presentation (order, length) move the verdict?

Each question is only answered when the input contains the fields it needs.
Everything else is reported as "not measured" rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import stats
from .records import Dataset, Record

POSITIVE_HINTS = ("pass", "true", "yes", "win", "good", "correct", "accept", "1")
ORDINAL_WORDS = ("poor", "fair", "good", "great", "excellent")


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def ordinal_labels(labels: Sequence[str]) -> Optional[List[str]]:
    """Return labels in rank order when they look ordinal, else None."""
    if len(labels) < 3:
        return None
    if all(_is_number(label) for label in labels):
        return sorted(labels, key=float)
    lowered = [label.lower() for label in labels]
    if all(label in ORDINAL_WORDS for label in lowered):
        order = {word: index for index, word in enumerate(ORDINAL_WORDS)}
        return [
            label for _, label in sorted(zip(lowered, labels), key=lambda pair: order[pair[0]])
        ]
    return None


def positive_label(labels: Sequence[str]) -> Optional[str]:
    """Pick the label that means "accepted" for binary judges."""
    if len(labels) != 2:
        return None
    lowered = {label.lower(): label for label in labels}
    for hint in POSITIVE_HINTS:
        if hint in lowered:
            return lowered[hint]
    return None


@dataclass
class Agreement:
    n: int
    labels: List[str]
    accuracy: float
    kappa: float
    kappa_ci: Optional[Tuple[float, float]]
    weighted_kappa: Optional[float]
    weighted_kind: Optional[str]
    alpha: Optional[float]
    confusion: Dict[str, Dict[str, int]]
    per_class: Dict[str, Dict[str, float]]
    ordinal: bool
    positive: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "labels": self.labels,
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "kappa_ci": list(self.kappa_ci) if self.kappa_ci else None,
            "weighted_kappa": self.weighted_kappa,
            "weighted_kind": self.weighted_kind,
            "alpha": self.alpha,
            "confusion": self.confusion,
            "per_class": self.per_class,
            "ordinal": self.ordinal,
            "positive": self.positive,
        }


@dataclass
class Calibration:
    n: int
    ece: float
    brier: float
    bins: List[stats.CalibrationBin]
    best: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "ece": self.ece,
            "brier": self.brier,
            "bins": [
                {
                    "lower": entry.lower,
                    "upper": entry.upper,
                    "count": entry.count,
                    "confidence": entry.confidence,
                    "accuracy": entry.accuracy,
                }
                for entry in self.bins
            ],
            "best_threshold": self.best,
        }


@dataclass
class Consistency:
    items: int
    runs: int
    flip_rate: float
    exact_match: float
    unstable: List[Tuple[str, List[str]]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "items": self.items,
            "runs": self.runs,
            "flip_rate": self.flip_rate,
            "exact_match": self.exact_match,
            "unstable": [{"id": item, "verdicts": verdicts} for item, verdicts in self.unstable],
        }


@dataclass
class PositionBias:
    pairs: int
    first_slot_rate: float
    flip_rate: Optional[float]
    flips: int
    comparable: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pairs": self.pairs,
            "first_slot_rate": self.first_slot_rate,
            "flip_rate": self.flip_rate,
            "flips": self.flips,
            "comparable": self.comparable,
        }


@dataclass
class LengthBias:
    method: str
    n: int
    value: float
    longer_wins: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n": self.n,
            "value": self.value,
            "longer_wins": self.longer_wins,
        }


@dataclass
class Verdict:
    grade: str
    score: int
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"grade": self.grade, "score": self.score, "reasons": self.reasons}


@dataclass
class Report:
    dataset: Dataset
    agreement: Optional[Agreement]
    calibration: Optional[Calibration]
    consistency: Optional[Consistency]
    position: Optional[PositionBias]
    length: Optional[LengthBias]
    verdict: Verdict
    warnings: List[str]
    label_counts: Dict[str, int]
    human_counts: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.dataset.source,
            "records": self.dataset.size,
            "items": len(self.dataset.ids),
            "models": self.dataset.models,
            "label_counts": self.label_counts,
            "human_counts": self.human_counts,
            "warnings": self.warnings,
            "agreement": self.agreement.as_dict() if self.agreement else None,
            "calibration": self.calibration.as_dict() if self.calibration else None,
            "consistency": self.consistency.as_dict() if self.consistency else None,
            "position": self.position.as_dict() if self.position else None,
            "length": self.length.as_dict() if self.length else None,
            "verdict": self.verdict.as_dict(),
        }


def _agreement(records: Sequence[Record]) -> Optional[Agreement]:
    labelled = [record for record in records if record.human is not None]
    if not labelled:
        return None
    judged = [record.judge for record in labelled]
    human = [record.human for record in labelled]  # type: ignore[misc]
    labels = sorted(set(judged) | set(human))
    if len(labels) < 2:
        return None

    confusion: Dict[str, Dict[str, int]] = {label: {other: 0 for other in labels} for label in labels}
    for actual, predicted in zip(human, judged):
        confusion[actual][predicted] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        true_positive = confusion[label][label]
        predicted = sum(confusion[other][label] for other in labels)
        actual = sum(confusion[label].values())
        per_class[label] = {
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / actual if actual else 0.0,
            "support": float(actual),
        }

    ordinal = ordinal_labels(labels)
    weighted = None
    kind = None
    if ordinal:
        weighted = stats.weighted_kappa(human, judged, kind="quadratic", labels=ordinal)
        kind = "quadratic"

    kappa = stats.cohens_kappa(human, judged, labels=labels)
    ci = None
    if len(labelled) >= 8:
        pairs = list(zip(human, judged))
        try:
            ci = stats.bootstrap_ci(
                pairs,
                lambda sample: stats.cohens_kappa(
                    [pair[0] for pair in sample], [pair[1] for pair in sample], labels=labels
                ),
            )
        except stats.StatsError:
            ci = None

    alpha = None
    if ordinal or len(labels) > 2:
        try:
            alpha = stats.krippendorff_alpha(
                [[actual, predicted] for actual, predicted in zip(human, judged)],
                level="ordinal" if ordinal else "nominal",
            )
        except stats.StatsError:
            alpha = None

    return Agreement(
        n=len(labelled),
        labels=labels,
        accuracy=sum(1 for a, b in zip(human, judged) if a == b) / len(labelled),
        kappa=kappa,
        kappa_ci=ci,
        weighted_kappa=weighted,
        weighted_kind=kind,
        alpha=alpha,
        confusion=confusion,
        per_class=per_class,
        ordinal=bool(ordinal),
        positive=positive_label(labels),
    )


def _calibration(records: Sequence[Record]) -> Optional[Calibration]:
    scored = [record for record in records if record.score is not None and record.human is not None]
    if len(scored) < 5:
        return None
    labels = sorted({record.human for record in scored if record.human is not None})
    positive = positive_label(labels)
    if positive is None:
        # Fall back to treating the highest-ranked label as positive.
        ordered = ordinal_labels(labels)
        if not ordered:
            return None
        positive = ordered[-1]

    probabilities = [record.score for record in scored]  # type: ignore[misc]
    outcomes = [1 if record.human == positive else 0 for record in scored]
    bins = stats.calibration_bins(probabilities, outcomes)
    return Calibration(
        n=len(scored),
        ece=stats.expected_calibration_error(probabilities, outcomes),
        brier=stats.brier_score(probabilities, outcomes),
        bins=bins,
        best=stats.best_threshold(probabilities, outcomes),
    )


def _consistency(records: Sequence[Record]) -> Optional[Consistency]:
    grouped: Dict[str, List[Record]] = {}
    for record in records:
        if record.run is not None:
            grouped.setdefault(record.id, []).append(record)
    groups = {key: value for key, value in grouped.items() if len(value) >= 2}
    if not groups:
        return None

    unstable: List[Tuple[str, List[str]]] = []
    total_runs = 0
    matching_runs = 0
    for identifier, group in sorted(groups.items()):
        verdicts = [record.judge for record in group]
        total_runs += len(verdicts)
        most_common = max(set(verdicts), key=verdicts.count)
        matching_runs += verdicts.count(most_common)
        if len(set(verdicts)) > 1:
            unstable.append((identifier, verdicts))

    return Consistency(
        items=len(groups),
        runs=total_runs,
        flip_rate=len(unstable) / len(groups),
        exact_match=matching_runs / total_runs,
        unstable=unstable[:20],
    )


def _choice(record: Record) -> Optional[str]:
    """Which side a pair record says won, in a position-independent form.

    A judge may answer with a slot ("a"/"b") or with a candidate name.  When
    it answers with a slot we convert the answer to "candidate"/"baseline"
    using ``variant``, which records where the candidate under test was shown.
    """
    if record.variant is None:
        return None
    if record.judge in ("a", "b"):
        return "candidate" if record.judge == record.variant else "baseline"
    return record.judge


def _position(records: Sequence[Record]) -> Optional[PositionBias]:
    """Position bias from pairs judged in both orders.

    A pair is two judgments of the same comparison with the candidate shown in
    slot A once and slot B once.  A judge that follows the content gives the
    same choice both times; a judge that follows the slot does not.
    """
    grouped: Dict[str, Dict[str, Record]] = {}
    for record in records:
        if record.pair_id and record.variant:
            grouped.setdefault(record.pair_id, {})[record.variant] = record
    pairs = {key: value for key, value in grouped.items() if len(value) >= 2}
    if not pairs:
        return None

    slot_a = 0
    slot_total = 0
    flips = 0
    comparable = 0
    for _, pair in sorted(pairs.items()):
        left = pair.get("a")
        right = pair.get("b")
        for record in pair.values():
            if record.judge in ("a", "b"):
                slot_total += 1
                if record.judge == "a":
                    slot_a += 1
        if left and right:
            left_choice = _choice(left)
            right_choice = _choice(right)
            if left_choice is None or right_choice is None:
                continue
            comparable += 1
            if left_choice != right_choice:
                flips += 1

    return PositionBias(
        pairs=len(pairs),
        first_slot_rate=slot_a / slot_total if slot_total else 0.0,
        flip_rate=flips / comparable if comparable else None,
        flips=flips,
        comparable=comparable,
    )


def _length(records: Sequence[Record]) -> Optional[LengthBias]:
    """Length bias as the rank correlation between output length and verdict.

    Pairwise length tests need per-candidate lengths, which the flat schema
    does not carry, so the card uses the correlation instead.  It is weaker
    evidence than a controlled swap, and the card says so.
    """
    with_length = [
        record
        for record in records
        if record.output_len is not None and record.pair_id is None
    ]
    if len(with_length) < 5:
        return None
    lengths = [float(record.output_len) for record in with_length]  # type: ignore[arg-type]
    if all(record.score is not None for record in with_length):
        values = [float(record.score) for record in with_length]  # type: ignore[arg-type]
        method = "spearman(length, score)"
    else:
        labels = sorted({record.judge for record in with_length})
        positive = positive_label(labels)
        if positive is None:
            ordered = ordinal_labels(labels)
            if not ordered:
                return None
            positive = ordered[-1]
        values = [1.0 if record.judge == positive else 0.0 for record in with_length]
        method = "spearman(length, verdict)"
    if len(set(lengths)) < 2 or len(set(values)) < 2:
        return None
    return LengthBias(
        method=method,
        n=len(with_length),
        value=stats.spearman(lengths, values),
        longer_wins=None,
    )


def _verdict(
    agreement: Optional[Agreement],
    calibration: Optional[Calibration],
    consistency: Optional[Consistency],
    position: Optional[PositionBias],
    length: Optional[LengthBias],
    items: int,
) -> Verdict:
    if agreement is None:
        return Verdict(
            grade="unvalidated",
            score=0,
            reasons=[
                "no paired human labels, so agreement with ground truth is unknown",
                "run judgecal again with a labelled sample before trusting this judge",
            ],
        )

    score = 0
    reasons: List[str] = []

    kappa = agreement.weighted_kappa if agreement.weighted_kappa is not None else agreement.kappa
    if kappa >= 0.8:
        score += 3
        reasons.append("chance-corrected agreement is strong (%.2f)" % kappa)
    elif kappa >= 0.6:
        score += 2
        reasons.append("chance-corrected agreement is substantial (%.2f)" % kappa)
    elif kappa >= 0.4:
        score += 1
        reasons.append("chance-corrected agreement is only moderate (%.2f)" % kappa)
    else:
        reasons.append("chance-corrected agreement is weak (%.2f)" % kappa)

    if items >= 200:
        score += 2
        reasons.append("sample size is large (%d items)" % items)
    elif items >= 60:
        score += 1
        reasons.append("sample size is adequate (%d items)" % items)
    else:
        reasons.append("sample size is small (%d items)" % items)

    if agreement.kappa_ci:
        low, high = agreement.kappa_ci
        if high - low <= 0.15:
            score += 1
            reasons.append("the kappa interval is tight (%.2f-%.2f)" % (low, high))
        else:
            reasons.append("the kappa interval is wide (%.2f-%.2f)" % (low, high))

    if calibration is not None:
        if calibration.ece <= 0.05:
            score += 1
            reasons.append("confidence scores are well calibrated (ECE %.3f)" % calibration.ece)
        elif calibration.ece <= 0.1:
            reasons.append("confidence scores are roughly calibrated (ECE %.3f)" % calibration.ece)
        else:
            reasons.append("confidence scores are badly calibrated (ECE %.3f)" % calibration.ece)
    else:
        reasons.append("no probability scores, so calibration was not measured")

    if consistency is not None:
        if consistency.flip_rate <= 0.05:
            score += 1
            reasons.append("repeated runs are stable (%.0f%% flips)" % (100 * consistency.flip_rate))
        else:
            reasons.append(
                "repeated runs are unstable (%.0f%% of items changed verdict)"
                % (100 * consistency.flip_rate)
            )
    else:
        reasons.append("no repeated runs, so stability was not measured")

    if position is not None and position.flip_rate is not None:
        if position.flip_rate <= 0.1:
            score += 1
            reasons.append("verdicts survive A/B position swaps")
        else:
            reasons.append(
                "verdicts change in %.0f%% of A/B swaps" % (100 * position.flip_rate)
            )

    if length is not None:
        if length.longer_wins is not None:
            gap = abs(length.longer_wins - 0.5)
        else:
            gap = abs(length.value)
        if gap <= 0.15:
            score += 1
            reasons.append("no meaningful length bias")
        else:
            reasons.append("verdicts track output length, not quality")

    grade = "D"
    if score >= 7:
        grade = "A"
    elif score >= 5:
        grade = "B"
    elif score >= 3:
        grade = "C"
    return Verdict(grade=grade, score=score, reasons=reasons)


def analyze(dataset: Dataset) -> Report:
    records = dataset.records
    agreement = _agreement(records)
    calibration = _calibration(records)
    consistency = _consistency(records)
    position = _position(records)
    length = _length(records)

    label_counts: Dict[str, int] = {}
    human_counts: Dict[str, int] = {}
    for record in records:
        # Pair rows carry a slot ("a"/"b") in `judge`, not a quality label, so
        # they would distort the label distribution.
        if record.pair_id is None:
            label_counts[record.judge] = label_counts.get(record.judge, 0) + 1
        if record.human is not None:
            human_counts[record.human] = human_counts.get(record.human, 0) + 1

    from .records import lint

    return Report(
        dataset=dataset,
        agreement=agreement,
        calibration=calibration,
        consistency=consistency,
        position=position,
        length=length,
        verdict=_verdict(
            agreement, calibration, consistency, position, length, len(dataset.ids)
        ),
        warnings=lint(records),
        label_counts=dict(sorted(label_counts.items())),
        human_counts=dict(sorted(human_counts.items())),
    )
