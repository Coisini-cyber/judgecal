"""Reliability statistics used by judgecal.

Everything here is implemented from the standard definitions, in plain
Python, so results can be checked by hand:

- Cohen's kappa, linear and quadratic weighted kappa
- Krippendorff's alpha (nominal, ordinal, interval)
- Spearman's rho and Pearson's r with average ranks
- expected calibration error (ECE) and Brier score
- bootstrap confidence intervals

The functions take plain sequences and return floats or small dataclasses.
They never import third-party packages.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Number = float


class StatsError(ValueError):
    """Raised when a statistic is undefined for the given data."""


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise StatsError("not a number: %r" % (value,)) from error


def mean(values: Sequence[float]) -> float:
    if not values:
        raise StatsError("mean of an empty sequence")
    return sum(values) / len(values)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation.  Returns 0.0 when either side is constant."""
    if len(xs) != len(ys):
        raise StatsError("pearson needs equal-length sequences")
    n = len(xs)
    if n < 2:
        raise StatsError("pearson needs at least two pairs")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, 1-based, ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        # ranks are 1-based: positions position..end map to (position+1)..(end+1)
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman's rank correlation with average ranks for ties."""
    if len(xs) != len(ys):
        raise StatsError("spearman needs equal-length sequences")
    if len(xs) < 2:
        raise StatsError("spearman needs at least two pairs")
    return pearson(_ranks(xs), _ranks(ys))


def cohens_kappa(
    a: Sequence[object], b: Sequence[object], labels: Optional[Sequence[object]] = None
) -> float:
    """Unweighted Cohen's kappa for two raters.

    ``labels`` fixes the category order; when omitted it is derived from the
    union of both raters' values in sorted order.
    """
    if len(a) != len(b):
        raise StatsError("cohens_kappa needs equal-length sequences")
    if not a:
        raise StatsError("cohens_kappa needs at least one pair")
    if labels is None:
        labels = sorted(set(a) | set(b), key=lambda value: (str(type(value)), value))
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    for value in list(a) + list(b):
        if value not in index:
            raise StatsError("label %r is not in the label set" % (value,))
    if size < 2:
        return 1.0
    table = [[0] * size for _ in range(size)]
    for left, right in zip(a, b):
        try:
            table[index[left]][index[right]] += 1
        except KeyError as error:
            raise StatsError("label %r is not in the label set" % (left,)) from error
    total = len(a)
    observed = sum(table[i][i] for i in range(size)) / total
    row = [sum(table[i]) for i in range(size)]
    column = [sum(table[i][j] for i in range(size)) for j in range(size)]
    expected = sum(row[i] * column[i] for i in range(size)) / (total * total)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _weights(size: int, kind: str) -> List[List[float]]:
    if kind not in ("linear", "quadratic"):
        raise StatsError("weight kind must be 'linear' or 'quadratic'")
    span = max(size - 1, 1)
    table = []
    for i in range(size):
        row = []
        for j in range(size):
            distance = abs(i - j) / span
            row.append(distance if kind == "linear" else distance * distance)
        table.append(row)
    return table


def weighted_kappa(
    a: Sequence[object],
    b: Sequence[object],
    kind: str = "quadratic",
    labels: Optional[Sequence[object]] = None,
) -> float:
    """Weighted kappa for ordinal ratings.

    ``kind="quadratic"`` is the usual choice for 1-5 style rubric scores.
    """
    if len(a) != len(b):
        raise StatsError("weighted_kappa needs equal-length sequences")
    if not a:
        raise StatsError("weighted_kappa needs at least one pair")
    if labels is None:
        labels = sorted(set(a) | set(b))
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    if size < 2:
        return 1.0
    table = [[0] * size for _ in range(size)]
    for left, right in zip(a, b):
        try:
            table[index[left]][index[right]] += 1
        except KeyError as error:
            raise StatsError("label %r is not in the label set" % (left,)) from error
    weights = _weights(size, kind)
    total = len(a)
    row = [sum(table[i]) for i in range(size)]
    column = [sum(table[i][j] for i in range(size)) for j in range(size)]
    observed = sum(
        weights[i][j] * table[i][j] for i in range(size) for j in range(size)
    ) / total
    expected = sum(
        weights[i][j] * row[i] * column[j] for i in range(size) for j in range(size)
    ) / (total * total)
    if expected == 0.0:
        return 1.0
    return 1.0 - observed / expected


def krippendorff_alpha(units: Iterable[Sequence[object]], level: str = "nominal") -> float:
    """Krippendorff's alpha for any number of coders per unit.

    ``units`` is an iterable of per-unit rating lists.  Units with fewer than
    two ratings are ignored, as the standard requires.
    """
    if level not in ("nominal", "ordinal", "interval"):
        raise StatsError("level must be 'nominal', 'ordinal' or 'interval'")
    cleaned: List[List[object]] = []
    for unit in units:
        values = [value for value in unit if value is not None]
        if len(values) >= 2:
            cleaned.append(values)
    if not cleaned:
        raise StatsError("krippendorff_alpha needs at least one unit with two ratings")

    values = sorted({value for unit in cleaned for value in unit}, key=str)
    position = {value: i for i, value in enumerate(values)}
    size = len(values)
    if size == 1:
        return 1.0

    coincidence = [[0.0] * size for _ in range(size)]
    total = 0
    for unit in cleaned:
        count = Counter(unit)
        m = len(unit)
        total += m
        for left, left_n in count.items():
            for right, right_n in count.items():
                pairs = left_n * (left_n - 1) if left == right else left_n * right_n
                coincidence[position[left]][position[right]] += pairs / (m - 1)

    def delta(i: int, j: int) -> float:
        if level == "nominal":
            return 0.0 if i == j else 1.0
        if level == "interval":
            return float((i - j) ** 2)
        # ordinal: rank-based metric
        low, high = min(i, j), max(i, j)
        if low == high:
            return 0.0
        between = sum(sum(coincidence[r]) for r in range(low, high + 1))
        edge = (sum(coincidence[low]) + sum(coincidence[high])) / 2.0
        return (between - edge) ** 2

    observed = sum(
        coincidence[i][j] * delta(i, j) for i in range(size) for j in range(size)
    )
    marginals = [sum(coincidence[i]) for i in range(size)]
    expected = sum(
        marginals[i] * marginals[j] * delta(i, j)
        for i in range(size)
        for j in range(size)
    )
    if total <= 1:
        raise StatsError("krippendorff_alpha needs more than one rating")
    observed /= total
    expected /= total * (total - 1)
    if expected == 0.0:
        return 1.0
    return 1.0 - observed / expected


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    if len(probabilities) != len(outcomes):
        raise StatsError("brier_score needs equal-length sequences")
    if not probabilities:
        raise StatsError("brier_score needs at least one pair")
    return sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / len(probabilities)


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float


def calibration_bins(
    probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> List[CalibrationBin]:
    """Equal-width bins over [0, 1], empty bins omitted."""
    if len(probabilities) != len(outcomes):
        raise StatsError("calibration_bins needs equal-length sequences")
    if bins < 1:
        raise StatsError("bins must be at least 1")
    buckets: Dict[int, List[Tuple[float, int]]] = {}
    for probability, outcome in zip(probabilities, outcomes):
        p = _as_float(probability)
        if not 0.0 <= p <= 1.0:
            raise StatsError("probabilities must be in [0, 1]")
        slot = min(int(p * bins), bins - 1)
        buckets.setdefault(slot, []).append((p, int(outcome)))
    result = []
    width = 1.0 / bins
    for slot in sorted(buckets):
        pairs = buckets[slot]
        result.append(
            CalibrationBin(
                lower=slot * width,
                upper=(slot + 1) * width,
                count=len(pairs),
                confidence=sum(p for p, _ in pairs) / len(pairs),
                accuracy=sum(o for _, o in pairs) / len(pairs),
            )
        )
    return result


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> float:
    """ECE: the count-weighted mean gap between confidence and accuracy."""
    if not probabilities:
        raise StatsError("expected_calibration_error needs at least one pair")
    total = len(probabilities)
    return sum(
        (entry.count / total) * abs(entry.accuracy - entry.confidence)
        for entry in calibration_bins(probabilities, outcomes, bins)
    )


def bootstrap_ci(
    values: Sequence[float],
    statistic,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap interval.  Deterministic for a given seed."""
    if not values:
        raise StatsError("bootstrap_ci needs at least one value")
    if iterations < 1:
        raise StatsError("iterations must be at least 1")
    rng = random.Random(seed)
    size = len(values)
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        try:
            estimates.append(statistic(sample))
        except StatsError:
            continue
    if not estimates:
        raise StatsError("statistic was undefined for every bootstrap sample")
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, min(len(estimates) - 1, int(math.floor(alpha * len(estimates)))))
    high_index = max(
        0, min(len(estimates) - 1, int(math.ceil((1.0 - alpha) * len(estimates))) - 1)
    )
    return estimates[low_index], estimates[high_index]


def threshold_metrics(
    probabilities: Sequence[float], outcomes: Sequence[int], threshold: float
) -> Dict[str, float]:
    """Confusion-matrix summaries for a binary judge at a fixed threshold."""
    if len(probabilities) != len(outcomes):
        raise StatsError("threshold_metrics needs equal-length sequences")
    tp = fp = tn = fn = 0
    for probability, outcome in zip(probabilities, outcomes):
        predicted = 1 if probability >= threshold else 0
        actual = 1 if outcome else 0
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and not actual:
            tn += 1
        else:
            fn += 1
    positives = tp + fn
    negatives = tn + fp
    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": (tp + tn) / max(len(outcomes), 1),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positives if positives else 0.0,
        "specificity": tn / negatives if negatives else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
    }


def best_threshold(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> Dict[str, float]:
    """Threshold in 0.01 steps that maximises accuracy, ties broken by F1."""
    if not probabilities:
        raise StatsError("best_threshold needs at least one value")
    best: Optional[Dict[str, float]] = None
    for step in range(1, 100):
        candidate = threshold_metrics(probabilities, outcomes, step / 100.0)
        if best is None:
            best = candidate
            continue
        if (candidate["accuracy"], candidate["f1"]) > (best["accuracy"], best["f1"]):
            best = candidate
    assert best is not None
    return best
