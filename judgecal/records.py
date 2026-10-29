"""Loading, normalising, and linting judge outputs.

judgecal reads a flat table of judgments.  One row is one judgment of one
item by one judge.  The minimal row is ``{"id": ..., "judge": ...}``; every
other field unlocks an extra part of the reliability card.

Supported input: JSON Lines, a JSON array, or CSV.  No third-party packages.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

REQUIRED_FIELDS = ("id", "judge")

# Fields judgecal understands.  Anything else is kept in ``meta``.
KNOWN_FIELDS = {
    "id",
    "judge",
    "human",
    "score",
    "pair_id",
    "variant",
    "run",
    "model",
    "output_len",
    "output",
    "prompt",
    "meta",
}


class RecordError(ValueError):
    """Raised when an input row cannot be interpreted."""


@dataclass
class Record:
    id: str
    judge: str
    human: Optional[str] = None
    score: Optional[float] = None
    pair_id: Optional[str] = None
    variant: Optional[str] = None
    run: Optional[int] = None
    model: Optional[str] = None
    output_len: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    line: int = 0

    @property
    def has_human(self) -> bool:
        return self.human is not None


def label_of(value: Any) -> Optional[str]:
    """Canonical string label for a judge or human verdict.

    Booleans become ``"true"``/``"false"``, numbers are rendered without a
    trailing ``.0`` when they are integral, and strings are stripped.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return repr(value)
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return text or None


def _coerce_score(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RecordError("score is not a number: %r" % (value,)) from error
    if not 0.0 <= number <= 1.0:
        raise RecordError("score must be in [0, 1], got %r" % (value,))
    return number


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise RecordError("expected an integer, got %r" % (value,)) from error


def record_from_mapping(raw: Dict[str, Any], line: int = 0) -> Record:
    if not isinstance(raw, dict):
        raise RecordError("line %d: expected a JSON object" % line)
    missing = [name for name in REQUIRED_FIELDS if raw.get(name) in (None, "")]
    if missing:
        raise RecordError("line %d: missing %s" % (line, ", ".join(missing)))

    identifier = label_of(raw.get("id"))
    judge = label_of(raw.get("judge"))
    if identifier is None:
        raise RecordError("line %d: empty id" % line)
    if judge is None:
        raise RecordError("line %d: empty judge value" % line)

    variant = label_of(raw.get("variant"))
    if variant is not None:
        variant = variant.lower()
        if variant not in ("a", "b", "1", "2", "left", "right"):
            raise RecordError(
                "line %d: variant must be a/b/left/right, got %r" % (line, variant)
            )
        variant = {"a": "a", "b": "b", "1": "a", "2": "b", "left": "a", "right": "b"}[variant]

    output_len = _coerce_int(raw.get("output_len"))
    output = raw.get("output")
    if output_len is None and isinstance(output, str):
        output_len = len(output)

    meta = raw.get("meta")
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise RecordError("line %d: meta must be an object" % line)

    extra = {
        key: value
        for key, value in raw.items()
        if key not in KNOWN_FIELDS and value not in (None, "")
    }
    if extra:
        meta = dict(meta)
        meta.update(extra)

    return Record(
        id=identifier,
        judge=judge,
        human=label_of(raw.get("human")),
        score=_coerce_score(raw.get("score")),
        pair_id=label_of(raw.get("pair_id")),
        variant=variant,
        run=_coerce_int(raw.get("run")),
        model=label_of(raw.get("model")),
        output_len=output_len,
        meta=meta,
        line=line,
    )


def _rows_from_json(text: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    stripped = text.lstrip()
    if not stripped:
        return
    if stripped[0] == "[":
        try:
            data = json.loads(stripped)
        except ValueError as error:
            raise RecordError("invalid JSON: %s" % error)
        if not isinstance(data, list):
            raise RecordError("top-level JSON must be an array or one object per line")
        for index, row in enumerate(data, start=1):
            yield index, row
        return
    for index, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise RecordError("line %d: invalid JSON: %s" % (index, error))
        yield index, row


def _rows_from_csv(text: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise RecordError("CSV has no header row")
    for index, row in enumerate(reader, start=2):
        yield index, {key: value for key, value in row.items() if key is not None}


def load_records(path: str) -> List[Record]:
    """Load judgments from ``path``, choosing the parser by content."""
    if not os.path.exists(path):
        raise RecordError("file not found: %s" % path)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        text = handle.read()
    return load_records_from_string(text, source=path)


def load_records_from_string(text: str, source: str = "<memory>") -> List[Record]:
    """Parse judgments from a string, choosing the parser by content."""
    lines = text.splitlines()
    first = text.lstrip()[:1]
    if first in ("{", "["):
        rows = _rows_from_json(text)
    elif lines and "," in lines[0]:
        rows = _rows_from_csv(text)
    else:
        rows = _rows_from_json(text)

    records: List[Record] = []
    for line, raw in rows:
        records.append(record_from_mapping(raw, line))
    if not records:
        raise RecordError("no records found in %s" % source)
    return records


@dataclass
class Dataset:
    records: List[Record]
    source: str = "<memory>"

    def __post_init__(self) -> None:
        self.records = sorted(
            self.records,
            key=lambda record: (record.id, record.run if record.run is not None else -1),
        )

    # -- basic views -----------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def ids(self) -> List[str]:
        seen: List[str] = []
        known = set()
        for record in self.records:
            if record.id not in known:
                known.add(record.id)
                seen.append(record.id)
        return seen

    @property
    def models(self) -> List[str]:
        return sorted({record.model for record in self.records if record.model})

    def labelled(self) -> List[Record]:
        """Records that have both a judge verdict and a human verdict."""
        return [record for record in self.records if record.human is not None]

    def scored(self) -> List[Record]:
        return [record for record in self.records if record.score is not None]

    def paired(self) -> List[Record]:
        return [record for record in self.records if record.pair_id and record.variant]

    def repeated(self) -> List[Record]:
        return [record for record in self.records if record.run is not None]

    def with_lengths(self) -> List[Record]:
        return [record for record in self.records if record.output_len is not None]

    def labels(self) -> List[str]:
        values = {record.judge for record in self.records}
        values.update(record.human for record in self.records if record.human is not None)
        return sorted(values)

    def groups(self) -> Dict[str, List[Record]]:
        """Group repeated judgments of the same item by ``id``."""
        grouped: Dict[str, List[Record]] = {}
        for record in self.records:
            grouped.setdefault(record.id, []).append(record)
        return grouped

    def pair_groups(self) -> Dict[str, List[Record]]:
        grouped: Dict[str, List[Record]] = {}
        for record in self.paired():
            assert record.pair_id is not None
            grouped.setdefault(record.pair_id, []).append(record)
        return grouped


def lint(records: Sequence[Record]) -> List[str]:
    """Return human-readable warnings about coverage and consistency."""
    warnings: List[str] = []
    if not records:
        return ["no records"]

    ids = {record.id for record in records}
    labelled = [record for record in records if record.human is not None]
    if not labelled:
        warnings.append(
            "no human labels: agreement and calibration cannot be computed, "
            "so the judge is unvalidated by this dataset"
        )
    elif len(labelled) < len(records):
        warnings.append(
            "%d of %d records have no human label and are excluded from agreement"
            % (len(records) - len(labelled), len(records))
        )

    if len(ids) < 30:
        warnings.append(
            "only %d distinct items: confidence intervals will be wide" % len(ids)
        )

    judge_labels = {record.judge for record in records}
    if len(judge_labels) == 1:
        warnings.append(
            "the judge only ever produced %r, so no discrimination can be measured"
            % next(iter(judge_labels))
        )

    seen: Dict[Tuple[str, Optional[int]], int] = {}
    for record in records:
        key = (record.id, record.run)
        if key in seen:
            warnings.append(
                "duplicate judgment for id=%r run=%r on lines %d and %d"
                % (record.id, record.run, seen[key], record.line)
            )
        else:
            seen[key] = record.line

    models = {record.model for record in records if record.model}
    if len(models) > 1:
        warnings.append(
            "records mix %d judge models (%s); agreement numbers pool them"
            % (len(models), ", ".join(sorted(models)))
        )

    scored = [record for record in records if record.score is not None]
    if scored and len(scored) < len(records):
        warnings.append(
            "only %d of %d records carry a probability score" % (len(scored), len(records))
        )

    paired = [record for record in records if record.pair_id]
    if paired:
        by_pair: Dict[str, set] = {}
        for record in paired:
            by_pair.setdefault(record.pair_id or "", set()).add(record.variant)
        incomplete = [pair for pair, variants in by_pair.items() if len(variants) < 2]
        if incomplete:
            warnings.append(
                "%d pair(s) are missing an A/B counterpart, so position bias skips them"
                % len(incomplete)
            )
    return warnings


def to_jsonl(records: Iterable[Record]) -> str:
    lines = []
    for record in records:
        row: Dict[str, Any] = {"id": record.id, "judge": record.judge}
        for name in ("human", "score", "pair_id", "variant", "run", "model", "output_len"):
            value = getattr(record, name)
            if value is not None:
                row[name] = value
        if record.meta:
            row["meta"] = record.meta
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + "\n"
