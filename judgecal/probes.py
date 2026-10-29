"""Generate a probe suite for a judge.

A reliability card can only measure what the data contains.  Most teams have
no position-swap pairs, no repeated runs, and no length-controlled examples,
so their card says "not measured" everywhere.

``judgecal probes`` writes a JSONL file you can feed to your judge.  Each row
is a judgment task with the bookkeeping fields (``pair_id``, ``variant``,
``run``) already filled in, so the answers drop straight back into
``judgecal card``.

The probes are deliberately mundane: they test whether a judge is stable and
unbiased, not whether it is clever.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List

# Two answers of clearly different quality.  The good one is specific and
# checks the input; the bad one is confident and wrong.
GOOD_ANSWER = (
    "Validate the email before inserting it: strip whitespace, reject addresses "
    "longer than 254 characters, and require exactly one '@' with a non-empty "
    "local part and a domain containing a dot. Return HTTP 400 with a field-level "
    "error message instead of raising a 500."
)
BAD_ANSWER = (
    "Just insert the email. The database will reject anything invalid and the user "
    "can retry, so no validation is needed here."
)

FILLER = (
    " In summary, this approach is generally considered best practice across the "
    "industry and should be applied consistently in all similar situations."
)

EDGE_CASES = [
    {
        "note": "empty answer",
        "output": "",
        "expected": "reject",
    },
    {
        "note": "refusal",
        "output": "I cannot help with that request.",
        "expected": "reject",
    },
    {
        "note": "off-topic but fluent",
        "output": (
            "The history of the semicolon is fascinating: it began as a rhetorical "
            "pause in Greek writing and only later became a statement terminator."
        ),
        "expected": "reject",
    },
    {
        "note": "correct but terse",
        "output": "Validate the email with a regex and return 400 on failure.",
        "expected": "accept",
    },
]

PROMPT = "Review this answer to 'how should the signup endpoint validate email?'"


@dataclass
class Probe:
    id: str
    prompt: str
    output: str
    pair_id: str = ""
    variant: str = ""
    run: int = 0
    category: str = ""
    note: str = ""

    def as_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "id": self.id,
            "prompt": self.prompt,
            "output": self.output,
            "output_len": len(self.output),
            "meta": {"category": self.category, "note": self.note},
        }
        if self.pair_id:
            row["pair_id"] = self.pair_id
            row["variant"] = self.variant
        if self.run:
            row["run"] = self.run
        return row


def build_probes(rounds: int = 3, seed: int = 0) -> List[Probe]:
    """Build position, length, stability, and edge-case probes."""
    if rounds < 2:
        rounds = 2
    rng = random.Random(seed)
    probes: List[Probe] = []

    # 1. Position swaps: identical content, both orders, several times.
    for index in range(4):
        pair_id = "pos-%02d" % (index + 1)
        probes.append(
            Probe(
                id="%s-a" % pair_id,
                prompt=PROMPT,
                output=GOOD_ANSWER,
                pair_id=pair_id,
                variant="a",
                category="position",
                note="good answer in slot A",
            )
        )
        probes.append(
            Probe(
                id="%s-b" % pair_id,
                prompt=PROMPT,
                output=BAD_ANSWER,
                pair_id=pair_id,
                variant="b",
                category="position",
                note="bad answer in slot B",
            )
        )
        swapped = "pos-%02d-swapped" % (index + 1)
        probes.append(
            Probe(
                id="%s-a" % swapped,
                prompt=PROMPT,
                output=BAD_ANSWER,
                pair_id=swapped,
                variant="a",
                category="position",
                note="bad answer in slot A",
            )
        )
        probes.append(
            Probe(
                id="%s-b" % swapped,
                prompt=PROMPT,
                output=GOOD_ANSWER,
                pair_id=swapped,
                variant="b",
                category="position",
                note="good answer in slot B",
            )
        )

    # 2. Length perturbation: same content, one side padded.
    for index in range(3):
        pair_id = "len-%02d" % (index + 1)
        probes.append(
            Probe(
                id="%s-a" % pair_id,
                prompt=PROMPT,
                output=GOOD_ANSWER,
                pair_id=pair_id,
                variant="a",
                category="length",
                note="good answer, normal length",
            )
        )
        probes.append(
            Probe(
                id="%s-b" % pair_id,
                prompt=PROMPT,
                output=BAD_ANSWER + FILLER * (index + 1),
                pair_id=pair_id,
                variant="b",
                category="length",
                note="bad answer padded with filler",
            )
        )

    # 3. Stability: the same item judged several times.
    for index in range(3):
        base = index + 1
        for run in range(1, rounds + 1):
            probes.append(
                Probe(
                    id="stab-%02d" % base,
                    prompt=PROMPT,
                    output=GOOD_ANSWER if base % 2 else BAD_ANSWER,
                    run=run,
                    category="stability",
                    note="repeated judgment, run %d" % run,
                )
            )

    # 4. Edge cases: empty, refusal, off-topic, terse-but-correct.
    for index, case in enumerate(EDGE_CASES, start=1):
        probes.append(
            Probe(
                id="edge-%02d" % index,
                prompt=PROMPT,
                output=case["output"],
                category="edge",
                note="%s (expected: %s)" % (case["note"], case["expected"]),
            )
        )

    rng.shuffle(probes)
    return probes


def to_jsonl(probes: List[Probe]) -> str:
    return "\n".join(json.dumps(probe.as_row(), ensure_ascii=False, sort_keys=True) for probe in probes) + "\n"
