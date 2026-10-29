"""Build a synthetic dataset so `judgecal demo` shows a realistic card.

The judge in the demo is deliberately imperfect: it agrees with humans about
85% of the time, is overconfident, flips on a few repeated runs, and has a
mild position bias.  That is what a real judge looks like, and it exercises
every section of the card.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List


def build_demo(seed: int = 7, items: int = 120) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []

    for index in range(1, items + 1):
        identifier = "item-%03d" % index
        human = "pass" if rng.random() < 0.6 else "fail"
        # 85% agreement, with errors concentrated on borderline items.
        if rng.random() < 0.85:
            judge = human
        else:
            judge = "fail" if human == "pass" else "pass"

        if judge == "pass":
            score = min(0.99, max(0.01, rng.gauss(0.82, 0.09)))
        else:
            score = min(0.99, max(0.01, rng.gauss(0.32, 0.12)))

        rows.append(
            {
                "id": identifier,
                "judge": judge,
                "human": human,
                "score": round(score, 3),
                "model": "demo-judge-v1",
                "output_len": rng.randint(80, 900),
            }
        )

    # Repeated runs on 30 items, with a small flip rate.
    for index in range(1, 31):
        identifier = "item-%03d" % index
        base = next(row for row in rows if row["id"] == identifier)
        for run in (2, 3):
            judge = base["judge"]
            if rng.random() < 0.08:
                judge = "fail" if judge == "pass" else "pass"
            rows.append(
                {
                    "id": identifier,
                    "judge": judge,
                    "human": base["human"],
                    "score": round(min(0.99, max(0.01, rng.gauss(0.6, 0.15))), 3),
                    "model": "demo-judge-v1",
                    "run": run,
                    "output_len": base["output_len"],
                }
            )

    # A/B pairs judged in both orders.  The candidate under test sits in slot A
    # once and slot B once.  The judge follows the content 85% of the time and
    # otherwise just picks slot A, which is the bias the card should detect.
    for index in range(1, 21):
        pair_id = "pair-%02d" % index
        good = rng.random() < 0.5
        for order, slot in ((1, "a"), (2, "b")):
            follows_content = rng.random() < 0.85
            if follows_content:
                winner = slot if good else ("b" if slot == "a" else "a")
            else:
                winner = "a"
            rows.append(
                {
                    "id": "%s-order%d" % (pair_id, order),
                    "judge": winner,
                    "pair_id": pair_id,
                    "variant": slot,
                    "model": "demo-judge-v1",
                    "output_len": rng.randint(200, 800),
                }
            )

    return rows


def to_jsonl(rows: List[Dict[str, object]]) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
