# judgecal

[![ci](https://github.com/Coisini-cyber/judgecal/actions/workflows/ci.yml/badge.svg)](https://github.com/Coisini-cyber/judgecal/actions/workflows/ci.yml)

**Reliability cards for LLM judges.**

Your eval suite says the new prompt is 12% better. Your judge is an LLM. Has
anyone checked whether that judge agrees with a human, stays stable across
runs, or just prefers the longer answer?

`judgecal` reads the judgments you already store and produces one page that
answers that, with chance-corrected statistics and explicit "not measured"
where your data cannot answer.

```bash
pip install judgecal          # or: python -m pip install -e .
judgecal demo -o demo.jsonl
judgecal card demo.jsonl
```

```text
## Verdict: C

> Weak evidence. Fine for triage, not for gating or reporting quality.

- chance-corrected agreement is substantial (0.77)
- repeated runs are unstable (20% of items changed verdict)
- verdicts change in 20% of A/B swaps
```

## Why

LLM-as-a-judge is now the default way to score model output, but the judge is
itself a model with biases: it prefers the first option, prefers longer
answers, is overconfident, and is not reproducible at temperature > 0. Teams
run thousands of judgments and never measure any of that.

Existing tools run judges (promptfoo, Langfuse, Braintrust). `judgecal` audits
the judge. It is complementary: keep your runner, export the judgments, get a
reliability card.

## What it measures

| Question | Statistic | Needs |
| --- | --- | --- |
| Does it agree with humans, beyond chance? | Cohen's kappa, quadratic weighted kappa, Krippendorff's alpha, per-class precision/recall, bootstrap CI | `human` |
| When it says 0.9, is it right 90% of the time? | ECE, Brier score, reliability bins, best threshold | `human` + `score` |
| Does the same item get the same answer twice? | flip rate, modal-verdict match | `run` |
| Does presentation change the verdict? | A/B swap flip rate, slot-A rate | `pair_id` + `variant` |
| Does length drive the verdict? | Spearman(length, score or verdict) | `output_len` or `output` |

Anything your data cannot support is reported as **not measured**, never
guessed. A card with no human labels gets the grade **unvalidated** and says so
at the top.

## Input

One JSON object per line (JSON Lines), a JSON array, or CSV. Only `id` and
`judge` are required.

```json
{"id": "q-001", "judge": "pass", "human": "pass", "score": 0.91, "model": "gpt-judge", "output_len": 412}
{"id": "q-002", "judge": "fail", "human": "pass", "score": 0.22, "model": "gpt-judge", "output_len": 188}
{"id": "q-001", "judge": "pass", "human": "pass", "score": 0.88, "model": "gpt-judge", "run": 2}
```

| field | type | meaning |
| --- | --- | --- |
| `id` | string | item identifier (required) |
| `judge` | string/number/bool | the judge's verdict (required) |
| `human` | string/number/bool | the human verdict, for agreement |
| `score` | 0-1 | judge confidence in the positive class |
| `run` | integer | repeat index; same `id` + different `run` = a repeat |
| `pair_id` | string | groups the two orders of one A/B comparison |
| `variant` | `a`/`b`/`left`/`right` | which slot the candidate under test was shown in |
| `model` | string | judge model name, reported in the card |
| `output_len` | integer | length of the judged output (derived from `output` if present) |
| `output` | string | the judged text; only its length is used |
| `meta` | object | anything else you want to keep |

Unknown fields are preserved under `meta`, so a dataset with extra columns
still loads.

### A/B pairs

A pair is **the same comparison judged twice with the order swapped**. `variant`
records where the candidate under test sat, and `judge` is the winner:

```json
{"id": "p1-order1", "pair_id": "p1", "variant": "a", "judge": "a"}
{"id": "p1-order2", "pair_id": "p1", "variant": "b", "judge": "b"}
```

Here the judge picked the candidate both times, so it is not position-biased.
If the second row said `"judge": "a"`, the judge picked slot A both times and
the card reports a 100% flip rate. `judge` may also name a candidate
(`"model_x"`) instead of a slot.

## Commands

```bash
judgecal card FILE                     # Markdown to stdout
judgecal card FILE -f html -o card.html # self-contained HTML, no CDN
judgecal card FILE -f json             # machine-readable report
judgecal stats FILE                    # same as card -f json
judgecal lint FILE                     # coverage and data-quality warnings
judgecal probes -o probes.jsonl        # generate position/length/stability probes
judgecal demo -o demo.jsonl            # synthetic dataset that exercises everything
judgecal compare before.jsonl after.jsonl
```

### Gating CI

`card` exits with status **3** when the verdict is below a floor, so a judge
regression fails the build:

```bash
judgecal card judgments.jsonl --min-grade b
judgecal card judgments.jsonl --min-kappa 0.65
```

Exit codes: `0` ok, `1` bad input, `2` usage, `3` gate failed.

### Probing a judge you have not measured yet

Most teams have no swapped-order pairs and no repeated runs, so the first card
is empty. Generate the missing evidence:

```bash
judgecal probes -o probes.jsonl
# run probes.jsonl through your judge, write the verdict into "judge"
judgecal card probes.judged.jsonl
```

The suite contains A/B swaps, a padded bad answer, repeated items, and four
edge cases (empty, refusal, off-topic, terse-but-correct).

## Reading a card

- **Grade A** — strong chance-corrected agreement, tight interval, calibrated
  confidence, stable, no measured bias. Safe for this task, on this sample.
- **B** — usable with known gaps.
- **C** — weak evidence; triage only.
- **D** — does not track human judgement well enough to be a metric.
- **unvalidated** — no human labels. Not a judgement about the judge, just an
  absence of evidence.

The score behind the grade is additive and every point is listed in the card,
so you can disagree with a specific reason instead of a black box.

Kappa conventions (Landis & Koch): `<0.4` weak, `0.4-0.6` moderate,
`0.6-0.8` substantial, `>0.8` strong. judgecal prints the number and the
interval, not just the word.

## Statistics

Implemented from the standard definitions, in plain Python, with hand-checked
tests:

- Cohen's kappa — `(po - pe) / (1 - pe)` (Cohen, 1960).
- Weighted kappa — linear and quadratic disagreement weights.
- Krippendorff's alpha — nominal, ordinal, and interval, any number of coders
  per unit (Krippendorff, 2004; 2011).
- Bootstrap CI — percentile interval over resampled pairs, seeded and
  deterministic.
- ECE — count-weighted `|accuracy - confidence|` over equal-width bins
  (Naeini et al., 2015; Guo et al., 2017).
- Brier score, precision/recall/F1, threshold sweep.
- Spearman's rho with average ranks for ties.

See [docs/metrics.md](docs/metrics.md) for the formulas and the exact
definitions used for bias.

## Limitations

- Small samples give wide intervals. The card reports the interval; 40 items
  cannot separate a 0.6 judge from a 0.8 judge.
- Agreement is measured against *your* labels, which have their own error.
  A judge that matches a noisy labeller inherits that noise.
- Length bias is a rank correlation, not a controlled experiment. It shows
  association; use the generated probes if you need causation.
- One card describes one dataset. Re-run it when the rubric, model, or task
  changes.
- No network calls, no telemetry, no dependencies.

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q judgecal
```

117 tests, no third-party packages, Python 3.9+.

## License

MIT.
