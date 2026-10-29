"""Render a reliability report as Markdown, HTML, or JSON.

The HTML output is a single self-contained file: no CDN, no web fonts, no
remote images, so it can be attached to a PR or opened offline.
"""

from __future__ import annotations

import html
import json
from typing import List, Optional

from .metrics import Report

GRADE_COLORS = {
    "A": "#1a7f37",
    "B": "#2da44e",
    "C": "#bf8700",
    "D": "#cf222e",
    "unvalidated": "#6e7781",
}

GRADE_BLURB = {
    "A": "Trustworthy for this task, within the measured sample.",
    "B": "Usable, with known gaps. Prefer a human spot-check on high-stakes items.",
    "C": "Weak evidence. Fine for triage, not for gating or reporting quality.",
    "D": "Does not track human judgement well enough to use as a metric.",
    "unvalidated": "No human labels: this judge has not been checked against anything.",
}


def _pct(value: float) -> str:
    return "%.1f%%" % (100 * value)


def _num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return ("%." + str(digits) + "f") % value


def _status(available: bool) -> str:
    return "measured" if available else "not measured"


def _agreement_section(report: Report, level: str = "##") -> List[str]:
    agreement = report.agreement
    lines: List[str] = []
    if agreement is None:
        lines.append("%s Agreement with humans" % level)
        lines.append("")
        lines.append(
            "Not measured: no record pairs a `judge` verdict with a `human` verdict."
        )
        lines.append("")
        return lines

    lines.append("%s Agreement with humans" % level)
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append("| paired judgments | %d |" % agreement.n)
    lines.append("| raw accuracy | %s |" % _pct(agreement.accuracy))
    lines.append("| Cohen's kappa | %s |" % _num(agreement.kappa))
    if agreement.kappa_ci:
        lines.append(
            "| kappa 95%% bootstrap interval | %s to %s |"
            % (_num(agreement.kappa_ci[0]), _num(agreement.kappa_ci[1]))
        )
    if agreement.weighted_kappa is not None:
        lines.append(
            "| weighted kappa (%s) | %s |"
            % (agreement.weighted_kind, _num(agreement.weighted_kappa))
        )
    if agreement.alpha is not None:
        lines.append("| Krippendorff's alpha | %s |" % _num(agreement.alpha))
    lines.append("")

    labels = agreement.labels
    lines.append("Confusion matrix (rows: human, columns: judge):")
    lines.append("")
    lines.append("| human \\ judge | " + " | ".join(labels) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in labels) + " |")
    for actual in labels:
        cells = [str(agreement.confusion[actual][predicted]) for predicted in labels]
        lines.append("| **%s** | %s |" % (actual, " | ".join(cells)))
    lines.append("")

    lines.append("| label | precision | recall | support |")
    lines.append("| --- | --- | --- | --- |")
    for label in labels:
        entry = agreement.per_class[label]
        lines.append(
            "| %s | %s | %s | %d |"
            % (
                label,
                _num(entry["precision"], 2),
                _num(entry["recall"], 2),
                int(entry["support"]),
            )
        )
    lines.append("")
    return lines


def _calibration_section(report: Report, level: str = "##") -> List[str]:
    calibration = report.calibration
    lines: List[str] = []
    lines.append("%s Calibration of confidence scores" % level)
    lines.append("")
    if calibration is None:
        lines.append(
            "Not measured: needs at least five records with both `score` (0-1) and "
            "`human`, and a binary or ordinal label set."
        )
        lines.append("")
        return lines

    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append("| scored judgments | %d |" % calibration.n)
    lines.append("| expected calibration error | %s |" % _num(calibration.ece))
    lines.append("| Brier score | %s |" % _num(calibration.brier))
    lines.append(
        "| best threshold | %s (accuracy %s, F1 %s) |"
        % (
            _num(calibration.best["threshold"], 2),
            _num(calibration.best["accuracy"], 3),
            _num(calibration.best["f1"], 3),
        )
    )
    lines.append("")
    lines.append("| confidence bin | n | mean confidence | observed accuracy | gap |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in calibration.bins:
        gap = entry.accuracy - entry.confidence
        lines.append(
            "| %.2f-%.2f | %d | %s | %s | %+.3f |"
            % (entry.lower, entry.upper, entry.count, _num(entry.confidence), _num(entry.accuracy), gap)
        )
    lines.append("")
    return lines


def _consistency_section(report: Report, level: str = "##") -> List[str]:
    consistency = report.consistency
    lines: List[str] = []
    lines.append("%s Stability across repeated runs" % level)
    lines.append("")
    if consistency is None:
        lines.append(
            "Not measured: no item appears twice with different `run` values. "
            "Re-run the judge 3-5 times on the same items to measure this."
        )
        lines.append("")
        return lines
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append("| items with repeated runs | %d |" % consistency.items)
    lines.append("| judgments compared | %d |" % consistency.runs)
    lines.append("| items whose verdict changed | %s |" % _pct(consistency.flip_rate))
    lines.append("| judgments matching the item's modal verdict | %s |" % _pct(consistency.exact_match))
    lines.append("")
    if consistency.unstable:
        lines.append("Most unstable items:")
        lines.append("")
        for identifier, verdicts in consistency.unstable[:5]:
            lines.append("- `%s`: %s" % (identifier, ", ".join(verdicts)))
        lines.append("")
    return lines


def _position_section(report: Report, level: str = "##") -> List[str]:
    position = report.position
    lines: List[str] = []
    lines.append("%s Position bias" % level)
    lines.append("")
    if position is None:
        lines.append(
            "Not measured: needs `pair_id` and `variant` (a/b) on at least one pair."
        )
        lines.append("")
        return lines
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append("| pairs judged in both orders | %d |" % position.pairs)
    lines.append("| judgments that chose slot A | %s |" % _pct(position.first_slot_rate))
    if position.flip_rate is None:
        lines.append("| choice changed when the order was swapped | n/a (no complete pairs) |")
    else:
        lines.append(
            "| choice changed when the order was swapped | %s (%d of %d) |"
            % (_pct(position.flip_rate), position.flips, position.comparable)
        )
    lines.append("")
    return lines


def _length_section(report: Report, level: str = "##") -> List[str]:
    length = report.length
    lines: List[str] = []
    lines.append("%s Length bias" % level)
    lines.append("")
    if length is None:
        lines.append(
            "Not measured: needs `output_len` (or `output`) on at least five records."
        )
        lines.append("")
        return lines
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append("| method | %s |" % length.method)
    lines.append("| n | %d |" % length.n)
    if length.longer_wins is not None:
        lines.append("| longer output wins | %s |" % _pct(length.longer_wins))
        lines.append("| distance from a coin flip | %s |" % _num(abs(length.longer_wins - 0.5)))
    else:
        lines.append("| correlation with length | %s |" % _num(length.value))
    lines.append("")
    return lines


def render_markdown(report: Report) -> str:
    verdict = report.verdict
    lines: List[str] = []
    lines.append("# Judge reliability card")
    lines.append("")
    lines.append("**Source:** `%s`  " % report.dataset.source)
    lines.append(
        "**Data:** %d judgments, %d distinct items, %d judge model(s)"
        % (report.dataset.size, len(report.dataset.ids), max(len(report.dataset.models), 1))
    )
    if report.dataset.models:
        lines.append("**Models:** %s  " % ", ".join(report.dataset.models))
    lines.append("")
    lines.append("## Verdict: %s" % verdict.grade.upper())
    lines.append("")
    lines.append("> %s" % GRADE_BLURB[verdict.grade])
    lines.append("")
    lines.append("Score: %d. Reasons:" % verdict.score)
    lines.append("")
    for reason in verdict.reasons:
        lines.append("- %s" % reason)
    lines.append("")

    lines.append("## What was measured")
    lines.append("")
    lines.append("| question | status |")
    lines.append("| --- | --- |")
    lines.append("| agreement with humans | %s |" % _status(report.agreement is not None))
    lines.append("| confidence calibration | %s |" % _status(report.calibration is not None))
    lines.append("| stability across runs | %s |" % _status(report.consistency is not None))
    lines.append("| position bias | %s |" % _status(report.position is not None))
    lines.append("| length bias | %s |" % _status(report.length is not None))
    lines.append("")

    lines.append("## Label distribution")
    lines.append("")
    lines.append("| label | judge count | human count |")
    lines.append("| --- | --- | --- |")
    for label in sorted(set(report.label_counts) | set(report.human_counts)):
        lines.append(
            "| %s | %d | %d |"
            % (label, report.label_counts.get(label, 0), report.human_counts.get(label, 0))
        )
    lines.append("")

    lines.extend(_agreement_section(report))
    lines.extend(_calibration_section(report))
    lines.extend(_consistency_section(report))
    lines.extend(_position_section(report))
    lines.extend(_length_section(report))

    lines.append("## Data warnings")
    lines.append("")
    if report.warnings:
        for warning in report.warnings:
            lines.append("- %s" % warning)
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## What to do next")
    lines.append("")
    for action in _actions(report):
        lines.append("- %s" % action)
    lines.append("")
    lines.append(
        "Generated by [judgecal](https://github.com/Coisini-cyber/judgecal). "
        "Numbers describe this dataset, not the judge in general."
    )
    lines.append("")
    return "\n".join(lines)


def _actions(report: Report) -> List[str]:
    actions: List[str] = []
    if report.agreement is None:
        actions.append(
            "Label 100-200 items by hand and re-run: without that, no agreement number exists."
        )
    elif len(report.dataset.ids) < 60:
        actions.append("Grow the labelled sample to at least 60 items, ideally 200.")
    if report.agreement is not None and report.agreement.kappa < 0.6:
        actions.append(
            "Rewrite the rubric: name the decision each label implies and add one "
            "positive and one negative example per label."
        )
    if report.calibration is None:
        actions.append(
            "Ask the judge for a 0-1 confidence score and store it in `score`; "
            "calibration is the cheapest way to catch an overconfident judge."
        )
    elif report.calibration.ece > 0.1:
        actions.append(
            "Recalibrate: fit a monotone mapping from score to observed accuracy, "
            "or lower every reported confidence."
        )
    if report.consistency is None:
        actions.append("Run the judge 3-5 times per item to measure verdict stability.")
    elif report.consistency.flip_rate > 0.05:
        actions.append(
            "Reduce sampling temperature or use majority voting over three runs."
        )
    if report.position is not None and report.position.flip_rate and report.position.flip_rate > 0.1:
        actions.append("Randomise or double-run A/B order and average the two verdicts.")
    if report.length is not None:
        gap = (
            abs(report.length.longer_wins - 0.5)
            if report.length.longer_wins is not None
            else abs(report.length.value)
        )
        if gap > 0.15:
            actions.append(
                "Length is moving the verdict: add length-controlled pairs to the rubric "
                "or truncate candidates to a common budget."
            )
    if not actions:
        actions.append("Keep the current sampling and re-run this card when the rubric changes.")
    return actions


def _svg_calibration(report: Report) -> str:
    """A small reliability diagram, drawn inline so the card stays standalone."""
    calibration = report.calibration
    width = 340
    height = 260
    plot = 180
    x0 = (width - plot) / 2.0
    y0 = (height - plot) / 2.0 - 6

    def sx(value: float) -> float:
        return x0 + plot * value

    def sy(value: float) -> float:
        return y0 + plot * (1.0 - value)

    parts = [
        '<svg viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
        'aria-label="Reliability diagram">' % (width, height, width, height),
        '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (width, height),
    ]
    for step in range(5):
        value = step / 4.0
        parts.append(
            '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#eaeef2" '
            'stroke-width="1"/>' % (sx(value), sy(0), sx(value), sy(1))
        )
        parts.append(
            '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#eaeef2" '
            'stroke-width="1"/>' % (sx(0), sy(value), sx(1), sy(value))
        )
    parts.append(
        '<rect x="%.1f" y="%.1f" width="%d" height="%d" fill="none" '
        'stroke="#d0d7de" stroke-width="1"/>' % (x0, y0, plot, plot)
    )
    parts.append(
        '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#8c959f" '
        'stroke-width="1" stroke-dasharray="4 4"/>' % (sx(0), sy(0), sx(1), sy(1))
    )
    if calibration is not None:
        points = []
        for entry in calibration.bins:
            radius = 3.0 + min(6.0, entry.count ** 0.5)
            points.append((sx(entry.confidence), sy(entry.accuracy), radius))
        if len(points) > 1:
            path = " ".join("%.1f,%.1f" % (x, y) for x, y, _ in points)
            parts.append(
                '<polyline points="%s" fill="none" stroke="#0969da" stroke-width="2"/>' % path
            )
        for x, y, radius in points:
            parts.append(
                '<circle cx="%.1f" cy="%.1f" r="%.1f" fill="#0969da" fill-opacity="0.75"/>'
                % (x, y, radius)
            )
    parts.append(
        '<text x="%.1f" y="%.1f" text-anchor="middle" font-family="system-ui, sans-serif" '
        'font-size="11" fill="#57606a">confidence</text>' % (x0 + plot / 2.0, y0 + plot + 22)
    )
    parts.append(
        '<text x="%.1f" y="%.1f" text-anchor="middle" font-family="system-ui, sans-serif" '
        'font-size="11" fill="#57606a" transform="rotate(-90 %.1f %.1f)">accuracy</text>'
        % (x0 - 22, y0 + plot / 2.0, x0 - 22, y0 + plot / 2.0)
    )
    parts.append("</svg>")
    return "".join(parts)


def render_html(report: Report) -> str:
    verdict = report.verdict
    color = GRADE_COLORS[verdict.grade]
    body = render_markdown(report)
    # Convert the markdown body into HTML with a deliberately small converter:
    # headings, tables, lists, blockquotes, and paragraphs.  The card only ever
    # renders content this module produced, so the surface is small.
    html_body = _markdown_to_html(body)
    if report.calibration is not None:
        figure = (
            "<figure>%s<figcaption>Reliability diagram: a perfectly calibrated judge "
            "follows the dashed diagonal. Bubble size is the number of judgments in the "
            "bin.</figcaption></figure>" % _svg_calibration(report)
        )
    else:
        figure = ""
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judge reliability card</title>
<style>
:root { color-scheme: light; }
body { margin: 0; padding: 40px 20px; background: #f6f8fa;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: #1f2328; line-height: 1.55; }
main { max-width: 860px; margin: 0 auto; background: #fff; padding: 40px 44px;
  border: 1px solid #d0d7de; border-radius: 12px; }
h1 { font-size: 26px; margin: 0 0 8px; }
h2 { font-size: 19px; margin: 32px 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid #eaeef2; }
table { border-collapse: collapse; width: 100%%; margin: 12px 0 20px; font-size: 14px; }
th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }
th { background: #f6f8fa; font-weight: 600; }
code { background: #f6f8fa; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
blockquote { margin: 12px 0; padding: 10px 16px; border-left: 4px solid %s;
  background: #f6f8fa; color: #424a53; }
.grade { display: inline-block; padding: 2px 12px; border-radius: 999px;
  background: %s; color: #fff; font-weight: 600; letter-spacing: .02em; }
figure { margin: 16px 0 24px; }
figcaption { font-size: 13px; color: #57606a; }
ul { padding-left: 22px; }
</style>
</head>
<body>
<main>
<div class="grade">%s</div>
%s
<figure>%s<figcaption>Reliability diagram: a perfectly calibrated judge
follows the dashed diagonal. Bubble size is the number of judgments in the bin.</figcaption></figure>
</main>
</body>
</html>
""" % (
        color,
        color,
        html.escape(verdict.grade.upper()),
        html_body,
        _svg_calibration(report),
    )


def _markdown_to_html(text: str) -> str:
    """Minimal Markdown renderer for the subset used by render_markdown."""
    lines = text.splitlines()
    out: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("## "):
            out.append("<h2>%s</h2>" % _inline(stripped[3:]))
            index += 1
            continue
        if stripped.startswith("# "):
            out.append("<h1>%s</h1>" % _inline(stripped[2:]))
            index += 1
            continue
        if stripped.startswith("> "):
            out.append("<blockquote>%s</blockquote>" % _inline(stripped[2:]))
            index += 1
            continue
        if stripped.startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index].strip())
                index += 1
            out.append(_table(block))
            continue
        if stripped.startswith("- "):
            items = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(lines[index].strip()[2:])
                index += 1
            out.append("<ul>%s</ul>" % "".join("<li>%s</li>" % _inline(item) for item in items))
            continue
        paragraph = []
        while (
            index < len(lines)
            and lines[index].strip()
            and not lines[index].strip().startswith(("#", "|", "- ", "> "))
        ):
            paragraph.append(lines[index].strip())
            index += 1
        out.append("<p>%s</p>" % _inline(" ".join(paragraph)))
    return "\n".join(out)


def _table(rows: List[str]) -> str:
    cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]
    if len(cells) >= 2 and all(set(cell) <= set("-: ") for cell in cells[1]):
        header, body = cells[0], cells[2:]
    else:
        header, body = cells[0], cells[1:]
    out = ["<table><thead><tr>"]
    out.extend("<th>%s</th>" % _inline(cell) for cell in header)
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        out.extend("<td>%s</td>" % _inline(cell) for cell in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = escaped.replace("&lt;br&gt;", "<br>")
    escaped = _links(escaped)
    # Bold **x** first, then inline code `x`, so code spans keep their content.
    parts = escaped.split("`")
    rebuilt = []
    for position, part in enumerate(parts):
        if position % 2 == 1:
            rebuilt.append("<code>%s</code>" % part)
        else:
            rebuilt.append(_bold(part))
    return "".join(rebuilt)


def _links(text: str) -> str:
    """Turn ``[label](url)`` into an anchor, on already-escaped text."""
    out: List[str] = []
    index = 0
    while True:
        start = text.find("[", index)
        if start == -1:
            out.append(text[index:])
            break
        middle = text.find("](", start)
        end = text.find(")", middle + 2) if middle != -1 else -1
        if middle == -1 or end == -1:
            out.append(text[index:])
            break
        url = text[middle + 2 : end]
        if not url.startswith(("http://", "https://")):
            out.append(text[index : end + 1])
            index = end + 1
            continue
        out.append(text[index:start])
        out.append('<a href="%s">%s</a>' % (url, text[start + 1 : middle]))
        index = end + 1
    return "".join(out)


def _bold(text: str) -> str:
    pieces = text.split("**")
    if len(pieces) % 2 == 0:
        return text
    return "".join(
        ("<strong>%s</strong>" % piece) if position % 2 else piece
        for position, piece in enumerate(pieces)
    )


def render_json(report: Report) -> str:
    return json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n"
