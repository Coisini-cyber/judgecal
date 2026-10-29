"""judgecal command line interface."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

from . import __version__
from .card import render_html, render_json, render_markdown
from .demo import build_demo, to_jsonl as demo_jsonl
from .metrics import Report, analyze
from .probes import build_probes, to_jsonl as probes_jsonl
from .records import Dataset, RecordError, lint, load_records

GRADE_ORDER = {"a": 4, "b": 3, "c": 2, "d": 1, "unvalidated": 0}


def _read(path: str) -> Dataset:
    return Dataset(load_records(path), source=os.path.basename(path))


def _write(text: str, path: Optional[str]) -> None:
    if path:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print("wrote %s" % path, file=sys.stderr)
    else:
        sys.stdout.write(text)


def cmd_card(args: argparse.Namespace) -> int:
    dataset = _read(args.file)
    report = analyze(dataset)
    if args.format == "html":
        text = render_html(report)
    elif args.format == "json":
        text = render_json(report)
    else:
        text = render_markdown(report)
    _write(text, args.out)

    grade = report.verdict.grade
    if args.min_grade:
        floor = args.min_grade.lower()
        if GRADE_ORDER[grade] < GRADE_ORDER[floor]:
            print(
                "gate failed: grade %s is below the required %s"
                % (grade.upper(), floor.upper()),
                file=sys.stderr,
            )
            return 3
    if args.min_kappa is not None:
        if report.agreement is None:
            print("gate failed: no agreement data, so kappa is undefined", file=sys.stderr)
            return 3
        kappa = (
            report.agreement.weighted_kappa
            if report.agreement.weighted_kappa is not None
            else report.agreement.kappa
        )
        if kappa < args.min_kappa:
            print(
                "gate failed: kappa %.3f is below the required %.3f"
                % (kappa, args.min_kappa),
                file=sys.stderr,
            )
            return 3
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    dataset = _read(args.file)
    report = analyze(dataset)
    _write(render_json(report), args.out)
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    dataset = _read(args.file)
    warnings = lint(dataset.records)
    print("records: %d" % dataset.size)
    print("items:   %d" % len(dataset.ids))
    print("labels:  %s" % ", ".join(dataset.labels()))
    if warnings:
        print("")
        for warning in warnings:
            print("warning: %s" % warning)
    else:
        print("")
        print("no warnings")
    return 0


def cmd_probes(args: argparse.Namespace) -> int:
    probes = build_probes(rounds=args.rounds, seed=args.seed)
    _write(probes_jsonl(probes), args.out or "probes.jsonl")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    rows = build_demo(seed=args.seed, items=args.items)
    _write(demo_jsonl(rows), args.out or "demo.jsonl")
    return 0


def _summary(report: Report) -> dict:
    agreement = report.agreement
    calibration = report.calibration
    consistency = report.consistency
    position = report.position
    length = report.length
    kappa = None
    weighted = None
    if agreement is not None:
        kappa = agreement.kappa
        weighted = agreement.weighted_kappa
    return {
        "grade": report.verdict.grade.upper(),
        "items": len(report.dataset.ids),
        "kappa": kappa,
        "weighted_kappa": weighted,
        "ece": calibration.ece if calibration else None,
        "flip_rate": consistency.flip_rate if consistency else None,
        "position_flip": position.flip_rate if position else None,
        "length_r": length.value if length else None,
    }


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.3f" % value


def cmd_compare(args: argparse.Namespace) -> int:
    left = _summary(analyze(_read(args.left)))
    right = _summary(analyze(_read(args.right)))
    keys = [
        ("grade", "grade", "text"),
        ("items", "items", "int"),
        ("kappa", "kappa", "float"),
        ("weighted_kappa", "weighted kappa", "float"),
        ("ece", "ECE", "float"),
        ("flip_rate", "run flip rate", "float"),
        ("position_flip", "position flip rate", "float"),
        ("length_r", "length correlation", "float"),
    ]
    print("| metric | %s | %s | delta |" % (args.left, args.right))
    print("| --- | --- | --- | --- |")
    for key, label, kind in keys:
        left_value = left[key]
        right_value = right[key]
        if kind == "text":
            print("| %s | %s | %s | |" % (label, left_value, right_value))
            continue
        if kind == "int":
            print("| %s | %d | %d | %+d |" % (label, left_value, right_value, right_value - left_value))
            continue
        if left_value is None or right_value is None:
            print("| %s | %s | %s | |" % (label, _fmt(left_value), _fmt(right_value)))
            continue
        print(
            "| %s | %s | %s | %+.3f |"
            % (label, _fmt(left_value), _fmt(right_value), right_value - left_value)
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judgecal",
        description="Reliability cards for LLM judges: agreement, calibration, stability, bias.",
    )
    parser.add_argument("--version", action="version", version="judgecal %s" % __version__)
    subparsers = parser.add_subparsers(dest="command")

    card = subparsers.add_parser("card", help="render a reliability card")
    card.add_argument("file", help="judgments as JSONL, JSON array, or CSV")
    card.add_argument(
        "-f",
        "--format",
        choices=("md", "html", "json"),
        default="md",
        help="output format (default: md)",
    )
    card.add_argument("-o", "--out", help="write to this file instead of stdout")
    card.add_argument(
        "--min-grade",
        choices=("a", "b", "c", "d", "unvalidated"),
        help="exit with status 3 if the verdict is below this grade",
    )
    card.add_argument(
        "--min-kappa",
        type=float,
        help="exit with status 3 if chance-corrected agreement is below this value",
    )
    card.set_defaults(func=cmd_card)

    stats_parser = subparsers.add_parser("stats", help="print the report as JSON")
    stats_parser.add_argument("file")
    stats_parser.add_argument("-o", "--out")
    stats_parser.set_defaults(func=cmd_stats)

    lint_parser = subparsers.add_parser("lint", help="check coverage and data quality")
    lint_parser.add_argument("file")
    lint_parser.set_defaults(func=cmd_lint)

    probes = subparsers.add_parser(
        "probes", help="write a probe suite for position, length, and stability bias"
    )
    probes.add_argument("-o", "--out", help="output file (default: probes.jsonl)")
    probes.add_argument("--rounds", type=int, default=3, help="repeats per stability item")
    probes.add_argument("--seed", type=int, default=0)
    probes.set_defaults(func=cmd_probes)

    demo = subparsers.add_parser("demo", help="write a synthetic dataset to try the card")
    demo.add_argument("-o", "--out", help="output file (default: demo.jsonl)")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--items", type=int, default=120)
    demo.set_defaults(func=cmd_demo)

    compare = subparsers.add_parser("compare", help="compare two judgment files")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except RecordError as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
