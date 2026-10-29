import contextlib
import io
import json
import os
import tempfile
import unittest

from judgecal.cli import main


def run(argv):
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class DemoAndCardTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "demo.jsonl")
        code, _, _ = run(["demo", "-o", self.path])
        self.assertEqual(code, 0)

    def tearDown(self):
        self.directory.cleanup()

    def test_demo_writes_jsonl(self):
        with open(self.path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertGreater(len(rows), 100)
        self.assertIn("judge", rows[0])

    def test_card_markdown(self):
        code, out, _ = run(["card", self.path])
        self.assertEqual(code, 0)
        self.assertIn("# Judge reliability card", out)
        self.assertIn("## Verdict:", out)

    def test_card_json(self):
        code, out, _ = run(["card", self.path, "-f", "json"])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertIn("agreement", data)

    def test_card_html_writes_file(self):
        target = os.path.join(self.directory.name, "card.html")
        code, _, _ = run(["card", self.path, "-f", "html", "-o", target])
        self.assertEqual(code, 0)
        with open(target, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("<!doctype html>", text)

    def test_stats(self):
        code, out, _ = run(["stats", self.path])
        self.assertEqual(code, 0)
        self.assertIn('"verdict"', out)

    def test_lint(self):
        code, out, _ = run(["lint", self.path])
        self.assertEqual(code, 0)
        self.assertIn("records:", out)


class GateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.directory.cleanup()

    def write(self, rows):
        path = os.path.join(self.directory.name, "rows.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_min_grade_fails_on_unvalidated(self):
        path = self.write([{"id": str(i), "judge": "pass"} for i in range(20)])
        code, _, err = run(["card", path, "--min-grade", "c"])
        self.assertEqual(code, 3)
        self.assertIn("gate failed", err)

    def test_min_grade_passes_on_unvalidated_when_asked(self):
        path = self.write([{"id": str(i), "judge": "pass"} for i in range(20)])
        code, _, _ = run(["card", path, "--min-grade", "unvalidated"])
        self.assertEqual(code, 0)

    def test_min_kappa_without_labels_fails(self):
        path = self.write([{"id": str(i), "judge": "pass"} for i in range(20)])
        code, _, err = run(["card", path, "--min-kappa", "0.5"])
        self.assertEqual(code, 3)
        self.assertIn("kappa", err)

    def test_min_kappa_passes(self):
        rows = []
        for index in range(60):
            human = "pass" if index % 2 else "fail"
            rows.append({"id": str(index), "judge": human, "human": human})
        path = self.write(rows)
        code, _, _ = run(["card", path, "--min-kappa", "0.9"])
        self.assertEqual(code, 0)


class ProbesTest(unittest.TestCase):
    def test_probes_are_written_and_parseable(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "probes.jsonl")
            code, _, _ = run(["probes", "-o", target])
            self.assertEqual(code, 0)
            with open(target, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        categories = {row["meta"]["category"] for row in rows}
        self.assertEqual(categories, {"position", "length", "stability", "edge"})
        position = [row for row in rows if row["meta"]["category"] == "position"]
        self.assertTrue(any(row.get("variant") == "b" for row in position))

    def test_probe_rows_carry_the_bookkeeping_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "probes.jsonl")
            run(["probes", "-o", target])
            with open(target, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("id", row)
            self.assertIn("prompt", row)
            self.assertIn("output", row)
            self.assertEqual(row["output_len"], len(row["output"]))
            self.assertIn("category", row["meta"])
        stability = [row for row in rows if row["meta"]["category"] == "stability"]
        self.assertTrue(any(row.get("run", 0) >= 2 for row in stability))


class CompareTest(unittest.TestCase):
    def test_compare_prints_a_table(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "a.jsonl")
            second = os.path.join(directory, "b.jsonl")
            run(["demo", "-o", first, "--seed", "1"])
            run(["demo", "-o", second, "--seed", "2"])
            code, out, _ = run(["compare", first, second])
        self.assertEqual(code, 0)
        self.assertIn("| metric |", out)
        self.assertIn("kappa", out)


class ErrorTest(unittest.TestCase):
    def test_missing_file(self):
        code, _, err = run(["card", "does-not-exist.jsonl"])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not json}\n")
            code, _, err = run(["card", path])
        self.assertEqual(code, 1)
        self.assertIn("invalid JSON", err)

    def test_no_command_prints_help(self):
        code, out, _ = run([])
        self.assertEqual(code, 2)
        self.assertIn("usage:", out)


if __name__ == "__main__":
    unittest.main()
