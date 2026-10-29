import json
import unittest

from judgecal.card import render_html, render_json, render_markdown
from judgecal.metrics import analyze
from judgecal.records import Dataset, record_from_mapping


def full_dataset():
    rows = []
    for index in range(40):
        human = "pass" if index % 3 else "fail"
        judge = human if index >= 4 else ("fail" if human == "pass" else "pass")
        rows.append(
            {
                "id": "item-%02d" % index,
                "judge": judge,
                "human": human,
                "score": 0.85 if judge == "pass" else 0.2,
                "output_len": 100 + index * 7,
                "model": "test-judge",
            }
        )
    for index in range(6):
        for run in (2, 3):
            rows.append(
                {
                    "id": "item-%02d" % index,
                    "judge": "pass" if (index + run) % 2 else "fail",
                    "human": "pass",
                    "run": run,
                }
            )
    for index in range(4):
        rows.append(
            {"id": "p%d-a" % index, "judge": "a", "pair_id": "p%d" % index, "variant": "a"}
        )
        rows.append(
            {"id": "p%d-b" % index, "judge": "a", "pair_id": "p%d" % index, "variant": "b"}
        )
    return Dataset([record_from_mapping(row) for row in rows], source="fixture.jsonl")


class MarkdownTest(unittest.TestCase):
    def setUp(self):
        self.report = analyze(full_dataset())
        self.text = render_markdown(self.report)

    def test_has_every_section(self):
        for heading in (
            "# Judge reliability card",
            "## Verdict:",
            "## What was measured",
            "## Agreement with humans",
            "## Calibration of confidence scores",
            "## Stability across repeated runs",
            "## Position bias",
            "## Length bias",
            "## Data warnings",
            "## What to do next",
        ):
            self.assertIn(heading, self.text)

    def test_reports_measured_sections(self):
        self.assertIn("| agreement with humans | measured |", self.text)
        self.assertIn("| stability across runs | measured |", self.text)
        self.assertIn("| position bias | measured |", self.text)

    def test_confusion_matrix_is_rendered(self):
        self.assertIn("| human \\ judge | fail | pass |", self.text)

    def test_numbers_are_formatted(self):
        self.assertIn("Cohen's kappa", self.text)
        self.assertNotIn("nan", self.text.lower())

    def test_footer_names_the_tool(self):
        self.assertIn("judgecal", self.text)


class EmptyishTest(unittest.TestCase):
    def test_unlabelled_dataset_says_not_measured(self):
        rows = [{"id": str(index), "judge": "pass"} for index in range(10)]
        report = analyze(Dataset([record_from_mapping(row) for row in rows]))
        text = render_markdown(report)
        self.assertIn("Not measured", text)
        self.assertIn("unvalidated", text.lower())

    def test_minimal_markdown_is_valid(self):
        rows = [{"id": "a", "judge": "pass"}]
        report = analyze(Dataset([record_from_mapping(row) for row in rows]))
        text = render_markdown(report)
        self.assertTrue(text.endswith("\n"))


class HtmlTest(unittest.TestCase):
    def setUp(self):
        self.html = render_html(analyze(full_dataset()))

    def test_is_a_complete_document(self):
        self.assertTrue(self.html.startswith("<!doctype html>"))
        self.assertIn("</html>", self.html)

    def test_is_self_contained(self):
        lowered = self.html.lower()
        for forbidden in ("<script", "<link", "@import", "src=", "url(", "cdn."):
            self.assertNotIn(forbidden, lowered)

    def test_has_inline_styles_and_svg(self):
        self.assertIn("<style>", self.html)
        self.assertIn("<svg", self.html)

    def test_escapes_html_in_data(self):
        rows = [{"id": "a", "judge": "pass", "human": "pass"}]
        dataset = Dataset(
            [record_from_mapping(row) for row in rows],
            source="<script>alert(1)</script>.jsonl",
        )
        output = render_html(analyze(dataset))
        self.assertNotIn("<script>alert(1)</script>", output)
        self.assertIn("&lt;script&gt;", output)

    def test_tables_are_rendered(self):
        self.assertIn("<table>", self.html)
        self.assertIn("<th>", self.html)


class JsonTest(unittest.TestCase):
    def test_round_trips(self):
        report = analyze(full_dataset())
        data = json.loads(render_json(report))
        self.assertEqual(data["records"], report.dataset.size)
        self.assertIn("verdict", data)
        self.assertIn("agreement", data)
        self.assertEqual(data["verdict"]["grade"], report.verdict.grade)

    def test_unmeasured_sections_are_null(self):
        rows = [{"id": str(index), "judge": "pass"} for index in range(10)]
        report = analyze(Dataset([record_from_mapping(row) for row in rows]))
        data = json.loads(render_json(report))
        self.assertIsNone(data["agreement"])
        self.assertIsNone(data["calibration"])


if __name__ == "__main__":
    unittest.main()
