import json
import os
import tempfile
import unittest

from judgecal.records import (
    Dataset,
    RecordError,
    label_of,
    lint,
    load_records,
    load_records_from_string,
    record_from_mapping,
    to_jsonl,
)


class LabelTest(unittest.TestCase):
    def test_booleans(self):
        self.assertEqual(label_of(True), "true")
        self.assertEqual(label_of(False), "false")

    def test_integral_floats_lose_the_decimal(self):
        self.assertEqual(label_of(3.0), "3")

    def test_strings_are_stripped(self):
        self.assertEqual(label_of("  pass  "), "pass")

    def test_empty_is_none(self):
        self.assertIsNone(label_of(""))
        self.assertIsNone(label_of(None))


class ParsingTest(unittest.TestCase):
    def test_jsonl(self):
        text = '{"id": "a", "judge": "pass"}\n{"id": "b", "judge": "fail"}\n'
        records = load_records_from_string(text)
        self.assertEqual([record.judge for record in records], ["pass", "fail"])

    def test_json_array(self):
        text = '[{"id": "a", "judge": "pass"}]'
        self.assertEqual(len(load_records_from_string(text)), 1)

    def test_csv(self):
        text = "id,judge,human\na,pass,pass\nb,fail,pass\n"
        records = load_records_from_string(text)
        self.assertEqual(records[1].human, "pass")

    def test_comments_and_blank_lines_are_skipped(self):
        text = '# header\n\n{"id": "a", "judge": "x"}\n'
        self.assertEqual(len(load_records_from_string(text)), 1)

    def test_missing_required_field_reports_line(self):
        with self.assertRaises(RecordError) as context:
            load_records_from_string('{"id": "a"}\n')
        self.assertIn("line 1", str(context.exception))
        self.assertIn("judge", str(context.exception))

    def test_variant_normalisation(self):
        self.assertEqual(record_from_mapping({"id": "a", "judge": "x", "variant": "left"}).variant, "a")
        self.assertEqual(record_from_mapping({"id": "a", "judge": "x", "variant": "2"}).variant, "b")

    def test_bad_variant_rejected(self):
        with self.assertRaises(RecordError):
            record_from_mapping({"id": "a", "judge": "x", "variant": "middle"})

    def test_output_length_is_derived_from_output(self):
        record = record_from_mapping({"id": "a", "judge": "x", "output": "abcd"})
        self.assertEqual(record.output_len, 4)

    def test_score_out_of_range(self):
        with self.assertRaises(RecordError):
            record_from_mapping({"id": "a", "judge": "x", "score": 1.4})

    def test_extra_fields_land_in_meta(self):
        record = record_from_mapping({"id": "a", "judge": "x", "dataset": "mmlu"})
        self.assertEqual(record.meta["dataset"], "mmlu")

    def test_duplicate_run_index_is_preserved(self):
        record = record_from_mapping({"id": "a", "judge": "x", "run": 2})
        self.assertEqual(record.run, 2)

    def test_missing_file(self):
        with self.assertRaises(RecordError):
            load_records(os.path.join(tempfile.gettempdir(), "judgecal-does-not-exist.jsonl"))

    def test_file_round_trip(self):
        rows = [{"id": "a", "judge": "pass", "human": "pass", "score": 0.9}]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rows.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(json.dumps(row) for row in rows) + "\n")
            records = load_records(path)
        self.assertEqual(records[0].score, 0.9)


class DatasetTest(unittest.TestCase):
    def make(self):
        return Dataset(
            [
                record_from_mapping({"id": "a", "judge": "pass", "human": "pass"}),
                record_from_mapping({"id": "a", "judge": "pass", "human": "pass", "run": 2}),
                record_from_mapping({"id": "b", "judge": "fail", "human": "pass"}),
            ]
        )

    def test_ids_are_unique_and_ordered(self):
        self.assertEqual(self.make().ids, ["a", "b"])

    def test_labelled(self):
        self.assertEqual(len(self.make().labelled()), 3)

    def test_repeated(self):
        self.assertEqual(len(self.make().repeated()), 1)

    def test_groups(self):
        self.assertEqual(len(self.make().groups()["a"]), 2)

    def test_models_empty(self):
        self.assertEqual(self.make().models, [])


class LintTest(unittest.TestCase):
    def test_no_human_labels(self):
        records = [record_from_mapping({"id": str(i), "judge": "pass"}) for i in range(40)]
        warnings = lint(records)
        self.assertTrue(any("no human labels" in warning for warning in warnings))

    def test_single_verdict(self):
        records = [
            record_from_mapping({"id": str(i), "judge": "pass", "human": "pass"})
            for i in range(40)
        ]
        warnings = lint(records)
        self.assertTrue(any("only ever produced" in warning for warning in warnings))

    def test_small_sample(self):
        records = [record_from_mapping({"id": "a", "judge": "pass", "human": "pass"})]
        self.assertTrue(any("distinct items" in warning for warning in lint(records)))

    def test_duplicate_detection(self):
        records = [
            record_from_mapping({"id": "a", "judge": "pass"}, 1),
            record_from_mapping({"id": "a", "judge": "fail"}, 2),
        ]
        self.assertTrue(any("duplicate judgment" in warning for warning in lint(records)))

    def test_incomplete_pairs(self):
        records = [
            record_from_mapping({"id": "a", "judge": "a", "pair_id": "p", "variant": "a"}),
            record_from_mapping({"id": "b", "judge": "a", "pair_id": "q", "variant": "a"}),
        ]
        self.assertTrue(any("missing an A/B counterpart" in warning for warning in lint(records)))

    def test_clean_dataset(self):
        records = [
            record_from_mapping({"id": str(i), "judge": "pass" if i % 2 else "fail", "human": "pass" if i % 3 else "fail"})
            for i in range(50)
        ]
        self.assertEqual(lint(records), [])


class SerialisationTest(unittest.TestCase):
    def test_round_trip(self):
        record = record_from_mapping(
            {"id": "a", "judge": "pass", "human": "pass", "score": 0.5, "model": "m"}
        )
        text = to_jsonl([record])
        reloaded = load_records_from_string(text)[0]
        self.assertEqual(reloaded.judge, "pass")
        self.assertEqual(reloaded.score, 0.5)
        self.assertEqual(reloaded.model, "m")


if __name__ == "__main__":
    unittest.main()
