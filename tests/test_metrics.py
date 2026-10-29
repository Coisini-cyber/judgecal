import unittest

from judgecal.metrics import analyze, ordinal_labels, positive_label
from judgecal.records import Dataset, record_from_mapping


def dataset(rows):
    return Dataset([record_from_mapping(row) for row in rows])


def labelled_rows(n=40, mistakes=0):
    """n items where the judge is right except for ``mistakes`` items."""
    rows = []
    for index in range(n):
        human = "pass" if index % 3 else "fail"
        judge = human
        if index < mistakes:
            judge = "fail" if human == "pass" else "pass"
        rows.append({"id": "item-%03d" % index, "judge": judge, "human": human})
    return rows


class LabelHelpersTest(unittest.TestCase):
    def test_numeric_labels_are_ordinal(self):
        self.assertEqual(ordinal_labels(["3", "1", "2"]), ["1", "2", "3"])

    def test_word_labels_are_ordinal(self):
        self.assertEqual(
            ordinal_labels(["good", "poor", "excellent"]),
            ["poor", "good", "excellent"],
        )

    def test_binary_is_not_ordinal(self):
        self.assertIsNone(ordinal_labels(["pass", "fail"]))

    def test_positive_label(self):
        self.assertEqual(positive_label(["fail", "pass"]), "pass")
        self.assertEqual(positive_label(["no", "yes"]), "yes")
        self.assertIsNone(positive_label(["a", "b", "c"]))


class AgreementTest(unittest.TestCase):
    def test_perfect_agreement(self):
        report = analyze(dataset(labelled_rows(40, mistakes=0)))
        self.assertAlmostEqual(report.agreement.accuracy, 1.0)
        self.assertAlmostEqual(report.agreement.kappa, 1.0)

    def test_known_confusion(self):
        # 3 pass/pass, 1 pass/fail, 1 fail/pass, 2 fail/fail.
        rows = [
            {"id": "1", "judge": "pass", "human": "pass"},
            {"id": "2", "judge": "pass", "human": "pass"},
            {"id": "3", "judge": "pass", "human": "pass"},
            {"id": "4", "judge": "fail", "human": "pass"},
            {"id": "5", "judge": "pass", "human": "fail"},
            {"id": "6", "judge": "fail", "human": "fail"},
            {"id": "7", "judge": "fail", "human": "fail"},
        ]
        report = analyze(dataset(rows))
        self.assertEqual(report.agreement.confusion["pass"]["pass"], 3)
        self.assertEqual(report.agreement.confusion["pass"]["fail"], 1)
        self.assertAlmostEqual(report.agreement.accuracy, 5 / 7)
        # po=5/7, pe=(4*4 + 3*3)/49 = 25/49, kappa=(5/7-25/49)/(1-25/49)
        expected = (5 / 7 - 25 / 49) / (1 - 25 / 49)
        self.assertAlmostEqual(report.agreement.kappa, expected)

    def test_per_class_metrics(self):
        report = analyze(dataset(labelled_rows(30, mistakes=3)))
        self.assertIn("precision", report.agreement.per_class["pass"])
        self.assertGreater(report.agreement.per_class["pass"]["support"], 0)

    def test_confidence_interval_is_reported(self):
        report = analyze(dataset(labelled_rows(60, mistakes=6)))
        self.assertIsNotNone(report.agreement.kappa_ci)
        low, high = report.agreement.kappa_ci
        self.assertLessEqual(low, high)

    def test_ordinal_data_gets_weighted_kappa(self):
        rows = [
            {"id": str(index), "judge": str((index % 5) + 1), "human": str((index % 5) + 1)}
            for index in range(30)
        ]
        rows[0] = {"id": "0", "judge": "2", "human": "1"}
        report = analyze(dataset(rows))
        self.assertTrue(report.agreement.ordinal)
        self.assertIsNotNone(report.agreement.weighted_kappa)

    def test_no_human_labels_means_no_agreement(self):
        rows = [{"id": str(index), "judge": "pass"} for index in range(10)]
        report = analyze(dataset(rows))
        self.assertIsNone(report.agreement)
        self.assertEqual(report.verdict.grade, "unvalidated")


class CalibrationTest(unittest.TestCase):
    def test_calibration_uses_positive_label(self):
        rows = [
            {"id": "1", "judge": "pass", "human": "pass", "score": 0.9},
            {"id": "2", "judge": "pass", "human": "pass", "score": 0.8},
            {"id": "3", "judge": "fail", "human": "fail", "score": 0.2},
            {"id": "4", "judge": "fail", "human": "fail", "score": 0.1},
            {"id": "5", "judge": "pass", "human": "pass", "score": 0.7},
        ]
        report = analyze(dataset(rows))
        self.assertIsNotNone(report.calibration)
        self.assertGreater(report.calibration.ece, 0.0)
        self.assertLess(report.calibration.ece, 0.2)
        self.assertEqual(report.calibration.n, 5)

    def test_too_few_scores(self):
        rows = [
            {"id": "1", "judge": "pass", "human": "pass", "score": 0.9},
        ]
        self.assertIsNone(analyze(dataset(rows)).calibration)


class ConsistencyTest(unittest.TestCase):
    def test_stable_judge(self):
        rows = []
        for index in range(5):
            for run in (1, 2, 3):
                rows.append({"id": "i%d" % index, "judge": "pass", "run": run})
        report = analyze(dataset(rows))
        self.assertEqual(report.consistency.flip_rate, 0.0)
        self.assertEqual(report.consistency.exact_match, 1.0)

    def test_unstable_judge(self):
        rows = []
        for index in range(4):
            for run in (1, 2, 3):
                judge = "pass" if (index + run) % 2 else "fail"
                rows.append({"id": "i%d" % index, "judge": judge, "run": run})
        report = analyze(dataset(rows))
        self.assertEqual(report.consistency.flip_rate, 1.0)
        self.assertTrue(report.consistency.unstable)

    def test_no_repeats(self):
        rows = [{"id": "i%d" % index, "judge": "pass"} for index in range(5)]
        self.assertIsNone(analyze(dataset(rows)).consistency)


class PositionTest(unittest.TestCase):
    def test_content_following_judge_has_no_flip(self):
        rows = [
            # candidate is in slot A for order 1 and slot B for order 2
            {"id": "p1-a", "judge": "a", "pair_id": "p1", "variant": "a"},
            {"id": "p1-b", "judge": "b", "pair_id": "p1", "variant": "b"},
        ]
        report = analyze(dataset(rows))
        self.assertEqual(report.position.pairs, 1)
        self.assertEqual(report.position.flip_rate, 0.0)

    def test_slot_a_biased_judge_flips(self):
        rows = [
            {"id": "p1-a", "judge": "a", "pair_id": "p1", "variant": "a"},
            {"id": "p1-b", "judge": "a", "pair_id": "p1", "variant": "b"},
        ]
        report = analyze(dataset(rows))
        self.assertEqual(report.position.flip_rate, 1.0)
        self.assertEqual(report.position.first_slot_rate, 1.0)

    def test_candidate_name_answers(self):
        rows = [
            {"id": "p1-a", "judge": "model_x", "pair_id": "p1", "variant": "a"},
            {"id": "p1-b", "judge": "model_x", "pair_id": "p1", "variant": "b"},
        ]
        report = analyze(dataset(rows))
        self.assertEqual(report.position.flip_rate, 0.0)

    def test_incomplete_pair_is_skipped(self):
        rows = [{"id": "p1-a", "judge": "a", "pair_id": "p1", "variant": "a"}]
        self.assertIsNone(analyze(dataset(rows)).position)


class LengthTest(unittest.TestCase):
    def test_longer_outputs_score_higher(self):
        rows = [
            {
                "id": str(index),
                "judge": "pass" if index > 4 else "fail",
                "score": index / 10.0,
                "output_len": index * 10,
            }
            for index in range(10)
        ]
        report = analyze(dataset(rows))
        self.assertIsNotNone(report.length)
        self.assertGreater(report.length.value, 0.9)

    def test_no_relationship(self):
        rows = [
            {"id": str(index), "judge": "pass" if index % 2 else "fail", "output_len": 100}
            for index in range(10)
        ]
        # Constant length is not informative, so no measurement is claimed.
        self.assertIsNone(analyze(dataset(rows)).length)


class VerdictTest(unittest.TestCase):
    def test_unvalidated_without_labels(self):
        rows = [{"id": str(index), "judge": "pass"} for index in range(50)]
        self.assertEqual(analyze(dataset(rows)).verdict.grade, "unvalidated")

    def test_strong_dataset_gets_a_high_grade(self):
        rows = labelled_rows(240, mistakes=4)
        report = analyze(dataset(rows))
        self.assertIn(report.verdict.grade, ("A", "B"))

    def test_weak_dataset_gets_a_low_grade(self):
        rows = labelled_rows(20, mistakes=10)
        report = analyze(dataset(rows))
        self.assertIn(report.verdict.grade, ("C", "D"))

    def test_reasons_are_always_present(self):
        report = analyze(dataset(labelled_rows(40, mistakes=5)))
        self.assertTrue(report.verdict.reasons)


if __name__ == "__main__":
    unittest.main()
