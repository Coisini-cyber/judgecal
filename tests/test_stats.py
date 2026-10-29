import math
import unittest

from judgecal import stats


class KappaTest(unittest.TestCase):
    def test_hand_computed_kappa(self):
        # 4 units, two raters.  po = 0.75, pe = 0.5, so kappa = 0.5.
        a = [1, 1, 0, 1]
        b = [1, 1, 0, 0]
        self.assertAlmostEqual(stats.cohens_kappa(a, b), 0.5)

    def test_perfect_agreement(self):
        self.assertEqual(stats.cohens_kappa(["x", "y", "z"], ["x", "y", "z"]), 1.0)

    def test_single_label_is_perfect(self):
        self.assertEqual(stats.cohens_kappa(["x", "x"], ["x", "x"]), 1.0)

    def test_unknown_label_rejected(self):
        with self.assertRaises(stats.StatsError):
            stats.cohens_kappa(["a"], ["b"], labels=["a"])

    def test_weighted_kappa_quadratic(self):
        a = [1, 2, 3, 3, 2]
        b = [1, 3, 3, 2, 2]
        # Observed disagreement 0.1, expected 0.28.
        self.assertAlmostEqual(stats.weighted_kappa(a, b, "quadratic"), 0.6428571428571428)

    def test_weighted_kappa_linear(self):
        a = [1, 2, 3, 3, 2]
        b = [1, 3, 3, 2, 2]
        # Observed disagreement 0.2, expected 0.4.
        self.assertAlmostEqual(stats.weighted_kappa(a, b, "linear"), 0.5)

    def test_weighted_kappa_beats_unweighted_for_near_misses(self):
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 4, 4]
        self.assertGreater(
            stats.weighted_kappa(a, b, "quadratic"), stats.cohens_kappa(a, b)
        )


class AlphaTest(unittest.TestCase):
    def test_hand_computed_nominal_alpha(self):
        # Same 4 units as the kappa test: Do = 0.25, De = 30/56.
        units = [[1, 1], [1, 1], [0, 0], [1, 0]]
        expected = 1 - 0.25 / (30 / 56)
        self.assertAlmostEqual(stats.krippendorff_alpha(units, "nominal"), expected)

    def test_alpha_differs_from_kappa(self):
        units = [[1, 1], [1, 1], [0, 0], [1, 0]]
        alpha = stats.krippendorff_alpha(units, "nominal")
        kappa = stats.cohens_kappa([1, 1, 0, 1], [1, 1, 0, 0])
        self.assertNotAlmostEqual(alpha, kappa, places=3)

    def test_perfect_alpha(self):
        units = [["a", "a"], ["b", "b"], ["c", "c"]]
        self.assertEqual(stats.krippendorff_alpha(units, "nominal"), 1.0)

    def test_units_with_one_rating_are_ignored(self):
        units = [["a", "a"], ["b"], ["c", "c"]]
        self.assertEqual(stats.krippendorff_alpha(units, "nominal"), 1.0)

    def test_three_coders_per_unit(self):
        units = [["a", "a", "a"], ["b", "b", "c"], ["c", "c", "c"]]
        value = stats.krippendorff_alpha(units, "nominal")
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)

    def test_ordinal_alpha_between_nominal_and_one(self):
        units = [[1, 2], [2, 3], [3, 3], [4, 5], [1, 1]]
        nominal = stats.krippendorff_alpha(units, "nominal")
        ordinal = stats.krippendorff_alpha(units, "ordinal")
        self.assertGreater(ordinal, nominal)
        self.assertLessEqual(ordinal, 1.0)

    def test_interval_alpha(self):
        units = [[1, 2], [2, 3], [3, 3]]
        value = stats.krippendorff_alpha(units, "interval")
        self.assertGreater(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_bad_level_rejected(self):
        with self.assertRaises(stats.StatsError):
            stats.krippendorff_alpha([[1, 2]], "ratio")


class CorrelationTest(unittest.TestCase):
    def test_pearson_perfect(self):
        self.assertAlmostEqual(stats.pearson([1, 2, 3], [2, 4, 6]), 1.0)

    def test_pearson_constant_is_zero(self):
        self.assertEqual(stats.pearson([1, 1, 1], [1, 2, 3]), 0.0)

    def test_spearman_hand_computed(self):
        # ys has a tie, so its ranks are 1, 2.5, 2.5, 4.
        value = stats.spearman([1, 2, 3, 4], [1, 2, 2, 3])
        self.assertAlmostEqual(value, 4.5 / math.sqrt(22.5))

    def test_spearman_detects_monotone_non_linear(self):
        xs = [1, 2, 3, 4, 5]
        ys = [1, 4, 9, 16, 25]
        self.assertAlmostEqual(stats.spearman(xs, ys), 1.0)

    def test_ranks_average_ties(self):
        self.assertEqual(stats._ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(stats.StatsError):
            stats.pearson([1, 2], [1])


class CalibrationTest(unittest.TestCase):
    def test_ece_hand_computed(self):
        probabilities = [0.9, 0.9, 0.6, 0.6, 0.3]
        outcomes = [1, 1, 0, 0, 0]
        self.assertAlmostEqual(
            stats.expected_calibration_error(probabilities, outcomes), 0.34
        )

    def test_brier_hand_computed(self):
        probabilities = [0.9, 0.9, 0.6, 0.6, 0.3]
        outcomes = [1, 1, 0, 0, 0]
        self.assertAlmostEqual(stats.brier_score(probabilities, outcomes), 0.166)

    def test_perfectly_calibrated_has_zero_ece(self):
        # Every bin contains one hit and one miss at the same confidence.
        probabilities = [0.5, 0.5, 0.5, 0.5]
        outcomes = [1, 0, 1, 0]
        self.assertAlmostEqual(
            stats.expected_calibration_error(probabilities, outcomes), 0.0
        )

    def test_probability_out_of_range(self):
        with self.assertRaises(stats.StatsError):
            stats.calibration_bins([1.5], [1])

    def test_bins_are_equal_width(self):
        bins = stats.calibration_bins([0.05, 0.15, 0.95], [1, 0, 1], bins=10)
        self.assertEqual(len(bins), 3)
        self.assertAlmostEqual(bins[0].upper - bins[0].lower, 0.1)

    def test_threshold_metrics(self):
        result = stats.threshold_metrics([0.9, 0.8, 0.2, 0.1], [1, 0, 0, 0], 0.5)
        self.assertEqual(result["tp"], 1.0)
        self.assertEqual(result["fp"], 1.0)
        self.assertEqual(result["tn"], 2.0)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_best_threshold_finds_separation(self):
        probabilities = [0.9, 0.85, 0.2, 0.15]
        outcomes = [1, 1, 0, 0]
        best = stats.best_threshold(probabilities, outcomes)
        self.assertGreaterEqual(best["threshold"], 0.2)
        self.assertLessEqual(best["threshold"], 0.9)
        self.assertEqual(best["accuracy"], 1.0)


class BootstrapTest(unittest.TestCase):
    def test_deterministic_for_seed(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        first = stats.bootstrap_ci(values, stats.mean, iterations=200, seed=3)
        second = stats.bootstrap_ci(values, stats.mean, iterations=200, seed=3)
        self.assertEqual(first, second)

    def test_interval_brackets_the_estimate(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        low, high = stats.bootstrap_ci(values, stats.mean, iterations=500, seed=1)
        self.assertLess(low, stats.mean(values))
        self.assertGreater(high, stats.mean(values))

    def test_empty_rejected(self):
        with self.assertRaises(stats.StatsError):
            stats.bootstrap_ci([], stats.mean)


if __name__ == "__main__":
    unittest.main()
