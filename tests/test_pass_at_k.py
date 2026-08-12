"""Tests for the pass@k estimator (Chen et al. 2021, unbiased)."""

import math
import pytest

from evaluator import pass_at_k, compute_pass_at_k_batch


# ---------------------------------------------------------------------------
# Basic known values
# ---------------------------------------------------------------------------

class TestPassAtK:
    def test_all_correct(self):
        """n=10, c=10 → pass@1 = 1.0 for all k."""
        assert pass_at_k(10, 10, 1) == 1.0
        assert pass_at_k(10, 10, 5) == 1.0
        assert pass_at_k(10, 10, 10) == 1.0

    def test_none_correct(self):
        """n=10, c=0 → pass@k = 0.0 for all k."""
        assert pass_at_k(10, 0, 1) == 0.0
        assert pass_at_k(10, 0, 5) == 0.0
        assert pass_at_k(10, 0, 10) == 0.0

    def test_half_correct(self):
        """n=10, c=5 → pass@1 = 0.5, pass@5 > 0.5, pass@10 = 1.0."""
        p1 = pass_at_k(10, 5, 1)
        p5 = pass_at_k(10, 5, 5)
        p10 = pass_at_k(10, 5, 10)

        assert p1 == pytest.approx(0.5, abs=1e-9)
        assert p5 > 0.5
        assert p5 < 1.0
        assert p10 == 1.0

    def test_pass_at_k_monotonic(self):
        """pass@k should increase with k for fixed n, c."""
        p1 = pass_at_k(10, 3, 1)
        p5 = pass_at_k(10, 3, 5)
        p10 = pass_at_k(10, 3, 10)
        assert p1 <= p5 <= p10

    def test_formula_manual(self):
        """Verify against manual calculation: pass@1 = 1 - C(n-c,1)/C(n,1) = c/n."""
        # pass@1 = 1 - C(n-c, 1) / C(n, 1) = 1 - (n-c)/n = c/n
        assert pass_at_k(10, 3, 1) == pytest.approx(3 / 10)
        assert pass_at_k(10, 7, 1) == pytest.approx(7 / 10)
        assert pass_at_k(20, 5, 1) == pytest.approx(5 / 20)

    def test_pass_at_5_manual(self):
        """Verify pass@5 against manual combinatorics."""
        # pass@5 = 1 - C(7, 5) / C(10, 5)
        expected = 1.0 - math.comb(7, 5) / math.comb(10, 5)
        assert pass_at_k(10, 3, 5) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPassAtKEdgeCases:
    def test_single_sample_correct(self):
        assert pass_at_k(1, 1, 1) == 1.0

    def test_single_sample_incorrect(self):
        assert pass_at_k(1, 0, 1) == 0.0

    def test_k_equals_n(self):
        """When k == n, pass@k = 1.0 if c >= 1."""
        assert pass_at_k(5, 1, 5) == 1.0
        assert pass_at_k(5, 0, 5) == 0.0

    def test_k_greater_than_n_raises(self):
        with pytest.raises(ValueError, match="n=.*must be >= k="):
            pass_at_k(5, 3, 10)

    def test_negative_c_raises(self):
        with pytest.raises(ValueError, match="c=.*must be in"):
            pass_at_k(10, -1, 1)

    def test_c_greater_than_n_raises(self):
        with pytest.raises(ValueError, match="c=.*must be in"):
            pass_at_k(10, 11, 1)

    def test_c_just_above_threshold(self):
        """n - c < k should return 1.0 (guaranteed at least one correct in k draws)."""
        # n=10, c=8, k=5 → n-c=2 < k=5 → 1.0
        assert pass_at_k(10, 8, 5) == 1.0


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

class TestPassAtKBatch:
    def test_single_program(self):
        result = compute_pass_at_k_batch([(10, 5)], k_values=[1, 5, 10])
        assert 1 in result
        assert 5 in result
        assert 10 in result
        assert result[1] == pytest.approx(0.5)

    def test_multiple_programs_average(self):
        """Average of pass@1 for two programs: one all-correct, one all-wrong."""
        result = compute_pass_at_k_batch([(10, 10), (10, 0)], k_values=[1])
        assert result[1] == pytest.approx(0.5)

    def test_empty_results(self):
        result = compute_pass_at_k_batch([], k_values=[1])
        assert result[1] == 0.0

    def test_skips_programs_with_too_few_samples(self):
        """Programs with n < k should be skipped, not crash."""
        result = compute_pass_at_k_batch([(3, 2)], k_values=[1, 5])
        assert 1 in result
        assert result[1] == pytest.approx(2 / 3)
        # k=5 should have no valid programs, so 0.0
        assert result[5] == 0.0
