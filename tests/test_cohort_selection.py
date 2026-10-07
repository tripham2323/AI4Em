"""Matched subject cohorts and reserved-prefix exclusion for comparisons."""
import numpy as np
import pytest

from src.datasets import cohorts as sm


def test_p0_cohort_is_the_intersection_and_reports_who_was_dropped():
    out = sm.select_p0_cohort({"A": ["s1", "s2", "s3"], "D": ["s1", "s2"]})
    assert out["cohort"] == ["s1", "s2"] and out["dropped_by_experiment"] == {"A": ["s3"]}
    assert out["identical_across_experiments"] is False
    same = sm.select_p0_cohort({"A": ["s1"], "B": ["s1"]})
    assert same["identical_across_experiments"] is True


def test_p1_cohort_excludes_blocked_subjects_but_reports_them():
    out = sm.select_p1_comparison_cohort(["s1", "s2", "s51"], ["s1", "s2"])
    assert out["cohort"] == ["s1", "s2"] and out["p1_blocked_subjects"] == ["s51"]
    assert out["full_cohort_p1_coverage"] == pytest.approx(2 / 3)


def test_p1_cohort_with_no_valid_profile_is_empty_not_faked():
    out = sm.select_p1_comparison_cohort(["s5"], [])
    assert out["cohort"] == [] and out["full_cohort_p1_coverage"] == 0.0


def test_prefix_windows_are_removed_for_matched_d_comparison():
    keep = sm.keep_after_calibration_prefix(["a", "a", "a", "b"], [0, 29_999, 30_000, 5], {"a": 30_000})
    assert keep.tolist() == [False, False, True, True]  # subject b has no prefix: untouched
    with pytest.raises(ValueError):
        sm.keep_after_calibration_prefix(["a"], [1, 2], {})


def test_foreign_valid_profiles_do_not_inflate_p1_coverage():
    out = sm.select_p1_comparison_cohort(["a", "b"], ["a", "outside"])
    assert out["cohort"] == ["a"] and out["p1_blocked_subjects"] == ["b"]
    assert out["full_cohort_p1_coverage"] == 0.5
    empty = sm.select_p1_comparison_cohort([], ["outside"])
    assert empty["cohort"] == [] and empty["full_cohort_p1_coverage"] is None
