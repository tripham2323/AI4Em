"""Subject intersections and matched calibration-prefix masks, independent of summaries."""
from __future__ import annotations

import numpy as np


def select_p0_cohort(subjects_by_experiment):
    sets = {name: {str(subject) for subject in subjects}
            for name, subjects in subjects_by_experiment.items()}
    common = set.intersection(*sets.values()) if sets else set()
    dropped = {name: sorted(subjects - common) for name, subjects in sets.items()
               if subjects - common}
    return dict(cohort=sorted(common), dropped_by_experiment=dropped,
                identical_across_experiments=bool(sets) and not dropped)


def select_p1_comparison_cohort(p0_subjects, valid_p1_subjects):
    full = {str(subject) for subject in p0_subjects}
    valid = {str(subject) for subject in valid_p1_subjects}
    matched = full & valid
    return dict(cohort=sorted(matched), p1_blocked_subjects=sorted(full - valid),
                full_cohort_p1_coverage=len(matched) / len(full) if full else None)


def keep_after_calibration_prefix(subject_ids, window_start_ms, prefix_end_ms):
    starts = np.asarray(window_start_ms, dtype=float)
    subjects = list(subject_ids)
    if starts.ndim != 1 or len(subjects) != len(starts):
        raise ValueError('Subjects and window starts must have matching lengths')
    keep = np.ones(len(starts), dtype=bool)
    for index, (subject, start) in enumerate(zip(subjects, starts)):
        end = prefix_end_ms.get(subject, prefix_end_ms.get(str(subject)))
        if end is not None:
            keep[index] = start >= end
    return keep
