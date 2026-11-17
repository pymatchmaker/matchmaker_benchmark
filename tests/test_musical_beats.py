"""TR divides position errors by the reference meter's musical-beat factor."""

import numpy as np
import pytest
from partitura.score import Note, Part, TimeSignature

from matchmaker_eval.verify_tracking import check_tracking, musical_beat_factors, score_meter_map


def score(signatures=((0, 6, 8), (12, 4, 4))):
    """6/8 for six denominator-note beats, then 4/4."""
    part = Part("P1", quarter_duration=4)
    for start, numerator, denominator in signatures:
        part.add(TimeSignature(numerator, denominator), start=start)
    part.add(Note("C", 4, id="n0"), start=0, end=4)
    part.add(Note("D", 4, id="n1"), start=12, end=16)
    part.add(Note("E", 4, id="n2"), start=24, end=28)
    return part


def tracked(wp, gt, part, threshold=1.0):
    factors = musical_beat_factors(gt[:, 1], score_meter_map(part))
    return check_tracking(wp, gt, threshold=threshold, beat_factors=factors)


def test_factor_follows_the_meter_at_each_position():
    meter = score_meter_map(score())
    # A pickup uses the first meter; a return to an earlier position uses its meter again.
    assert musical_beat_factors([-1, 0, 5.99, 6, 10, 2], meter).tolist() == [3, 3, 3, 1, 1, 3]


@pytest.mark.parametrize("numerator,denominator,expected", [
    (6, 8, True), (9, 8, True), (12, 16, True), (3, 4, False), (3, 8, False),
])
def test_only_compound_meters_group_three_beats(numerator, denominator, expected):
    gt = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    wp = gt + [0.0, 2.0]
    assert tracked(wp, gt, score(((0, numerator, denominator),)))["tracked"] is expected


def test_a_meter_change_inside_a_window_applies_per_onset():
    gt = np.array([[0.0, 5.0], [1.0, 5.5], [2.0, 6.0]])
    wp = gt + np.array([[0, 2.4], [0, 2.4], [0, 1.5]])
    # 2.4 / 3 = 0.8 for the two 6/8 onsets, 1.5 for the 4/4 onset: median 0.8.
    result = tracked(wp, gt, score())
    assert result["tracked"]
    assert result["worst_window"]["deviation"] == pytest.approx(0.8)


def test_the_factor_comes_from_the_ground_truth_position():
    gt = np.array([[0.0, 6.1], [1.0, 6.2], [2.0, 6.3]])
    wp = gt - [0.0, 2.0]
    # The estimates fall back in 6/8, but the reference is already in 4/4.
    assert not tracked(wp, gt, score())["tracked"]
