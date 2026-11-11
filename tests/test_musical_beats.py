"""TR follows the ground-truth meter without changing position coordinates."""
import numpy as np
import pytest
from partitura.score import Note, Part, TimeSignature

from matchmaker_eval.verify_tracking import musical_beat_factors, score_meter_map
from matchmaker_eval.verify_tracking import check_tracking


def score(signatures=((0, 6, 8), (12, 4, 4))):
    part = Part('P1', quarter_duration=4)
    for start, numerator, denominator in signatures:
        part.add(TimeSignature(numerator, denominator), start=start)
    part.add(Note('C', 4, id='n0'), start=0, end=4)
    part.add(Note('D', 4, id='n1'), start=12, end=16)
    part.add(Note('E', 4, id='n2'), start=24, end=28)
    return part


def test_meter_changes_and_returns_preserve_score_coordinates():
    part = score()
    before = part.note_array().copy()
    meter = score_meter_map(part)
    # At the change and after a return, use the meter at that score position.
    assert np.array_equal(musical_beat_factors([-1, 0, 5.99, 6, 10, 2], meter), [3,3,3,1,1,3])
    assert np.array_equal(part.note_array(), before)


@pytest.mark.parametrize('threshold,error', [(1.,2.0),(.1,.25)])
@pytest.mark.parametrize('numerator,denominator,tracked', [
    (6,8,True), (9,8,True), (12,16,True), (3,4,False), (3,8,False),
])
def test_tr_uses_compound_groups_but_not_simple_triple_meter(
    threshold, error, numerator, denominator, tracked,
):
    gt = np.array([[0.,0.],[1.,1.],[2.,2.]])
    wp = gt.copy();wp[:,1] += error
    before = wp.copy()
    meter = score_meter_map(score(((0,numerator,denominator),)))
    result = check_tracking(
        wp, gt, threshold=threshold, beat_factors=musical_beat_factors(gt[:,1], meter),
    )
    assert result['tracked'] == tracked
    assert np.array_equal(wp,before)


def test_change_inside_a_window_is_applied_before_taking_median():
    gt = np.array([[0.,5.0],[1.,5.5],[2.,6.0]])
    wp = gt.copy();wp[:,1] += [2.4,2.4,1.5]
    # Two compound onsets are each 0.8 musical beat; the simple onset is 1.5.
    meter = score_meter_map(score())
    result = check_tracking(wp,gt,threshold=1.,beat_factors=musical_beat_factors(gt[:,1],meter))
    assert result['tracked']
    assert result['worst_window']['deviation'] == pytest.approx(.8)


def test_factor_follows_ground_truth_not_incorrect_predicted_region():
    meter = score_meter_map(score())
    gt = np.array([[0.,6.1],[1.,6.2],[2.,6.3]])
    wp = gt.copy();wp[:,1] -= 2.
    # Estimates lie back in compound meter, but ground truth is now in 4/4.
    assert not check_tracking(wp,gt,threshold=1.,beat_factors=musical_beat_factors(gt[:,1],meter))['tracked']
