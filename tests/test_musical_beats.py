"""TR follows the reference meter while position-error coordinates stay fixed."""
import numpy as np
import pytest
from partitura.score import Note, Part, TimeSignature

from matchmaker_eval.beat_units import (
    TRACKING_THRESHOLDS, denominator_notes_per_beat, musical_beat_factors,
    score_meter_map,
)
from matchmaker_eval.verify_tracking import check_tracking


def score(signatures=((0, 6, 8), (12, 4, 4))):
    part = Part('P1', quarter_duration=4)
    for start, numerator, denominator in signatures:
        part.add(TimeSignature(numerator, denominator), start=start)
    part.add(Note('C', 4, id='n0'), start=0, end=4)
    part.add(Note('D', 4, id='n1'), start=12, end=16)
    part.add(Note('E', 4, id='n2'), start=24, end=28)
    return part


@pytest.mark.parametrize('numerator,factor', [(2,1),(3,1),(4,1),(5,1),(6,3),(7,1),(9,3),(12,3),(15,3)])
def test_grouping(numerator, factor):
    assert denominator_notes_per_beat(numerator) == factor


def test_mixed_meter_uses_each_reference_position_without_mutating_score():
    part = score()
    before = part.note_array().copy()
    meter = score_meter_map(part)
    assert [r['start_score_beat'] for r in meter['changes']] == [0, 6]
    assert np.array_equal(musical_beat_factors([-1, 0, 5.99, 6, 10, 2], meter), [3,3,3,1,1,3])
    assert np.array_equal(part.note_array(), before)
    assert not part._use_musical_beat


def test_simple_triple_meter_does_not_group_three_beats():
    meter = score_meter_map(score(((0,3,8),)))
    assert musical_beat_factors([0,1,2], meter).tolist() == [1,1,1]


def test_missing_meter_is_explicitly_recorded():
    meter = score_meter_map(score(()))
    assert meter['assumed_time_signature'] is True
    assert musical_beat_factors([0,12], meter).tolist() == [1,1]


def test_already_converted_score_is_rejected():
    part = score()
    part.use_musical_beat()
    with pytest.raises(ValueError, match='denominator-note'):
        score_meter_map(part)


@pytest.mark.parametrize('threshold,error', [(1.,2.0),(.1,.25)])
def test_same_native_error_passes_in_compound_but_fails_in_simple(threshold,error):
    gt = np.array([[0.,0.],[1.,1.],[2.,2.]])
    wp = gt.copy();wp[:,1] += error
    before = wp.copy()
    assert not check_tracking(wp,gt,threshold=threshold,beat_factors=np.ones(3))['tracked']
    grouped = check_tracking(wp,gt,threshold=threshold,beat_factors=np.full(3,3.))
    assert grouped['tracked'] and grouped['error_unit']=='musical_beat'
    assert np.array_equal(wp,before)


def test_change_inside_a_window_is_applied_before_taking_median():
    gt = np.array([[0.,0.],[1.,1.],[2.,2.]])
    wp = gt.copy();wp[:,1] += [2.4,2.4,1.5]
    # Two compound onsets are each 0.8 musical beat; the simple onset is 1.5.
    result = check_tracking(wp,gt,beat_factors=np.array([3.,3.,1.]))
    assert result['tracked']
    assert result['worst_window']['deviation'] == pytest.approx(.8)


def test_factor_follows_ground_truth_not_incorrect_predicted_region():
    meter = score_meter_map(score())
    gt = np.array([[0.,6.1],[1.,6.2],[2.,6.3]])
    wp = gt.copy();wp[:,1] -= 2.
    # Estimates lie back in compound meter, but the reference is now in 4/4.
    assert not check_tracking(wp,gt,beat_factors=musical_beat_factors(gt[:,1],meter))['tracked']


def test_missing_initial_estimate_stays_infinite_after_grouping():
    gt = np.array([[0.,0.],[1.,1.],[2.,2.]])
    wp = np.array([[2.,2.]])
    result = check_tracking(wp,gt,beat_factors=np.full(3,3.))
    assert not result['tracked']
    assert np.isinf(result['max_deviation'])


def test_duplicate_reports_keep_the_last_causal_decision():
    gt = np.array([[0.,0.],[1.,1.],[2.,2.]])
    wp = np.array([[0.,50.],[0.,0.],[1.,50.],[1.,1.],[2.,2.]])
    assert check_tracking(wp,gt,threshold=.1,beat_factors=np.full(3,3.))['tracked']


@pytest.mark.parametrize('bad', [np.ones(2),np.array([1.,0.,3.]),np.array([1.,np.nan,3.])])
def test_invalid_factors_fail_loudly(bad):
    gt = np.array([[0.,0.],[1.,1.],[2.,2.]])
    with pytest.raises(ValueError, match='beat_factors'):
        check_tracking(gt,gt,beat_factors=bad)


def test_modality_thresholds():
    assert TRACKING_THRESHOLDS == {'audio':1.,'midi':.1}
