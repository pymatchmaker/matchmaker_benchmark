from types import SimpleNamespace
import numpy as np
import pytest
from matchmaker_eval.run_ophmm_playthrough import controlled_factory

@pytest.mark.parametrize('condition', ['on', 'off'])
def test_only_global_factor_changes(condition):
    f = SimpleNamespace(n_states=3, S=np.ones(3)/3, r=np.ones(3)/3,
                        alpha=np.eye(3), b_table=np.arange(9).reshape(3, 3))
    original = {k: v.copy() for k, v in vars(f).items() if isinstance(v, np.ndarray)}
    seen = []
    mm = SimpleNamespace(score_follower=f)
    factory = controlled_factory(condition, seen, base=lambda **kw: mm)
    assert factory(method='outerhmm', input_type='midi', performance_file='p.mid') is mm
    for key in ('r', 'alpha', 'b_table'):
        np.testing.assert_array_equal(getattr(f, key), original[key])
    np.testing.assert_array_equal(f.S, np.zeros(3) if condition == 'off' else original['S'])
    assert seen[0]['S_nonzero'] == (0 if condition == 'off' else 3)

def test_rejects_other_methods():
    with pytest.raises(ValueError, match='restricted'):
        controlled_factory('off', [], base=lambda **kw: None)(method='arzt', input_type='midi')
