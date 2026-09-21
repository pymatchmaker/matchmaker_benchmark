import numpy as np

from matchmaker.dp.oltw_soft import SoftOnlineTimeWarping
from matchmaker_eval.eval import save_imm_diagnostics


def test_diagnostics_match_online_priors_and_skip_initial_silence(tmp_path):
    features = np.eye(12, dtype=np.float32)[np.arange(80) % 12]
    follower = SoftOnlineTimeWarping(features, ref_frame_to_beat=np.arange(80) / 10,
                                    frame_rate=10, path_tempo=True)
    follower.kalman.pause_ranges = [(0.0, 1.0)]
    observations = np.vstack((np.zeros((3, 12), dtype=np.float32), features[:20]))
    priors, posteriors, times = [], [], []
    for frame, observation in enumerate(observations):
        time = frame / 10 + 2
        follower(observation, time)
        times.append(time)
        if follower._music_started:
            priors.append(follower.kalman.c_bar.copy())
            posteriors.append(follower.kalman.mu.copy())
    path = tmp_path / 'imm.tsv'
    save_imm_diagnostics(follower, np.array(times), path)
    saved = np.loadtxt(path, skiprows=1)
    np.testing.assert_allclose(saved[:, 0], times[3:])
    np.testing.assert_allclose(saved[:, 4:7], priors)
    np.testing.assert_allclose(saved[:, 7:10], posteriors)
    np.testing.assert_allclose(saved[:, 2], follower.alignment_path[1, 3:])
