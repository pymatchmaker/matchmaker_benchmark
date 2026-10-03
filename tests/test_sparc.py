"""SPARC must rank smoothness, ignore a piece's length and give no score to a still path."""

import numpy as np

from utils import compute_sparc

FPS = 30.0


def path(seconds, position):
    times = np.arange(0.0, seconds, 1.0 / FPS)
    return position(times), times


def test_a_path_that_never_moves_has_no_score():
    assert compute_sparc(*path(120, np.zeros_like)) is None


def test_jitter_lowers_the_score():
    rng = np.random.default_rng(0)
    steady = compute_sparc(*path(120, lambda t: 2 * t))
    jittery = compute_sparc(*path(120, lambda t: 2 * t + rng.normal(0, 0.3, len(t))))
    assert jittery < steady < 0


def test_the_score_does_not_grow_with_the_length_of_the_piece():
    def motion(t):
        return 2 * t + 0.5 * np.sin(2 * np.pi * 0.3 * t)

    short, long = compute_sparc(*path(60, motion)), compute_sparc(*path(600, motion))
    assert abs(short - long) < 0.05


def test_a_backward_jump_counts_by_its_size_not_its_sign():
    forward = compute_sparc(*path(60, lambda t: 2 * t + 20 * (t > 30)))
    backward = compute_sparc(*path(60, lambda t: 2 * t - 20 * (t > 30)))
    assert abs(forward - backward) < 0.05
