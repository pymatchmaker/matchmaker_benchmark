"""Constant-tempo baseline for audio: the floor every audio follower must clear.

The audio counterpart of ``baseline-constant-tempo``. It is the same follower —
it ignores the performance entirely, assumes the performer starts at the
beginning of the score and plays at exactly the notated tempo, and reads its
score position off the clock. What differs is only the stream it is registered
for.

Having the same open-loop follower on both leaderboards makes the two input
types comparable: a MIDI and an audio submission are then measured against the
same no-information reference, not against two different ones.

Like the MIDI baseline it deliberately uses nothing beyond matchmaker and numpy.
"""

import numpy as np
from matchmaker import register_method
from matchmaker.base import OnlineAlignment


class ConstantTempoFollower(OnlineAlignment):
    """Advance through the score at the notated tempo, open-loop."""

    def __init__(self, score_positions, beats_per_second, queue=None, **kwargs):
        super().__init__(score_positions=score_positions, queue=queue, **kwargs)
        self.beats_per_second = float(beats_per_second)
        self.start_beat = float(score_positions[0])

    def get_current_position(self) -> float:
        # current_perf_time is set by the base class before every step().
        return self.start_beat + self.beats_per_second * self.current_perf_time

    def step(self, features) -> None:
        # Keep current_index in sync with the predicted beat: the base class
        # uses it to decide when the score has run out (is_still_following).
        beat = self.get_current_position()
        index = int(np.searchsorted(self.score_positions, beat, side="right")) - 1
        self.current_index = int(np.clip(index, 0, len(self.score_positions) - 1))


def build_follower(mm):
    return ConstantTempoFollower(
        score_positions=mm.score_positions,
        beats_per_second=mm.tempo / 60.0,
        queue=mm.stream.queue,
        reference_features=mm.reference_features,
    )


register_method(
    "baseline-constant-tempo-audio",
    input_type="audio",
    build_follower=build_follower,
    # No build_reference: the follower never looks at the score-side features,
    # so the note array is left as it is rather than paying for a synthesised
    # rendering of the score it would ignore.
    default_kwargs={"processor": "chroma"},
)
