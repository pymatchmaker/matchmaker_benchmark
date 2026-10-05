"""A small but real score follower: match pitches forward, coast on tempo.

The idea, in two sentences. Keep a pointer into the list of score chords; when a
note arrives, look a few chords ahead for one that contains that pitch and move
the pointer there. Between those jumps, keep the reported position moving at the
tempo estimated from the last two jumps, so the follower does not freeze during
rests or notes it fails to match.

This is deliberately naive — it has no notion of note durations, cannot go
backwards, and a single wrong jump is permanent. It is here to show what a
submission looks like end to end, and to give a target that a serious method
should beat.
"""

import numpy as np
from matchmaker import register_method
from matchmaker.base import OnlineAlignment
from matchmaker.features.midi import PitchProcessor

PIANO_OFFSET = 21  # MIDI pitch of A0; piano-range features are pitch - 21


class PitchMatchFollower(OnlineAlignment):
    """Forward pitch matching with a tempo-interpolated position.

    Parameters
    ----------
    chord_pitches : list[set[int]]
        Piano-range pitches sounding at each score onset, one entry per
        entry of ``score_positions``.
    window : int
        How many chords ahead to search for a match. This is the parameter
        that matters: too large and the pointer races ahead on coincidental
        pitch matches, too small and it cannot get past an ornament. Tuned on
        the validation fold — tuning it on a single piece picks the wrong value.
    smoothing : float
        Weight of the previous tempo estimate when a new one arrives.
    """

    def __init__(
        self,
        score_positions,
        chord_pitches,
        beats_per_second,
        queue=None,
        window=2,
        smoothing=0.5,
        **kwargs,
    ):
        super().__init__(score_positions=score_positions, queue=queue, **kwargs)
        self.chord_pitches = chord_pitches
        self.window = int(window)
        self.smoothing = float(smoothing)
        self.beats_per_second = float(beats_per_second)
        self._anchor_beat = float(score_positions[0])
        self._anchor_time = 0.0

    def step(self, features) -> None:
        observed = {int(p) for p in np.atleast_1d(features)}
        if not observed:
            return

        # Take the first chord ahead that shares a pitch with what we just
        # heard. The lookahead has to stay small: music reuses every pitch
        # constantly, so "the next chord containing this note" is only
        # trustworthy within a note or two. On the validation fold, a lookahead of
        # 2 tracks 14 of 20 pieces; 3 tracks 8; 5 tracks none, because the
        # pointer starts jumping on coincidental matches and never recovers.
        end = min(self.current_index + 1 + self.window, len(self.chord_pitches))
        for candidate in range(self.current_index + 1, end):
            if self.chord_pitches[candidate] & observed:
                self._anchor(candidate)
                return
        # Nothing matched — an ornament, a wrong note, or a passage we already
        # lost. Hold the pointer; get_current_position keeps the estimate
        # moving at the current tempo.

    def _anchor(self, index: int) -> None:
        """Pin the position to a matched chord and re-estimate the tempo."""
        beat = float(self.score_positions[index])
        elapsed = self.current_perf_time - self._anchor_time
        advanced = beat - self._anchor_beat

        # Ignore implausible estimates: two notes 3 ms apart say nothing about
        # tempo, and a huge jump is more likely a mismatch than a fast passage.
        if elapsed > 0.05 and 0 < advanced < 8:
            rate = advanced / elapsed
            self.beats_per_second = (
                self.smoothing * self.beats_per_second + (1 - self.smoothing) * rate
            )

        self.current_index = index
        self._anchor_beat = beat
        self._anchor_time = self.current_perf_time

    def get_current_position(self) -> float:
        """Interpolate past the last matched chord at the estimated tempo.

        Without this the follower would report a staircase that stalls
        whenever a note fails to match, and the beat error would be dominated
        by those stalls rather than by the matching itself.
        """
        drift = self.beats_per_second * (self.current_perf_time - self._anchor_time)
        return self._anchor_beat + drift


def build_processor(mm):
    # return_pitch_list gives MIDI pitch indices instead of a one-hot vector,
    # which is what the matching in step() wants. The standard processors are
    # named by string in default_kwargs; this one needs an argument they do not
    # expose, so it is built here.
    return PitchProcessor(piano_range=True, return_pitch_list=True)


def build_reference(mm):
    """Pitches sounding at each score onset, ordered like mm.score_positions."""
    note_array = mm.score_part.note_array()
    onsets = note_array["onset_beat"]
    pitches = note_array["pitch"] - PIANO_OFFSET
    return [set(pitches[onsets == onset]) for onset in np.unique(onsets)]


def build_follower(mm):
    return PitchMatchFollower(
        score_positions=mm.score_positions,
        chord_pitches=mm.reference_features,
        beats_per_second=mm.tempo / 60.0,
        queue=mm.stream.queue,
        reference_features=mm.reference_features,
    )


register_method(
    "example-pitch-matcher",
    input_type="midi",
    build_follower=build_follower,
    build_processor=build_processor,
    build_reference=build_reference,
    # Group note-ons within 50 ms into one observation, so a chord arrives as
    # one set of pitches rather than as three separate notes.
    default_kwargs={"polling_period": 0.05},
)
