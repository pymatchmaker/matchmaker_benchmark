"""Template submission — copy this directory, rename it, and edit.

    cp -r submissions/_template submissions/your-name

A submission has two halves, the same two every score follower has:

1. a **follower** — a ``matchmaker.base.OnlineAlignment`` subclass that turns a
   stream of observations into a score position;
2. a **processor** — turns raw MIDI messages or audio frames into the features
   your follower consumes. Naming a standard one in ``default_kwargs`` is
   usually enough.

You then call ``register_method`` once, and everything else — score loading,
streaming, evaluation — is done for you and is identical to what the built-in
methods get. Nothing here is benchmark-specific: this file works with plain
matchmaker too.

Test it without touching a dataset:

    python matchmaker_eval/validate_submission.py submissions/your-name --smoke

Then tune on the validation fold (never the eval fold — see docs/eval-protocol.md):

    python matchmaker_eval/run_submission.py submissions/your-name --fold valid
"""

from matchmaker import register_method
from matchmaker.base import OnlineAlignment


class MyFollower(OnlineAlignment):
    """Replace this with your score follower.

    The base class gives you:

    ``self.current_index``      index into ``score_positions``; you update it
    ``self.current_perf_time``  seconds since the performance started
    ``self.score_positions``    score beat of each state (ascending)
    ``self.reference_features`` the score-side features

    and takes care of ``alignment_path``, ``run()`` and termination. See
    HOW_TO_MAKE_CUSTOM_SCORE_FOLLOWERS.md in the matchmaker repository.
    """

    def step(self, features) -> None:
        """Consume one observation and update the current score position.

        Called once per incoming frame (audio) or note/chord (MIDI). It must
        return quickly — a real-time follower has one frame period to work in,
        and the benchmark reports your real-time factor.
        """
        self.current_index = min(self.current_index + 1, len(self.score_positions) - 1)

    def get_current_position(self) -> float:
        """Optional: report a position between score states.

        The default snaps to ``score_positions[current_index]``. Override it if
        your follower can interpolate — the metrics are in beats, so
        sub-state precision shows up directly in the score.
        """
        return float(self.score_positions[self.current_index])


def build_follower(mm):
    """Build a fresh follower for one piece.

    ``mm`` is the ``Matchmaker`` running this piece. It carries the score side
    of the problem — ``mm.score_positions``, ``mm.reference_features``,
    ``mm.score_part``, ``mm.tempo``, ``mm.frame_rate``, ``mm.config`` — and the
    stream's ``mm.stream.queue``.

    It deliberately gives you no way to read the performance ahead of time: the
    observations arrive one at a time through the queue, which is what online
    score following means.
    """
    return MyFollower(
        score_positions=mm.score_positions,
        queue=mm.stream.queue,
        reference_features=mm.reference_features,
    )


register_method(
    # The name that identifies your method. Use your submission directory name.
    "your-name",
    # "midi" or "audio" — must match metadata.yaml.
    input_type="midi",
    build_follower=build_follower,
    # Stream and feature settings.
    #   midi:  processor ("pitch", "pianoroll", "chord_onset", "pitchclass"),
    #          piano_range, polling_period (None = one observation per message)
    #   audio: processor ("chroma", "mfcc", "cqt", "lse", ...),
    #          sample_rate, frame_rate (or hop_length)
    # Anything else you put here reaches your follower as mm.config.
    default_kwargs={"processor": "pitch", "piano_range": True},
    #
    # ---- optional hooks -----------------------------------------------------
    #
    # build_processor=lambda mm: MyProcessor(...),
    #     Only needed for a processor of your own; naming a standard one in
    #     default_kwargs above covers most cases.
    #
    # build_reference=build_reference,
    #     Score-side features. Defaults to the partitura note array. Audio
    #     followers usually align against a synthesised score rendering:
    #
    #     def build_reference(mm):
    #         import numpy as np
    #         from matchmaker.utils.misc import generate_score_audio
    #         audio = generate_score_audio(mm.score_part, mm.tempo, mm.sample_rate)
    #         features, _ = mm.processor((audio.astype(np.float32), 0.0))
    #         mm.processor.reset()   # the same processor handles the live input
    #         return features
    #
    #     Frame-based audio followers usually also want mm.ref_frame_to_beat().
)
