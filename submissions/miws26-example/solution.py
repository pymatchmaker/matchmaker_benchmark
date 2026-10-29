"""MIWS26 example: on-line time warping, written as a submission.

A plain-Python version of on-line time warping (Dixon, 2005) in the
Arzt-style variant behind matchmaker's built-in ``arzt`` audio method: one
column per input frame, a search window around the current position and a
step-size constraint, with no tempo model. It is written from scratch so that
the whole algorithm reads in one ``step()``, and follows
HOW_TO_MAKE_CUSTOM_SCORE_FOLLOWERS.md in the matchmaker repository:

* the follower subclasses ``OnlineAlignment`` and implements ``step()``, and
  overrides ``get_current_position()`` to report frame-level beats;
* the processor is the standard ``chroma`` one, named in ``default_kwargs``;
* ``build_reference`` synthesises the score and runs that processor over it, so
  the follower aligns live chroma frames against score chroma frames.
"""

import time

import numpy as np
from matchmaker import register_method
from matchmaker.base import OnlineAlignment
from matchmaker.utils.misc import generate_score_audio


class OLTWExample(OnlineAlignment):
    def __init__(
        self,
        reference_features,
        score_positions,
        ref_frame_to_beat,
        frame_rate,
        queue=None,
        window_size=10.0,
        step_size=3,
    ):
        super().__init__(reference_features, score_positions, queue)
        self.ref_frame_to_beat = ref_frame_to_beat
        self.window = int(round(window_size * frame_rate))
        self.step_size = step_size
        self.ref_frame = 0
        self.input_index = 0
        self.previous = np.full(len(reference_features), np.inf)
        self.latency_stats = {"total_latency": 0.0, "total_frames": 0}

    def step(self, features) -> None:
        started = time.perf_counter()
        t = self.input_index
        start = max(self.ref_frame - self.window, 0)
        end = min(self.ref_frame + self.window, len(self.reference_features))
        distance = np.abs(self.reference_features[start:end] - features).sum(axis=1)

        column = np.full_like(self.previous, np.inf)
        best_frame, best_cost = self.ref_frame, np.inf
        for j in range(start, end):
            if j == 0:
                predecessor = 0.0 if t == 0 else self.previous[0]
            else:
                predecessor = min(column[j - 1], self.previous[j], self.previous[j - 1])
            column[j] = distance[j - start] + predecessor
            cost = column[j] / (t + j + 1)
            if cost < best_cost:
                best_frame, best_cost = j, cost

        lowest, highest = self.ref_frame, self.ref_frame + self.step_size
        self.ref_frame = min(max(best_frame, lowest), highest)
        self.previous = column
        self.input_index += 1

        beat = self.get_current_position()
        index = np.searchsorted(self.score_positions, beat, side="right") - 1
        self.current_index = max(int(index), 0)

        self.latency_stats["total_latency"] += time.perf_counter() - started
        self.latency_stats["total_frames"] = self.input_index

    def get_current_position(self) -> float:
        return float(self.ref_frame_to_beat[self.ref_frame])


def build_reference(mm):
    """Chroma frames of the score, synthesised at its notated tempo."""
    audio = generate_score_audio(mm.score_part, mm.tempo, mm.sample_rate)
    features, _ = mm.processor((audio.astype(np.float32), 0.0))
    mm.processor.reset()  # the same processor then handles the live input
    return features


def build_follower(mm):
    return OLTWExample(
        reference_features=mm.reference_features,
        score_positions=mm.score_positions,
        ref_frame_to_beat=mm.ref_frame_to_beat(),
        frame_rate=mm.frame_rate,
        queue=mm.stream.queue,
        window_size=mm.config["window_size"],
        step_size=mm.config["step_size"],
    )


register_method(
    "miws26-example",
    input_type="audio",
    build_follower=build_follower,
    build_reference=build_reference,
    # The follower's settings travel with the stream's, so they can be changed
    # through Matchmaker(kwargs=...) without editing this file.
    default_kwargs={
        "processor": "chroma",
        "window_size": 10.0,
        "step_size": 3,
    },
)
