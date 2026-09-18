"""Soft-min Online Time Warping (Soft-OLTW) with IMM Tempo Tracking."""

import numpy as np
from matchmaker import register_method
from matchmaker.dp.oltw_soft import SoftOnlineTimeWarping
from matchmaker.utils.misc import generate_score_audio


def build_reference(mm):
    """Features of a synthesised rendering of the score."""
    score_audio = generate_score_audio(mm.score_part, mm.tempo, mm.sample_rate)
    features, _ = mm.processor((score_audio.astype(np.float32), 0.0))
    mm.processor.reset()
    return features


def build_follower(mm):
    """Build a SoftOnlineTimeWarping follower for one piece."""
    return SoftOnlineTimeWarping(
        reference_features=mm.reference_features,
        score_positions=mm.score_positions,
        queue=mm.stream.queue,
        frame_rate=mm.frame_rate,
        ref_frame_to_beat=mm.ref_frame_to_beat(),
        score_part=mm.score_part,
        tempo=mm.tempo,
        window_size=mm.config.get("window_size", 10),
        step_size=mm.config.get("step_size", 3),
    )


register_method(
    "soft-oltw",
    input_type="audio",
    build_follower=build_follower,
    build_reference=build_reference,
    default_kwargs={
        "processor": "chroma",
        "window_size": 10,
        "step_size": 3,
    },
)
