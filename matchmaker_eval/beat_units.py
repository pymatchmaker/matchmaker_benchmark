"""Musical-beat tolerances on the benchmark's denominator-note coordinates.

Follower outputs and position-accuracy metrics keep their existing coordinate
unit. Only the TR error is divided by the number of denominator notes in the
musical beat prevailing at the ground-truth score position. The meter map is
built from the very same unfolded score used by the follower, so repeats and
time-signature changes are retained without changing the follower's inputs.
"""

import numpy as np
from partitura.score import TimeSignature

TRACKING_THRESHOLDS = {"audio": 1.0, "midi": 0.1}
TRACKING_ERROR_UNIT = "musical_beat"
POSITION_ERROR_UNIT = "denominator_note"


def denominator_notes_per_beat(numerator: int) -> int:
    """Three in compound meters (6, 9, 12, ...); one in other meters."""
    if numerator <= 0 or int(numerator) != numerator:
        raise ValueError("time-signature numerator must be a positive integer")
    return 3 if numerator >= 6 and numerator % 3 == 0 else 1


def score_meter_map(part) -> dict:
    """Serializable meter changes in the follower's score-beat coordinates.

    ``part`` must be the native, denominator-note score part passed to the
    follower. Do not call ``use_musical_beat()`` on it: that would also change
    its note array and the algorithm's behavior, rather than just evaluation.
    A score without time signatures follows Partitura's 4/4 default, explicitly
    recorded in the map instead of being mistaken for annotated meter.
    """
    if getattr(part, "_use_musical_beat", False):
        raise ValueError("TR expects score positions in denominator-note beats")
    signatures = list(part.iter_all(TimeSignature))
    changes = []
    for ts in sorted(signatures, key=lambda ts: ts.start.t):
        if ts.beat_type <= 0:
            raise ValueError("time-signature denominator must be positive")
        change = {
            "start_score_beat": float(part.beat_map(ts.start.t)),
            "numerator": int(ts.beats),
            "denominator": int(ts.beat_type),
            "denominator_notes_per_beat": denominator_notes_per_beat(ts.beats),
        }
        if changes and change["start_score_beat"] == changes[-1]["start_score_beat"]:
            if change != changes[-1]:
                raise ValueError("conflicting time signatures at one score position")
            continue
        changes.append(change)
    if not changes:
        changes = [{"start_score_beat": 0.0, "numerator": 4, "denominator": 4,
                    "denominator_notes_per_beat": 1}]
    return {
        "schema": 1,
        "position_unit": POSITION_ERROR_UNIT,
        "tracking_error_unit": TRACKING_ERROR_UNIT,
        "assumed_time_signature": not bool(signatures),
        "changes": changes,
    }


def musical_beat_factors(score_positions, meter: dict) -> np.ndarray:
    """The reference meter's factor at each position, right-inclusive at changes.

    Positions need not increase: the same lookup also applies when a performed
    repeat returns to an earlier score position. Pickups before the first
    explicit signature use that first signature.
    """
    positions = np.asarray(score_positions, dtype=float)
    if not np.all(np.isfinite(positions)):
        raise ValueError("ground-truth score positions must be finite")
    if meter.get("position_unit") != POSITION_ERROR_UNIT:
        raise ValueError("meter map must use denominator-note score positions")
    changes = meter["changes"]
    starts = np.asarray([r["start_score_beat"] for r in changes], dtype=float)
    factors = np.asarray([r["denominator_notes_per_beat"] for r in changes], dtype=float)
    if (len(starts) == 0 or not np.all(np.isfinite(starts))
            or np.any(np.diff(starts) <= 0) or not np.all(np.isin(factors, [1., 3.]))):
        raise ValueError("meter map must have ordered, finite changes and factors 1 or 3")
    indices = np.maximum(np.searchsorted(starts, positions, side="right") - 1, 0)
    return factors[indices]
