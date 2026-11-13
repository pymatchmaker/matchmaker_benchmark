"""
Check if a tracker followed the score throughout a piece.

Evaluates in the score domain. At every annotated onset it
looks up the last score position emitted at or before that time, then computes
the median absolute beat error in a forward window starting at each unique
annotated performance onset. The window slides onset-wise until one first
reaches the final annotation. A piece fails when any window median exceeds the
selected beat tolerance. Benchmark callers supply ``beat_factors`` from the
reference score: errors are divided by three in compound-meter passages and
by one elsewhere, before the window median is taken. Without the factors,
errors stay in denominator-note beats.

Input:
  wp.tsv   Warping path (tab-separated)
           Column 0 = performance time (seconds), Column 1 = score position (beats)
  gt.tsv   Ground truth (tab-separated)
           Column 0 = perf time (seconds), Column 1 = score position (beats)

Usage:
  python verify_tracking.py --wp output/.../wp_1.tsv --gt output/.../gt_1.tsv
  python verify_tracking.py --wp output/.../wp_1.tsv --gt output/.../gt_1.tsv --save out.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matchmaker.utils.eval import ONSET_WINDOW_DURATION, onset_wise_windows
from partitura.score import TimeSignature

BEAT_ERROR_THRESHOLD = 1.0  # threshold for window median beat error (beats)
SEGMENT_DURATION = ONSET_WINDOW_DURATION  # shared window duration (seconds)


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


def _onset_wise_window_profile(
    times: np.ndarray,
    errors: np.ndarray,
    window_duration: float,
    threshold: float,
) -> list[dict]:
    """Evaluate forward windows starting at successive annotated onsets."""
    if window_duration <= 0:
        raise ValueError("segment_duration/window_duration must be positive")

    evaluable = np.isfinite(times) & ~np.isnan(errors)
    times = np.asarray(times[evaluable], dtype=float)
    errors = np.asarray(errors[evaluable], dtype=float)
    if len(times) == 0:
        return []

    order = np.argsort(times, kind="stable")
    times = times[order]
    errors = errors[order]

    windows = []
    for window in onset_wise_windows(times, window_duration):
        left = window["onset_index_start"]
        right = window["onset_index_stop"]
        deviation = float(np.median(errors[left:right]))
        windows.append(
            {**window, "deviation": deviation, "failed": deviation > threshold}
        )
    return windows


def check_tracking(
    wp: np.ndarray,
    gt: np.ndarray,
    segment_duration: float = SEGMENT_DURATION,
    threshold: float = BEAT_ERROR_THRESHOLD,
    *,
    beat_factors: np.ndarray = None,
) -> dict:
    """
    Check tracking quality using an onset-wise sliding-window median.

    Evaluates in the **score domain** (perf→score): at each GT performance
    onset time, looks up the tracker's last-known score position from the
    alignment path and computes the absolute beat error.

    Parameters
    ----------
    wp : (N, 2) array — col 0: perf time (seconds), col 1: score position
    gt : (M, 2) array — col 0: perf time (seconds), col 1: score position (beats)
    segment_duration : float — window duration W in seconds (default: 30)
    threshold : float — maximum median absolute beat error theta (default: 1.0)
    beat_factors : (M,) array, optional
        Denominator notes per musical beat at each GT score position. The
        benchmark supplies this even for simple meter; without it, errors
        remain in denominator-note units.

    Returns
    -------
    dict with: windows, mean_onsets_per_window, worst_window, max_deviation,
    tracked, reason
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if beat_factors is not None:
        beat_factors = np.asarray(beat_factors, dtype=float)
        if (beat_factors.shape != (len(gt),) or not np.all(np.isfinite(beat_factors))
                or np.any(beat_factors <= 0)):
            raise ValueError("beat_factors must contain one positive finite factor per GT onset")
    unit = "musical_beat" if beat_factors is not None else "denominator_note"
    criterion = {"error_unit": unit, "threshold": float(threshold),
                 "window_duration": float(segment_duration)}
    if wp.size == 0 or gt.size == 0:
        return {
            **criterion,
            "windows": [],
            "n_windows": 0,
            "mean_onsets_per_window": 0.0,
            "n_failed": 1,
            "max_deviation": float("inf"),
            "worst_window": None,
            "tracked": False,
            "reason": "no evaluable alignment or ground-truth onsets",
        }
    wp_score = wp[:, 1].astype(float)
    wp_perf = wp[:, 0].astype(float)  # seconds

    gt_perf = gt[:, 0]
    gt_score = gt[:, 1]

    # Reverse lookup (perf → score): at each GT perf time, find tracker's
    # last-known score position (step-function, no future information).
    sort_idx = np.argsort(wp_perf, kind="stable")
    wp_perf_sorted = wp_perf[sort_idx]
    wp_score_sorted = wp_score[sort_idx]

    unique_times, first_idx = np.unique(wp_perf_sorted, return_index=True)
    reduced_scores = np.empty(len(unique_times))
    for g in range(len(unique_times)):
        start = first_idx[g]
        end = first_idx[g + 1] if g + 1 < len(unique_times) else len(wp_score_sorted)
        reduced_scores[g] = wp_score_sorted[end - 1]  # last (final decision)

    indices = np.searchsorted(unique_times, gt_perf, side="right") - 1
    predicted_score = np.full(len(gt_score), np.nan)
    valid = indices >= 0
    predicted_score[valid] = reduced_scores[indices[valid]]

    errors_all = np.abs(predicted_score - gt_score)  # beats
    errors_all[~valid] = np.inf  # no causal estimate at an annotation
    if beat_factors is not None:
        errors_all = errors_all / beat_factors

    windows = _onset_wise_window_profile(
        gt_perf, errors_all, segment_duration, threshold
    )
    if not windows:
        max_dev = float("inf")
        worst_window = None
        n_failed = 1
        tracked = False
        reason = "no causally evaluable annotated onsets"
    else:
        worst_window = max(windows, key=lambda window: window["deviation"])
        max_dev = float(worst_window["deviation"])
        n_failed = sum(window["failed"] for window in windows)
        tracked = n_failed == 0
        reason = (
            "OK"
            if tracked
            else (
                f"worst {segment_duration:g}s-window median {max_dev:.4f}b "
                f"> {threshold:g}b"
            )
        )

    return {
        **criterion,
        "windows": windows,
        "n_windows": len(windows),
        "mean_onsets_per_window": float(
            np.mean([window["n_onsets"] for window in windows])
        ),
        "n_failed": n_failed,
        "max_deviation": round(max_dev, 4),
        "worst_window": worst_window,
        "tracked": tracked,
        "reason": reason,
    }


def _auto_title(wp_path: Path) -> str:
    """Generate title from wp path (e.g. 'dixon #11')."""
    idx = wp_path.stem.replace("wp_", "")
    parent = (
        wp_path.parent.parent.name
        if wp_path.parent.name.isdigit()
        else wp_path.parent.name
    )
    method = ""
    if "[best-" in parent:
        method = parent.split("[best-")[1].split("-")[0]
    return f"{method} #{idx}" if method else f"#{idx}"


def plot_tracking(
    wp: np.ndarray,
    gt: np.ndarray,
    title: str = "",
    save_path: Path = None,
    segment_duration: float = SEGMENT_DURATION,
    threshold: float = BEAT_ERROR_THRESHOLD,
    *,
    beat_factors: np.ndarray = None,
):
    """Plot the alignment path and onset-wise window median-error profile."""
    result = check_tracking(
        wp,
        gt,
        segment_duration=segment_duration,
        threshold=threshold,
        beat_factors=beat_factors,
    )
    windows = result["windows"]
    worst_window = result["worst_window"]
    max_dev = result["max_deviation"]
    tracked = result["tracked"]

    wp_score = wp[:, 1].astype(float)
    wp_perf = wp[:, 0].astype(float)  # seconds
    gt_perf = gt[:, 0]
    gt_score = gt[:, 1]

    # Reverse lookup (perf → score, same as check_tracking)
    sort_idx = np.argsort(wp_perf, kind="stable")
    wp_pf_sorted = wp_perf[sort_idx]
    wp_sc_sorted = wp_score[sort_idx]
    unique_times, first_idx = np.unique(wp_pf_sorted, return_index=True)
    reduced_scores = np.empty(len(unique_times))
    for g in range(len(unique_times)):
        start = first_idx[g]
        end = first_idx[g + 1] if g + 1 < len(unique_times) else len(wp_sc_sorted)
        reduced_scores[g] = wp_sc_sorted[end - 1]
    indices = np.searchsorted(unique_times, gt_perf, side="right") - 1
    predicted_score = np.full(len(gt_score), np.nan)
    valid_idx = indices >= 0
    predicted_score[valid_idx] = reduced_scores[indices[valid_idx]]

    # Figure (3-panel)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.5], hspace=0.35)
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]

    # Panel 1: alignment path + GT; highlight the worst window only on failure.
    if not tracked and worst_window is not None:
        ax1.axvspan(
            worst_window["t0"],
            worst_window["t1"],
            color="red",
            alpha=0.15,
        )
    ax1.scatter(gt_perf, gt_score, c="red", s=12, alpha=0.5, zorder=2, label="GT")
    ax1.scatter(
        wp_perf,
        wp_score,
        color="navy",
        s=3,
        alpha=0.65,
        zorder=3,
        label="Tracker",
    )
    # Error lines: vertical (same perf time, different score positions)
    for gs_val, gp_val, ps_val in zip(gt_score, gt_perf, predicted_score):
        if np.isfinite(ps_val):
            ax1.plot(
                [gp_val, gp_val], [gs_val, ps_val], color="red", alpha=0.15, lw=0.5
            )
    status = "TRACKED" if tracked else f"FAILED ({result['reason']})"
    ax1.set_title(
        f"{title}  —  {status}  [{segment_duration:.0f}s onset-wise median]",
        fontsize=14,
        color="green" if tracked else "red",
        fontweight="bold",
    )
    ax1.set_ylabel("Score position (denominator-note beats)")
    ax1.set_xlabel("Performance time (s)")
    ax1.legend(loc="upper left")

    # Panel 2: median absolute error for successive onset-wise windows.
    starts = [window["t0"] for window in windows]
    deviations = [window["deviation"] for window in windows]
    ax2.plot(starts, deviations, color="navy", lw=1.0)
    ax2.axhline(
        threshold, color="red", ls="--", lw=1.5, label=f"threshold={threshold:.1f}b"
    )
    ax2.axhline(0, color="black", lw=0.5)
    beat_label = "musical beats" if beat_factors is not None else "denominator-note beats"
    ax2.set_ylabel(f"Sliding median error ({beat_label})")
    yl = max(threshold * 2, max_dev * 1.3) if max_dev > 0 else threshold * 2
    ax2.set_ylim(0, yl)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}b"))
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: avoid highlighting a sub-threshold "worst" window on success.
    ax3.axis("off")
    if tracked:
        headers = [
            "Windows",
            "Window duration",
            "Threshold",
            "Failed windows",
            "Status",
        ]
        rows = [
            [
                str(len(windows)),
                f"{segment_duration:g}s",
                f"{threshold:g}b",
                "0",
                "TRACKED",
            ]
        ]
    elif worst_window is not None:
        headers = ["Windows", "Worst interval", "N pts", "Worst median", "Status"]
        rows = [
            [
                str(len(windows)),
                f"{worst_window['t0']:.2f}-{worst_window['t1']:.2f}s",
                str(worst_window["n_points"]),
                f"{worst_window['deviation']:.2f}b",
                "FAILED",
            ]
        ]
    else:
        headers = [
            "Windows",
            "Window duration",
            "Threshold",
            "Failed windows",
            "Status",
        ]
        rows = [["0", f"{segment_duration:g}s", f"{threshold:g}b", "1", "FAILED"]]
    tbl = ax3.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    if not tracked:
        for j in range(len(headers)):
            tbl[1, j].set_facecolor("#ffcccc")
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#e0e0e0")
        tbl[0, j].set_text_props(fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Check tracking with an onset-wise sliding-window median"
    )
    parser.add_argument("--wp", type=Path, required=True, help="Warping path TSV")
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth TSV")
    parser.add_argument(
        "--meter", type=Path, default=None,
        help="Saved meter_<index>.json from the benchmark; applies musical-beat TR",
    )
    parser.add_argument(
        "--save", type=Path, default=None, help="Save plot to file (default: show)"
    )
    parser.add_argument(
        "--window-duration",
        type=float,
        default=SEGMENT_DURATION,
        help="Sliding-window duration W in seconds (default: 30)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=BEAT_ERROR_THRESHOLD,
        help="Median beat-error tolerance theta (default: 1.0)",
    )
    args = parser.parse_args()

    wp = np.loadtxt(args.wp, delimiter="\t", skiprows=1, ndmin=2)
    gt = np.loadtxt(args.gt, delimiter="\t", skiprows=1, ndmin=2)
    factors = None
    if args.meter is not None:
        import json
        factors = musical_beat_factors(gt[:, 1], json.loads(args.meter.read_text()))

    result = check_tracking(
        wp,
        gt,
        segment_duration=args.window_duration,
        threshold=args.threshold,
        beat_factors=factors,
    )
    title = _auto_title(args.wp)

    # Print result
    if result["tracked"]:
        print(
            f"{title}  Tracked: True  |  "
            f"{result['n_windows']} onset-wise windows evaluated  |  OK"
        )
    else:
        print(
            f"{title}  Tracked: False  |  max_dev: "
            f"{result['max_deviation']:.2f}b  |  {result['reason']}"
        )
    worst = result["worst_window"]
    if not result["tracked"] and worst is not None:
        print(
            f"  Worst [{worst['t0']:.3f}-{worst['t1']:.3f}s]: "
            f"{worst['deviation']:.2f}b ({worst['n_points']} pts); "
            f"{result['n_windows']} onset-wise windows evaluated"
        )

    plot_tracking(
        wp,
        gt,
        title=title,
        save_path=args.save,
        segment_duration=args.window_duration,
        threshold=args.threshold,
        beat_factors=factors,
    )


if __name__ == "__main__":
    main()
