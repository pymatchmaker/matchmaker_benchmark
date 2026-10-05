"""
Check if a tracker followed the score throughout a piece.

Evaluates in the score domain, in beat unit. At every annotated onset it
looks up the last score position emitted at or before that time, then computes
the median absolute beat error in a forward window starting at each unique
annotated performance onset. The window slides onset-wise until one first
reaches the final annotation. A piece fails when any window median exceeds the
selected beat tolerance.

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

BEAT_ERROR_THRESHOLD = 1.0  # threshold for window median beat error (beats)
SEGMENT_DURATION = 30.0  # window duration (seconds)


def musical_beat_converter(score_part):
    """Map score beats to musical beats for the tracking criterion.

    Score beats follow the time signature's denominator (an eighth in 6/8), so
    the same deviation threshold is stricter in compound meters. The tracking
    criterion is applied in musical beats instead (a dotted quarter in 6/8 and
    12/8, a bar in 3/8; simple meters are unchanged). The map is built from the
    note onsets of a copy of ``score_part``, so time-signature changes are
    followed and the part itself is left as it is; positions between onsets are
    interpolated and positions beyond them extrapolated at the end slopes.
    """
    import copy

    part = copy.deepcopy(score_part)
    notated = part.note_array()["onset_beat"].astype(float)
    part.use_musical_beat()
    musical = part.note_array()["onset_beat"].astype(float)
    order = np.argsort(notated, kind="stable")
    notated, idx = np.unique(notated[order], return_index=True)
    musical = musical[order][idx]

    def convert(beats):
        beats = np.asarray(beats, dtype=float)
        if len(notated) < 2:
            return beats.copy()
        out = np.interp(beats, notated, musical)
        lo, hi = beats < notated[0], beats > notated[-1]
        out[lo] = musical[0] + (beats[lo] - notated[0]) * (musical[1] - musical[0]) / (notated[1] - notated[0])
        out[hi] = musical[-1] + (beats[hi] - notated[-1]) * (musical[-1] - musical[-2]) / (notated[-1] - notated[-2])
        return out

    return convert


def in_musical_beats(path: np.ndarray, convert) -> np.ndarray:
    """An (N, 2) array of (perf_sec, score_beat) with its beats converted."""
    path = np.array(path, dtype=float, copy=True)
    path[:, 1] = convert(path[:, 1])
    return path


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
    final_onset = float(times[-1])
    unique_starts = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
    for left in unique_starts:
        start = float(times[left])
        nominal_stop = start + window_duration
        right = int(np.searchsorted(times, nominal_stop, side="right"))
        reaches_final_onset = right == len(times)
        deviation = float(np.median(errors[left:right]))
        n_onsets = int(
            np.count_nonzero(
                np.r_[True, times[left + 1 : right] != times[left : right - 1]]
            )
        )
        windows.append(
            {
                "t0": start,
                "t1": min(nominal_stop, final_onset),
                "onset_index_start": int(left),
                "onset_index_stop": right,
                "n_points": right - left,
                "n_onsets": n_onsets,
                "deviation": deviation,
                "failed": deviation > threshold,
                "short_piece_fallback": left == 0 and reaches_final_onset,
                "reaches_final_onset": reaches_final_onset,
            }
        )
        if reaches_final_onset:
            break
    return windows


def check_tracking(
    wp: np.ndarray,
    gt: np.ndarray,
    segment_duration: float = SEGMENT_DURATION,
    threshold: float = BEAT_ERROR_THRESHOLD,
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

    Returns
    -------
    dict with: windows, mean_onsets_per_window, worst_window, max_deviation,
    tracked, reason
    """
    if wp.size == 0 or gt.size == 0:
        return {
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
):
    """Plot the alignment path and onset-wise window median-error profile."""
    result = check_tracking(
        wp,
        gt,
        segment_duration=segment_duration,
        threshold=threshold,
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
    ax1.set_ylabel("Score position")
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
    ax2.set_ylabel("Sliding median |beat error|")
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

    wp = np.loadtxt(args.wp, delimiter="\t", skiprows=1)
    gt = np.loadtxt(args.gt, delimiter="\t", skiprows=1)

    result = check_tracking(
        wp,
        gt,
        segment_duration=args.window_duration,
        threshold=args.threshold,
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
    )


if __name__ == "__main__":
    main()
