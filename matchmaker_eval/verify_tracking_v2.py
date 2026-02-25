"""
Check if a tracker followed the score throughout a piece (v2).

Divides performance time into fixed 30-second segments.  For each GT
annotation in a segment, interpolates the warping path to get the
tracker's predicted score position, computes |predicted - gt|, and
reports the median of those absolute errors as the segment deviation.
A piece is FAILED if any segment's median error exceeds the threshold.

Key difference from v1:
  - v1: N equal segments, compares median positions
  - v2: Fixed 30-second segments, per-point absolute alignment errors

Score axis conversion (no 0-1 normalization):
  - frame mode (arzt, dixon): wp score frames / frame_rate → seconds
  - state mode (outerhmm): wp state indices → score positions via state_space

Input:
  wp.tsv   Warping path (tab-separated, 1 header row)
           Column 0 = score position (frames or state indices), Column 1 = perf position (frame index)
  gt.tsv   Ground truth (tab-separated, 1 header row)
           Column 0 = perf time (seconds), Column 1 = score position (seconds)

Usage:
  python verify_tracking_v2.py --wp output/.../wp_1.tsv --gt data/gt/valid/gt_1.tsv --frame-rate 30
  python verify_tracking_v2.py --wp output/.../wp_1.tsv --gt data/gt/valid/gt_1.tsv --frame-rate 30 --save out.png
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

SEGMENT_DURATION = 30.0  # seconds
SEGMENT_THRESHOLD = 1.0  # max allowed median |error| per segment (seconds)
MIN_FAILS = 2  # piece fails if >= this many segments exceed SEGMENT_THRESHOLD
MAX_DEV_CAP = 2.0  # piece fails if any segment exceeds this (seconds)


def _wp_to_score(
    wp: np.ndarray,
    frame_rate: float,
    mode: str = "frame",
    state_space: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert wp score axis to the same units as GT score axis.

    Parameters
    ----------
    wp : (N, 2) — col 0: score (frames or state indices), col 1: perf frames
    frame_rate : float
    mode : "frame" or "state"
    state_space : array mapping state index → score position (seconds or beats).
                  Required when mode="state".

    Returns
    -------
    wp_score : (N,) array in GT-compatible units
    """
    if mode == "state" and state_space is not None:
        state_idx = wp[:, 0].astype(int)
        offset = int(state_idx.min())
        mapped = np.clip(state_idx - offset, 0, len(state_space) - 1)
        return state_space[mapped].astype(float)
    else:
        # Frame mode: score frames → seconds
        return wp[:, 0] / frame_rate


def check_tracking(
    wp: np.ndarray,
    gt: np.ndarray,
    frame_rate: float,
    segment_duration: float = SEGMENT_DURATION,
    threshold: float = SEGMENT_THRESHOLD,
    mode: str = "frame",
    state_space: Optional[np.ndarray] = None,
    min_fails: int = MIN_FAILS,
    max_dev_cap: float = MAX_DEV_CAP,
) -> dict:
    """
    Check tracking quality using fixed-duration segments.

    For each GT point in a segment, interpolates the warping path to get
    the tracker's predicted score position and computes the absolute error
    against the GT score position.  No 0-1 normalization — errors are in
    the same units as gt[:, 1] (typically seconds).

    Parameters
    ----------
    wp : (N, 2) array — col 0: score position (frames or state indices), col 1: perf frames
    gt : (M, 2) array — col 0: perf time (seconds), col 1: score position (seconds)
    frame_rate : float — converts frames to seconds
    segment_duration : float — segment length in seconds (default: 30)
    threshold : float — max allowed median absolute error per segment
    mode : "frame" or "state"
        "frame": wp[:, 0] are score frames (÷ frame_rate → seconds)
        "state": wp[:, 0] are HMM state indices (mapped via state_space)
    state_space : optional array mapping state index → score position.
        Required when mode="state".

    Returns
    -------
    dict with: segments, max_deviation, tracked, reason
    """
    wp_perf = wp[:, 1] / frame_rate

    gt_perf = gt[:, 0]
    gt_score = gt[:, 1]

    wp_score = _wp_to_score(wp, frame_rate, mode, state_space)

    # Causal lookup: for each GT time, use the last WP entry at or before that time.
    # GT points before the first WP entry or after the last get NaN → treated as errors.
    indices = np.searchsorted(wp_perf, gt_perf, side="right") - 1
    predicted_score = np.full(len(gt_perf), np.nan)
    valid = indices >= 0
    predicted_score[valid] = wp_score[indices[valid]]
    errors_all = np.abs(predicted_score - gt_score)

    # Cover full performance duration (max of WP and GT), so early tracker death is penalized
    total_dur = max(wp_perf[-1], gt_perf[-1])
    n_segments = max(1, int(np.ceil(total_dur / segment_duration)))
    segments = []

    for s in range(n_segments):
        t0 = s * segment_duration
        t1 = min((s + 1) * segment_duration, total_dur)

        gt_mask = (gt_perf >= t0) & (gt_perf < t1)
        n_points = int(gt_mask.sum())

        if n_points > 0:
            seg_errors = errors_all[gt_mask]
            finite = seg_errors[np.isfinite(seg_errors)]
            dev = float(np.median(finite)) if len(finite) > 0 else np.nan
        else:
            dev = np.nan

        failed = dev > threshold if not np.isnan(dev) else False

        segments.append(
            {
                "t0": t0,
                "t1": t1,
                "n_points": n_points,
                "deviation": dev,
                "failed": failed,
            }
        )

    valid = [seg["deviation"] for seg in segments if not np.isnan(seg["deviation"])]
    max_dev = max(valid) if valid else 0.0
    n_failed = sum(1 for seg in segments if seg["failed"])
    tracked = n_failed < min_fails and max_dev <= max_dev_cap

    if not tracked:
        reasons = []
        if n_failed >= min_fails:
            reasons.append(f"{n_failed}/{n_segments} segments failed (>={min_fails})")
        if max_dev > max_dev_cap:
            reasons.append(f"max_dev={max_dev:.1f}s > cap {max_dev_cap:.0f}s")
        reason = "; ".join(reasons)
    else:
        reason = "OK"

    return {
        "segments": segments,
        "n_segments": n_segments,
        "n_failed": n_failed,
        "max_deviation": max_dev,
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
    frame_rate: float,
    title: str = "",
    save_path: Path = None,
    segment_duration: float = SEGMENT_DURATION,
    threshold: float = SEGMENT_THRESHOLD,
    mode: str = "frame",
    state_space: Optional[np.ndarray] = None,
    min_fails: int = MIN_FAILS,
    max_dev_cap: float = MAX_DEV_CAP,
):
    """Plot warping path vs GT with per-point error analysis per segment."""
    import matplotlib.pyplot as plt

    result = check_tracking(
        wp,
        gt,
        frame_rate,
        segment_duration=segment_duration,
        threshold=threshold,
        mode=mode,
        state_space=state_space,
        min_fails=min_fails,
        max_dev_cap=max_dev_cap,
    )
    segments = result["segments"]
    max_dev = result["max_deviation"]
    tracked = result["tracked"]

    wp_perf = wp[:, 1] / frame_rate
    wp_score = _wp_to_score(wp, frame_rate, mode, state_space)
    gt_perf = gt[:, 0]
    gt_score = gt[:, 1]

    # Causal lookup (same as check_tracking)
    indices = np.searchsorted(wp_perf, gt_perf, side="right") - 1
    predicted_score = np.full(len(gt_perf), np.nan)
    valid_idx = indices >= 0
    predicted_score[valid_idx] = wp_score[indices[valid_idx]]

    # Figure (3-panel)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.5], hspace=0.35)
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]

    # Panel 1: warping path + GT
    for seg in segments:
        ax1.axvspan(
            seg["t0"],
            seg["t1"],
            color="red" if seg["failed"] else "green",
            alpha=0.15 if seg["failed"] else 0.05,
        )
        ax1.axvline(seg["t0"], color="gray", alpha=0.3, ls="--", lw=0.5)
    ax1.axvline(segments[-1]["t1"], color="gray", alpha=0.3, ls="--", lw=0.5)
    ax1.scatter(gt_perf, gt_score, c="red", s=12, alpha=0.5, zorder=2, label="GT")
    ax1.plot(
        wp_perf, wp_score, color="navy", lw=1.2, alpha=0.9, zorder=3, label="Tracker"
    )
    for t, gs_val, ps_val in zip(gt_perf, gt_score, predicted_score):
        ax1.plot([t, t], [gs_val, ps_val], color="red", alpha=0.15, lw=0.5)
    n_failed = result["n_failed"]
    status = "TRACKED" if tracked else f"FAILED ({result['reason']})"
    mf_label = f", min_fails={min_fails}" if min_fails > 1 else ""
    cap_label = f", cap={max_dev_cap:.0f}s" if max_dev_cap < float("inf") else ""
    ax1.set_title(
        f"{title}  —  {status}  [{segment_duration:.0f}s segments{mf_label}{cap_label}]",
        fontsize=14,
        color="green" if tracked else "red",
        fontweight="bold",
    )
    ax1.set_ylabel("Score position (s)")
    ax1.set_xlabel("Performance time (s)")
    ax1.legend(loc="upper left")

    # Panel 2: median absolute error bars
    for seg in segments:
        t_mid = (seg["t0"] + seg["t1"]) / 2
        v = seg["deviation"] if not np.isnan(seg["deviation"]) else 0
        ax2.bar(
            t_mid,
            v,
            width=segment_duration * 0.8,
            color="red" if seg["failed"] else "green",
            alpha=0.7,
            edgecolor="black",
            lw=0.5,
        )
    ax2.axhline(
        threshold, color="red", ls="--", lw=1.5, label=f"threshold={threshold:.0f}s"
    )
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Median |error| per segment (s)")
    yl = max(threshold * 2, max_dev * 1.3)
    ax2.set_ylim(0, yl)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}s"))
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: numeric table
    ax3.axis("off")
    headers = ["Seg", "Time", "N pts", "Med |err|", "Status"]
    rows = []
    for i, seg in enumerate(segments):
        rows.append(
            [
                f"S{i+1}",
                f"{seg['t0']:.0f}-{seg['t1']:.0f}s",
                f"{seg['n_points']}",
                f"{seg['deviation']:.2f}s" if not np.isnan(seg["deviation"]) else "N/A",
                "FAIL" if seg["failed"] else "OK",
            ]
        )
    tbl = ax3.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    for i, seg in enumerate(segments):
        if seg["failed"]:
            for j in range(len(headers)):
                tbl[i + 1, j].set_facecolor("#ffcccc")
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
        description="Check if tracker followed the score (v2: fixed-duration segments)"
    )
    parser.add_argument("--wp", type=Path, required=True, help="Warping path TSV")
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth TSV")
    parser.add_argument("--frame-rate", type=float, required=True, help="Frame rate")
    parser.add_argument(
        "--mode",
        choices=["frame", "state"],
        default="frame",
        help="Warping path mode: 'frame' for arzt/dixon, 'state' for outerhmm",
    )
    parser.add_argument(
        "--state-space",
        type=Path,
        default=None,
        help="State space file (one score position per line). Required for --mode state",
    )
    parser.add_argument(
        "--save", type=Path, default=None, help="Save plot to file (default: show)"
    )
    args = parser.parse_args()

    wp = np.loadtxt(args.wp, delimiter="\t", skiprows=1)
    gt = np.loadtxt(args.gt, delimiter="\t", skiprows=1)

    state_space = None
    if args.state_space is not None:
        state_space = np.loadtxt(args.state_space)

    result = check_tracking(
        wp,
        gt,
        args.frame_rate,
        mode=args.mode,
        state_space=state_space,
    )
    title = _auto_title(args.wp)

    # Print result
    print(
        f"{title}  Tracked: {result['tracked']}  |  max_dev: {result['max_deviation']:.2f}s  |  {result['reason']}"
    )
    for i, seg in enumerate(result["segments"]):
        d = seg["deviation"]
        flag = " !" if not np.isnan(d) and d > SEGMENT_THRESHOLD else ""
        if not np.isnan(d):
            print(
                f"  S{i+1:>2} [{seg['t0']:>5.0f}-{seg['t1']:>5.0f}s]: {d:.2f}s ({seg['n_points']} pts){flag}"
            )
        else:
            print(f"  S{i+1:>2} [{seg['t0']:>5.0f}-{seg['t1']:>5.0f}s]: nan (0 pts)")

    plot_tracking(
        wp,
        gt,
        args.frame_rate,
        title=title,
        save_path=args.save,
        mode=args.mode,
        state_space=state_space,
    )


if __name__ == "__main__":
    main()
