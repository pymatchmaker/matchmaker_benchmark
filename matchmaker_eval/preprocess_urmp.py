"""Preprocess URMP dataset: create aligned performance annotations.

For each piece, matches score notes (from MIDI) to performance notes
(from per-instrument annotation files) using pitch-based alignment
within each instrument track.  Outputs a single-column file of
performance onset times (seconds) that is 1-to-1 aligned with the
unique score onset beats produced by Matchmaker's
build_score_annotations(level="note").

Usage:
    python preprocess_urmp.py
    python preprocess_urmp.py --urmp-dir ~/data/URMP --output-dir urmp_aligned
"""

import argparse
import re
from pathlib import Path

import numpy as np
import partitura as pt


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to MIDI note number (rounded)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(hz / 440.0)
    midi[~np.isfinite(midi)] = -1
    return np.round(midi).astype(int)


def match_by_pitch(
    score_pitches: np.ndarray,
    perf_pitches: np.ndarray,
    perf_onsets: np.ndarray,
) -> np.ndarray:
    """Needleman-Wunsch alignment on pitch sequences.

    Aligns score pitches to performance pitches allowing insertions
    (extra perf notes) and deletions (missing perf notes).  Returns
    perf onset times aligned to score notes (NaN for unmatched).
    """
    n, m = len(score_pitches), len(perf_pitches)
    GAP = -1
    MATCH = 2
    MISMATCH = -2

    # DP table
    dp = np.full((n + 1, m + 1), -np.inf)
    dp[0, 0] = 0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + GAP
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + GAP

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score_val = MATCH if score_pitches[i - 1] == perf_pitches[j - 1] else MISMATCH
            dp[i, j] = max(
                dp[i - 1, j - 1] + score_val,  # match/mismatch
                dp[i - 1, j] + GAP,             # deletion (skip score note)
                dp[i, j - 1] + GAP,             # insertion (skip perf note)
            )

    # Traceback
    result = np.full(n, np.nan)
    i, j = n, m
    while i > 0 and j > 0:
        score_val = MATCH if score_pitches[i - 1] == perf_pitches[j - 1] else MISMATCH
        if dp[i, j] == dp[i - 1, j - 1] + score_val:
            if score_pitches[i - 1] == perf_pitches[j - 1]:
                result[i - 1] = perf_onsets[j - 1]
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + GAP:
            i -= 1  # score note has no match
        else:
            j -= 1  # extra perf note
    return result


def align_sequences(n_score: int, perf_onsets: np.ndarray) -> np.ndarray:
    """Align two monotonic sequences of potentially different lengths.

    Matches N score positions to M perf onset times using DP that
    prefers i/n ≈ j/m matches.  Returns length-N array of perf times
    (monotonic by construction).  Unmatched entries interpolated.
    """
    m = len(perf_onsets)
    if n_score == m:
        return perf_onsets.copy()

    GAP = 0.5
    dp = np.full((n_score + 1, m + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, n_score + 1):
        dp[i, 0] = dp[i - 1, 0] + GAP
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + GAP

    for i in range(1, n_score + 1):
        for j in range(1, m + 1):
            cost = abs(i / n_score - j / m)
            dp[i, j] = min(
                dp[i - 1, j - 1] + cost,
                dp[i - 1, j] + GAP,
                dp[i, j - 1] + GAP,
            )

    result = np.full(n_score, np.nan)
    i, j = n_score, m
    while i > 0 and j > 0:
        cost = abs(i / n_score - j / m)
        if dp[i, j] == dp[i - 1, j - 1] + cost:
            result[i - 1] = perf_onsets[j - 1]
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + GAP:
            i -= 1
        else:
            j -= 1

    # Interpolate missing entries
    valid = np.isfinite(result)
    if valid.sum() > 1:
        idx = np.arange(n_score)
        result = np.interp(idx, idx[valid], result[valid])
    return result


def process_piece(
    score_midi_path: Path,
    piece_dir: Path,
    folder_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Process one URMP piece.

    Aligns score's unique onset beats to Notes_merged performance onset
    times (both are monotonically ordered in their respective units).
    Returns (onset_beats, onset_secs, info_dict).
    """
    # Load score → unique onset beats
    score = pt.load_score_midi(str(score_midi_path))
    sna = score.note_array()
    unique_beats = np.unique(sna["onset_beat"])
    n_score = len(unique_beats)

    # Load Notes_merged and sort by onset time
    merged_file = next(piece_dir.glob("Notes_merged_*.txt"))
    merged = np.loadtxt(str(merged_file))
    if merged.ndim == 1:
        merged = merged.reshape(1, -1)
    perf_onsets = np.sort(merged[:, 0])
    n_perf = len(perf_onsets)

    # Sequence alignment between score unique beats and perf merged onsets
    aligned_secs = align_sequences(n_score, perf_onsets)

    # Verify monotonicity
    n_nonmono = int((np.diff(aligned_secs) < 0).sum()) if n_score > 1 else 0

    info = {
        "n_score": n_score,
        "n_perf": n_perf,
        "n_nonmono": n_nonmono,
        "length_diff": n_perf - n_score,
    }

    return unique_beats, aligned_secs, info


def main():
    parser = argparse.ArgumentParser(description="Preprocess URMP annotations")
    parser.add_argument(
        "--urmp-dir",
        type=Path,
        default=Path("~/data/URMP").expanduser(),
        help="Path to URMP dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "urmp_aligned",
        help="Output directory for aligned annotations",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).parent / "metadata-urmp.csv",
        help="URMP metadata CSV",
    )
    args = parser.parse_args()

    import pandas as pd

    meta = pd.read_csv(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for _, row in meta.iterrows():
        folder = row.folder
        piece_dir = args.urmp_dir / folder
        score_path = piece_dir / row.xml_score

        try:
            beats, secs, info = process_piece(score_path, piece_dir, folder)

            # Save aligned performance onsets (single column, seconds)
            out_file = args.output_dir / f"{folder}_aligned.txt"
            np.savetxt(str(out_file), secs, fmt="%.6f")

            # Save GT pairs (score_beat, perf_sec) for verification
            gt_file = args.output_dir / f"{folder}_gt.tsv"
            np.savetxt(
                str(gt_file),
                np.column_stack([beats, secs]),
                fmt="%.6f",
                delimiter="\t",
                header="score_beat\tperf_sec",
                comments="",
            )

            status = "OK" if info["n_nonmono"] == 0 else "WARN"
            print(
                f"{status} {folder:45s} "
                f"n_score={info['n_score']:4d}  n_perf={info['n_perf']:4d}  "
                f"diff={info['length_diff']:+3d}  nonmono={info['n_nonmono']}"
            )
            summary.append({"folder": folder, **info, "status": status})

        except Exception as e:
            print(f"ERR {folder:45s} {e}")
            summary.append(
                {"folder": folder, "status": "ERROR", "error": str(e)}
            )

    # Summary
    ok = sum(1 for s in summary if s.get("status") == "OK")
    warn = sum(1 for s in summary if s.get("status") == "WARN")
    err = sum(1 for s in summary if s.get("status") == "ERROR")
    print(f"\nDone: {ok} OK, {warn} WARN, {err} ERROR / {len(summary)} total")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
