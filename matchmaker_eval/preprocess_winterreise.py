"""Preprocess Winterreise dataset: create aligned performance annotations.

Winterreise note annotations are semicolon-delimited CSVs with columns:
    start; end; pitch; pitchclass; instrument
where pitch is MIDI note number.

For each (song, performance) pair, matches score notes (from MusicXML)
to performance notes using pitch-based Needleman-Wunsch alignment,
then outputs a single-column file of performance onset times aligned
with the score's unique onset beats.

Usage:
    python preprocess_winterreise.py
"""

import argparse
import sys
from pathlib import Path

sys.setrecursionlimit(10000)

import numpy as np
import pandas as pd
import partitura as pt
from utils import source_annotation


def match_by_pitch(
    score_pitches: np.ndarray,
    perf_pitches: np.ndarray,
    perf_onsets: np.ndarray,
) -> np.ndarray:
    """Needleman-Wunsch alignment on pitch sequences."""
    n, m = len(score_pitches), len(perf_pitches)
    GAP, MATCH, MISMATCH = -1, 2, -2

    dp = np.full((n + 1, m + 1), -np.inf)
    dp[0, 0] = 0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + GAP
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + GAP

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = MATCH if score_pitches[i - 1] == perf_pitches[j - 1] else MISMATCH
            dp[i, j] = max(
                dp[i - 1, j - 1] + s,
                dp[i - 1, j] + GAP,
                dp[i, j - 1] + GAP,
            )

    result = np.full(n, np.nan)
    i, j = n, m
    while i > 0 and j > 0:
        s = MATCH if score_pitches[i - 1] == perf_pitches[j - 1] else MISMATCH
        if dp[i, j] == dp[i - 1, j - 1] + s:
            if score_pitches[i - 1] == perf_pitches[j - 1]:
                result[i - 1] = perf_onsets[j - 1]
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + GAP:
            i -= 1
        else:
            j -= 1
    return result


def process_piece(score_xml_path: Path, note_csv_path: Path) -> tuple:
    """Match score notes to performance notes and return aligned onset times."""
    # Load score
    score = pt.load_musicxml(str(score_xml_path), ignore_invisible_objects=True)
    score = pt.score.unfold_part_maximal(score, update_ids=False, ignore_leaps=False)
    sna = score.note_array()
    score_pitches = sna["pitch"]
    score_beats = sna["onset_beat"]

    # Load performance note annotations (semicolon-delimited)
    perf_df = pd.read_csv(note_csv_path, sep=";", skipinitialspace=True)
    perf_df.columns = perf_df.columns.str.strip()
    perf_onsets = perf_df["start"].values.astype(float)
    perf_pitches = perf_df["pitch"].values.astype(int)

    # Sort both by onset then pitch for consistent ordering
    s_order = np.lexsort((score_pitches, score_beats))
    score_pitches = score_pitches[s_order]
    score_beats = score_beats[s_order]

    p_order = np.lexsort((perf_pitches, perf_onsets))
    perf_pitches = perf_pitches[p_order]
    perf_onsets = perf_onsets[p_order]

    n_score = len(score_pitches)
    n_perf = len(perf_pitches)

    # Match
    if n_score == n_perf:
        matched_onsets = perf_onsets
    else:
        matched_onsets = match_by_pitch(score_pitches, perf_pitches, perf_onsets)

    # Build pairs and group by unique onset beat
    valid = np.isfinite(matched_onsets)
    pairs = np.column_stack([score_beats[valid], matched_onsets[valid]])
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]

    unique_beats = np.unique(pairs[:, 0])
    aligned_secs = np.empty(len(unique_beats))
    for i, b in enumerate(unique_beats):
        mask = pairs[:, 0] == b
        aligned_secs[i] = pairs[mask, 1].min()

    # Remove non-monotonic points (keep longest monotonically increasing subsequence)
    mono_mask = np.ones(len(unique_beats), dtype=bool)
    max_so_far = -np.inf
    for i, t in enumerate(aligned_secs):
        if t > max_so_far:
            max_so_far = t
        else:
            mono_mask[i] = False
    unique_beats = unique_beats[mono_mask]
    aligned_secs = aligned_secs[mono_mask]

    n_matched = int(valid.sum())
    match_rate = n_matched / n_score if n_score > 0 else 0

    return (
        unique_beats,
        aligned_secs,
        {
            "n_score": n_score,
            "n_perf": n_perf,
            "n_matched": n_matched,
            "match_rate": match_rate,
            "n_unique_beats": len(unique_beats),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess Winterreise annotations")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("~/data/winterreise").expanduser()
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "winterreise_aligned"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).parent / "metadata-winterreise.csv",
    )
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")

    meta = pd.read_csv(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    for _, row in meta.iterrows():
        score_path = args.data_dir / row.xml_score
        note_path = args.data_dir / source_annotation(row, "winterreise")
        label = f"{row.title}_{Path(row.audio_performance).stem.split('_')[-1]}"

        try:
            beats, secs, info = process_piece(score_path, note_path)

            out_file = args.output_dir / f"{label}_aligned.txt"
            np.savetxt(str(out_file), secs, fmt="%.6f")

            gt_file = args.output_dir / f"{label}_gt.tsv"
            np.savetxt(
                str(gt_file),
                np.column_stack([beats, secs]),
                fmt="%.6f",
                delimiter="\t",
                header="score_beat\tperf_sec",
                comments="",
            )

            status = "OK" if info["match_rate"] > 0.95 else "WARN"
            print(
                f"{status} {label:30s} matched={info['n_matched']}/{info['n_score']} "
                f"({info['match_rate']:.1%})  unique_beats={info['n_unique_beats']}"
            )

            new_row = row.copy()
            new_row["performance_annotations"] = f"{label}_aligned.txt"
            new_rows.append(new_row)

        except Exception as e:
            print(f"ERR {label:30s} {e}")

    # Save updated metadata pointing to aligned files
    new_meta = pd.DataFrame(new_rows)
    # Copy aligned files to winterreise data dir for pipeline access
    import shutil

    for _, r in new_meta.iterrows():
        src = args.output_dir / r.performance_annotations
        # Put in a subfolder of winterreise
        dest_dir = args.data_dir / "aligned"
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(src, dest_dir / r.performance_annotations)

    new_meta["performance_annotations"] = (
        "aligned/" + new_meta["performance_annotations"]
    )
    out_meta = Path(__file__).parent / "metadata-winterreise-aligned.csv"
    new_meta.to_csv(out_meta, index=False)
    print(f"\nSaved {len(new_meta)} rows to {out_meta}")


if __name__ == "__main__":
    main()
