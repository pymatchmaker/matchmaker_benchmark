"""Preprocess URMP from the original per-instrument annotations.

For each piece, merges the per-instrument Notes_<k>_*.txt onsets/pitches,
aligns them to the score notes by pitch (Needleman-Wunsch), collapses to the
unique score onset beats, re-zeroes the leading silence (0.5s margin), and writes:
  - data/preprocessed/urmp/ground_truth/trimmed/<i>.tsv
  - <urmp-dir>/trimmed_audio/AuMix_<folder>_trim.wav
  - data/metadata-urmp-trimmed.csv
The original metadata and per-piece raw URMP files remain unchanged.

Usage:
    python preprocess_urmp.py
    python preprocess_urmp.py --urmp-dir ~/data/URMP
"""

import argparse
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from matchmaker import Matchmaker

WORKING_DIR = Path(__file__).parent.parent
PREPROCESSED_DIR = WORKING_DIR / "data" / "preprocessed" / "urmp"
GT_DIR = PREPROCESSED_DIR / "ground_truth" / "trimmed"
SOURCE_METADATA = WORKING_DIR / "data" / "metadata-urmp.csv"
OUTPUT_METADATA = WORKING_DIR / "data" / "metadata-urmp-trimmed.csv"
MARGIN = 0.5


def hz_to_midi(hz):
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(hz / 440.0)
    midi[~np.isfinite(midi)] = -1
    return np.round(midi).astype(int)


def match_by_pitch(score_pitches, perf_pitches, perf_onsets):
    n, m = len(score_pitches), len(perf_pitches)
    GAP, MATCH, MISMATCH = -1, 2, -2
    dp = np.full((n + 1, m + 1), -np.inf)
    dp[0, 0] = 0
    dp[1:, 0] = np.arange(1, n + 1) * GAP
    dp[0, 1:] = np.arange(1, m + 1) * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = MATCH if score_pitches[i - 1] == perf_pitches[j - 1] else MISMATCH
            dp[i, j] = max(dp[i - 1, j - 1] + s, dp[i - 1, j] + GAP, dp[i, j - 1] + GAP)
    res = np.full(n, np.nan)
    i, j = n, m
    while i > 0 and j > 0:
        s = MATCH if score_pitches[i - 1] == perf_pitches[j - 1] else MISMATCH
        if dp[i, j] == dp[i - 1, j - 1] + s:
            if score_pitches[i - 1] == perf_pitches[j - 1]:
                res[i - 1] = perf_onsets[j - 1]
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + GAP:
            i -= 1
        else:
            j -= 1
    return res


def load_merged_perf(piece_dir):
    onsets, pitches = [], []
    for f in sorted(glob.glob(str(piece_dir / "Notes_[0-9]*_*.txt"))):
        a = np.loadtxt(f, ndmin=2)
        onsets.append(a[:, 0])
        pitches.append(hz_to_midi(a[:, 1]))
    onsets = np.concatenate(onsets)
    pitches = np.concatenate(pitches)
    order = np.argsort(onsets, kind="stable")
    return onsets[order], pitches[order]


def main():
    parser = argparse.ArgumentParser(description="Build URMP GT + audio from raw notes")
    parser.add_argument(
        "--urmp-dir", type=Path, default=Path("~/data/URMP").expanduser()
    )
    args = parser.parse_args()
    audio_dir = args.urmp_dir / "trimmed_audio"

    meta = pd.read_csv(SOURCE_METADATA, skipinitialspace=True)
    meta.columns = meta.columns.str.strip()
    GT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_col = []
    gt_col = []
    for i, row in enumerate(meta.itertuples(), 1):
        piece_dir = args.urmp_dir / row.folder
        original = piece_dir / f"AuMix_{row.folder}.wav"

        mm = Matchmaker(
            score_file=piece_dir / row.xml_score,
            performance_file=original,
            input_type="audio",
            unfold_score=False,
        )
        na = mm.score_part.note_array()
        order = np.lexsort((na["pitch"], na["onset_div"]))
        note_beats, note_pitches = na["onset_beat"][order], na["pitch"][order]

        perf_onsets, perf_pitches = load_merged_perf(piece_dir)
        aligned = match_by_pitch(note_pitches, perf_pitches, perf_onsets)
        valid = np.isfinite(aligned)
        aligned = np.interp(np.arange(len(aligned)), np.flatnonzero(valid), aligned[valid])

        beats = np.unique(note_beats)
        perf = np.array([aligned[note_beats == b].min() for b in beats])

        trim_offset = max(0, perf[0] - MARGIN)
        audio_name = original.name
        if trim_offset > 1.0:
            perf = perf - trim_offset
            audio, sr = sf.read(str(original))
            trimmed = audio_dir / f"AuMix_{row.folder}_trim.wav"
            sf.write(str(trimmed), audio[int(trim_offset * sr):], sr)
            audio_name = f"dataset_root/trimmed_audio/{trimmed.name}"
            print(f"[{i}] {row.folder}: trimmed {trim_offset:.2f}s")

        gt = np.column_stack([perf, beats])
        np.savetxt(
            GT_DIR / f"{i}.tsv", gt, fmt="%.6f", delimiter="\t",
            header="perf_sec\tscore_beat", comments="",
        )
        audio_col.append(audio_name)
        gt_col.append(f"data/preprocessed/urmp/ground_truth/trimmed/{i}.tsv")

    meta["audio_performance"] = audio_col
    meta["performance_annotations"] = gt_col
    meta.to_csv(OUTPUT_METADATA, index=False)
    print(f"Updated {OUTPUT_METADATA} and {PREPROCESSED_DIR} ({len(meta)} pieces)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
