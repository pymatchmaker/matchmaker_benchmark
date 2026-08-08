"""Build KRAISLER note-onset GT and 60-row audio metadata.

Piano score/performance onsets come from ``alignment_match/*_PF.match``.
Violin score notes are aligned to ``annotation_csv/*_notes_VN.csv`` by pitch,
order, and the official beat-time map. Piano and violin annotations at the same
score onset are aggregated with the earliest performance onset.

The output has one note-level GT file per track and three mix variants per track:
``mix_dry``, ``mix_hall``, and ``mix_studio``.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import partitura as pt


WORKING_DIR = Path(__file__).parent.parent
OUTPUT_METADATA = WORKING_DIR / "data" / "metadata-kraisler.csv"
GT_DIR = WORKING_DIR / "data" / "preprocessed" / "kraisler" / "ground_truth"
ROOMS = ("dry", "hall", "studio")

MATCHED_NOTE = re.compile(
    r"^snote\([^,]+,\[[^]]+\],-?\d+,[^,]+,[^,]+,[^,]+,"
    r"(-?\d+(?:\.\d+)?),-?\d+(?:\.\d+)?,\[[^]]*\]\)-"
    r"note\([^,]+,-?\d+,(-?\d+),"
)


def align_violin_notes(
    score_pitches: np.ndarray,
    expected_times: np.ndarray,
    perf_pitches: np.ndarray,
    perf_times: np.ndarray,
) -> list[tuple[int, int]]:
    """Needleman-Wunsch alignment using pitch, order, and beat-time anchors."""
    n_score, n_perf = len(score_pitches), len(perf_pitches)
    scores = np.empty((n_score + 1, n_perf + 1), dtype=float)
    scores[:, 0] = -np.arange(n_score + 1)
    scores[0, :] = -np.arange(n_perf + 1)
    back = np.zeros((n_score + 1, n_perf + 1), dtype=np.int8)

    for i in range(1, n_score + 1):
        for j in range(1, n_perf + 1):
            if score_pitches[i - 1] == perf_pitches[j - 1]:
                time_error = abs(expected_times[i - 1] - perf_times[j - 1])
                match_score = 3 - min(time_error / 0.25, 6)
            else:
                match_score = -3
            candidates = (
                scores[i - 1, j - 1] + match_score,
                scores[i - 1, j] - 1,
                scores[i, j - 1] - 1,
            )
            move = int(np.argmax(candidates))
            scores[i, j] = candidates[move]
            back[i, j] = move

    pairs = []
    i, j = n_score, n_perf
    while i or j:
        move = int(back[i, j]) if i and j else (1 if i else 2)
        if move == 0:
            time_error = abs(expected_times[i - 1] - perf_times[j - 1])
            if score_pitches[i - 1] == perf_pitches[j - 1] and time_error < 1.5:
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def parse_piano_match(match_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract score beats and performance seconds from matched piano notes."""
    text = match_path.read_text()
    units_match = re.search(r"info\(midiClockUnits,(\d+)\)", text)
    rate_match = re.search(r"info\(midiClockRate,(\d+)\)", text)
    if units_match is None or rate_match is None:
        raise ValueError(f"Missing MIDI clock metadata in {match_path}")
    units = int(units_match.group(1))
    rate = int(rate_match.group(1))

    matched_lines = [line for line in text.splitlines() if ")-note(" in line]
    matches = [MATCHED_NOTE.match(line) for line in matched_lines]
    if not matches or any(match is None for match in matches):
        raise ValueError(f"Could not parse every matched piano note in {match_path}")

    beats = np.round([float(match.group(1)) for match in matches], 4)
    ticks = np.array([int(match.group(2)) for match in matches], dtype=float)
    seconds = ticks * rate / (units * 1_000_000)
    return beats, seconds


def beat_time_map(
    score_path: Path, beat_csv: Path
) -> tuple[np.ndarray, np.ndarray, object]:
    """Map official musical beats back to the score's native note coordinates."""
    native_part = pt.load_score_as_part(str(score_path))
    musical_part = pt.load_score_as_part(str(score_path))
    musical_part.use_musical_beat()

    beat_times = pd.read_csv(beat_csv)["beat_time"].to_numpy(dtype=float)
    start = np.ceil(float(musical_part.note_array()["onset_beat"].min()))
    musical_beats = np.arange(start, start + len(beat_times))
    native_beats = native_part.beat_map(musical_part.inv_beat_map(musical_beats))
    return np.asarray(native_beats, dtype=float), beat_times, native_part


def build_track_gt(dataset_dir: Path, track: str) -> tuple[np.ndarray, dict]:
    score_path = dataset_dir / "score_musicxml" / f"{track}.musicxml"
    beat_beats, beat_times, native_part = beat_time_map(
        score_path, dataset_dir / "annotation_csv" / f"{track}_beats.csv"
    )

    piano_beats, piano_times = parse_piano_match(
        dataset_dir / "alignment_match" / f"{track}_PF.match"
    )

    violin_score = pt.load_score(
        str(dataset_dir / "score_musicxml" / f"{track}_VN.musicxml")
    ).note_array()
    violin_perf = pd.read_csv(
        dataset_dir / "annotation_csv" / f"{track}_notes_VN.csv"
    )
    expected_times = np.interp(
        violin_score["onset_beat"], beat_beats, beat_times
    )
    violin_pairs = align_violin_notes(
        violin_score["pitch"].astype(int),
        expected_times,
        violin_perf["midi"].to_numpy(dtype=int),
        violin_perf["onset"].to_numpy(dtype=float),
    )
    violin_beats = np.round(
        [float(violin_score["onset_beat"][i]) for i, _ in violin_pairs], 4
    )
    violin_times = np.array(
        [float(violin_perf["onset"].iloc[j]) for _, j in violin_pairs]
    )

    all_beats = np.concatenate([piano_beats, violin_beats])
    all_times = np.concatenate([piano_times, violin_times])
    score_beats = np.unique(all_beats)
    perf_times = np.array(
        [all_times[all_beats == beat].min() for beat in score_beats]
    )
    gt = np.column_stack([perf_times, score_beats])

    native_onsets = np.unique(
        np.round(native_part.note_array()["onset_beat"].astype(float), 4)
    )
    coverage = float(np.isin(native_onsets, score_beats).mean())
    valid = (beat_beats >= score_beats.min()) & (beat_beats <= score_beats.max())
    beat_error = np.abs(
        np.interp(beat_beats[valid], score_beats, perf_times) - beat_times[valid]
    )
    stats = {
        "violin_matched": len(violin_pairs),
        "violin_score": len(violin_score),
        "violin_performance": len(violin_perf),
        "onset_coverage": coverage,
        "beat_error_median": float(np.median(beat_error)),
        "beat_error_p95": float(np.quantile(beat_error, 0.95)),
    }
    return gt, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kraisler-dir",
        type=Path,
        default=Path("~/data/KRAISLER").expanduser(),
    )
    args = parser.parse_args()

    source = pd.read_csv(OUTPUT_METADATA)
    source["track"] = source["xml_score"].map(lambda path: Path(path).stem[-2:])
    source = source.drop_duplicates("track").sort_values("track")
    if len(source) != 20:
        raise ValueError(f"Expected 20 source tracks, found {len(source)}")

    official = pd.read_csv(args.kraisler_dir / "metadata.csv", dtype={"track": str})
    official["track"] = official["track"].str.zfill(2)
    official = official.set_index("track")

    GT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in source.itertuples(index=False):
        track = row.track
        gt, stats = build_track_gt(args.kraisler_dir, track)
        gt_path = GT_DIR / f"{track}.tsv"
        np.savetxt(
            gt_path,
            gt,
            fmt="%.6f",
            delimiter="\t",
            header="perf_sec\tscore_beat",
            comments="",
        )
        print(
            f"[{track}] {len(gt)} onsets; "
            f"VN {stats['violin_matched']}/{stats['violin_score']}/"
            f"{stats['violin_performance']}; coverage "
            f"{stats['onset_coverage']:.1%}; beat p95 "
            f"{stats['beat_error_p95']:.3f}s"
        )

        for room in ROOMS:
            rows.append(
                {
                    "composer": row.composer,
                    "title": row.title,
                    "audio_variant": f"mix_{room}",
                    "xml_score": official.at[track, "score_musicxml"],
                    "performance_annotations": (
                        "data/preprocessed/kraisler/ground_truth/"
                        f"{track}.tsv"
                    ),
                    "audio_performance": official.at[
                        track, f"audio_mix_{room}"
                    ],
                }
            )

    pd.DataFrame(rows).to_csv(OUTPUT_METADATA, index=False)
    print(f"Wrote {len(rows)} metadata rows to {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
