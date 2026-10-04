"""Note-level KRAISLER ground truth from the original dataset.

For each of the 20 violin--piano pieces, every score onset of the full score gets
one performance time:

- piano: the score-to-performance note pairs of ``alignment_match/<k>_PF.match``;
- violin: ``annotation_csv/<k>_notes_VN.csv`` aligned to the violin part by pitch
  (the Needleman-Wunsch alignment of ``preprocess_urmp.py``).

A score onset takes the median time of its paired notes. Notes the score marks
as tremolo are used only when nothing else sounds at that onset: their repeated
strokes are paired with the wrong repetition in places. A time is then kept only
if it lies within a quarter of a score beat of the manual beat annotation
(``annotation_csv/<k>_beats.csv``), and the remaining inversions are removed by
keeping the longest increasing subsequence. The three mixes of a piece share one
file. Writes ``<out-dir>/<k>.tsv`` (perf_sec, score_beat)::

    python matchmaker_eval/build_kraisler_gt.py --kraisler-dir ~/data/KRAISLER \\
        --out-dir <data repo>/kraisler/annotations
"""

import argparse
import bisect
import csv
import re
import sys
from pathlib import Path

import numpy as np
import partitura as pt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_urmp import match_by_pitch  # noqa: E402

#: Beat-annotation units tried against the score beat, and their offsets.
BEAT_SCALES = (0.5, 1, 2, 3, 1 / 3)
BEAT_OFFSETS = np.arange(-6, 6.01, 0.25)
#: Largest distance from the beat annotation, in score beats.
TOLERANCE_BEATS = 0.25


def longest_increasing(times: np.ndarray) -> np.ndarray:
    """Mask of a longest strictly increasing subsequence."""
    tails, tail_idx, prev = [], [], [-1] * len(times)
    for i, t in enumerate(times):
        j = bisect.bisect_left(tails, t)
        if j == len(tails):
            tails.append(t)
            tail_idx.append(i)
        else:
            tails[j], tail_idx[j] = t, i
        prev[i] = tail_idx[j - 1] if j > 0 else -1
    keep = np.zeros(len(times), dtype=bool)
    k = tail_idx[-1]
    while k != -1:
        keep[k] = True
        k = prev[k]
    return keep


def onset_pairs(root: Path, k: str):
    """(score beat, performance time, tremolo) for every paired note of piece k."""
    load = lambda name: pt.load_musicxml(  # noqa: E731
        str(root / "score_musicxml" / name), force_note_ids="keep"
    ).note_array()
    piano, violin = load(f"{k}_PF.musicxml"), load(f"{k}_VN.musicxml")

    match_path = root / "alignment_match" / f"{k}_PF.match"
    tremolo = set(
        re.findall(r"^snote\(([^,]+),.*tremolo", match_path.read_text(), re.M)
    )
    performance, alignment = pt.load_match(str(match_path))[:2]
    note_on = {
        str(n["id"]): float(n["note_on"])
        for part in performance.performedparts
        for n in part.notes
    }
    beat_of = {str(n["id"]): float(n["onset_beat"]) for n in piano}
    pairs = [
        (beat_of[a["score_id"]], note_on[a["performance_id"]], a["score_id"] in tremolo)
        for a in alignment
        if a.get("label") == "match"
        and a["score_id"] in beat_of
        and a["performance_id"] in note_on
    ]

    order = np.lexsort((violin["pitch"], violin["onset_div"]))
    with open(root / "annotation_csv" / f"{k}_notes_VN.csv") as f:
        notes = [(float(r["onset"]), int(r["midi"])) for r in csv.DictReader(f)]
    onsets = np.array([n[0] for n in notes])
    pitches = np.array([n[1] for n in notes])
    aligned = match_by_pitch(violin["pitch"][order], pitches, onsets)
    pairs += [
        (b, t, False)
        for b, t in zip(violin["onset_beat"][order], aligned)
        if np.isfinite(t)
    ]
    return pairs


def piece_gt(root: Path, k: str):
    pairs = onset_pairs(root, k)
    beats_all = np.array([p[0] for p in pairs])
    times_all = np.array([p[1] for p in pairs])
    tremolo = np.array([p[2] for p in pairs])

    beats = np.unique(beats_all)
    times = np.empty(len(beats))
    for i, b in enumerate(beats):
        at = beats_all == b
        steady = at & ~tremolo
        times[i] = np.median(times_all[steady if steady.any() else at])

    with open(root / "annotation_csv" / f"{k}_beats.csv") as f:
        beat_times = np.array([float(r["beat_time"]) for r in csv.DictReader(f)])
    grid = np.arange(len(beat_times))
    best = None
    for scale in BEAT_SCALES:
        for offset in BEAT_OFFSETS:
            index = (beats - beats[0]) * scale + offset
            inside = (index >= 0) & (index <= len(beat_times) - 1)
            if inside.sum() < 0.7 * len(beats):
                continue
            err = np.median(np.abs(times[inside] - np.interp(index[inside], grid, beat_times)))
            if best is None or err < best[0]:
                best = (err, scale, offset)
    _, scale, offset = best
    index = (beats - beats[0]) * scale + offset
    inside = (index >= 0) & (index <= len(beat_times) - 1)
    reference = np.interp(index, grid, beat_times)
    interval = np.interp(index, grid[:-1] + 0.5, np.diff(beat_times))
    close = np.abs(times - reference) <= TOLERANCE_BEATS * interval / scale
    keep = ~inside | close

    beats, times = beats[keep], times[keep]
    increasing = longest_increasing(times)
    stats = {
        "onsets": len(keep),
        "off_beat_annotation": int((~keep).sum()),
        "inversions": int((~increasing).sum()),
    }
    return times[increasing], beats[increasing], stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kraisler-dir", type=Path, default=Path("~/data/KRAISLER").expanduser())
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = {"onsets": 0, "off_beat_annotation": 0, "inversions": 0}
    for i in range(1, 21):
        k = f"{i:02d}"
        times, beats, stats = piece_gt(args.kraisler_dir, k)
        np.savetxt(
            args.out_dir / f"{k}.tsv",
            np.column_stack([times, beats]),
            fmt="%.6f",
            delimiter="\t",
            header="perf_sec\tscore_beat",
            comments="",
        )
        for key in total:
            total[key] += stats[key]
        print(f"{k}: {len(beats)}/{stats['onsets']} onsets kept {stats}")
    kept = total["onsets"] - total["off_beat_annotation"] - total["inversions"]
    print(f"total: {kept}/{total['onsets']} onsets kept {total}")


if __name__ == "__main__":
    main()
