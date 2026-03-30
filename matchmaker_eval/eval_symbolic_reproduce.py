"""Reproduce eval_symbolic.py results on validation set with local paths.

Follows eval_symbolic.py's align() approach exactly:
1. Load match file for GT time maps
2. Create Matchmaker (sets up score_follower)
3. Compute features separately via compute_features_from_symbolic
4. Call score_follower(frame) directly (no mm.run())
5. Compute error as mapped_ponsets - tracked_ponsets
"""
import sys
sys.setrecursionlimit(2**31 - 1)
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import partitura as pt
from partitura.io.importmatch import load_matchfile
from partitura.musicanalysis.performance_codec import (
    get_time_maps_from_alignment,
    to_matched_score,
)
from partitura.performance import PerformedPart
from partitura.score import Part, Score, merge_parts
from partitura.utils.misc import PathLike

from matchmaker import Matchmaker
from matchmaker.features.midi import (
    PianoRollProcessor,
    PitchIOIProcessor,
    PitchProcessor,
)
from matchmaker.utils.eval import get_evaluation_results
from matchmaker.utils.symbolic import (
    framed_midi_messages_from_performance,
    midi_messages_from_performance,
)
from matchmaker.utils.tempo_models import KalmanTempoModel

WORKING_DIR = Path(__file__).parent.parent
DATASET_DIR = {
    "asap": Path("~/data/asap-dataset-matchmaker").expanduser(),
    "batik": Path("~/data/batik_plays_mozart").expanduser(),
    "vienna": Path("~/data/vienna4x22").expanduser(),
}
METADATA_PATH = WORKING_DIR / "data/metadata-validation.csv"

# Same KWARGS as eval_symbolic.py
KWARGS = {
    "audio": {
        "arzt": {"window_size": 5, "start_window_size": 0.25, "step_size": 5},
        "dixon": {"window_size": 10},
    },
    "midi": {
        "arzt": {
            "feature_type": "pianoroll",
            "processor": "pianoroll",
            "piano_range": True,
            "window_size": 2,
            "start_window_size": 2,
            "step_size": 5,
        },
        "dixon": {
            "feature_type": "pianoroll",
            "processor": "pianoroll",
            "piano_range": True,
            "window_size": 0.3,
        },
        "hmm": {
            "processor": "pitch_ioi",
            "tempo_model": KalmanTempoModel,
            "piano_range": True,
        },
        "pthmm": {
            "processor": "pitch_ioi",
            "piano_range": True,
        },
        "outerhmm": {
            "processor": "pitch_ioi",
            "piano_range": True,
        },
    },
}


def compute_features_from_symbolic(ref_info, processor_name, processor_kwargs=None, polling_period=0.01):
    """Exactly as in eval_symbolic.py."""
    processor_mapping = {
        "pitch": PitchProcessor,
        "pitch_ioi": PitchIOIProcessor,
        "pianoroll": PianoRollProcessor,
    }
    if processor_kwargs is None:
        processor_kwargs = {}

    feature_processor = processor_mapping[processor_name](**processor_kwargs)

    if polling_period is not None:
        frames_array, frame_times = framed_midi_messages_from_performance(
            perf=ref_info, polling_period=polling_period
        )
        outputs = []
        for frame, f_time in zip(frames_array, frame_times):
            output = feature_processor((frame, f_time))
            outputs.append(output)
    else:
        frames_array, frame_times = midi_messages_from_performance(perf=ref_info)
        frames_array = np.array(list(zip(frames_array, frame_times)))
        outputs = []
        for frame, f_time in frames_array:
            output = feature_processor(([[frame, f_time]], f_time))
            outputs.append(output)
    return outputs, frame_times


def compute_pianoroll_features(note_info, polling_period):
    """Exactly as in eval_symbolic.py (partitura method)."""
    if isinstance(note_info, (Score, Part)):
        ppart = pt.utils.music.performance_from_part(note_info, bpm=60)
    else:
        ppart = note_info
    ppart.sustain_pedal_threshold = 127
    ref_frames = (
        pt.utils.music.compute_pianoroll(
            note_info=ppart,
            time_unit="sec",
            time_div=int(np.round(1 / polling_period)),
            binary=True,
            piano_range=True,
        )
        .toarray()
        .T
    )
    return ref_frames


def align(match_fn, score_xml_fn, perf_midi_fn, method, kwargs):
    """Follow eval_symbolic.py's align() logic exactly."""
    config = kwargs["midi"][method]

    solo_perf, alignment = pt.load_match(
        filename=match_fn, create_score=False, first_note_at_zero=True,
    )
    solo_ppart = solo_perf[0]
    score = pt.load_musicxml(score_xml_fn, ignore_invisible_objects=True)

    # unfold
    update_ids = '-' in load_matchfile(match_fn).snotes[0].Anchor and '-' not in score.note_array()['id'][0]
    score = pt.score.unfold_part_maximal(score, update_ids=update_ids, ignore_leaps=False)

    # If musicxml has no note IDs (e.g. vienna), fall back to match-file score
    sna_ids = set(score.note_array()["id"])
    use_match_score = sna_ids == {"None"} or len(sna_ids) <= 1
    if use_match_score:
        perf_fb, alignment_fb, score_fb = pt.load_match(
            filename=match_fn, create_score=True, first_note_at_zero=True,
        )
        ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
            ppart_or_note_array=perf_fb.note_array(),
            spart_or_note_array=score_fb.note_array(),
            alignment=alignment_fb,
        )
        matched_array = to_matched_score(score_fb, perf_fb[0], alignment_fb)[0]['onset']
    else:
        ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
            ppart_or_note_array=solo_ppart,
            spart_or_note_array=score,
            alignment=alignment,
        )
        matched_array = to_matched_score(score, solo_ppart, alignment)[0]['onset']

    sna = score.note_array()
    pna = solo_ppart.note_array()

    processor_kwargs = dict({'piano_range': config['piano_range']})
    feature_type = config["processor"]
    POLLING_PERIOD = None if method == "outerhmm" else 0.01

    if method == "hmm":
        processor_kwargs["return_pitch_list"] = True
    elif method == "pthmm":
        config = dict(config)  # copy to avoid mutation
        config["processor"] = 'pitch'
        processor_kwargs["return_pitch_list"] = False
        feature_type = "pitch_ioi"
    elif method == "outerhmm":
        processor_kwargs["return_pitch_list"] = False

    if config["processor"] == "pianoroll":
        ref_frames = compute_pianoroll_features(note_info=solo_ppart, polling_period=POLLING_PERIOD)
        input_signal = np.array(ref_frames).astype(np.float32)
        frame_times = np.arange(1, len(input_signal) + 1) * POLLING_PERIOD
    else:
        input_signal, frame_times = compute_features_from_symbolic(
            ref_info=solo_ppart,
            processor_name=config["processor"],
            processor_kwargs=processor_kwargs,
            polling_period=POLLING_PERIOD,
        )

    mm = Matchmaker(
        score_file=score_xml_fn,
        performance_file=perf_midi_fn,
        input_type="midi",
        feature_type=feature_type,
        method=method,
        kwargs=kwargs,
    )

    tracked_sonsets, tracked_ponsets = [], []

    if config["processor"] == "pianoroll":
        current_idx = 0
        for i, frame in enumerate(input_signal):
            if frame.sum() != 0:
                current_state = mm.score_follower(frame)
                score_position = mm.score_follower.state_to_ref_time_map(current_state * POLLING_PERIOD)
                if score_position is not None:
                    try:
                        current_onset = mm.score_follower.state_space[current_idx]
                    except IndexError:
                        current_onset = mm.score_follower.state_space[-1]
                    if score_position >= current_onset:
                        if current_onset not in tracked_sonsets:
                            tracked_sonsets.append(current_onset)
                            tracked_ponsets.append(frame_times[i])
                            current_idx += 1
    else:
        for i, frame in enumerate(input_signal):
            if frame is not None:
                current_state = mm.score_follower(frame)
                mm_score_position = mm.score_follower.state_space[current_state]
                tracked_sonsets.append(mm_score_position)
                tracked_ponsets.append(frame_times[i])

    tracked_sonsets = np.array(tracked_sonsets)
    tracked_ponsets = np.array(tracked_ponsets)

    set_matched_sonsets = set(matched_array)
    set_tracked_sonsets = set(tracked_sonsets)
    tracking_ratio = len(set_matched_sonsets.intersection(set_tracked_sonsets)) / len(set_matched_sonsets)

    mapped_ponsets = stime_to_ptime_map(tracked_sonsets)

    tolerances_in_seconds = [10, 25, 50, 100, 200, 300, 500, 1000, 2000]
    results_in_seconds = get_evaluation_results(
        mapped_ponsets,
        tracked_ponsets,
        total_counts=len(tracked_sonsets),
        tolerances=tolerances_in_seconds,
        in_seconds=True,
    )

    mapped_sonsets = ptime_to_stime_map(tracked_ponsets)
    beat_tolerances = [0.05, 0.1, 0.3, 0.5, 1, 2]
    beat_results = get_evaluation_results(
        tracked_sonsets,
        mapped_sonsets,
        total_counts=len(tracked_sonsets),
        tolerances=beat_tolerances,
        in_seconds=False,
    )

    # Build GT from match file: (score_beat, perf_time_sec)
    unique_onsets = np.unique(sna["onset_beat"])
    gt_perf_times = stime_to_ptime_map(unique_onsets)
    gt_valid = np.isfinite(gt_perf_times)
    gt = np.column_stack([unique_onsets[gt_valid], gt_perf_times[gt_valid]])

    # Sparse WP: (score_beat, perf_time_sec)
    wp = np.column_stack([tracked_sonsets, tracked_ponsets])

    return {
        "tracking_ratio": tracking_ratio,
        "MAE_ms": results_in_seconds["median"],
        "AAE_ms": results_in_seconds["mean"],
        "std_ms": results_in_seconds["std"],
        "lt_50ms": results_in_seconds.get("50ms", 0),
        "lt_100ms": results_in_seconds.get("100ms", 0),
        "lt_300ms": results_in_seconds.get("300ms", 0),
        "lt_500ms": results_in_seconds.get("500ms", 0),
        "lt_1000ms": results_in_seconds.get("1000ms", 0),
        "MAE_beats": beat_results["median"],
        "count": results_in_seconds.get("count", len(tracked_sonsets)),
        "wp": wp,
        "gt": gt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="outerhmm", choices=["outerhmm", "hmm", "pthmm", "arzt", "dixon"])
    args = parser.parse_args()
    method = args.method

    metadata = pd.read_csv(METADATA_PATH, skipinitialspace=True)
    metadata.columns = metadata.columns.str.strip()
    str_cols = metadata.select_dtypes(include=["object"]).columns
    metadata[str_cols] = metadata[str_cols].apply(lambda x: x.str.strip())

    # Output directory for wp/gt files
    save_dir = WORKING_DIR / "output" / f"symbolic_{method}_valid"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, row in enumerate(metadata.itertuples(), 1):
        dataset = row.dataset
        dataset_dir = DATASET_DIR[dataset]
        score_xml = dataset_dir / row.xml_score
        perf_midi = dataset_dir / row.midi_performance
        match = dataset_dir / row.match

        print(f"[{i}/{len(metadata)}] {row.title} ({dataset})...", end=" ", flush=True)

        try:
            res = align(
                match_fn=str(match),
                score_xml_fn=str(score_xml),
                perf_midi_fn=str(perf_midi),
                method=method,
                kwargs=KWARGS,
            )
            res["title"] = row.title

            # Save wp and gt
            np.savetxt(save_dir / f"wp_{i}.tsv", res["wp"], delimiter="\t", fmt="%.6f")
            np.savetxt(save_dir / f"gt_{i}.tsv", res["gt"], delimiter="\t", fmt="%.6f")

            # Remove large arrays before adding to results list
            res_summary = {k: v for k, v in res.items() if k not in ("wp", "gt")}
            all_results.append(res_summary)
            print(f"MAE={res['MAE_ms']:.1f}ms, tracking={res['tracking_ratio']:.3f}, <300ms={res['lt_300ms']:.3f}")
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(save_dir / "results.csv", index=False)
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"Pieces: {len(df)}")
        print(f"Tracking ratio (avg): {df['tracking_ratio'].mean():.3f}")
        print(f"MAE ms (avg): {df['MAE_ms'].mean():.3f}")
        print(f"AAE ms (avg): {df['AAE_ms'].mean():.3f}")
        print(f"<50ms (avg): {df['lt_50ms'].mean()*100:.1f}%")
        print(f"<100ms (avg): {df['lt_100ms'].mean()*100:.1f}%")
        print(f"<300ms (avg): {df['lt_300ms'].mean()*100:.1f}%")
        print(f"<1000ms (avg): {df['lt_1000ms'].mean()*100:.1f}%")
        print(f"MAE beats (avg): {df['MAE_beats'].mean():.3f}")
        print(f"\nSaved wp/gt to: {save_dir}")


if __name__ == "__main__":
    main()
