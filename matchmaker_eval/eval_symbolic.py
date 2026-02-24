# -*- coding: utf-8 -*-
import sys
sys.setrecursionlimit(2**31 - 1)
import argparse
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict
from pathlib import Path
import wandb

import pdb

import pandas as pd

import parangonar as pa

import numpy as np
import partitura as pt
import matplotlib.pyplot as plt
import yaml
import csv


import scipy

from partitura.performance import Performance, PerformedPart, PerformanceLike
from partitura.score import Score, Part, ScoreLike, merge_parts
from partitura.utils.misc import PathLike

from partitura.utils.music import performance_from_part, compute_pianoroll
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment, to_matched_score
from partitura.io.importmatch import load_matchfile

from partitura_utils import (
    partitura_to_framed_midi_custom as partitura_to_framed_midi,
)

from matchmaker import Matchmaker

from matchmaker.utils.symbolic import (
    framed_midi_messages_from_performance,
    midi_messages_from_performance,
)
from matchmaker.features.midi import PitchIOIProcessor, PitchProcessor, PianoRollProcessor, PitchClassPianoRollProcessor

"""from matchmaker.utils.accompanist_tempo_models import   (KalmanTempoSyncModel, 
                                                        ReactiveSyncModel,
                                                        MovingAverageSyncModel,
                                                        LinearSyncModel,
                                                        JointAdaptationAnticipationSyncModel)"""
from matchmaker.utils.tempo_models import (
    KalmanTempoModel,
    LinearTempoModel,
    MovingAverageTempoModel,
    ReactiveTempoModel,
    JointAdaptationAnticipationModel
)

from matchmaker.utils.eval import (
    TOLERANCES_IN_BEATS,
    TOLERANCES_IN_MILLISECONDS,
    get_evaluation_results,
    #transfer_from_perf_to_predicted_score,
    #transfer_from_score_to_predicted_perf,
)

from mido import Message
InputMIDIFrame = Tuple[List[Tuple[Message, float]], float]

import warnings
warnings.filterwarnings("ignore")


DATASET_DIR = {
    "validation": Path("/home/alexander-neuhauser/datasets"),
    "asap": Path("/home/alexander-neuhauser/datasets/asap-dataset-matchmaker"),
    "batik": Path("/home/alexander-neuhauser/datasets/Batik_Audio"),
    "vienna": Path("/home/alexander-neuhauser/datasets/vienna4x22"),
}
METADATA_PATH = {
    "validation": "../ismir2025_matchmaker/data/metadata-validation.csv",
    "asap": "../ismir2025_matchmaker/data/reduced/metadata-asap.csv",
    "batik": "../ismir2025_matchmaker/data/reduced/metadata-batik.csv",
    "vienna":"../ismir2025_matchmaker/data/reduced/metadata-vienna.csv",
}

POLLING_PERIOD = None#0.01

SAVE_RESULTS = True
SAVE_ALIGNMENTS = True

KWARGS = {
    "audio":
        {"arzt":
            {"window_size": 5,
             "start_window_size": 0.25,
             "step_size" : 5,
             },
        "dixon":
            {"window_size": 10,
             },
        },
    "midi": 
        {"arzt": 
            {"processor": "pianoroll",
             "piano_range": True,
             "window_size": 200,
             "start_window_size": 200,
             "step_size": 5,
             },
        "dixon":
            {"processor": "pianoroll",
             "piano_range": True,
             "window_size": 30,
             },
        "hmm": 
            {"processor": "pitch_ioi",
             "tempo_model": KalmanTempoModel,
             "piano_range": True,
             },
        "pthmm":
            {"processor": "pitch_ioi", # "pitch"
             "piano_range": True,
             },
        "outerhmm":
            {"processor": "pitch_ioi",
             "piano_range": True,
             },
        },
}

# ============= helper functions =============

def setup_match_file(
    solo_fn: List[PathLike],
) -> Tuple[List[Tuple[PerformedPart, Callable, Callable]], Score]:
    """
    Setup the score objects.
    """
    solo_parts = []
    for i, fn in enumerate(solo_fn):
        if fn.endswith(".match"):
            if i == 0:
                solo_perf, alignment, solo_score = pt.load_match(
                    filename=fn,
                    create_score=True,
                    first_note_at_zero=True,
                )
                solo_ppart = solo_perf[0]
                solo_spart = solo_score[0]
            else:
                solo_perf, alignment = pt.load_match(
                    filename=fn,
                    create_score=False,
                    first_note_at_zero=True,
                )
                solo_ppart = solo_perf[0]
                

            ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
                ppart_or_note_array=solo_ppart,
                spart_or_note_array=solo_spart,
                alignment=alignment,
            )
            solo_parts.append((solo_ppart, ptime_to_stime_map, stime_to_ptime_map))
    return solo_parts


def compute_features_from_symbolic(
    ref_info: Union[ScoreLike, PerformanceLike, np.ndarray, str],
    processor_name: str,
    processor_kwargs: Optional[dict] = None,
    polling_period: Optional[float] = 0.01,
    bpm: Optional[float] = 120,
):
    processor_mapping = {
        "pitch": PitchProcessor,
        "pitch_ioi": PitchIOIProcessor,
        "pianoroll": PianoRollProcessor,
        "pitch_class_pianoroll": PitchClassPianoRollProcessor,
    }
    if processor_kwargs is None:
        processor_kwargs = {}

    feature_processor = processor_mapping[processor_name](**processor_kwargs)

    if isinstance(ref_info, Score):
        ref_info = performance_from_part(
            part=merge_parts(ref_info) if len(ref_info) > 1 else ref_info[0],
            bpm=bpm,
        )
    elif isinstance(ref_info, Part):
        ref_info = performance_from_part(
            part=ref_info,
            bpm=bpm,
        )
    elif isinstance(ref_info, str):
        # This method assumes that all paths are to
        # performance files.
        ref_info = pt.load_performance(ref_info)

    elif isinstance(ref_info, np.ndarray):
        ref_info = PerformedPart.from_note_array(ref_info)

    if polling_period is not None:
        frames_array, frame_times = framed_midi_messages_from_performance(
            perf=ref_info, polling_period=polling_period
        )
        outputs = []
        for frame, f_time in zip(frames_array, frame_times):
            output = feature_processor((frame, f_time))

            outputs.append(output)
    else:
        frames_array, frame_times = midi_messages_from_performance(
            perf=ref_info,
        )

        # Get same format as expected by the input processors
        frames_array = np.array(list(zip(frames_array, frame_times)))

        outputs = []
        for frame, f_time in frames_array:
            output = feature_processor(([[frame, f_time]], f_time))    # TODO: CHANGE IN matchmaker.features.midi as well ?

            outputs.append(output)
    return outputs, frame_times

def compute_pianoroll_features(
    note_info: Union[ScoreLike, PerformanceLike],
    polling_period: float,
    pianoroll_method: str = "partitura",
):
    """
    Features for OLTW
    """
    if pianoroll_method == "accompanion":
        ref_frames = partitura_to_framed_midi(
            part_or_notearray_or_filename=note_info,
            is_performance=True,
            pipeline=PianoRollProcessor(piano_range=True),
            polling_period=polling_period,
        )[0]

    elif pianoroll_method == "partitura":

        if isinstance(note_info, (Score, Part)):
            ppart = pt.utils.music.performance_from_part(note_info, bpm=60)
        else:
            ppart = note_info
        # Use real note off instead of sound off
        ppart.sustain_pedal_threshold = 127
        ref_frames = (
            compute_pianoroll(
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

# ============= main alignment function for a single performance =============

def align(
    solo_perf_fn: PathLike,
    reference_fn: Union[PathLike, List[PathLike]],
    perf_midi,
    args,
    kwargs,
    save_alignments=False
):  
    config = kwargs[args.input_type][args.method]

    solo_perf, alignment = pt.load_match(
                    filename=solo_perf_fn,
                    create_score=False,
                    first_note_at_zero=True,
                )
    solo_ppart = solo_perf[0]
    score = pt.load_musicxml(reference_fn, ignore_invisible_objects=True)

    if True: # unfold piece - only necessary if piece is not unfolded yet
        update_ids = '-' in load_matchfile(solo_perf_fn).snotes[0].Anchor and '-' not in score.note_array()['id'][0]
        score = pt.score.unfold_part_maximal(score, update_ids=update_ids, ignore_leaps=False)

    ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
                ppart_or_note_array=solo_ppart,
                spart_or_note_array=score,
                alignment=alignment,
            )
    
    sna = score.note_array()
    pna = solo_ppart.note_array()
    
    matched_array = to_matched_score(score, solo_ppart, alignment)[0]['onset']
    
    processor_kwargs = dict({'piano_range': config['piano_range']})

    feature_type = config["processor"]
    POLLING_PERIOD = None if args.method == "outerhmm" else 0.01

    if args.method == "hmm":
        processor_kwargs["return_pitch_list"] = True
        
    elif args.method == "pthmm":
        config["processor"] = 'pitch'
        processor_kwargs["return_pitch_list"] = False
        feature_type = "pitch_ioi"
    
    elif args.method == "outerhmm":
        processor_kwargs["return_pitch_list"] = False
        
    if config["processor"] == "pianoroll" :
        ref_frames = compute_pianoroll_features(note_info=solo_ppart, polling_period=POLLING_PERIOD)

        input_signal = np.array(ref_frames).astype(np.float32)
        frame_times = np.arange(1, len(input_signal)+1) * POLLING_PERIOD
        
    else:
        input_signal, frame_times = compute_features_from_symbolic(ref_info=solo_ppart,processor_name=config["processor"], 
                                                                processor_kwargs=processor_kwargs,
                                                                polling_period=POLLING_PERIOD)


    mm = Matchmaker(
    score_file=reference_fn, # the score file (musicxml) is used as reference feature
    performance_file=perf_midi,
    input_type=args.input_type,
    feature_type=feature_type,
    method=args.method,
    kwargs=kwargs
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
                mm_score_position = mm.score_follower.state_space[current_state] # (= current_position)
                tracked_sonsets.append(mm_score_position)
                tracked_ponsets.append(frame_times[i])

    tracked_sonsets = np.array(tracked_sonsets)
    tracked_ponsets = np.array(tracked_ponsets)

    set_matched_sonsets = set(matched_array)
    set_tracked_sonsets = set(tracked_sonsets)
    tracking_ratio = len(set_matched_sonsets.intersection(set_tracked_sonsets))/len(set_matched_sonsets)

    mapped_ponsets = stime_to_ptime_map(tracked_sonsets)    

    list_of_nans = []
    asynchrony = mapped_ponsets - tracked_ponsets
    if np.count_nonzero(np.isnan(asynchrony)) > 0:
        list_of_nans = np.where(np.isnan(asynchrony))
        print(f'Warning: asynchrony array contains {np.count_nonzero(np.isnan(asynchrony))} NaNs!')
    
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
    gt_ponsets = pna["onset_sec"]
    gt_sonsets = sna["onset_beat"]

    predicted_alignments = tracked_ponsets, mapped_ponsets, tracked_sonsets, mapped_sonsets, gt_ponsets, gt_sonsets

    if save_alignments:
        results_path = os.path.join("predicted_alignments", args.method, args.dataset)
        os.makedirs(results_path, exist_ok=True)

        alignment_fn = solo_perf_fn.replace(str(DATASET_DIR[args.dataset]),'')[1:].replace("/", "-").replace(".match", ".csv")
        alignment_fn = os.path.join(results_path, alignment_fn)

        with open(alignment_fn, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['tracked_ponsets', 'mapped_ponsets', 'tracked_sonsets', 'mapped_sonsets'])
            writer.writerows(zip(predicted_alignments))

        plt.plot(gt_sonsets, gt_sonsets, label="gt")
        plt.plot(tracked_sonsets, mapped_sonsets, label="predicted")
        plt.xlabel('tracked sonsets')
        plt.ylabel('mapped sonsets')
        plt.title('Sonsets')
        plt.legend()
        plt.savefig(alignment_fn.replace(".csv", ".png"))
        plt.clf()

        """plt.plot(gt_ponsets, gt_ponsets, label="gt")
        plt.plot(mapped_ponsets, tracked_ponsets, label="predicted")
        plt.xlabel('mapped ponsets')
        plt.ylabel('tracked ponsets')
        plt.title('Ponsets')
        plt.legend()
        plt.savefig(alignment_fn.replace(".csv", "-p.png"))
        plt.clf()"""
        
        config_fn = os.path.join(results_path, "config.yaml")
        with open(config_fn, 'w') as f:
            yaml.dump(mm.config, f, default_flow_style=False)  

    result = dict({'tracking_ratio': f'{tracking_ratio:.3f}',
                    'AAE (ms) ± σ': f'{results_in_seconds["mean"]:.3f}±{results_in_seconds["std"]:.3f}',
                    'MAE (ms)': f'{results_in_seconds["median"]:.3f}', 
                    'skewness (ms)': f'{results_in_seconds["skewness"]:.3f}',
                    'kurtosis (ms)': f'{results_in_seconds["kurtosis"]:.3f}',
                    '<10ms': f'{results_in_seconds["10ms"]*100:.1f}',
                    '<25ms': f'{results_in_seconds["25ms"]*100:.1f}',
                    '<50ms': f'{results_in_seconds["50ms"]*100:.1f}',
                    '<100ms': f'{results_in_seconds["100ms"]*100:.1f}',
                    '<500ms': f'{results_in_seconds["500ms"]*100:.1f}',
                    '<1000ms': f'{results_in_seconds["1000ms"]*100:.1f}',
                    '<2000ms': f'{results_in_seconds["2000ms"]*100:.1f}',
                    'count (ms)': f'{results_in_seconds["count"]}',
                    'pcr (ms)': f'{results_in_seconds["pcr"]:.3f}',
                    'AAE (beats) ± σ': f'{beat_results["mean"]:.2f}±{beat_results["std"]:.2f}',
                    'MAE (beats)': f'{beat_results["median"]:.2f}', 
                    '<0.05b': f'{beat_results["0.05b"]*100:.1f}',
                    '<0.1b': f'{beat_results["0.1b"]*100:.1f}',
                    '<0.5b': f'{beat_results["0.5b"]*100:.1f}',
                    '<1b': f'{beat_results["1b"]*100:.1f}',
                    '<2b': f'{beat_results["2b"]*100:.1f}',
                    'count (beats)': f'{beat_results["count"]}',
                    'pcr (beats)': f'{beat_results["pcr"]:.3f}',
                    
                })
    
    result_ext = dict({'tracking_ratio': tracking_ratio,
                    'AAE': results_in_seconds["mean"], 
                    'std': results_in_seconds["std"],
                    'MAE': results_in_seconds["median"], 
                    'skewness (ms)': results_in_seconds["skewness"],
                    'kurtosis (ms)': results_in_seconds["kurtosis"],
                    '%_lt_10ms': results_in_seconds["10ms"]*100,
                    '%_lt_25ms': results_in_seconds["25ms"]*100,
                    '%_lt_50ms': results_in_seconds["50ms"]*100,
                    '%_lt_100ms': results_in_seconds["100ms"]*100,
                    '%_lt_200ms': results_in_seconds["200ms"]*100,
                    '%_lt_300ms': results_in_seconds["300ms"]*100,
                    '%_lt_500ms': results_in_seconds["500ms"]*100,
                    '%_lt_1000ms': results_in_seconds["1000ms"]*100,
                    '%_lt_2000ms': results_in_seconds["2000ms"]*100,
                    'count (ms)': results_in_seconds["count"],
                    'pcr (ms)': results_in_seconds["pcr"],
                    'AAE (beats)': beat_results["mean"],
                    'MAE (beats)': beat_results["median"], 
                    'std (beats)': beat_results["std"],
                    'skewness (beats)': beat_results["skewness"],
                    'kurtosis (beats)': beat_results["kurtosis"],
                    '%_lt_<0.05b': beat_results["0.05b"]*100,
                    '%_lt_<0.1b': beat_results["0.1b"]*100,
                    '%_lt_<0.3b': beat_results["0.3b"]*100,
                    '%_lt_<0.5b': beat_results["0.5b"]*100,
                    '%_lt_<1b': beat_results["1b"]*100,
                    '%_lt_<2b': beat_results["2b"]*100,
                    'count (beats)': beat_results["count"],
                    'pcr (beats)': beat_results["pcr"],
                    })

    
    return result, result_ext, list_of_nans, predicted_alignments

# ============= run the alignment for each performance in a specified dataset =============

def run_tests_and_eval_by_dataset(args, kwargs):
    path = os.path.join(f'results')
    os.makedirs(path, exist_ok=True)

    keys = '88' if kwargs[args.input_type][args.method]["piano_range"] else '128' 
    if args.method == "hmm":
        name = f'{args.dataset}:{args.method}-{keys}-{args.input_type}-{kwargs[args.input_type][args.method]["tempo_model"].__name__}'
    else:
        name = f'{args.dataset}:{args.method}-{keys}-{args.input_type}'
        
    csv_path = os.path.join(path, name + '.csv')
    issue_path = os.path.join(path, 'issues-' + name + '.csv')
    
    dataset_dir = DATASET_DIR[args.dataset]
    metadata = pd.read_csv(METADATA_PATH[args.dataset])
    results = pd.DataFrame()
    issues = pd.DataFrame()

    if args.wandb:
        wandb.init(
        entity="darthalexus",
        project=f'matchmaker-symbolic',
        config = kwargs[args.input_type][args.method],
        name=name,
        )

    print(f'{name} - {SAVE_RESULTS=} - {SAVE_ALIGNMENTS=}')
    
    for i, row in enumerate(metadata.itertuples(), 1):
        #if i < 17:
        #    continue
        if args.dataset == "validation":
            dataset_dir = DATASET_DIR[row.dataset]

        score_xml = dataset_dir / row.xml_score
        #if args.dataset == 'vienna':
        #    score_xml = dataset_dir / Path('musicxml_corrected'+row.xml_score[8:])

        match = dataset_dir / row.match
        perf_midi = dataset_dir / row.midi_performance
        perf_audio = dataset_dir / row.audio_performance

        if args.input_type == "midi":
            perf = perf_midi
        else:
            perf = perf_audio
        
        result = dict({'composer': row.composer, 'title': row.title, 'performance': perf.stem})
        result_extended = dict({'composer': row.composer, 'title': row.title, 'performance': perf.stem})

        print(row.title)

        ######################################## the magic happens here: ########################################
        res, res_extended, list_of_nans, _ = align(solo_perf_fn=str(match), 
                                  reference_fn=str(score_xml), 
                                  perf_midi=str(perf_midi),
                                  args=args,
                                  kwargs=kwargs,
                                  save_alignments=SAVE_ALIGNMENTS)
        #########################################################################################################

        if issues.empty:
            issues = pd.DataFrame(columns=['composer', 'title', 'performance', '#Nans', 'Nans'])
        if len(list_of_nans) > 0:
            issues.loc[len(issues)] = dict({'composer': row.composer, 'title': row.title, 'performance': perf.stem, '#Nans': len(list_of_nans[0]), 'Nans': list_of_nans[0]})
            if SAVE_RESULTS:
                issues.to_csv(issue_path, index=False)

        result.update(res)
        result_extended.update(res_extended)
        
        if results.empty:
            results = pd.DataFrame(columns=result.keys())
            results_extended = pd.DataFrame(columns=result_extended.keys())
            
        results.loc[len(results)] = result
        results_extended.loc[len(results)] = result_extended
        
        print(results)
        if SAVE_RESULTS:
            results.to_csv(csv_path, index=False)

    final_res = dict({
                    'tracking ratio': f'{np.mean(results_extended["tracking_ratio"]):.3f}',
                    'AAE (ms) ± σ': f'{np.mean(results_extended["AAE"]):.3f}±{np.mean(results_extended["std"]):.3f}', 
                    'MAE (ms)': f'{np.mean(results_extended["MAE"]):.3f}', 
                    'skewness (ms)': f'{np.mean(results_extended["skewness (ms)"]):.3f}',
                    'kurtosis (ms)': f'{np.mean(results_extended["kurtosis (ms)"]):.3f}',
                    '<10ms': f'{np.mean(results_extended["%_lt_10ms"]):.1f}',
                    '<25ms': f'{np.mean(results_extended["%_lt_25ms"]):.1f}',
                    '<50ms': f'{np.mean(results_extended["%_lt_50ms"]):.1f}',
                    '<100ms': f'{np.mean(results_extended["%_lt_100ms"]):.1f}',
                    '<500ms': f'{np.mean(results_extended["%_lt_500ms"]):.1f}',
                    '<1000ms': f'{np.mean(results_extended["%_lt_1000ms"]):.1f}',
                    '<2000ms': f'{np.mean(results_extended["%_lt_2000ms"]):.1f}',
                    'count (ms)': f'{np.mean(results_extended["count (ms)"]):.3f}',
                    'pcr (ms)': f'{np.mean(results_extended["pcr (ms)"]):.3f}',
                    'AAE (beats) ± σ': f'{np.mean(results_extended["AAE (beats)"]):.3f}±{np.mean(results_extended["std (beats)"]):.3f}', 
                    'MAE (beats)': f'{np.mean(results_extended["MAE (beats)"]):.3f}', 
                    'skewness (beats)': f'{np.mean(results_extended["skewness (beats)"]):.3f}',
                    'kurtosis (beats)': f'{np.mean(results_extended["kurtosis (beats)"]):.3f}',
                    '<0.05b': f'{np.mean(results_extended["%_lt_<0.05b"]):.1f}',
                    '<0.1b': f'{np.mean(results_extended["%_lt_<0.1b"]):.1f}',
                    '<0.5b': f'{np.mean(results_extended["%_lt_<0.5b"]):.1f}',
                    '<1b': f'{np.mean(results_extended["%_lt_<1b"]):.1f}',
                    '<2b': f'{np.mean(results_extended["%_lt_<2b"]):.1f}',
                    'count (beats)': f'{np.mean(results_extended["count (beats)"]):.3f}',
                    'pcr (beats)': f'{np.mean(results_extended["pcr (beats)"]):.3f}',
                    })

    final_results = pd.DataFrame(final_res, index=[args.dataset])

    print(f'finished.\n\nFINAL_RESULTS:\n{name} - SAVE_RESULTS={SAVE_RESULTS}\n')
    print(final_results)
    if SAVE_RESULTS:
        final_csv_path = os.path.join(path,'final')
        os.makedirs(final_csv_path, exist_ok=True)
        final_csv_path = os.path.join(final_csv_path, f'{name}.csv')
        final_results.to_csv(final_csv_path, index=False)

    if args.wandb:
        individual_table = wandb.Table(results.columns, results)
        wandb.log({f'individual_results - {name}': individual_table})
        final_table = wandb.Table(final_results.columns, final_results)
        wandb.log({f'final_results - {name}': final_table})
        wandb.finish()
    
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run an offline alignment experiment."# For further configs, use config/experiment.yaml"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["validation", "asap", "vienna", "batik"],
        default="asap",
        help="Dataset to use (asap, vienna, or batik)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hmm", "outerhmm", "pthmm", "arzt", "dixon"],
        default="hmm",
        help="Method to use (hmm, dixon, arzt, or offline)",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["audio", "midi"],
        default="midi",
        help="Input type to use (audio or midi)",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="report results to wandb", default=False
    )
    args = parser.parse_args()

    run_tests_and_eval_by_dataset(args, kwargs=KWARGS)