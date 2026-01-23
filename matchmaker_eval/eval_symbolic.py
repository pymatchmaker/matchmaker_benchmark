
# -*- coding: utf-8 -*-
import argparse
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict
from pathlib import Path
import wandb
import sys
sys.setrecursionlimit(2**31 -1)

import pdb

import pandas as pd

import numpy as np
import partitura as pt
import matplotlib.pyplot as plt
import yaml

import scipy

from partitura.performance import Performance, PerformedPart, PerformanceLike
from partitura.score import Score, Part, ScoreLike, merge_parts
from partitura.utils.misc import PathLike

from partitura.utils.music import performance_from_part
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment, to_matched_score

from partitura_utils import (
    partitura_to_framed_midi_custom as partitura_to_framed_midi,
)

from matchmaker import Matchmaker

from matchmaker.utils.symbolic import (
    framed_midi_messages_from_performance,
    midi_messages_from_performance,
)
from matchmaker.features.midi import PitchIOIProcessor, PitchProcessor, PianoRollProcessor

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
      "asap": Path("/home/alexander-neuhauser/datasets/asap-dataset-matchmaker"),
     "batik": Path("/home/alexander-neuhauser/datasets/Batik_Audio"),
    "vienna": Path("/home/alexander-neuhauser/datasets/vienna4x22"),
}
METADATA_PATH = {
    "asap": "../ismir2025_matchmaker/data/metadata-asap.csv",
    "batik": "../ismir2025_matchmaker/data/metadata-batik.csv",
    "vienna":"../ismir2025_matchmaker/data/metadata-vienna.csv",
}

DEFAULT_LOCAL_COST = "Manhattan"
WINDOW_SIZE = 100
STEP_SIZE = 5
START_WINDOW_SIZE = 60
POLLING_PERIOD = None#0.01

CONFIG = dict(
    follower_type="hmm",
    ioi_precision=0.1,
    gumbel_transition_matrix_scale=1.5,
    init_bp=0.8,
    trans_par=2,
    trans_var=0.5,
    obs_var=1,
    init_var=0.5,
    tempo_model=KalmanTempoModel,
    piano_range=True,
    save_results = True,
)

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
        #"pitch_class_pianoroll": PitchClassPianoRollProcessor,
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

# ============= main alignment function for a single performance =============

def align(
    solo_perf_fn: PathLike,
    reference_fn: Union[PathLike, List[PathLike]],
    perf_midi,
    tempo_model,
    method,
    unfold = False,
    piano_range = False,
):
    solo_ppart, ptime_to_stime_map, stime_to_ptime_map = setup_match_file([solo_perf_fn])[0]

    
    _, alignment, solo_score = pt.load_match(
                    filename=solo_perf_fn,
                    create_score=True,
                    first_note_at_zero=True,
                )

    
    score = pt.load_musicxml(reference_fn)
    if unfold:
        score = pt.score.unfold_part_maximal(score)
    
    matched_array = to_matched_score(score,#.note_array(), 
                                     solo_ppart, alignment)[0]['onset']
    processor_kwargs = dict({'piano_range': piano_range})

    if method == "arzt":
        processor_name = 'pianoroll'
        POLLING_PERIOD = None
        
    elif method == "hmm":
        processor_name = 'pitch_ioi'
        POLLING_PERIOD = 0.01
        processor_kwargs["return_pitch_list"] = True
        
    elif method == "pthmm":
        processor_name = 'pitch'
        POLLING_PERIOD = 0.01
        processor_kwargs["return_pitch_list"] = False
    
    elif method == "outerhmm":
        processor_name = 'pitch_ioi'
        POLLING_PERIOD = None
        processor_kwargs["return_pitch_list"] = False
        
    input_signal, frame_times = compute_features_from_symbolic(ref_info=solo_ppart,processor_name=processor_name, 
                                                                processor_kwargs=processor_kwargs,
                                                                polling_period=POLLING_PERIOD)
    

    mm = Matchmaker(
    score_file=reference_fn, # the score file (musicxml) is used as reference feature
    performance_file=perf_midi,
    input_type=config["input_type"],
    tempo_model=tempo_model,
    feature_type="pitch_ioi",
    method=method,
    piano_range=piano_range,
    )
    

    tracked_sonsets, tracked_ponsets = [], []

    
    for i, frame in enumerate(input_signal):
        if frame is not None:
            current_state = mm.score_follower(frame)
            mm_score_position = mm.score_follower.state_space[current_state] # (= current_position)

        else:
            mm_score_position = None

        if mm_score_position != None:
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
        total_length=len(tracked_sonsets),
        tolerances=tolerances_in_seconds,
        in_seconds=True,
    )

    mapped_sonsets = ptime_to_stime_map(tracked_ponsets)
    beat_tolerances = [0.05, 0.1, 0.3, 0.5, 1, 2]
    beat_results = get_evaluation_results(
        tracked_sonsets,
        mapped_sonsets,
        total_length=len(tracked_sonsets),
        tolerances=beat_tolerances,
        in_seconds=False,
    )

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
                    'AAE (beats) ± σ': f'{beat_results["mean"]:.2f}±{beat_results["std"]:.2f}',
                    'MAE (beats)': f'{beat_results["median"]:.2f}', 
                    '<0.05b': f'{beat_results["0.05b"]*100:.1f}',
                    '<0.1b': f'{beat_results["0.1b"]*100:.1f}',
                    '<0.5b': f'{beat_results["0.5b"]*100:.1f}',
                    '<1b': f'{beat_results["1b"]*100:.1f}',
                    '<2b': f'{beat_results["2b"]*100:.1f}', 
                })
    
    result_ext = dict({'tracking_ratio': f'{tracking_ratio:.3f}',
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
                    })

    return result, result_ext, list_of_nans

# ============= run the alignment for each performance in a specified dataset =============

def run_tests_and_eval_by_dataset(args, config):
    path = os.path.join(f'results/experiment/hmm-preprocessing/compute_ft_fr_symb-has_insertions_but_no_eval-pipeline-differences', f'{config["dataset"]}')
    os.makedirs(path, exist_ok=True)

    keys = '88' if config["piano_range"] else '128' 
    if config["method"] == "hmm":
        name = f'{config["method"]}-{keys}-{config["input_type"]}-{config["tempo_model"].__name__}'
    else:
        name = f'{config["method"]}-{keys}-{config["input_type"]}'
        
    csv_path = os.path.join(path, name + '.csv')
    issue_path = os.path.join(path, 'issues-' + name + '.csv')
    
    dataset_dir = DATASET_DIR[config["dataset"]]
    metadata = pd.read_csv(METADATA_PATH[config["dataset"]])
    results = pd.DataFrame()
    issues = pd.DataFrame()
    use_musical_beat = config["dataset"] == "asap"

    if args.wandb:
        wandb.init(
        entity="darthalexus",
        project=f'matchmaker',
        config = config,
        name=name,
        )

    print(f'{name} - SAVE_RESULTS={config["save_results"]}')
    
    for i, row in enumerate(metadata.itertuples(), 1):
        score_xml = dataset_dir / row.xml_score
        if config["dataset"] == 'vienna':
            score_xml = dataset_dir / Path('musicxml_corrected'+row.xml_score[8:])
            #print(score_xml)

        match = dataset_dir / row.match
        perf_midi = dataset_dir / row.midi_performance
        perf_audio = dataset_dir / row.audio_performance

        if config["input_type"] == "midi":
            perf = perf_midi
        else:
            perf = perf_audio
        
        result = dict({'composer': row.composer, 'title': row.title, 'performance': perf.stem})
        result_extended = dict({'composer': row.composer, 'title': row.title, 'performance': perf.stem})

        print(row.title)

        ######################################## the magic happens here: ########################################
        res, res_extended, list_of_nans = align(solo_perf_fn=str(match), 
                                  reference_fn=str(score_xml), 
                                  perf_midi=str(perf_midi), 
                                  tempo_model=config["tempo_model"], 
                                  method=config["method"], 
                                  piano_range=config["piano_range"],
                                  #filtered=config["filtered"],
                                  unfold = False if config["dataset"] == "vienna" else True)
        #########################################################################################################

        if issues.empty:
            issues = pd.DataFrame(columns=['composer', 'title', 'performance', '#Nans', 'Nans'])
        if len(list_of_nans) > 0:
            issues.loc[len(issues)] = dict({'composer': row.composer, 'title': row.title, 'performance': perf.stem, '#Nans': len(list_of_nans[0]), 'Nans': list_of_nans[0]})
            if config["save_results"]:
                issues.to_csv(issue_path, index=False)

        result.update(res)
        result_extended.update(res_extended)
        
        if results.empty:
            results = pd.DataFrame(columns=result.keys())
            results_extended = pd.DataFrame(columns=result_extended.keys())
            
        results.loc[len(results)] = result
        results_extended.loc[len(results)] = result_extended
        
        print(results)
        if config["save_results"]:
            results.to_csv(csv_path, index=False)

    final_res = dict({
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
                    'AAE (beats) ± σ': f'{np.mean(results_extended["AAE (beats)"]):.3f}±{np.mean(results_extended["std (beats)"]):.3f}', 
                    'MAE (beats)': f'{np.mean(results_extended["MAE (beats)"]):.3f}', 
                    'skewness (beats)': f'{np.mean(results_extended["skewness (beats)"]):.3f}',
                    'kurtosis (beats)': f'{np.mean(results_extended["kurtosis (beats)"]):.3f}',
                    '<0.05b': f'{np.mean(results_extended["%_lt_<0.05b"]):.1f}',
                    '<0.1b': f'{np.mean(results_extended["%_lt_<0.1b"]):.1f}',
                    '<0.5b': f'{np.mean(results_extended["%_lt_<0.5b"]):.1f}',
                    '<1b': f'{np.mean(results_extended["%_lt_<1b"]):.1f}',
                    '<2b': f'{np.mean(results_extended["%_lt_<2b"]):.1f}',
                    })

    final_results = pd.DataFrame(final_res, index=[config["dataset"]])

    print(f'finished.\n\nFINAL_RESULTS:\n{name} - SAVE_RESULTS={config["save_results"]}\n')
    print(final_results)
    if config["save_results"]:
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
    '''
    parser.add_argument(
        "--solo",
        "-s",
        help="Input Solo performance (as a match file)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--reference",
        "-r",
        help="Reference (as a list of match files)",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Config file (YAML)",
        type=str,
        default=None,
    )
    '''
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["asap", "vienna", "batik"],
        default="asap",
        help="Dataset to use (asap, vienna, or batik)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hmm", "outerhmm", "pthmm", "arzt"],
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
    parser.add_argument(
        "--wandb_ext", action="store_true", help="report extensive results to wandb", default=False
    )    

    args = parser.parse_args()

    print('Use this script together with the feature/PitchIOIHMM branch of the matchmaker repository.')

    config = CONFIG
    config["method"] = args.method
    config["input_type"] = args.input_type
    config["dataset"] = args.dataset

    config["piano_range"] = True
    run_tests_and_eval_by_dataset(args, config)
    raise KeyboardInterrupt
    
    for method in ["pthmm", "outerhmm", "hmm"]:
        config["method"] = method
        print(f'METHOD = {config["method"]}')
        for dataset in ["vienna", "asap", "batik"]:
            config["dataset"] = dataset
            for piano_range in [True, False]:
                config["piano_range"] = piano_range
                if config["method"] == "hmm":
                    for tempo_model in [#KalmanTempoSyncModel, 
                                        #ReactiveSyncModel,
                                        #MovingAverageSyncModel,
                                        #LinearSyncModel,
                                        #JointAdaptationAnticipationSyncModel,
                                        KalmanTempoModel,
                                        LinearTempoModel,
                                        MovingAverageTempoModel,
                                        ReactiveTempoModel]:
                        config["tempo_model"] = tempo_model
                        print(f'{config["tempo_model"]=}')
                        results = run_tests_and_eval_by_dataset(args, config)
                else:
                    results = run_tests_and_eval_by_dataset(args, config)
