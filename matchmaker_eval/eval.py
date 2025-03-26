import json
import time
import traceback
from pathlib import Path
from typing import Optional, Union

import librosa
import mido
import numpy as np
import pandas as pd
import partitura as pt
import scipy
from matchmaker import Matchmaker
from matchmaker.utils.eval import get_evaluation_results
from numpy.typing import NDArray
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from utils import MatchmakerEvalConfig

TOLERANCES = [50, 100, 300, 500, 1000, 2000]
METRICS = ["mean", "median", "std", "skewness", "kurtosis"] + [
    f"{t}ms" for t in TOLERANCES
]


def transfer_positions(wp, perf_annots, frame_rate):
    perf_annots_frame = np.round(perf_annots * frame_rate)
    x, y = wp[0], wp[1]
    return (
        scipy.interpolate.interp1d(y, x, kind="linear")(perf_annots_frame) / frame_rate
    )


def _get_DLNCO_features_from_audio(audio, feature_sequence_length, Fs, feature_rate):
    f_pitch_onset = audio_to_pitch_onset_features(f_audio=audio, Fs=Fs)
    f_DLNCO = pitch_onset_features_to_DLNCO(
        f_peaks=f_pitch_onset,
        feature_rate=feature_rate,
        feature_sequence_length=feature_sequence_length,
        visualize=False,
    )

    return f_DLNCO


def run_offline_alignment(
    score_path: Path,
    perf_path: Path,
    perf_beat_ann,
    config,
    use_musical_beat,
):
    mm = Matchmaker(
        score_file=score_path,
        performance_file=perf_path,
        input_type="audio",
    )
    # read audio
    audio_1 = mm.score_audio
    audio_2, _ = librosa.load(perf_path.as_posix(), sr=config.sample_rate)

    # extract chroma cens features
    f_chroma_librosa_1 = librosa.feature.chroma_cens(
        y=audio_1,
        sr=config.sample_rate,
        hop_length=config.hop_length,
    )
    f_chroma_librosa_2 = librosa.feature.chroma_cens(
        y=audio_2,
        sr=config.sample_rate,
        hop_length=config.hop_length,
    )
    # generate DLNCO features
    f_DLNCO_1 = _get_DLNCO_features_from_audio(
        audio_1, f_chroma_librosa_1.shape[1], config.sample_rate, config.frame_rate
    )

    f_DLNCO_2 = _get_DLNCO_features_from_audio(
        audio_2, f_chroma_librosa_2.shape[1], config.sample_rate, config.frame_rate
    )

    wp = sync_via_mrmsdtw(
        f_chroma1=f_chroma_librosa_1,
        f_onset1=f_DLNCO_1,
        f_chroma2=f_chroma_librosa_2,
        f_onset2=f_DLNCO_2,
        input_feature_rate=config.frame_rate,
        verbose=False,
    )
    # wp = compute_strict_alignment_path_mask(wp.T).T
    score_annots = mm.build_score_annotations(musical_beat=use_musical_beat)
    perf_annots = np.loadtxt(fname=perf_beat_ann, delimiter="\t", usecols=0)

    min_length = min(len(score_annots), len(perf_annots))
    score_annots = score_annots[:min_length]
    perf_annots = perf_annots[:min_length]

    predicted_score_annots = transfer_positions(
        wp,
        perf_annots,
        config.frame_rate,
    )
    results = get_evaluation_results(
        score_annots,
        predicted_score_annots,
        TOLERANCES,
    )
    return results


def regenerate_tempo_adjusted_midi(midi_path: Path, target_duration: float) -> Path:
    mid = mido.MidiFile(midi_path.as_posix())
    ratio = target_duration / mid.length
    print(f"Tempo ratio (target audio / midi length): {ratio}")

    has_set_tempo = False
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                has_set_tempo = True
                print(f"Original tempo: {mido.tempo2bpm(msg.tempo)}, {msg.tempo}")
                new_tempo = mido.bpm2tempo(mido.tempo2bpm(msg.tempo) / ratio)
                print(f"New tempo: {mido.tempo2bpm(new_tempo)}, {new_tempo}")
                msg.tempo = int(new_tempo)
    if not has_set_tempo:
        # If the track does not have a 'set_tempo' message, add one with the default BPM
        default_bpm = 120
        print(f"[Default] Original tempo: {default_bpm}, 500000")
        new_tempo = mido.bpm2tempo(default_bpm / ratio)
        print(f"New tempo: {new_tempo}")
        for track in mid.tracks:
            track.insert(0, mido.MetaMessage("set_tempo", tempo=int(new_tempo)))

    new_midi_path = midi_path.with_stem("midi_score_adjusted")
    mid.save(new_midi_path)
    return new_midi_path


def run_score_following(
    score_path: Path,
    perf_path: Union[Path, str],
    perf_beat_ann: Path,
    config: MatchmakerEvalConfig,
    use_musical_beat: bool = False,
    dry_run: bool = False,
    save_dir: Optional[Path] = None,
    run_name: str = "",
) -> NDArray[np.float32]:
    """
    Run score following on the score audio and the performance file.

    Parameters
    ----------
    score_path : Path
        path to the score file (.mid or .xml).
    perf_path : Path or str
        path to the performance file (.wav or .mid), or empty string for live performance mode.

    Returns
    -------
    warping_path: np.ndarray [shape=(2, T)]
        Resulting warping path with pairs of indices of the reference and target audio.
    """
    mm = Matchmaker(
        score_file=score_path,
        performance_file=perf_path,
        input_type="audio",
        method=config.method,
        distance_func=config.distance_func,
        frame_rate=config.frame_rate,
        sample_rate=config.sample_rate,
        feature_type=config.feature_type,
        wait=False,
    )

    try:
        alignment_positions = list(mm.run())
    except Exception as e:
        print(f"Error: {type(e)}, {e}")
        traceback.print_exc()
        mm._has_run = True

    results = mm.run_evaluation(
        perf_beat_ann,
        tolerances=TOLERANCES,
        musical_beat=use_musical_beat,
        debug=not dry_run,
        save_dir=save_dir,
        run_name=run_name,
    )
    print(f"RESULTS: {json.dumps(results, indent=4)}")
    return results
