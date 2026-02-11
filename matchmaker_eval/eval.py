import json
import traceback
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import partitura as pt
import scipy
from matchmaker import Matchmaker
from matchmaker.utils.eval import get_evaluation_results
from numpy.typing import NDArray
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from utils import MatchmakerEvalConfig

TOLERANCES_IN_MS = [50, 100, 300, 500, 1000, 2000]
TOLERANCES_IN_BEATS = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
METRICS = (
    ["mean", "median", "std", "skewness", "kurtosis"]
    + [f"{t}ms" for t in TOLERANCES_IN_MS]
    + [f"{t}b" for t in TOLERANCES_IN_BEATS]
)


def parse_match_file_for_note_onsets(
    match_file: Union[str, Path], level: str = "note"
) -> np.ndarray:
    """
    Parse a match file and extract performed note onset times in seconds
    using partitura's alignment mapping.

    Uses partitura to properly load the match file and create a mapping
    from score time to performance time based on the alignment.

    Parameters
    ----------
    match_file : Union[str, Path]
        Path to the match file
    level : str
        "note" for note-level onsets, "beat" for beat-level onsets

    Returns
    -------
    np.ndarray
        Array of performed times in seconds, corresponding to unique
        score onset positions.
    """
    match_file = Path(match_file)

    perf, alignment, score = pt.load_match(
        filename=str(match_file),
        create_score=True,
    )

    pnote_array = perf.note_array()
    snote_array = score.note_array()

    ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
        ppart_or_note_array=pnote_array,
        spart_or_note_array=snote_array,
        alignment=alignment,
    )

    # Extract annotations based on level
    if level == "beat":
        start_beat = np.ceil(snote_array["onset_beat"].min())
        end_beat = np.floor(snote_array["onset_beat"].max())
        beats = np.arange(start_beat, end_beat + 1)
        performed_times = stime_to_ptime_map(beats)
    elif level == "note":
        unique_onsets = np.unique(snote_array["onset_beat"])
        performed_times = stime_to_ptime_map(unique_onsets)
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'beat' or 'note'.")

    return performed_times


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
    match_file: Path,
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
    perf_annots = parse_match_file_for_note_onsets(match_file)

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
        TOLERANCES_IN_MS,
    )
    return results


def run_score_following(
    score_path: Path,
    perf_path: Union[Path, str],
    match_file: Path,
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
    match_file : Path
        path to the match file.

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
        auto_adjust_tempo=getattr(config, "adjust_tempo", False),
    )

    try:
        alignment_positions = list(mm.run())
    except Exception as e:
        print(f"Error: {type(e)}, {e}")
        traceback.print_exc()
        mm._has_run = True

    # Parse performance annotations from match file
    perf_annotations = parse_match_file_for_note_onsets(match_file)

    # Performance domain evaluation (ms-based tolerances)
    results = mm.run_evaluation(
        perf_annotations,
        tolerances=TOLERANCES_IN_MS,
        musical_beat=use_musical_beat,
        debug=not dry_run,
        save_dir=save_dir,
        run_name=run_name,
        level="note",
    )

    # Score domain evaluation (beat-based tolerances)
    score_results = mm.run_evaluation(
        perf_annotations,
        tolerances=TOLERANCES_IN_BEATS,
        musical_beat=use_musical_beat,
        domain="score",
        debug=False,
        save_dir=save_dir,
        run_name=run_name,
        level="note",
    )

    beat_tolerance_keys = {f"{t}b" for t in TOLERANCES_IN_BEATS}
    for key, value in score_results.items():
        if key in beat_tolerance_keys:
            results[key] = value
        else:
            results[f"{key}_b"] = value

    print(f"RESULTS: {json.dumps(results, indent=4)}")
    return results
