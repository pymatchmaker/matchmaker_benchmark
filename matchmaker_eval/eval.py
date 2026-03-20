import json
import traceback
from pathlib import Path
from typing import Callable, Optional, Union

import librosa
import numpy as np
import pandas as pd
import partitura as pt
import scipy
from matchmaker import Matchmaker
from matchmaker.utils.eval import get_evaluation_results
from numpy.typing import NDArray
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment

from utils import TOLERANCES_IN_BEATS, TOLERANCES_IN_MS, MatchmakerEvalConfig
from verify_tracking import check_tracking, plot_tracking


def parse_match_file_for_note_onsets(
    match_file: Union[str, Path],
    level: str = "note",
    score_onset_beats: Optional[np.ndarray] = None,
    onset_aggregation_fun: Callable = np.min,
) -> np.ndarray:
    """
    Parse a match file and extract performed note onset times in seconds
    using partitura's alignment mapping.

    Parameters
    ----------
    match_file : Union[str, Path]
        Path to the match file
    level : str
        "note" for note-level onsets, "beat" for beat-level onsets
    score_onset_beats : np.ndarray, optional
        Onset beat positions from the MusicXML score. If provided, these
        are mapped through stime_to_ptime_map instead of using the match
        file's own score onsets.

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
        onset_aggregation_fun=onset_aggregation_fun,
    )

    if score_onset_beats is not None:
        performed_times = stime_to_ptime_map(score_onset_beats)
    elif level == "beat":
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


def parse_annotation_csv(
    annotation_file: Union[str, Path],
) -> np.ndarray:
    """
    Parse a note annotation file.

    Supports:
    - CSV with a TIME column header (e.g., chorale-bricks)
    - Tab-delimited without header, first column = onset time (e.g., URMP Notes)
    """
    annotation_file = Path(annotation_file)
    with open(annotation_file, "r") as f:
        first_line = f.readline().strip()

    if "TIME" in first_line.upper():
        df = pd.read_csv(annotation_file)
        # Find the time column: exact "TIME" or fallback to first column containing "time"
        if "TIME" in df.columns:
            return df["TIME"].values
        time_cols = [c for c in df.columns if "time" in c.lower()]
        if time_cols:
            return df[time_cols[0]].values
        # Last resort: use the first column
        return df.iloc[:, 0].values
    else:
        # Whitespace-delimited, no header; first column = onset time in seconds
        try:
            data = np.loadtxt(annotation_file)
        except ValueError:
            # Has non-numeric columns (e.g., ASAP beat annotations); read first column only
            data = np.loadtxt(annotation_file, usecols=(0,))
        if data.ndim == 1:
            return data  # single column: all values are onset times
        return data[:, 0]


def transfer_offline_positions(wp, perf_annots, frame_rate):
    perf_annots_frame = np.round(perf_annots * frame_rate)
    x, y = wp[0], wp[1]
    return (
        scipy.interpolate.interp1d(y, x, kind="linear")(perf_annots_frame) / frame_rate
    )


def _get_DLNCO_features_from_audio(audio, feature_sequence_length, Fs, feature_rate):
    from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
    from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features

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
    match_file: Optional[Path],
    config,
    use_musical_beat,
    perf_annotations: Optional[np.ndarray] = None,
    granularity: str = "note",
):
    mm = Matchmaker(
        score_file=score_path,
        performance_file=perf_path,
        input_type="audio",
        unfold_score=True,
    )
    # read audio
    audio_1 = mm.score_audio
    audio_2, _ = librosa.load(perf_path.as_posix(), sr=config.sample_rate)

    # extract CQT log-magnitude features
    cqt_1 = np.abs(
        librosa.cqt(
            y=audio_1,
            sr=config.sample_rate,
            hop_length=config.hop_length,
            n_bins=84,
            bins_per_octave=12,
        )
    )
    cqt_2 = np.abs(
        librosa.cqt(
            y=audio_2,
            sr=config.sample_rate,
            hop_length=config.hop_length,
            n_bins=84,
            bins_per_octave=12,
        )
    )
    cqt_1_db = librosa.amplitude_to_db(cqt_1, ref=np.max)
    cqt_2_db = librosa.amplitude_to_db(cqt_2, ref=np.max)
    # L2 normalize per frame
    cqt_1_norm = cqt_1_db / (np.linalg.norm(cqt_1_db, axis=0, keepdims=True) + 1e-10)
    cqt_2_norm = cqt_2_db / (np.linalg.norm(cqt_2_db, axis=0, keepdims=True) + 1e-10)

    # librosa DTW with cosine distance
    _, wp_raw = librosa.sequence.dtw(
        X=cqt_1_norm, Y=cqt_2_norm, metric="cosine", backtrack=True
    )
    wp = wp_raw[::-1].T  # reverse and transpose to [2, T] format
    # wp = compute_strict_alignment_path_mask(wp.T).T
    score_annots = mm.build_score_annotations(
        level=granularity, musical_beat=use_musical_beat, return_type="seconds"
    )

    # Determine how many intro beats to skip in performance annotations.
    # build_score_annotations uses np.arange(ceil(min_onset), floor(max_onset)+1),
    # so when the score's first note starts at beat N>0 (e.g. piano intro before
    # the scored section), the annotation CSV has N extra beats at the beginning
    # that don't correspond to any score beat.
    # However, we must not skip more beats than the difference between perf and
    # score counts — if they already match (e.g. both 108), offset must be 0.
    na = mm.score_part.note_array()
    start_beat = max(0, int(np.ceil(np.unique(na["onset_beat"]).min())))
    n_perf = len(perf_annotations) if perf_annotations is not None else 0
    n_score = len(score_annots)
    intro_offset = max(0, min(start_beat, n_perf - n_score))

    if perf_annotations is not None:
        perf_annots = perf_annotations[intro_offset:]
    elif match_file is not None:
        # Pass score_onset_beats to ensure alignment with build_score_annotations,
        # especially when ignore_invisible_objects reduces the note count.
        score_onset_beats = mm.build_score_annotations(
            level=granularity, musical_beat=use_musical_beat, return_type="beats"
        )
        perf_annots = parse_match_file_for_note_onsets(
            match_file, level=granularity, score_onset_beats=score_onset_beats
        )
    else:
        raise ValueError("Either match_file or perf_annotations must be provided")

    min_length = min(len(score_annots), len(perf_annots))
    score_annots = score_annots[:min_length]
    perf_annots = perf_annots[:min_length]

    # Filter out invalid pairs (NaN or outside warping path range)
    max_perf_frame = wp[1].max()
    max_perf_time = max_perf_frame / config.frame_rate
    valid = (
        np.isfinite(perf_annots)
        & np.isfinite(score_annots)
        & (perf_annots <= max_perf_time)
        & (perf_annots >= 0)
    )
    score_annots = score_annots[valid]
    perf_annots = perf_annots[valid]

    predicted_score_annots = transfer_offline_positions(
        wp,
        perf_annots,
        config.frame_rate,
    )
    results = get_evaluation_results(
        score_annots,
        predicted_score_annots,
        total_counts=len(score_annots),
        tolerances=TOLERANCES_IN_MS,
    )
    return results


def run_score_following(
    score_path: Path,
    perf_path: Union[Path, str],
    config: MatchmakerEvalConfig,
    use_musical_beat: bool = False,
    dry_run: bool = False,
    save_dir: Optional[Path] = None,
    run_name: str = "",
    match_file: Optional[Path] = None,
    perf_annotations: Optional[np.ndarray] = None,
    matchmaker_kwargs: Optional[dict] = None,
    granularity: str = "note",
) -> NDArray[np.float32]:
    """
    Run score following on the score audio and the performance file.

    Parameters
    ----------
    score_path : Path
        path to the score file (.mid or .xml).
    perf_path : Path or str
        path to the performance file (.wav or .mid), or empty string for live performance mode.
    config : MatchmakerEvalConfig
        configuration for the evaluation.
    use_musical_beat : bool
        whether to use musical beat for the evaluation.
    dry_run : bool
        whether to run the evaluation in dry run mode.
    save_dir : Optional[Path]
        directory to save the evaluation results.
    run_name : str
        name of the run.
    match_file : Optional[Path]
        path to the match file.
    perf_annotations : Optional[np.ndarray]
        performance annotations.

    Returns
    -------
    results : dict
        evaluation results.
    """
    extra = {}
    if matchmaker_kwargs is not None:
        extra["kwargs"] = matchmaker_kwargs

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
        unfold_score=True,
        **extra,
    )

    try:
        alignment_positions = list(mm.run())
    except Exception as e:
        print(f"Error during mm.run(): {type(e)}, {e}")
        traceback.print_exc()
        mm._has_run = True

    # Check if warping path is empty (e.g. audio_outerhmm failed to produce alignment)
    wp = mm.score_follower.warping_path
    if wp is None or (hasattr(wp, "size") and wp.size == 0) or len(wp) == 0:
        raise RuntimeError(
            f"Empty warping path after mm.run() — score follower produced no alignment"
        )

    # Parse performance annotations from match file or use pre-parsed annotations
    if perf_annotations is None:
        score_onset_beats = mm.build_score_annotations(
            level="note", musical_beat=use_musical_beat, return_type="beats"
        )
        perf_annotations = parse_match_file_for_note_onsets(
            match_file, score_onset_beats=score_onset_beats
        )

    # Performance domain evaluation (ms-based tolerances)
    results = mm.run_evaluation(
        perf_annotations,
        tolerances=TOLERANCES_IN_MS,
        musical_beat=use_musical_beat,
        debug=not dry_run,
        save_dir=save_dir,
        run_name=run_name,
        level=granularity,
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
        level=granularity,
    )

    beat_tolerance_keys = {f"{t}b" for t in TOLERANCES_IN_BEATS}
    for key, value in score_results.items():
        if key in beat_tolerance_keys:
            results[key] = value
        else:
            results[f"{key}_b"] = value

    # Tracking verification + plot
    try:
        wp_for_check = wp.T  # (2, T) → (T, 2)

        score_annots_for_gt = mm.build_score_annotations(
            level=granularity, musical_beat=use_musical_beat, return_type="beats"
        )

        min_len = min(len(score_annots_for_gt), len(perf_annotations))
        gt_for_check = np.column_stack(
            [
                score_annots_for_gt[:min_len],
                perf_annotations[:min_len],
            ]
        )
        tracking = check_tracking(
            wp_for_check,
            gt_for_check,
            config.frame_rate,
            mode="beat",
        )
        results["tracked"] = tracking["tracked"]
        results["max_deviation"] = tracking["max_deviation"]
        results["n_failed_segments"] = tracking["n_failed"]

        # Save tracking plot
        if save_dir is not None and run_name:
            plot_tracking(
                wp_for_check,
                gt_for_check,
                config.frame_rate,
                title=f"{config.method} #{run_name}",
                save_path=Path(save_dir) / f"tracking_{run_name}.png",
                mode="beat",
            )
    except Exception:
        pass

    # Overwrite individual JSON with merged ms + beat results
    if save_dir is not None and run_name and not dry_run:
        with open(Path(save_dir) / f"{run_name}.json", "w") as f:
            json.dump(results, f, indent=4)

    print(f"RESULTS: {json.dumps(results, indent=4)}")
    return results
