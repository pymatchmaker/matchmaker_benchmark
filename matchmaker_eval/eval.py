"""Single-piece alignment functions for audio and symbolic score following.

Audio functions (used by test_audio.py):
  - parse_match_file_for_note_onsets
  - run_score_following (input_type="audio")
  - run_offline_alignment

Symbolic functions (used by test_symbolic.py):
  - run_score_following (input_type="midi", HMM methods)
  - run_event_alignment (event-level OLTW for MIDI)
  - build_gt (ground truth from match file)
"""

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
from matchmaker.utils.eval import (
    evaluate_alignment,
    get_evaluation_results,
    resolve_gt,
    transfer_positions,
)
from numpy.typing import NDArray
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment

from utils import (
    TOLERANCES_IN_BEATS,
    TOLERANCES_IN_MS,
    AudioEvalConfig,
    save_debug_results,
)
from verify_tracking import check_tracking, plot_tracking


# ---------------------------------------------------------------------------
# Evaluation against ground truth (post-processing of a completed Matchmaker run)
# ---------------------------------------------------------------------------


def run_evaluation(
    mm: Matchmaker,
    gt: Union[str, Path, np.ndarray] = None,
    tolerances: Optional[list] = None,
    musical_beat: bool = False,
    debug: bool = False,
    save_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
    domain: str = "score",
    plot_dist_matrix: bool = True,
    make_plot: bool = True,
    level: str = "note",
) -> dict:
    """Evaluate a completed Matchmaker run against ground truth.

    When domain="score" (default), returns beat-based metrics as primary
    and ms-based metrics under "ms" key. When domain="performance",
    returns ms-based metrics only (legacy behavior).

    Parameters
    ----------
    mm : Matchmaker
        A Matchmaker instance that has already been run (``mm.run()``).
    gt : PathLike or np.ndarray
        Ground truth: a .match file, a .tsv (perf_sec, score_beat) file,
        or an (N, 2) array of [perf_sec, score_beat].
    tolerances : list or None
        Tolerances for evaluation. If None, uses default for the domain.
    debug : bool
        Whether to save debug outputs.
    domain : str
        "score" (default, beat-based primary) or "performance" (ms-based, legacy).

    Returns
    -------
    dict
        Evaluation results. If domain="score", includes both beat and ms metrics.
    """
    if tolerances is None:
        tolerances = TOLERANCES_IN_BEATS if domain == "score" else TOLERANCES_IN_MS
    if not mm._has_run:
        raise ValueError("Must call run() before evaluation")

    wp = mm.score_follower.alignment_path
    score_beat = wp[1].astype(float)
    perf_sec = mm._wp_perf_to_seconds(wp[0].astype(float))

    perf_annots, score_annots_beats = resolve_gt(gt, mm.score_part.note_array())

    eval_results = evaluate_alignment(
        score_beat,
        perf_sec,
        score_annots_beats,
        perf_annots,
        beat_tolerances=tolerances if domain == "score" else TOLERANCES_IN_BEATS,
        ms_tolerances=TOLERANCES_IN_MS,
    )

    # Real-Time Factor (domain-independent)
    if mm.alignment_duration is not None:
        finite_perf = perf_annots[np.isfinite(perf_annots)]
        if len(finite_perf) > 0:
            perf_duration = float(np.max(finite_perf) - np.min(finite_perf))
            if perf_duration > 0:
                eval_results["rtf"] = float(
                    f"{mm.alignment_duration / perf_duration:.4f}"
                )

    if mm.input_type == "audio":
        eval_results.update(mm.get_latency_stats())

    if debug and save_dir is not None:
        wp_sec = np.array([perf_sec, score_beat])
        sf = mm.score_follower
        save_debug_results(
            alignment_path=wp_sec,
            score_annots=score_annots_beats,
            perf_annots=perf_annots,
            perf_annots_predicted=transfer_positions(
                wp_sec,
                score_annots_beats,
                frame_rate=1,
                domain="performance",
            ),
            eval_results=eval_results,
            frame_rate=mm.frame_rate,
            save_dir=save_dir,
            run_name=run_name or "results",
            score_positions=sf.score_positions,
            ref_features=sf.reference_features if plot_dist_matrix else None,
            input_features=(
                getattr(sf, "input_features", None) if plot_dist_matrix else None
            ),
            distance_func=getattr(sf, "distance_func", None),
            ref_frame_to_beat=getattr(sf, "_ref_frame_to_beat", None),
            make_plot=make_plot,
        )

    return eval_results


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def parse_annotation_csv(annotation_file: Union[str, Path]) -> np.ndarray:
    """Parse a note annotation file (CSV with TIME header or tab-delimited)."""
    annotation_file = Path(annotation_file)
    with open(annotation_file, "r") as f:
        first_line = f.readline().strip()
    if "TIME" in first_line.upper():
        df = pd.read_csv(annotation_file)
        if "TIME" in df.columns:
            return df["TIME"].values
        time_cols = [c for c in df.columns if "time" in c.lower()]
        if time_cols:
            return df[time_cols[0]].values
        return df.iloc[:, 0].values
    else:
        try:
            data = np.loadtxt(annotation_file)
        except ValueError:
            data = np.loadtxt(annotation_file, usecols=(0,))
        return data if data.ndim == 1 else data[:, 0]


def parse_match_file_for_note_onsets(
    match_file: Union[str, Path],
    level: str = "note",
    score_onset_beats: Optional[np.ndarray] = None,
    onset_aggregation_fun: Callable = np.min,
) -> np.ndarray:
    """Parse a match file and extract performed note onset times in seconds."""
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
        return stime_to_ptime_map(score_onset_beats)
    elif level == "beat":
        start_beat = np.ceil(snote_array["onset_beat"].min())
        end_beat = np.floor(snote_array["onset_beat"].max())
        return stime_to_ptime_map(np.arange(start_beat, end_beat + 1))
    elif level == "note":
        return stime_to_ptime_map(np.unique(snote_array["onset_beat"]))
    else:
        raise ValueError(f"Invalid level: {level}")


# ---------------------------------------------------------------------------
# Ground truth from match file
# ---------------------------------------------------------------------------


def build_gt(score_part, perf_ppart, alignment):
    """Build GT array: (perf_sec, score_beat) for each unique score onset."""
    _, stime_to_ptime = get_time_maps_from_alignment(
        perf_ppart, score_part, alignment, onset_aggregation_fun=np.min
    )
    gt_beats = np.unique(score_part.note_array()["onset_beat"])
    gt_times = stime_to_ptime(gt_beats)
    valid = np.isfinite(gt_times)
    return np.column_stack([gt_times[valid], gt_beats[valid]])


# ---------------------------------------------------------------------------
# Audio: score following via Matchmaker
# ---------------------------------------------------------------------------


def run_score_following(
    score_path: Path,
    perf_path: Union[Path, str],
    config: AudioEvalConfig,
    use_musical_beat: bool = False,
    *,
    input_type: str = "audio",
    dry_run: bool = False,
    save_dir: Optional[Path] = None,
    run_name: str = "",
    match_file: Optional[Path] = None,
    perf_annotations: Optional[np.ndarray] = None,
    matchmaker_kwargs: Optional[dict] = None,
    granularity: str = "note",
    save_plots: bool = True,
    gt: Optional[Union[str, Path, np.ndarray]] = None,
) -> dict:
    """Run score following via Matchmaker (audio or MIDI methods)."""
    from matchmaker import DEFAULT_KWARGS as _MM_DEFAULTS

    mm_kwargs = dict(_MM_DEFAULTS.get(input_type, {}).get(config.method, {}))
    if input_type == "audio":
        mm_kwargs["sample_rate"] = config.sample_rate
        mm_kwargs["frame_rate"] = config.frame_rate
    if matchmaker_kwargs is not None:
        mm_kwargs.update(matchmaker_kwargs)

    mm = Matchmaker(
        score_file=score_path,
        performance_file=perf_path,
        input_type=input_type,
        method=config.method,
        wait=False,
        unfold_score=True,
        kwargs=mm_kwargs,
    )

    try:
        alignment_positions = list(mm.run())
    except Exception as e:
        print(f"Error during mm.run(): {type(e)}, {e}")
        traceback.print_exc()
        mm._has_run = True

    wp = mm.score_follower.alignment_path
    if wp is None or (hasattr(wp, "size") and wp.size == 0) or len(wp) == 0:
        raise RuntimeError(
            "Empty alignment path — score follower produced no alignment"
        )

    gt_pairs = None
    if gt is not None:
        ps, sb = resolve_gt(gt, mm.score_part.note_array())
        gt_pairs = np.column_stack([ps, sb])

    nested = run_evaluation(
        mm,
        gt=gt_pairs,
        tolerances=TOLERANCES_IN_BEATS,
        musical_beat=use_musical_beat,
        domain="score",
        debug=save_dir is not None and not dry_run,
        make_plot=save_plots,
        save_dir=save_dir,
        run_name=run_name,
        level=granularity,
    )
    # Flatten nested {"beat": {...}, "ms": {...}} into a single dict for the
    # per-piece results table, prefixing metric keys with beat_ / ms_.
    results = {}
    for k, v in nested.get("beat", {}).items():
        results[f"beat_{k}"] = v
    for k, v in nested.get("ms", {}).items():
        results[f"ms_{k}"] = v
    for k, v in nested.items():
        if k not in ("beat", "ms"):
            results[k] = v

    # Tracking verification
    try:
        wp_for_check = wp.T
        if gt_pairs is not None:
            gt_for_check = gt_pairs
        else:
            score_annots_for_gt = mm.build_score_annotations(
                level=granularity, musical_beat=use_musical_beat, return_type="beats"
            )
            min_len = min(len(score_annots_for_gt), len(perf_annotations))
            gt_for_check = np.column_stack(
                [perf_annotations[:min_len], score_annots_for_gt[:min_len]]
            )
        tracking = check_tracking(
            wp_for_check, gt_for_check, config.frame_rate, mode="beat"
        )
        results["tracked"] = tracking["tracked"]
        results["max_deviation"] = tracking["max_deviation"]
        results["n_failed_segments"] = tracking["n_failed"]
        nested["tracked"] = tracking["tracked"]
        nested["max_deviation"] = tracking["max_deviation"]
        nested["n_failed_segments"] = tracking["n_failed"]

        if save_plots and save_dir is not None and run_name:
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

    if save_dir is not None and run_name and not dry_run:
        with open(Path(save_dir) / f"{run_name}.json", "w") as f:
            json.dump(nested, f, indent=4)

    return results


# ---------------------------------------------------------------------------
# Audio: offline alignment (librosa DTW)
# ---------------------------------------------------------------------------


def transfer_offline_positions(wp, perf_annots, frame_rate):
    perf_annots_frame = np.round(perf_annots * frame_rate)
    x, y = wp[0], wp[1]
    return (
        scipy.interpolate.interp1d(y, x, kind="linear")(perf_annots_frame) / frame_rate
    )


def run_offline_alignment(
    score_path,
    perf_path,
    match_file,
    config,
    use_musical_beat,
    perf_annotations=None,
    granularity="note",
):
    """Offline DTW alignment using librosa (audio only)."""
    from matchmaker.utils.misc import generate_score_audio

    mm = Matchmaker(
        score_file=score_path,
        performance_file=perf_path,
        input_type="audio",
        unfold_score=True,
    )
    audio_1 = generate_score_audio(mm.score_part, mm.tempo, config.sample_rate).astype(
        np.float32
    )
    audio_2, _ = librosa.load(perf_path.as_posix(), sr=config.sample_rate)

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
    cqt_1_norm = cqt_1_db / (np.linalg.norm(cqt_1_db, axis=0, keepdims=True) + 1e-10)
    cqt_2_norm = cqt_2_db / (np.linalg.norm(cqt_2_db, axis=0, keepdims=True) + 1e-10)

    _, wp_raw = librosa.sequence.dtw(
        X=cqt_1_norm, Y=cqt_2_norm, metric="cosine", backtrack=True
    )
    wp = wp_raw[::-1].T

    score_annots = mm.build_score_annotations(
        level=granularity, musical_beat=use_musical_beat, return_type="seconds"
    )
    score_beats = mm.build_score_annotations(
        level=granularity, musical_beat=use_musical_beat, return_type="beats"
    )
    na = mm.score_part.note_array()
    start_beat = max(0, int(np.ceil(np.unique(na["onset_beat"]).min())))
    n_perf = len(perf_annotations) if perf_annotations is not None else 0
    n_score = len(score_annots)
    intro_offset = max(0, min(start_beat, n_perf - n_score))

    if perf_annotations is not None:
        perf_annots = perf_annotations[intro_offset:]
    elif match_file is not None:
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
    score_beats = score_beats[:min_length]
    perf_annots = perf_annots[:min_length]

    max_perf_time = wp[1].max() / config.frame_rate
    valid = (
        np.isfinite(perf_annots)
        & np.isfinite(score_annots)
        & (perf_annots <= max_perf_time)
        & (perf_annots >= 0)
    )
    score_annots = score_annots[valid]
    score_beats = score_beats[valid]
    perf_annots = perf_annots[valid]

    predicted = transfer_offline_positions(wp, perf_annots, config.frame_rate)
    predicted_beats = np.interp(predicted, score_annots, score_beats)
    beat_res = get_evaluation_results(
        score_beats, predicted_beats, total_counts=len(score_beats),
        tolerances=TOLERANCES_IN_BEATS, in_seconds=False,
    )
    ms_res = get_evaluation_results(
        score_annots, predicted, total_counts=len(score_annots),
        tolerances=TOLERANCES_IN_MS,
    )
    results = {f"beat_{k}": v for k, v in beat_res.items()}
    results.update({f"ms_{k}": v for k, v in ms_res.items()})
    return results
