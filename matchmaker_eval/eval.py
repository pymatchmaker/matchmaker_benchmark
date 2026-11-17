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
from matchmaker import Matchmaker
from matchmaker.utils.eval import (
    evaluate_alignment,
    resolve_gt,
    transfer_positions,
)
from numpy.typing import NDArray
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment

from matchmaker_eval.utils import (
    TOLERANCES_IN_BEATS,
    AudioEvalConfig,
    save_debug_results,
)
from matchmaker_eval.verify_tracking import (
    TRACKING_THRESHOLDS, check_tracking, plot_tracking, musical_beat_factors, score_meter_map,
)

# ---------------------------------------------------------------------------
# Evaluation against ground truth (post-processing of a completed Matchmaker run)
# ---------------------------------------------------------------------------


def latency_stats(mm) -> dict:
    """Per-frame latency, when the follower reports it.

    ``latency_stats`` is not part of the ``OnlineAlignment`` contract — the
    built-in audio followers each maintain their own, and a submission written
    to the documented base class has none. It is a nice-to-have measurement, so
    a follower without it is evaluated on everything else rather than failing.
    """
    if mm.input_type != "audio":
        return {}
    try:
        return mm.get_latency_stats()
    except (AttributeError, ZeroDivisionError, KeyError, TypeError):
        return {}


def run_evaluation(
    mm: Matchmaker,
    gt: Union[str, Path, np.ndarray] = None,
    tolerances: Optional[list] = None,
    debug: bool = False,
    save_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
    domain: str = "score",
    plot_dist_matrix: bool = True,
    make_plot: bool = True,
    level: str = "note",
) -> dict:
    """Evaluate a completed Matchmaker run against ground truth.

    Return matchmaker's beat accuracy and scalar smoothness fields directly,
    adding only runtime and audio latency measurements.

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
        Retained for runner compatibility; results use the shared metric schema.

    Returns
    -------
    dict
        Beat accuracy, scalar smoothness, runtime, and available latency metrics.
    """
    if tolerances is None:
        tolerances = TOLERANCES_IN_BEATS
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
        beat_tolerances=tolerances,
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

    eval_results.update(latency_stats(mm))

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
    tempo: Optional[float] = 120.0,
) -> dict:
    """Run score following via Matchmaker (audio or MIDI methods)."""
    from methods import default_kwargs

    mm_kwargs = default_kwargs(input_type, config.method)
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
        tempo=tempo,
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

    results = run_evaluation(
        mm,
        gt=gt_pairs,
        tolerances=TOLERANCES_IN_BEATS,
        domain="score",
        debug=save_dir is not None and not dry_run,
        make_plot=save_plots,
        save_dir=save_dir,
        run_name=run_name,
        level=granularity,
    )
    # Tracking verification
    try:
        wp_for_check = wp.T
        if gt_pairs is not None:
            gt_for_check = gt_pairs
        else:
            score_annots_for_gt = mm.build_score_annotations(
                level=granularity, return_type="beats"
            )
            min_len = min(len(score_annots_for_gt), len(perf_annotations))
            gt_for_check = np.column_stack(
                [perf_annotations[:min_len], score_annots_for_gt[:min_len]]
            )
        meter = score_meter_map(mm.score_part)
        factors = musical_beat_factors(gt_for_check[:, 1], meter)
        threshold = TRACKING_THRESHOLDS[mm.input_type]
        tracking = check_tracking(
            wp_for_check, gt_for_check, threshold=threshold, beat_factors=factors
        )
        results["tracked"] = tracking["tracked"]
        results["max_deviation"] = tracking["max_deviation"]
        results["n_failed_segments"] = tracking["n_failed"]
        results["tracking_criterion"] = {
            key: tracking[key] for key in ("error_unit", "threshold", "window_duration")
        }
        if save_dir is not None and run_name:
            meter_path = Path(save_dir) / f"meter_{run_name}.json"
            meter_path.write_text(json.dumps(meter, indent=2) + "\n")

        if save_plots and save_dir is not None and run_name:
            plot_tracking(
                wp_for_check,
                gt_for_check,
                title=f"{config.method} #{run_name}",
                save_path=Path(save_dir) / f"tracking_{run_name}.png",
                threshold=threshold,
                beat_factors=factors,
            )
    except Exception:
        pass

    if save_dir is not None and run_name and not dry_run:
        with open(Path(save_dir) / f"{run_name}.json", "w") as f:
            json.dump(results, f, indent=4)

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
        level=granularity, return_type="seconds"
    )
    score_beats = mm.build_score_annotations(
        level=granularity, return_type="beats"
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
            level=granularity, return_type="beats"
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

    frame_to_beat = mm.ref_frame_to_beat()
    reference_frames = wp[0].astype(int)
    if np.any(reference_frames < 0) or np.any(reference_frames >= len(frame_to_beat)):
        raise ValueError(
            "Offline alignment reference frames exceed the frame-to-beat mapping "
            f"of length {len(frame_to_beat)}"
        )
    return evaluate_alignment(
        frame_to_beat[reference_frames], wp[1] / config.frame_rate,
        score_beats, perf_annots, beat_tolerances=TOLERANCES_IN_BEATS,
    )
