import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import partitura
import partitura as pt
import scipy
import yaml
from matchmaker.prob.hmm import (
    BernoulliGaussianPitchIOIObservationModel,
    PitchIOIHMM,
    compute_discrete_pitch_profiles,
    compute_ioi_matrix,
    gumbel_init_dist,
    gumbel_transition_matrix,
)
from matchmaker.utils.eval import get_evaluation_results, transfer_positions
from matchmaker.utils.tempo_models import KalmanTempoModel
from numpy.typing import NDArray
from pydantic_settings import BaseSettings

WORKING_DIR = Path(__file__).parent.parent
TOLERANCES_IN_MS = [50, 100, 300, 500, 1000, 2000]
TOLERANCES_IN_BEATS = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
METRICS_MS = ["mean", "median", "std", "skewness", "kurtosis"] + [
    f"{t}ms" for t in TOLERANCES_IN_MS
]
METRICS_BEAT = ["mean_b", "median_b", "std_b", "skewness_b", "kurtosis_b"] + [
    f"{t}b" for t in TOLERANCES_IN_BEATS
]
METRICS_TRACKING = ["tracked", "max_deviation", "n_failed_segments"]
METRICS_ALL = METRICS_MS + METRICS_BEAT + METRICS_TRACKING


class AudioEvalConfig(BaseSettings):
    method: str
    sample_rate: int = 44100
    frame_rate: float = 30
    dataset: Optional[str] = None

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate // self.frame_rate)

    # attributes for experiment (for logging purpose)
    attr_exp: list[str] = [
        "method",
        "sample_rate",
        "frame_rate",
        "dataset",
    ]


class SymbolicEvalConfig(BaseSettings):
    method: str
    dataset: Optional[str] = None
    processor: Optional[str] = None

    attr_exp: list[str] = ["method", "processor", "dataset"]


def save_config(config, save_dir):
    config_path = save_dir / "config.yaml"
    with open(config_path, "w") as f:
        config_dict = {
            k: v
            for k, v in config.__dict__.items()
            if not k.startswith("__") and not k.startswith("attr_")
        }
        yaml.dump(config_dict, f)


def compute_sparc(
    score_beat: np.ndarray,
    perf_sec: np.ndarray,
    fs: float = 50.0,
    fc: float = 10.0,
    padlevel: int = 4,
) -> float:
    """Compute Spectral Arc Length (SPARC) for an alignment path."""
    score_beat = np.asarray(score_beat, dtype=float)
    perf_sec = np.asarray(perf_sec, dtype=float)

    valid = np.isfinite(score_beat) & np.isfinite(perf_sec)
    score_beat = score_beat[valid]
    perf_sec = perf_sec[valid]

    if len(perf_sec) < 2:
        return 0.0

    t_start, t_end = float(perf_sec[0]), float(perf_sec[-1])
    duration = t_end - t_start
    if duration <= 0.1:
        return 0.0

    n_pts = max(int(np.round(duration * fs)), 2)
    t_uniform = np.linspace(t_start, t_end, n_pts)
    s_interp = np.interp(t_uniform, perf_sec, score_beat)
    vel = np.gradient(s_interp, 1.0 / fs)

    nfft = int(2 ** (np.ceil(np.log2(len(vel))) + padlevel))
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mask = freq <= fc
    freq_filtered = freq[mask]

    Mf = np.abs(np.fft.rfft(vel, n=nfft))[mask]
    max_mf = Mf.max()
    if max_mf > 0:
        Mf = Mf / max_mf

    d_freq = np.diff(freq_filtered) / fc
    d_mf = np.diff(Mf)
    arc = np.sum(np.sqrt(d_freq ** 2 + d_mf ** 2))
    return float(-arc)


def compute_event_pooled_summary(
    results: dict,
    run_dir: Path,
    tracked_only: bool = True,
    common_indices: Optional[set] = None,
) -> dict:
    """
    Compute event-wise pooled metrics.

    Accuracy metrics (mean, median, tolerances): pooled across all events.
    RTF/latency/SPARC metrics: piece-wise averaged/median.

    Parameters
    ----------
    tracked_only : bool
        If True, only include tracked pieces. If False, include all pieces.
    common_indices : set or None
        If provided, only include pieces whose index is in common_indices.
    """
    n_total = len(results["Index"])
    tracked_flags = results.get("tracked", [False] * n_total)

    # Collect events from selected pieces
    all_gt_perf = []
    all_pred_perf = []
    all_gt_score_beats = []
    all_pred_score_beats = []
    selected_indices = []
    sparc_values = []

    for idx, is_tracked in zip(results["Index"], tracked_flags):
        if common_indices is not None and idx not in common_indices:
            continue
        if tracked_only and not is_tracked:
            continue
        selected_indices.append(idx)

        wp_file = run_dir / f"wp_{idx}.tsv"
        gt_file = run_dir / f"gt_{idx}.tsv"
        if not wp_file.exists() or not gt_file.exists():
            continue

        wp = np.loadtxt(wp_file, delimiter="\t", skiprows=1)
        gt = np.loadtxt(gt_file, delimiter="\t", skiprows=1)

        gt_perf = gt[:, 0]  # perf times (seconds) — col 0 in new format
        gt_score = gt[:, 1]  # score positions (beats) — col 1 in new format

        # Score → perf prediction (ms metrics)
        # wp TSV col order: perf_sec, score_beat — already matches transfer_positions input
        wp_t = wp.T
        pred_perf = transfer_positions(wp_t, gt_score, 1, domain="performance")
        valid = np.isfinite(pred_perf) & np.isfinite(gt_perf)
        all_gt_perf.append(gt_perf[valid])
        all_pred_perf.append(pred_perf[valid])

        # Perf → score prediction (beat metrics)
        valid_gt_perf = np.isfinite(gt_perf)
        pred_score = transfer_positions(wp_t, gt_perf[valid_gt_perf], 1, domain="score")
        valid_b = np.isfinite(pred_score)
        all_gt_score_beats.append(gt_score[valid_gt_perf][valid_b])
        all_pred_score_beats.append(pred_score[valid_b])

        # Compute SPARC for the alignment path
        sparc_val = compute_sparc(wp_t[1], wp_t[0])
        sparc_values.append(sparc_val)

    summary = {}
    n_tracked = sum(1 for t in tracked_flags if t)
    summary["tracking_rate"] = round(n_tracked / n_total, 4) if n_total > 0 else 0.0

    # Beat metrics (primary)
    if all_gt_score_beats:
        pooled_gt_b = np.concatenate(all_gt_score_beats)
        pooled_pred_b = np.concatenate(all_pred_score_beats)
        total_count_b = len(pooled_gt_b)
        beat_results = get_evaluation_results(
            pooled_gt_b,
            pooled_pred_b,
            total_count_b,
            tolerances=TOLERANCES_IN_BEATS,
            in_seconds=False,
        )
        summary["beat"] = beat_results

    # Ms metrics (secondary)
    if all_gt_perf:
        pooled_gt = np.concatenate(all_gt_perf)
        pooled_pred = np.concatenate(all_pred_perf)
        total_count = len(pooled_gt)
        ms_results = get_evaluation_results(
            pooled_gt,
            pooled_pred,
            total_count,
            tolerances=TOLERANCES_IN_MS,
            in_seconds=True,
        )
        summary["ms"] = ms_results

    # SPARC: piece-wise mean across selected pieces
    if sparc_values:
        summary["sparc"] = float(f"{np.mean(sparc_values):.2f}")

    # RTF and latency: piece-wise average of selected pieces
    for key in ["rtf", "f_avg_latency", "i_avg_latency"]:
        if key in results:
            if common_indices is not None:
                vals = [v for v, idx, t in zip(results[key], results["Index"], tracked_flags) if idx in common_indices and (not tracked_only or t)]
            elif tracked_only:
                vals = [v for v, t in zip(results[key], tracked_flags) if t]
            else:
                vals = list(results[key])
            if vals:
                summary[key] = float(f"{np.nanmean(vals):.4f}")

    summary["piece_count"] = n_total
    summary["tracked_count"] = n_tracked
    summary["selected_count"] = len(selected_indices)
    return summary


#: Per-piece timing columns the pooled summary averages piece-wise.
TIMING_KEYS = ("rtf", "f_avg_latency", "i_avg_latency")


def pooled_summaries(pieces: list, run_dir: Path) -> dict:
    """Pool a run's metrics over every piece and over the tracked ones.

    ``pieces`` are the per-piece records of a run — each with ``index``,
    ``tracked`` and, when the piece ran, its timing columns. Returns the
    ``summary_all`` / ``summary_tracked`` pair that ``metrics.json`` carries.

    A piece that crashed has no timing columns; it is padded with NaN so the
    columns stay aligned with ``Index`` and the piece-wise averages skip it.
    """
    results = defaultdict(list)
    timing = [key for key in TIMING_KEYS if any(key in piece for piece in pieces)]
    for piece in pieces:
        results["Index"].append(piece["index"])
        results["tracked"].append(bool(piece.get("tracked")))
        for key in timing:
            results[key].append(piece.get(key, float("nan")))
    return {
        "summary_all": compute_event_pooled_summary(
            results, run_dir, tracked_only=False
        ),
        "summary_tracked": compute_event_pooled_summary(
            results, run_dir, tracked_only=True
        ),
    }


def dataset_summaries(pieces: list, run_dir: Path) -> dict:
    """The same pooled summaries, once per dataset.

    Each dataset's block has the shape of the run's own record — piece counts
    plus ``summary_all`` and ``summary_tracked`` — and is computed by the same
    function over that dataset's pieces alone, so a per-dataset number means
    exactly what the number for the whole run means.
    """
    by_dataset = defaultdict(list)
    for piece in pieces:
        by_dataset[piece.get("dataset") or "unknown"].append(piece)

    summaries = {}
    for dataset, rows in sorted(by_dataset.items()):
        summaries[dataset] = {
            "n_pieces": len(rows),
            "n_tracked": sum(1 for r in rows if r.get("tracked")),
            "n_failed": sum(1 for r in rows if r.get("error")),
            **pooled_summaries(rows, run_dir),
        }
    return summaries


def save_results_to_csv(results: dict, save_path: str):
    from itertools import zip_longest

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(results.keys())
        writer.writerows(zip_longest(*results.values(), fillvalue=""))


def save_nparray_to_csv(array: NDArray, save_path: str, header: Optional[str] = None):
    with open(save_path, "w") as csvfile:
        if header is not None:
            csvfile.write(header + "\n")
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerows(array)


def _beats_to_frames(
    beats: np.ndarray,
    ref_frame_to_beat: np.ndarray,
) -> np.ndarray:
    """Convert beat positions to (float) frame indices via inverse interpolation."""
    frames = np.arange(len(ref_frame_to_beat), dtype=float)
    return np.interp(beats, ref_frame_to_beat, frames)


def plot_alignment(
    alignment_path: np.ndarray,
    perf_annots: np.ndarray,
    perf_annots_predicted: np.ndarray,
    save_dir: Path,
    name: str,
    score_y: Optional[np.ndarray] = None,
    frame_rate: float = 1.0,
    score_positions: Optional[np.ndarray] = None,
    ref_features: Optional[np.ndarray] = None,
    input_features: Optional[np.ndarray] = None,
    distance_func=None,
    ref_frame_to_beat: Optional[np.ndarray] = None,
):
    """Plot alignment path, GT annotations, and predicted points."""
    label_fontsize = 7
    tick_fontsize = 5
    legend_fontsize = 6
    title_fontsize = 7

    save_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(perf_annots, dtype=float)
    pred = np.asarray(perf_annots_predicted, dtype=float)
    n = min(len(gt), len(pred))
    gt, pred = gt[:n], pred[:n]

    # Figure size + dpi scale with performance duration (song length):
    perf_times = np.asarray(alignment_path[0], dtype=float)
    perf_times = perf_times[np.isfinite(perf_times)]
    perf_dur = float(perf_times.max()) if perf_times.size else 0.0
    t = float(np.clip((perf_dur - 30.0) / (540.0 - 30.0), 0.0, 1.0))
    fig_side = 4.0 + t * (16.0 - 4.0)
    fig_dpi = int(100 + t * (200 - 100))

    fig, ax = plt.subplots(figsize=(fig_side, fig_side))

    # Distance matrix background
    show_dist = False
    if (
        ref_features is not None
        and input_features is not None
        and distance_func is not None
    ):
        try:
            ref_features = np.asarray(ref_features, dtype=np.float32)
            input_features = np.asarray(input_features, dtype=np.float32)
            if ref_features.ndim > 2:
                ref_features = ref_features.reshape(ref_features.shape[0], -1)
            if input_features.ndim > 2:
                input_features = input_features.reshape(input_features.shape[0], -1)
            if isinstance(distance_func, str):
                dist = scipy.spatial.distance.cdist(
                    ref_features, input_features, metric=distance_func
                )
            else:
                dist = np.array(
                    [
                        [distance_func(r, i) for i in input_features]
                        for r in ref_features
                    ],
                    dtype=np.float32,
                )
            n_input = input_features.shape[0]
            n_ref = ref_features.shape[0]
            ax.imshow(
                dist,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(0, n_input - 1, 0, n_ref - 1),
            )
            show_dist = True
        except Exception:
            pass

    # x-axis: performance time in frames
    x_gt = gt * float(frame_rate)
    wp_x = alignment_path[0] * float(frame_rate)

    # y-axis: score position (beats)
    wp_in_beats = np.issubdtype(alignment_path[1].dtype, np.floating)
    if score_positions is not None and not wp_in_beats:
        wp_y = score_positions[alignment_path[1]]
    elif show_dist and wp_in_beats and ref_frame_to_beat is not None:
        wp_y = _beats_to_frames(alignment_path[1], ref_frame_to_beat)
    else:
        wp_y = alignment_path[1]

    # GT score positions (y-axis for annotation dots)
    if score_y is not None:
        y_gt = np.asarray(score_y, dtype=float)[:n]
        if show_dist and wp_in_beats and ref_frame_to_beat is not None:
            y_gt = _beats_to_frames(y_gt, ref_frame_to_beat)
    else:
        y_gt = np.arange(n)

    # Predicted score positions at GT perf times (perf→score direction)
    wp_x_sorted = np.asarray(wp_x, dtype=float)
    wp_y_sorted = np.asarray(wp_y, dtype=float)
    if len(wp_x_sorted) > 1:
        y_pred = np.interp(x_gt, wp_x_sorted, wp_y_sorted)
    else:
        y_pred = y_gt

    # Plot layers. GT and the dense alignment path share one length-scaled
    # size (~10 short -> ~4 long) so they look consistent.
    t_gt = float(np.clip((perf_dur - 20.0) / (75.0 - 20.0), 0.0, 1.0))
    gt_size = 10.0 - t_gt * (10.0 - 4.0)
    gt_lw = 1.0 - t_gt * (1.0 - 0.8)
    ax.scatter(
        wp_x,
        wp_y,
        label="alignment path",
        s=gt_size,
        color="white" if show_dist else "limegreen",
        alpha=0.7 if show_dist else 0.85,
        linewidths=0,
        zorder=3,
    )
    ax.scatter(
        x_gt,
        y_pred,
        label="predicted",
        s=6,
        marker="o",
        color="royalblue",
        zorder=4,
    )
    ax.scatter(
        x_gt,
        y_gt,
        label="ground truth",
        s=gt_size,
        alpha=0.9,
        marker="x",
        color="red",
        linewidths=gt_lw,
        zorder=5,
    )

    if show_dist:
        ax.set_xlim(0, input_features.shape[0] - 1)
        ax.set_ylim(0, ref_features.shape[0] - 1)
        ax.set_box_aspect(1)

    # Beat tick labels when projected to frame space
    if show_dist and wp_in_beats and ref_frame_to_beat is not None:
        finite_beats = ref_frame_to_beat[np.isfinite(ref_frame_to_beat)]
        beat_min, beat_max = (
            finite_beats[0],
            finite_beats[-1] if len(finite_beats) > 0 else (0, 1),
        )
        n_ticks = max(2, min(12, int(beat_max - beat_min) + 1))
        beat_ticks = np.unique(
            np.round(np.linspace(beat_min, beat_max, n_ticks)).astype(int)
        )
        ax.set_yticks(_beats_to_frames(beat_ticks.astype(float), ref_frame_to_beat))
        ax.set_yticklabels([str(b) for b in beat_ticks], fontsize=tick_fontsize)
    ax.set_xlabel("performance frame", fontsize=label_fontsize)
    ax.set_ylabel("score position (beats)", fontsize=label_fontsize)
    ax.set_title(f"[{save_dir.name}] alignment ({name})", fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=0.5, length=2)
    ax.grid(True, alpha=0.25, linewidth=0.3)
    ax.legend(loc="best", fontsize=legend_fontsize, markerscale=1.4, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(
        save_dir / f"{name}.png", dpi=fig_dpi, bbox_inches="tight", pad_inches=0.05
    )
    plt.close(fig)


def save_debug_results(
    alignment_path: np.ndarray,
    score_annots: np.ndarray,
    perf_annots: np.ndarray,
    perf_annots_predicted: np.ndarray,
    eval_results: dict,
    frame_rate: float,
    save_dir: Path,
    run_name: str = "results",
    score_positions: Optional[np.ndarray] = None,
    ref_features: Optional[np.ndarray] = None,
    input_features: Optional[np.ndarray] = None,
    distance_func=None,
    ref_frame_to_beat: Optional[np.ndarray] = None,
    make_plot: bool = True,
):
    """Save debug outputs: alignment path TSV, results JSON, and (optional) plot."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Alignment path TSV + results JSON + GT annotations
    # Column order: perf_sec, score_beat (alignment_path is already [perf, score])
    save_nparray_to_csv(
        alignment_path.T,
        (save_dir / f"wp_{run_name}.tsv").as_posix(),
        header="perf_sec\tscore_beat",
    )
    gt_pairs = np.column_stack([perf_annots, score_annots])
    save_nparray_to_csv(
        gt_pairs,
        (save_dir / f"gt_{run_name}.tsv").as_posix(),
        header="perf_sec\tscore_beat",
    )
    with open(save_dir / f"{run_name}.json", "w") as f:
        json.dump(eval_results, f, indent=4)

    if not make_plot:
        return

    # 2. Alignment plot
    # score_y = beat positions for each annotation (y-axis of the plot).
    # Not gated on monotonicity: note-level GT dips locally at chords.
    sx = np.asarray(score_annots, dtype=float)
    score_y = sx if sx.ndim == 1 and len(sx) == len(perf_annots) else None
    plot_alignment(
        alignment_path,
        perf_annots,
        perf_annots_predicted,
        save_dir,
        run_name,
        score_y=score_y,
        frame_rate=frame_rate,
        score_positions=score_positions,
        ref_features=ref_features,
        input_features=input_features,
        distance_func=distance_func,
        ref_frame_to_beat=ref_frame_to_beat,
    )


def save_score_following_result(
    model, save_dir, score_annots, perf_ann_path: Path, frame_rate, name=None
):
    run_name = name or "results"
    save_path = save_dir / f"wp_{run_name}.tsv"
    save_nparray_to_csv(model.alignment_path.T, save_path.as_posix())

    dist = scipy.spatial.distance.cdist(
        model.reference_features,
        model.input_features[: model.alignment_path[1][-1]],
        metric=model.distance_func,
    )  # [d, wy]
    plt.figure(figsize=(15, 15))
    plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(
        f"[{save_dir.name}] \n Matchmaker alignment path with ground-truth labels",
        fontsize=25,
    )
    plt.xlabel("Performance Audio frame", fontsize=15)
    plt.ylabel("Score Audio frame", fontsize=15)

    # plot online DTW path
    ref_paths, target_paths = model.alignment_path[0], model.alignment_path[1]
    for n in range(len(ref_paths)):
        plt.plot(
            target_paths[n], ref_paths[n], ".", color="lime", alpha=0.5, markersize=3
        )

    # plot ground-truth labels
    perf_annots = pd.read_csv(
        filepath_or_buffer=perf_ann_path, delimiter="\t", header=None
    )[0]
    for i, (ref, target) in enumerate(zip(score_annots, perf_annots)):
        plt.plot(
            target * frame_rate, ref * frame_rate, "x", color="r", alpha=1, markersize=3
        )
    plt.savefig(save_dir / f"{run_name}.png")


# def plot_and_save_score_following_result(
#     wp,
#     ref_features,
#     input_features,
#     distance_func,
#     save_dir,
#     score_annots,
#     perf_ann_path: Path,
#     frame_rate,
#     name=None,
# ):
#     run_name = name or "results"
#     save_path = save_dir / f"wp_{run_name}.tsv"
#     save_nparray_to_csv(wp.T, save_path.as_posix())

#     dist = scipy.spatial.distance.cdist(
#         ref_features,
#         input_features[: wp[1][-1]],
#         metric=distance_func,
#     )  # [d, wy]
#     plt.figure(figsize=(15, 15))
#     plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
#     plt.title(
#         f"[{save_dir.name}] \n Matchmaker alignment path with ground-truth labels",
#         fontsize=25,
#     )
#     plt.xlabel("Performance Audio frame", fontsize=15)
#     plt.ylabel("Score Audio frame", fontsize=15)

#     # plot online DTW path
#     ref_paths, target_paths = wp[0], wp[1]
#     for n in range(len(ref_paths)):
#         plt.plot(
#             target_paths[n], ref_paths[n], ".", color="purple", alpha=0.5, markersize=3
#         )

#     # plot ground-truth labels
#     perf_annots = pd.read_csv(
#         filepath_or_buffer=perf_ann_path, delimiter="\t", header=None
#     )[0]
#     for i, (ref, target) in enumerate(zip(score_annots, perf_annots)):
#         plt.plot(
#             target * frame_rate, ref * frame_rate, "x", color="r", alpha=1, markersize=3
#         )
#     plt.savefig(save_dir / f"{run_name}.png")


def build_matchmaker_hmm(score_path):
    bpm = 100

    snote_array = pt.load_score_midi(score_path).note_array()
    unique_sonsets = np.unique(snote_array["onset_beat"])

    unique_sonset_idxs = [
        np.where(snote_array["onset_beat"] == ui)[0] for ui in unique_sonsets
    ]

    chord_pitches = [snote_array["pitch"][uix] for uix in unique_sonset_idxs]

    pitch_profiles = compute_discrete_pitch_profiles(
        chord_pitches=chord_pitches,
        piano_range=True,
        inserted_states=True,
    )

    ioi_matrix = compute_ioi_matrix(
        unique_onsets=unique_sonsets,
        inserted_states=True,
    )

    score_positions = ioi_matrix[0]
    n_states = len(score_positions)

    observation_model = BernoulliGaussianPitchIOIObservationModel(
        pitch_profiles=pitch_profiles,
        ioi_matrix=ioi_matrix,
        ioi_precision=1,
    )

    transition_matrix = (
        gumbel_transition_matrix(  # TODO another possibilities? (Gaussian/Nakamura)
            n_states=n_states,
            inserted_states=True,
        )
    )

    initial_probabilities = gumbel_init_dist(
        n_states=n_states,
    )

    tempo_model = KalmanTempoModel(  # TODO another possibilities? (LinearSMS, JADAM)
        init_score_onset=unique_sonsets.min(),
        init_beat_period=60 / bpm,
    )

    matchmaker = PitchIOIHMM(
        observation_model=observation_model,
        transition_matrix=transition_matrix,
        score_onsets=score_positions,
        initial_probabilities=initial_probabilities,
        has_insertions=True,
        tempo_model=tempo_model,
    )
    return matchmaker


def source_annotation(row, dataset):
    audio = Path(row.audio_performance)
    if dataset == "kraisler":
        return f"annotations/{audio.parent.name}_beats.csv"
    if dataset == "chorale":
        return f"annotations/{audio.stem}_notes.csv"
    if dataset == "winterreise":
        return f"aligned/{row.title}_{audio.stem.split('_')[-1]}_aligned.txt"
    return row.performance_annotations
