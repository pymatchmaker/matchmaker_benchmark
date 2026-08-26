"""End-to-end audio -> AMT -> PTHMM benchmark with RTF measurement.

For each piece in a dataset:
1. Load audio (.wav)
2. Run onlineAMT to transcribe to in-memory notes
3. Save to a temporary MIDI file
4. Run matchmaker score-following (default: pthmm) on the temp MIDI
5. Measure total wall time and report RTF = wall_time / audio_duration

Reuses the same metrics machinery as test_symbolic.py.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from eval import run_evaluation
from matchmaker import Matchmaker
from matchmaker.matchmaker import DEFAULT_KWARGS
from matchmaker.utils.eval import resolve_gt
from utils import (
    DATASET_DIR,
    OUTPUT_DIR,
    SYMBOLIC_METADATA_PATH as METADATA_PATH,
    TOLERANCES_IN_BEATS,
    WORKING_DIR,
    SymbolicEvalConfig,
    compute_event_pooled_summary,
    save_config,
    save_results_to_csv,
)
from verify_tracking import check_tracking, plot_tracking

ONLINE_AMT_ROOT = Path(
    os.environ.get("ONLINE_AMT_ROOT", "~/workspace/onlineAMT")
).expanduser()
sys.path.insert(0, str(ONLINE_AMT_ROOT))

sys.setrecursionlimit(10000)

TRACKING_THRESHOLD = 0.5  # beats
DEFAULT_AMT_MODEL_PATH = ONLINE_AMT_ROOT / "amt/model.pt"
DEFAULT_AMT_DEVICE = "cpu"


def load_audio_for_amt(path, sr):
    """Load and resample audio without torchaudio's TorchCodec backend."""
    import torch
    from scipy.signal import resample_poly

    samples, original_sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    mono = samples.mean(axis=1)
    if original_sr != sr:
        mono = resample_poly(mono, sr, original_sr).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)


def load_online_amt():
    """Import onlineAMT only when the pipeline is actually executed."""
    try:
        from amt import inference
        from amt.transcribe import load_model
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "onlineAMT dependencies are unavailable in this environment "
            f"(missing {error.name!r})"
        ) from error

    inference._load_audio = load_audio_for_amt
    return load_model, inference.transcribe, inference.save_midi


def get_audio_duration(audio_path):
    """Return duration (seconds) of an audio file."""
    info = sf.info(str(audio_path))
    return info.frames / float(info.samplerate)


def resolve_asset_path(path_value, dataset_root):
    """Resolve dataset-relative and benchmark-generated asset paths."""
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    if path.parts[:2] == ("data", "preprocessed"):
        return WORKING_DIR / path
    if path.parts[:1] == ("dataset_root",):
        return dataset_root.joinpath(*path.parts[1:])
    return dataset_root / path


def run_pipeline_one(
    model,
    audio_path,
    score_xml,
    match_path,
    method,
    run_dir,
    idx,
    device,
    transcribe_fn,
    save_midi_fn,
    save_plots=True,
):
    """Transcribe one performance, follow the score, and evaluate the result."""
    audio_duration = get_audio_duration(audio_path)
    if audio_duration <= 0:
        raise ValueError(f"Audio has no duration: {audio_path}")

    tmp_midi_path = None
    t_start = time.time()
    try:
        t_amt_start = time.time()
        notes = transcribe_fn(model, str(audio_path), device=device)
        t_amt = time.time() - t_amt_start

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_midi_path = tmp.name
        save_midi_fn(notes, tmp_midi_path)

        mm_kwargs = copy.deepcopy(DEFAULT_KWARGS["midi"].get(method, {}))
        t_mm_start = time.time()
        mm = Matchmaker(
            score_file=str(score_xml),
            performance_file=tmp_midi_path,
            input_type="midi",
            method=method,
            kwargs=mm_kwargs if mm_kwargs else None,
        )
        list(mm.run(verbose=False))
        t_mm = time.time() - t_mm_start
        pipeline_sec = time.time() - t_start

        wp = np.asarray(mm.score_follower.alignment_path, dtype=float)
        if wp.ndim != 2 or wp.shape[0] != 2 or wp.shape[1] == 0:
            raise RuntimeError("Score follower produced an empty alignment path")
        wp_T = np.column_stack(
            [
                mm._wp_perf_to_seconds(wp[0]),
                wp[1],
            ]
        )

        perf_sec, score_beat = resolve_gt(
            match_path,
            mm.score_part.note_array(),
        )
        gt = np.column_stack([perf_sec, score_beat])

        tracking = check_tracking(
            wp_T,
            gt,
            segment_duration=30,
            threshold=TRACKING_THRESHOLD,
        )
        nested = run_evaluation(
            mm,
            gt=gt,
            tolerances=TOLERANCES_IN_BEATS,
            domain="score",
            debug=run_dir is not None,
            save_dir=run_dir,
            run_name=str(idx),
            make_plot=save_plots,
        )
        nested["tracked"] = tracking["tracked"]
        nested["max_deviation"] = float(tracking["max_deviation"])
        nested["n_failed_segments"] = int(tracking["n_failed"])

        timings = {
            "amt_sec": float(t_amt),
            "mm_sec": float(t_mm),
            "pipeline_sec": float(pipeline_sec),
            "audio_sec": float(audio_duration),
            "rtf": float(pipeline_sec / audio_duration),
        }
        nested.update(timings)

        piece_result = {}
        for key, value in nested.get("beat", {}).items():
            piece_result[f"beat_{key}"] = value
        for key, value in nested.get("ms", {}).items():
            piece_result[f"ms_{key}"] = value
        for key, value in nested.items():
            if key not in ("beat", "ms"):
                piece_result[key] = value

        return piece_result, nested, wp_T, gt, tracking
    finally:
        if tmp_midi_path is not None:
            Path(tmp_midi_path).unlink(missing_ok=True)


def run_dataset(
    dataset_type,
    method,
    run_dir,
    model,
    device,
    transcribe_fn,
    save_midi_fn,
    save_plots=True,
):
    """Run the complete AMT-to-score-following pipeline for one dataset."""
    metadata = pd.read_csv(METADATA_PATH[dataset_type])
    is_validation = dataset_type in ("valid", "example")
    results = defaultdict(list)

    for i, row in enumerate(metadata.itertuples(), 1):
        dataset_name = row.dataset if is_validation else dataset_type
        dataset_root = DATASET_DIR[dataset_name]
        score_xml = resolve_asset_path(row.xml_score, dataset_root)
        match_path = resolve_asset_path(row.match, dataset_root)
        audio_path = resolve_asset_path(row.audio_performance, dataset_root)

        print(f"[{i}/{len(metadata)}] {row.title}", flush=True)
        missing = [
            path
            for path in (audio_path, score_xml, match_path)
            if not path.exists()
        ]
        if missing:
            print(f"  SKIP: missing asset(s): {missing}", flush=True)
            continue

        try:
            piece_result, nested, wp_T, gt, tracking = run_pipeline_one(
                model,
                audio_path,
                score_xml,
                match_path,
                method,
                run_dir,
                i,
                device,
                transcribe_fn,
                save_midi_fn,
                save_plots=save_plots,
            )

            np.savetxt(
                run_dir / f"wp_{i}.tsv",
                wp_T,
                delimiter="\t",
                fmt="%.6f",
                header="perf_sec\tscore_beat",
                comments="",
            )
            np.savetxt(
                run_dir / f"gt_{i}.tsv",
                gt,
                delimiter="\t",
                fmt="%.6f",
                header="perf_sec\tscore_beat",
                comments="",
            )
            with open(run_dir / f"{i}.json", "w") as file:
                json.dump(nested, file, indent=4, default=float)
            if save_plots:
                plot_tracking(
                    wp_T,
                    gt,
                    title=f"AMT + {method} #{i}",
                    save_path=run_dir / f"tracking_{i}.png",
                    threshold=TRACKING_THRESHOLD,
                )

            status = "TRACKED" if tracking["tracked"] else "FAILED"
            print(
                f"  {status}  rtf={piece_result['rtf']:.3f}  "
                f"amt={piece_result['amt_sec']:.1f}s  "
                f"mm={piece_result['mm_sec']:.1f}s  "
                f"audio={piece_result['audio_sec']:.1f}s  "
                f"max_dev={tracking['max_deviation']:.3f}b",
                flush=True,
            )

            results["Index"].append(i)
            results["Piece"].append(row.title)
            for key, value in piece_result.items():
                results[key].append(value)
        except Exception as error:
            print(f"  ERROR: {error}", flush=True)
            traceback.print_exc()

    if not results["Index"]:
        raise RuntimeError(f"No pieces completed successfully for {dataset_type}")
    return results


def add_timing_summary(summary, results, tracked_only):
    """Add pipeline timing statistics using the same tracked-piece filter."""
    tracked = results.get("tracked", [])
    for key in ("rtf", "amt_sec", "mm_sec", "pipeline_sec", "audio_sec"):
        values = results.get(key, [])
        if tracked_only:
            values = [value for value, is_tracked in zip(values, tracked) if is_tracked]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_median"] = float(np.median(values))


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end audio -> AMT -> MIDI score-following benchmark"
    )
    parser.add_argument(
        "--datasets",
        default="valid,asap,batik,vienna",
        help="Comma-separated datasets to run",
    )
    parser.add_argument(
        "--method",
        default="pthmm",
        choices=sorted(DEFAULT_KWARGS["midi"]),
        help="Symbolic score-following method",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_AMT_MODEL_PATH,
        help="onlineAMT checkpoint",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_AMT_DEVICE,
        help="Torch device used by onlineAMT",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip per-piece evaluation and tracking plots",
    )
    args = parser.parse_args()

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown_datasets = sorted(set(datasets) - set(METADATA_PATH))
    if unknown_datasets:
        parser.error(f"unknown dataset(s): {', '.join(unknown_datasets)}")

    model_path = args.model_path.expanduser()
    if not model_path.exists():
        parser.error(f"AMT model not found: {model_path}")

    try:
        load_model_fn, transcribe_fn, save_midi_fn = load_online_amt()
    except RuntimeError as error:
        parser.error(str(error))

    print(f"Loading AMT model from {model_path} ...", flush=True)
    model, _ = load_model_fn(str(model_path), args.device)
    print(f"Model loaded on {args.device}", flush=True)

    method_kwargs = DEFAULT_KWARGS["midi"].get(args.method, {})
    processor = method_kwargs.get("processor")

    for dataset in datasets:
        config = SymbolicEvalConfig(
            method=args.method,
            dataset=dataset,
            processor=processor,
        )
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        run_dir = (
            OUTPUT_DIR
            / f"test_{timestamp}_amt_pipeline_{args.method}_{dataset}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n========== Method: {args.method}, Dataset: {dataset}, "
            f"Output: {run_dir} ==========",
            flush=True,
        )
        results = run_dataset(
            dataset,
            args.method,
            run_dir,
            model,
            args.device,
            transcribe_fn,
            save_midi_fn,
            save_plots=not args.no_plots,
        )

        n_total = len(results["Index"])
        n_tracked = sum(results["tracked"])
        print(f"\nTracked: {n_tracked}/{n_total}", flush=True)

        summary_all = compute_event_pooled_summary(
            results,
            run_dir,
            tracked_only=False,
        )
        summary_tracked = compute_event_pooled_summary(
            results,
            run_dir,
            tracked_only=True,
        )
        add_timing_summary(summary_all, results, tracked_only=False)
        add_timing_summary(summary_tracked, results, tracked_only=True)

        for filename, summary in (
            ("summary_all.json", summary_all),
            ("summary_tracked.json", summary_tracked),
        ):
            with open(run_dir / filename, "w") as file:
                json.dump(summary, file, indent=4)

        results_path = run_dir / "test_results.tsv"
        save_results_to_csv(results, results_path)
        save_config(config, run_dir)
        print(f"Results saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
