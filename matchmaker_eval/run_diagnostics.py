"""Run and merge the manuscript diagnostic suites with the benchmark evaluator.

Inputs come from a manifest pinned to one commit of the data repository.
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
import traceback
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "matchmaker_eval"):
    sys.path.insert(0, str(p))

PERTURBATION_METHODS = {
    "audio": ["dixon", "arzt", "outerhmm", "skf"],
    "midi": ["pthmm", "hmm", "outerhmm", "dixon", "arzt", "SLT_OLTW", "dannenberg"],
}
CONFIGS = {
    **{f"perturbations-{t}-{m}": {"suite": "perturbations", "input_type": t, "method": m}
       for t, methods in PERTURBATION_METHODS.items() for m in methods},
    **{f"repeat_examples-audio-{m}": {"suite": "repeat_examples", "input_type": "audio", "method": m}
       for m in ["arzt", "arzt_multi_fold", "outerhmm"]},
    **{f"folded_batik-audio-{m}": {"suite": "folded_batik", "input_type": "audio", "method": m,
                                   "kwargs": {"s_j": 1e-5} if m == "outerhmm" else {}}
       for m in ["dixon", "arzt", "skf", "outerhmm", "arzt_multi_fold"]},
    **{f"global_skip-midi-{s}": {"suite": "global_skip", "input_type": "midi", "method": "outerhmm",
                                 "disable_global_skip": s == "off"}
       for s in ["on", "off"]},
}


def select_configs(only="", input_type="both"):
    """The configurations for the given methods (all if empty) and input type."""
    methods = set(only.split())
    return {name: c for name, c in CONFIGS.items()
            if input_type in ("both", c["input_type"]) and (not methods or c["method"] in methods)}


def select_cases(manifest, config, shard=0, shards=1):
    """The (index, case) pairs of one shard, round-robin over the suite's cases."""
    cases = [c for c in manifest["cases"]
             if c["suite"] == config["suite"] and c["input_type"] == config["input_type"]]
    return [(i, c) for i, c in enumerate(cases) if i % shards == shard]


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out)


def fetch_input(name, cache, base_url):
    """Download one input into ``cache`` unless it is already there."""
    path = cache / name
    if not path.exists():
        download(base_url + name, path)
    return path


def load_manifest(cache):
    """The pinned manifest, the pin itself, and the URL its files are served from."""
    from matchmaker_eval.fetch_data import frozen_source

    pin = frozen_source("diagnostics")
    base_url = f"https://raw.githubusercontent.com/{pin['repository']}/{pin['commit']}/diagnostics/"
    path = cache / "manifest.json"
    if not path.exists():
        download(base_url + "manifest.json", path)
    return pin, base_url, json.loads(path.read_text())


def run_case(case, config, paths, out):
    import numpy as np
    from matchmaker import Matchmaker
    from matchmaker.matchmaker import DEFAULT_KWARGS
    from matchmaker.utils.eval import resolve_gt

    from matchmaker_eval.eval import run_evaluation
    from matchmaker_eval.run_submission import check_follower, piece_time_budget
    from matchmaker_eval.utils import TOLERANCES_IN_BEATS
    from matchmaker_eval.verify_tracking import (
        TRACKING_THRESHOLDS, check_tracking, musical_beat_factors, score_meter_map,
    )

    input_type, method = config["input_type"], config["method"]
    performance = paths[f"{input_type}_performance"]
    kwargs = dict(DEFAULT_KWARGS[input_type][method])
    if input_type == "audio" and method == "outerhmm":
        kwargs["s_j"] = 1e-100  # the diagnostics test jumps, which the default disables
    kwargs.update(config.get("kwargs", {}))
    out.mkdir(parents=True, exist_ok=True)

    with piece_time_budget(2400):
        mm = Matchmaker(score_file=str(paths["xml_score"]), performance_file=str(performance),
                        method=method, input_type=input_type, wait=False,
                        unfold_score=case["unfold_score"], tempo=case["tempo"], kwargs=kwargs)
        check_follower(mm.score_follower)
        if config.get("disable_global_skip"):
            mm.score_follower.S[:] = 0.0  # OPHMM's skip-from distribution
        for _ in mm.run(verbose=False):
            pass
        times, beats = resolve_gt(paths["match"], mm.score_part.note_array(),
                                  performance=str(performance) if input_type == "midi" else None)
        gt = np.column_stack((times, beats))
        metrics = run_evaluation(mm, gt=gt, tolerances=TOLERANCES_IN_BEATS, domain="score",
                                 debug=True, save_dir=out, run_name="0",
                                 make_plot=False, plot_dist_matrix=False)
        wp = mm.score_follower.alignment_path.T
        meter = score_meter_map(mm.score_part)
        tracking = check_tracking(wp, gt, threshold=TRACKING_THRESHOLDS[input_type],
                                  segment_duration=30.0,
                                  beat_factors=musical_beat_factors(gt[:, 1], meter))

    (out / "meter_0.json").write_text(json.dumps(meter, indent=2) + "\n")
    for name, array in [("wp_0", wp), ("gt_0", gt)]:
        np.savetxt(out / f"{name}.tsv", array, delimiter="\t", fmt="%.12g",
                   header="perf_sec\tscore_beat", comments="")

    wp = wp[np.argsort(wp[:, 0], kind="stable")]
    index = np.searchsorted(wp[:, 0], gt[:, 0], side="right") - 1
    valid = index >= 0
    errors = np.full(len(gt), np.nan)
    errors[valid] = np.abs(gt[valid, 1] - wp[index[valid], 1])

    intervals = {}
    if "repeat_metadata" in case:
        start = case["repeat_metadata"]["replay_start_sec"]
        end = case["repeat_metadata"]["replay_end_sec"]
        t = gt[:, 0]
        for name, mask in [("before", t < start), ("replay", (t >= start) & (t < end)),
                           ("after", t >= end)]:
            intervals[name] = {"onsets": int(mask.sum()),
                               "within_1_beat": float(np.mean(errors[mask] <= 1)),
                               "median_beat_error": float(np.nanmedian(errors[mask]))}

    return {
        "case_id": case["id"], "category": case["category"],
        "method": method, "input_type": input_type, "kwargs": kwargs,
        "unfold_score": case["unfold_score"], "tempo": case["tempo"],
        "disable_global_skip": config.get("disable_global_skip", False),
        "tracked": bool(tracking["tracked"]), "max_deviation": float(tracking["max_deviation"]),
        "tracking_criterion": {k: tracking[k] for k in ("error_unit", "threshold", "window_duration")},
        "mean_b": float(np.nanmean(errors)), "n_events": int(valid.sum()),
        "n_annotated_onsets": len(gt), "metrics": metrics, "intervals": intervals,
    }


def write_json(path, value):
    def default(v):
        return f"{v.__module__}.{v.__qualname__}" if isinstance(v, type) else float(v)

    path.write_text(json.dumps(value, indent=2, default=default) + "\n")


def run(args):
    from matchmaker_eval.run_submission import environment_info

    args.cache.mkdir(parents=True, exist_ok=True)
    pin, base_url, manifest = load_manifest(args.cache)
    config = CONFIGS[args.config]
    selected = select_cases(manifest, config, args.shard, args.shards)
    inputs = ["xml_score", "match", f"{config['input_type']}_performance"]
    result = {
        "config": args.config, "definition": config, "shard": args.shard, "shards": args.shards,
        "data": pin, "environment": environment_info(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "pieces": [], "failures": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)

    for i, case in selected:
        case_dir = args.output / f"{i:03d}"
        case_dir.mkdir(exist_ok=True)
        try:
            paths = {k: fetch_input(case["paths"][k], args.cache, base_url) for k in inputs}
            with (case_dir / "run.log").open("w") as log, \
                    contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                record = run_case(case, config, paths, case_dir)
            record["directory"] = case_dir.name
            write_json(case_dir / "result.json", record)
            result["pieces"].append(record)
            print(args.config, case["id"], record["tracked"], record["mean_b"], flush=True)
        except Exception:
            error = traceback.format_exc()
            (case_dir / "error.txt").write_text(error)
            result["failures"].append({"case_id": case["id"], "error": error})
            print(error, flush=True)
    write_json(args.output / "metrics.json", result)


def summarize(pieces):
    """Per category: how many cases were tracked, and their mean beat error."""
    import numpy as np

    by_category = defaultdict(list)
    for piece in pieces:
        by_category[piece["category"]].append(piece)
    summary = {}
    for category, group in by_category.items():
        tracked = [p["mean_b"] for p in group if p["tracked"]]
        summary[category] = {"total": len(group), "tracked": len(tracked),
                             "mean_b_tracked": float(np.mean(tracked)) if tracked else None,
                             "mean_b_all": float(np.mean([p["mean_b"] for p in group]))}
    return summary


def merge(args):
    records = [json.loads(p.read_text()) for p in args.source.glob("*/metrics.json")]
    args.output.mkdir(parents=True, exist_ok=True)

    pieces = defaultdict(list)
    for record in records:
        pieces[record["config"]].extend(record["pieces"])
    summary = {name: summarize(group) for name, group in sorted(pieces.items())}

    for shard_dir in args.source.iterdir():
        if shard_dir.is_dir():
            shutil.copytree(shard_dir, args.output / "raw" / shard_dir.name, dirs_exist_ok=True)
    write_json(args.output / "summary.json", {"configurations": summary})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run one shard of one configuration")
    run_parser.add_argument("--config", choices=CONFIGS, required=True)
    run_parser.add_argument("--shard", type=int, default=0)
    run_parser.add_argument("--shards", type=int, default=1)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--cache", type=Path, default=Path("diagnostic-data"))
    merge_parser = commands.add_parser("merge", help="combine downloaded shards")
    merge_parser.add_argument("--source", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
