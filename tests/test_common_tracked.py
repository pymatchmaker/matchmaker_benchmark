"""Precision on the pieces every compared method tracks."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common_tracked import CommonTrackedError, build_report, format_table, main

GT = [(0.5 * i, 0.5 * i) for i in range(1, 21)]
DATASETS = ["asap", "asap", "batik"]


def write_run(root: Path, name: str, offsets, tracked, fold="f0", partial=False):
    """One run over three pieces: each path is the ground truth shifted by an offset."""
    run = root / name
    run.mkdir(parents=True)
    pieces = []
    for index, (offset, is_tracked, dataset) in enumerate(zip(offsets, tracked, DATASETS), 1):
        for kind, shift in (("gt", 0.0), ("wp", offset)):
            rows = ["perf_sec\tscore_beat"] + [f"{a}\t{b + shift}" for a, b in GT]
            (run / f"{kind}_{index}.tsv").write_text("\n".join(rows) + "\n")
        pieces.append({"index": index, "piece_id": f"{dataset}/p{index}",
                       "dataset": dataset, "tracked": is_tracked, "rtf": 0.1})
    (run / "metrics.json").write_text(json.dumps({
        "submission": name, "fold_sha256": fold, "input_type": "audio",
        "partial": partial, "pieces": pieces,
    }))
    return run


def test_common_subset_is_the_intersection_of_the_defining_runs(tmp_path):
    a = write_run(tmp_path, "a", [0.0, 0.2, 0.0], [True, True, False])
    b = write_run(tmp_path, "b", [0.1, 0.0, 0.0], [True, False, True])
    report = build_report([a, b])
    assert report["n_common"] == 1
    assert report["common_pieces"] == ["asap/p1"]
    assert report["n_common_by_dataset"] == {"asap": 1, "batik": 0}
    assert report["methods"]["a"]["overall"]["common"]["beat"]["mean"] == 0.0
    assert report["methods"]["b"]["overall"]["common"]["beat"]["mean"] == pytest.approx(0.1)


def test_tracked_subset_is_each_methods_own(tmp_path):
    a = write_run(tmp_path, "a", [0.0, 0.2, 0.0], [True, True, False])
    b = write_run(tmp_path, "b", [0.1, 0.0, 0.0], [True, False, True])
    overall = build_report([a, b])["methods"]["a"]["overall"]
    assert overall["n_tracked"] == 2
    assert overall["tracked"]["beat"]["mean"] == pytest.approx(0.1)


def test_report_entries_do_not_shape_the_subset(tmp_path):
    a = write_run(tmp_path, "a", [0.0, 0.0, 0.0], [True, True, True])
    side = write_run(tmp_path, "side", [0.5, 0.0, 0.0], [False, True, False])
    report = build_report([a], [side])
    assert report["n_common"] == 3
    common = report["methods"]["side"]["overall"]["common"]
    # every common piece is pooled, including the two this entry lost
    assert common["selected_count"] == 3
    assert report["methods"]["side"]["defines_subset"] is False


def test_per_dataset_blocks(tmp_path):
    a = write_run(tmp_path, "a", [0.0, 0.2, 0.4], [True, True, True])
    datasets = build_report([a])["methods"]["a"]["datasets"]
    assert datasets["asap"]["n_common"] == 2
    assert datasets["batik"]["common"]["beat"]["mean"] == pytest.approx(0.4)


def test_runs_on_different_folds_are_refused(tmp_path):
    a = write_run(tmp_path, "a", [0, 0, 0], [True] * 3, fold="f0")
    b = write_run(tmp_path, "b", [0, 0, 0], [True] * 3, fold="f1")
    with pytest.raises(CommonTrackedError, match="different fold"):
        build_report([a, b])


def test_partial_runs_are_refused(tmp_path):
    a = write_run(tmp_path, "a", [0, 0, 0], [True] * 3, partial=True)
    with pytest.raises(CommonTrackedError, match="partial"):
        build_report([a])


def test_cli_writes_json_and_table(tmp_path):
    a = write_run(tmp_path, "a", [0.0, 0.2, 0.0], [True, True, False])
    b = write_run(tmp_path, "b", [0.1, 0.0, 0.0], [True, True, True])
    out = tmp_path / "out" / "audio.json"
    main(["--runs", str(a), str(b), "--output", str(out)])
    assert json.loads(out.read_text())["n_common"] == 2
    table = out.with_suffix(".md").read_text()
    assert "| overall | a |" in table
    assert format_table(build_report([a, b])) in table
