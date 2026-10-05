"""The per-dataset view of a result.

A run's ``metrics.json`` carries its pooled summaries once for the whole fold
and once per dataset, in the same shape and from the same function. Everything
downstream — the ``datasets`` block on a leaderboard row, the per-dataset
leaderboard files, the detail file the page reads — is cut from that block,
so a number reported for one dataset can never be pooled differently from the
number above it.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matchmaker_eval.leaderboard as L
from merge_shards import merge
from utils import dataset_summaries, pooled_summaries

GT = [(0.5 * i, 0.5 * i) for i in range(1, 21)]


def write_path(run_dir: Path, name: str, rows) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = ["perf_sec\tscore_beat"] + [f"{a}\t{b}" for a, b in rows]
    (run_dir / name).write_text("\n".join(lines) + "\n")


@pytest.fixture
def run_dir(tmp_path):
    """Three pieces on two datasets: one exact, one 0.2 beats late, one crashed."""
    run = tmp_path / "run"
    write_path(run, "gt_0.tsv", GT)
    write_path(run, "wp_0.tsv", GT)
    write_path(run, "gt_1.tsv", GT)
    write_path(run, "wp_1.tsv", [(a, b + 0.2) for a, b in GT])
    return run


PIECES = [
    {"index": 0, "piece_id": "asap/a", "dataset": "asap", "tracked": True, "rtf": 0.1},
    {"index": 1, "piece_id": "batik/b", "dataset": "batik", "tracked": True, "rtf": 0.3},
    {"index": 2, "piece_id": "batik/c", "dataset": "batik", "tracked": False, "error": "boom"},
]


class TestPooling:
    def test_each_dataset_is_pooled_over_its_own_pieces(self, run_dir):
        blocks = dataset_summaries(PIECES, run_dir)
        assert list(blocks) == ["asap", "batik"]
        assert blocks["asap"]["summary_tracked"]["beat"]["median"] == 0.0
        assert blocks["batik"]["summary_tracked"]["beat"]["median"] == pytest.approx(0.2)

    def test_a_dataset_block_has_the_shape_of_the_run_record(self, run_dir):
        block = dataset_summaries(PIECES, run_dir)["batik"]
        assert block["n_pieces"] == 2
        assert block["n_tracked"] == 1
        assert block["n_failed"] == 1
        assert block["summary_all"]["tracking_rate"] == 0.5
        assert set(block["summary_tracked"]) >= {"beat", "ms", "rtf", "sparc"}

    def test_a_crashed_piece_does_not_shift_the_timing_columns(self, run_dir):
        """The crashed piece has no rtf; the others' must still line up."""
        assert pooled_summaries(PIECES, run_dir)["summary_tracked"]["rtf"] == 0.2
        assert dataset_summaries(PIECES, run_dir)["batik"]["summary_tracked"]["rtf"] == 0.3


def shard(index: int, piece: dict, run_dir: Path, root: Path) -> None:
    d = root / f"shard-{index}"
    d.mkdir(parents=True)
    for prefix in ("wp", "gt"):
        src = run_dir / f"{prefix}_{piece['index']}.tsv"
        if src.exists():
            (d / src.name).write_text(src.read_text())
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "submission": "demo-midi",
                "method": "demo",
                "kind": "submission",
                "metadata": {"name": "Demo", "authors": ["a"]},
                "fold": "eval",
                "fold_file": "data/folds/eval.csv",
                "fold_sha256": "abc",
                "input_type": "midi",
                "environment": {"matchmaker": "0.3.0"},
                "timestamp": "2026-01-01T00:00:00+00:00",
                "fold_size": len(PIECES),
                "n_pieces": 1,
                "partial": True,
                "failures": [],
                "pieces": [piece],
            }
        )
    )


@pytest.fixture
def merged(tmp_path, run_dir):
    """A complete record as the evaluation workflow produces it: from shards."""
    root = tmp_path / "shards"
    for i, piece in enumerate(PIECES):
        shard(i, piece, run_dir, root)
    return merge(root, tmp_path / "merged")


class TestTheRecord:
    def test_the_merge_carries_the_per_dataset_block(self, merged):
        assert set(merged["datasets"]) == {"asap", "batik"}
        assert merged["datasets"]["asap"]["summary_tracked"]["beat"]["median"] == 0.0

    def test_the_row_flattens_every_dataset_like_the_headline(self, merged):
        row = L.row_from_metrics(merged)
        headline = L.metric_columns(merged)
        assert set(row["datasets"]["batik"]) == set(headline)
        assert row["datasets"]["batik"]["tracking_rate"] == 0.5
        assert row["datasets"]["batik"]["n_failed"] == 1
        assert row["datasets"]["asap"]["beat_median"] == 0.0

    def test_the_headline_is_pooled_over_the_whole_run(self, merged):
        row = L.row_from_metrics(merged)
        assert row["n_pieces"] == 3
        assert row["tracking_rate"] == pytest.approx(2 / 3, abs=1e-4)


@pytest.fixture
def results(tmp_path, merged, monkeypatch):
    """A results/ tree with two entries, one of which never ran on batik."""
    results = tmp_path / "results"
    monkeypatch.setattr(L, "RESULTS_DIR", results)
    monkeypatch.setattr(L, "SUBMISSION_RESULTS", results / "submissions")
    monkeypatch.setattr(L, "LEADERBOARD_JSON", results / "leaderboard.json")
    monkeypatch.setattr(L, "LEADERBOARD_CSV", results / "leaderboard.csv")

    for name, record in (("demo-midi", merged), ("asap-only", dict(merged))):
        if name == "asap-only":
            record = json.loads(json.dumps(merged))
            record["submission"] = name
            record["datasets"] = {"asap": record["datasets"]["asap"]}
            record["datasets"]["asap"]["summary_tracked"]["beat"]["median"] = 0.1
        d = results / "submissions" / name
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps(record))
    monkeypatch.setattr(L, "read_retractions", lambda: [])
    return results


class TestDatasetBoards:
    def test_one_board_per_dataset_seen_in_any_run(self, results):
        boards = L.dataset_boards(L.build())
        assert list(boards) == ["asap", "batik"]
        assert [e["submission"] for e in boards["batik"]["entries"]] == ["demo-midi"]

    def test_a_board_ranks_within_its_dataset(self, results):
        board = L.dataset_boards(L.build())["asap"]
        ranks = [(e["rank"], e["submission"], e["beat_median"]) for e in board["entries"]]
        assert ranks == [(1, "demo-midi", 0.0), (2, "asap-only", 0.1)]

    def test_a_board_row_carries_the_dataset_columns_not_the_headline(self, results):
        entry = L.dataset_boards(L.build())["batik"]["entries"][0]
        assert entry["n_pieces"] == 2
        assert entry["tracking_rate"] == 0.5
        assert "datasets" not in entry

    def test_a_board_row_keeps_its_place_on_the_whole_fold(self, results):
        """First on asap, second overall: the two entries tie on the headline
        and the tie breaks on name, which is exactly what the column exposes."""
        entry = L.dataset_boards(L.build())["asap"]["entries"][0]
        assert entry["submission"] == "demo-midi"
        assert entry["rank"] == 1
        assert entry["overall_rank"] == 2
        assert entry["details"] == "details/demo-midi.json"
        assert "overall_rank" in L.DATASET_CSV_COLUMNS

    def test_write_produces_a_pair_per_dataset_and_removes_stale_ones(self, results):
        stale = results / "leaderboard-winterreise.json"
        stale.write_text("{}")
        written = L.write(L.build())
        assert (results / "leaderboard-asap.json").exists()
        assert (results / "leaderboard-batik.csv").exists()
        assert not stale.exists()
        assert "leaderboard-asap.csv" in written
        header = (results / "leaderboard-asap.csv").read_text().splitlines()[0]
        assert header.split(",")[:2] == ["rank", "overall_rank"]

    def test_the_index_row_carries_the_per_dataset_columns(self, results):
        L.write(L.build())
        board = json.loads((results / "leaderboard.json").read_text())
        entry = next(e for e in board["entries"] if e["submission"] == "demo-midi")
        assert entry["datasets"]["batik"]["beat_0.5b"] is not None
        assert entry["datasets"]["batik"]["sparc"] is not None


class TestTheDetailFile:
    def test_it_copies_the_run_and_per_dataset_summaries(self, tmp_path, merged):
        import export_details as E

        run = tmp_path / "merged"
        detail = E.export(run, include_paths=False)
        assert set(detail["datasets"]) == {"asap", "batik"}
        # Compared as JSON: a skewness over two events is NaN, and NaN != NaN.
        assert json.dumps(detail["datasets"]["batik"]["summary_tracked"]) == (
            json.dumps(merged["datasets"]["batik"]["summary_tracked"])
        )
        assert json.dumps(detail["overall"]["summary_tracked"]) == (
            json.dumps(merged["summary_tracked"])
        )
        assert detail["overall"]["n_pieces"] == 3


class TestPublishedJsonIsStrict:
    """A browser's response.json() rejects NaN, and with it the whole file."""

    def test_nan_and_infinity_become_null(self):
        value = {"a": [float("nan"), 1.0], "b": {"c": float("inf")}}
        assert L.without_nan(value) == {"a": [None, 1.0], "b": {"c": None}}

    def test_a_nan_skewness_does_not_reach_the_published_files(self, results):
        record_path = results / "submissions" / "demo-midi" / "metrics.json"
        record = json.loads(record_path.read_text())
        record["datasets"]["batik"]["summary_tracked"]["ms"]["skewness"] = float("nan")
        record_path.write_text(json.dumps(record))
        L.write(L.build())
        assert "NaN" not in (results / "leaderboard.json").read_text()
        assert "NaN" not in (results / "leaderboard-batik.json").read_text()

    def test_the_detail_file_is_written_the_same_way(self, tmp_path, monkeypatch, merged):
        import export_details as E

        run = tmp_path / "merged"
        record = json.loads((run / "metrics.json").read_text())
        record["summary_tracked"]["ms"]["kurtosis"] = float("nan")
        (run / "metrics.json").write_text(json.dumps(record))
        monkeypatch.setattr(E, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(E, "SUBMISSION_RESULTS", tmp_path)
        monkeypatch.setattr(E, "DETAILS_DIR", tmp_path / "details")
        monkeypatch.setattr(sys, "argv", ["export_details.py", "--no-paths"])
        assert E.main() == 0
        text = (tmp_path / "details" / "demo-midi.json").read_text()
        assert "NaN" not in text
        assert json.loads(text)["overall"]["summary_tracked"]["ms"]["kurtosis"] is None


class TestTheSite:
    def test_it_copies_the_per_dataset_files(self, tmp_path, results, monkeypatch):
        import build_site

        L.write(L.build())
        monkeypatch.setattr(build_site, "RESULTS_DIR", results)
        summary = build_site.build(tmp_path / "site")
        assert (tmp_path / "site" / "leaderboard-asap.csv").exists()
        assert (tmp_path / "site" / "leaderboard-batik.json").exists()
        assert sorted(summary["boards"]) == [
            "leaderboard-asap.csv",
            "leaderboard-asap.json",
            "leaderboard-batik.csv",
            "leaderboard-batik.json",
        ]
