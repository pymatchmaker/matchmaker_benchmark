"""The evaluation key: one hash for the conditions a run was scored under.

Rows are only ranked against rows that share it, a key is never computed over
part of the data, shards scored under different conditions are never merged,
and a key's leaderboard stays in the archive once a newer key takes over.
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matchmaker_eval.leaderboard as L
from matchmaker_eval import provenance as P
from matchmaker_eval.export_details import export
from merge_shards import MergeError, merge

CODE = {
    "scheme": "ek1",
    "fold_sha256": "f" * 64,
    "scoring_sha256": "s" * 64,
    "matchmaker_sha256": "m" * 64,
    "partitura": "1.9.0",
}


class FakePiece:
    """The parts of folds.Piece the data hash reads."""

    def __init__(self, root: Path):
        self.root = root
        self.score_path = root / "score" / "a.musicxml"
        self.match_path = root / "match" / "a.match"
        self._performance = root / "midi" / "a.mid"
        for path, data in (
            (self.score_path, b"<score/>"),
            (self.match_path, b"match"),
            (self._performance, b"MThd"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def performance_path(self, input_type):
        return self._performance


def records(n=3, prefix="h"):
    return [
        {"index": i, "piece_id": f"asap/{i}", "data_sha256": f"{prefix}{i}"}
        for i in range(n)
    ]


class TestTheKey:
    def test_it_names_its_scheme_and_is_short(self):
        key = P.evaluation_record(CODE, records())["key"]
        assert re.fullmatch(r"ek1-[0-9a-f]{12}", key)

    def test_the_same_conditions_give_the_same_key(self):
        assert (
            P.evaluation_record(CODE, records())["key"]
            == P.evaluation_record(dict(CODE), records())["key"]
        )

    @pytest.mark.parametrize("component", sorted(set(CODE) - {"scheme"}))
    def test_any_changed_component_changes_the_key(self, component):
        changed = {**CODE, component: CODE[component] + "x"}
        assert (
            P.evaluation_record(changed, records())["key"]
            != P.evaluation_record(CODE, records())["key"]
        )

    def test_changed_data_changes_the_key(self):
        assert (
            P.evaluation_record(CODE, records(prefix="other"))["key"]
            != P.evaluation_record(CODE, records())["key"]
        )

    def test_record_order_does_not_matter(self):
        assert (
            P.evaluation_record(CODE, list(reversed(records())))["key"]
            == P.evaluation_record(CODE, records())["key"]
        )

    def test_no_key_over_part_of_the_data(self):
        pieces = records()
        pieces[1]["data_sha256"] = None
        record = P.evaluation_record(CODE, pieces)
        assert record["key"] is None
        assert record["components"]["data_sha256"] is None

    def test_no_key_without_every_code_component(self):
        assert P.evaluation_record({**CODE, "partitura": None}, records())["key"] is None


class TestTheComponents:
    def test_a_piece_hash_follows_its_bytes(self, tmp_path):
        piece = FakePiece(tmp_path / "asap")
        before = P.piece_data_sha256(piece, "midi")
        assert P.piece_data_sha256(piece, "midi") == before
        piece.match_path.write_bytes(b"corrected match")
        assert P.piece_data_sha256(piece, "midi") != before

    def test_the_scoring_hash_follows_a_threshold(self):
        base = {"TRACKING_THRESHOLD": {"midi": 0.5, "audio": 1.0}}
        tighter = {"TRACKING_THRESHOLD": {"midi": 0.4, "audio": 1.0}}
        assert P.scoring_sha256(base) != P.scoring_sha256(tighter)

    def test_the_scoring_hash_follows_a_function(self):
        def scorer():
            return 1

        def other_scorer():
            return 2

        assert P.scoring_sha256({"run_piece": scorer}) != P.scoring_sha256(
            {"run_piece": other_scorer}
        )

    def test_the_matchmaker_hash_is_stable(self):
        assert P.matchmaker_sha256() == P.matchmaker_sha256()

    def test_a_run_records_every_code_component(self):
        from matchmaker_eval.run_submission import scoring_parts

        code = P.code_components("f" * 64, scoring_parts())
        assert set(code) == set(P.SHARED_COMPONENTS)
        assert all(code.values())


def write_shard(root: Path, index: int, piece: dict, code: dict) -> None:
    d = root / f"shard-{index}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "submission": "demo-midi",
        "method": "demo",
        "kind": "submission",
        "metadata": {"name": "Demo", "authors": ["a"]},
        "fold": "eval",
        "fold_sha256": code["fold_sha256"],
        "input_type": "midi",
        "environment": {"matchmaker": "0.3.0"},
        "timestamp": "2026-01-01T00:00:00+00:00",
        "fold_size": 2,
        "n_pieces": 1,
        "partial": True,
        "failures": [],
        "evaluation": P.evaluation_record(code, [piece]),
        "pieces": [piece],
    }))


class TestMerging:
    def test_the_merged_key_is_the_whole_fold_key(self, tmp_path):
        pieces = [
            {"index": 0, "piece_id": "asap/a", "dataset": "asap", "tracked": False, "data_sha256": "h0"},
            {"index": 1, "piece_id": "asap/b", "dataset": "asap", "tracked": False, "data_sha256": "h1"},
        ]
        for i, piece in enumerate(pieces):
            write_shard(tmp_path / "shards", i, piece, CODE)
        merged = merge(tmp_path / "shards", tmp_path / "out")
        assert merged["evaluation"] == P.evaluation_record(CODE, pieces)
        assert merged["evaluation"]["key"] is not None

    def test_shards_scored_differently_are_not_merged(self, tmp_path):
        piece = {"index": 0, "piece_id": "asap/a", "dataset": "asap", "tracked": False, "data_sha256": "h0"}
        write_shard(tmp_path / "shards", 0, piece, CODE)
        write_shard(
            tmp_path / "shards", 1, {**piece, "index": 1},
            {**CODE, "scoring_sha256": "t" * 64},
        )
        with pytest.raises(MergeError, match="scoring_sha256"):
            merge(tmp_path / "shards", tmp_path / "out")


def entry(
    name: str, key, evaluated: str, components=None, tracking=1.0, input_type="midi"
) -> dict:
    """A complete eval-fold metrics.json, as leaderboard.collect() reads it."""
    return {
        "submission": name,
        "kind": "submission",
        "method": name,
        "metadata": {"name": name, "authors": ["a"]},
        "fold": "eval",
        "fold_sha256": "f" * 64,
        "input_type": input_type,
        "timestamp": evaluated,
        "environment": {"matchmaker": "0.3.0", "matchmaker_commit": "c" * 40},
        "evaluation": {"key": key, "components": components or {}},
        "partial": False,
        "n_pieces": 2,
        "n_tracked": 2,
        "summary_tracked": {"tracking_rate": tracking, "median_b": 0.1, "smoothness": -3.0},
        "summary_all": {"tracking_rate": tracking, "median_b": 0.1, "smoothness": -3.0},
        "datasets": {},
        "pieces": [],
    }


@pytest.fixture
def results(tmp_path, monkeypatch):
    results = tmp_path / "results"
    monkeypatch.setattr(L, "RESULTS_DIR", results)
    monkeypatch.setattr(L, "SUBMISSION_RESULTS", results / "submissions")
    monkeypatch.setattr(L, "LEADERBOARD_JSON", results / "leaderboard.json")
    monkeypatch.setattr(L, "LEADERBOARD_CSV", results / "leaderboard.csv")
    monkeypatch.setattr(L, "read_retractions", lambda: [])

    def add(record):
        d = results / "submissions" / record["submission"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps(record))

    return add


OLD = {**CODE, "data_sha256": "d" * 64}
NEW = {**OLD, "scoring_sha256": "n" * 64}


class TestTheLeaderboard:
    def test_only_rows_with_the_newest_key_are_ranked(self, results):
        results(entry("old-ref", "ek1-aaaaaaaaaaaa", "2026-10-01T00:00:00+00:00", OLD))
        results(entry("new-ref", "ek1-bbbbbbbbbbbb", "2026-10-28T00:00:00+00:00", NEW, 0.5))
        board = L.build()
        assert board["evaluation_keys"] == {"midi": "ek1-bbbbbbbbbbbb"}
        assert [(e["submission"], e["rank"]) for e in board["entries"]] == [("new-ref", 1)]
        (earlier,) = board["earlier_conditions"]
        assert earlier["submission"] == "old-ref"
        assert earlier["rank"] is None
        assert earlier["evaluation_changed"] == ["scoring_sha256"]

    def test_a_row_from_before_keys_is_not_ranked_with_keyed_rows(self, results):
        results(entry("legacy", None, "2026-10-01T00:00:00+00:00"))
        results(entry("new-ref", "ek1-bbbbbbbbbbbb", "2026-10-28T00:00:00+00:00", NEW))
        board = L.build()
        assert [e["submission"] for e in board["entries"]] == ["new-ref"]
        (legacy,) = board["earlier_conditions"]
        assert legacy["submission"] == "legacy"
        assert legacy["evaluation_changed"] == []

    def test_without_any_key_every_row_is_ranked(self, results):
        results(entry("a", None, "2026-10-01T00:00:00+00:00"))
        results(entry("b", None, "2026-10-02T00:00:00+00:00", tracking=0.5))
        board = L.build()
        assert [e["rank"] for e in board["entries"]] == [1, 2]
        assert board["earlier_conditions"] == []

    def test_midi_and_audio_rank_together_under_their_own_keys(self, results):
        results(entry("midi-ref", "ek1-aaaaaaaaaaaa", "2026-10-28T00:00:00+00:00", OLD))
        results(entry(
            "audio-ref", "ek1-cccccccccccc", "2026-10-28T01:00:00+00:00",
            {**OLD, "data_sha256": "a" * 64}, 0.5, "audio",
        ))
        board = L.build()
        assert board["evaluation_keys"] == {
            "audio": "ek1-cccccccccccc", "midi": "ek1-aaaaaaaaaaaa",
        }
        assert [(e["submission"], e["rank"]) for e in board["entries"]] == [
            ("midi-ref", 1), ("audio-ref", 2),
        ]
        assert board["earlier_conditions"] == []

    def test_an_input_type_scored_by_older_code_is_not_ranked(self, results):
        results(entry(
            "audio-ref", "ek1-cccccccccccc", "2026-10-01T00:00:00+00:00",
            {**OLD, "data_sha256": "a" * 64}, input_type="audio",
        ))
        results(entry("midi-ref", "ek1-bbbbbbbbbbbb", "2026-10-28T00:00:00+00:00", NEW))
        board = L.build()
        assert board["evaluation_keys"] == {"midi": "ek1-bbbbbbbbbbbb"}
        assert [e["submission"] for e in board["entries"]] == ["midi-ref"]
        (earlier,) = board["earlier_conditions"]
        assert earlier["submission"] == "audio-ref"
        # Compared on code only: MIDI's data hash says nothing about audio.
        assert earlier["evaluation_changed"] == ["scoring_sha256"]

    def test_the_key_is_a_csv_column(self, results):
        results(entry("new-ref", "ek1-bbbbbbbbbbbb", "2026-10-28T00:00:00+00:00", NEW))
        L.write(L.build())
        header = L.LEADERBOARD_CSV.read_text().splitlines()[0].split(",")
        assert "evaluation_key" in header


class TestTheArchive:
    def test_a_key_keeps_its_table_after_a_newer_key(self, results, tmp_path):
        results(entry("ref", "ek1-aaaaaaaaaaaa", "2026-10-01T00:00:00+00:00", OLD))
        L.write(L.build())
        archive = L.RESULTS_DIR / "archive"
        first = (archive / "ek1-aaaaaaaaaaaa" / "leaderboard.json").read_text()
        assert json.loads(first)["entries"][0]["submission"] == "ref"

        results(entry("ref", "ek1-bbbbbbbbbbbb", "2026-10-28T00:00:00+00:00", NEW))
        L.write(L.build())
        assert (archive / "ek1-aaaaaaaaaaaa" / "leaderboard.json").read_text() == first
        assert (archive / "ek1-bbbbbbbbbbbb" / "leaderboard.json").exists()
        index = json.loads((archive / "index.json").read_text())["archives"]
        assert [a["key"] for a in index] == ["ek1-aaaaaaaaaaaa", "ek1-bbbbbbbbbbbb"]
        assert index[1]["conditions"]["scoring_sha256"] == "n" * 64
        assert index[1]["input_type"] == "midi"

    def test_each_input_type_is_archived_under_its_own_key(self, results):
        results(entry("midi-ref", "ek1-aaaaaaaaaaaa", "2026-10-28T00:00:00+00:00", OLD))
        results(entry(
            "audio-ref", "ek1-cccccccccccc", "2026-10-28T01:00:00+00:00",
            {**OLD, "data_sha256": "a" * 64}, 0.5, "audio",
        ))
        L.write(L.build())
        archive = L.RESULTS_DIR / "archive"
        audio = json.loads((archive / "ek1-cccccccccccc" / "leaderboard.json").read_text())
        assert audio["input_type"] == "audio"
        assert audio["evaluation_conditions"]["data_sha256"] == "a" * 64
        assert [(e["submission"], e["rank"]) for e in audio["entries"]] == [("audio-ref", 1)]
        index = json.loads((archive / "index.json").read_text())["archives"]
        assert {a["key"]: a["input_type"] for a in index} == {
            "ek1-aaaaaaaaaaaa": "midi", "ek1-cccccccccccc": "audio",
        }

    def test_nothing_is_archived_without_a_key(self, results):
        results(entry("legacy", None, "2026-10-01T00:00:00+00:00"))
        L.write(L.build())
        assert not (L.RESULTS_DIR / "archive").exists()


class TestTheDetailExport:
    def test_the_data_hash_is_provenance_not_a_metric(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        (run / "metrics.json").write_text(json.dumps({
            "submission": "demo", "input_type": "midi",
            "evaluation": P.evaluation_record(CODE, records(1)),
            "pieces": [{"index": 0, "piece_id": "asap/0", "dataset": "asap",
                        "tracked": True, "mean_b": 0.1, "data_sha256": "h0"}],
        }))
        detail = export(run, include_paths=False)
        assert "data_sha256" not in detail["metrics"]
        assert detail["pieces"][0]["data_sha256"] == "h0"
        assert detail["evaluation"]["key"].startswith("ek1-")
