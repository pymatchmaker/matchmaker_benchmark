"""Withdrawing a published result.

A leaderboard row is derived from ``results/submissions/<entry>/metrics.json``,
so withdrawing one is deleting that directory and rebuilding. These tests pin
the parts that are easy to get wrong: the per-piece detail file must go too, the
withdrawal must be recorded, and it must be reversible.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matchmaker_eval.retract as R


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A results/ tree with one published entry, isolated from the repo."""
    results = tmp_path / "results"
    submissions = results / "submissions"
    details = results / "details"
    (submissions / "demo-midi").mkdir(parents=True)
    details.mkdir(parents=True)

    (submissions / "demo-midi" / "metrics.json").write_text(
        json.dumps(
            {
                "submission": "demo-midi",
                "method": "demo",
                "kind": "reference",
                "input_type": "midi",
                "fold": "eval",
                "fold_sha256": "deadbeef",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "n_pieces": 146,
                "n_tracked": 100,
                "summary_all": {"tracking_rate": 0.685},
                "summary_tracked": {"beat": {"median": 0.05}},
                "environment": {"matchmaker": "0.3.0"},
            }
        )
    )
    (submissions / "demo-midi" / "wp_1.tsv").write_text("perf_sec\tscore_beat\n0\t0\n")
    (details / "demo-midi.json").write_text('{"submission":"demo-midi"}')

    monkeypatch.setattr(R, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(R, "RESULTS_DIR", results)
    monkeypatch.setattr(R, "SUBMISSION_RESULTS", submissions)
    monkeypatch.setattr(R, "DETAILS_DIR", details)
    monkeypatch.setattr(R, "RETRACTED_PATH", results / "retracted.json")
    monkeypatch.setattr(R, "ARCHIVE_DIR", results / "retracted")
    return tmp_path


class TestRetract:
    def test_the_run_leaves_results_submissions(self, workspace):
        R.retract("demo-midi", "a bug")
        assert not (workspace / "results" / "submissions" / "demo-midi").exists()

    def test_the_detail_file_goes_too(self, workspace):
        """Otherwise a withdrawn result is still readable at its details URL."""
        detail = workspace / "results" / "details" / "demo-midi.json"
        assert detail.exists()
        R.retract("demo-midi", "a bug")
        assert not detail.exists()

    def test_the_withdrawal_is_recorded_with_its_reason(self, workspace):
        R.retract("demo-midi", "tempo model bug")
        records = R.read_retractions()
        assert len(records) == 1
        assert records[0]["submission"] == "demo-midi"
        assert records[0]["reason"] == "tempo model bug"
        assert records[0]["retracted_at"]

    def test_the_published_numbers_are_kept_for_audit(self, workspace):
        R.retract("demo-midi", "a bug")
        published = R.read_retractions()[0]["published"]
        assert published["n_pieces"] == 146
        assert published["n_tracked"] == 100
        assert published["tracking_rate"] == 0.685
        assert published["fold_sha256"] == "deadbeef"
        assert published["environment"]["matchmaker"] == "0.3.0"

    def test_it_is_archived_and_restorable(self, workspace):
        R.retract("demo-midi", "a bug")
        archived = workspace / "results" / "retracted" / "demo-midi" / "metrics.json"
        assert archived.exists()

        R.undo("demo-midi")
        assert (
            workspace / "results" / "submissions" / "demo-midi" / "metrics.json"
        ).exists()
        assert R.read_retractions() == []

    def test_purge_leaves_nothing_to_restore(self, workspace):
        R.retract("demo-midi", "a bug", archive=False)
        assert not (workspace / "results" / "retracted").exists()
        with pytest.raises(R.RetractionError, match="no archived result"):
            R.undo("demo-midi")

    def test_withdrawing_twice_keeps_one_record(self, workspace):
        R.retract("demo-midi", "first")
        R.undo("demo-midi")
        R.retract("demo-midi", "second")
        records = R.read_retractions()
        assert len(records) == 1
        assert records[0]["reason"] == "second"

    def test_an_unknown_entry_is_refused(self, workspace):
        with pytest.raises(R.RetractionError, match="not a published result"):
            R.retract("no-such-entry", "a bug")

    def test_undo_refuses_to_overwrite_a_live_result(self, workspace):
        R.retract("demo-midi", "a bug")
        (workspace / "results" / "submissions" / "demo-midi").mkdir()
        with pytest.raises(R.RetractionError, match="already exists"):
            R.undo("demo-midi")


class TestReasonIsRequired:
    def test_the_cli_refuses_a_withdrawal_with_no_reason(self):
        import subprocess

        done = subprocess.run(
            [sys.executable, str(REPO_ROOT / "matchmaker_eval" / "retract.py"), "x"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert done.returncode != 0
        assert "--reason is required" in done.stderr


class TestOrphanedDetails:
    def test_export_details_removes_a_detail_with_no_run(self, tmp_path, monkeypatch):
        import export_details as E

        details = tmp_path / "details"
        details.mkdir()
        (details / "gone.json").write_text("{}")
        (details / "kept.json").write_text("{}")
        monkeypatch.setattr(E, "DETAILS_DIR", details)

        removed = E.remove_orphans({"kept.json"})
        assert removed == ["gone.json"]
        assert not (details / "gone.json").exists()
        assert (details / "kept.json").exists()

    def test_dry_run_reports_without_deleting(self, tmp_path, monkeypatch):
        import export_details as E

        details = tmp_path / "details"
        details.mkdir()
        (details / "gone.json").write_text("{}")
        monkeypatch.setattr(E, "DETAILS_DIR", details)

        removed = E.remove_orphans(set(), dry_run=True)
        assert removed == ["gone.json"]
        assert (details / "gone.json").exists()
