"""Progress reporting for a long reference run.

A full audio fold takes hours per method, and the children are captured so
their output does not interleave -- which meant nothing was printed at all
until a method finished. These cover the parts that make a run followable.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_references as R


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(R, "RESULTS_DIR", tmp_path / "results" / "submissions")
    monkeypatch.setattr(R, "LOG_DIR", tmp_path / "results" / "logs")
    return tmp_path


def write_pieces(root: Path, n: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"wp_{i}.tsv").write_text("perf_sec\tscore_beat\n")


class TestRunDir:
    def test_the_eval_fold_writes_where_the_leaderboard_reads(self, results):
        assert R.run_dir_for("arzt", "audio", "eval") == (
            results / "results" / "submissions" / "arzt-audio"
        )

    def test_another_fold_writes_elsewhere(self, results):
        """A validation run must not overwrite a published record."""
        assert R.run_dir_for("arzt", "audio", "valid") == (
            results / "results" / "runs" / "valid" / "arzt-audio"
        )

    def test_metrics_path_follows_the_run_dir(self, results):
        assert R.metrics_path_for("arzt", "audio", "eval").parent == R.run_dir_for(
            "arzt", "audio", "eval"
        )


class TestPiecesDone:
    def test_it_counts_finished_pieces(self, results):
        write_pieces(R.run_dir_for("arzt", "audio", "eval"), 7)
        assert R.pieces_done("arzt", "audio", "eval") == 7

    def test_a_run_that_has_not_started_is_zero(self, results):
        assert R.pieces_done("nothing", "audio", "eval") == 0

    def test_other_files_are_not_counted(self, results):
        d = R.run_dir_for("arzt", "audio", "eval")
        write_pieces(d, 2)
        (d / "gt_0.tsv").write_text("")
        (d / "metrics.json").write_text("{}")
        assert R.pieces_done("arzt", "audio", "eval") == 2


class TestRerun:
    """A rerun starts from a clean run directory.

    The previous run's wp_*.tsv made pieces_done() read 146/146 from the
    first second, and would have been pooled into summary_all for any piece
    that crashed the second time.
    """

    def test_a_previous_runs_piece_files_are_cleared(self, tmp_path):
        from run_submission import clear_piece_outputs

        d = tmp_path / "run"
        write_pieces(d, 3)
        for name in ("gt_0.tsv", "0.json", "tracking_0.png"):
            (d / name).write_text("")
        (d / "metrics.json").write_text("{}")
        assert clear_piece_outputs(d) == 6
        assert sorted(f.name for f in d.iterdir()) == ["metrics.json"]

    def test_the_log_is_written_while_the_child_runs(self, results, monkeypatch):
        """tail -f must show this run, not the previous one until it ends."""
        seen = {}

        def fake_run(command, stdout, **kwargs):
            seen["streamed"] = stdout.writable() and not stdout.closed
            stdout.write("[1/2] piece\nboom\n")

            class Done:
                returncode = 1

            return Done()

        monkeypatch.setattr(R.subprocess, "run", fake_run)
        record = R.run_one("arzt", "midi", "eval", [])
        assert seen["streamed"]
        assert (R.LOG_DIR / "arzt-midi.log").read_text() == "[1/2] piece\nboom\n"
        assert record["tail"] == ["[1/2] piece", "boom"]


class TestProgress:
    def test_it_reports_every_job(self, results):
        write_pieces(R.run_dir_for("arzt", "audio", "eval"), 10)
        write_pieces(R.run_dir_for("skf", "audio", "eval"), 3)
        progress = R.Progress(
            [("arzt", "audio"), ("skf", "audio")], "eval", total=146
        )
        assert progress.snapshot() == [
            ("arzt-audio", 10, 146),
            ("skf-audio", 3, 146),
        ]

    def test_the_rendering_names_each_method_and_its_count(self, results):
        write_pieces(R.run_dir_for("arzt", "audio", "eval"), 10)
        text = R.Progress([("arzt", "audio")], "eval", total=146).render()
        assert "arzt-audio" in text
        assert "10/146" in text

    def test_it_totals_across_methods(self, results):
        write_pieces(R.run_dir_for("arzt", "audio", "eval"), 10)
        write_pieces(R.run_dir_for("skf", "audio", "eval"), 20)
        text = R.Progress(
            [("arzt", "audio"), ("skf", "audio")], "eval", total=100
        ).render()
        assert "30/200" in text

    def test_it_survives_a_run_directory_that_does_not_exist(self, results):
        text = R.Progress([("ghost", "midi")], "eval", total=146).render()
        assert "0/146" in text

    def test_a_zero_total_does_not_divide_by_zero(self, results):
        text = R.Progress([("arzt", "audio")], "eval", total=0).render()
        assert "arzt-audio" in text

    def test_the_monitor_thread_starts_and_stops(self, results):
        import io

        stream = io.StringIO()
        progress = R.Progress(
            [("arzt", "audio")], "eval", total=146, interval=0.01, stream=stream
        )
        with progress:
            import time

            time.sleep(0.05)
        assert "arzt-audio" in stream.getvalue()
        assert not progress._thread.is_alive()


class TestProgressCanBeSwitchedOff:
    def test_the_flag_exists_with_a_sensible_default(self):
        import subprocess

        done = subprocess.run(
            [sys.executable, str(REPO_ROOT / "matchmaker_eval" / "run_references.py"),
             "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert "--progress-interval" in done.stdout
        assert "--watch" in done.stdout
