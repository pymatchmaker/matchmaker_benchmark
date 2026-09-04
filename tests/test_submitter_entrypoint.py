"""The one command a submitter runs.

run_submission.py is the engine and carries twelve CI-oriented flags; a
submitter needs one of them. This front door is locked to the tuning fold on
purpose — the eval fold is what the leaderboard measures, and not providing a
convenient way to develop against it is the cheapest way to keep the
declaration in a submission's metadata.yaml true.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_submission as entry

SCRIPT = REPO_ROOT / "matchmaker_eval" / "test_submission.py"


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestItIsLockedToTuning:
    def test_there_is_no_fold_option(self):
        assert "--fold" not in run("--help").stdout, (
            "a submitter must not be able to point this at the eval fold"
        )

    def test_the_fold_is_tuning(self):
        assert entry.TUNING_FOLD == "tuning"

    def test_the_help_says_which_fold_and_why(self):
        text = run("--help").stdout
        assert "tuning" in text


class TestArguments:
    def test_a_directory_or_a_method_but_not_both(self):
        done = run("submissions/baseline-constant-tempo", "--method", "pthmm")
        assert done.returncode != 0
        assert "not both" in done.stderr

    def test_neither_is_refused(self):
        done = run()
        assert done.returncode != 0
        assert "not both" in done.stderr


class TestTheSummary:
    def metrics(self, **over):
        record = {
            "submission": "demo",
            "input_type": "midi",
            "n_tracked": 3,
            "n_pieces": 4,
            "summary_all": {
                "tracking_rate": 0.75, "rtf": 0.21, "beat": {"median": 0.4}
            },
            "summary_tracked": {"beat": {"median": 0.05}},
            "pieces": [
                {"piece_id": "asap/a/a", "dataset": "asap", "tracked": True},
                {"piece_id": "batik/b/b", "dataset": "batik", "tracked": True},
                {"piece_id": "batik/c/c", "dataset": "batik", "tracked": True},
                {"piece_id": "vienna/d/d", "dataset": "vienna", "tracked": False},
            ],
            "failures": [],
        }
        record.update(over)
        return record

    def test_it_leads_with_the_tracking_verdict(self):
        text = entry.summarise(self.metrics())
        assert "3/4 pieces" in text and "75%" in text

    def test_it_reports_both_beat_medians(self):
        text = entry.summarise(self.metrics())
        assert "0.050" in text and "0.400" in text

    def test_it_breaks_down_by_dataset(self):
        text = entry.summarise(self.metrics())
        assert "batik" in text and "2/2 tracked" in text

    def test_it_names_the_pieces_that_were_lost(self):
        text = entry.summarise(self.metrics())
        assert "vienna/d/d" in text

    def test_it_reports_pieces_that_did_not_run(self):
        record = self.metrics(
            failures=[{"piece_id": "x/y/z", "error": "ValueError: boom"}]
        )
        text = entry.summarise(record)
        assert "did not run" in text and "x/y/z" in text

    def test_a_perfect_run_says_nothing_about_losses(self):
        record = self.metrics(
            n_tracked=4,
            pieces=[
                dict(p, tracked=True) for p in self.metrics()["pieces"]
            ],
        )
        assert "lost the performance" not in entry.summarise(record)

    def test_it_survives_a_run_with_no_tracked_pieces(self):
        record = self.metrics(
            n_tracked=0,
            summary_tracked={},
            pieces=[dict(p, tracked=False) for p in self.metrics()["pieces"]],
        )
        text = entry.summarise(record)
        assert "0/4 pieces" in text


class TestDocsPointHere:
    def test_submitting_names_the_command(self):
        text = (REPO_ROOT / "docs" / "submitting.md").read_text()
        assert "test_submission.py" in text, (
            "the submitter guide should name the command submitters run"
        )
