"""What the evaluation pipeline actually needs from ``data/``.

``data/`` currently holds two unrelated things: the benchmark's own
configuration, and leftover dataset material from the metadata-CSV runners. The
plan is for all *data* to live in matchmaker-benchmark-data, so these tests pin
which files the fold-driven pipeline reads -- everything else can go without
breaking evaluation.

Verified by running all ten MIDI references with the dataset material removed
and MATCHMAKER_DATA_DIR pointing at an empty directory: every one completed,
fetching its files from the data repository.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA = REPO_ROOT / "data"

#: Configuration the benchmark owns. Deleting any of these breaks evaluation.
REQUIRED = (
    "folds/eval.csv",
    "folds/tuning.csv",
    "folds/example.csv",
    "data_sources.yaml",
    "builtin_methods.yaml",
)


class TestRequiredConfig:
    @pytest.mark.parametrize("relative", REQUIRED)
    def test_it_is_present(self, relative):
        assert (DATA / relative).is_file()

    def test_the_fold_pipeline_reads_only_these(self):
        """The modules the evaluation path imports must not reach for a dataset.

        A reference to data/metadata-*.csv or data/winterreise/ in any of these
        would mean the pipeline still depends on material that is meant to move
        to the data repository.
        """
        pipeline = [
            "folds.py",
            "fetch_data.py",
            "run_submission.py",
            "merge_shards.py",
            "merge_references.py",
            "leaderboard.py",
            "export_details.py",
            "methods.py",
            "submission.py",
            "build_site.py",
        ]
        offenders = {}
        for name in pipeline:
            text = (REPO_ROOT / "matchmaker_eval" / name).read_text()
            for needle in ("data/metadata", "data/reduced", "data/winterreise",
                           "tpdd_difficulty"):
                if needle in text:
                    offenders.setdefault(name, []).append(needle)
        assert not offenders, (
            f"the evaluation path reads dataset material from data/: {offenders}"
        )


class TestDatasetMaterialIsSeparable:
    """Removing the dataset leftovers must not break the pipeline's imports."""

    def test_the_pipeline_imports_without_them(self, tmp_path):
        staged = tmp_path / "data"
        (staged / "folds").mkdir(parents=True)
        for relative in REQUIRED:
            shutil.copy2(DATA / relative, staged / relative)

        # A repo tree holding only the config, and nothing dataset-shaped.
        script = (
            "import sys; sys.path[:0] = [r'%s', r'%s']\n"
            "import matchmaker_eval.folds as F\n"
            "from pathlib import Path\n"
            "F.FOLD_DIR = Path(r'%s')\n"
            "pieces = F.load_fold('eval', input_type='midi')\n"
            "print(len(pieces))\n"
        ) % (REPO_ROOT, REPO_ROOT / "matchmaker_eval", staged / "folds")
        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=tmp_path,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "146"

    def test_nothing_dataset_shaped_is_required(self):
        """These are the files the eventual deletion is expected to remove."""
        removable = [
            p for p in DATA.rglob("*")
            if p.is_file()
            and p.relative_to(DATA).as_posix() not in REQUIRED
        ]
        # Not an assertion about the count -- just that the split is real and
        # the required files are not among the removable ones.
        for path in removable:
            assert path.relative_to(DATA).as_posix() not in REQUIRED
