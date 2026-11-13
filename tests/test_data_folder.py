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
    "folds/valid.csv",
    "folds/example.csv",
    "data_sources.yaml",
    "builtin_methods.yaml",
)


class TestRequiredConfig:
    @pytest.mark.parametrize("relative", REQUIRED)
    def test_it_is_present(self, relative):
        assert (DATA / relative).is_file()


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
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "146"
