"""Check a submission before it is merged.

    python matchmaker_eval/validate_submission.py submissions/alice
    python matchmaker_eval/validate_submission.py --all
    python matchmaker_eval/validate_submission.py submissions/alice --smoke

Runs three passes:

1. **Structure** — the files a submission must have, and a well-formed
   ``metadata.yaml`` including the eval-fold declaration.
2. **Contract** — ``solution.py`` imports, defines exactly one ``Submission``,
   and implements ``build_follower``.
3. **Smoke run** (``--smoke``) — the submission follows the one piece committed
   under ``resources/``. This needs no dataset download, so CI can run it on
   every pull request. It checks that the follower *runs*, not that it is good.

Review flags are printed for anything a human should look at (network access,
subprocesses, ground-truth file access). They are a reading aid for the
maintainer doing the review, not a sandbox: this repository executes submitted
code, and merge is the trust boundary. See ``docs/eval-protocol.md``.
"""

# Entry-point path setup — see the note in run_submission.py.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matchmaker_eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import re

from matchmaker_eval.folds import REPO_ROOT
from matchmaker_eval.submission import (
    SubmissionError,
    load_solution,
    read_metadata,
)

SUBMISSIONS_DIR = REPO_ROOT / "submissions"
REQUIRED_FILES = ("solution.py", "metadata.yaml")
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,48}$")

#: Patterns worth a second pair of eyes. Each is (regex, why it matters).
REVIEW_PATTERNS = [
    (
        r"\bimport\s+(requests|urllib|httpx|socket|ftplib)\b|\bfrom\s+urllib\b",
        "network access — a submission must run offline and deterministically",
    ),
    (r"\bimport\s+subprocess\b|\bos\.system\b|\bos\.popen\b", "spawns processes"),
    (r"\.match\b", "touches .match files — that is the ground truth"),
    (
        r"data/folds/eval|folds\.load_fold|metadata-validation",
        "reads the benchmark's own fold definitions",
    ),
    (
        r"MATCHMAKER_DATA_DIR|~/data|os\.environ",
        "reaches for the dataset directory rather than using the streamed input",
    ),
    (r"\bexec\(|\beval\(|__import__\(|base64\.b64decode", "dynamic code execution"),
]

#: pip options that would make an install unreproducible or unreviewable.
REQUIREMENT_FLAGS = [
    ("git+", "installs straight from a git URL — pin a released version instead"),
    ("--index-url", "points pip at another package index"),
    ("--extra-index-url", "points pip at another package index"),
    ("-e ", "editable install"),
    ("http://", "plain-HTTP download"),
]


class Report:
    """Collects the outcome of one submission's checks."""

    def __init__(self, name: str):
        self.name = name
        self.errors, self.warnings, self.flags = [], [], []

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def flag(self, message):
        self.flags.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"{'PASS' if self.ok else 'FAIL'}  {self.name}"]
        for message in self.errors:
            lines.append(f"  error  {message}")
        for message in self.warnings:
            lines.append(f"  warn   {message}")
        for message in self.flags:
            lines.append(f"  review {message}")
        return "\n".join(lines)


def check_structure(directory: Path, report: Report) -> dict:
    if not NAME_PATTERN.match(directory.name):
        report.error(
            f"directory name '{directory.name}' must be lowercase letters, digits, "
            "'-', '_' or '.' (2-49 characters)"
        )
    for filename in REQUIRED_FILES:
        if not (directory / filename).exists():
            report.error(f"{filename} is missing")

    unexpected = [
        p.name
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix in {".wav", ".mp3", ".mid", ".midi", ".match", ".pt", ".ckpt"}
    ]
    if unexpected:
        report.warn(
            f"data/model files committed in the submission directory: {unexpected}. "
            "Small model weights are fine; audio, MIDI and .match files are not."
        )

    metadata = {}
    if (directory / "metadata.yaml").exists():
        try:
            metadata = read_metadata(directory)
        except SubmissionError as e:
            report.error(str(e))
    return metadata


def check_requirements(directory: Path, report: Report) -> None:
    path = directory / "requirements.txt"
    if not path.exists():
        return
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for needle, why in REQUIREMENT_FLAGS:
            if needle in line:
                report.flag(f"requirements.txt:{lineno} {why}: {line}")
        if not any(
            op in line for op in ("==", ">=", "~=", "<=")
        ) and not line.startswith("-"):
            report.warn(
                f"requirements.txt:{lineno} '{line}' is unpinned — pin a version so "
                "the run can be reproduced"
            )


def scan_for_review(directory: Path, report: Report) -> None:
    for path in sorted(directory.rglob("*.py")):
        try:
            source = path.read_text()
        except UnicodeDecodeError:
            report.warn(f"{path.name} is not valid UTF-8")
            continue
        for lineno, line in enumerate(source.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern, why in REVIEW_PATTERNS:
                if re.search(pattern, line):
                    rel = path.relative_to(directory)
                    report.flag(f"{rel}:{lineno} {why}: {line.strip()[:70]}")


def check_contract(directory: Path, metadata: dict, report: Report):
    if not (directory / "solution.py").exists():
        return None
    try:
        method, input_type = load_solution(directory / "solution.py")
    except SubmissionError as e:
        report.error(str(e))
        return None
    if metadata and input_type != metadata.get("input_type"):
        report.error(
            f"metadata.yaml says input_type '{metadata.get('input_type')}' but "
            f"solution.py registered '{input_type}'"
        )
    if method != directory.name:
        report.warn(
            f"registered method name '{method}' differs from the directory name "
            f"'{directory.name}' — keeping them equal makes results easier to trace"
        )
    return method


def run_smoke(directory: Path, report: Report) -> None:
    from matchmaker_eval.run_submission import evaluate_submission

    try:
        metrics = evaluate_submission(
            directory,
            fold="example",
            run_dir=REPO_ROOT / "results" / "smoke" / directory.name,
            save_plots=False,
            piece_timeout=600,
        )
    except Exception as e:
        report.error(f"smoke run raised {type(e).__name__}: {e}")
        return
    if metrics["n_failed"]:
        report.error(
            f"smoke run could not complete the example piece: "
            f"{metrics['failures'][0]['error']}"
        )
        return
    beat = metrics["summary_all"].get("beat", {})
    report.warn(
        "smoke run completed (median beat error "
        f"{beat.get('median', float('nan')):.2f}b) — this only proves the "
        "follower runs"
    )


def display_name(directory: Path) -> str:
    """Repo-relative path when possible, so reports read the same in CI."""
    try:
        return str(directory.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(directory)


def validate(directory: Path, smoke: bool = False) -> Report:
    directory = Path(directory)
    report = Report(display_name(directory))
    if not directory.is_dir():
        report.error("not a directory")
        return report

    metadata = check_structure(directory, report)
    check_requirements(directory, report)
    scan_for_review(directory, report)
    if report.ok:
        check_contract(directory, metadata, report)
    if report.ok and smoke:
        run_smoke(directory, report)
    return report


def discover() -> list:
    return sorted(
        p
        for p in SUBMISSIONS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "submissions", type=Path, nargs="*", help="submission directories"
    )
    parser.add_argument("--all", action="store_true", help="validate every submission")
    parser.add_argument(
        "--smoke", action="store_true", help="also run the example-fold smoke test"
    )
    args = parser.parse_args()

    targets = discover() if args.all else args.submissions
    if not targets:
        parser.error("pass at least one submission directory, or --all")

    reports = [validate(directory, smoke=args.smoke) for directory in targets]
    print("\n".join(report.render() for report in reports))

    failed = [r for r in reports if not r.ok]
    flagged = [r for r in reports if r.flags]
    print(f"\n{len(reports) - len(failed)}/{len(reports)} passed")
    if flagged:
        print(
            f"{len(flagged)} submission(s) raised review flags — a maintainer must "
            "read the code before merging."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
