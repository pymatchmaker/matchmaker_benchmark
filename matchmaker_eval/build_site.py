"""Assemble the leaderboard site into a directory.

    python matchmaker_eval/build_site.py _site      # what Pages publishes
    python matchmaker_eval/build_site.py --serve    # ...and open it locally

The site is ``docs/`` plus the three things the page reads: the leaderboard
JSON, the CSV, and the per-entry detail files. Both the publish workflow and the
staging preview call this, so what you look at before merging is assembled by
the same code as what goes live.

The page also reads ``../results/`` when it is served from a checkout, so
``--serve`` from the repository root works without assembling anything; the
assembled form is what Pages uploads.
"""

# Entry-point path setup — see the note in run_submission.py.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matchmaker_eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import shutil

from matchmaker_eval.folds import REPO_ROOT

DOCS_DIR = REPO_ROOT / "docs"
RESULTS_DIR = REPO_ROOT / "results"


class SiteError(Exception):
    """Raised when the site cannot be assembled."""


def build(destination: Path) -> dict:
    """Copy docs/ and the result files into ``destination``. Returns a summary."""
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    if not DOCS_DIR.is_dir():
        raise SiteError(f"{DOCS_DIR} is missing.")
    shutil.copytree(DOCS_DIR, destination)

    summary = {"details": 0, "missing": []}
    for name in ("leaderboard.json", "leaderboard.csv"):
        source = RESULTS_DIR / name
        if source.exists():
            shutil.copy2(source, destination / name)
        else:
            summary["missing"].append(name)

    details = RESULTS_DIR / "details"
    if details.is_dir():
        target = destination / "details"
        target.mkdir(exist_ok=True)
        for path in sorted(details.glob("*.json")):
            shutil.copy2(path, target / path.name)
            summary["details"] += 1

    if "leaderboard.json" in summary["missing"]:
        raise SiteError(
            "results/leaderboard.json is missing — run "
            "matchmaker_eval/leaderboard.py first."
        )
    return summary


def serve(directory: Path, port: int) -> None:
    import functools
    import http.server
    import socketserver

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"\nserving {directory} at http://localhost:{port}/  (ctrl-c to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "destination",
        nargs="?",
        default="_site",
        help="where to assemble the site (default: _site)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="serve the assembled site over HTTP and keep running",
    )
    parser.add_argument("--port", type=int, default=8000, help="port for --serve")
    args = parser.parse_args()

    try:
        summary = build(Path(args.destination))
    except SiteError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

    print(f"assembled {args.destination} from docs/ + results/")
    print(f"  leaderboard.json, leaderboard.csv, {summary['details']} detail file(s)")
    for name in summary["missing"]:
        print(f"  note: results/{name} was not there")
    if args.serve:
        serve(Path(args.destination), args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
