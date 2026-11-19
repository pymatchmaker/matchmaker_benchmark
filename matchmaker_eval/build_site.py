"""Assemble the leaderboard site into a directory.

    python matchmaker_eval/build_site.py _site      # what Pages publishes
    python matchmaker_eval/build_site.py --serve    # ...and open it locally
    python matchmaker_eval/build_site.py _site --versions DIR   # plus versions/<tag>/

The site is ``docs/`` plus what the page reads and links to: the leaderboard
JSON and CSV, the same pair per dataset, and the per-entry detail files. Both
the publish workflow and the staging preview call this, so what you look at
before merging is assembled by the same code as what goes live.

With ``--versions``, every published release is also assembled under
``versions/<tag>/`` (see docs/maintaining.md, "Older result versions").

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
import json
import shutil

from matchmaker_eval.folds import REPO_ROOT, RESULTS_DIR

DOCS_DIR = REPO_ROOT / "docs"

#: Releases that keep their per-piece views under versions/: the newest few,
#: plus any pinned ones, such as the release our paper cites.
NEWEST_FULL_VERSIONS = 3
PINNED_FULL_VERSIONS = ("v1.0.0",)


class SiteError(Exception):
    """Raised when the site cannot be assembled."""


def build(destination: Path, results_dir: Path = None, details: bool = True) -> dict:
    """Copy docs/ and the result files into ``destination``. Returns a summary.

    ``details=False`` leaves out the per-piece views, which are almost all of
    a version's size; the tables and archived keys are always included.
    """
    destination = Path(destination)
    results_dir = Path(results_dir or RESULTS_DIR)
    if destination.exists():
        shutil.rmtree(destination)
    if not DOCS_DIR.is_dir():
        raise SiteError(f"{DOCS_DIR} is missing.")
    shutil.copytree(DOCS_DIR, destination)

    summary = {"boards": [], "details": 0, "archives": 0, "missing": []}
    for name in ("leaderboard.json", "leaderboard.csv", "release.json"):
        source = results_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
        else:
            summary["missing"].append(name)
    # leaderboard-<dataset>.json/.csv: the page links to them for download.
    for source in sorted(results_dir.glob("leaderboard-*.*")):
        if source.suffix in (".json", ".csv"):
            shutil.copy2(source, destination / source.name)
            summary["boards"].append(source.name)

    details_dir = results_dir / "details"
    if details and details_dir.is_dir():
        target = destination / "details"
        target.mkdir(exist_ok=True)
        for path in sorted(details_dir.glob("*.json")):
            shutil.copy2(path, target / path.name)
            summary["details"] += 1

    # Every leaderboard ever published under an evaluation key, so a key cited
    # in a paper resolves to archive/<key>/leaderboard.json on the site.
    archive = results_dir / "archive"
    if archive.is_dir():
        shutil.copytree(archive, destination / "archive")
        summary["archives"] = sum(1 for path in archive.iterdir() if path.is_dir())

    if "leaderboard.json" in summary["missing"]:
        raise SiteError(
            "results/leaderboard.json is missing — run "
            "matchmaker_eval/leaderboard.py first."
        )
    return summary


def add_versions(destination: Path, versions_dir: Path) -> list:
    """Assemble each release in ``versions_dir/versions.json`` under ``versions/<tag>/``.

    The newest ``NEWEST_FULL_VERSIONS`` and the ``PINNED_FULL_VERSIONS`` keep
    their per-piece views; the others keep their tables only. The list is also written to ``versions.json``.
    """
    destination, versions_dir = Path(destination), Path(versions_dir)
    listed = json.loads((versions_dir / "versions.json").read_text())["versions"]
    for index, version in enumerate(listed):
        version["full"] = index < NEWEST_FULL_VERSIONS or version["tag"] in PINNED_FULL_VERSIONS
        build(destination / "versions" / version["tag"], versions_dir / version["tag"],
              details=version["full"])
    (destination / "versions.json").write_text(json.dumps({"versions": listed}, indent=2) + "\n")
    return listed


def serve(directory: Path, port: int) -> None:
    import functools
    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        # Without this a browser may keep showing the page from the previous
        # build: with no cache headers it treats a file as fresh for a share
        # of its age, and index.html is often days old.
        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    handler = functools.partial(Handler, directory=str(directory))
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
    parser.add_argument(
        "--versions",
        type=Path,
        default=None,
        help="folder of downloaded releases (release_results.py --versions-dir); "
             "each is added under versions/<tag>/",
    )
    args = parser.parse_args()

    try:
        summary = build(Path(args.destination))
        versions = add_versions(Path(args.destination), args.versions) if args.versions else []
    except SiteError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

    print(f"assembled {args.destination} from docs/ + results/")
    print(
        f"  leaderboard.json, leaderboard.csv, {len(summary['boards'])} "
        f"per-dataset file(s), {summary['details']} detail file(s), "
        f"{summary['archives']} archived key(s)"
    )
    if args.versions:
        print(f"  {len(versions)} published version(s) under versions/: "
              + (", ".join(v["tag"] + ("" if v["full"] else " (tables only)") for v in versions)
                 or "none"))
    for name in summary["missing"]:
        print(f"  note: results/{name} was not there")
    if args.serve:
        serve(Path(args.destination), args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
