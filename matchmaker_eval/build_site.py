"""Assemble the leaderboard site into a directory.

    python matchmaker_eval/build_site.py _site      # what Pages publishes
    python matchmaker_eval/build_site.py --serve    # ...and open it locally
    python matchmaker_eval/build_site.py _site --versions DIR   # plus v/<tag>/

The site is ``docs/`` plus what the page reads and links to: the leaderboard
JSON and CSV, the same pair per dataset, and the per-entry detail files. Both
the publish workflow and the staging preview call this, so what you look at
before merging is assembled by the same code as what goes live.

With ``--versions``, every published result release found in DIR (as written
by ``release_results.py --versions-dir``) is also assembled under
``v/<tag>/``, and ``versions.json`` lists them for the page's version menu.
``data/full_versions.yaml`` decides which versions keep their per-piece views
(the newest few and any cited version); the others show their tables only.
Each version uses the current page with that release's data, so a link such
as ``v/benchmark-v1/`` keeps showing the numbers a paper cites.

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
import re
import shutil

from matchmaker_eval.load_folds import REPO_ROOT, RESULTS_DIR

DOCS_DIR = REPO_ROOT / "docs"
FULL_VERSIONS = REPO_ROOT / "data" / "full_versions.yaml"


class SiteError(Exception):
    """Raised when the site cannot be assembled."""


#: A published result tag, as release_results.py creates them.
VERSION_TAG = re.compile(r"benchmark-[A-Za-z0-9][A-Za-z0-9._-]*")


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


def full_version_rule(path: Path = None) -> dict:
    """Read data/full_versions.yaml: how many newest versions, and which tags,
    keep their per-piece views on the site."""
    import yaml

    path = Path(path or FULL_VERSIONS)
    config = (yaml.safe_load(path.read_text()) if path.is_file() else None) or {}
    newest = config.get("newest", 3)
    keep = config.get("keep") or []
    if not isinstance(newest, int) or newest < 1:
        raise SiteError(f"{path}: 'newest' must be a positive whole number")
    if not isinstance(keep, list) or not all(isinstance(t, str) for t in keep):
        raise SiteError(f"{path}: 'keep' must be a list of release tags")
    return {"newest": newest, "keep": keep}


def add_versions(destination: Path, versions_dir: Path, rule: dict = None) -> list:
    """Assemble every downloaded release under ``destination/v/<tag>/``.

    ``versions_dir/versions.json`` (newest first) decides what is listed; a
    release whose snapshot is missing is left out rather than linked broken.
    The newest versions and those kept by name (``full_version_rule``) are
    assembled in full, the rest without per-piece views. Returns the listed
    versions, also written to ``destination/versions.json``.
    """
    destination, versions_dir = Path(destination), Path(versions_dir)
    rule = rule or full_version_rule()
    index = versions_dir / "versions.json"
    if not index.is_file():
        raise SiteError(f"{index} is missing — run release_results.py --versions-dir first.")
    listed = []
    for version in json.loads(index.read_text()).get("versions", []):
        tag = version.get("tag", "")
        if not VERSION_TAG.fullmatch(tag) or not (versions_dir / tag / "leaderboard.json").is_file():
            continue
        full = len(listed) < rule["newest"] or tag in rule["keep"]
        summary = build(destination / "v" / tag, versions_dir / tag, details=full)
        if "release.json" in summary["missing"]:
            raise SiteError(f"release {tag} has no release.json")
        listed.append({**{key: version.get(key) for key in ("tag", "published_at", "url", "evaluation_keys")},
                       "full": full})
    missing = [tag for tag in rule["keep"] if tag not in {v["tag"] for v in listed}]
    for tag in missing:
        # Expected until that version is published (e.g. the paper's release).
        print(f"::warning::{tag} is kept in full by data/full_versions.yaml but is not published yet")
    (destination / "versions.json").write_text(
        json.dumps({"versions": listed}, indent=2) + "\n")
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
             "each is added under v/<tag>/",
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
        print(f"  {len(versions)} published version(s) under v/: "
              + (", ".join(v["tag"] + ("" if v["full"] else " (tables only)") for v in versions)
                 or "none"))
    for name in summary["missing"]:
        print(f"  note: results/{name} was not there")
    if args.serve:
        serve(Path(args.destination), args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
