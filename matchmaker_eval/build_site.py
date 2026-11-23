"""Assemble the leaderboard site into a directory.

    python matchmaker_eval/build_site.py _site      # what Pages publishes
    python matchmaker_eval/build_site.py --serve    # ...and open it locally
    python matchmaker_eval/build_site.py _site --versions DIR   # plus versions/<tag>/

The site is ``docs/`` plus what the page reads and links to: the leaderboard
JSON and CSV, the same pair per dataset, the per-entry detail files, and
``per-piece.zip``, the detail files' per-piece metrics as TSV. Both the publish
workflow and the staging preview call this, so what you look at
before merging is assembled by the same code as what goes live.

With ``--versions``, each downloaded release is also assembled under
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
import csv
import io
import json
import shutil
import zipfile
from collections import Counter

from matchmaker_eval.folds import REPO_ROOT, RESULTS_DIR

DOCS_DIR = REPO_ROOT / "docs"

#: Columns that lead every row of the per-piece TSVs, naming the entry.
ENTRY_COLUMNS = ("submission", "method", "input_type")
#: Per-piece fields left out of the TSVs: the paths the page plots.
PATH_FIELDS = ("wp", "gt")

class SiteError(Exception):
    """Raised when the site cannot be assembled."""


def build(destination: Path, results_dir: Path = None) -> dict:
    """Copy docs/ and the result files into ``destination``. Returns a summary."""
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
    if details_dir.is_dir():
        target = destination / "details"
        target.mkdir(exist_ok=True)
        for path in sorted(details_dir.glob("*.json")):
            shutil.copy2(path, target / path.name)
            summary["details"] += 1
        release = results_dir / "release.json"
        tag = json.loads(release.read_text()).get("tag") if release.exists() else None
        summary["pieces_zip"] = write_pieces_zip(details_dir, destination / "per-piece.zip",
                                                 folder=pieces_folder_name(tag))

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


def pieces_folder_name(tag: str = None) -> str:
    """Name of the per-piece zip's top folder, and of the file the page saves it as."""
    return f"matchmaker-benchmark-{tag}-per-piece-results" if tag else "matchmaker-benchmark-per-piece-results"


def write_pieces_zip(details_dir: Path, destination: Path, folder: str = None) -> int:
    """Write every detail file's per-piece metrics to a zip of TSVs.

    Everything sits under one top ``folder`` (default ``pieces_folder_name()``),
    so unpacking gives a folder named after the release. Each dataset has a folder with one ``<dataset>-<input type>-<method>.tsv``
    per entry, and ``all-entries.tsv`` at the top holds every row, one per
    entry and piece, for comparing methods piece by piece. Returns the number
    of entries written.
    """
    entries = []
    for path in sorted(Path(details_dir).glob("*.json")):
        detail = json.loads(path.read_text())
        if detail.get("pieces"):
            rows = [
                {**{column: detail.get(column) for column in ENTRY_COLUMNS},
                 **{key: value for key, value in piece.items() if key not in PATH_FIELDS}}
                for piece in detail["pieces"]
            ]
            entries.append((detail, rows))

    columns = list(ENTRY_COLUMNS)
    for _, rows in entries:
        for row in rows:
            columns += [key for key in row if key not in columns]

    # Entries sharing a method and input type, e.g. a submission named after a
    # reference method, are told apart by their submission names instead.
    methods = Counter((detail.get("input_type"), detail.get("method")) for detail, _ in entries)
    files = {}
    for detail, rows in entries:
        input_type, name = detail.get("input_type"), detail.get("method")
        if methods[input_type, name] > 1:
            name = detail.get("submission")
        for dataset in sorted({row["dataset"] for row in rows}):
            files[f"{dataset}/{dataset}-{input_type}-{name}.tsv"] = [
                row for row in rows if row["dataset"] == dataset]
    files["all-entries.tsv"] = [row for _, rows in entries for row in rows]

    folder = folder or pieces_folder_name()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.writestr(f"{folder}/{path}", format_tsv(files[path], columns))
    return len(entries)


def format_tsv(rows: list, columns: list) -> str:
    """Rows of dicts as TSV text; a missing value is an empty cell."""
    text = io.StringIO()
    writer = csv.DictWriter(text, columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue()


def add_versions(destination: Path, versions_dir: Path) -> list:
    """Assemble each release in ``versions_dir/versions.json`` under ``versions/<tag>/``.

    The list is also written to ``versions.json``.
    """
    destination, versions_dir = Path(destination), Path(versions_dir)
    listed = json.loads((versions_dir / "versions.json").read_text())["versions"]
    for version in listed:
        build(destination / "versions" / version["tag"], versions_dir / version["tag"])
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
    if "pieces_zip" in summary:
        print(f"  per-piece.zip with {summary['pieces_zip']} entr(y/ies)")
    if args.versions:
        print(f"  {len(versions)} published version(s) under versions/: "
              + (", ".join(v["tag"] for v in versions) or "none"))
    for name in summary["missing"]:
        print(f"  note: results/{name} was not there")
    if args.serve:
        serve(Path(args.destination), args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
