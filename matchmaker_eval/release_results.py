"""Publish completed CI runs as a release, or fetch a published snapshot.

Called by pages.yml with the GitHub CLI and GH_TOKEN; it never runs a follower.
Results are written to MATCHMAKER_RESULTS_DIR, outside the source checkout.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from matchmaker_eval.folds import REPO_ROOT, RESULTS_DIR

SNAPSHOT = "benchmark-snapshot.zip"

#: Result release tags, vMAJOR.MINOR.PATCH (see docs/maintaining.md).
TAG = re.compile(r"v\d+\.\d+\.\d+")

#: How many of the newest releases the site serves under versions/<tag>/.
SITE_VERSIONS = 5

PINNED_VERSIONS = ("v0.1.0",)


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True).strip()


def api(path):
    return json.loads(gh("api", path))


def releases(repo):
    pages = json.loads(gh("api", "--paginate", "--slurp", f"repos/{repo}/releases"))
    return [release for page in pages for release in page]


def published(records):
    """Published result releases with a website snapshot, newest first."""
    candidates = [r for r in records if TAG.fullmatch(r["tag_name"])
                  and not r["draft"] and not r["prerelease"]
                  and any(a["name"] == SNAPSHOT for a in r["assets"])]
    return sorted(candidates, key=lambda r: r["published_at"], reverse=True)


def latest(records):
    return next(iter(published(records)), None)


def extract(archive, destination):
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)


def fetch(repo, release, destination):
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        gh("release", "download", release["tag_name"], "--repo", repo,
           "--pattern", SNAPSHOT, "--dir", str(work))
        if destination.exists():
            shutil.rmtree(destination)
        extract(work / SNAPSHOT, destination)


def stage_piano(source, destination):
    """Later selected runs replace their entries; other methods are retained."""
    entries = []
    for path in sorted(source.glob("*/metrics.json")):
        target = destination / "submissions" / path.parent.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path.parent, target)
        entries.append(path.parent.name)
    return entries


def snapshot(results, output):
    """Store website data and metrics; full paths stay in the original run ZIPs."""
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(results.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(results)
            if relative.parts[0] == "submissions" and path.name != "metrics.json":
                continue
            bundle.write(path, str(relative))


def changed_keys(previous, current):
    """Input types whose evaluation key in ``previous`` differs in ``current``."""
    return sorted(t for t, key in previous.items() if current.get(t) != key)


def publish(repo, tag, run_ids, records, destination):
    base = latest(records)
    adding = not tag
    if tag and not TAG.fullmatch(tag):
        raise ValueError(f"Release tag must look like v1.0.0, not {tag!r}")
    if adding and not base:
        raise ValueError("The first release needs a tag such as v0.1.0")
    runs = [api(f"repos/{repo}/actions/runs/{run}") for run in run_ids]
    if any(run["conclusion"] != "success" for run in runs):
        raise ValueError("Only successful runs can be published")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()

    previous = {}
    if base:
        fetch(repo, base, destination)
        previous = json.loads((destination / "release.json").read_text())
    else:
        destination.mkdir(parents=True)
        old_archive = REPO_ROOT / "results" / "archive"
        if old_archive.exists():
            shutil.copytree(old_archive, destination / "archive")
    entry_sources = previous.get("entry_sources", {})

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        assets = work / "assets"
        assets.mkdir()
        sources = []
        for run in runs:
            pages = json.loads(gh("api", "--paginate", "--slurp",
                                 f"repos/{repo}/actions/runs/{run['id']}/artifacts"))
            artifacts = [a for page in pages for a in page["artifacts"]
                         if a["name"].startswith("benchmark-results-")]
            for artifact in artifacts:
                output = assets / f"run-{run['id']}-{artifact['name']}.zip"
                with output.open("wb") as stream:
                    subprocess.run(["gh", "api", f"repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
                                   stdout=stream, check=True)
                if artifact["name"] == "benchmark-results-eval":
                    unpacked = work / str(artifact["id"])
                    extract(output, unpacked)
                    for entry in stage_piano(unpacked, destination):
                        entry_sources[entry] = run["id"]
            sources.append({"id": run["id"], "url": run["html_url"],
                            "benchmark_commit": run["head_sha"]})

        env = {**os.environ, "MATCHMAKER_RESULTS_DIR": str(destination)}
        for script in ("export_details.py", "leaderboard.py"):
            subprocess.run([sys.executable, str(REPO_ROOT / "matchmaker_eval" / script)],
                           env=env, check=True)
        board = json.loads((destination / "leaderboard.json").read_text())
        if adding:
            changed = changed_keys(previous.get("evaluation_keys", {}), board["evaluation_keys"])
            if changed:
                raise ValueError(f"These runs change the {', '.join(changed)} evaluation key of "
                                 f"{base['tag_name']}; publish them as a new release")
            tag = base["tag_name"]
            sources = previous["source_runs"] + sources
        manifest = {"tag": tag, "publisher_commit": commit, "source_runs": sources,
                    "entry_sources": entry_sources,
                    "evaluation_keys": board["evaluation_keys"]}
        (destination / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
        snapshot(destination, assets / SNAPSHOT)
        notes = work / "release.md"
        notes.write_text(
            f"Benchmark results published from `{commit}`.\n\n"
            + "\n".join(f"- [Run {s['id']}]({s['url']}), evaluated at `{s['benchmark_commit']}`"
                        for s in sources)
            + "\n"
        )
        files = [str(p) for p in sorted(assets.iterdir())]
        if adding:
            gh("release", "upload", tag, *files, "--repo", repo, "--clobber")
            gh("release", "edit", tag, "--repo", repo, "--notes-file", str(notes))
        else:
            gh("release", "create", tag, *files,
               "--repo", repo, "--target", commit, "--title", tag, "--notes-file", str(notes))
    return tag


def fetch_versions(repo, records, destination):
    """Download the newest ``SITE_VERSIONS`` snapshots to ``destination/<tag>/`` for the site.

    ``versions.json`` lists them newest first, for the page's version menu.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    versions = []
    kept = [release for rank, release in enumerate(published(records))
            if rank < SITE_VERSIONS or release["tag_name"] in PINNED_VERSIONS]
    for release in kept:
        tag = release["tag_name"]
        fetch(repo, release, destination / tag)
        manifest = json.loads((destination / tag / "release.json").read_text())
        versions.append({"tag": tag, "published_at": release["published_at"],
                         "url": release["html_url"],
                         "evaluation_keys": manifest.get("evaluation_keys", {})})
    (destination / "versions.json").write_text(json.dumps({"versions": versions}, indent=2) + "\n")
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--source-runs", default="", help="space-separated completed CI run IDs")
    parser.add_argument("--versions-dir", type=Path, default=None,
                        help="also download the newest releases here, for the site's version menu")
    args = parser.parse_args()

    records = releases(args.repo)
    if args.source_runs:
        tag = publish(args.repo, args.tag, args.source_runs.split(), records, RESULTS_DIR)
    else:
        # Without new runs, redeploy the latest release; older ones stay at versions/<tag>/.
        if args.tag:
            parser.error("--tag needs --source-runs; an older release is served at versions/<tag>/")
        release = latest(records)
        tag = release["tag_name"] if release else ""
        if release:
            fetch(args.repo, release, RESULTS_DIR)
    if tag and args.versions_dir:
        # Listed again: a release created just now is not in `records`.
        versions = fetch_versions(args.repo, releases(args.repo), args.versions_dir)
        print(f"Versions for the site: {', '.join(v['tag'] for v in versions)}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"ready={'true' if tag else 'false'}\ntag={tag}\n")
    print(f"Results: {tag}" if tag else "No benchmark release yet; leave the existing website unchanged.")


if __name__ == "__main__":
    main()
