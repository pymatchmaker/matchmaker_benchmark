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
from matchmaker_eval.evaluation_key import file_sha256
from matchmaker_eval.load_folds import REPO_ROOT, RESULTS_DIR

SNAPSHOT = "benchmark-snapshot.zip"


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True).strip()


def api(path):
    return json.loads(gh("api", path))


def releases(repo):
    pages = json.loads(gh("api", "--paginate", "--slurp", f"repos/{repo}/releases"))
    return [release for page in pages for release in page]


def latest(records):
    candidates = [r for r in records if r["tag_name"].startswith("benchmark-")
                  and not r["draft"] and not r["prerelease"]
                  and any(a["name"] == SNAPSHOT for a in r["assets"])]
    return max(candidates, key=lambda r: r["published_at"], default=None)


def extract(archive, destination):
    """Reject paths outside the destination before extracting any member."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or member.filename.startswith("/"):
                raise ValueError(f"Unsafe archive path: {member.filename}")
        bundle.extractall(destination)


def fetch(repo, release, destination):
    if release["draft"] or release["prerelease"]:
        raise ValueError("Select a published benchmark release")
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        gh("release", "download", release["tag_name"], "--repo", repo,
           "--pattern", SNAPSHOT, "--pattern", "SHA256SUMS", "--dir", str(work))
        hashes = dict(line.split(maxsplit=1)[::-1]
                      for line in (work / "SHA256SUMS").read_text().splitlines())
        if file_sha256(work / SNAPSHOT) != hashes.get(SNAPSHOT):
            raise ValueError("Published snapshot checksum does not match")
        unpacked = work / "unpacked"
        extract(work / SNAPSHOT, unpacked)
        if not (unpacked / "leaderboard.json").is_file():
            raise ValueError("Published snapshot has no leaderboard")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(unpacked, destination)


def validate_runs(runs, repo):
    if not runs or any(r["status"] != "completed" or r["conclusion"] != "success"
                       or r["head_repository"]["full_name"] != repo for r in runs):
        raise ValueError("Only successful, completed runs from this repository can be released")


def stage_piano(source, destination, commit):
    """Later selected runs replace complete entries; other methods are retained."""
    metrics_paths = sorted(source.glob("*/metrics.json"))
    if not metrics_paths:
        raise ValueError("Piano artifact contains no merged entries")
    for path in metrics_paths:
        record = json.loads(path.read_text())
        entry = path.parent.name
        if (record.get("submission") != entry or record.get("fold") != "eval"
                or record.get("partial") is not False or record.get("n_failed") != 0
                or not record.get("fold_size")
                or record.get("n_pieces") != record["fold_size"]):
            raise ValueError(f"Incomplete or invalid piano entry: {entry}")
        if record.get("environment", {}).get("benchmark_commit") != commit:
            raise ValueError(f"Result does not identify its source run's commit: {entry}")
    for path in metrics_paths:
        target = destination / "submissions" / path.parent.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path.parent, target)
    return [path.parent.name for path in metrics_paths]


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


def publish(repo, tag, run_ids, records, destination):
    if not re.fullmatch(r"benchmark-[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("Use a new result tag such as benchmark-v1")
    if any(r["tag_name"] == tag for r in records):
        raise ValueError(f"Release {tag} already exists; published results are never overwritten")
    if not run_ids or any(not re.fullmatch(r"[0-9]+", run) for run in run_ids):
        raise ValueError("source_runs must contain numeric run IDs separated by spaces")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Select each source run only once")
    runs = [api(f"repos/{repo}/actions/runs/{run}") for run in run_ids]
    validate_runs(runs, repo)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=REPO_ROOT, check=True)
    if any(ref["ref"] == f"refs/tags/{tag}"
           for ref in api(f"repos/{repo}/git/matching-refs/tags/{tag}")):
        raise ValueError(f"Tag {tag} already exists; choose a new result version")

    base = latest(records)
    entry_sources = {}
    if base:
        fetch(repo, base, destination)
        entry_sources = json.loads((destination / "release.json").read_text()).get("entry_sources", {})
    else:
        destination.mkdir(parents=True, exist_ok=False)
        # With no earlier release, start from the archive tracked in the repository.
        old_archive = REPO_ROOT / "results" / "archive"
        if old_archive.exists():
            shutil.copytree(old_archive, destination / "archive")

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        assets = work / "assets"
        assets.mkdir()
        sources = []
        for run in runs:
            pages = json.loads(gh("api", "--paginate", "--slurp",
                                 f"repos/{repo}/actions/runs/{run['id']}/artifacts"))
            artifacts = [a for page in pages for a in page["artifacts"]
                         if a["name"].startswith("benchmark-results-")
                         or a["name"] == "ophmm-playthrough-comparison"]
            if not artifacts or any(a["expired"] for a in artifacts):
                raise ValueError(f"Run {run['id']} has no available complete result artifact")
            saved = []
            for artifact in artifacts:
                name = artifact["name"]
                if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                    raise ValueError(f"Unexpected artifact name: {name}")
                output = assets / f"run-{run['id']}-{name}.zip"
                with output.open("wb") as stream:
                    subprocess.run(["gh", "api", f"repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
                                   stdout=stream, check=True)
                if name == "benchmark-results-eval":
                    unpacked = work / str(artifact["id"])
                    extract(output, unpacked)
                    for entry in stage_piano(unpacked, destination, run["head_sha"]):
                        entry_sources[entry] = run["id"]
                saved.append({"id": artifact["id"], "asset": output.name,
                              "sha256": file_sha256(output)})
            sources.append({"id": run["id"], "url": run["html_url"],
                            "benchmark_commit": run["head_sha"], "branch": run["head_branch"],
                            "attempt": run["run_attempt"], "artifacts": saved})

        if not list((destination / "submissions").glob("*/metrics.json")):
            raise ValueError("A release must include a complete piano evaluation")
        env = {**os.environ, "MATCHMAKER_RESULTS_DIR": str(destination)}
        for script in ("export_details.py", "leaderboard.py"):
            subprocess.run([sys.executable, str(REPO_ROOT / "matchmaker_eval" / script)],
                           env=env, check=True)
        board = json.loads((destination / "leaderboard.json").read_text())
        if not board["entries"] or board["skipped"]:
            raise ValueError("The leaderboard is empty or contains invalid entries")
        manifest = {"tag": tag, "publisher_commit": commit, "source_runs": sources,
                    "entry_sources": entry_sources,
                    "base_release": base["html_url"] if base else None,
                    "evaluation_keys": board["evaluation_keys"]}
        (destination / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.run([sys.executable, str(REPO_ROOT / "matchmaker_eval" / "build_site.py"),
                        str(work / "site")], env=env, check=True)
        snapshot(destination, assets / SNAPSHOT)
        checksums = assets / "SHA256SUMS"
        checksums.write_text("".join(f"{file_sha256(p)}  {p.name}\n" for p in sorted(assets.iterdir())))
        notes = work / "release.md"
        notes.write_text(
            f"Benchmark results published from `{commit}`.\n\n"
            + "\n".join(f"- [Run {s['id']}]({s['url']}), evaluated at `{s['benchmark_commit']}`"
                        for s in sources)
            + "\n"
        )
        # Upload everything as a draft; expose it only after every upload succeeds.
        gh("release", "create", tag, *(str(p) for p in sorted(assets.iterdir())),
           "--repo", repo, "--target", commit, "--title", tag,
           "--notes-file", str(notes), "--draft", "--latest=false")
        gh("release", "edit", tag, "--repo", repo, "--draft=false", "--latest=false")
    return tag


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--source-runs", default="", help="space-separated completed CI run IDs")
    args = parser.parse_args()
    if not args.repo or not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error("Set GITHUB_REPOSITORY or --repo owner/repository")
    if RESULTS_DIR.resolve().is_relative_to(REPO_ROOT):
        parser.error("Set MATCHMAKER_RESULTS_DIR to a directory outside the source checkout")
    try:
        records = releases(args.repo)
        if args.source_runs:
            tag = publish(args.repo, args.tag, args.source_runs.split(), records, RESULTS_DIR)
        else:
            release = next((r for r in records if r["tag_name"] == args.tag), None) if args.tag else latest(records)
            if args.tag and release is None:
                raise ValueError(f"Release {args.tag} was not found")
            tag = release["tag_name"] if release else ""
            if release:
                fetch(args.repo, release, RESULTS_DIR)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
                stream.write(f"ready={'true' if tag else 'false'}\ntag={tag}\n")
        print(f"Results: {tag}" if tag else "No benchmark release yet; leave the existing website unchanged.")
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
