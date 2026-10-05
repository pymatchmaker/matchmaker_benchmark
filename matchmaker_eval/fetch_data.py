"""Fetch exactly the data a fold needs, and nothing else.

    python matchmaker_eval/fetch_data.py --fold eval --input-type midi
    python matchmaker_eval/fetch_data.py --fold eval --input-type audio \
        --shard 0 --num-shards 8
    python matchmaker_eval/fetch_data.py --fold valid --input-type audio --dry-run

The point of this module is that nobody has to download the whole corpus. A
contributor testing a submission needs no data at all (the ``example`` fold is
committed under ``resources/``). The evaluation workflow fetches only the pieces
in its shard, caches them, and reuses them on the next run.

Files already present under ``MATCHMAKER_DATA_DIR`` are left alone, so a local
copy of the original datasets is used as-is.

Where the files come from is configured in ``data/data_sources.yaml``.
"""

# Entry-point path setup — see the note in run_submission.py.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matchmaker_eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from matchmaker_eval.folds import DATA_ROOT, REPO_ROOT, load_fold, shard_of

CONFIG_PATH = REPO_ROOT / "data" / "data_sources.yaml"
CHECKSUM_DIR = REPO_ROOT / "data" / "checksums"
RETRIES = 3
TIMEOUT = 120

#: How GitHub serves a file, depending on how it was committed. A Git LFS file
#: fetched from raw.githubusercontent.com comes back as the pointer text
#: ("version https://git-lfs.github.com/spec/v1 ...") rather than the bytes, so
#: the two hosts are not interchangeable.
GITHUB_URLS = {
    "plain": "https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{location}",
    # The REST contents endpoint honours a personal access token where
    # raw.githubusercontent.com sometimes does not, so it is the dependable
    # route into a private repository. Serves files up to 100 MB.
    "api": (
        "https://api.github.com/repos/{owner}/{repo}/contents/"
        "{location}?ref={branch}"
    ),
    "lfs": (
        "https://media.githubusercontent.com/media/"
        "{owner}/{repo}/{branch}/{location}"
    ),
}
STORAGE_ORDER = {
    "plain": ["plain"],
    "api": ["api"],
    "lfs": ["lfs"],
    "auto": ["plain", "api", "lfs"],
}

#: The contents endpoint returns JSON metadata unless asked for the bytes.
API_ACCEPT = "application/vnd.github.raw"

#: A Git LFS pointer is a small text file that starts with this line. If one
#: arrives where audio was expected, the wrong host was used.
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


class FetchError(Exception):
    """Raised when data cannot be fetched or fails verification."""


@dataclass
class Wanted:
    """One file the benchmark needs, and where it would come from."""

    dataset: str
    path: str  # relative to the dataset root
    local: Path
    alternatives: List[Path]  # other containers that would also satisfy this

    @property
    def satisfied(self) -> bool:
        return self.local.exists() or any(p.exists() for p in self.alternatives)


#: Environment overrides, so a personal fork can be tested without editing the
#: file everyone else shares.
ENV_OVERRIDES = {
    "owner": "MATCHMAKER_DATA_OWNER",
    "repo": "MATCHMAKER_DATA_REPO",
    "branch": "MATCHMAKER_DATA_BRANCH",
    "layout": "MATCHMAKER_DATA_LAYOUT",
    "storage": "MATCHMAKER_DATA_STORAGE",
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FetchError(f"{CONFIG_PATH} is missing.")
    config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    for key, variable in ENV_OVERRIDES.items():
        value = os.environ.get(variable)
        if value:
            config[key] = value
            # An override of the address implies fetching is wanted, even if the
            # committed config still has it switched off.
            config["enabled"] = True
    return config


#: Prefixes GitHub uses for access tokens. If one of these turns up where an
#: environment variable *name* belongs, someone has pasted the secret itself.
TOKEN_PREFIXES = ("github_pat_", "ghp_", "gho_", "ghs_", "ghu_", "ghr_")


def auth_token(config: dict) -> Optional[str]:
    """Read the access token for a private data repository, if one is set.

    ``token_env`` names an environment variable; it must never hold the token,
    which would then be committed. Pasting the secret there is an easy misread
    of the field name, and it fails silently — the lookup just misses — so it
    is caught explicitly here.
    """
    name = config.get("token_env") or "MATCHMAKER_DATA_TOKEN"
    if name.startswith(TOKEN_PREFIXES) or len(name) > 60:
        raise FetchError(
            "data/data_sources.yaml: `token_env` must hold the NAME of an "
            "environment variable, not the token itself.\n\n"
            "  token_env: MATCHMAKER_DATA_TOKEN        <- this file\n"
            "  export MATCHMAKER_DATA_TOKEN=github_pat_...   <- your shell\n\n"
            "The value currently there looks like a real token. Revoke it: it "
            "has been written to a file that gets committed."
        )
    return os.environ.get(name) or None


def branch_for(config: dict, fold: Optional[str] = None) -> str:
    """Which branch of the data repository this fold's files live on.

    Folds can be kept on separate branches — validation material on one, evaluation
    material on another — so that a clone or a checkout brings down only what
    that fold needs. ``branches:`` maps fold name to branch; ``branch:`` is the
    fallback for anything unlisted.

    Note this organises the data, it does not restrict access: on a public
    repository any branch can be checked out by anyone. Keeping evaluation
    material genuinely unseen needs a separate private repository, not a
    branch.
    """
    mapping = config.get("branches") or {}
    if fold and fold in mapping:
        return str(mapping[fold])
    return str(config.get("branch", "main"))


def source_coordinates(fold: Optional[str] = None) -> Dict[str, str]:
    """Which repository and branch hold ``fold``'s data.

    The data repository mirrors the layout the benchmark expects, so a plain
    checkout of it *is* a ``MATCHMAKER_DATA_DIR``. CI uses this to clone the
    right branch once instead of fetching several hundred files one at a time;
    reading it from here keeps ``data/data_sources.yaml`` the only place the
    address is written down.
    """
    config = load_config()
    return {
        "owner": str(config.get("owner", "")),
        "repo": str(config.get("repo", "")),
        "branch": branch_for(config, fold),
        "enabled": "true" if config.get("enabled", True) else "false",
    }


def candidate_urls(
    config: dict, dataset: str, path: str, fold: Optional[str] = None
) -> List[str]:
    """Every URL that might serve ``dataset/path``, in the order to try them."""
    location = (config.get("layout") or "{dataset}/{path}").format(
        dataset=dataset, path=path
    )

    override = config.get("url_template")
    if override:
        return [override.format(dataset=dataset, path=path)]

    missing = [k for k in ("owner", "repo") if not config.get(k)]
    if missing:
        raise FetchError(
            f"data/data_sources.yaml is missing {missing}. Set owner/repo/branch, "
            "or set url_template to fetch from somewhere other than GitHub."
        )
    storage = config.get("storage", "auto")
    if storage not in STORAGE_ORDER:
        raise FetchError(
            f"storage must be one of {sorted(STORAGE_ORDER)}, got '{storage}'."
        )
    return [
        GITHUB_URLS[kind].format(
            owner=config["owner"],
            repo=config["repo"],
            branch=branch_for(config, fold),
            location=location,
        )
        for kind in STORAGE_ORDER[storage]
    ]


def wanted_files(fold: str, input_type: str, pieces=None) -> List[Wanted]:
    """Every file the given fold needs, deduplicated and in fold order."""
    config = load_config()
    local_datasets = set(config.get("local_datasets") or [])
    pieces = pieces if pieces is not None else load_fold(fold, input_type=input_type)

    wanted, seen = [], set()
    for piece in pieces:
        if piece.dataset in local_datasets:
            continue
        entries = [
            (piece.xml_score, piece.score_path, []),
            (piece.match, piece.match_path, []),
        ]
        if input_type == "audio":
            canonical = piece.root / piece.audio_performance
            entries.append(
                (piece.audio_performance, canonical, piece.audio_candidates())
            )
        else:
            entries.append((piece.midi_performance, piece.performance_path("midi"), []))

        for relative, local, alternatives in entries:
            if not relative:
                continue
            key = (piece.dataset, relative)
            if key in seen:
                continue
            seen.add(key)
            wanted.append(
                Wanted(
                    dataset=piece.dataset,
                    path=relative,
                    local=local,
                    alternatives=[p for p in alternatives if p != local],
                )
            )
    return wanted


def load_checksums(dataset: str) -> Dict[str, str]:
    """Read ``data/checksums/<dataset>.sha256`` if it exists."""
    path = CHECKSUM_DIR / f"{dataset}.sha256"
    if not path.exists():
        return {}
    checksums = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        if name:
            checksums[name.strip()] = digest.strip()
    return checksums


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, token: Optional[str] = None) -> None:
    """Download to a temporary file and rename, so a partial file is never
    mistaken for a cached one."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if url.startswith("https://api.github.com/"):
        headers["Accept"] = API_ACCEPT
    last_error: Optional[Exception] = None
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                with open(temporary, "wb") as f:
                    first = response.read(1 << 20)
                    # Catch a Git LFS pointer served in place of the real file:
                    # it is a ~130-byte text stub, and writing it would look
                    # like a successful download of a corrupt piece.
                    if first.startswith(LFS_POINTER_PREFIX):
                        temporary.unlink(missing_ok=True)
                        raise FetchError(
                            f"{url} returned a Git LFS pointer, not the file. "
                            "Set `storage: lfs` (or `auto`) in "
                            "data/data_sources.yaml."
                        )
                    f.write(first)
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
            temporary.replace(destination)
            return
        except FetchError:
            raise
        except urllib.error.HTTPError as e:
            # 404 means the file is not there and 401/403 means the token is
            # wrong or absent; retrying changes neither, and the audio-container
            # fallback depends on failing fast.
            temporary.unlink(missing_ok=True)
            if e.code == 404:
                raise FetchError(f"not found: {url}") from e
            if e.code in (401, 403):
                raise FetchError(
                    f"access denied for {url} (HTTP {e.code}). If the data "
                    "repository is private, export a token in "
                    "$MATCHMAKER_DATA_TOKEN with read access to it."
                ) from e
            last_error = e
            if attempt < RETRIES:
                print(f"    retry {attempt}/{RETRIES - 1}: {e}")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_error = e
            temporary.unlink(missing_ok=True)
            if attempt < RETRIES:
                print(f"    retry {attempt}/{RETRIES - 1}: {e}")
    raise FetchError(f"could not download {url}: {last_error}")


def metadata_path(dataset: str, branch: str = "main") -> Path:
    """Where this dataset's metadata CSV is cached locally.

    Keyed by branch: two branches can describe the same dataset differently
    (one listing the validation material, another the evaluation material), and
    caching both to one path would let whichever was fetched first silently
    stand in for the other.
    """
    from matchmaker_eval.folds import dataset_root

    root = dataset_root(dataset)
    if branch in ("main", "master"):
        return root / f"metadata-{dataset}.csv"
    return root / f"metadata-{dataset}.{branch}.csv"


#: Root-level manifest names to try when a branch keeps one file for all
#: datasets instead of one per dataset folder. A manifest may be named after
#: the fold that reads it or after the branch it sits on — the two need not
#: match, and both conventions are in use — so try each.
ROOT_METADATA_NAMES = (
    "metadata-{fold}.csv",
    "metadata-{branch}.csv",
    "metadata.csv",
)


def fetch_root_metadata(fold: Optional[str], force: bool = False) -> Optional[Path]:
    """Fetch a repository-root metadata manifest, if the branch has one.

    Some branches keep one file covering every dataset, with a ``dataset``
    column, rather than one file inside each dataset folder. Both are
    reasonable shapes; this looks for the root form and returns None when the
    branch does not use it, so the caller can fall back to the per-dataset one.
    """
    from matchmaker_eval.folds import DATA_ROOT

    config = load_config()
    if not config.get("enabled"):
        return None
    branch = branch_for(config, fold)
    token = auth_token(config)

    seen = set()
    for template in ROOT_METADATA_NAMES:
        name = template.format(fold=fold or "root", branch=branch)
        if name in seen:
            continue
        seen.add(name)
        # Cached per branch: two branches may carry different manifests.
        destination = DATA_ROOT / f"{branch}--{name}"
        if destination.exists() and not force:
            return destination
        for kind in STORAGE_ORDER[config.get("storage", "auto")]:
            url = GITHUB_URLS[kind].format(
                owner=config["owner"],
                repo=config["repo"],
                branch=branch,
                location=name,
            )
            try:
                download(url, destination, token=token)
                return destination
            except FetchError:
                continue
    return None


def fetch_metadata(
    dataset: str, force: bool = False, fold: Optional[str] = None
) -> Path:
    """Download ``<dataset>/metadata-<dataset>.csv`` from the data repository.

    The repository describes its own layout: these CSVs name the paths of every
    file it holds, so the benchmark never has to encode a filename convention
    that the repository is free to change.
    """
    config = load_config()
    branch = branch_for(config, fold)
    destination = metadata_path(dataset, branch)
    if destination.exists() and not force:
        return destination
    if not config.get("enabled"):
        raise FetchError(
            f"{destination} is not present and fetching is off "
            "(data/data_sources.yaml: enabled: false)."
        )

    token = auth_token(config)
    name = f"metadata-{dataset}.csv"
    errors = []
    for url in candidate_urls(config, dataset, name, fold=fold):
        try:
            download(url, destination, token=token)
            return destination
        except FetchError as e:
            errors.append(str(e))
    raise FetchError(f"could not fetch {dataset}/{name}\n  " + "\n  ".join(errors))


def fetch_for_pieces(
    pieces,
    input_type: str,
    label: str = "",
    dry_run: bool = False,
    fold: Optional[str] = None,
) -> int:
    """Fetch whatever these specific pieces need. Returns files downloaded.

    Used both by the CLI and by ``run_submission.py``, which fetches for exactly
    the pieces it is about to evaluate rather than for the whole fold.
    """
    return _fetch(load_config(), pieces, input_type, label, dry_run, fold)


def verify(
    fold: str,
    input_type: str,
    shard: Optional[int] = None,
    num_shards: int = 1,
) -> int:
    """Report what a fold (or one shard of it) is missing. Downloads nothing.

    The preflight for a run whose data came from a checkout of the data
    repository rather than from :func:`fetch`: it names every absent file up
    front instead of letting the run discover them one piece at a time.
    Non-zero exit means something is missing.
    """
    pieces = load_fold(fold, input_type=input_type)
    if shard is not None:
        pieces = shard_of(pieces, shard, num_shards)
    label = f"{fold}/{input_type}" + (
        f" shard {shard + 1}/{num_shards}" if shard is not None else ""
    )

    wanted = wanted_files(None, input_type, pieces=pieces)
    missing = [w for w in wanted if not w.satisfied]
    print(
        f"{label}: {len(pieces)} pieces, {len(wanted)} files, "
        f"{len(missing)} missing (data root: {DATA_ROOT})"
    )
    if not missing:
        print("All present.")
        return 0
    for w in missing[:20]:
        print(f"  missing {w.dataset}/{w.path}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    print(
        f"\n{len(missing)} file(s) missing. Either MATCHMAKER_DATA_DIR does not "
        "point at a checkout of the data repository, or that branch does not "
        "carry this fold. Run fetch_data.py --source --fold "
        f"{fold} to see which branch is expected."
    )
    return 1


def fetch(
    fold: str,
    input_type: str,
    shard: Optional[int] = None,
    num_shards: int = 1,
    dry_run: bool = False,
) -> int:
    """Fetch what is missing for a fold, or one shard of it."""
    config = load_config()
    pieces = load_fold(fold, input_type=input_type)
    if shard is not None:
        pieces = shard_of(pieces, shard, num_shards)

    label = f"{fold}/{input_type}" + (
        f" shard {shard + 1}/{num_shards}" if shard is not None else ""
    )
    return _fetch(config, pieces, input_type, label, dry_run, fold)


def _fetch(
    config: dict,
    pieces,
    input_type: str,
    label: str,
    dry_run: bool,
    fold: Optional[str] = None,
) -> int:
    wanted = wanted_files(None, input_type, pieces=pieces)
    missing = [w for w in wanted if not w.satisfied]

    print(
        f"{label or input_type}: {len(pieces)} pieces, {len(wanted)} files, "
        f"{len(missing)} missing (data root: {DATA_ROOT})"
    )
    if not missing:
        return 0

    if not config.get("enabled"):
        listing = "\n  ".join(str(w.local) for w in missing[:10])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FetchError(
            f"{len(missing)} file(s) are missing and automatic fetching is off "
            f"(data/data_sources.yaml: enabled: false):\n  {listing}{more}\n\n"
            "Until the benchmark data repository is published, point "
            "MATCHMAKER_DATA_DIR at a local copy of the datasets, or use "
            "matchmaker_eval/download_data.py."
        )

    if dry_run:
        for w in missing:
            print(f"  would fetch {w.dataset}/{w.path}")
        return 0

    audio_extensions = config.get("audio_extensions") or [""]
    verify = config.get("verify_checksums", True)
    token = auth_token(config)
    checksums = {w.dataset: load_checksums(w.dataset) for w in missing}

    fetched = 0
    for index, w in enumerate(missing, 1):
        print(f"  [{index}/{len(missing)}] {w.dataset}/{w.path}", flush=True)

        # Audio: the repository may hold a different container than the fold
        # names. Try each in turn and keep whichever exists.
        if w.alternatives:
            candidates = [
                (Path(w.path).with_suffix(ext).as_posix(), w.local.with_suffix(ext))
                for ext in audio_extensions
            ]
        else:
            candidates = [(w.path, w.local)]

        errors = []
        for remote_path, destination in candidates:
            for url in candidate_urls(config, w.dataset, remote_path, fold=fold):
                try:
                    download(url, destination, token=token)
                except FetchError as e:
                    errors.append(str(e))
                    continue

                expected = checksums.get(w.dataset, {}).get(remote_path)
                if verify and expected and sha256_of(destination) != expected:
                    destination.unlink(missing_ok=True)
                    raise FetchError(f"checksum mismatch for {w.dataset}/{remote_path}")
                fetched += 1
                break
            else:
                continue
            break
        else:
            # GitHub answers an unauthenticated request for a private
            # repository with 404, which is indistinguishable from a genuinely
            # missing file. Say so rather than let people hunt for a typo.
            hint = ""
            if not token and any("not found" in e for e in errors):
                var = config.get("token_env", "MATCHMAKER_DATA_TOKEN")
                hint = (
                    f"\n\nNo token is set (${var}). A private repository "
                    "returns 404 to unauthenticated requests, so this may be an "
                    "access problem rather than a missing file. Export a token "
                    f"with read access:\n  export {var}=github_pat_...\n"
                    "Then check the layout with:\n"
                    "  python matchmaker_eval/fetch_data.py --probe "
                    f"--fold eval --input-type {input_type}"
                )
            raise FetchError(
                f"could not fetch {w.dataset}/{w.path}; tried "
                f"{[c[0] for c in candidates]}\n  " + "\n  ".join(errors) + hint
            )

    print(f"fetched {fetched} file(s) into {DATA_ROOT}")
    return fetched


def exists_in_repo(config: dict, dataset: str, path: str, token, fold=None) -> bool:
    """Is this file actually in the data repository? One HEAD request."""
    for url in candidate_urls(config, dataset, path, fold=fold):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if url.startswith("https://api.github.com/"):
            headers["Accept"] = API_ACCEPT
        try:
            request = urllib.request.Request(url, method="HEAD", headers=headers)
            with urllib.request.urlopen(request, timeout=30):
                return True
        except Exception:
            continue
    return False


def probe(fold: str, input_type: str, count: int = 3) -> int:
    """Check the data source without downloading a corpus.

    Prints the URLs the configuration produces for the first few files and
    reports what the server says about each. Run this first when the data
    repository is new, renamed, or restructured.
    """
    config = load_config()
    token = auth_token(config)
    wanted = wanted_files(fold, input_type)[:count]

    overridden = [v for k, v in ENV_OVERRIDES.items() if os.environ.get(v)]
    print(
        f"repository : {config.get('owner')}/{config.get('repo')}"
        f"@{branch_for(config, fold)}"
        + (f"   [from ${', $'.join(overridden)}]" if overridden else "")
        + "\n"
        f"layout     : {config.get('layout')}\n"
        f"storage    : {config.get('storage', 'auto')}\n"
        f"token      : {'set' if token else 'not set'} "
        f"(${config.get('token_env', 'MATCHMAKER_DATA_TOKEN')})\n"
    )

    # Ask about the repository itself first. A 404 on a file is ambiguous —
    # missing file, or a repository the token cannot see — and this separates
    # the two before anyone starts editing paths.
    reachable = None
    if config.get("owner") and config.get("repo"):
        repo_url = f"https://api.github.com/repos/{config['owner']}/{config['repo']}"
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            request = urllib.request.Request(repo_url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                info = json.loads(response.read())
                reachable = True
                print(
                    f"repository is reachable: private={info.get('private')}, "
                    f"default branch '{info.get('default_branch')}'"
                )
                fold_branch = branch_for(config, fold)
                if info.get("default_branch") != fold_branch:
                    print(
                        f"  note: fold '{fold}' reads branch "
                        f"'{fold_branch}'; the repository default is "
                        f"'{info.get('default_branch')}'"
                    )
        except urllib.error.HTTPError as e:
            reachable = False
            print(f"repository is NOT reachable ({repo_url} -> HTTP {e.code})")
        except Exception as e:
            print(f"repository check failed: {type(e).__name__}: {e}")
        print()

    audio_extensions = config.get("audio_extensions") or [""]
    problems = 0
    codes = set()
    for w in wanted:
        remote_paths = (
            [Path(w.path).with_suffix(e).as_posix() for e in audio_extensions]
            if w.alternatives
            else [w.path]
        )
        found = False
        for remote_path in remote_paths:
            for url in candidate_urls(config, w.dataset, remote_path, fold=fold):
                try:
                    probe_headers = (
                        {"Authorization": f"Bearer {token}"} if token else {}
                    )
                    if url.startswith("https://api.github.com/"):
                        probe_headers["Accept"] = API_ACCEPT
                    request = urllib.request.Request(
                        url, method="HEAD", headers=probe_headers
                    )
                    with urllib.request.urlopen(request, timeout=30) as response:
                        size = response.headers.get("Content-Length", "?")
                        print(f"  OK   {url}\n         {size} bytes")
                        found = True
                        break
                except urllib.error.HTTPError as e:
                    codes.add(e.code)
                    print(f"  {e.code:<4} {url}")
                except Exception as e:  # network, DNS, TLS
                    print(f"  ERR  {url}\n         {type(e).__name__}: {e}")
            if found:
                break
        if not found:
            problems += 1

    print()
    if problems:
        print(f"{problems}/{len(wanted)} file(s) could not be located.")
        if not token:
            var = config.get("token_env", "MATCHMAKER_DATA_TOKEN")
            print(
                "\nNo token is set. A private repository answers "
                "unauthenticated\nrequests with 404, so these may be access "
                f"failures rather than\nmissing files. Export ${var} and probe "
                "again."
            )
        elif codes & {401, 403}:
            print(
                "\nThe server rejected the token (HTTP "
                f"{sorted(codes & {401, 403})}). Check that it has not expired "
                "and that\nit grants read access to this repository — a "
                "fine-grained token must list\nthis repository explicitly and "
                "carry the 'Contents: read' permission."
            )
        elif reachable is False:
            print(
                "\nThe repository itself could not be read. Check owner/repo, "
                "and that the\ntoken's resource owner and repository access "
                "cover it."
            )
        elif reachable:
            print(
                "\nThe repository is readable, so these are path errors, not "
                "access errors.\nThe fold CSVs still hold the old layout. "
                "Rebuild them from the\nrepository's own metadata:\n\n"
                "  python matchmaker_eval/fetch_data.py --metadata\n"
                "  python matchmaker_eval/make_folds.py --from-repo\n"
                "  python matchmaker_eval/fetch_data.py --probe --fold eval "
                "--input-type midi"
            )
        else:
            print(
                "\nCheck owner/repo/branch and `layout` in "
                "data/data_sources.yaml against\nthe repository's actual "
                "directory structure."
            )
        return 1
    print(f"All {len(wanted)} probed file(s) resolved.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fold", default="eval", help="fold name or CSV path")
    parser.add_argument(
        "--input-type",
        choices=("audio", "midi"),
        default="midi",
        help="which performances",
    )
    parser.add_argument("--shard", type=int, default=None, help="fetch only this shard")
    parser.add_argument("--num-shards", type=int, default=1, help="total shards")
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be fetched"
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="fetch the data repository's own metadata CSVs and exit",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if cached"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="check the configured data source against a few files and exit",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="report what the fold is missing and exit non-zero if anything "
        "is; downloads nothing. The preflight when the data came from a "
        "checkout rather than from this script",
    )
    parser.add_argument(
        "--source",
        action="store_true",
        help="print this fold's data repository as KEY=VALUE lines and exit "
        "(owner, repo, branch); for CI, which checks the repository out "
        "instead of downloading files one at a time",
    )
    args = parser.parse_args()

    try:
        if args.source:
            for key, value in source_coordinates(args.fold).items():
                print(f"{key}={value}")
            return 0
        if args.metadata:
            branch = branch_for(load_config(), args.fold)
            print(f"fold '{args.fold}' reads branch '{branch}'\n")
            for dataset in ("asap", "batik", "vienna"):
                path = fetch_metadata(dataset, force=args.force, fold=args.fold)
                header = path.read_text().splitlines()[0]
                print(f"{dataset:8} -> {path}")
                print(f"         columns: {header}")
            return 0
        if args.probe:
            return probe(args.fold, args.input_type)
        if args.verify:
            return verify(
                args.fold,
                args.input_type,
                shard=args.shard,
                num_shards=args.num_shards,
            )
        fetch(
            args.fold,
            args.input_type,
            shard=args.shard,
            num_shards=args.num_shards,
            dry_run=args.dry_run,
        )
    except FetchError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
