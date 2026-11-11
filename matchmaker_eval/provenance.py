"""The evaluation key: one short hash naming the conditions a run was scored under.

Two leaderboard rows are only comparable when everything that decides their
numbers — apart from the follower itself — was the same. That is:

    fold_sha256         the fold CSV: which pieces, their paths, estimated_bpm
    data_sha256         the bytes of every file a piece was scored with
                        (score, the performance that was streamed, the match /
                        annotation file), over the whole fold
    scoring_sha256      the benchmark's own scoring code: the evaluation and
                        tracking modules, and the parts of run_submission.py
                        that decide ground truth and the tracked verdict
    matchmaker_scoring_sha256
                        Matchmaker's shared score loading and evaluation:
                        ground truth, beat errors, tracking windows and SPARC
    partitura           how every score and match file is parsed

``evaluation_key()`` folds these into ``ek2-<12 hex>``. Same key, same
conditions; a different key means at least one of them changed. The ``ek2``
prefix is the scheme: if what goes into the key ever changes, the prefix does
too, so a key never silently changes meaning.

The full ``matchmaker_sha256`` is also recorded, and must agree across shards
of one method's run. It is provenance, not a comparison condition: adding a
follower must not exclude existing methods from the leaderboard. Legacy ek1
records retain their original meaning and are never relabelled as ek2.
"""

import hashlib
import inspect
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Optional

from matchmaker_eval.folds import REPO_ROOT

#: Bumped whenever the set of components, or how one is computed, changes.
SCHEME = "ek2"

#: Benchmark modules whose every line can move a score. Hashed whole.
SCORING_MODULES = (
    "matchmaker_eval/eval.py",
    "matchmaker_eval/utils.py",
    "matchmaker_eval/verify_tracking.py",
)

#: Matchmaker source files that are part of the package's behaviour. Compiled
#: extensions and generated C are left out: they differ between builds of the
#: same source, and the .pyx they are built from is hashed instead.
MATCHMAKER_SOURCE_SUFFIXES = (".py", ".pyx", ".pxd", ".yaml", ".yml", ".json")

# Shared score loading and evaluation live here. Method implementations,
# registration and defaults remain in the full-package provenance hash.
MATCHMAKER_SCORING_MODULES = ("matchmaker.py", "utils/eval.py")

#: The fields of the key, in the order they are documented.
COMPONENTS = (
    "scheme",
    "fold_sha256",
    "data_sha256",
    "scoring_sha256",
    "matchmaker_scoring_sha256",
    "partitura",
)

#: The components every shard of one run must share. data_sha256 is the one
#: left out: a shard only hashes its own pieces, and merge_shards.py combines
#: the per-piece hashes into the fold's.
SHARED_COMPONENTS = tuple(c for c in COMPONENTS if c != "data_sha256") + (
    "matchmaker_sha256",
)


def comparison_components(components: dict) -> dict:
    """Comparison conditions only, interpreted under the recorded scheme."""
    if components.get("scheme") == "ek1":
        names = tuple(
            "matchmaker_sha256" if c == "matchmaker_scoring_sha256" else c
            for c in COMPONENTS
        )
    elif components.get("scheme") == SCHEME:
        names = COMPONENTS
    else:
        return {}
    return {name: components.get(name) for name in names}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_lines(lines: Iterable[str]) -> str:
    return sha256_bytes("\n".join(lines).encode())


def piece_files(piece, input_type: str) -> dict:
    """The files a piece is scored with, by role."""
    return {
        "score": piece.score_path,
        "performance": piece.performance_path(input_type),
        "ground_truth": piece.match_path,
    }


def piece_data_sha256(piece, input_type: str) -> str:
    """One hash over the bytes of every file this piece is scored with.

    The path is part of it too, relative to the dataset: the same bytes under
    another name are another fold row.
    """
    lines = []
    for role, path in sorted(piece_files(piece, input_type).items()):
        relative = Path(path).relative_to(piece.root)
        lines.append(f"{role}\t{relative.as_posix()}\t{file_sha256(path)}")
    return _digest_lines(lines)


def data_sha256(pieces: Iterable[dict]) -> Optional[str]:
    """The fold's data hash, from the per-piece hashes of a run's records.

    None when any piece has no hash: a key that silently covers only part of
    the data would claim more than it knows.
    """
    lines = []
    for record in sorted(pieces, key=lambda r: r["index"]):
        piece_hash = record.get("data_sha256")
        if not piece_hash:
            return None
        lines.append(f"{record['index']}\t{record['piece_id']}\t{piece_hash}")
    return _digest_lines(lines) if lines else None


def scoring_sha256(decisive: dict) -> str:
    """The benchmark's scoring code.

    The scoring modules whole, plus ``decisive``: what run_submission.py
    contributes to a score — its tracked-verdict thresholds and the functions
    that resolve ground truth and call the evaluation. Functions are hashed by
    their source. The rest of that file (argument parsing, logging, fetching)
    can change without changing a number, so it is left out.
    """
    lines = [
        f"{module}\t{file_sha256(REPO_ROOT / module)}" for module in SCORING_MODULES
    ]
    spelled = {
        name: inspect.getsource(value) if callable(value) else value
        for name, value in decisive.items()
    }
    lines.append(
        "run_submission\t"
        + sha256_bytes(json.dumps(spelled, sort_keys=True, default=str).encode())
    )
    return _digest_lines(lines)


def matchmaker_sha256() -> str:
    """The installed matchmaker package's sources, as one hash."""
    import matchmaker

    root = Path(matchmaker.__file__).resolve().parent
    lines = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.suffix in MATCHMAKER_SOURCE_SUFFIXES
            and "__pycache__" not in path.parts
        ):
            lines.append(f"{path.relative_to(root).as_posix()}\t{file_sha256(path)}")
    return _digest_lines(lines)


def matchmaker_scoring_sha256() -> str:
    """Hash shared score loading and evaluation independently of followers."""
    import matchmaker

    root = Path(matchmaker.__file__).resolve().parent
    return _digest_lines(
        f"{module}\t{file_sha256(root / module)}"
        for module in MATCHMAKER_SCORING_MODULES
    )


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def code_components(fold_sha256: str, decisive: dict) -> dict:
    """Every component except the data, which needs the run's pieces."""
    return {
        "scheme": SCHEME,
        "fold_sha256": fold_sha256,
        "scoring_sha256": scoring_sha256(decisive),
        "matchmaker_sha256": matchmaker_sha256(),
        "matchmaker_scoring_sha256": matchmaker_scoring_sha256(),
        "partitura": package_version("partitura"),
    }


def evaluation_key(components: dict) -> Optional[str]:
    """A scheme-qualified key for complete comparison conditions, else None."""
    conditions = comparison_components(components)
    if not conditions or not all(conditions.values()):
        return None
    canonical = json.dumps(conditions, sort_keys=True)
    return f"{components['scheme']}-{sha256_bytes(canonical.encode())[:12]}"


def evaluation_record(code: dict, pieces: list) -> dict:
    """The ``evaluation`` block a metrics.json carries: the key and what made it."""
    components = {**code, "data_sha256": data_sha256(pieces)}
    components = {name: components.get(name) for name in (*SHARED_COMPONENTS, "data_sha256")}
    return {"key": evaluation_key(components), "components": components}
