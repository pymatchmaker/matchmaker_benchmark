"""Generate and check the benchmark folds in ``data/folds/``.

    python matchmaker_eval/make_folds.py            # regenerate + check
    python matchmaker_eval/make_folds.py --check    # check only, no writes

The folds are derived from the dataset metadata already in ``data/``:

``valid``   <- ``data/metadata-validation.csv`` (the set the sweeps use)
``eval``     <- ``data/reduced/metadata-{asap,batik,vienna}.csv``
``example``  <- the single piece committed under ``resources/``

``--from-repo`` instead rebuilds the folds from the data repository's own
manifests. That is the authoritative path, and the only one that fills in
``estimated_bpm``: the repository publishes a tempo for every performance it
holds, and every one of them reproduces from the score and performance MIDI it
ships. The default path leaves the column blank rather than guess, and the
check step reports how many rows came out that way.

The generated CSVs are committed so that any change to what the benchmark
measures arrives as a reviewable diff rather than as a silent shift in the
leaderboard. Regenerate them only deliberately: ``eval`` is frozen, and
changing it invalidates every published number.

The check step enforces the property the leaderboard depends on — no
performance appears in both the validation and the eval fold — and reports the
*score*-level overlap, which is unavoidable for datasets such as vienna4x22
(4 pieces, 22 pianists each). See ``docs/eval-protocol.md``.
"""

# Entry-point path setup. The repository root goes on sys.path so this file and
# submissions agree on one `matchmaker_eval.*` module identity (importing the
# same file twice under two names would give two unrelated `Submission`
# classes, and every isinstance check would quietly fail). The package
# directory goes on too, because the older eval modules import each other by
# bare module name.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matchmaker_eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import csv
from collections import Counter, defaultdict

from matchmaker_eval.folds import FOLD_DIR, REPO_ROOT, Piece, load_fold, write_fold

METADATA_DIR = REPO_ROOT / "data"

TUNING_SOURCES = [("<from column>", METADATA_DIR / "metadata-validation.csv")]
EVAL_SOURCES = [
    ("asap", METADATA_DIR / "reduced" / "metadata-asap.csv"),
    ("batik", METADATA_DIR / "reduced" / "metadata-batik.csv"),
    ("vienna", METADATA_DIR / "reduced" / "metadata-vienna.csv"),
]

EXAMPLE_PIECE = Piece(
    piece_id="local/Fugue_bwv_858/ex_VuV01M",
    dataset="local",
    composer="Bach",
    title="Fugue_bwv_858",
    xml_score="resources/ex_score.musicxml",
    midi_performance="resources/ex_VuV01M.mid",
    audio_performance="resources/ex_VuV01M.wav",
    match="resources/ex_VuV01M.match",
    difficulty="3",
    estimated_bpm="58",
)


#: Column names the repository's metadata might use for each role. The first
#: one present wins, so the repository can rename a column without this
#: needing an edit — and an unrecognised schema is reported rather than
#: guessed at.
REPO_COLUMNS = {
    "xml_score": ("xml_score", "score", "score_file", "musicxml"),
    "midi_performance": ("midi_performance", "midi", "performance_midi"),
    "audio_performance": ("audio_performance", "audio", "performance_audio", "mp3"),
    "match": ("match", "match_file", "alignment"),
    "title": ("title", "piece", "name"),
    "composer": ("composer",),
    "difficulty": ("difficulty",),
    "estimated_bpm": ("estimated_bpm", "tempo", "bpm"),
}


def repo_columns() -> dict:
    """:data:`REPO_COLUMNS`, with the configured tempo column name first.

    ``data/data_sources.yaml`` may name the tempo column explicitly, for a data
    branch caught mid-rename. Honouring it here rather than at run time keeps
    the guessing in the one place that reads the repository's manifests.
    """
    columns = dict(REPO_COLUMNS)
    try:
        import yaml

        config = (
            yaml.safe_load((METADATA_DIR / "data_sources.yaml").read_text()) or {}
        )
    except Exception:
        return columns
    configured = config.get("estimated_bpm_column")
    if not configured:
        return columns
    names = (
        [configured] if isinstance(configured, str) else [str(c) for c in configured]
    )
    columns["estimated_bpm"] = tuple(
        dict.fromkeys([*names, *columns["estimated_bpm"]])
    )
    return columns


def resolve_columns(header) -> dict:
    """Map each role to the column that carries it, or None."""
    present = {h.strip(): h.strip() for h in header}
    resolved = {}
    for role, candidates in repo_columns().items():
        resolved[role] = next((c for c in candidates if c in present), None)
    return resolved


def repo_key(relative: str) -> str:
    """Identity of a repository file: its stem.

    The repository flattens the original directories into the filename for asap
    (``Bach/Fugue/bwv_858/Zhang01M.mid`` -> ``midi/Bach_Fugue_bwv_858_Zhang01M.mid``)
    and drops them for batik and vienna. The stem survives either way.
    """
    return Path(relative).stem


def fold_keys(relative: str) -> list:
    """Identities a fold path might be known by in the repository.

    Flattened first: an asap performance stem such as ``Zhang01M`` is reused
    across pieces, so the flattened form is the unambiguous one.
    """
    path = Path(relative)
    return ["_".join(path.with_suffix("").parts), path.stem]


def learn_directories(pieces) -> dict:
    """Which folder each kind of file sits in, learned from the metadata.

    Read off the repository's own paths rather than hard-coded, so a rename of
    ``score/`` to ``scores/`` needs no change here.
    """
    dirs = {}
    for piece in pieces:
        for role, value in (
            ("xml_score", piece.xml_score),
            ("midi_performance", piece.midi_performance),
            ("audio_performance", piece.audio_performance),
            ("match", piece.match),
        ):
            if value and role not in dirs:
                dirs[role] = (Path(value).parent.as_posix(), Path(value).suffix)
    return dirs


def recover_piece(dataset, piece, dirs, config, token, fold=None):
    """Find a piece the metadata does not list, if its files are there anyway.

    The metadata may cover only part of what the repository holds. Rather than
    assume either way, this constructs the paths the repository's own naming
    would give and asks the server whether each one exists. Nothing is accepted
    unverified.
    """
    from matchmaker_eval.fetch_data import exists_in_repo

    found = {}
    for role, source in (
        ("xml_score", piece.xml_score),
        ("midi_performance", piece.midi_performance),
        ("audio_performance", piece.audio_performance),
        ("match", piece.match),
    ):
        if not source or role not in dirs:
            continue
        folder, suffix = dirs[role]
        for stem in fold_keys(source):
            candidate = f"{folder}/{stem}{suffix}"
            if exists_in_repo(config, dataset, candidate, token, fold=fold):
                found[role] = candidate
                break
    return found


def normalise_path(dataset: str, value: str) -> str:
    """Make a metadata path relative to its dataset folder.

    Per-dataset manifests write ``audio/x.mp3``; a root manifest covering every
    dataset writes ``batik/audio/x.mp3``. Stripping the leading dataset segment
    leaves one internal form, so nothing downstream has to know which shape the
    branch used.
    """
    value = (value or "").strip()
    prefix = f"{dataset}/"
    return value.removeprefix(prefix) if value.startswith(prefix) else value


def read_root_metadata(path: Path, datasets=("asap", "batik", "vienna")):
    """Read a root manifest covering every dataset, keyed by a ``dataset`` column.

    Yields the same :class:`Piece` objects as the per-dataset reader, with paths
    normalised to be dataset-relative.
    """
    with open(path, newline="") as f:
        rows = [_clean(r) for r in csv.DictReader(f, skipinitialspace=True)]
    if not rows:
        raise SystemExit(f"{path} is empty.")
    if "dataset" not in rows[0]:
        raise SystemExit(
            f"{path}: a root manifest needs a 'dataset' column naming which "
            f"dataset each row belongs to. Columns present: {sorted(rows[0])}"
        )

    columns = resolve_columns(rows[0].keys())
    seen = Counter()
    for row in rows:
        dataset = row.get("dataset", "")
        if datasets and dataset not in datasets:
            continue

        def value(role):
            column = columns[role]
            return normalise_path(dataset, row.get(column, "")) if column else ""

        performance = value("midi_performance") or value("audio_performance")
        stem = Path(performance).stem if performance else "unknown"
        title = (row.get(columns["title"], "") if columns["title"] else "") or stem
        piece_id = f"{dataset}/{title}/{stem}"
        seen[piece_id] += 1
        if seen[piece_id] > 1:
            piece_id = f"{piece_id}#{seen[piece_id]}"
        yield Piece(
            piece_id=piece_id,
            dataset=dataset,
            composer=row.get(columns["composer"], "") if columns["composer"] else "",
            title=title,
            xml_score=value("xml_score"),
            midi_performance=value("midi_performance"),
            audio_performance=value("audio_performance"),
            match=value("match"),
            difficulty=(
                row.get(columns["difficulty"], "") if columns["difficulty"] else ""
            ),
            estimated_bpm=(
                row.get(columns["estimated_bpm"], "")
                if columns["estimated_bpm"]
                else ""
            ),
        )


def read_repo_metadata(dataset: str, path: Path):
    """Build pieces from the data repository's own metadata CSV.

    The repository names the paths of everything it holds, so the fold records
    exactly those. Nothing here encodes a filename convention: restructure the
    repository and only its CSV changes.
    """
    with open(path, newline="") as f:
        rows = [_clean(r) for r in csv.DictReader(f, skipinitialspace=True)]
    if not rows:
        raise SystemExit(f"{path} is empty.")

    columns = resolve_columns(rows[0].keys())
    required = ("xml_score", "match")
    missing = [r for r in required if not columns[r]]
    if missing:
        raise SystemExit(
            f"{path}: could not find a column for {missing}.\n"
            f"  columns present: {sorted(rows[0])}\n"
            f"  add the name to REPO_COLUMNS in make_folds.py"
        )
    if not columns["midi_performance"] and not columns["audio_performance"]:
        raise SystemExit(f"{path}: no midi or audio performance column found.")

    seen = Counter()
    for row in rows:

        def value(role):
            column = columns[role]
            return normalise_path(dataset, row.get(column, "")) if column else ""

        performance = value("midi_performance") or value("audio_performance")
        stem = Path(performance).stem if performance else "unknown"
        title = (row.get(columns["title"], "") if columns["title"] else "") or stem
        piece_id = f"{dataset}/{title}/{stem}"
        seen[piece_id] += 1
        if seen[piece_id] > 1:
            piece_id = f"{piece_id}#{seen[piece_id]}"
        yield Piece(
            piece_id=piece_id,
            dataset=dataset,
            composer=value("composer"),
            title=title,
            xml_score=value("xml_score"),
            midi_performance=value("midi_performance"),
            audio_performance=value("audio_performance"),
            match=value("match"),
            difficulty=value("difficulty"),
            estimated_bpm=(
                row.get(columns["estimated_bpm"], "")
                if columns["estimated_bpm"]
                else ""
            ),
        )


def _clean(row: dict) -> dict:
    return {(k or "").strip(): (v or "").strip() for k, v in row.items()}


def read_metadata_rows(dataset: str, path: Path):
    """Yield :class:`Piece` objects from one metadata CSV."""
    with open(path, newline="") as f:
        rows = [_clean(r) for r in csv.DictReader(f, skipinitialspace=True)]

    seen = Counter()
    for row in rows:
        # metadata-validation.csv mixes datasets and names the source in a column.
        name = row.get("dataset") or dataset
        perf = row.get("midi_performance") or row.get("audio_performance")
        stem = Path(perf).stem if perf else "unknown"
        piece_id = f"{name}/{row['title']}/{stem}"
        seen[piece_id] += 1
        if seen[piece_id] > 1:  # defensive: keep ids unique even on duplicates
            piece_id = f"{piece_id}#{seen[piece_id]}"
        yield Piece(
            piece_id=piece_id,
            dataset=name,
            composer=row.get("composer", ""),
            title=row.get("title", ""),
            xml_score=row.get("xml_score", ""),
            midi_performance=row.get("midi_performance", ""),
            audio_performance=row.get("audio_performance", ""),
            match=row.get("match", ""),
            difficulty=row.get("difficulty", ""),
            # Left blank deliberately. data/perf_tempo_estimate/ holds tempi
            # keyed by these very paths, but they were measured against the
            # upstream corpora and four vienna rows disagree with the data
            # repository's by 1-3 BPM. A blank stops an `estimated_bpm: true`
            # run outright; a plausible wrong number would just quietly shift
            # what it measured. --from-repo fills these in.
            estimated_bpm="",
        )


def build(sources) -> list:
    pieces = []
    for dataset, path in sources:
        if not path.exists():
            raise SystemExit(f"metadata source not found: {path}")
        pieces.extend(read_metadata_rows(dataset, path))
    return pieces


def check(eval_pieces, valid_pieces) -> int:
    """Report fold hygiene. Returns the number of hard failures."""
    failures = 0

    incomplete = [
        p.piece_id
        for p in eval_pieces + valid_pieces
        if not p.xml_score
        or not p.match
        or not (p.midi_performance or p.audio_performance)
    ]
    if incomplete:
        print(f"FAIL  {len(incomplete)} fold row(s) lack a score, GT or performance:")
        for piece_id in incomplete[:10]:
            print(f"        {piece_id}")
        failures += 1

    eval_ids = {p.piece_id for p in eval_pieces}
    shared_performances = sorted(eval_ids & {p.piece_id for p in valid_pieces})
    if shared_performances:
        print(f"FAIL  {len(shared_performances)} performance(s) are in both folds:")
        for piece_id in shared_performances[:10]:
            print(f"        {piece_id}")
        failures += 1
    else:
        print("ok    no performance appears in both the validation and the eval fold")

    duplicates = [
        pid for pid, n in Counter(p.piece_id for p in eval_pieces).items() if n > 1
    ]
    if duplicates:
        print(f"FAIL  duplicate piece ids in the eval fold: {duplicates[:10]}")
        failures += 1

    # A blank tempo is not a failure — most followers never ask for it — but it
    # silently disarms every entry that declares `estimated_bpm: true`, so it
    # has to be visible when a fold is regenerated.
    without_tempo = [p.piece_id for p in eval_pieces + valid_pieces
                     if not p.estimated_bpm]
    if without_tempo:
        total = len(eval_pieces) + len(valid_pieces)
        print(
            f"note  {len(without_tempo)}/{total} row(s) carry no estimated_bpm, so a "
            "follower declaring\n      it uses the tempo cannot run on them:"
        )
        for piece_id in without_tempo[:5]:
            print(f"        {piece_id}")
        print(
            "      Only --from-repo fills this column in:\n"
            "        python matchmaker_eval/make_folds.py --from-repo"
        )
    else:
        print("ok    every row carries an estimated_bpm")

    eval_titles = {(p.dataset, p.title) for p in eval_pieces}
    shared_scores = sorted(eval_titles & {(p.dataset, p.title) for p in valid_pieces})
    if shared_scores:
        print(
            f"note  {len(shared_scores)} score(s) appear in both folds, played by "
            "different performers:"
        )
        by_dataset = defaultdict(list)
        for dataset, title in shared_scores:
            by_dataset[dataset].append(title)
        for dataset, titles in sorted(by_dataset.items()):
            print(
                f"        {dataset}: {len(titles)} ({', '.join(sorted(titles)[:4])}...)"
            )
        print(
            "      This is inherent to the datasets (vienna4x22 has 4 scores in\n"
            "      total). Tuning therefore sees these scores, never these\n"
            "      performances. Documented in docs/eval-protocol.md."
        )

    counts = Counter(p.dataset for p in eval_pieces)
    print(
        f"\neval fold   : {len(eval_pieces)} performances "
        f"{dict(sorted(counts.items()))}"
    )
    counts = Counter(p.dataset for p in valid_pieces)
    print(
        f"validation fold : {len(valid_pieces)} performances "
        f"{dict(sorted(counts.items()))}"
    )
    return failures


def repo_pieces_for(fold: str) -> dict:
    """Read a fold's branch metadata, whichever shape that branch uses.

    A branch may keep one manifest inside each dataset folder, or a single
    manifest at the repository root with a ``dataset`` column. Per-dataset is
    tried first and the root form is the fallback; both are normalised to the
    same result, so nothing downstream depends on which was found.
    """
    from matchmaker_eval.fetch_data import (
        FetchError,
        fetch_metadata,
        fetch_root_metadata,
    )

    by_dataset = {}
    for dataset in ("asap", "batik", "vienna"):
        try:
            path = fetch_metadata(dataset, fold=fold)
        except (FetchError, SystemExit):
            continue
        by_dataset[dataset] = list(read_repo_metadata(dataset, path))
    if by_dataset:
        return by_dataset

    root = fetch_root_metadata(fold)
    if root is None:
        raise SystemExit(
            f"No metadata found for fold '{fold}'. Its branch has neither "
            "<dataset>/metadata-<dataset>.csv nor a root manifest "
            "(metadata-<fold>.csv or metadata.csv)."
        )
    print(f"           root manifest {root.name}")
    for piece in read_root_metadata(root):
        by_dataset.setdefault(piece.dataset, []).append(piece)
    return by_dataset


def build_from_repo() -> int:
    """Regenerate the folds against the data repository's layout.

    The eval and validation folds keep exactly the pieces they already name — the
    benchmark's contents do not change — but their paths are replaced with the
    ones the repository publishes. Membership is matched on (dataset, title,
    performance stem), which is what ``piece_id`` is built from and is stable
    across a repository restructure.
    """
    # The repository metadata carries only paths, no titles, so identity comes
    # from the filenames. Each fold is read from its own branch, so a repository
    # that keeps validation and evaluation material apart is handled naturally.
    repo_pieces, directories = {}, {}
    for fold in ("eval", "valid"):
        pieces = repo_pieces_for(fold)
        for dataset, items in pieces.items():
            for piece in items:
                key = piece.midi_performance or piece.audio_performance
                if key:
                    repo_pieces[(dataset, repo_key(key))] = piece
            directories.setdefault(dataset, learn_directories(items))
        counts = {k: len(v) for k, v in sorted(pieces.items())}
        print(f"{fold:10} {sum(counts.values()):>3} rows {counts}")

    from matchmaker_eval.fetch_data import auth_token, load_config

    config = load_config()
    token = auth_token(config)

    failed_folds = []
    for name in ("eval", "valid"):
        current = load_fold(name)
        rebuilt, missing = [], []
        recovered_count = 0
        for piece in current:
            anchor = piece.midi_performance or piece.audio_performance
            match = None
            for key in fold_keys(anchor):
                match = repo_pieces.get((piece.dataset, key))
                if match is not None:
                    break
            if match is None:
                recovered = recover_piece(
                    piece.dataset,
                    piece,
                    directories.get(piece.dataset, {}),
                    config,
                    token,
                    fold=name,
                )
                if len(recovered) >= 2 and "match" in recovered:
                    rebuilt.append(
                        Piece(
                            piece_id=piece.piece_id,
                            dataset=piece.dataset,
                            composer=piece.composer,
                            title=piece.title,
                            xml_score=recovered.get("xml_score", ""),
                            midi_performance=recovered.get("midi_performance", ""),
                            audio_performance=recovered.get("audio_performance", ""),
                            match=recovered.get("match", ""),
                            difficulty=piece.difficulty,
                            estimated_bpm=piece.estimated_bpm,
                        )
                    )
                    recovered_count += 1
                    continue
                missing.append(piece.piece_id)
                continue
            # Keep the fold's own identity, take the repository's paths.
            rebuilt.append(
                Piece(
                    piece_id=piece.piece_id,
                    dataset=piece.dataset,
                    composer=match.composer or piece.composer,
                    title=piece.title,
                    xml_score=match.xml_score,
                    midi_performance=match.midi_performance,
                    audio_performance=match.audio_performance,
                    match=match.match,
                    difficulty=match.difficulty or piece.difficulty,
                    # The repository measured these on the audio it actually
                    # ships, so its value wins; the fold keeps whatever it
                    # already had when the manifest carries no tempo.
                    estimated_bpm=match.estimated_bpm or piece.estimated_bpm,
                )
            )
        if missing:
            failed_folds.append(name)
            print(
                f"\n{name}: {len(missing)}/{len(current)} piece(s) are not in the "
                f"repository, e.g.\n  " + "\n  ".join(missing[:5])
            )
            print(f"  {name}.csv left untouched.")
        elif recovered_count:
            path = FOLD_DIR / f"{name}.csv"
            write_fold(rebuilt, path)
            print(
                f"{name:8} rewrote {len(rebuilt)} rows "
                f"({recovered_count} found in the repository but absent from its "
                f"metadata) -> {path.relative_to(REPO_ROOT)}"
            )
        else:
            path = FOLD_DIR / f"{name}.csv"
            write_fold(rebuilt, path)
            print(
                f"{name:8} rewrote {len(rebuilt)} rows -> {path.relative_to(REPO_ROOT)}"
            )

    if "eval" in failed_folds:
        print(
            "\nThe eval fold was not rewritten: a fold must keep every piece it "
            "names, or\npublished numbers stop being comparable."
        )
        return 1
    if failed_folds:
        print(
            f"\n{', '.join(failed_folds)} could not be rebuilt, but eval could, so "
            "the leaderboard is\nunaffected. Those folds keep their local paths and "
            "still work against a local\ndataset copy. Publish their pieces to the "
            "data repository to switch them over."
        )
    print(
        "\nVerify before running anything long:\n"
        "  python matchmaker_eval/fetch_data.py --probe --fold eval --input-type audio"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the committed folds without rewriting them",
    )
    parser.add_argument(
        "--from-repo",
        action="store_true",
        help="rebuild the folds from the data repository's own metadata CSVs, "
        "so the paths are whatever the repository says they are",
    )
    args = parser.parse_args()

    if args.from_repo:
        return build_from_repo()

    if args.check:
        eval_pieces = load_fold("eval")
        valid_pieces = load_fold("valid")
    else:
        eval_pieces = build(EVAL_SOURCES)
        valid_pieces = build(TUNING_SOURCES)

    failures = check(eval_pieces, valid_pieces)
    if failures:
        print(f"\n{failures} check(s) failed.")
        return 1

    if not args.check:
        for name, pieces in [
            ("eval", eval_pieces),
            ("valid", valid_pieces),
            ("example", [EXAMPLE_PIECE]),
        ]:
            path = FOLD_DIR / f"{name}.csv"
            n = write_fold(pieces, path)
            print(f"wrote {n:4d} rows -> {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
