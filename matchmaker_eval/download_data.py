"""Download benchmark datasets for matchmaker-benchmark.

Usage:
    python matchmaker_eval/download_data.py             # all, into $MATCHMAKER_DATA_DIR
    python matchmaker_eval/download_data.py --dataset asap
    python matchmaker_eval/download_data.py --data-dir ~/my_data
"""

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from folds import DATA_ROOT

# TODO: update URLs when data hosting is finalized
DATASETS = {
    "asap": {
        "url": "https://github.com/CPJKU/asap-dataset",
        "target": "asap-dataset",
    },
    "batik": {
        "url": "https://github.com/huispaty/batik_plays_mozart",
        "target": "batik_plays_mozart",
    },
    "vienna": {
        "url": "https://github.com/CPJKU/vienna4x22",
        "target": "vienna4x22",
    },
}

DEFAULT_DATA_DIR = DATA_ROOT


def download_dataset(name: str, data_dir: Path) -> None:
    info = DATASETS[name]
    target = data_dir / info["target"]

    if target.exists():
        print(f"[{name}] Already exists at {target}, skipping.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    url = info["url"]
    archive = data_dir / Path(url).name

    print(f"[{name}] Downloading from {url} ...")
    urlretrieve(url, archive)

    print(f"[{name}] Extracting to {data_dir} ...")
    if archive.suffix == ".gz" and archive.stem.endswith(".tar"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(data_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive.name}")

    archive.unlink()
    print(f"[{name}] Done -> {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Download matchmaker benchmark datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
        help="Dataset to download (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root directory for datasets (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else list(DATASETS.keys())
    data_dir = args.data_dir.expanduser()

    for name in targets:
        download_dataset(name, data_dir)


if __name__ == "__main__":
    main()
