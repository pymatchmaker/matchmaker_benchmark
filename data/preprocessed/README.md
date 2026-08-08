# Benchmark-preprocessed data

This directory contains only assets that were derived or adjusted specifically for
this benchmark. Keep each downloaded raw dataset in its own local directory under
`~/data`; do not copy complete raw datasets into this repository.

Metadata CSV files remain in `data/`. Paths beginning with
`data/preprocessed/` are resolved relative to the repository root. Paths beginning
with `dataset_root/` are resolved relative to that dataset's directory. All other
score and audio paths are resolved relative to the raw dataset directory declared
in `matchmaker_eval/test_audio.py`.

The shared archive has normalized folder names under `datasets/`:

```text
datasets/
├── chorale/
├── kraisler/
├── urmp/
│   └── trimmed_audio/
└── winterreise/
```

After extracting it, point the benchmark at that directory:

```bash
export MATCHMAKER_DATASETS_ROOT=/path/to/extracted/datasets
```

Without the environment variable, the runners use `~/data` and retain fallback
support for the legacy local names `URMP`, `KRAISLER`, and `chorale-bricks`.

## Layout

```text
preprocessed/
├── chorale/ground_truth/
├── kraisler/ground_truth/        # piano + violin note-onset GT
├── urmp/
│   ├── ground_truth/original/    # GT on the original audio timeline
│   ├── ground_truth/trimmed/     # GT for the benchmark timeline
│   └── trim_offsets.csv
└── winterreise/
    ├── ground_truth/
    └── musicxml/                 # scores transposed for each performance
```

## Dataset notes

- **URMP:** `python matchmaker_eval/test_audio.py --dataset urmp ...` uses
  `data/metadata-urmp-trimmed.csv`. Thirty-eight rows use WAV files in
  `urmp/trimmed_audio/`; the six rows that needed no initial trim still use the
  original `AuMix_*.wav` in `~/data/URMP/<piece>/`. `data/metadata-urmp.csv` preserves
  the fully original timeline and uses `ground_truth/original/`. Run it with
  `--dataset urmp-original`.
- **Winterreise:** audio remains in `~/data/winterreise`. The benchmark uses the
  performance-specific transposed scores in `winterreise/musicxml/`.
- **KRAISLER:** 20 scores are evaluated with `mix_dry`, `mix_hall`, and
  `mix_studio`, producing 60 metadata rows. Each track's note-level GT combines
  piano `.match` onsets and aligned violin note annotations.
- **Chorale:** scores and audio remain in the raw dataset folder; only the
  generated benchmark ground truth is stored here.

## Sharing and regeneration

Ground-truth TSV and MusicXML files are small and can be versioned. URMP WAV
files stay with the local dataset under `~/data/URMP/trimmed_audio/`; the shared
dataset archive contains the same directory as `datasets/urmp/trimmed_audio/`.
Regenerate the WAV files and trimmed metadata from the original URMP download
with:

```bash
conda run -n matchmaker-benchmark python matchmaker_eval/preprocess_urmp.py
```
