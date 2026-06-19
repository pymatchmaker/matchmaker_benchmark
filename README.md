# A Systematic Comparison of Methods for Real-time Music Alignment

A benchmark for real-time music alignment using the [matchmaker](https://github.com/pymatchmaker/matchmaker) package, supporting both audio and MIDI (symbolic) score following.

## Setup

### Setting up the code

Setting up the experiments as described here requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Follow the instructions for your OS.

To setup the experiments, use the following script.

```bash
git clone https://github.com/pymatchmaker/matchmaker_benchmark.git
git clone https://github.com/pymatchmaker/matchmaker.git

conda env create -f environment.yml
conda activate matchmaker-benchmark

cd ../matchmaker
pip install -e ".[dev]"

conda install -c conda-forge gcc=12.1.0 glib fluidsynth
```

### Datasets

> **Note:** Dataset hosting will be moved to a dedicated repository.

Download all datasets with:

```bash
python matchmaker_eval/download_data.py                    # all datasets
python matchmaker_eval/download_data.py --dataset asap     # single dataset
python matchmaker_eval/download_data.py --data-dir ~/data  # custom directory
```

Datasets are saved under `~/datasets/` by default. To use a different location, pass `--data-dir` and set the matching paths in `matchmaker_eval/test_audio.py` and `matchmaker_eval/test_symbolic.py`:
Metadata CSV files are in `./data/`.

## Running experiments

### Audio score following

```bash
# Single dataset + method
python matchmaker_eval/test_audio.py --dataset asap --method arzt

# Available methods: arzt, dixon, outerhmm
# Available datasets: valid, asap, batik, vienna, urmp, chorale
```

Methods use frame-level features (chroma, LSE, CQT). Results saved in `output/`.

### MIDI (symbolic) score following

```bash
# HMM methods (note-level features)
python matchmaker_eval/test_symbolic.py --dataset asap --method hmm

# Event-level OLTW methods (onset pianoroll)
python matchmaker_eval/test_symbolic.py --dataset asap --method arzt

# Available methods: hmm, pthmm, outerhmm, arzt, dixon
# Available datasets: valid, asap, batik, vienna
```

For MIDI, `arzt` and `dixon` use event-level OLTW variants (`OnlineTimeWarpingArztEvent` / `OnlineTimeWarpingDixonEvent`) which align onset-by-onset rather than frame-by-frame. HMM methods (`hmm`, `pthmm`, `outerhmm`) use Matchmaker's standard pipeline.

### Output

Each run creates a directory in `output/` containing:
- `wp_{i}.tsv` — warping path per piece (perf_sec, score_beat)
- `gt_{i}.tsv` — ground truth per piece (perf_sec, score_beat)
- `{i}.json` — per-piece tracking result
- `summary_tracked.json` — event-pooled summary over tracked pieces

### Evaluation protocol

- **Primary metric**: beat error (perf→score direction)
- **Secondary metric**: ms error (score→perf direction)
- **Tracking**: 30-second segments, median beat error per segment
  - Audio threshold: 1.0 beat, min_fails=2
  - MIDI threshold: 0.5 beat, min_fails=2
- **Aggregation**: event-pooled across tracked pieces

## Project structure

```
matchmaker_eval/
  eval.py              — single-piece alignment functions (audio + symbolic)
  test_audio.py        — audio benchmark runner
  test_symbolic.py     — symbolic benchmark runner
  utils.py             — shared utilities (config, summary, metrics)
  verify_tracking.py   — segment-based tracking verification
data/
  metadata-*.csv       — dataset metadata (full)
  reduced/             — reduced metadata (test set)
```

## Acknowledgments

This work has been supported by the Austrian Science Fund (FWF), grant agreement PAT 8820923 ("Rach3: A Computational Approach to Study Piano Rehearsals"). Additionally, this work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. NRF-2023R1A2C3007605).
