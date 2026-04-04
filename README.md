# matchmaker-benchmark

Benchmark for real-time music alignment using the [matchmaker](https://github.com/pymatchmaker/matchmaker) package. Supports both audio and MIDI (symbolic) score following.

## Setup

### Code

```bash
git clone https://github.com/pymatchmaker/matchmaker-benchmark.git
git clone https://github.com/pymatchmaker/matchmaker.git

conda env create -f environment.yml
conda activate matchmaker-benchmark

cd ../matchmaker
pip install -e ".[dev]"

conda install -c conda-forge gcc=12.1.0 glib fluidsynth
```

### Datasets

Set dataset paths in `matchmaker_eval/test_audio.py` and `matchmaker_eval/test_symbolic.py`:

```python
DATASET_DIR = {
    "asap": Path("~/data/asap-dataset-matchmaker").expanduser(),
    "batik": Path("~/data/batik_plays_mozart").expanduser(),
    "vienna": Path("~/data/vienna4x22").expanduser(),
}
```

Metadata CSV files are in `data/` (full) and `data/reduced/` (test set).

## Running experiments

### Audio score following

```bash
# Single dataset + method
python matchmaker_eval/test_audio.py --dataset asap --method arzt

# Available methods: arzt, dixon, outerhmm
# Available datasets: valid, asap, batik, vienna
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
- `wp_{i}.tsv` — warping path per piece (score_beat, perf_time)
- `gt_{i}.tsv` — ground truth per piece (score_beat, perf_time)
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
  eval_symbolic.py     — symbolic eval (legacy, Alex's original code)
  utils.py             — shared utilities (config, summary, metrics)
  verify_tracking.py   — segment-based tracking verification
data/
  metadata-*.csv       — dataset metadata (full)
  reduced/             — reduced metadata (test set)
output/                — experiment results
final_results/         — summary CSVs for paper tables
scripts/               — utility scripts
```

## Acknowledgments

This work has been supported by the Austrian Science Fund (FWF), grant agreement PAT 8820923 ("Rach3: A Computational Approach to Study Piano Rehearsals"). Additionally, this work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. NRF-2023R1A2C3007605).
