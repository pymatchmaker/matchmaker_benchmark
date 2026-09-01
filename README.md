# Matchmaker Benchmark

An open leaderboard for **real-time score following**: how well can a machine
follow a live performer through a score?

**[→ Leaderboard](https://pymatchmaker.github.io/matchmaker_benchmark/)**

Write one Python file, open a pull request, and after review your score follower
is evaluated on 146 fixed performances and published. **You never download any
data** — the evaluation runs on GitHub's runners, which fetch what they need.

| | |
| --- | --- |
| **Evaluation set** | 146 performances (ASAP · Batik · Vienna 4x22), audio and MIDI |
| **Data** | [matchmaker-benchmark-data](https://github.com/pymatchmaker/matchmaker-benchmark-data), fetched on demand |
| **Library** | [matchmaker](https://github.com/pymatchmaker/matchmaker) |
| **Metrics** | beat error, ms error, tracking rate, real-time factor |

## Submit a score follower

```bash
git clone https://github.com/pymatchmaker/matchmaker_benchmark.git
cd matchmaker_benchmark
conda env create -f environment.yml && conda activate matchmaker-benchmark
pip install "pymatchmaker @ git+https://github.com/pymatchmaker/matchmaker.git@main"

cp -r submissions/_template submissions/your-name
```

Edit `solution.py`. The whole required surface is a follower and one call:

```python
from matchmaker import register_method
from matchmaker.base import OnlineAlignment

class MyFollower(OnlineAlignment):
    def step(self, features) -> None:
        # one note or audio frame in; update self.current_index
        ...

register_method(
    "your-name",
    input_type="midi",                       # or "audio"
    build_follower=lambda mm: MyFollower(
        score_positions=mm.score_positions,
        queue=mm.stream.queue,
        reference_features=mm.reference_features,
    ),
    default_kwargs={"processor": "pitch", "piano_range": True},
)
```

`register_method` is a plain matchmaker function, so your submission is an
ordinary matchmaker plugin — the same file works outside this repository.

Check it in about thirty seconds, with no dataset:

```bash
python matchmaker_eval/validate_submission.py submissions/your-name --smoke
```

Then open a pull request. CI runs that same check; a maintainer reviews the
code; after merge it is evaluated and appears on the leaderboard.

| Read next | |
| --- | --- |
| [docs/submitting.md](docs/submitting.md) | The full walkthrough |
| [docs/submission-api.md](docs/submission-api.md) | What `mm` carries, audio submissions, the contract |
| [docs/eval-protocol.md](docs/eval-protocol.md) | Folds, metrics, ranking, and what "do not tune on the eval fold" means |
| [docs/maintaining.md](docs/maintaining.md) | Review, runners, data sources, publishing |

## How it works

```
 you              GitHub                    maintainer          GitHub Actions
  │                 │                            │                    │
  │  pull request   │                            │                    │
  ├────────────────►│  structure + contract      │                    │
  │                 ├──► smoke test on the ──────┤                    │
  │                 │    committed example       │ reads the code     │
  │                 │    (no data needed)        │ and merges         │
  │                 │                            ├───────────────────►│
  │                 │                            │        fetches its own data,
  │                 │                            │        runs 146 pieces in
  │                 │                            │        parallel shards,
  │                 │                            │        commits the leaderboard
  │                 │                            │                    │
  └──────────────── leaderboard on GitHub Pages ◄─────────────────────┘
```

Review is the trust boundary: CI executes submitted code, it does not sandbox
it. Everything after merge is automatic.

## Folds

The benchmark is defined by frozen lists in `data/folds/`, committed so that any
change to what it measures arrives as a reviewable diff.

| Fold | Pieces | Data branch | Use |
| --- | ---: | --- | --- |
| `example` | 1 | committed in `resources/` | smoke tests; no download |
| `tuning` | 20 | [`experiment`](https://github.com/pymatchmaker/matchmaker-benchmark-data/tree/experiment) | development and sweeps |
| `eval` | 146 | [`benchmark`](https://github.com/pymatchmaker/matchmaker-benchmark-data/tree/benchmark) | the leaderboard, and nothing else |

No performance appears in both `tuning` and `eval`; CI checks it on every pull
request (`make_folds.py --check`).

## Data

Audio and MIDI live in
[matchmaker-benchmark-data](https://github.com/pymatchmaker/matchmaker-benchmark-data)
and are fetched **file by file over HTTPS**, from the branch that fold lives on.
Nothing is cloned, and only what a run touches is downloaded.

```bash
# only the 20 tuning performances, from the `experiment` branch
python matchmaker_eval/fetch_data.py --fold tuning --input-type midi

# check the configuration without downloading a corpus
python matchmaker_eval/fetch_data.py --probe --fold eval --input-type audio
```

`run_submission.py` fetches what it needs on its own, so even that is optional.
Files already under `MATCHMAKER_DATA_DIR` (default `~/data`) are used as-is, so
a local copy of the corpora still works. The address is four lines in
[data/data_sources.yaml](data/data_sources.yaml).

## Running things

```bash
# one submission on a fold
python matchmaker_eval/run_submission.py submissions/<name> --fold tuning

# the built-in methods, as leaderboard reference rows, then republish
python matchmaker_eval/run_references.py --input-type midi
python matchmaker_eval/run_references.py --input-type audio

# split a fold across parallel runs, then recombine
python matchmaker_eval/run_submission.py submissions/<name> \
    --shard 0 --num-shards 8 --output results/shards/shard-0
python matchmaker_eval/merge_shards.py results/shards

# rebuild the published ranking and the per-piece detail behind it
python matchmaker_eval/leaderboard.py
python matchmaker_eval/export_details.py
```

The original per-dataset runners for the built-in methods are unchanged:

```bash
python matchmaker_eval/test_audio.py    --dataset asap --method arzt
python matchmaker_eval/test_symbolic.py --dataset asap --method hmm
```

Submissions and built-in methods go through the *same* `Matchmaker` call and the
*same* evaluation functions — `run_evaluation`, `check_tracking`,
`compute_event_pooled_summary`. `verify_equivalence.py` proves it by running one
built-in method through both paths and comparing every metric.

## Evaluation protocol

- **Primary metric**: beat error (performance → score)
- **Secondary metric**: ms error (score → performance)
- **Tracking**: 30-second segments, median beat error per segment; a piece is
  lost if 2 or more segments exceed the threshold (audio 1.0 beat, MIDI 0.5)
- **Aggregation**: event-pooled across tracked pieces
- **Ranking**: tracking rate first, then median beat error

Full detail, including fold hygiene and the anti-tuning policy, in
[docs/eval-protocol.md](docs/eval-protocol.md).

## Project structure

```
matchmaker_eval/
  eval.py                 — single-piece alignment functions (audio + symbolic)
  test_audio.py           — audio runner for the built-in methods
  test_symbolic.py        — symbolic runner for the built-in methods
  submission.py           — loads submissions/<name>/ and validates metadata
  run_submission.py       — evaluate one submission on one fold -> metrics.json
  run_references.py       — evaluate every built-in method, then republish
  validate_submission.py  — pull-request checks: structure, contract, smoke run
  merge_shards.py         — recombine parallel shards into one metrics record
  leaderboard.py          — metrics.json files -> leaderboard.json / .csv
  export_details.py       — per-dataset/per-piece detail + decimated paths
  folds.py, make_folds.py — fold definitions and their integrity checks
  fetch_data.py           — download only the files a fold needs
  verify_equivalence.py   — proves submissions are scored like built-in methods
  utils.py, verify_tracking.py, sweep.py
data/
  folds/                  — frozen fold definitions (example, tuning, eval)
  data_sources.yaml       — where performance data is fetched from
  builtin_methods.yaml    — the reference methods shown on the leaderboard
submissions/
  _template/              — copy-me skeleton
  baseline-constant-tempo/  — the no-information floor
  example-pitch-matcher/    — a worked example that actually listens
results/
  leaderboard.json/.csv   — the published ranking
  details/<name>.json     — per-dataset, per-piece and alignment-path detail
  submissions/<name>/     — per-submission metrics.json
docs/                     — contributor and maintainer documentation, and the
                            GitHub Pages leaderboard
.github/workflows/
  validate-submission.yml — pull-request checks (no secrets, read-only token)
  evaluate.yml            — post-merge evaluation in parallel shards
  pages.yml               — publishes the leaderboard
```

## Acknowledgments

This work has been supported by the Austrian Science Fund (FWF), grant agreement PAT 8820923 ("Rach3: A Computational Approach to Study Piano Rehearsals"). Additionally, this work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. NRF-2023R1A2C3007605).
