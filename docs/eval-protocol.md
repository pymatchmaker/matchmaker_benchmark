# Evaluation protocol

What the benchmark measures, on which data, and what "do not tune on the eval
fold" means in practice.

## Folds

The benchmark is defined by three CSV files in `data/folds/`. Each row is one
(score, performance, ground truth) triple, together with the two measurements
published about that performance — `difficulty` and `estimated_bpm`. The lists
are committed, so any change to what the benchmark measures arrives as a
reviewable diff instead of a silent shift in the leaderboard, and a run reads
nothing the fold did not name.

| Fold | Rows | Contents | Use it for |
| --- | --- | --- | --- |
| `example` | 1 | The piece committed under `resources/` | Smoke tests. No download needed. |
| `valid` | 20 | ASAP, Batik, Vienna4x22 | Development, hyper-parameter sweeps, model selection. |
| `eval` | 146 | ASAP (32), Batik (30), Vienna4x22 (84) | The leaderboard. Nothing else. |

The `example` fold is a member of the `valid` fold — the same score and the
same performance — so smoke-testing a submission never touches eval data.

The audio and MIDI files themselves are not in this repository; they are far too
large to version. They live in a separate data repository and are fetched on
demand — the address is in `data/data_sources.yaml` — into
`MATCHMAKER_DATA_DIR` (default `~/data`):

```bash
python matchmaker_eval/fetch_data.py --fold valid --input-type midi
```

Only the files the fold actually names are fetched, and anything already on disk
is left alone — so a local copy of the original datasets is used as-is. **A
contributor never has to run this**: the pull-request check uses the committed
`example` piece, and the post-merge evaluation fetches what it needs on GitHub's
runners.

`matchmaker_eval/make_folds.py` regenerates the folds from `data/metadata-*.csv`
and checks them; `--from-repo` instead rebuilds them from the data repository's
own manifests, which is where the paths and the `estimated_bpm` values come
from. `--check` verifies the committed files without rewriting them (CI runs
this on every pull request).

### The audio is not recorded the same way across datasets

Worth knowing before reading any audio result. The three datasets' recordings
have three different origins:

| Dataset | Recordings |
| --- | --- |
| ASAP | MAESTRO concert recordings — many pianists, instruments and halls |
| Batik | The corpus MIDI replayed on a Disklavier reproducing piano at JKU and recorded there |
| Vienna 4x22 | The original corpus recordings |

So an audio follower's score on Batik is measured on a reproducing-piano
rendering with uniform acoustics and audio that is synchronous with the MIDI by
construction, while its ASAP score is measured on varied concert recordings.
The comparison between methods on the same dataset is sound; a comparison of
one method's numbers *across* datasets is partly a comparison of recording
conditions.

This does not affect MIDI results, where the performance data is identical in
all three cases.

### What the split guarantees, and what it does not

**Guaranteed: no performance is in both folds.** `make_folds.py --check` fails
if one is, so a submission tuned on the validation fold has never seen an eval-fold
recording.

**Not guaranteed: score disjointness.** Ten scores appear in both folds, played
by different pianists — six in ASAP, four in Vienna4x22. This is inherent to the
data: Vienna4x22 consists of 4 pieces played by 22 pianists each, so a
score-disjoint split would delete the dataset. Tuning therefore sees these
*scores*, never these *performances*.

This matters when you read the leaderboard: a follower that memorised score
structure from the validation fold has a small edge on those ten scores. It does not
matter for the comparison between submissions, since every submission is offered
the same deal.

## Metrics

The benchmark uses matchmaker's evaluation result format directly and adds tracking verdicts and aggregates. **Beat error** is primary.

| Metric | Direction | Meaning |
| --- | --- | --- |
| `mean_b`, `median_b`, `std_b` | performance → score | Mean, median, and standard deviation of absolute score-position error, in beats. |
| `accuracy_0.05b`, `accuracy_0.1b`, `accuracy_0.3b`, `accuracy_0.5b`, `accuracy_1b`, `accuracy_2b` | performance → score | Fraction of annotated onsets within each beat-error tolerance; the same flat fields are used in JSON and CSV. |
| `skewness`, `kurtosis` | — | Unitless statistics of the signed score-position error distribution. |
| `smoothness` | — | Trajectory smoothness computed with SPARC; higher (less negative) is smoother. |
| `rtf` | — | Processing time divided by the annotated performance span. |
| `feature_latency_ms`, `inference_latency_ms` | — | Mean feature-extraction and inference time per frame, in milliseconds (audio). |
| `tracking_rate` | — | Share of pieces that pass the tracking check (see below). |

Accuracy is **event-pooled**: errors from every onset of every tracked piece go
into one distribution, so long pieces weigh more than short ones. `rtf` and
latency are averaged per piece instead.

### Smoothness (SPARC)

Matchmaker's `evaluate_alignment()` returns `smoothness`, calculated using SPARC (Spectral Arc Length); the benchmark keeps that name and value.
Positions are resampled at 50 Hz, holding the last reported position between updates.
Windows start at ground-truth performance onsets, following the TR window policy; their duration is configurable and is set to 30 seconds for this benchmark.
Short pieces use one whole-piece window, and the final window is clipped at the last annotated onset.
Each piece averages its valid window values, and the headline score averages these per-piece values over tracked pieces only.
Fully stationary or otherwise undefined windows are skipped; `smoothness` is `null` when no window can be scored.
Compare smoothness only within the same input type (audio or MIDI), since position-reporting patterns differ between modalities.

Per-piece `performance_bpm` comes from the fold's `estimated_bpm` column and is used only to display approximate ms equivalents of beat errors; the run-level `estimated_bpm` remains the boolean indicating whether the follower received the tempo.

### What a follower is told

By default a follower is told nothing about the performance it is about to
hear. It gets the score, and the stream. The notated tempo comes from the score
like any other marking; it is not a measurement of the recording.

A follower may instead be given the performance's **estimated tempo**
(`estimated_bpm`) — a measurement of the recording, frozen as a column of the
fold CSV beside the paths it belongs to, so `fold_sha256` covers it: two runs
reporting the same fold hash were handed the same tempi. This is a materially
easier task, so it is the exception and it is always visible:

- the entry declares it — `estimated_bpm: true` in a submission's
  `metadata.yaml`, or in `data/builtin_methods.yaml` for a reference;
- the run records it, and the leaderboard marks the row with an asterisk and a
  footnote.

A piece whose fold row carries no tempo still runs: matchmaker falls back to
the score's notated marking, then to 120 BPM, which is what every entry that
never asked for a tempo already uses. The run says so — per piece as it goes,
and once in the summary — and records how many pieces actually had one in
`metrics.json` as `n_estimated_bpm`, so a partly-supplied run is visible rather
than assumed. That only arises on a fold CSV written before the column existed,
or by hand; regenerate it with `make_folds.py --from-repo`.

The fallback keeps the run alive; it does not make the result equivalent. On
the example piece, `pfkorz` goes from a tracked `median_b` of 0.067 with the
measured tempo to a lost piece at 39.0 without it, because the notated marking
is nowhere near the played tempo. Read a run with a non-zero shortfall as a
mixture of two conditions, not as a clean measurement.

The value is one per performance, not one per container: it is score beats
divided by how long the playing took, and the mp3 and the MIDI of a performance
are two renderings of the same playing. A follower's audio and MIDI runs are
therefore given the same number.

Among the reference methods only the Korzeniowski particle filter (`pfkorz`)
uses it, because it tracks tempo as part of its state.

A submission using it is not disqualified and is not ranked separately — it
sits in the same table, marked. Readers can then judge the comparison for
themselves, which they cannot do if the difference is invisible.

### Tracked vs. lost

A follower that loses the performance produces meaningless errors, so pieces are
classified first. The performance is walked in onset-wise forward windows of 30
seconds — from the first annotated onset through the first window that reaches
the last one. A window fails if the median absolute beat error inside it exceeds
the threshold, and the piece counts as **lost** if any evaluated window fails.

| Input | Threshold |
| --- | --- |
| audio | 1.0 musical beat |
| MIDI | 0.1 musical beat |

A musical beat contains three denominator-note units in compound time
signatures (numerators 6, 9, 12, and larger multiples of three), and one in
other meters. Simple triple meters such as 3/4 and 3/8 are not grouped.
At every reference onset, the absolute position error is divided by the factor
for the time signature at that reference score position **before** computing
the window median. This uses the follower's actual unfolded score, retaining
meter changes within a performance, even inside a single evaluation window.
The factor follows the reference position, not a potentially incorrect estimate.

Thus the native-coordinate tolerance in a compound passage is 3.0
denominator-note beats for audio and 0.3 for MIDI. The output paths, ground
truth, MeanAE, MedAE, standard deviations, and tolerance accuracies retain
denominator-note units; only TR uses musical beats. Each run saves a
`meter_<index>.json` alongside its full paths so the criterion can be reproduced.
Scores without a time signature use the parser's 4/4 default, recorded as
`assumed_time_signature` in that file.

Accuracy metrics are reported over tracked pieces only. A piece that crashes or
times out counts as lost.

### Ranking

Rows are ordered by **tracking rate first, median beat error second**.

Ranking on accuracy alone would reward a follower that gives up on every hard
piece, because accuracy is only measured where it stayed with the performance.
Tracking rate is the primary key for that reason, and the accuracy columns
describe how well a submission does on the pieces it did track.

## Tuning hygiene

**The eval fold must not be used to develop, tune, select or validate a
submission.** Every submission declares this in `metadata.yaml`:

```yaml
eval_fold_untouched: true
```

That includes the indirect forms, which are the ones people fall into by
accident:

- running the eval fold repeatedly and keeping the variant that scored best —
  that is model selection on the test set, even if no gradient touched it;
- choosing hyper-parameters, features or thresholds by eval-fold score;
- training on any corpus that contains the eval-fold recordings (ASAP, Batik and
  Vienna4x22 are all public, so check what your pre-trained components saw);
- reading per-piece eval results to decide what to fix next.

Use the `valid` fold for all of that, or your own data.

### Why this is on the honour system

The eval fold is public — its contents are in this repository and its audio is
downloadable. We cannot verify a declaration at submission time, and we are not
going to pretend otherwise. What we can do:

1. **Reproduce.** Maintainers periodically re-run submitted followers, including
   under sweeps on the validation fold.
2. **Compare the folds.** A follower that does much better on eval than on
   tuning, beyond what the difficulty difference explains, gets a closer look.
   This is a weak signal — the folds differ in size and content — so it starts a
   conversation, it does not decide anything.
3. **Read the code.** Every submission is reviewed before merge.

A submission found to have been tuned on the eval fold is removed from the
leaderboard, and the reason is recorded. If you realise after submitting that
you crossed the line, say so — an honest correction costs a resubmission, not a
reputation.

### The stronger fix, if we need it

Honour-system integrity holds as long as the stakes are low. If the leaderboard
starts to matter — a paper deadline, a prize — the answer is a **held-out fold**
whose audio is never published, evaluated by maintainers on request. The fold
mechanism is already generic (`data/folds/*.csv`, `--fold <name>`), so adding
one is a matter of data hosting and policy, not code. It is not in place today.

## The evaluation key

Two rows are only comparable when everything that decides their numbers,
apart from the follower itself, was the same. Every run records those
conditions and hashes them into one **evaluation key**, e.g. `ek1-3f9c0a1b2d4e`:

| Component | What it pins |
| --- | --- |
| `fold_sha256` | The fold CSV: which pieces, their paths, `estimated_bpm` |
| `data_sha256` | The bytes of every file a piece was scored with: the score, the performance that was streamed, the ground truth |
| `scoring_sha256` | The benchmark's scoring code: `eval.py`, `utils.py`, `beat_units.py`, `verify_tracking.py`, and the thresholds and per-piece scoring in `run_submission.py` |
| `matchmaker_sha256` | The installed matchmaker sources: ground-truth resolution, beat errors, tracking windows, SPARC, and the reference followers |
| `partitura` | The version that parses every score and match file |

They are content hashes, not commit ids, on purpose: a results commit or a
README edit changes a commit but not a number, while any change to what is
read or how it is scored changes the key. The `ek1` prefix names the scheme; if
the set of components ever changes, so does the prefix.

MIDI and audio runs stream different performance files, so their keys always
differ in `data_sha256`: each input type has its own current key. The current
conditions are set by the most recent run. Its code components (`fold_sha256`,
`scoring_sha256`, `matchmaker_sha256`, `partitura`) apply to every input type,
and each input type's key is that of its most recent run under them. The
leaderboard ranks a row only when it has the current key of its input type, so
every ranked row was scored by the same code, and rows of one input type also
on the same data. A row with another key — from before a scoring fix, a data
correction or a fold change — is listed below the table as scored under
earlier conditions, with the components that differ, and rejoins the ranking
once it is re-run.

### Citing results

Every published key keeps its table: `results/archive/<key>/leaderboard.json`
(and `.csv`), the ranked entries of that key and so of one input type, also
served on the site at `archive/<key>/leaderboard.json`, with
`results/archive/index.json` listing each key, its input type, when it was
first and last published, and its components. A key's archive is updated while it is current
and frozen once a newer key takes over.

So cite a number together with the key of its input type, e.g. *"tracking rate 0.84 (benchmark
evaluation key `ek1-3f9c0a1b2d4e`)"*. Anyone can then find the exact table it
came from, and see whether a later leaderboard was scored under the same
conditions or not. The per-piece records of a key are in the repository's
history at the `benchmark_commit` its rows record.

## Reproducing a leaderboard row

Every `metrics.json` records the matchmaker version and commit, the benchmark
commit, the SHA-256 of the fold file it ran against, and the evaluation key
with its components. To reproduce a row:

```bash
git checkout <benchmark_commit>
pip install -r submissions/<name>/requirements.txt
python matchmaker_eval/run_submission.py submissions/<name> --fold eval
```

If the evaluation key of your run differs from the one in the published record,
something that decides the numbers changed since; its components say what.
