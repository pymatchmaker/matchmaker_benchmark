# Evaluation protocol

What the benchmark measures, on which data, and what "do not tune on the eval
fold" means in practice.

## Folds

The benchmark is defined by three CSV files in `data/folds/`. Each row is one
(score, performance, ground truth) triple. The lists are committed, so any
change to what the benchmark measures arrives as a reviewable diff instead of a
silent shift in the leaderboard.

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
and checks them; `--check` verifies the committed files without rewriting them
(CI runs this on every pull request).

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

Alignment is scored in both directions. The **beat error** is primary.

| Metric | Direction | Meaning |
| --- | --- | --- |
| `beat_median`, `beat_mean` | performance → score | Absolute error in beats, at each ground-truth performance onset. The headline number. |
| `beat_0.3b`, `beat_0.5b`, `beat_1.0b` | performance → score | Share of onsets within that many beats. |
| `ms_median`, `ms_300ms` | score → performance | Absolute error in milliseconds. Secondary; comparable to the score-following literature. |
| `rtf` | — | Real-time factor: processing time / performance duration. Must be well under 1.0 to be usable live. |
| `tracking_rate` | — | Share of pieces followed to the end (see below). |

Accuracy is **event-pooled**: errors from every onset of every tracked piece go
into one distribution, so long pieces weigh more than short ones. `rtf` and
latency are averaged per piece instead.

### What a follower is told

By default a follower is told nothing about the performance it is about to
hear. It gets the score, and the stream. The notated tempo comes from the score
like any other marking; it is not a measurement of the recording.

A follower may instead be given the performance's **estimated tempo**
(`estimated_bpm`) — a measurement of the recording, published as a column in the
data repository's metadata. This is a materially easier task, so it is the
exception and it is always visible:

- the entry declares it — `estimated_bpm: true` in a submission's
  `metadata.yaml`, or in `data/builtin_methods.yaml` for a reference;
- the run records it, and the leaderboard marks the row with an asterisk and a
  footnote.

Declaring it and then not having the data available fails the run rather than
quietly falling back to the notated tempo: a row labelled as having the tempo
must actually have had it.

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
| audio | 1.0 beat |
| MIDI | 0.5 beat |

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

## Reproducing a leaderboard row

Every `metrics.json` records the matchmaker version, the benchmark commit, and
the SHA-256 of the fold file it ran against. To reproduce a row:

```bash
git checkout <benchmark_commit>
pip install -r submissions/<name>/requirements.txt
python matchmaker_eval/run_submission.py submissions/<name> --fold eval
```

If the fold hash in your run differs from the one in the published record, the
fold changed since — the numbers are not comparable.
