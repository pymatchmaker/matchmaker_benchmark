# Running the manuscript diagnostics

In Actions, choose **Evaluate references**, set `fold` to `diagnostics`, and
use four shards. Select the branch containing this workflow. No leaderboard
files are changed by this run. `benchmark-results-diagnosis` contains all raw
paths, ground truth, per-case metrics, logs, input hashes and a merged summary.
The merge fails if any case or shard of a selected configuration is missing,
duplicated or failed. Leave `only` empty and `input_type=both` for all suites;
set `only=dannenberg` and `input_type=midi` for its 34 perturbation conditions only.

For non-piano evaluation use the same workflow with `fold=beyond_piano`,
`input_type=audio`. Its artifact is `benchmark-results-beyond_piano`.

Diagnostic inputs are pinned under `frozen_sources.diagnostics` in
`data/data_sources.yaml`. The runner verifies
the manifest hash and each selected input, including losslessly encoded audio.
The source data release documents the original generator scripts and licenses.
No regeneration with an unrecorded soundfont occurs in CI.

The diagnostic configurations are in `matchmaker_eval/diagnostics.py`:

- Perturbations: four audio and seven MIDI followers on 41/34 conditions.
- Repeat examples: Arzt, Multi-Fold Arzt and OPHMM on two K.282 excerpts.
- Folded Batik: Dixon, Arzt, SKF, OPHMM and Multi-Fold Arzt on 24 test movements.
- Global skip: symbolic OPHMM on two K.279 conditions, changing only its
  skip-from distribution to disable the global transition.

Audio OPHMM uses `s_j=1e-100` for the synthetic diagnostic suites, where jumps
are part of the experiment. The folded-Batik OPHMM configuration uses `s_j=1e-5`, as in the original
folded-score diagnostic protocol. The play-through baseline registry uses zero.
Nominal tempo is provided only for the two K.282 illustrative cases (60 BPM).
The synthetic perturbations and recorded folded cases receive the normal
score-based tempo initialization, not the performance's estimated tempo.

All suites use the same `run_evaluation`, causal onset sampling, 3 kHz SPARC
implementation, and musical-beat tracking criterion as the main benchmark.
Accuracy remains in score denominator-note beats. The input manifest specifies
whether the score is unfolded. Repeat ground truth retains backward jumps.
The perturbation summary averages per-condition MeanAE over tracked cases;
raw per-case paths also support pooled accuracy and common-tracked analyses.
Timing measured on these runners is retained separately from previously
reported RTF. A new runner measurement must not silently replace an earlier
measurement or be selected because it is smaller.

Local reproduction:

```sh
python matchmaker_eval/diagnostics.py run \
  --config perturbations-audio-arzt --shard 0 --shards 4 --output results/diagnostics/arzt-0
```

Use the library revision installed by the workflow. Artifacts record both
library and benchmark commits and the GitHub run ID. A raw-path re-score from
an old local experiment is not equivalent to a CI execution of a follower.

## Offline reference

Set `fold=offline` (eight shards) to run the offline reference on the piano
and beyond-piano folds. `benchmark-results-offline` includes the merged
results and original shards. The offline pipeline is migrated from the saved
server script, using its verified synctoolbox 1.3.2 environment, 50 Hz features,
22,050 Hz onset filterbank and DTW weights [1.5, 1.5, 2.0]. It is not an online
follower. These results are archived, never inserted into the online leaderboard.

The archived offline numerical environment uses NumPy 1.26.4, SciPy 1.14.1,
pandas 2.2.3, librosa 0.10.2, numba 0.64.0 and libfmp 1.3.0, verified on the
original server. Synctoolbox 1.3.2's old dependency upper bounds do not describe
that working environment. CI installs the two pure-Python toolbox packages
without their obsolete dependency pins, then runs the example alignment as a
preflight. The exact installed package list is saved in each artifact. This
compatibility setup is confined to offline reference jobs.

## Frozen AMT outputs

Set `fold=amt_reference` (four shards) for the 146 piano transcriptions with
OLTWArzt, PTHMM and SLT_OLTW. The source MIDI is public and pinned by
`data/data_sources.yaml` under `frozen_sources.amt`. CI executes the symbolic
followers, not the private AMT model.
It uses the original audio ground truth and the audio TR tolerance. The saved
MIDI timestamps are unchanged. New runner timings measure the follower only
and must not replace end-to-end AMT RTF in the manuscript.
The artifact is `benchmark-results-amt-reference`.
