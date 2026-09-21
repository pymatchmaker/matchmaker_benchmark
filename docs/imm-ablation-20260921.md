# IMM design validation — 2026-09-21

TR is the primary outcome; tracked-only accuracy is secondary. Nine fixed ablations cover all six motion-model sets containing CV or CA, plus score gating, correlated observation error, output filtering, and hard-min path aggregation. No parameters or default model were selected from the 146 test performances.

## Full 146 results

| Configuration | TR | Tracked MeanAE (b) | MedAE (b) | ≤0.5b | ≤1.0b |
|---|---:|---:|---:|---:|---:|
| **SoftOLTW + IMM (Ours)** | **91.78% (134/146)** | 0.2680 | 0.0840 | 88.13% | 95.00% |
| CV only | 34.25% (50/146) | 0.3787 | 0.2701 | 72.98% | 94.33% |
| CA only | 89.73% (131/146) | 0.2732 | 0.0856 | 87.34% | 94.74% |
| w/o CV (CA+ZV) | 89.04% (130/146) | 0.2722 | 0.0856 | 87.37% | 94.81% |
| w/o CA (CV+ZV) | 47.26% (69/146) | 0.2999 | 0.1908 | 82.35% | **96.33%** |
| w/o ZV (CV+CA) | **91.78% (134/146)** | 0.2673 | 0.0839 | 88.20% | 95.02% |
| ZV allowed everywhere | 90.41% (132/146) | 0.2741 | 0.0863 | 87.51% | 94.91% |
| w/o correlated error | 89.04% (130/146) | 0.2743 | 0.0885 | 87.17% | 94.69% |
| w/o output IMM | 89.73% (131/146) | 0.2758 | 0.1000 | 87.54% | 94.98% |
| Hard-min path aggregation | 90.41% (132/146) | **0.2642** | **0.0833** | **88.31%** | 95.15% |
| SoftOLTW | 86.99% (127/146) | 0.2938 | 0.1000 | 86.57% | 94.33% |

The full-model row uses the identical reference trajectories as manuscript Table 1. Errors are event-pooled over each configuration's successful performances using `scripts/evaluate_and_report_146.py --native`. Every source run uses `matchmaker_eval/test_audio.py --no-plots`. Runtime is not compared across these concurrent and historical runs.

## What the evidence supports

- Combining CV and CA improves TR over the strongest tested single model: 134 versus 131 for CA alone (CV alone: 50). The three additional successes are Vienna 27, 37, and 41. On the 131 common successes, full IMM MeanAE is 0.2638 versus 0.2732, MedAE 0.0834 versus 0.0856, ≤0.5b 88.29% versus 87.34%, and ≤1b 95.10% versus 94.74%.
- ZV is not necessary for the observed TR. CV+CA tracks exactly the same 134 performances and slightly improves every reported beat-accuracy metric. These experiments do not establish a benefit from adding ZV. The frozen default remains unchanged; the test set is not used to select a replacement.
- Conditional on retaining all three modes, score-based ZV gating helps: allowing ZV everywhere reduces TR to 132, losing ASAP 16 and Vienna 27. On the 132 common successes, gated versus ungated MeanAE is 0.2642 versus 0.2741. The ASAP failure is a small threshold crossing: worst-window median error rises from 0.9862b to 1.0402b, not a catastrophic loss of position.
- Removing output IMM or correlated observation error reduces TR to 131 or 130. This separates output filtering from the retained path-conditioned Kalman lattice.
- Hard-min aggregation tracks 132 despite better own-tracked accuracy. On the 132 common successes, full soft aggregation has MeanAE 0.2637 versus 0.2642, MedAE 0.0832 versus 0.0833, ≤0.5b 88.34% versus 88.31%, and identical ≤1b 95.15%. Its lower own-tracked error reflects a different subset, not evidence of better matched-set precision.
- Removing CA leaves 69 successes. Their higher own-tracked 1b rate does not compensate for losing 65 tracked performances. On the same 69 performances, full IMM has MeanAE 0.1763 versus 0.2999 and ≤1b 97.16% versus 96.33%.

## Controlled implementation

Implementation commits: `5204643` (component ablations) and `9aa221b` (score gate). Benchmark diagnostics: `821f1a2`. No new method registrations were added.

Mode removal preserves residence times and redistributes exit rates across remaining destinations. CV and ZV retain zero motion process noise; CA retains its original white-jerk noise. CV-only is therefore a deletion ablation, not a claim about all possible CV Kalman filters. CA-only supplies an adaptive single-model comparator. ZV alone cannot follow moving score position.

Hard-min retains the existing acoustic/transition cost ratio and selects one minimum-cost incoming branch. It isolates soft aggregation, not a simultaneous change of all likelihood scales. The no-output-IMM condition retains IMM computation but emits DP position, so it is not a runtime ablation. The ungated condition uses the unchanged three-mode transition matrix everywhere.

Independent scalar tests verify input mixing, between-model covariance, correlated-error correction, likelihood weighting and fused covariance. Single-mode settings match ordinary Kalman recursion. The ablation switches preserve default behavior in 500 IMM and 400 follower steps; diagnostic inference also reproduced all 20 saved final validation trajectories byte for byte.

## Posterior analysis

`matchmaker_eval/eval.py` saves diagnostic TSV files after inference. Actual performance times are aligned with raw DP position, filtered position, ZV eligibility, and predictive/posterior mode probabilities; initial silence is excluded. A test checks saved priors and posteriors directly against the running filter. The existing `scripts/analyze_posterior_probabilities.py` summarizes these traces.

All 20 validation performances are included, including the failure. Of 117,629 active frames, 2,636 permit ZV, across 13 performances. Mean posteriors (CV, CA, ZV) are (0.7721, 0.2279, 0) outside these intervals and (0.6346, 0.2048, 0.1606) within them. These are model-use statistics, not ground-truth regime classification, posterior calibration, or transition-delay validation. The previous manuscript's contextual probability/delay table had no verified provenance for the final implementation and was replaced.

Both gated and ungated validation runs track the same 19/20 performances and receive identical DP observations. Gated versus ungated MeanAE is 0.1994 versus 0.2080, MedAE 0.0731 versus 0.0747, ≤0.5b 90.30% versus 89.54%, and ≤1b 95.92% versus 95.71%. Validation TR is tied; its evidence favors gating only on secondary accuracy.

## Sources and reproducibility

Canonical artifacts: `output/imm_ablation_146_20260921/{ablation_metrics.json,tracking_changes.json,ablation_table.md,protocol/}`. Dataset directories contain the native completion records, individual metrics, trajectories, ground truth, kwargs and logs. Pairwise common-subset reports are retained under the corresponding variant directories. The reference is `output/paper_reproduction_146_20260920/softoltw`; baseline trajectories come from `output/imm_refactor_verification_20260920/softoltw_no_imm`.

Three completed runs under `output/imm_ablation_20260921` were reused after source inspection and byte-identical reproduction: CV-only (48 paths), no output IMM (14), and no correlated error (14). Their remaining duplicate jobs were deliberately terminated; the complete canonical runs each contain 146 performances. Verification evidence is in `protocol/reproduction_checks/` and `protocol/reused_ablations.json`. Hard-min was fully reevaluated because the previous variant averaged tied minima, whereas the current variant retains one survivor.

The comparison design follows the distinction between motion-model comparisons and posterior interpretation in [Kolat et al. (2022)](https://doi.org/10.3390/s22010347) and [Lim et al. (2026, preprint), IV-A2 and IV-C1](https://arxiv.org/html/2609.13307v1). Score-dependent mode availability is analogous to [Kirubarajan et al. (2000)](https://ieee-aess.org/media/ground-target-tracking-variable-structure-imm-estimator), whose model sets depend on estimated position and road context. This analogy motivates gating; it does not prove that our particular gate or ZV mode is necessary.
