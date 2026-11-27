## Submission

<!-- Delete this section if your pull request is not a submission. -->

- **Name:**
- **Input type:** midi / audio
- **What is new about it:** <!-- one or two sentences -->

### Checklist

- [ ] The submission lives in one new directory, `submissions/<name>/`, and this
      pull request changes nothing else.
- [ ] `metadata.yaml` is filled in, including `eval_fold_untouched: true`.
- [ ] `requirements.txt` pins exact versions (or is empty).
- [ ] `python matchmaker_eval/validate_submission.py submissions/<name> --smoke`
      passes locally.
- [ ] I tuned on the **valid** fold (or my own data) and did not use the eval
      fold to develop, tune, select or validate this score follower —
      see [docs/eval-protocol.md](../docs/eval-protocol.md).
- [ ] The code is mine to submit under the repository's licence, or I have said
      in `metadata.yaml` where it comes from.
