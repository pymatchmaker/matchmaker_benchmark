# Submissions

One directory per score follower:

```
submissions/your-name/
    solution.py       your follower (required)
    metadata.yaml     who wrote it, how it was tuned (required)
    requirements.txt  extra pip dependencies (optional)
```

`solution.py` defines a `matchmaker.base.OnlineAlignment` subclass and registers
it with `matchmaker.register_method()`. Nothing in it is benchmark-specific, so
the same file works with plain matchmaker too.

Start from the template:

```bash
cp -r submissions/_template submissions/your-name
```

Then read [docs/submitting.md](../docs/submitting.md) for the full walkthrough
and [docs/submission-api.md](../docs/submission-api.md) for the API.

Check your work before opening a pull request — this needs no dataset download:

```bash
python matchmaker_eval/validate_submission.py submissions/your-name --smoke
```

Directories starting with `_` are templates and are not evaluated.

## What is here

| Directory | What it is |
| --- | --- |
| `_template/` | Copy-me skeleton with a commented walkthrough. |
| `baseline-constant-tempo/` | The no-information floor: reads the score position off the clock without listening. Every real follower should beat it. |
| `example-pitch-matcher/` | A worked example that actually listens: matches each note against the next couple of score chords and coasts on the estimated tempo in between. Read this one first. |
