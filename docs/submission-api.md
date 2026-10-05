# Submission API

The contract between a submission and the benchmark.

The short version: your `solution.py` calls `matchmaker.register_method()` once.
That is a plain matchmaker call, not a benchmark concept — the same file works
outside this repository, and the benchmark runs your follower through
`Matchmaker` with no adapter in between. Everything the built-in methods get,
you get.

## The shape of a submission

```
submissions/<name>/
    solution.py       registers exactly one method
    metadata.yaml     validated at pull-request time
    requirements.txt  optional, pinned
```

`<name>` is lowercase letters, digits, `-`, `_` or `.`, and becomes the
leaderboard's identifier. Register your method under the same name. Directories
starting with `_` are templates and are never evaluated.

## `register_method`

```python
from matchmaker import register_method
from matchmaker.base import OnlineAlignment

class MyFollower(OnlineAlignment):
    def step(self, features) -> None:
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

| Argument | Required | What it does |
| --- | --- | --- |
| `name` | yes | The method name. Use your directory name. |
| `input_type` | yes | `"midi"` or `"audio"`. Must match `metadata.yaml`. |
| `build_follower(mm)` | yes | Returns the `OnlineAlignment` for one piece. |
| `build_processor(mm)` | no | Only for a processor of your own; naming a standard one in `default_kwargs` covers most cases. |
| `build_reference(mm)` | no | Score-side features. Defaults to the note array. |
| `default_kwargs` | no | Stream and feature settings; also reaches your follower as `mm.config`. |

The hooks are called once per piece, in the order processor → reference →
follower. Keep per-piece state on the follower, never in module globals — the
same registration is reused for all 146 pieces.

### `default_kwargs`

| Key | Input | Meaning |
| --- | --- | --- |
| `processor` | both | Standard processor name: `pitch`, `pianoroll`, `chord_onset`, `pitchclass` (MIDI); `chroma`, `mfcc`, `cqt`, `lse`, … (audio). |
| `piano_range` | MIDI | Restrict to the 88-key range. |
| `polling_period` | MIDI | Seconds per observation window. `None` = one observation per MIDI message. Default 0.01. |
| `sample_rate` | audio | Stream sample rate. Default 44100. |
| `frame_rate` / `hop_length` | audio | Frames per second, or the hop directly. |

## What `mm` carries

`build_*` receives the `Matchmaker` running the piece:

| Attribute | What it is |
| --- | --- |
| `mm.score_part` | The unfolded, merged partitura `Part` |
| `mm.score_positions` | Ascending score beat of every note onset — the follower's states |
| `mm.reference_features` | Score-side features (available from `build_follower` on) |
| `mm.tempo` | Notated tempo in BPM, or 120 if the score has no marking |
| `mm.frame_rate` | Frames per second (1 for MIDI) |
| `mm.sample_rate`, `mm.hop_length` | Audio stream settings |
| `mm.config` | Whatever `default_kwargs` supplied |
| `mm.stream.queue` | The stream's queue — pass it to your follower |
| `mm.ref_frame_to_beat()` | Score beat of each reference *frame* (audio) |
| `mm.processor` | The processor, once built |

**What is deliberately absent** is a way to read the performance ahead of time.
Observations arrive one at a time through the queue, because that is what online
score following means. A follower that reads the whole performance up front is
not solving this problem.

## The follower

Subclass `matchmaker.base.OnlineAlignment` and implement `step(features)`.
The base class supplies:

| | |
| --- | --- |
| `self.current_index` | Index into `score_positions`; your `step()` updates it |
| `self.current_perf_time` | Seconds since the performance started, set before each `step()` |
| `self.score_positions` | Ascending score beats |
| `self.reference_features` | Whatever `build_reference` returned |
| `self.get_current_position()` | Defaults to `score_positions[current_index]`; override to report between states |
| `self.alignment_path` | Accumulated `(2, T)` array of (perf seconds, score beats) — do not touch |
| `self.is_still_following()` | `current_index < len(score_positions) - 1` |

Pass `queue=mm.stream.queue` and `score_positions=mm.score_positions` when you
construct it — without the queue nothing feeds it, and without the positions the
base class cannot tell when the score has ended.

Full details in
[HOW_TO_MAKE_CUSTOM_SCORE_FOLLOWERS.md](https://github.com/pymatchmaker/matchmaker/blob/main/HOW_TO_MAKE_CUSTOM_SCORE_FOLLOWERS.md).

## Audio submissions

Audio followers usually align against a synthesised rendering of the score
rather than against the note array:

```python
import numpy as np
from matchmaker import register_method
from matchmaker.utils.misc import generate_score_audio


def build_reference(mm):
    audio = generate_score_audio(mm.score_part, mm.tempo, mm.sample_rate)
    features, _ = mm.processor((audio.astype(np.float32), 0.0))
    mm.processor.reset()   # the same processor then handles the live input
    return features


def build_follower(mm):
    return MyAudioFollower(
        reference_features=mm.reference_features,
        score_positions=mm.score_positions,
        queue=mm.stream.queue,
        frame_rate=mm.frame_rate,
        ref_frame_to_beat=mm.ref_frame_to_beat(),
    )


register_method(
    "your-name",
    input_type="audio",
    build_follower=build_follower,
    build_reference=build_reference,
    default_kwargs={"processor": "chroma", "sample_rate": 44100, "frame_rate": 30},
)
```

## How a submission is run

There is no submission-specific evaluation path. Per piece:

1. `run_submission.py` imports `solution.py`, which registers your method.
2. It builds `Matchmaker(method=<your name>)` — the same call
   `test_symbolic.py` and `test_audio.py` make for a built-in method.
3. The stream plays the performance into your follower's queue.
4. `run_evaluation` (in `eval.py`) scores `alignment_path` against the ground
   truth, and `check_tracking` decides whether the piece counts as tracked.

`verify_equivalence.py` checks this claim by re-registering a built-in
follower under a new name and confirming both paths produce identical metrics.

A piece that raises, or exceeds its 15-minute budget, is recorded as a lost
piece with the error message, and the run continues.
