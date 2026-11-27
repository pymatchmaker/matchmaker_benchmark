import numpy as np
from typing import Tuple, Set, List
from matchmaker import register_method
from matchmaker.base import OnlineAlignment
from matchmaker.features.midi import PitchProcessor

PIANO_OFFSET = 21  # MIDI pitch of A0; piano-range features are pitch - 21

ROW = 1
COLUMN = 2
BOTH = 3

class OLTWDixon(OnlineAlignment):
    def __init__(
        self,
        score_positions,
        chord_pitches,
        queue=None,
        search_width=50,
        max_run_count=50,
        **kwargs,
    ):
        super().__init__(score_positions=score_positions, queue=queue, **kwargs)
        self.chord_pitches = chord_pitches
        self.search_width = search_width
        self.max_run_count: int = max_run_count
        self.score_len: int = len(self.chord_pitches)
        self.perf_inx: int = -1
        self.run_count: int = 0
        self.score_inx: int = 0 
        self.best_cell: Tuple[int, int] = (0,0)
        self.previous: int = 0
        self.x: List[int] = []
        self.cost_matrix: List[np.ndarray] = []
        self.path_lengths: List[np.ndarray] = []

    def distance(self, p, s):
        # {dist1}
        # return 0 if self.chord_pitches[s] & self.x[p] else 1

        # {dist2}
        distances = []
        for pitch in self.x[p]:
            distances.append(0 if pitch in self.chord_pitches[s] else 1)
        return np.mean(distances)

    def eval_path_cost(self, p, s):
        if p == 0 and s == 0:
            self.cost_matrix[p][s] = self.distance(p, s)
            return

        indices = [(p-1,s-1), (p-1,s), (p,s-1)]
        iinx = np.argmin([
            self.cost_matrix[p-1][s-1] if p > 0 and s > 0 else np.inf,
            self.cost_matrix[p-1][s] if p > 0 else np.inf,
            self.cost_matrix[p][s-1] if s > 0 else np.inf,
        ])
        min_inx = indices[iinx]

        # {path_len1}
        path_len = 1

        # {path_len2}
        # path_len = 1 if iinx > 0 else np.sqrt(2)

        self.cost_matrix[p][s] = self.distance(p, s) + \
            self.cost_matrix[min_inx[0]][min_inx[1]]
        self.path_lengths[p][s] = self.path_lengths[min_inx[0]][min_inx[1]] + path_len

    def cost_norm(self, p, s):
        return self.cost_matrix[p][s] / self.path_lengths[p][s]

    def get_inc(self, p, s):
        w = self.search_width
        min_row = min(
            range(p, max(-1, p - w), -1), # reversed to make (p,s) win ties
            key=lambda r: self.cost_norm(r,s)
        )
        min_col = min(
            range(s, max(-1, s - w), -1), # reversed to make (p,s) win ties
            key=lambda c: self.cost_norm(p,c)
        )
        r,c = (
            (min_row,s)
            if self.cost_norm(min_row,s) < self.cost_norm(p,min_col)
            else (p,min_col)
        )
        self.best_cell = (r,c)

        if s == self.score_len - 1:
            return ROW
        if p < self.search_width:
            return BOTH
        if self.run_count > self.max_run_count:
            if self.previous == ROW:
                return COLUMN
            else:
                return ROW

        # This seems to be the wrong way around in the paper's pseudo code?
        if r < p:
            return COLUMN
        elif c < s:
            return ROW
        else:
            return BOTH

    def step(self, features) -> None:
        perf_pitches: Set[int] = {int(p) for p in np.atleast_1d(features)}
        if not perf_pitches:
            return
        self.x.append(perf_pitches)
        self.cost_matrix.append(np.full(self.score_len, np.inf))
        self.path_lengths.append(np.ones(self.score_len))

        # --- new row ---
        self.perf_inx += 1
        for score_k in range(
            max(0, self.score_inx - self.search_width + 1),
            self.score_inx + 1
        ):
            self.eval_path_cost(self.perf_inx, score_k)

        if self.previous != ROW:
            self.run_count = 0
        self.previous = ROW
        self.run_count += 1

        # --- new col ---
        while (mode := self.get_inc(self.perf_inx, self.score_inx)) != ROW:
            self.score_inx += 1
            for perf_k in range(
                max(0, self.perf_inx - self.search_width + 1),
                self.perf_inx + 1
            ):
                self.eval_path_cost(perf_k, self.score_inx)

            if self.previous != COLUMN:
                self.run_count = 0
                self.previous = COLUMN
            self.run_count += 1

            if mode == BOTH:
                # next row directly after one column 
                self.run_count = 0
                self.previous = BOTH
                break
        
        self.current_index = min(self.best_cell[1], len(self.score_positions)-1)

    def get_current_position(self) -> float:
        return float(self.score_positions[self.current_index])

def build_processor(mm):
    return PitchProcessor(piano_range=True, return_pitch_list=True)

def build_reference(mm):
    """Pitches sounding at each score onset, ordered like mm.score_positions."""
    note_array = mm.score_part.note_array()
    onsets = note_array["onset_beat"]
    pitches = note_array["pitch"] - PIANO_OFFSET
    return [set(pitches[onsets == onset]) for onset in np.unique(onsets)]

def build_follower(mm):
    return OLTWDixon(
        score_positions=mm.score_positions,
        chord_pitches=mm.reference_features,
        queue=mm.stream.queue,
        reference_features=mm.reference_features,
    )

register_method(
    "miles-davis",
    input_type="midi",
    build_follower=build_follower,
    default_kwargs={"processor": "pitch", "piano_range": True},
    build_processor=build_processor,
    build_reference=build_reference,
)
