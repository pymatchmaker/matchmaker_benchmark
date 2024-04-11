#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Example of how to extract the performed time of the beats
from a match file.
"""
import partitura as pt
import numpy as np

from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment
from matchmaker import EXAMPLE_MATCH


if __name__ == "__main__":

    match_fn = EXAMPLE_MATCH

    # Load performance, score and alignment info
    # from match file. The performed MIDI files are aligned with
    # the audio, so the alignment also corresponds
    # to the alignment in the audio.
    perf, alignment, score = pt.load_match(
        filename=match_fn,
        create_score=True,
    )

    # Get a structured numpy array of notes in the
    # score and notes in the performance
    pnote_array = perf.note_array()
    snote_array = score.note_array()

    # Get interp1d object that map time in the performance
    # in seconds to time in the score in beats and the 
    # other way around
    ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
        ppart_or_note_array=pnote_array,
        spart_or_note_array=snote_array,
        alignment=alignment,
    )

    # To get, e.g., the times of the beats in the
    # performance
    start_beat = np.ceil(snote_array["onset_beat"].min())
    end_beat = np.floor(snote_array["onset_beat"].max())

    # beats in the score
    beats = np.arange(start_beat, end_beat + 1)

    performed_beat_times = stime_to_ptime_map(beats)


