import argparse
from pathlib import Path

import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import partitura as pt

WORKING_DIR = Path(__file__).parent.parent
DATASET_DIR = {
    "asap": Path("~/Documents/Datasets/datasets/asap-dataset-matchmaker").expanduser(),
    "batik": Path("~/datasets/batik_plays_mozart").expanduser(),
    "vienna": Path("~/Documents/Datasets/datasets/vienna4x22").expanduser(),
}
METADATA_PATH = {
    "valid": WORKING_DIR / "data/metadata-validation.csv",
    "asap": WORKING_DIR / "data/reduced/metadata-asap.csv",
    "batik": WORKING_DIR / "data/reduced/metadata-batik.csv",
    "vienna": WORKING_DIR / "data/reduced/metadata-vienna.csv",
    "example": WORKING_DIR / "data/metadata-example.csv",
}

TEMPO_METADATA_PATH = WORKING_DIR / "data/perf_tempo_estimate"


def main():
    parser = argparse.ArgumentParser(
        description="MIDI score following benchmark (mirrors test_audio.py)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="asap",
        help="Dataset (valid, example, asap, batik, vienna)",
    )
    args = parser.parse_args()

    dataset = args.dataset

    metadata = pd.read_csv(METADATA_PATH[dataset])

    tempo_metadata_fn = TEMPO_METADATA_PATH / f"{dataset}_tempo_estimates.csv"

    tempo_metadata = []

    for i, row in enumerate(metadata.itertuples(), 1):
        dataset_dir = DATASET_DIR[dataset]

        score_xml = dataset_dir / row.xml_score
        perf_midi = dataset_dir / row.midi_performance

        score = pt.load_musicxml(score_xml)
        perf = pt.load_performance_midi(perf_midi)

        spart = score.parts[0]
        ppart = perf.performedparts[0]

        sna = spart.note_array()
        pna = ppart.note_array()

        if len(pna) == 0:
            ppart = perf.performedparts[1]
            pna = ppart.note_array()

        perf_end_time = pna[-1]["onset_sec"] + pna[-1]["duration_sec"]

        total_num_beats_score = sna[-1]["onset_beat"] + sna[-1]["duration_beat"]

        bpm = round(total_num_beats_score / perf_end_time * 60.0)

        tempo_metadata.append(
            {
                "dataset": dataset,
                "title": row.title,
                "score_file": row.xml_score,
                "midi_performance_file": row.midi_performance,
                "audio_performance_file": row.audio_performance,
                "estimated_bpm": bpm,
            }
        )


    df = pd.DataFrame(tempo_metadata)
    df.to_csv(tempo_metadata_fn, index=False)

    return

if __name__ == "__main__":
    main()