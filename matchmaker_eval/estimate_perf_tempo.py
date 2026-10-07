import argparse
from pathlib import Path

import pandas as pd
import librosa
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import partitura as pt
from partitura.score import merge_parts

from folds import DATA_ROOT, nested_dataset_root

WORKING_DIR = Path(__file__).parent.parent
#: The `example` dataset is the one piece committed to this repository under
#: resources/, so the runners have a smoke test that needs no corpus at all.
DATASET_DIR = {
    "local": WORKING_DIR / "resources",
    "asap": nested_dataset_root("asap"),
    "batik": nested_dataset_root("batik"),
    "vienna": nested_dataset_root("vienna"),
    # upstream copies under MATCHMAKER_DATA_DIR, by their original names
    "kraisler": DATA_ROOT / "KRAISLER",
    "chorale": DATA_ROOT / "chorale-bricks",
    "urmp": DATA_ROOT / "URMP",
    "winterreise": DATA_ROOT / "winterreise",
    "zeilinger": DATA_ROOT / "Zeilinger_data",
}
METADATA_PATH = {
    "valid": WORKING_DIR / "data/metadata-validation.csv",
    "asap": WORKING_DIR / "data/reduced/metadata-asap.csv",
    "batik": WORKING_DIR / "data/reduced/metadata-batik.csv",
    "vienna": WORKING_DIR / "data/reduced/metadata-vienna.csv",
    "example": WORKING_DIR / "data/metadata-example.csv",
    "kraisler": WORKING_DIR / "data/metadata-kraisler.csv",
    "chorale": WORKING_DIR / "data/metadata-chorale.csv",
    "urmp": WORKING_DIR / "data/metadata-urmp-trimmed.csv",
    "winterreise": WORKING_DIR / "data/metadata-winterreise.csv",
    "zeilinger": WORKING_DIR / "data/metadata-zeilinger.csv",
}

TEMPO_METADATA_PATH = WORKING_DIR / "data/perf_tempo_estimate"


def main():
    parser = argparse.ArgumentParser(
        description="MIDI score following benchmark (mirrors test_audio.py)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vienna",
        help="Dataset (valid, example, asap, batik, vienna, kraisler, chorale, urmp, winterreise, zeilinger)",
    )
    args = parser.parse_args()

    dataset = args.dataset

    metadata = pd.read_csv(METADATA_PATH[dataset])

    tempo_metadata_fn = TEMPO_METADATA_PATH / f"{dataset}_tempo_estimates.csv"

    tempo_metadata = []

    for i, row in enumerate(metadata.itertuples(), 1):
        dataset_dir = DATASET_DIR[dataset]
        if dataset == "winterreise":
            score_xml = WORKING_DIR / row.xml_score
        else:
            if not hasattr(row, 'folder'):   
                score_xml = dataset_dir / row.xml_score
            else:
                score_xml = dataset_dir / row.folder / row.xml_score

        if score_xml.suffix == ".mid":
            score = pt.load_score_midi(score_xml)
        else:
            score = pt.load_musicxml(score_xml)
        spart = merge_parts(score.parts)
        sna = spart.note_array()
        score_end = (sna["onset_beat"] + sna["duration_beat"]).max()
        score_start = sna["onset_beat"].min()
        total_num_beats_score = score_end - score_start

        if dataset in ["asap", "batik", "vienna"]:
            perf_midi = dataset_dir / row.midi_performance
            perf = pt.load_performance_midi(perf_midi)
            ppart = perf.performedparts[0]
            pna = ppart.note_array()

            if len(pna) == 0:
                ppart = perf.performedparts[1]
                pna = ppart.note_array()

            perf_end_time = (pna["onset_sec"] + pna["duration_sec"]).max()

        else:
            if dataset == "urmp":
                if row.audio_performance.startswith("dataset_root"):
                    perf_audio = dataset_dir / row.audio_performance.replace("dataset_root/", "")
                else:
                    perf_audio = dataset_dir / row.folder / row.audio_performance
            else:
                # check if 'folder' is a field in the metadata
                if hasattr(row, 'folder'):
                    perf_audio = dataset_dir / row.folder / row.audio_performance
                else:
                    perf_audio = dataset_dir / row.audio_performance
            y, sr = librosa.load(perf_audio, sr=None)
            perf_end_time = len(y) / sr

        bpm = round(total_num_beats_score / perf_end_time * 60.0)
        metadata_dict = dict()
        metadata_dict["dataset"] = dataset
        metadata_dict["title"] = row.title
        metadata_dict["audio_performance_file"] = row.audio_performance
        metadata_dict["estimated_bpm"] = bpm
        if dataset == "urmp":
            if hasattr(row, 'folder'):
                metadata_dict["folder"] = row.folder
            else:
                metadata_dict["folder"] = None
        else:
            if hasattr(row, 'folder'):
                metadata_dict["folder"] = row.folder
        tempo_metadata.append(metadata_dict)

    df = pd.DataFrame(tempo_metadata)
    df.to_csv(tempo_metadata_fn, index=False)

    return

if __name__ == "__main__":
    main()