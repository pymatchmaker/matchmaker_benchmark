import json
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from matchmaker.features.audio import FRAME_RATE
from matchmaker.utils.misc import save_nparray_to_csv
from matchmaker.utils.eval import get_evaluation_results, TOLERANCES, transfer_positions
from matchmaker.utils.misc import save_mixed_audio
from matchmaker.prob.hmm import (
    GaussianAudioPitchHMM,
    GaussianAudioPitchTempoHMM,
    CosineExpGaussianAudioPitchTempoObservationModel,
)
from scipy.spatial import distance
from scipy.signal import fftconvolve

import warnings

from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing


warnings.filterwarnings("ignore")

WORKING_DIR = Path(__file__).parent.parent
DATASET_DIR = {
    "asap": Path("../datasets/asap-dataset-matchmaker"),
    "batik": Path("../datasets/Batik_Audio"),
    "vienna": Path("../datasets/vienna4x22"),
}
METADATA_PATH = {
    "asap": WORKING_DIR / "data/metadata-asap.csv",
    "batik": WORKING_DIR / "data/metadata-batik.csv",
    "vienna": WORKING_DIR / "data/metadata-vienna.csv",
}

METADATA = {
    "asap": pd.read_csv(METADATA_PATH["asap"]),
    "batik": pd.read_csv(METADATA_PATH["batik"]),
    "vienna": pd.read_csv(METADATA_PATH["vienna"]),
}
OUTPUT_DIR = WORKING_DIR / "pitchtempohmm_results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def adjust_predictions(perf_annots, perf_annots_predicted):
    if len(perf_annots_predicted) < len(perf_annots):
        pl = len(perf_annots_predicted)

        # Reverse the predicted_pos for cross-correlation
        reversed_pred = perf_annots_predicted[::-1]

        # Cross-correlation via FFT
        cross_corr = fftconvolve(perf_annots, reversed_pred, mode="valid")

        # Sum of squares of perf_annots in sliding windows
        perf_squared = fftconvolve(perf_annots**2, np.ones(pl), mode="valid")

        # Sum of squares of predicted_pos (constant)
        predicted_squared_sum = np.sum(perf_annots_predicted**2)

        # Compute MSE
        mse = (perf_squared + predicted_squared_sum - 2 * cross_corr) / pl

        best_idx = np.argmin(mse)

        # Align predicted_pos
        pred_pos = np.zeros(len(perf_annots))
        pred_pos[best_idx : best_idx + pl] = perf_annots_predicted
        perf_annots_predicted = pred_pos
    return perf_annots_predicted


def plot_and_save_score_following_result(
    wp,
    ref_features,
    input_features,
    distance_func,
    score_annots,
    perf_annots,
    perf_annots_pred,
    frame_rate,
    out_fn: Path,
    name: str,
    plot_cdist: bool = False,
):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"

    plt.figure(figsize=(15, 15))
    if plot_cdist:
        dist = distance.cdist(
            ref_features,
            input_features[: wp[1][-1]],
            metric=distance_func,
        )  # [d, wy]

        plt.imshow(
            dist,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="plasma",
        )
    plt.title(
        f"[{name}] \n Alignment path with ground-truth labels",
        fontsize=25,
    )
    plt.xlabel("Performance Audio frame", fontsize=15)
    plt.ylabel("Score Audio frame", fontsize=15)

    # plot online DTW path
    # ref_paths, target_paths = wp[0], wp[1]
    # for n in range(len(ref_paths)):
    #     plt.plot(
    #         target_paths[n],
    #         ref_paths[n],
    #         ".",
    #         color="lime",
    #         alpha=0.5,
    #         markersize=5,
    #     )
    for i, (ref, target) in enumerate(zip(score_annots, perf_annots_pred)):
        plt.plot(
            target * frame_rate,
            ref * frame_rate,
            ".",
            color="lime",
            alpha=0.9,
            markersize=20,
        )
    # plot ground-truth labels
    for i, (ref, target) in enumerate(zip(score_annots, perf_annots)):
        plt.plot(
            target * frame_rate,
            ref * frame_rate,
            "x",
            color="r",
            alpha=1,
            markersize=20,
        )
    plt.tight_layout()
    plt.savefig(out_fn)
    plt.clf()
    plt.close()


def test_alignment_piece(
    dataset,
    proc_name,
    afn,
    pfeat_fn,
    rfeat_fn,
    safn,
    pafn,
    model,
):
    try:
        pred_annots_path = OUTPUT_DIR / Path(
            f"{dataset}/{proc_name}/{afn.stem}_{model}_{proc_name}_predicted_annotations.txt"
        )
        warping_path_path = OUTPUT_DIR / Path(
            f"{dataset}/{proc_name}/{afn.stem}_{model}_{proc_name}_warping_path.txt"
        )
        mixed_fn = OUTPUT_DIR / Path(
            f"{dataset}/{proc_name}/{afn.stem}_{model}_{proc_name}.wav"
        )
        results_path = OUTPUT_DIR / Path(
            f"{dataset}/{proc_name}/{afn.stem}_{model}_{proc_name}_results.json"
        )
        plot_path = OUTPUT_DIR / Path(
            f"{dataset}/{proc_name}/{afn.stem}_{model}_{proc_name}_wp.pdf"
        )

        if all(
            path.exists()
            for path in [
                pred_annots_path,
                warping_path_path,
                mixed_fn,
                results_path,
                plot_path,
            ]
        ):
            print(f"All output files already exist for {afn.stem}. Skipping...")
            return

        audio_frames = np.load(pfeat_fn, allow_pickle=True)["frames"]
        ref_frames = np.load(rfeat_fn, allow_pickle=True)["frames"]

        ref_features = np.vstack([rf[0] for rf in ref_frames])
        input_features = np.vstack([ff[0] for ff in audio_frames])
        score_annots = np.loadtxt(safn, dtype=float)

        if dataset != "asap":
            perf_annots = np.loadtxt(pafn, dtype=float)
        else:
            perf_annots = np.loadtxt(pafn, usecols=0, dtype=float)

        if model == "phmm":
            score_follower = GaussianAudioPitchHMM(
                reference_features=ref_features,
                precision=2,
            )
        elif model == "pthmm":

            if proc_name != "chroma":
                obs_model = CosineExpGaussianAudioPitchTempoObservationModel(
                    audio_features=ref_features,
                    pitch_rate=2,
                    ioi_precision=2,
                )
            else:
                obs_model = None
            score_follower = GaussianAudioPitchTempoHMM(
                reference_features=ref_features,
                observation_model=obs_model,
                # pitch_precision=0.5,
                # ioi_precision=2,
                transition_scale=0.05,
            )

        current_positions = []
        for frame, f_time in audio_frames:
            current_pos = score_follower((frame, f_time))

            current_positions.append((f_time, current_pos))

        perf_annots_predicted = transfer_positions(
            score_follower.warping_path,
            perf_annots,
            frame_rate=FRAME_RATE,
        )

        perf_annots_predicted = adjust_predictions(
            perf_annots=perf_annots,
            perf_annots_predicted=perf_annots_predicted,
        )

        out_dir = OUTPUT_DIR / Path(f"{dataset}/{proc_name}/")

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        # Save predicted annotations to a text file
        np.savetxt(pred_annots_path, perf_annots_predicted, fmt="%.6f", delimiter="\t")

        # Save warping path to a text file
        np.savetxt(
            warping_path_path,
            score_follower.warping_path,
            fmt="%.6f",
            delimiter="\t",
        )

        mixed_fn.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        save_mixed_audio(
            afn,
            perf_annots_predicted,
            save_path=mixed_fn,
        )

        results = get_evaluation_results(
            perf_annots=perf_annots,
            perf_annots_predicted=perf_annots_predicted,
            tolerances=TOLERANCES,
        )

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        plot_and_save_score_following_result(
            wp=score_follower.warping_path,
            ref_features=ref_features,
            input_features=input_features,
            distance_func=distance.cosine,
            score_annots=score_annots,
            perf_annots=perf_annots,
            perf_annots_pred=perf_annots_predicted,
            frame_rate=FRAME_RATE,
            out_fn=plot_path,
            name=f"{afn.stem}_{model}_{proc_name}",
        )
    except Exception as e:
        raise e
        return (dataset, proc_name, afn, pfeat_fn, rfeat_fn, safn, pafn, model, e)


if __name__ == "__main__":

    missing_features = []
    missing_rfeatures = []
    missing_score_annotations = []

    tasks = []
    for dataset in [
        # "vienna",
        "asap",
        "batik",
    ]:

        for proc_name in [
            # "chroma",
            "mel",
            # "lse",
            # "mfcc",
        ]:

            for ix, row in METADATA[dataset].iterrows():

                afn = DATASET_DIR[dataset] / Path(row["audio_performance"])
                rfn = DATASET_DIR[dataset] / Path(row["xml_score"])

                # audio features
                pfeat_fn = afn.with_name(f"{afn.stem}_{proc_name}.npz")
                # reference features
                rfeat_fn = afn.with_name(f"{afn.stem}_scoresynth_{proc_name}.npz")
                # score annotations
                safn = afn.with_name(f"{afn.stem}_score_annotations.txt")
                pafn = DATASET_DIR[dataset] / Path(row["performance_annotations"])

                if not pfeat_fn.exists():
                    print(f"Features do not exist for {pfeat_fn}")
                    missing_features.append(pfeat_fn)

                if not rfeat_fn.exists():
                    print(f"Reference features do not exist for {rfeat_fn}")
                    missing_rfeatures.append(rfeat_fn)

                if not safn.exists():
                    print("Score annotations do not exist")
                    missing_score_annotations.append(safn)

                if pfeat_fn.exists() and rfeat_fn.exists() and safn.exists():

                    tasks.append(
                        (dataset, afn, proc_name, pfeat_fn, rfeat_fn, safn, pafn)
                    )

    # tasks = np.array(tasks, dtype=object)
    # random_state = np.random.RandomState(seed=1984)
    # tasks_idxs = random_state.choice(np.arange(len(tasks)), size=5, replace=False)
    # tasks = tasks[30:]
    model = "pthmm"

    for dataset, afn, proc_name, pfeat_fn, rfeat_fn, safn, pafn in tasks:
        print(dataset, afn, proc_name)
        test_alignment_piece(
            dataset=dataset,
            proc_name=proc_name,
            afn=afn,
            pfeat_fn=pfeat_fn,
            rfeat_fn=rfeat_fn,
            safn=safn,
            pafn=pafn,
            model=model,
        )

    # num_cores = multiprocessing.cpu_count()

    # failed_pieces = Parallel(n_jobs=num_cores)(
    #     delayed(test_alignment_piece)(
    #         dataset, proc_name, afn, pfeat_fn, rfeat_fn, safn, pafn, model
    #     )
    #     for dataset, afn, proc_name, pfeat_fn, rfeat_fn, safn, pafn in tqdm(
    #         tasks, desc="Processing alignments"
    #     )
    # )

    # print(failed_pieces)
