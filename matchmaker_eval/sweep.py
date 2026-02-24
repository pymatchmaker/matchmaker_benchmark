from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import wandb

import yaml
import os

from eval_symbolic import align


DATASET_DIR = {
    "validation": Path("/home/alexander-neuhauser/datasets"),
    "asap": Path("/home/alexander-neuhauser/datasets/asap-dataset-matchmaker"),
    "batik": Path("/home/alexander-neuhauser/datasets/Batik_Audio"),
    "vienna": Path("/home/alexander-neuhauser/datasets/vienna4x22"),
}

METADATA_PATH = "data/metadata-validation.csv"

SAVE_RESULTS = True
UPLOAD_ALIGNMENTS = True

QUICK_TEST = False


def run_tests_and_eval_by_dataset(args, kwargs):
    metadata = pd.read_csv(METADATA_PATH)
    results = pd.DataFrame()
    
    for i, row in enumerate(metadata.itertuples(), 1):
        ####################################
        if QUICK_TEST: # test only two short pieces
            if i < 10 or i > 11: # QUICK TEST
                continue
        ####################################
        dataset_dir = DATASET_DIR[row.dataset]

        score_xml = dataset_dir / row.xml_score

        match = dataset_dir / row.match
        perf_midi = dataset_dir / row.midi_performance
        perf_audio = dataset_dir / row.audio_performance

        if args.input_type == "midi":
            perf = perf_midi
        else:
            perf = perf_audio
        
        result = dict({'composer': row.composer, 'title': row.title, 'performance': perf.stem})

        print(row.title)

        _, res_extended, _, predicted_alignments = align(solo_perf_fn=str(match), 
                                  reference_fn=str(score_xml), 
                                  perf_midi=str(perf_midi),
                                  args=args,
                                  kwargs=kwargs)

        result.update(res_extended)

        wandb.log({f'individual/{row.title}': result})

        if UPLOAD_ALIGNMENTS:
            pred_alignment = [[a,b,c,d] for (a,b,c,d) in zip(predicted_alignments[0], 
                                                             predicted_alignments[1], 
                                                             predicted_alignments[2], 
                                                             predicted_alignments[3])]
            
            table = wandb.Table(data=pred_alignment, columns=["tracked_ponsets", "mapped_ponsets", "tracked_sonsets", "mapped_sonsets"])
            wandb.log(
                {f"alignments/{row.title}/ponsets": wandb.plot.line(
                        table, "tracked_ponsets", "mapped_ponsets", title=f"alignments/{row.title}/ponsets")}
                    )
            wandb.log(
                {f"alignments/{row.title}/sonsets": wandb.plot.line(
                        table, "tracked_sonsets", "mapped_sonsets", title=f"alignments/{row.title}/sonsets")}
                    )
        
        if results.empty:
            results = pd.DataFrame(columns=result.keys())
            if SAVE_RESULTS:
                results_path = os.path.join('sweep-results', wandb.config.method, wandb.run.name)
                individual_csv = os.path.join(results_path, 'individual.csv')
                average_csv = os.path.join(results_path, 'average.csv')
                os.makedirs(results_path)
                config_fn = os.path.join(results_path, "config.yaml")
                with open(config_fn, 'w') as f:
                    yaml.dump(wandb.config._items, f, default_flow_style=False) 
            
        results.loc[len(results)] = result
        if SAVE_RESULTS:
            results.to_csv(individual_csv, index=False)
        

    final_res = dict({
                    'tracking ratio': np.mean(results["tracking_ratio"]),
                    'AAE (ms)': np.mean(results["AAE"]),
                    'std (ms)': np.mean(results["std"]), 
                    'MAE (ms)': np.mean(results["MAE"]), 
                    'skewness (ms)': np.mean(results["skewness (ms)"]),
                    'kurtosis (ms)': np.mean(results["kurtosis (ms)"]),
                    '<10ms': np.mean(results["%_lt_10ms"]),
                    '<25ms': np.mean(results["%_lt_25ms"]),
                    '<50ms': np.mean(results["%_lt_50ms"]),
                    '<100ms': np.mean(results["%_lt_100ms"]),
                    '<500ms': np.mean(results["%_lt_500ms"]),
                    '<1000ms': np.mean(results["%_lt_1000ms"]),
                    '<2000ms': np.mean(results["%_lt_2000ms"]),
                    'count (ms)': np.mean(results["count (ms)"]),
                    'pcr (ms)': np.mean(results["pcr (ms)"]),
                    'AAE (beats)': np.mean(results["AAE (beats)"]),
                    'std (beats)': np.mean(results["std (beats)"]), 
                    'MAE (beats)': np.mean(results["MAE (beats)"]), 
                    'skewness (beats)': np.mean(results["skewness (beats)"]),
                    'kurtosis (beats)': np.mean(results["kurtosis (beats)"]),
                    '<0.05b': np.mean(results["%_lt_<0.05b"]),
                    '<0.1b': np.mean(results["%_lt_<0.1b"]),
                    '<0.5b': np.mean(results["%_lt_<0.5b"]),
                    '<1b': np.mean(results["%_lt_<1b"]),
                    '<2b': np.mean(results["%_lt_<2b"]),
                    'count (beats)': np.mean(results["count (beats)"]),
                    'pcr (beats)': np.mean(results["pcr (beats)"]),
                    })

    wandb.log({'average': final_res})

    final_results = pd.DataFrame(columns=final_res.keys())
    final_results.loc[len(final_results)] = final_res

    print(f'finished.\n{final_results}')
    if SAVE_RESULTS:
        final_results.to_csv(average_csv, index=False)

    return results


def get_tempo_model(args, kwargs):
    from matchmaker.utils.tempo_models import KalmanTempoModel, LinearTempoModel, MovingAverageTempoModel, ReactiveTempoModel, JointAdaptationAnticipationModel

    tempo_models = {"KalmanTempoModel": KalmanTempoModel,
                    "LinearTempoModel" : LinearTempoModel,
                    "MovingAverageTempoModel" : MovingAverageTempoModel,
                    "ReactiveTempoModel" : ReactiveTempoModel,
                    "JointAdaptationAnticipationModel" : JointAdaptationAnticipationModel,
                    }

    kwargs[args.input_type][args.method]["tempo_model"] = tempo_models[kwargs[args.input_type][args.method]["tempo_model"]]
    return kwargs


if __name__ == "__main__":

    with wandb.init(
    ) as run:
        kwargs = {run.config["input_type"]: {run.config["method"]: dict(run.config)}}

        args = SimpleNamespace(input_type = run.config["input_type"], method = run.config["method"], dataset = "validation")

        if args.method == 'hmm':
            kwargs = get_tempo_model(args, kwargs)

        run_tests_and_eval_by_dataset(args = args, kwargs = kwargs)