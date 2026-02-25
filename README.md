# A Systematic Comparison of Methods for Real-time Music Alignment

This repository contains experiments for a systematic comparison of methods for real-time music alignment using the matchmaker package.

## Setup

### Setting up the code

Setting up the experiments as described here requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Follow the instructions for your OS.

To setup the experiments, use the following script.

```bash
# Download this repository
git clone https://github.com/neosatrapahereje/ismir2025_matchmaker.git
 
# Clone matchmaker
git clone https://github.com/pymatchmaker/matchmaker.git

# Create and activate conda environment
conda env create -f environment.yml

conda activate ismir2024_matchmaker

# Go to matchmaker directory
cd ../matchmaker

# Install matchmaker in editable mode
pip install -e ."[dev]"

# Install GCC
conda install -c conda-forge gcc=12.1.0

# Install glib and fluidsynth
conda install -c conda-forge glib fluidsynth

```

### Setting up the datasets

Please set the `DATASET_DIR` in `matchmaker_eval/test.py` to the path of the dataset you want to use.

```python
# matchmaker_eval/test.py
DATASET_DIR = {
    "asap": Path("~/data/asap-dataset-matchmaker").expanduser(),
    "batik": Path("~/data/Batik_Audio").expanduser(),
    "vienna": Path("~/data/vienna4x22").expanduser(),
}
```

## Running the experiments

### Inference for a single file
You can run the following command to run inference of a single performance. For a quick test, you can use the following command:

```bash
python matchmaker_eval/infer.py --eval
```

For a specific performance file, you can run the following command with arguments:

```bash
python matchmaker_eval/infer.py --score ./resources/ex_score.mid --perf ./resources/ex_VuV01M.wav --perf-annots ./resources/ex_VuV01M._annotations.txt --eval
```

The results will be saved in the `output` directory, and the results will include the following metrics:

```javascript
{
    "mean": 16.3835,  // mean of the absolute alignment error (in ms)
    "median": 16.8721, // median of the absolute alignment error (in ms)
    "std": 6.5937, // standard deviation of the absolute alignment error (in ms)
    "skewness": 1.7935, // skewness of the absolute alignment error (in ms)
    "kurtosis": 9.2273, // kurtosis of the absolute alignment error (in ms)
    "50ms": 0.9926, // percentage of the alignment error within 50ms
    "100ms": 1.0, // percentage of the alignment error within 100ms
    "300ms": 1.0, // percentage of the alignment error within 300ms
    "500ms": 1.0, // percentage of the alignment error within 500ms
    "1000ms": 1.0, // percentage of the alignment error within 1000ms
    "2000ms": 1.0, // percentage of the alignment error within 2000ms
    "count": 136 // number of aligned events in the performance (beats)
}
```

### Experiment with a dataset (input type: audio)

You can run the following command to run the experiments.

```bash
python matchmaker_eval/test.py --dataset asap --method arzt
```

You can also report the results to wandb.

```bash
python matchmaker_eval/test.py --dataset vienna --method dixon --wandb
```

All the results will be saved in the `output` directory.

If you want to change the config of the inference, you can change the configurations in `matchmaker_eval/config/experiment.yaml`.

## Acknowledgments

This work has been supported by the Austrian Science Fund (FWF), grant agreement PAT 8820923 (“Rach3: A Computational Approach to Study Piano Rehearsals”). Additionally, this work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. NRF-2023R1A2C3007605).
