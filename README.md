# A Systematic Comparison of Methods for Real-time Music Alignment

This repository contains experiments for a systematic comparison of methods for real-time music alignment using the matchmaker package.

## Setup

### Setting up the code

Setting up the experiments as described here requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Follow the instructions for your OS.

To setup the experiments, use the following script.

```bash
# Download this repository
git clone https://github.com/neosatrapahereje/ismir2024_matchmaker.git
 
# Clone matchmaker
git clone https://github.com/neosatrapahereje/matchmaker.git

cd matchmaker_eval

conda env create -f environment.yml

# Go to matchmaker directory
# cd path/to/matchmaker
cd ../matchmaker

# Install matchmaker (TODO update this part when matchmaker is published)
pip install -e .

# Install soundfont for fluidsynth
mkdir -p ~/soundfonts/sf2
wget ftp://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2 ~/soundfonts/sf2/
```

### Setting up the datasets

TBD

## Running the experiments

You can run the following command to run inference of a single audio file with a midi file provided on resources.

```bash
# If score is in MIDI format, and performance input is an audio file,
python matchmaker_eval/infer.py --score ./resources/ex_score.mid --perf ./resources/ex_VuV01M.wav --eval
```

If you want to change the config of the inference, you can change the configurations in `matchmaker_eval/config/default.yaml`.

```bash
sample_rate: 44100
frame_rate: 30
chunk_size: 1
window_size: 5
features: ["chroma"]
distance_func: "euclidean"
max_run_count: 30
dataset: "asap"
algorithm: "hmm"  # hmm, oltw_dixon, oltw_arzt
```

## Acknowledgments

TBD
