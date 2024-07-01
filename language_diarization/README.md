
# Baseline for automatic speech recognition


# Prerequisites


- Python >= 3.9
- Kaldi
- pyannote.audio 0.0.1 [installing higher versions may generate inaccurate results]
- dscore
- fastcluster
- ffmpeg
- openai-whisper

Note: install packages using "pip install"

# Displace Audio Directory Structure
```

		data
		 |
		 |
    ---------------------------
    |                         |
    |                         |
    |                         |
    |                         |
   dev                       eval
    |                         |
    |                         |
    |                         |
    |                         |
audio files              audio files
```

# Data Preprocessing 


Run VAD on the audio files to detect the speech regions

VAD output is stored in .pkl files

Convert them to segment files and then to Kaldi formatted segment files


Finally create subsegments from the segments

# Feature Extraction

Based on the created subsegments, run the feature extraction to get the whisper posteriors

# Clustering

Run the clustering algorithm (AHC) and perform VBHMM on top of it to generate RTTMs


# Scoring

Comparing groundtruth RTTMs with the system generated RTTMs

# How to run

1. Place audio files in data directory according the file structure shown above

2. clone the dscore repository in the output directory

3. run the run_all.sh inside code directory

4. The run_all.sh takes audio files as input and gives the DER as output 



