<div align="left"><img src="displace_img.png" width="550"/></div>

# About the Challenge
Inspired by the broad participation in the DISPLACE 2023 challenge and the need for continued research to advance speech technology within natural multilingual conversations, we announced the second season of the DISPLACE challenge. The current challenge includes an additional track on automatic speech recognition (ASR) in code-switched multi-accent conversational scenarios along with speaker diarization (SD) in multilingual settings and language diarization (LD) in multi-speaker settings, using the same underlying dataset.  Further details about the challenge can be found at [DISPLACE 2024](https://displace2024.github.io/). 



# Updates
[01/07/2024] : The Second DISPLACE challenge ([paper](http://www.arxiv.org/pdf/2406.09494))
[20/02/2024]: Missing files in Speaker Diarization baseline have been updated. 

[8/02/2024]: Track 3 (ASR) baseline details and results on DEV data updated. 

[20/01/2024]: We have released the Baseline codes for speaker diarization and language diarization.

# Challenge Baseline Paper

- Kalluri, Shareef Babu, et al. "The Second DISPLACE Challenge: DIarization of SPeaker and LAnguage in Conversational Environments." arXiv preprint arXiv:2406.09494 (2024) ([paper](http://www.arxiv.org/pdf/2406.09494)).
  
# Baseline for speaker diarization (Track 1)

The implementation of the speaker diarization baseline is largely similar to the  [DISPLACE 2023](https://github.com/displace2023/DISPLACE_Baselines). 
This baseline has been described in the DISPLACE 2023 challenge paper :
- Baghel, Shikha et al., “The DISPLACE Challenge 2023 - DIarization of SPeaker and LAnguage in
Conversational Environments,” in Proc. INTERSPEECH, 2023. ([paper](https://www.isca-speech.org/archive/pdfs/interspeech_2023/baghel23_interspeech.pdf))

The steps involve speech activity detection, front-end feature extraction, x-vector extraction, PLDA scoring followed by Spectral Clustering (SC). The resegmentation is applied to refine speaker assignment using [VB-HMM](https://www.fit.vutbr.cz/research/groups/speech/publi/2018/diez_odyssey2018_63.pdf). 

The major changes in the DISPLACE 2024 baseline are:
1. Speech activity detection using [Pyannote SAD model](https://github.com/pyannote/pyannote-audio)
2. Overlap handling using [Pyannote overlap detector](https://github.com/pyannote/pyannote-audio) and VB-HMM together in the final stage.





# Baseline for Language Dizarization (Track 2)
The implementation of the language diarization baseline is based on an Agglomerative Hierarchical Clustering over language embeddings extracted from a spoken language recognition model trained on the VoxLingua107 dataset using SpeechBrain. The model was based on the ECAPA-TDNN architecture ([1](https://arxiv.org/abs/2005.07143)). VoxLingua covers 107 different languages. We used this model as a feature (embeddings) extractor. We experimented with this model on our own data with a range of different hop lengths and frame sizes. 
The steps involved in language diarization are speech activity detection, utterance-level feature extraction, and followed by Agglomerative Hierarchical Clustering (AHC). 
```
@inproceedings{valk2021slt,
  title={{VoxLingua107}: a Dataset for Spoken Language Recognition},
  author={J{\"o}rgen Valk and Tanel Alum{\"a}e},
  booktitle={Proc. IEEE SLT Workshop},
  year={2021},
}

@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
# Baseline Details for Automatic Speech Recognition (Track 3)
We have implemented the Google Speech to Text cloud services for our baseline system using the close field recordings of development data. 
```
 https://cloud.google.com/speech-to-text 
```
To replicate the results on the close field recordings of DEV data, we kindly request the users create your own Google Cloud services account and make use of Google Speech-to-Text API using the following settings **Speaker diarization**  "off"  and  **language** as follows <br />

తెలుగు (భారతదేశం) -- for Telugu <br />
ಕನ್ನಡ (ಭಾರತ) -- for Kannada <br />
বাংলা (ভারত) -- for Bengali <br />
हिन्दी (भारत) -- for Hindi <br />
English (India) -- for English <br />
Besides the above settings, we have used the audio segments greater than or equal to 2 seconds while giving as input to the API. <br />

We computed the Word Error Rate (WER) for close field recording per language (by concatenating all the generated transcripts per language into one single file) <br/>

|  Language                 | WER (Dev)   | 
| --------------------------| ----------- |
| Hindi                     |   58.5      |       
| Bengali                   |   63.5      |
| Telugu                    |   71.2      |
| Kannada                   |   80.8      |
| English from all sessions | 66.5        |
| Overall                   | 66.7        |


# Installation
  
## Step 1: Clone the repo and create a new virtual environment

Clone the repo:

```bash
https://github.com/displace2024/Displace2024_baseline.git
cd Baseline
```

While not required, we recommend running the recipes from a fresh virtual environment. If using ``virtualenv``:

```bash
virtualenv venv
source venv/bin/activate
```

Alternately, you could use ``conda`` or ``pipenv``. Make sure to activate the environment before proceeding.



## Step 2: Installing Python dependencies

Run the following command to install the required Python packages:

```bash
pip install -r requirements/core.txt
```

```bash
pip install -r requirements/sad.txt
```


## Step 3: Installing remaining dependencies

We also need to install [Kaldi](https://github.com/kaldi-asr/kaldi) and [dscore](https://github.com/nryant/dscore). To do so, run the installation scripts in the ``tools/`` directory:

```bash
cd tools
./install_kaldi.sh
./install_dscore.sh
cd ..
```

Please check the output of these scripts to ensure that installation has succeeded. If succesful, you should see ``Successfully installed {Kaldi,dscore}.`` printed at the end. If the installation of a component fails, please consult the output of the relevant installation script for additional details. If you already have the packages installed creating a softlink to the packages also works.


## Step 4: Running the baselines

Navigate to the ```speaker_diarization``` or ```language_diarization``` directories and follow the instructions in ```README.md``` to run the respective baseline systems.
  
<!-- ## Pretrained SAD model

We have placed a copy of the TDNN+stats SAD model used to produce these results on [Zenodo](https://zenodo.org/). To use this model, download and unarchive the [tarball](https://zenodo.org/record/4299009), then move it to ``speaker_diarization/exp``. -->
