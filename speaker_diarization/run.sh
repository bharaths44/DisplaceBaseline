#!/bin/bash

set -e  # Exit on error.

PYTHON=python  # Python to use; defaults to system Python.

################################################################################
# Configuration
################################################################################
nj=4
decode_nj=4 #eval
stage=4
sad_train_stage=0
sad_decode_stage=0
diarization_stage=0
vb_hmm_stage=0
overlap=1
# If following is "true", then SAD output will be evaluated against reference
# following decoding stage. This step requires the following Python packages be
# installed:
#
# - pyannote.core
# - pyannote.metrics
# - pandas
eval_sad=false


################################################################################
# Paths to DATA releases
################################################################################
DISPLACE_DEV_DIR=DISPLACE_dev
DISPLACE_EVAL_DIR=DISPLACE_eval

pyannote_pretrained_model=../vad_benchmarking/VAD_model/pytorch_model.bin # SAD and Overlap pyannote model.

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

################################################################################
# Prepare data directories
################################################################################
if [ $stage -le 0 ]; then
  echo "$0: Preparing data directories..."

  # dev
  if [ -d "$DISPLACE_DEV_DIR" ]; then 
  local/make_data_dir.py \
  --rttm-dir $DISPLACE_DEV_DIR/data/rttm_sd \
    data/displace_dev \
    $DISPLACE_DEV_DIR/data/wav
  ./create_utt2spk_spk2utt.sh data/displace_dev $DISPLACE_DEV_DIR
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/displace_dev/
  else
    echo "${DISPLACE_DEV_DIR} does not exist"
    exit 1
  fi

  # eval
  if [ -d "$DISPLACE_EVAL_DIR" ]; then 
  local/make_data_dir.py \
    --rttm-dir $DISPLACE_EVAL_DIR/data/rttm_sd \
    data/displace_eval \
    $DISPLACE_EVAL_DIR/data/wav 
   ./create_utt2spk_spk2utt.sh data/displace_eval $DISPLACE_EVAL_DIR
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/displace_eval/
  else
    echo "${DISPLACE_EVAL_DIR} does not exist"
  fi
fi

#####################################
# SAD decoding. Pyannote
#####################################
if [ $stage -le 1 ]; then
  echo "$0: Applying SAD model to DEV/EVAL..."
  sad_model=$pyannote_pretrained_model
  dset=dev
 
    ../vad_benchmarking/run_pyannote_SAD.sh \
      --nj $nj --stage $sad_decode_stage \
      data/displace_${dset} data/displace_${dset}_seg \
      $sad_model 
    
  if [ -d $DISPLACE_EVAL_DIR/data/wav ]; then
     dset=eval
     ../vad_benchmarking/run_pyannote_SAD.sh \
      --nj $nj --stage $sad_decode_stage \
      data/displace_${dset} data/displace_${dset}_seg \
      $sad_model
  fi
fi

################################################################################
# Perform first-pass diarization using AHC/SC.
################################################################################
period=0.25

scoretype=laplacian
outdevdir=exp/displace_diarization_nnet_1a_dev_spectral
outevaldir=exp/displace_diarization_nnet_1a_eval_spectral

if [ $stage -le 2 ]; then
  # cp -r exp/displace_diarization_nnet_1a_dev_fbank_spectral $outdevdir
  echo "$0: Performing first-pass diarization of DEV..."
  local/diarize_fbank.sh \
    --nj $nj --stage $diarization_stage \
    --tune true --period $period --clustering spectral \
    --scoretype $scoretype \
    exp/xvector_nnet_1a_tdnn_fbank/ exp/xvector_nnet_1a_tdnn_fbank/plda_model_full \
    data/displace_dev_seg/ $outdevdir
fi


if [ $stage -le 3 ] && [ -d $DISPLACE_EVAL_DIR/data/wav ]; then
  # cp -r exp/displace_diarization_nnet_1a_eval_fbank_spectral $outevaldir
  echo "$0: Performing first-pass diarization of EVAL using threshold "
  echo "$0: obtained by tuning on DEV..."
  thresh=$(cat ${outdevdir}/tuning_${period}/thresh_best)
  local/diarize_fbank.sh \
    --nj $decode_nj --stage $diarization_stage \
    --thresh $thresh --tune false --period $period  --clustering spectral \
    --scoretype $scoretype \
    exp/xvector_nnet_1a_tdnn_fbank/ exp/xvector_nnet_1a_tdnn_fbank/plda_model_full \
    data/displace_eval_seg/ $outevaldir
fi


################################################################################
# Evaluate first-pass diarization.
# ################################################################################
if [ $stage -le 4 ]; then
  cat $DISPLACE_DEV_DIR/data/rttm_sd/*.rttm > $DISPLACE_DEV_DIR/data/final.rttm
  echo "$0: Scoring first-pass diarization on DEV..."
  local/diarization/score_diarization.sh \
    --scores-dir $outdevdir/scoring \
    $DISPLACE_DEV_DIR $outdevdir/per_file_rttm
fi


if [ $stage -le 5 ] && [ -d $DISPLACE_EVAL_DIR/data/rttm_sd ]; then
  cat $DISPLACE_EVAL_DIR/data/rttm_sd/*.rttm > $DISPLACE_EVAL_DIR/data/final.rttm
  echo "$0: Scoring first-pass diarization on EVAL..."
  local/diarization/score_diarization.sh \
    --scores-dir $outevaldir/scoring \
    $DISPLACE_EVAL_DIR $outevaldir/per_file_rttm
fi

 
if [ $overlap -eq 0 ]; then
  ################################################################################
  # Refined first-pass diarization using VB-HMM resegmentation
  ################################################################################
  dubm_model=exp/xvec_init_gauss_1024_ivec_400/model/diag_ubm.pkl
  ie_model=exp/xvec_init_gauss_1024_ivec_400/model/ie.pkl

  outdevdir_vb=exp/displace_diarization_nnet_1a_vbhmm_dev_spectral
  outevaldir_vb=exp/displace_diarization_nnet_1a_vbhmm_eval_spectral

  if [ $stage -le 6 ]; then
    echo "$0: Performing VB-HMM resegmentation of DEV..."
    statScale=10
    loop=0.45
    maxiters=1
    echo "statScale=$statScale loop=$loop maxiters=$maxiters" 
    local/resegment_vbhmm.sh \
        --nj $nj --stage $vb_hmm_stage --statscale $statScale --loop $loop --max-iters $maxiters \
        data/displace_dev $outdevdir/rttm \
        $dubm_model $ie_model $outdevdir_vb/
  fi


  if [ $stage -le 7 ]; then
    echo "$0: Performing VB-HMM resegmentation of EVAL..."
    local/resegment_vbhmm.sh \
        --nj $decode_nj --stage $vb_hmm_stage \
        data/displace_eval $outevaldir/rttm \
        $dubm_model $ie_model $outevaldir_vb/
  fi


  ################################################################################
  # Evaluate VB-HMM resegmentation.
  ################################################################################
  if [ $stage -le 8 ]; then
    echo "$0: Scoring VB-HMM resegmentation on DEV..."
    local/diarization/score_diarization.sh \
      --scores-dir $outdevdir_vb/scoring \
      $DISPLACE_DEV_DIR $outdevdir_vb/per_file_rttm
  fi


  if [ $stage -le 9 ] && [ -d $DISPLACE_EVAL_DIR/data/rttm ]; then
    if [ -d $DISPLACE_EVAL_DIR/data/rttm/ ]; then
      echo "$0: Scoring VB-HMM resegmentation on EVAL..."
      local/diarization/score_diarization.sh \
        --scores-dir $outevaldir_vb/scoring \
        $DISPLACE_EVAL_DIR $outevaldir_vb/per_file_rttm
    fi
  fi

else
  #####  WITH OVERLAP ################################
  ################################################################################
  # Refined first-pass diarization using VB-HMM resegmentation with pyannote overlap
  ################################################################################
  dubm_model=exp/xvec_init_gauss_1024_ivec_400/model/diag_ubm.pkl
  ie_model=exp/xvec_init_gauss_1024_ivec_400/model/ie.pkl

  outdevdir_vb=exp/displace_diarization_nnet_1a_vbhmm_dev_spectral_overlap
  outevaldir_vb=exp/displace_diarization_nnet_1a_vbhmm_eval_spectral_overlap

  statScale=10
  loop=0.45
  maxiters=1
  echo "statScale=$statScale loop=$loop maxiters=$maxiters" 

  if [ $stage -le 6 ]; then
    echo "$0: Performing VB-HMM resegmentation of DEV..."
    mkdir -p $outdevdir_vb
    
    # cp -r exp_full_final/displace_diarization_nnet_1a_vbhmm_dev_overlap/per_file_overlap $outdevdir_vb/
    local/resegment_vbhmm.sh \
        --nj $nj --stage $vb_hmm_stage --statscale $statScale --loop $loop --max-iters $maxiters \
        --overlap 1 --PYTHON $PYTHON \
        --pyannote_pretrained_model $pyannote_pretrained_model \
        data/displace_dev $outdevdir/rttm \
        $dubm_model $ie_model $outdevdir_vb/ 
        
       
  fi

  if [ $stage -le 7 ]; then
    echo "$0: Performing VB-HMM resegmentation of EVAL..."
    mkdir -p $outevaldir_vb
   
    local/resegment_vbhmm.sh \
        --nj $decode_nj --stage $vb_hmm_stage \
        --statscale $statScale --loop $loop --max-iters $maxiters \
        --overlap 1 --PYTHON $PYTHON \
        --pyannote_pretrained_model $pyannote_pretrained_model \
        data/displace_eval $outevaldir/rttm \
        $dubm_model $ie_model $outevaldir_vb/
  fi


  ################################################################################
  # Evaluate VB-HMM resegmentation.
  ################################################################################
  echo "in the VBHMM scoring stage"
  if [ $stage -le 8 ]; then
    echo "$0: Scoring VB-HMM resegmentation on DEV..."
    local/diarization/score_diarization.sh \
      --scores-dir $outdevdir_vb/scoring \
      $DISPLACE_DEV_DIR $outdevdir_vb/per_file_rttm
  fi


  if [ $stage -le 9 ] && [ -d $DISPLACE_EVAL_DIR/data/rttm_sd ]; then
    if [ -d $DISPLACE_EVAL_DIR/data/rttm_sd/ ]; then
      echo "$0: Scoring VB-HMM resegmentation on EVAL..."
      local/diarization/score_diarization.sh \
        --scores-dir $outevaldir_vb/scoring \
        $DISPLACE_EVAL_DIR $outevaldir_vb/per_file_rttm
    fi
  fi

fi
