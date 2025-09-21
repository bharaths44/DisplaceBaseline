#  
#


#  This script runs the baseline system for language diarization for 
#  DISPLACE 2024 challenge 
# 
#  This bash script is a wrapper for different  python3 scripts
#  It uses libraries 
#      - pyannote.audio 0.01, 
#      - openai-whisper 
#      - ffmpeg 
#      -fastcluster	

# Change to the directory where this script is located
cd "$(dirname "$0")"

echo "=== DISPLACE 2024 Language Diarization Baseline System ==="
echo "Starting script execution from directory: $(pwd)"

#------------------------------------------------
#    Configuration
#------------------------------------------------

stage=0    
      # stage 0 - data preparation
      # Stage 1 - feature extraction and vad 
      # Stage 2 - diarization 

PYTHON=/Users/bharaths/Downloads/Displace2024_baseline_updated-main/.venv/bin/python3.10 

SEG_DUR=10    # maximum segment duration in sec
SEG_SHIFT=0.4 # shift to the next segment in sec
SEG_OVRLAP=`echo "$SEG_DUR - $SEG_SHIFT" | bc -l` 



#------------------------------------------------
#    Dev and Eval folders 
#------------------------------------------------

DISPLACE_DEVDATA='../data/dev'
DISPLACE_EVALDATA='../data/eval'
OUTPUT_DIR='../../output'

echo "Configuration:"
echo "  Python executable: $PYTHON"
echo "  Dev data path: $DISPLACE_DEVDATA"
echo "  Output directory: $OUTPUT_DIR"
echo "  Segment duration: $SEG_DUR seconds"
echo "  Segment shift: $SEG_SHIFT seconds"
echo ""


# #------------------------------------------------
# #    VAD Preparation
# #------------------------------------------------

echo "=== Stage 1: VAD Preparation ==="
echo "Running Voice Activity Detection..."

VAD_SUBDIR='dev/'
SEG_SUBDIR='devseg/'
KAL_SUBDIR='kal_devseg/'

echo "Executing: $PYTHON pyannote_vad.py $DISPLACE_DEVDATA $OUTPUT_DIR $VAD_SUBDIR $SEG_SUBDIR"
$PYTHON pyannote_vad.py $DISPLACE_DEVDATA  $OUTPUT_DIR $VAD_SUBDIR $SEG_SUBDIR

echo "Converting segments to Kaldi format..."
path_segments="$OUTPUT_DIR/$SEG_SUBDIR"
path_new_kaldi_segs="$OUTPUT_DIR/$KAL_SUBDIR"
mkdir -p $path_new_kaldi_segs
for path in $(ls $path_segments)
do
    g=${path_segments}/${path}
    echo "Processing: $g -> ${path_new_kaldi_segs}${path}.txt"
    $PYTHON seg2kaldi.py $g ${path_new_kaldi_segs}${path}.txt
done
echo "VAD Preparation completed."
echo ""



# #-------------------------------------------------
# #    Subsegment Creation
# #-------------------------------------------------

echo "=== Stage 2: Subsegment Creation ==="

SUBSEG_SUBDIR='dev_subseg/'
path_subsegs="$OUTPUT_DIR/$SUBSEG_SUBDIR"
mkdir -p $path_subsegs


echo "Creating subsegments..."
echo "Command: $PYTHON create_subseg.py --input_folder $path_new_kaldi_segs --out_subsegments_folder $path_subsegs --max-segment-duration $SEG_DUR --overlap-duration $SEG_OVRLAP --max-remaining-duration $SEG_DUR --constant-duration False"
$PYTHON create_subseg.py --input_folder $path_new_kaldi_segs --out_subsegments_folder $path_subsegs --max-segment-duration $SEG_DUR --overlap-duration $SEG_OVRLAP --max-remaining-duration $SEG_DUR --constant-duration False
echo "Subsegment creation completed."
echo ""

# #-------------------------------------------------
# #    Feature Extraction
# #-------------------------------------------------

echo "=== Stage 3: Feature Extraction ==="

FEAT_SUBDIR='dev_feat/'
mkdir -p $OUTPUT_DIR/$FEAT_SUBDIR

echo "Extracting features..."
echo "Command: $PYTHON feat_extr.py --aud_path $DISPLACE_DEVDATA --out_dir $OUTPUT_DIR/$FEAT_SUBDIR --seg_dir $OUTPUT_DIR/$SUBSEG_SUBDIR"
$PYTHON feat_extr.py --aud_path $DISPLACE_DEVDATA  --out_dir $OUTPUT_DIR/$FEAT_SUBDIR --seg_dir  $OUTPUT_DIR/$SUBSEG_SUBDIR
echo "Feature extraction completed."
echo ""



# #-------------------------------------------------
# #    Clustering (AHC)
# #-------------------------------------------------

echo "=== Stage 4: Clustering (AHC) ==="

FEAT_SEG_TSV='dev_whisp_feat.tsv'
RTTM_OUT_DIR='dev_rttm_outdir'
CLUSTER_ALGO='AHC'
mkdir -p $OUTPUT_DIR/$RTTM_OUT_DIR

echo "Preparing feature and segment lists..."
ls -1 $OUTPUT_DIR/$SUBSEG_SUBDIR/*.txt  | sort > $OUTPUT_DIR/temp_segs
ls -1 $OUTPUT_DIR/$FEAT_SUBDIR/*.npy | sort > $OUTPUT_DIR/temp_posteriors
paste $OUTPUT_DIR/temp_posteriors $OUTPUT_DIR/temp_segs > $OUTPUT_DIR/$FEAT_SEG_TSV

echo "Running clustering algorithm: $CLUSTER_ALGO"
echo "Command: $PYTHON clustering.py $OUTPUT_DIR/$FEAT_SEG_TSV $OUTPUT_DIR/$RTTM_OUT_DIR $CLUSTER_ALGO"
$PYTHON clustering.py $OUTPUT_DIR/$FEAT_SEG_TSV $OUTPUT_DIR/$RTTM_OUT_DIR $CLUSTER_ALGO
echo "Clustering completed."
echo ""
	

#-------------------------------------------------
#    Scoring
#-------------------------------------------------

echo "=== Stage 5: Scoring ==="

# Create a combined reference RTTM file from individual RTTM files in ground_truth
ref_RTTM=$OUTPUT_DIR/dscore/ref_dev_combined.rttm
sys_RTTM=$OUTPUT_DIR/dscore/sys_dev_whisper.rttm

echo "Combining reference RTTM files..."
echo "Command: cat $OUTPUT_DIR/ground_truth/*.rttm > $ref_RTTM"
cat $OUTPUT_DIR/ground_truth/*.rttm > $ref_RTTM

echo "Combining system RTTM files..."
echo "Command: cat $OUTPUT_DIR/$RTTM_OUT_DIR/*.rttm > $sys_RTTM"
cat $OUTPUT_DIR/$RTTM_OUT_DIR/*.rttm > $sys_RTTM

echo "Running scoring..."
echo "Command: $PYTHON score.py -r $ref_RTTM -s $sys_RTTM"
$PYTHON score.py -r $ref_RTTM -s $sys_RTTM

echo ""
echo "=== Script execution completed ===" 
