. ./cmd.sh
. ./path.sh

eval_sad=false
PYTHON=python # requires pyannote
nj=27
stage=0
# default
min_duration_on=0.0554
min_duration_off=0.0979
onset=0.5
offset=0.5

. utils/parse_options.sh



datasetpath=$1 #/data1/shareefb/track2_cluster/data/$dataset
path_new_kaldi_segs=$2 #/data1/shareefb/track2_cluster/data/$kaldi_dataset
dataset=`basename $datasetpath` #displace_dev_fbank
kaldi_dataset=`basename $path_new_kaldi_segs` #displace_pyannote_dev_fbank_seg
pyannote_pretrained_model=$3 #vad_benchmarking/VAD_model/pytorch_model.bin

utils/split_data.sh $datasetpath $nj
outputdir=../pyannote_vad/${dataset} #/${hyper} 
mkdir -p $outputdir
echo "dataset=$dataset, kaldi_dataset=$kaldi_dataset"
if [ $stage -le 1 ]; then

    offset=$onset
    JOB=1
    echo "######################################################################"
    $decode_cmd JOB=1:$nj $outputdir/log/sad.JOB.log \
        $PYTHON ../vad_benchmarking/VAD.py \
        --in-audio=$datasetpath/split$nj/JOB/wav.scp \
        --in-VAD=Pyannote_VAD \
        --dataset $dataset \
        --tuning \
        --onset $onset \
        --offset $offset \
        --min_duration_on $min_duration_on \
        --min_duration_off $min_duration_off \
        --outputpath $outputdir \
        --pyannote_pretrained_model $pyannote_pretrained_model
        

    echo "######################################################################"
                    
fi

#####################################
# Generate SAD output in segments format.
#####################################
if [ $stage -le 2 ]; then
    echo "$0: convert to kaldi style segments ..."

    # generate pyannote segments 
    echo utils.py --vad_dir_path $outputdir
    $PYTHON ../vad_benchmarking/utils.py --vad_dir_path $outputdir

    # convert to kaldi style segments
    ../vad_benchmarking/run_kaldi_seg.sh $outputdir $kaldi_dataset $path_new_kaldi_segs/filewise_segments
    cat $path_new_kaldi_segs/filewise_segments/*.segments > $path_new_kaldi_segs/segments
             
fi
####################################################
echo copying wav.scp and creating utt2spk and spk2utt from segments folder
####################################################
if [ $stage -le 3 ]; then
    cp $datasetpath/wav.scp $path_new_kaldi_segs/.
    [[ -e $datasetpath/rttm ]] && cp $datasetpath/rttm $path_new_kaldi_segs/
    awk '{print $1,$2}'  $path_new_kaldi_segs/segments >  $path_new_kaldi_segs/utt2spk
    utils/utt2spk_to_spk2utt.pl $path_new_kaldi_segs/utt2spk > $path_new_kaldi_segs/spk2utt
fi

