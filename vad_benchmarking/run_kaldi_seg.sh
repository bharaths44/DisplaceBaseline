path_segments=$1/segments
kaldi_dataset=$2
# hyper=$3
# path_segments=/data1/prachis/Amrit_sharc/pyannote_vad/$dataset/segments
# path_new_kaldi_segs=/data1/prachis/Amrit_sharc/tools_diar/data/$kaldi_dataset/$hyper/filewise_segments
path_new_kaldi_segs=$3
mkdir -p $path_new_kaldi_segs
for path in $(ls $path_segments)
	do

			g=$path_segments/$path
			python3 ../vad_benchmarking/SEG_TO_KALDI_SEG.py $g $path_new_kaldi_segs/$path
		
	done