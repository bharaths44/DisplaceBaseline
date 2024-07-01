import os
import matplotlib.pyplot as plt
import fastcluster
import scipy.cluster.hierarchy as sch
import argparse
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import numpy as np
import scipy as sp
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import RandomizedSearchCV
from pathlib import Path
from pdb import set_trace as bp
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.special import softmax
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kaldi_io import read_vec_flt_ark, write_vec_flt
import os
import joblib
import pickle

import struct
from VBx import VBx
from kaldi_io import open_or_fd, BadSampleSize, UnknownMatrixHeader
from kaldi_io.kaldi_io import _read_compressed_mat, _read_mat_ascii



def _read_vec_binary(fd):
    # Data type,
    type = fd.read(3)
    if type == b'FV ':
        sample_size = 4  # floats
    elif type == b'DV ':
        sample_size = 8  # doubles
    else:
        raise BadSampleSize
    assert(sample_size > 0)
    # Dimension,
    assert fd.read(1) == b'\4'  # int-size
    vec_size = struct.unpack('<i', fd.read(4))[0]  # vector dim
    # Read whole vector,
    buf = fd.read(vec_size * sample_size)
    if sample_size == 4:
        ans = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8:
        ans = np.frombuffer(buf, dtype='float64')
    else:
        raise BadSampleSize
    return ans
    
    
def _read_mat_binary(fd):
    # Data type
    header = fd.read(3).decode()
    # 'CM', 'CM2', 'CM3' are possible values,
    if header.startswith('CM'):
        return _read_compressed_mat(fd, header)
    elif header.startswith('SM'):
        return _read_sparse_mat(fd, header)
    elif header == 'FM ':
        sample_size = 4  # floats
    elif header == 'DM ':
        sample_size = 8  # doubles
    else:
        raise UnknownMatrixHeader("The header contained '%s'" % header)
    assert(sample_size > 0)
    # Dimensions
    s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
    # Read whole matrix
    buf = fd.read(rows * cols * sample_size)
    if sample_size == 4:
        vec = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8:
        vec = np.frombuffer(buf, dtype='float64')
    else:
        raise BadSampleSize
    mat = np.reshape(vec, (rows, cols))
    return mat



def twoGMMcalib_lin(s, niters=20):
    """
    Train two-Gaussian GMM with shared variance for calibration of scores 's'
    Returns threshold for original scores 's' that "separates" the two gaussians
    and array of linearly callibrated log odds ratio scores.
    """
    weights = np.array([0.5, 0.5])
    means = np.mean(s) + np.std(s) * np.array([-1, 1])
    var = np.var(s)
    threshold = np.inf
    for _ in range(niters):
        lls = np.log(weights) - 0.5 * np.log(var) - 0.5 * (s[:, np.newaxis] - means)**2 / var
        gammas = softmax(lls, axis=1)
        cnts = np.sum(gammas, axis=0)
        weights = cnts / cnts.sum()
        means = s.dot(gammas) / cnts
        var = ((s**2).dot(gammas) / cnts - means**2).dot(weights)
        threshold = -0.5 * (np.log(weights**2 / var) - means**2 / var).dot([1, -1]) / (means/var).dot([1, -1])
    return threshold, lls[:, means.argmax()] - lls[:, means.argmin()]


def cos_similarity(x):
    """Compute cosine similarity matrix
    Args:
        x (np.ndarray): embeddings, 2D array, embeddings are in rows
    Returns:
        np.ndarray: cosine similarity matrix
    """

    assert x.ndim == 2, f'x has {x.ndim} dimensions, it must be matrix'
    x = x / (np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)) + 1.0e-32)
    assert np.allclose(np.ones_like(x[:, 0]), np.sum(np.square(x), axis=1))
    max_n_elm = 200000000
    step = max(max_n_elm // (x.shape[0] * x.shape[0]), 1)
    retval = np.zeros(shape=(x.shape[0], x.shape[0]), dtype=np.float64)
    x0 = np.expand_dims(x, 0)
    x1 = np.expand_dims(x, 1)
    for i in range(0, x.shape[1], step):
        product = x0[:, :, i:i+step] * x1[:, :, i:i+step]
        retval += np.sum(product, axis=2, keepdims=False)
    assert np.all(retval >= -1.0001), retval
    assert np.all(retval <= 1.0001), retval
    return retval

def read_plda(file_or_fd):
    """ Loads PLDA from a file in kaldi format (binary or text).
    Input:
        file_or_fd - file name or file handle with kaldi PLDA model.
    Output:
        Tuple (mu, tr, psi) defining a PLDA model using the kaldi parametrization:
        mu  - mean vector
        tr  - transform whitening within- and diagonalizing across-class covariance matrix
        psi - diagonal of the across-class covariance in the transformed space
    """
    fd = open_or_fd(file_or_fd)
    try:
        binary = fd.read(2)
        if binary == b'\x00B':
            assert(fd.read(7) == b'<Plda> ')
            plda_mean = _read_vec_binary(fd)
            plda_trans = _read_mat_binary(fd)
            plda_psi = _read_vec_binary(fd)
        else:
            assert(binary+fd.read(5) == b'<Plda> ')
            plda_mean = np.array(fd.readline().strip(' \n[]').split(), dtype=float)
            assert(fd.read(2) == b' [')
            plda_trans = _read_mat_ascii(fd)
            plda_psi = np.array(fd.readline().strip(' \n[]').split(), dtype=float)
        assert(fd.read(8) == b'</Plda> ')
    finally:
        if fd is not file_or_fd:
            fd.close()
    return plda_mean, plda_trans, plda_psi

def AHC(x):
    """Compute clustering labels using Agglomerative hierarichial clustering algorithm
    Args:
        x (np.ndarray): embeddings, 2D array, embeddings are in rows
    Returns:
        np.ndarray: labels
    """
    scr_mx = cos_similarity(x)
    thr, _ = twoGMMcalib_lin(scr_mx.ravel())
    scr_mx = squareform(-scr_mx, checks=False)
    lin_mat = fastcluster.linkage(scr_mx, method='average', preserve_input='False')
    del scr_mx
    adjust = abs(lin_mat[:, 2].min())
    lin_mat[:, 2] += adjust
    labels1st = fcluster(lin_mat, -(thr - 0.015) + adjust,criterion='distance') - 1
    qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
    qinit[range(len(labels1st)), labels1st] = 1.0
    #print(qinit,"\n",range(len(labels1st)),"\n", labels1st)
    qinit = softmax(qinit * 5.0, axis=1)
    #np.save("qinit_fromahc_exp.npy",qinit)
    fea = x
    #np.save('AHC_labels.npy',labels1st)
    psi=np.var(fea.T, axis=1, ddof=1)
    #np.save("psi_var.npy",psi)
    #np.save("AHC_qinit.npy",np.argsort(-qinit, axis=1)[:, 0])

    q, sp, L = VBx(
        fea, psi,
        pi=qinit.shape[1], gamma=qinit,
        maxIters=1, epsilon=1e-6,
        loopProb=0.99, Fa=1.5, Fb=2)
    labels1st = np.argsort(-q, axis=1)[:, 0]
    #print(q)
    #np.save("qint.npy",q)
    if q.shape[1] > 1:
        labels2nd = np.argsort(-q, axis=1)[:, 0]
        return labels2nd
#    print(len(labels1st),len(np.argsort(-qinit, axis=1)[:, 1]))
    return labels1st
    	




def labels_to_rttm(segments, labels, rttm_file, rttm_channel=1):
    labels = labels+1
    reco2segs = {}


    with open(segments, 'r') as segments_file:
        lines = segments_file.readlines()
    for line, label in zip(lines, labels):
        seg, reco, start, end = line.strip().split()
        start, end = float(start), float(end)
        
        try:
            if reco in reco2segs:
                reco2segs[reco] = "{} {},{},{}".format(reco2segs[reco],start,end,label)  #reco2segs[reco] + " " + start + "," + end + "," + label
            else:
                reco2segs[reco] = "{} {},{},{}".format(reco,start,end,label) #reco + " " + start + "," + end + "," + label
        except KeyError:
            raise RuntimeError("Missing label for segment {0}".format(seg))
        
    contiguous_segs = []
    for reco in sorted(reco2segs):
        segs = reco2segs[reco].strip().split()
        new_segs = ""
        for i in range(1, len(segs)-1):
            start, end, label = segs[i].split(',')
            next_start, next_end, next_label = segs[i+1].split(',')
            if float(end) > float(next_start):
                done = False
                avg = str((float(next_start) + float(end)) / 2.0)
                segs[i+1] = ','.join([avg, next_end, next_label])
                new_segs += " {},{},{}".format(start,avg,label)   #" " + start + "," + avg + "," + label
            else:
                new_segs += " {},{},{}".format(start,end,label)   #" " + start + "," + end + "," + label
        start, end, label = segs[-1].split(',')
        new_segs += " {},{},{}".format(start,end,label)  #" " + start + "," + end + "," + label
        contiguous_segs.append(reco + new_segs)
        
    merged_segs = []
    for reco_line in contiguous_segs:
        segs = reco_line.strip().split()
        reco = segs[0]
        new_segs = ""
        for i in range(1, len(segs)-1):
            start, end, label = segs[i].split(',')
            next_start, next_end, next_label = segs[i+1].split(',')
            if float(end) == float(next_start) and label == next_label:
                segs[i+1] = ','.join([start, next_end, next_label])
            else:
                new_segs += " {},{},{}".format(start,end,label)  #" " + start + "," + end + "," + label
        start, end, label = segs[-1].split(',')
        new_segs += " {},{},{}".format(start,end,label)  #" " + start + "," + end + "," + label
        merged_segs.append(reco + new_segs)
        
    with open(rttm_file, 'w') as rttm_writer:
        for reco_line in merged_segs:
            segs = reco_line.strip().split()
            reco = segs[0]
            for i in range(1, len(segs)):
                start, end, label = segs[i].strip().split(',')
                print("LANGUAGE {0} {1} {2:7.3f} {3:7.3f} <NA> <NA> L{4} <NA> <NA>".format(
                    reco, rttm_channel, float(start), float(end)-float(start), label), file=rttm_writer)



def merge_adjacent_labels(starts, ends, labels):
    """ Labeled segments defined as start and end times are compacted in such a way that
    adjacent or overlapping segments with the same label are merged. Overlapping
    segments with different labels are further adjusted not to overlap (the boundary
    is set in the middle of the original overlap).
    Input:
         starts - array of segment start times in seconds
         ends   - array of segment end times in seconds
         labels - array of segment labels (of any type)
    Outputs:
          starts, ends, labels - compacted and ajusted version of the input arrays
    """
    # Merge neighbouring (or overlaping) segments with the same label
    adjacent_or_overlap = np.logical_or(np.isclose(ends[:-1], starts[1:]), ends[:-1] > starts[1:])
    to_split = np.nonzero(np.logical_or(~adjacent_or_overlap, labels[1:] != labels[:-1]))[0]
    #print("starts shape:", starts.shape)
    #print("to_split shape:", to_split.shape)
    #print("to_split values:", to_split)
    starts = starts[np.r_[0, to_split+1]]
    ends = ends[np.r_[to_split, -1]]
    labels = labels[np.r_[0, to_split+1]]

    # Fix starts and ends times for overlapping segments
    overlaping = np.nonzero(starts[1:] < ends[:-1])[0]
    ends[overlaping] = starts[overlaping+1] = (ends[overlaping] + starts[overlaping+1]) / 2.0
    return starts, ends, labels

def spectral(X):
	autotune = AutoTune(
	    p_percentile_min=0.60,
	    p_percentile_max=0.95,
	    init_search_step=0.01,
	    search_level=3,
	    proxy=AutoTuneProxy.PercentileSqrtOverNME)
	refinement_options = RefinementOptions(
	    gaussian_blur_sigma=1,
	    p_percentile=0.95,
	    thresholding_soft_multiplier=0.01,
	    thresholding_type=ThresholdType.RowMax,
	    refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE)
	clusterer = SpectralClusterer(
	    min_clusters=2,
	    max_clusters=7,
	    autotune=autotune,
	    laplacian_type=None,
	    refinement_options=refinement_options,
	    custom_dist="cosine")
	labels = clusterer.predict(X)
	return labels
	
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_segments_list") # emb_segments.list or llk_segments.list
    parser.add_argument("rttm_path") # exps/exp1/ecapa_tdnn_voxlingua_speechbrain_language_embeddings/rttm_outputs
    parser.add_argument('mode', type=str, help='AHC,kmeans,spectral')
    args = parser.parse_args()
    embeddings_segments = np.genfromtxt(args.embeddings_segments_list, dtype=str)
    return embeddings_segments, args.rttm_path, args.mode

def out_filename(emb_file, rttm_path, clustering_hparams=""):
    rec_basename = os.path.splitext(os.path.basename(emb_file))[0]
    segment_basename = os.path.basename(os.path.dirname(emb_file))
    # bp()
    #out_file = os.path.join(rttm_path, segment_basename, f"{clustering_hparams}", f"{rec_basename}.rttm")
    out_file = os.path.join(rttm_path,  f"{rec_basename}.rttm")
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    return out_file

def read_embeddings(ark_list):
	embeddings = []
	labels = []
	for i in ark_list.keys():
		embeddings_dict = read_vec_flt_ark("indic_ark_files/"+i+".ark")
		for key, embedding in embeddings_dict:
			embeddings.append(embedding)
			label = ark_list[key.split("_")[0]]
			labels.append(label)
	embeddings = np.array(embeddings)
	labels = np.array(labels)
	return embeddings, labels
	
    
def main():
    embeddings_segments, rttm_path, mode = get_args()
    
    clustering_algo_dict = {"AHC": AHC}
    
    cluster = clustering_algo_dict[mode]
    
    
    for emb_file, segments in embeddings_segments:
        data = np.load(emb_file)
	
        labels = cluster(data)
        out_file = out_filename(emb_file, rttm_path, mode)
        labels_to_rttm(segments, labels, out_file)
        print(f"Clustering ({mode}) done for {emb_file}, saved to {out_file}\n")
        
main()
