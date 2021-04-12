import os
import numpy as np
import string
import tensorflow as tf
import math

# read A3M and convert letters into
# integers in the 0..20 range
def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename,"r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))
    print(type(seqs[0]))

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20
    # print(msa)


    return msa

# 1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
    beff = tf.reduce_sum(w)
    f_i = tf.reduce_sum(w[:,None,None]*msa1hot, axis=0) / beff + 1e-9
    h_i = tf.reduce_sum( -f_i * tf.math.log(f_i), axis=1)
    return tf.concat([f_i, h_i[:,None]], axis=1)


# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    with tf.name_scope('reweight'):

        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff

        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])

        id_mask = id_mtx > id_min

        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
    return w


# shrunk covariance inversion
def fast_dca(msa1hot, weights, penalty = 5):

    nr = tf.shape(msa1hot)[0]
    nc = tf.shape(msa1hot)[1]
    ns = tf.shape(msa1hot)[2]

    with tf.name_scope('covariance'):
        x = tf.reshape(msa1hot, (nr, nc * ns))
        num_points = tf.reduce_sum(weights) - tf.sqrt(tf.reduce_mean(weights))
        mean = tf.reduce_sum(x * weights[:,None], axis=0, keepdims=True) / num_points
        x = (x - mean) * tf.sqrt(weights[:,None])
        cov = tf.matmul(tf.transpose(x), x)/num_points

    with tf.name_scope('inv_convariance'):
        cov_reg = cov + tf.eye(nc * ns) * penalty / tf.sqrt(tf.reduce_sum(weights))
        inv_cov = tf.linalg.inv(cov_reg)
        
        x1 = tf.reshape(inv_cov,(nc, ns, nc, ns))
        x2 = tf.transpose(x1, [0,2,1,3])
        features = tf.reshape(x2, (nc, nc, ns * ns))
        
        x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(nc))
        apc = tf.reduce_sum(x3,0,keepdims=True) * tf.reduce_sum(x3,1,keepdims=True) / tf.reduce_sum(x3)
        contacts = (x3 - apc) * (1-tf.eye(nc))

    return tf.concat([features, contacts[:,:,None]], axis=2)

def cp(filename):
    # dict1= {'A': , 'b': 2, 'b': '3'}
    dict1={'A':0.31,	'R':-1.01	,'N':-0.6	,'D':-0.77,	'C':1.54,	'Q':-0.22,	'E':-0.64,	'G':0,	'H':0.13,	'I':1.8,	'L':1.7,
           'K':-0.99,	'M':1.23,	'F':1.79,	'P':0.72,	'S':-0.04,	'T':0.26,	'W':2.25,	'Y':0.96,	'V':1.22}
    dict2 = {'A': 1.28, 'R': 2.34, 'N': 1.6, 'D': 1.6, 'C': 1.77, 'Q': 1.56, 'E': 1.56, 'G': 0, 'H': 2.99,
             'I': 4.19, 'L': 2.59,
             'K': 1.89, 'M': 2.35, 'F': 2.94, 'P': 2.67, 'S': 1.31, 'T': 3.03, 'W': 3.21, 'Y': 2.94, 'V': 3.67}
    dict3 = {'A': 0.046, 'R': 0.291, 'N': 0.134, 'D': 0.105, 'C': 0.128, 'Q': 0.18, 'E': 0.151, 'G': 0, 'H': 0.23,
             'I': 0.186, 'L': 0.186,
             'K': 0.219, 'M': 0.221, 'F': 0.29, 'P': 0.131, 'S': 0.062, 'T': 0.108, 'W': 0.409, 'Y': 0.298, 'V': 0.14}
    dict4 = {'A': 1, 'R': 6.13, 'N': 2.95, 'D': 2.78, 'C': 2.43, 'Q': 3.95, 'E': 3.78, 'G': 0, 'H': 4.66,
             'I': 4, 'L': 4,
             'K': 4.77, 'M': 4.43, 'F': 5.89, 'P': 2.72, 'S': 1.6, 'T': 2.6, 'W': 8.08, 'Y': 6.46, 'V': 3}
    dict5 = {'A': 7.3, 'R': 11.1, 'N': 8, 'D': 9.2, 'C': 14.5, 'Q': 10.6, 'E': 11.4, 'G': 0, 'H': 10.2,
             'I': 16.1, 'L': 10.1,
             'K': 10.9, 'M': 10.4, 'F': 13.9, 'P': 17.8, 'S': 13.1, 'T': 16.7, 'W': 13.2, 'Y': 13.9, 'V': 17.2}
    dict6 = {'A': -0.01, 'R': 0.04, 'N': 0.06, 'D': 0.15, 'C': 0.12, 'Q': 0.05, 'E': 0.07, 'G': 0, 'H': 0.08,
             'I': -0.01, 'L': -0.01,
             'K': 0, 'M': 0.04, 'F': 0.03, 'P': 0, 'S': 0.11, 'T': 0.04, 'W': 0, 'Y': 0.03, 'V': 0.01}
    dict7 = {'A': 4.76, 'R': 4.3, 'N': 3.64, 'D': 5.69, 'C': 3.67, 'Q': 4.54, 'E': 5.48, 'G': 3.77, 'H': 2.84,
             'I': 4.81, 'L': 4.79,
             'K': 4.27, 'M': 4.25, 'F': 4.31, 'P': 0, 'S': 3.83, 'T': 3.87, 'W': 4.75, 'Y': 4.3, 'V': 4.86}
    dict8 = {'A': -5.1, 'R': 2.6, 'N': 4.7, 'D': 3.1, 'C': 3.8, 'Q': 0.2, 'E': -5.2, 'G': 5.6, 'H': -0.9,
             'I': -4.5, 'L': -5.4,
             'K': 1, 'M': -5.3, 'F': -2.4, 'P': 3.5, 'S': 3.2, 'T':0, 'W': 2.9, 'Y': 3.2, 'V': -6.3}
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    for line in open(filename,"r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))
    print(seqs[0])
    cp_feature=np.zeros((len(seqs[0]),8))

    for i in range (len(seqs[0])):
        for j in range (8):
            if j==0:
                cp_feature[i][j]=dict1[seqs[0][i]]
            if j==1:
                cp_feature[i][j]=dict2[seqs[0][i]]
            if j==2:
                cp_feature[i][j]=dict3[seqs[0][i]]
            if j==3:
                cp_feature[i][j]=dict4[seqs[0][i]]
            if j==4:
                cp_feature[i][j]=dict5[seqs[0][i]]
            if j==5:
                cp_feature[i][j]=dict6[seqs[0][i]]
            if j==6:
                cp_feature[i][j]=dict7[seqs[0][i]]
            if j==7:
                cp_feature[i][j]=dict8[seqs[0][i]]
    return cp_feature











