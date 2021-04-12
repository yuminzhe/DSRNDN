import tensorflow as tf
from network.utils import *
import numpy as np
np.set_printoptions(threshold=np.inf)
msa=parse_a3m('/Users/yuminzhe/Downloads/trRosetta-master/example/3t63M.a3m')

msa1hot  = tf.one_hot(msa, 21, dtype=tf.float32)
w = reweight(msa1hot, 0.8)
# 1D features
f1d_seq = msa1hot[0, :, :20]
# PSSM and positional entropy features
f1d_pssm = msa2pssm(msa1hot, w)
f1d=tf.Session().run(f1d_pssm)
# cp features
f1d_cp=cp('/Users/yuminzhe/Downloads/trRosetta-master/example/3t63M.a3m')
f1d_cp=tf.Session().run(f1d_cp)


# 2D features
f2d_dca=fast_dca(msa1hot,w)
f2d_dcad=tf.Session().run(f2d_dca)
