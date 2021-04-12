import tensorflow as tf
from network.utils import *
import numpy as np
# np.set_printoptions(threshold=np.inf)
msa=parse_a3m('example/T0957s2.a3m')

msa1hot  = tf.one_hot(msa, 21, dtype=tf.float32)
w = reweight(msa1hot, 0.8)
# 1D features
f1d_seq = msa1hot[0, :, :20]
# PSSM and positional entropy features
f1d_pssm = msa2pssm(msa1hot, w)
f1d=tf.Session().run(f1d_pssm)
print("PSSM and positional entropy:")
print(f1d.shape)
print(f1d)

# cp features
f1d_cp=cp('example/T0957s2.a3m')
print('physicochemical:')
print(f1d_cp.shape)
print(f1d_cp)


# 2D features
f2d_dca=fast_dca(msa1hot,w)
f2d_dca=tf.Session().run(f2d_dca)
print("Couplings:")
print(f2d_dca.shape)
print(f2d_dca)
