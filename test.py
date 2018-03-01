from __future__ import division, print_function, unicode_literals
import numpy as np 
import os, time
from scipy.special import gammaln, digamma, psi
from scipy.sparse import coo_matrix
import _pickle as pickle


a = np.random.rand(100, 15276)

st = time.time()
b = psi(a) - psi(np.sum(a, 1))[:, np.newaxis]
print("runtime ", time.time() - st)
print(b.shape)