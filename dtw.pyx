import numpy as np
from cython.parallel import prange
from joblib import Parallel, delayed

def idx_producer(const int m, const int n):
	cdef int i
	cdef int j
	for i in range(m):
		for j in range(n):
			yield (i,j)


def multi_dtw(list Xs, list Ys, delta, int num_jobs = 2):
	# perform dynamic time warping on all pairwise sequence combinations
	cdef int K = len(Xs)
	cdef int L = len(Ys)
	res = Parallel(n_jobs = num_jobs, pre_dispatch='1.5*n_jobs')(delayed(dtw)(Xs[k], Ys[l], delta) for (k,l) in idx_producer(K,L))
	return np.reshape(res, (K, L))

def dtw(x, y, delta):
	cdef int m = len(x)
	cdef int n = len(y)
	# First, compute all pairwise replacements using joblib for
	# parallelization
	res = Parallel(n_jobs = 2, pre_dispatch='1.5*n_jobs')(delayed(delta)(x[i], y[j]) for (i,j) in idx_producer(m,n))
	Delta = np.reshape(res, (m, n))

	# Then, compute the dynamic time warping
	# distance
	D = np.zeros((m,n))
	return dtw_c(Delta, D)

cdef double dtw_c(const double[:,:] Delta, double[:,:] D) nogil:
	cdef int i
	cdef int j
	# initialize last entry
	D[-1, -1] = Delta[-1, -1]
	# compute last column
	for i in range(D.shape[0]-2,-1,-1):
		D[i,-1] = Delta[i,-1] + D[i+1,-1]
	# compute last row
	for j in range(D.shape[1]-2,-1,-1):
		D[-1,j] = Delta[-1,j] + D[-1,j+1]
	# compute remaining matrix
	for i in range(D.shape[0]-2,-1,-1):
		for j in range(D.shape[1]-2,-1,-1):
			D[i,j] = Delta[i,j] + min3(D[i+1,j+1], D[i,j+1], D[i+1,j])
	# return dtw distance
	return D[0,0]

cdef double min3(double a, double b, double c) nogil:
	if(a < b):
		if(a < c):
			return a
		else:
			return c
	else:
		if(b < c):
			return b
		else:
			return c
