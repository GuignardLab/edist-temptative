import numpy as np
from cython.parallel import prange
import multiprocessing as mp

def idx_producer(const int m, const int n):
	cdef int i
	cdef int j
	for i in range(m):
		for j in range(n):
			yield (i,j)

def dtw_with_idxs(x, y, delta, k, l):
	return (k, l, dtw(x, y, delta))

def multi_dtw(list Xs, list Ys, delta, int num_jobs = 8):
	# perform dynamic time warping on all pairwise sequence combinations
	cdef int K = len(Xs)
	cdef int L = len(Ys)
	# set up a parallel processing pool
	pool = mp.Pool(num_jobs)
	# set up the result matrix
	D = np.zeros((K,L))
	cdef double[:,:] D_view = D

	# set up the dtw index function
	# and the callback function
	def callback(tpl):
		D_view[tpl[0], tpl[1]] = tpl[2]

	# start off all parallel processing jobs
	for k in range(K):
		for l in range(L):
			pool.apply_async(dtw_with_idxs, args=(Xs[k], Ys[l], delta, k, l), callback=callback)

	# wait for the jobs to finish
	pool.close()
	pool.join()

	return D

def delta_with_idxs(x, y, delta, i, j):
	return (i, j, delta(x, y))

def dtw(x, y, delta):
	cdef int m = len(x)
	cdef int n = len(y)
	# First, compute all pairwise replacements
	Delta = np.zeros((m, n))
	cdef double[:,:] Delta_view = Delta
	cdef int i
	cdef int j
	for i in range(m):
		for j in range(n):
			Delta_view[i,j] = delta(x[i], y[j])

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
