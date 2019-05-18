import numpy as np
from cython.parallel import prange
from libc.math cimport sqrt
cimport cython

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

@cython.boundscheck(False)
def dtw_numeric(double[:] x, double[:] y):
	cdef int m = len(x)
	cdef int n = len(y)
	# First, compute all pairwise replacements
	# using OMP parallelization
	cdef int i
	cdef int j
	Delta = np.zeros((m, n))
	cdef double[:,:] Delta_view = Delta
	for i in prange(m, nogil=True):
		for j in prange(n):
			if(x[i] > y[j]):
				Delta_view[i,j] = x[i] - y[j]
			else:
				Delta_view[i,j] = y[j] - x[i]
	# Then, compute the dynamic time warping
	# distance
	D = np.zeros((m,n))
	return dtw_c(Delta, D)

@cython.boundscheck(False)
def dtw_manhattan(double[:,:] x, double[:,:] y):
	cdef int m = x.shape[0]
	cdef int n = y.shape[0]
	cdef int K = x.shape[1]
	if(y.shape[1] != K):
		raise ValueError('x and y do not have the same dimensionality (%d versus %d)' % (x.shape[1], y.shape[1]))
	# First, compute all pairwise replacements
	# using OMP parallelization
	cdef int i
	cdef int j
	Delta = np.zeros((m, n))
	cdef double[:,:] Delta_view = Delta
	cdef int k
	cdef double diff
	for i in prange(m, nogil=True):
		for j in prange(n):
			for k in prange(K):
				diff = x[i,k] - y[j,k]
				if(diff < 0):
					Delta_view[i, j] -= diff
				else:
					Delta_view[i, j] += diff
	# Then, compute the dynamic time warping
	# distance
	D = np.zeros((m,n))
	return dtw_c(Delta, D)

@cython.boundscheck(False)
def dtw_euclidean(double[:,:] x, double[:,:] y):
	cdef int m = x.shape[0]
	cdef int n = y.shape[0]
	cdef int K = x.shape[1]
	if(y.shape[1] != K):
		raise ValueError('x and y do not have the same dimensionality (%d versus %d)' % (x.shape[1], y.shape[1]))
	# First, compute all pairwise replacements
	# using OMP parallelization
	cdef int i
	cdef int j
	Delta = np.zeros((m, n))
	cdef double[:,:] Delta_view = Delta
	cdef int k
	cdef double diff
	for i in prange(m, nogil=True):
		for j in prange(n):
			for k in prange(K):
				diff = x[i,k] - y[j,k]
				Delta_view[i, j] += diff * diff
			Delta_view[i, j] = sqrt(Delta_view[i, j])
	# Then, compute the dynamic time warping
	# distance
	D = np.zeros((m,n))
	return dtw_c(Delta, D)

@cython.boundscheck(False)
def dtw_string(str x, str y):
	cdef int m = len(x)
	cdef int n = len(y)
	# First, compute all pairwise replacements
	# using OMP parallelization
	cdef int i
	cdef int j
	Delta = np.zeros((m, n))
	cdef double[:,:] Delta_view = Delta
	for i in range(m):
		for j in range(n):
			if(x[i] == y[j]):
				Delta_view[i, j] = 0.
			else:
				Delta_view[i, j] = 1.
	# Then, compute the dynamic time warping
	# distance
	D = np.zeros((m,n))
	return dtw_c(Delta, D)

@cython.boundscheck(False)
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
