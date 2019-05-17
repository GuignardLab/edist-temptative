import numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free

def multi_dtw(double[:,:] Xs, double[:,:] Ys):
	# iterate over all sequence combinations using OMP parallelization
	cdef int m = len(Xs)
	cdef int n = len(Ys)
	D = np.zeros((m, n))
	cdef double [:,:] D_view = D
	cdef int i
	cdef int j
	cdef double d_ij
	for i in prange(m, nogil=True):
		for j in prange(n):
			D_view[i, j] = l1_dtw(Xs[i], Ys[j])
	return D

def dtw(x, y, delta):
	# First, compute all pairwise replacements
	Delta = np.zeros((len(x), len(y)))
	cdef double [:,:] Delta_view = Delta
	cdef int i
	cdef int j
	for i in range(len(x)):
		for j in range(len(y)):
			Delta_view[i, j] = delta(x[i], y[j])
	# Then, compute the dynamic time warping
	# distance
	D = np.zeros((len(x), len(y)))
	cdef double [:,:] D_view = D
	# initialize last entry
	D_view[-1, -1] = Delta_view[-1, -1]
	# initialize last column
	for i in range(len(x)-1):
		D_view[i,-1] = Delta_view[i,-1] + D_view[i+1,-1]
	# initialize last row
	for j in range(len(y)-1):
		D_view[-1,j] = Delta_view[-1,j] + D_view[-1,j+1]
	
	for i in range(len(x)-1):
		for j in range(len(y)-1):
			D_view[i,j] = Delta_view[i,j] + min3(D_view[i+1,j+1], D_view[i,j+1], D_view[i+1,j])
	return D_view[0,0]

cdef double l1_dtw(double[:] x, double[:] y) nogil:
	# Then, compute the dynamic time warping
	# distance
	cdef int m = len(x)
	cdef int n = len(y)
	cdef double **D = <double **> malloc(m * sizeof(double*))
	cdef int i
	cdef int j
	for i in range(m):
		D[i] = <double *> malloc(n * sizeof(double))
	try:
		# initialize last entry
		D[m-1][n-1] = l1_dist(x[-1], y[-1])
		# initialize last column
		for i in range(m-1):
			D[i][n-1] = l1_dist(x[i], y[-1]) + D[i+1][-1]
		# initialize last row
		for j in range(n-1):
			D[m-1][j] = l1_dist(x[-1], y[j]) + D[-1][j+1]

		for i in range(m-1):
			for j in range(n-1):
				D[i][j] = l1_dist(x[i], y[j]) + min3(D[i+1][j+1], D[i][j+1], D[i+1][j])
		return D[0][0]
	finally:
		for i in range(m):
			free(D[i])
		free(D)

cdef double l1_dist(double a, double b) nogil:
	cdef double d = a - b
	if(d < 0):
		return -d
	else:
		return d

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
