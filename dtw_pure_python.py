import numpy as np

def multi_dtw_pure_python(Xs, Ys, delta):
	D = np.zeros((len(Xs), len(Ys)))
	for i in range(len(Xs)):
		for j in range(len(Ys)):
			D[i, j] = dtw_pure_python(Xs[i], Ys[j], delta)
	return D

def dtw_pure_python(x, y, delta):
	# First, compute all pairwise replacements
	Delta = np.zeros((len(x), len(y)))
	for i in range(len(x)):
		for j in range(len(y)):
			Delta[i, j] = delta(x[i], y[j])
	# Then, compute the dynamic time warping
	# distance
	D = np.zeros((len(x), len(y)))
	# initialize last entry
	D[-1, -1] = Delta[-1, -1]
	# initialize last column
	for i in range(len(x)-2,-1,-1):
		D[i,-1] = Delta[i,-1] + D[i+1,-1]
	# initialize last row
	for j in range(len(y)-2,-1,-1):
		D[-1,j] = Delta[-1,j] + D[-1,j+1]

	for i in range(len(x)-2,-1,-1):
		for j in range(len(y)-2,-1,-1):
			D[i,j] = Delta[i,j] + min3(D[i+1,j+1], D[i,j+1], D[i+1,j])
	return D[0,0]

def min3(a, b, c):
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
