from dtw_pure_python import multi_dtw_pure_python
from dtw import dtw
from dtw import dtw_euclidean
from dtw import dtw_manhattan
from dtw import dtw_string
from multiprocess import pairwise_distances_symmetric
import numpy as np
import time

m = 10
K = 10
L = 100
seqs = []
for i in range(m):
	seqs.append(np.random.random((L, K)))

def l1_distance(x, y):
	return np.sum(np.abs(x - y))

def l2_distance(x, y):
	return np.sqrt(np.sum(np.square(x - y)))

start = time.time()
D = multi_dtw_pure_python(seqs, seqs, l2_distance)
end = time.time()
print('pure python computed Euclidean dtw distance between %d sequences of length %d in %g seconds.' % (m, L, end - start))

start = time.time()
D2 = pairwise_distances_symmetric(seqs, dtw_euclidean)
end = time.time()
print('cython computed Euclidean dtw distance between %d sequences of length %d in %g seconds.' % (m, L, end - start))

print('control: frobenius norm between dtw matrices: %g' % (np.sum(np.square(D - D2))))

start = time.time()
D = multi_dtw_pure_python(seqs, seqs, l1_distance)
end = time.time()
print('pure python computed Manhattan dtw distance between %d sequences of length %d in %g seconds.' % (m, L, end - start))

start = time.time()
D2 = pairwise_distances_symmetric(seqs, dtw_manhattan)
end = time.time()
print('cython computed Manhattan dtw distance between %d sequences of length %d in %g seconds.' % (m, L, end - start))

print('control: frobenius norm between dtw matrices: %g' % (np.sum(np.square(D - D2))))

m = 20
L = 100
seqs = []
for i in range(m):
	seq = []
	for l in range(L):
		if(np.random.random() < 0.5):
			seq.append('a')
		else:
			seq.append('b')
	seq = ''.join(seq)
	seqs.append(seq)

def kron_distance(x, y):
	if(x == y):
		return 0.
	else:
		return 1.

start = time.time()
D = multi_dtw_pure_python(seqs, seqs, kron_distance)
end = time.time()
print('pure python computed string dtw distance between %d sequences of length %d in %g seconds.' % (m, L, end - start))

start = time.time()
D2 = pairwise_distances_symmetric(seqs, dtw_string)
end = time.time()
print('cython computed string dtw distance between %d sequences of length %d in %g seconds.' % (m, L, end - start))

print('control: frobenius norm between dtw matrices: %g' % (np.sum(np.square(D - D2))))
