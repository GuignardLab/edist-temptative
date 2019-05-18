from dtw_pure_python import multi_dtw_pure_python
from dtw import multi_dtw
import numpy as np
import time

m = 30
L = 50
seqs = []
for i in range(m):
	seqs.append(np.random.random(L))

def l1_distance(x, y):
	return abs(x - y)

start = time.time()
D = multi_dtw_pure_python(seqs, seqs, l1_distance)
end = time.time()
print('pure python computed dtw distance between %d sequences of length %d in %g seconds.' % (m, L, end - start))

start = time.time()
D2 = multi_dtw(seqs, seqs, l1_distance, 4)
end = time.time()
print('cython computed dtw distance between %d sequences of length %d in %g seconds.' % (m, L, end - start))

print('control: frobenius norm between dtw matrices: %g' % (np.sum(np.square(D - D2))))
