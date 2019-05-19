import multiprocessing as mp
import numpy as np

def dist_with_indices(k, l, dist, x, y):
	print(dist(x, y))
	return (k, l, dist(x, y))

def dist_with_indices_and_delta(k, l, dist, x, y, delta):
	return (k, l, dist(x, y, delta))

def pairwise_distances(Xs, Ys, dist, delta = None, num_jobs = 8):
	K = len(Xs)
	L = len(Ys)
	# set up a parallel processing pool
	pool = mp.Pool(num_jobs)
	# set up the result matrix
	D = np.zeros((K,L))

	# set up the dtw index function
	# and the callback function
	def callback(tpl):
		D[tpl[0], tpl[1]] = tpl[2]

	# start off all parallel processing jobs
	if(not delta):
		for k in range(K):
			for l in range(L):
				pool.apply_async(dist_with_indices, args=(k, l, dist, Xs[k], Ys[l]), callback=callback)
	else:
		for k in range(K):
			for l in range(L):
				pool.apply_async(dist_with_indices_and_delta, args=(k, l, dist, Xs[k], Ys[l], delta), callback=callback)

	# wait for the jobs to finish
	pool.close()
	pool.join()

	# return the distance matrix
	return D

def pairwise_distances_symmetric(Xs, dist, delta = None, num_jobs = 8):
	K = len(Xs)
	# set up a parallel processing pool
	pool = mp.Pool(num_jobs)
	# set up the result matrix
	D = np.zeros((K,K))

	# set up the dtw index function
	# and the callback function
	def callback(tpl):
		D[tpl[0], tpl[1]] = tpl[2]

	# start off all parallel processing jobs
	if(not delta):
		for k in range(K):
			for l in range(k+1, K):
				pool.apply_async(dist_with_indices, args=(k, l, dist, Xs[k], Xs[l]), callback=callback)
	else:
		for k in range(K):
			for l in range(k+1, K):
				pool.apply_async(dist_with_indices_and_delta, args=(k, l, dist, Xs[k], Xs[l], delta), callback=callback)

	# wait for the jobs to finish
	pool.close()
	pool.join()

	# add the lower diagonal
	D += np.transpose(D)

	# return the distance matrix
	return D

def pairwise_computation(Xs, Ys, dist, delta = None, num_jobs = 8):
	K = len(Xs)
	L = len(Ys)
	# set up a parallel processing pool
	pool = mp.Pool(num_jobs)
	# set up the result matrix
	D = []
	for k in range(K):
		res_k = []
		for l in range(L):
			res_k.append(None)
		D.append(res_k)

	# set up the dtw index function
	# and the callback function
	def callback(tpl):
		D[tpl[0]][tpl[1]] = tpl[2]

	# start off all parallel processing jobs
	if(not delta):
		for k in range(K):
			for l in range(L):
				pool.apply_async(dist_with_indices, args=(k, l, dist, Xs[k], Ys[l]), callback=callback)
	else:
		for k in range(K):
			for l in range(L):
				pool.apply_async(dist_with_indices_and_delta, args=(k, l, dist, Xs[k], Ys[l], delta), callback=callback)

	# wait for the jobs to finish
	pool.close()
	pool.join()

	# return the distance matrix
	return D

def pairwise_computation_symmetric(Xs, dist, delta = None, num_jobs = 8):
	K = len(Xs)
	# set up a parallel processing pool
	pool = mp.Pool(num_jobs)
	# set up the result matrix
	D = []
	for k in range(K):
		res_k = []
		for l in range(K):
			res_k.append(None)
		D.append(res_k)

	# set up the dtw index function
	# and the callback function
	def callback(tpl):
		D[tpl[0]][tpl[1]] = tpl[2]

	# start off all parallel processing jobs
	if(not delta):
		for k in range(K):
			for l in range(k, K):
				pool.apply_async(dist_with_indices, args=(k, l, dist, Xs[k], Xs[l]), callback=callback)
	else:
		for k in range(K):
			for l in range(k, K):
				pool.apply_async(dist_with_indices_and_delta, args=(k, l, dist, Xs[k], Xs[l], delta), callback=callback)

	# wait for the jobs to finish
	pool.close()
	pool.join()

	# add the lower diagonal
	for k in range(K):
		for l in range(k):
			D[k][l] = D[l][k]

	# return the distance matrix
	return D
