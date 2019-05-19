#!python
#cython: language_level=3
import numpy as np

def edit_distance(x, y, grammar, deltas):
	cdef int m = len(x)
	cdef int n = len(y)
	# pre-compute all operation costs
	grammar.validate(deltas)

	# First, compute all pairwise replacements
	cdef int K_rep = len(grammar._reps)
	Deltas_rep = np.zeros((K_rep, m, n))
	cdef double[:,:,:] Deltas_rep_view = Deltas_rep
	cdef int i
	cdef int j
	cdef int k
	for k in range(K_rep):
		delta = deltas[grammar._reps[k]]
		for i in range(m):
			for j in range(n):
				Deltas_rep_view[k, i, j] = delta(x[i], y[j])

	# Then, compute all deletions
	cdef int K_del = len(grammar._dels)
	Deltas_del = np.zeros((K_del, m))
	cdef double[:,:] Deltas_del_view = Deltas_del
	for k in range(K_del):
		delta = deltas[grammar._dels[k]]
		for i in range(m):
			Deltas_del_view[k, i] = delta(x[i], None)

	# Then, compute all insertions
	cdef int K_ins = len(grammar._inss)
	Deltas_ins = np.zeros((K_ins, n))
	cdef double[:,:] Deltas_ins_view = Deltas_ins
	for k in range(K_ins):
		delta = deltas[grammar._inss[k]]
		for j in range(n):
			Deltas_ins_view[k, j] = delta(None, y[j])

	# retrieve the adjacency list representation for
	# the grammar
	start_idx, accpt_idxs, adj_rep, adj_del, adj_ins = grammar.adjacency_lists()

	# Initialize the dynamic programming matrices
	# for all nonterminals
	cdef int R = len(grammar._nonterminals)
	Ds = np.full((R, m+1, n+1), np.inf)
	cdef double[:,:,:] Ds_view = Ds
	# initialize last entry for all accepting symbols
	cdef int nont
	for nont in accpt_idxs:
		Ds_view[nont, m, n] = 0.

	# initialize last column for all symbols
	cdef int r
	cdef int s
	for i in range(m-1,-1,-1):
		for r in range(R):
			for (k, s) in adj_del[r]:
				Ds_view[r, i, n] = Deltas_del_view[k, i] + Ds_view[s, i+1, n]

	# initialize last row for all symbols
	for j in range(n-1,-1,-1):
		for r in range(R):
			for (k, s) in adj_ins[r]:
				Ds_view[r, m, j] = Deltas_ins_view[k, j] + Ds_view[s, m, j+1]

	# perform the remaining computation
	cdef double min_cost
	cdef double current_cost
	for i in range(m-1,-1,-1):
		for j in range(n-1,-1,-1):
			for r in range(R):
				min_cost = np.inf
				# first, consider replacements
				for (k, s) in adj_rep[r]:
					current_cost = Deltas_rep_view[k, i, j] + Ds_view[s, i+1, j+1]
					if(current_cost < min_cost):
						min_cost = current_cost
				# then, consider deletions
				for (k, s) in adj_del[r]:
					current_cost = Deltas_del_view[k, i] + Ds_view[s, i+1, j]
					if(current_cost < min_cost):
						min_cost = current_cost
				# finally, consider insertions
				for (k, s) in adj_ins[r]:
					current_cost = Deltas_ins_view[k, j] + Ds_view[s, i, j+1]
					if(current_cost < min_cost):
						min_cost = current_cost
				# set new entry to minimum
				Ds_view[r, i, j] = min_cost

	# return the cost at the start symbol
	return Ds[start_idx, 0, 0]
