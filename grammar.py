class RuleEntry:
	def __init__(self):
		self._reps = []
		self._dels = []
		self._inss = []

	def __repr__(self):
		op_str = 'replacements: '
		for rep_edge in self._reps:
			op_str += 'via ' + rep_edge[0] + ' to ' + rep_edge[1] + ' '
		op_str += '\ndeletions: '
		for del_edge in self._dels:
			op_str += 'via ' + del_edge[0] + ' to ' + del_edge[1] + ' '
		op_str += '\ninsertions: '
		for ins_edge in self._inss:
			op_str += 'via ' + ins_edge[0] + ' to ' + ins_edge[1] + ' '
		return op_str

	def __str__(self):
		return self.__repr__()


def string_to_index_map(lst):
	dct = {}
	for i in range(len(lst)):
		dct[lst[i]] = i
	return dct

def string_to_index_list(lst, dct):
	idx_list = []
	for e in lst:
		idx = dct[e]
		if(idx is None):
			raise ValueError('unknown string: %s' % str(e))
		idx_list.append(idx)
	return idx_list

def string_to_index_tuple_list(lst, op_dct, nont_dct):
	idx_list = []
	for e in lst:
		op_idx = op_dct[e[0]]
		if(op_idx is None):
			raise ValueError('unknown operation string: %s' % str(e[0]))
		nont_idx = nont_dct[e[1]]
		if(nont_idx is None):
			raise ValueError('unknown nonterminal string: %s' % str(e[1]))
		idx_list.append((op_idx, nont_idx))
	return idx_list

class Grammar:
	def __init__(self, start, accepting, nonterminals = None, reps = None, dels = None, inss = None, rules = None):
		self._start = start
		self._accepting = accepting
		if(nonterminals is None):
			self._nonterminals = []
		else:
			self._nonterminals = nonterminals
		if(reps is None):
			self._reps = []
		else:
			self._reps = reps
		if(dels is None):
			self._dels = []
		else:
			self._dels = dels
		if(inss is None):
			self._inss = []
		else:
			self._inss = inss
		if(rules is None):
			self._rules = {}
		else:
			self._rules = rules
		self._initialize_rule_entry(start)
		for nont in accepting:
			self._initialize_rule_entry(nont)

	def __repr__(self):
		op_str = 'Start at ' + str(self._start) + '. Rules:\n'
		for nont in self._nonterminals:
			op_str += 'From ' + nont 
			if(nont in self._accepting):
				op_str += ' (accepting)'
			op_str += ':\n'
			op_str += str(self._rules[nont])
		return op_str

	def __str__(self):
		return self.__repr__()

	def _initialize_rule_entry(self, source):
		if(source not in self._nonterminals):
			self._nonterminals.append(source)
		if(source not in self._rules):
			self._rules[source] = RuleEntry()

	def append_replacement(self, source, target, operation):
		if(operation not in self._reps):
			self._reps.append(operation)
		self._initialize_rule_entry(source)
		self._initialize_rule_entry(target)
		self._rules[source]._reps.append((operation, target))

	def append_deletion(self, source, target, operation):
		if(operation not in self._dels):
			self._dels.append(operation)
		self._initialize_rule_entry(source)
		self._initialize_rule_entry(target)
		self._rules[source]._dels.append((operation, target))

	def append_insertion(self, source, target, operation):
		if(operation not in self._inss):
			self._inss.append(operation)
		self._initialize_rule_entry(source)
		self._initialize_rule_entry(target)
		self._rules[source]._inss.append((operation, target))

	def size(self):
		return len(self._nonterminals)

	def start(self):
		return self._start

	def nonterminals(self):
		return self._nonterminals

	def adjacency_lists(self):
		# first, create maps from string to index representations
		nont_map = string_to_index_map(self._nonterminals)
		reps_map = string_to_index_map(self._reps)
		dels_map = string_to_index_map(self._dels)
		inss_map = string_to_index_map(self._inss)
		# translate the start symbol to an index
		start_idx = nont_map[self._start]
		# translate the accepting symbol list
		accpt_idxs = string_to_index_list(self._accepting, nont_map)
		# then, create an adjacency list representation for all replacements,
		# deletions, and insertions separately
		rep_adj = []
		del_adj = []
		ins_adj = []
		for nont in self._nonterminals:
			rule_entry = self._rules[nont]
			rep_adj.append(string_to_index_tuple_list(rule_entry._reps, reps_map, nont_map))
			del_adj.append(string_to_index_tuple_list(rule_entry._dels, dels_map, nont_map))
			ins_adj.append(string_to_index_tuple_list(rule_entry._inss, inss_map, nont_map))
		return start_idx, accpt_idxs, rep_adj, del_adj, ins_adj

