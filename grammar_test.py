import unittest
import edist.grammar as grammar

class TestGrammarMethods(unittest.TestCase):

	def test_construction(self):
		gra = grammar.Grammar('A', ['A'])
		gra.append_replacement('A', 'A', 'rep')
		gra.append_deletion('A', 'A', 'del')
		gra.append_insertion('A', 'A', 'ins')
		expected = 'Start at A. Rules:\nFrom A (accepting):\nreplacements: via rep to A \ndeletions: via del to A \ninsertions: via ins to A '
		self.assertEqual(expected, str(gra))
		self.assertEqual(1, gra.size())
		self.assertEqual('A', gra.start())
		self.assertEqual(['A'], gra.nonterminals())

	def test_string_to_index_map(self):
		lst = ['A', 'B', 'C']
		expected = {'A' : 0, 'B' : 1, 'C' : 2}

		actual = grammar.string_to_index_map(lst)
		self.assertEqual(expected, actual)

	def test_string_to_index_list(self):
		lst = ['A', 'B', 'A']
		dct = {'A' : 0, 'B' : 1}
		expected = [0, 1, 0]

		actual = grammar.string_to_index_list(lst, dct)
		self.assertEqual(expected, actual)

	def test_string_to_index_tuple_list(self):
		lst = [('a', 'A'), ('a', 'B'), ('b', 'A')]
		op_dct = {'a' : 0, 'b' : 1}
		nont_dct = {'A' : 0, 'B' : 1}
		expected = [(0, 0), (0, 1), (1, 0)]

		actual = grammar.string_to_index_tuple_list(lst, op_dct, nont_dct)
		self.assertEqual(expected, actual)

	def test_adjacency_lists(self):
		gra = grammar.Grammar('A', ['A'])
		gra.append_replacement('A', 'A', 'rep')
		gra.append_deletion('A', 'A', 'del')
		gra.append_insertion('A', 'A', 'ins')
		expected = [[(0,0)]]

		start_idx, accpt_idxs, actual_reps, actual_dels, actual_inss = gra.adjacency_lists()
		self.assertEqual(0, start_idx)
		self.assertEqual([0], accpt_idxs)
		self.assertEqual(expected, actual_reps)
		self.assertEqual(expected, actual_dels)
		self.assertEqual(expected, actual_inss)


if __name__ == '__main__':
	unittest.main()
