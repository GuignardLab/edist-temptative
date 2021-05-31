"""
Tests ADP grammars.

"""
# Copyright (C) 2019-2021
# Benjamin Paaßen
# AG Machine Learning
# Bielefeld University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import edist.adp as adp

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestGrammarMethods(unittest.TestCase):

	def test_construction(self):
		gra = adp.Grammar('A', ['A'])
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

		actual = adp.string_to_index_map(lst)
		self.assertEqual(expected, actual)

	def test_string_to_index_list(self):
		lst = ['A', 'B', 'A']
		dct = {'A' : 0, 'B' : 1}
		expected = [0, 1, 0]

		actual = adp.string_to_index_list(lst, dct)
		self.assertEqual(expected, actual)

	def test_string_to_index_tuple_list(self):
		lst = [('a', 'A'), ('a', 'B'), ('b', 'A')]
		op_dct = {'a' : 0, 'b' : 1}
		nont_dct = {'A' : 0, 'B' : 1}
		expected = [(0, 0), (0, 1), (1, 0)]

		actual = adp.string_to_index_tuple_list(lst, op_dct, nont_dct)
		self.assertEqual(expected, actual)

	def test_adjacency_lists(self):
		gra = adp.Grammar('A', ['A', 'B', 'C'])
		gra.append_replacement('A', 'A', 'rep')
		gra.append_deletion('A', 'B', 'del')
		gra.append_insertion('A', 'C', 'ins')
		gra.append_replacement('B', 'A', 'rep')
		gra.append_deletion('B', 'B', 'skdel')
		gra.append_insertion('B', 'C', 'ins')
		gra.append_replacement('C', 'A', 'rep')
		gra.append_insertion('C', 'C', 'skins')

		start_idx, accpt_idxs, actual_reps, actual_dels, actual_inss = gra.adjacency_lists()
		self.assertEqual(0, start_idx)
		self.assertEqual([0, 1, 2], accpt_idxs)
		expected = [[(0,0)], [(0,0)], [(0,0)]]
		self.assertEqual(expected, actual_reps)
		expected = [[(0,1)], [(1,1)], []]
		self.assertEqual(expected, actual_dels)
		expected = [[(0,2)], [(0,2)], [(1, 2)]]
		self.assertEqual(expected, actual_inss)

	def test_inverse_adjacency_lists(self):
		gra = adp.Grammar('A', ['A', 'B', 'C'])
		gra.append_replacement('A', 'A', 'rep')
		gra.append_deletion('A', 'B', 'del')
		gra.append_insertion('A', 'C', 'ins')
		gra.append_replacement('B', 'A', 'rep')
		gra.append_deletion('B', 'B', 'skdel')
		gra.append_insertion('B', 'C', 'ins')
		gra.append_replacement('C', 'A', 'rep')
		gra.append_insertion('C', 'C', 'skins')

		start_idx, accpt_idxs, actual_reps, actual_dels, actual_inss = gra.inverse_adjacency_lists()
		self.assertEqual(0, start_idx)
		self.assertEqual([0, 1, 2], accpt_idxs)
		expected = [[(0,0), (0, 1), (0, 2)], [], []]
		self.assertEqual(expected, actual_reps)
		expected = [[], [(0,0), (1, 1)], []]
		self.assertEqual(expected, actual_dels)
		expected = [[], [], [(0, 0), (0, 1), (1, 2)]]
		self.assertEqual(expected, actual_inss)


if __name__ == '__main__':
	unittest.main()
