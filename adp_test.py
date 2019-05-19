import unittest
import grammar
from adp import edit_distance

def kron_distance(x, y):
	if(x == y):
		return 0.
	else:
		return 1.

class TestGrammarMethods(unittest.TestCase):

	def test_edit_distance(self):
		gra = grammar.Grammar('A', ['A'])
		gra.append_replacement('A', 'A', 'rep')
		gra.append_deletion('A', 'A', 'del')
		gra.append_insertion('A', 'A', 'ins')
		left = ['a', 'b', 'c']
		right = ['a', 'd', 'e', 'c']
		deltas = {'rep' : kron_distance, 'del' : kron_distance, 'ins' : kron_distance}
		expected = 2.
		actual = edit_distance(left, right, gra, deltas)
		self.assertEqual(expected, actual)

if __name__ == '__main__':
	unittest.main()
