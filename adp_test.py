import unittest
from edist.alignment import Alignment
from edist.adp import Grammar
from edist.adp import edit_distance
from edist.adp import backtrace

def kron_distance(x, y):
    if(x == y):
        return 0.
    else:
        return 1.

class TestGrammarMethods(unittest.TestCase):

    def test_edit_distance(self):
        gra = Grammar('A', ['A'])
        gra.append_replacement('A', 'A', 'rep')
        gra.append_deletion('A', 'A', 'del')
        gra.append_insertion('A', 'A', 'ins')
        left = ['a', 'b', 'c']
        right = ['a', 'd', 'e', 'f', 'c']
        deltas = {'rep' : kron_distance, 'del' : kron_distance, 'ins' : kron_distance}
        expected = 3.
        actual = edit_distance(left, right, gra, deltas)
        self.assertEqual(expected, actual)
        actual = edit_distance(left, right, gra, kron_distance)
        self.assertEqual(expected, actual)

        # set up a different grammar and process the same two strings again
        skip_gra = Grammar('A', ['A', 'Sk'])
        skip_gra.append_replacement('A', 'A', 'rep')
        skip_gra.append_deletion('A', 'Sk', 'del')
        skip_gra.append_insertion('A', 'Sk', 'ins')
        skip_gra.append_replacement('Sk', 'A', 'rep')
        skip_gra.append_deletion('Sk', 'Sk', 'skdel')
        skip_gra.append_insertion('Sk', 'Sk', 'skins')

        def skcost(x, y):
            return 0.5
        deltas = {'rep' : kron_distance, 'del' : kron_distance, 'ins' : kron_distance, 'skdel' : skcost, 'skins' : skcost}

        expected = 2.5
        actual = edit_distance(left, right, skip_gra, deltas)
        self.assertEqual(expected, actual)

    def test_backtrace(self):
        gra = Grammar('A', ['A'])
        gra.append_replacement('A', 'A', 'rep')
        gra.append_deletion('A', 'A', 'del')
        gra.append_insertion('A', 'A', 'ins')
        left = ['a', 'b', 'c']
        right = ['a', 'd', 'e', 'f', 'c']

        # set up expected alignment
        expected_ali = Alignment()
        expected_ali.append_tuple(0, 0, 'rep')
        expected_ali.append_tuple(1, 1, 'rep')
        expected_ali.append_tuple(-1, 2, 'ins')
        expected_ali.append_tuple(-1, 3, 'ins')
        expected_ali.append_tuple(2, 4, 'rep')

        # compare to actual alignment
        actual_ali = backtrace(left, right, gra, kron_distance)
        self.assertEqual(expected_ali, actual_ali)

        # set up a different grammar and process the same two strings again
        skip_gra = Grammar('A', ['A', 'Sk'])
        skip_gra.append_replacement('A', 'A', 'rep')
        skip_gra.append_deletion('A', 'Sk', 'del')
        skip_gra.append_insertion('A', 'Sk', 'ins')
        skip_gra.append_replacement('Sk', 'A', 'rep')
        skip_gra.append_deletion('Sk', 'Sk', 'skdel')
        skip_gra.append_insertion('Sk', 'Sk', 'skins')

        def skcost(x, y):
            return 0.5
        deltas = {'rep' : kron_distance, 'del' : kron_distance, 'ins' : kron_distance, 'skdel' : skcost, 'skins' : skcost}

        # set up expected alignment
        expected_ali = Alignment()
        expected_ali.append_tuple(0, 0, 'rep')
        expected_ali.append_tuple(1, 1, 'rep')
        expected_ali.append_tuple(-1, 2, 'ins')
        expected_ali.append_tuple(-1, 3, 'skins')
        expected_ali.append_tuple(2, 4, 'rep')

        # compare to actual alignment
        actual_ali = backtrace(left, right, skip_gra, deltas)
        self.assertEqual(expected_ali, actual_ali)

if __name__ == '__main__':
    unittest.main()
