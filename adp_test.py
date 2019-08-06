import unittest
import numpy as np
from edist.alignment import Alignment
from edist.adp import Grammar
from edist.adp import edit_distance
from edist.adp import backtrace
from edist.adp import backtrace_stochastic

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

    def test_backtrace_sochastic(self):
        gra = Grammar('A', ['A'])
        gra.append_replacement('A', 'A', 'rep')
        gra.append_deletion('A', 'A', 'del')
        gra.append_insertion('A', 'A', 'ins')
        left  = ['a', 'b', 'c', 'd']
        right = ['a', 'd', 'c']

        # set up expected alignment
        expected_ali = Alignment()
        expected_ali.append_tuple(0, 0, 'rep')
        expected_ali.append_tuple(1, 1, 'rep')
        expected_ali.append_tuple(2, 2, 'rep')
        expected_ali.append_tuple(3 , -1, 'del')

        # compare to actual alignment
        actual_ali = backtrace_stochastic(left, right, gra, kron_distance)
        self.assertEqual(expected_ali, actual_ali)

        # set up a different grammar and process another two strings
        skip_gra = Grammar('A', ['A', 'Skdel', 'Skins'])
        skip_gra.append_replacement('A', 'A', 'rep')
        skip_gra.append_deletion('A', 'Skdel', 'del')
        skip_gra.append_insertion('A', 'Skins', 'ins')
        skip_gra.append_replacement('Skdel', 'A', 'rep')
        skip_gra.append_deletion('Skdel', 'Skdel', 'skdel')
        skip_gra.append_replacement('Skins', 'A', 'rep')
        skip_gra.append_insertion('Skins', 'Skins', 'skins')

        def skcost(x, y):
            return 0.5
        deltas = {'rep' : kron_distance, 'del' : kron_distance, 'ins' : kron_distance, 'skdel' : skcost, 'skins' : skcost}

        left  = ['a', 'b', 'c']
        right = ['a', 'b', 'e', 'f', 'c']

        # set up expected alignment
        expected_ali = Alignment()
        expected_ali.append_tuple(0, 0, 'rep')
        expected_ali.append_tuple(1, 1, 'rep')
        expected_ali.append_tuple(-1, 2, 'ins')
        expected_ali.append_tuple(-1, 3, 'skins')
        expected_ali.append_tuple(2, 4, 'rep')

        # compare to actual alignment
        actual_ali = backtrace_stochastic(left, right, skip_gra, deltas)
        self.assertEqual(expected_ali, actual_ali)

        # test an ambiguous case and make sure that the probability
        # distribution confirms to expectations
        x = 'aa'
        y = 'b'

        expected_alis = [Alignment(), Alignment()]
        expected_alis[0].append_tuple( 0,  0, 'rep')
        expected_alis[0].append_tuple( 1, -1, 'del')
        expected_alis[1].append_tuple( 0, -1, 'del')
        expected_alis[1].append_tuple( 1,  0, 'rep')
        actual_ali = backtrace_stochastic(x, y, skip_gra, deltas)
        T = 100
        histogram = np.zeros(len(expected_alis))
        for t in range(T):
            actual_ali = backtrace_stochastic(x, y, skip_gra, deltas)
            self.assertTrue(actual_ali in expected_alis, 'unexpected alignment: %s' % str(actual_ali))
            a = expected_alis.index(actual_ali)
            histogram[a] += 1
        histogram /= T

        # the last option will dominate the distribution because
        # there is no choice left if we decide to delete a in the
        # beginning.
        expected_histogram = np.array([0.5, 0.5])
        np.testing.assert_allclose(histogram, expected_histogram, atol=0.1)

if __name__ == '__main__':
    unittest.main()
