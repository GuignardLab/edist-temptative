"""
Tests ADP computations.

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
import numpy as np
from edist.alignment import Alignment
import edist.adp as adp

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def kron_distance(x, y):
    if(x == y):
        return 0.
    else:
        return 1.

class TestADP(unittest.TestCase):

    def test_edit_distance(self):
        gra = adp.Grammar('A', ['A'])
        gra.append_replacement('A', 'A', 'rep')
        gra.append_deletion('A', 'A', 'del')
        gra.append_insertion('A', 'A', 'ins')
        left = ['a', 'b', 'c']
        right = ['a', 'd', 'e', 'f', 'c']
        deltas = {'rep' : kron_distance, 'del' : kron_distance, 'ins' : kron_distance}
        expected = 3.
        actual = adp.edit_distance(left, right, gra, deltas)
        self.assertEqual(expected, actual)
        actual = adp.edit_distance(left, right, gra, kron_distance)
        self.assertEqual(expected, actual)

        # set up a different grammar and process the same two strings again
        skip_gra = adp.Grammar('A', ['A', 'Sk'])
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
        actual = adp.edit_distance(left, right, skip_gra, deltas)
        self.assertEqual(expected, actual)

    def test_backtrace(self):
        gra = adp.Grammar('A', ['A'])
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
        actual_ali = adp.backtrace(left, right, gra, kron_distance)
        self.assertEqual(expected_ali, actual_ali)

        # set up a different grammar and process the same two strings again
        skip_gra = adp.Grammar('A', ['A', 'Sk'])
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
        actual_ali = adp.backtrace(left, right, skip_gra, deltas)
        self.assertEqual(expected_ali, actual_ali)

    def test_backtrace_sochastic(self):
        gra = adp.Grammar('A', ['A'])
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
        actual_ali = adp.backtrace_stochastic(left, right, gra, kron_distance)
        self.assertEqual(expected_ali, actual_ali)

        # set up a different grammar and process another two strings
        skip_gra = adp.Grammar('A', ['A', 'Skdel', 'Skins'])
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
        actual_ali = adp.backtrace_stochastic(left, right, skip_gra, deltas)
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
        actual_ali = adp.backtrace_stochastic(x, y, skip_gra, deltas)
        T = 100
        histogram = np.zeros(len(expected_alis))
        for t in range(T):
            actual_ali = adp.backtrace_stochastic(x, y, skip_gra, deltas)
            self.assertTrue(actual_ali in expected_alis, 'unexpected alignment: %s' % str(actual_ali))
            a = expected_alis.index(actual_ali)
            histogram[a] += 1
        histogram /= T

        expected_histogram = np.array([0.5, 0.5])
        np.testing.assert_allclose(histogram, expected_histogram, atol=0.1)


    def test_backtrace_matrix(self):
        gra = adp.Grammar('A', ['A'])
        gra.append_replacement('A', 'A', 'rep')
        gra.append_deletion('A', 'A', 'del')
        gra.append_insertion('A', 'A', 'ins')
        left  = ['a', 'b', 'c', 'd']
        right = ['a', 'd', 'c']

        # set up expected matrices
        P_rep_expected = np.zeros((1, len(left), len(right)))
        P_rep_expected[0, 0, 0] = 1.
        P_rep_expected[0, 1, 1] = 1.
        P_rep_expected[0, 2, 2] = 1.
        P_del_expected = np.zeros((1, len(left)))
        P_del_expected[0, 3] = 1.
        P_ins_expected = np.zeros((1, len(right)))

        # compare to actual matrices
        P_rep, P_del, P_ins, K = adp.backtrace_matrix(left, right, gra, kron_distance)
        self.assertEqual(1, K)
        np.testing.assert_allclose(P_rep_expected, P_rep, atol=0.1)
        np.testing.assert_allclose(P_del_expected, P_del, atol=0.1)
        np.testing.assert_allclose(P_ins_expected, P_ins, atol=0.1)

        # set up a different grammar and process another two strings
        skip_gra = adp.Grammar('A', ['A', 'Skdel', 'Skins'])
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

        # set up expected matrices
        P_rep_expected = np.zeros((1, len(left), len(right)))
        P_rep_expected[0, 0, 0] = 1.
        P_rep_expected[0, 1, 1] = 1.
        P_rep_expected[0, 2, 4] = 1.
        P_del_expected = np.zeros((2, len(left)))
        P_ins_expected = np.zeros((2, len(right)))
        P_ins_expected[0, 2] = 1.
        P_ins_expected[1, 3] = 1.

        # compare to actual matrices
        P_rep, P_del, P_ins, K = adp.backtrace_matrix(left, right, skip_gra, deltas)
        self.assertEqual(1, K)
        np.testing.assert_allclose(P_rep_expected, P_rep, atol=0.1)
        np.testing.assert_allclose(P_del_expected, P_del, atol=0.1)
        np.testing.assert_allclose(P_ins_expected, P_ins, atol=0.1)

        # test an ambiguous case and make sure that the probability
        # distribution confirms to expectations
        left  = 'aa'
        right = 'b'

        # set up expected matrices
        P_rep_expected = np.zeros((1, len(left), len(right)))
        P_rep_expected[0, 0, 0] = 0.5
        P_rep_expected[0, 1, 0] = 0.5
        P_del_expected = np.zeros((2, len(left)))
        P_del_expected[0, 0] = 0.5
        P_del_expected[0, 1] = 0.5
        P_ins_expected = np.zeros((2, len(right)))

        # compare to actual matrices
        P_rep, P_del, P_ins, K = adp.backtrace_matrix(left, right, skip_gra, deltas)
        self.assertEqual(2, K)
        np.testing.assert_allclose(P_rep_expected, P_rep, atol=0.1)
        np.testing.assert_allclose(P_del_expected, P_del, atol=0.1)
        np.testing.assert_allclose(P_ins_expected, P_ins, atol=0.1)


if __name__ == '__main__':
    unittest.main()
