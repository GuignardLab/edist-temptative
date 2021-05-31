"""
Tests affine edit distance computations.

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
import edist.aed as aed

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestAED(unittest.TestCase):

    def test_aed(self):
        x = 'abc'
        y = 'adefc'
        expected = 2.5
        actual = aed.aed(x, y)
        self.assertEqual(expected, actual)

    def test_backtrace(self):

        x = 'abc'
        y = 'adefc'

        # set up expected alignment
        expected_ali = Alignment()
        expected_ali.append_tuple(0, 0, 'rep')
        expected_ali.append_tuple(1, 1, 'rep')
        expected_ali.append_tuple(-1, 2, 'ins')
        expected_ali.append_tuple(-1, 3, 'skins')
        expected_ali.append_tuple(2, 4, 'rep')

        # compare to actual alignment
        actual_ali = aed.aed_backtrace(x, y)
        self.assertEqual(expected_ali, actual_ali)

    def test_backtrace_sochastic(self):
        x = 'abc'
        y = 'abefc'

        # set up expected alignment
        expected_ali = Alignment()
        expected_ali.append_tuple(0, 0, 'rep')
        expected_ali.append_tuple(1, 1, 'rep')
        expected_ali.append_tuple(-1, 2, 'ins')
        expected_ali.append_tuple(-1, 3, 'skins')
        expected_ali.append_tuple(2, 4, 'rep')

        # compare to actual alignment
        actual_ali = aed.aed_backtrace_stochastic(x, y)
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
        T = 100
        histogram = np.zeros(len(expected_alis))
        for t in range(T):
            actual_ali = aed.aed_backtrace_stochastic(x, y)
            self.assertTrue(actual_ali in expected_alis, 'unexpected alignment: %s' % str(actual_ali))
            a = expected_alis.index(actual_ali)
            histogram[a] += 1
        histogram /= T

        expected_histogram = np.array([0.5, 0.5])
        np.testing.assert_allclose(histogram, expected_histogram, atol=0.1)


    def test_backtrace_matrix(self):
        x = 'abc'
        y = 'abefc'

        # set up expected matrix
        P_expected = np.zeros((len(x)+2, len(y)+2))
        P_expected[0, 0] = 1.
        P_expected[1, 1] = 1.
        P_expected[3, 2] = 1.
        P_expected[4, 3] = 1.
        P_expected[2, 4] = 1.

        # compare to actual matrix
        P_actual, k = aed.aed_backtrace_matrix(x, y)
        self.assertEqual(1, k)
        np.testing.assert_allclose(P_actual, P_expected, atol=0.1)

        # test an ambiguous case and make sure that the probability
        # distribution confirms to expectations
        x = 'aa'
        y = 'b'

        # set up expected matrix
        P_expected = np.zeros((len(x)+2, len(y)+2))
        P_expected[0, 0] = 0.5
        P_expected[1, 0] = 0.5
        P_expected[0, 1] = 0.5
        P_expected[1, 1] = 0.5

        # compare to actual matrices
        P_actual, k = aed.aed_backtrace_matrix(x, y)
        self.assertEqual(2, k)
        np.testing.assert_allclose(P_actual, P_expected, atol=0.1)


if __name__ == '__main__':
    unittest.main()
