#!/usr/bin/python3
"""
Tests the dynamic time warping implementation.

"""
# Copyright (C) 2019-2020
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
import time
import numpy as np
from edist.alignment import Alignment
import edist.dtw as dtw

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestDTW(unittest.TestCase):

    def test_dtw_numeric(self):
        x = [0, 0, 1, 2]
        y = [0, 1, 1, 3]
        expected = 1.
        actual = dtw.dtw_numeric(np.array(x, dtype=float), np.array(y, dtype=float))
        np.testing.assert_almost_equal(actual, expected, 3)

        def abs_delta(x, y):
            return abs(x - y)

        actual = dtw.dtw(x, y, delta=abs_delta)
        np.testing.assert_almost_equal(actual, expected, 3)

    def test_dtw_manhattan(self):
        x = np.expand_dims(np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float), 1)
        x = np.concatenate([x, x], axis=1)
        y = np.copy(x)
        y[:, 1] = np.array([0, 0, 0, 1, 2, 2, 2, 3], dtype=float)
        # if we compare the first and second dimension separately, we expect
        # a distance of 0
        expected = 0.
        actual = dtw.dtw_numeric(x[:, 0], y[:, 0]) + dtw.dtw_numeric(x[:, 1], y[:, 1])
        np.testing.assert_almost_equal(actual, expected, 3)
        # but if we compare the manhattan distance, we expect a distance of two
        expected = 2.
        actual = dtw.dtw_manhattan(x, y)
        np.testing.assert_almost_equal(actual, expected, 3)

        # if we compare single-element sequences, we expect simply the manhattan
        # distance between the entries
        x = np.array([[0., 0.]])
        y = np.array([[1., 1.]])
        expected = 2.
        actual = dtw.dtw_manhattan(x, y)
        np.testing.assert_almost_equal(actual, expected, 3)

    def test_dtw_euclidean(self):
        x = np.expand_dims(np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float), 1)
        x = np.concatenate([x, x], axis=1)
        y = np.copy(x)
        y[:, 1] = np.array([0, 0, 0, 1, 2, 2, 2, 3], dtype=float)
        # if we compare the first and second dimension separately, we expect
        # a distance of 0
        expected = 0.
        actual = dtw.dtw_numeric(x[:, 0], y[:, 0]) + dtw.dtw_numeric(x[:, 1], y[:, 1])
        np.testing.assert_almost_equal(actual, expected, 3)
        # but if we compare the Euclidean distance, we expect a distance of two
        expected = 2.
        actual = dtw.dtw_euclidean(x, y)
        np.testing.assert_almost_equal(actual, expected, 3)

        # if we compare single-element sequences, we expect simply the Euclidean
        # distance between the entries
        x = np.array([[0., 0.]])
        y = np.array([[1., 1.]])
        expected = np.sqrt(2)
        actual = dtw.dtw_euclidean(x, y)
        np.testing.assert_almost_equal(actual, expected, 3)

    def test_dtw_string(self):
        x = 'aabbccdd'
        y = 'aaabcccde'
        expected = 1.
        actual = dtw.dtw_string(x, y)
        np.testing.assert_almost_equal(actual, expected, 3)

        def kron_delta(x, y):
            if(x == y):
                return 0.
            else:
                return 1.

        actual = dtw.dtw(x, y, delta=kron_delta)
        np.testing.assert_almost_equal(actual, expected, 3)

    def test_dtw_backtrace(self):
        def kron_delta(x, y):
            if(x == y):
                return 0.
            else:
                return 1.

        x = 'aaa'
        y = 'aa'

        expected_ali = Alignment()
        expected_ali.append_tuple(0, 0)
        expected_ali.append_tuple(1, 1)
        expected_ali.append_tuple(2, 1)

        actual_ali = dtw.dtw_backtrace(x, y, kron_delta)
        self.assertEqual(expected_ali, actual_ali)

    def test_dtw_backtrace_stochastic(self):
        # test an ambiguous case and check whether the stochastic
        # backtracing does select all possible alignments with the
        # expected distribution

        def kron_delta(x, y):
            if(x == y):
                return 0.
            else:
                return 1.

        x = 'aa'
        y = 'aa'

        expected_alis = [Alignment(), Alignment(), Alignment()]
        expected_alis[0].append_tuple(0, 0)
        expected_alis[0].append_tuple(1, 1)
        expected_alis[1].append_tuple(0, 0)
        expected_alis[1].append_tuple(0, 1)
        expected_alis[1].append_tuple(1, 1)
        expected_alis[2].append_tuple(0, 0)
        expected_alis[2].append_tuple(1, 0)
        expected_alis[2].append_tuple(1, 1)

        T = 100
        histogram = np.zeros(len(expected_alis))
        for t in range(T):
            actual_ali = dtw.dtw_backtrace_stochastic(x, y, kron_delta)
            self.assertTrue(actual_ali in expected_alis)
            a = expected_alis.index(actual_ali)
            histogram[a] += 1
        histogram /= T

        # the last option will dominate the distribution because
        # there is no choice left if we decide to delete a in the
        # beginning.
        expected_histogram = np.array([0.33, 0.33, 0.33])
        np.testing.assert_allclose(histogram, expected_histogram, atol=0.1)

    def test_dtw_backtrace_matrix(self):

        def kron_delta(x, y):
            if(x == y):
                return 0.
            else:
                return 1.

        x = 'aaa'
        y = 'aa'

        # set up expected count matrix
        expected_K = np.array([[5, 3, 1], [1, 3, 5]]).T
        expected_k = 5

        P, K, k = dtw.dtw_backtrace_matrix(x, y, kron_delta)

        np.testing.assert_almost_equal(P, K / k, 2)
        np.testing.assert_almost_equal(K, expected_K, 2)
        self.assertEqual(expected_k, k)

        x = 'abc'
        y = 'aa'

        # set up expected count matrix
        expected_K = np.array([[3, 1, 0], [1, 2, 3]]).T
        expected_k = 3

        P, K, k = dtw.dtw_backtrace_matrix(x, y, kron_delta)

        np.testing.assert_almost_equal(P, K / k, 2)
        np.testing.assert_almost_equal(K, expected_K, 2)
        self.assertEqual(expected_k, k)

        x = 'abc'
        y = 'bc'

        # set up expected count matrix
        expected_K = np.array([[1, 1, 0], [0, 0, 1]]).T
        expected_k = 1

        P, K, k = dtw.dtw_backtrace_matrix(x, y, kron_delta)

        np.testing.assert_almost_equal(P, K / k, 2)
        np.testing.assert_almost_equal(K, expected_K, 2)
        self.assertEqual(expected_k, k)


if __name__ == '__main__':
    unittest.main()
