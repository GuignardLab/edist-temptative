#!/usr/bin/python3
"""
Tests the sequence edit distance implementation.

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
import time
import numpy as np
from edist.alignment import Alignment
import edist.sed as sed

__author__ = "Benjamin Paaßen"
__copyright__ = "Copyright (C) 2019-2021, Benjamin Paaßen"
__license__ = "GPLv3"
__maintainer__ = "Benjamin Paaßen"
__email__ = "bpaassen@techfak.uni-bielefeld.de"


class TestSED(unittest.TestCase):

    def test_sed_string(self):
        x = "aabbccdd"
        y = "aaabcccde"
        expected = 3.0
        actual = sed.sed_string(x, y)
        self.assertEqual(expected, actual)

        def kron_delta(x, y):
            if x == y:
                return 0.0
            else:
                return 1.0

        actual = sed.sed(x, y, delta=kron_delta)
        self.assertEqual(float(expected), actual)

    def test_sed_backtrace(self):
        x = "abcde"
        y = "bdef"

        expected_ali = Alignment()
        expected_ali.append_tuple(0, -1)
        expected_ali.append_tuple(1, 0)
        expected_ali.append_tuple(2, -1)
        expected_ali.append_tuple(3, 1)
        expected_ali.append_tuple(4, 2)
        expected_ali.append_tuple(-1, 3)

        def kron_delta(x, y):
            if x == y:
                return 0.0
            else:
                return 1.0

        actual_ali = sed.sed_backtrace(x, y, kron_delta)
        self.assertEqual(expected_ali, actual_ali)

        actual_ali = sed.sed_backtrace_stochastic(x, y, kron_delta)
        self.assertEqual(expected_ali, actual_ali)

    def test_sed_backtrace_stochastic(self):
        # test an ambiguous case and check whether the stochastic
        # backtracing does select all possible alignments with the
        # expected distribution

        def kron_delta(x, y):
            if x == y:
                return 0.0
            else:
                return 1.0

        x = "aaa"
        y = "aa"

        expected_alis = [Alignment(), Alignment(), Alignment()]
        expected_alis[0].append_tuple(0, 0)
        expected_alis[0].append_tuple(1, 1)
        expected_alis[0].append_tuple(2, -1)
        expected_alis[1].append_tuple(0, 0)
        expected_alis[1].append_tuple(1, -1)
        expected_alis[1].append_tuple(2, 1)
        expected_alis[2].append_tuple(0, -1)
        expected_alis[2].append_tuple(1, 0)
        expected_alis[2].append_tuple(2, 1)

        T = 100
        histogram = np.zeros(len(expected_alis))
        for t in range(T):
            actual_ali = sed.sed_backtrace_stochastic(x, y, kron_delta)
            self.assertTrue(actual_ali in expected_alis)
            a = expected_alis.index(actual_ali)
            histogram[a] += 1
        histogram /= T

        # the last option will dominate the distribution because
        # there is no choice left if we decide to delete a in the
        # beginning.
        expected_histogram = np.array([0.25, 0.25, 0.5])
        np.testing.assert_allclose(histogram, expected_histogram, atol=0.1)

    def test_sed_backtrace_matrix(self):

        def kron_delta(x, y):
            if x == y:
                return 0.0
            else:
                return 1.0

        x = "aaa"
        y = "aa"

        # set up expected count matrix
        expected_K = np.array([[2, 1, 0], [0, 1, 2]]).T
        expected_k = 3

        P, K, k = sed.sed_backtrace_matrix(x, y, kron_delta)

        np.testing.assert_almost_equal(P[: len(x), :][:, : len(y)], K / k, 2)
        np.testing.assert_almost_equal(
            np.sum(P[:, : len(y)], axis=0), np.ones(len(y)), 2
        )
        np.testing.assert_almost_equal(
            np.sum(P[: len(x), :], axis=1), np.ones(len(x)), 2
        )
        np.testing.assert_almost_equal(K, expected_K, 2)
        self.assertEqual(expected_k, k)

        x = "abc"
        y = "aa"

        # set up expected count matrix
        expected_K = np.array([[2, 0, 0], [0, 1, 1]]).T
        expected_k = 2

        P, K, k = sed.sed_backtrace_matrix(x, y, kron_delta)

        np.testing.assert_almost_equal(P[: len(x), :][:, : len(y)], K / k, 2)
        np.testing.assert_almost_equal(
            np.sum(P[:, : len(y)], axis=0), np.ones(len(y)), 2
        )
        np.testing.assert_almost_equal(
            np.sum(P[: len(x), :], axis=1), np.ones(len(x)), 2
        )
        np.testing.assert_almost_equal(K, expected_K, 2)
        self.assertEqual(expected_k, k)

        x = "abc"
        y = "bc"

        # set up expected count matrix
        expected_K = np.array([[0, 1, 0], [0, 0, 1]]).T
        expected_k = 1

        P, K, k = sed.sed_backtrace_matrix(x, y, kron_delta)

        np.testing.assert_almost_equal(P[: len(x), :][:, : len(y)], K / k, 2)
        np.testing.assert_almost_equal(
            np.sum(P[:, : len(y)], axis=0), np.ones(len(y)), 2
        )
        np.testing.assert_almost_equal(
            np.sum(P[: len(x), :], axis=1), np.ones(len(x)), 2
        )
        np.testing.assert_almost_equal(K, expected_K, 2)
        self.assertEqual(expected_k, k)

    def test_standard_sed(self):
        x = "aabbccdd"
        y = "aaabcccde"
        expected = 3
        actual = sed.sed_string(x, y)
        self.assertEqual(expected, actual)

        actual = sed.standard_sed(x, y)
        self.assertEqual(float(expected), actual)

    def test_standard_sed_backtrace(self):
        x = "abcde"
        y = "bdef"

        expected_ali = Alignment()
        expected_ali.append_tuple(0, -1)
        expected_ali.append_tuple(1, 0)
        expected_ali.append_tuple(2, -1)
        expected_ali.append_tuple(3, 1)
        expected_ali.append_tuple(4, 2)
        expected_ali.append_tuple(-1, 3)

        actual_ali = sed.standard_sed_backtrace(x, y)
        self.assertEqual(expected_ali, actual_ali)

        actual_ali = sed.standard_sed_backtrace_stochastic(x, y)
        self.assertEqual(expected_ali, actual_ali)

    def test_standard_sed_backtrace_stochastic(self):
        # test an ambiguous case and check whether the stochastic
        # backtracing does select all possible alignments with the
        # expected distribution

        x = "aaa"
        y = "aa"

        expected_alis = [Alignment(), Alignment(), Alignment()]
        expected_alis[0].append_tuple(0, 0)
        expected_alis[0].append_tuple(1, 1)
        expected_alis[0].append_tuple(2, -1)
        expected_alis[1].append_tuple(0, 0)
        expected_alis[1].append_tuple(1, -1)
        expected_alis[1].append_tuple(2, 1)
        expected_alis[2].append_tuple(0, -1)
        expected_alis[2].append_tuple(1, 0)
        expected_alis[2].append_tuple(2, 1)

        T = 100
        histogram = np.zeros(len(expected_alis))
        for t in range(T):
            actual_ali = sed.standard_sed_backtrace_stochastic(x, y)
            self.assertTrue(actual_ali in expected_alis)
            a = expected_alis.index(actual_ali)
            histogram[a] += 1
        histogram /= T

        # the last option will dominate the distribution because
        # there is no choice left if we decide to delete a in the
        # beginning.
        expected_histogram = np.array([0.25, 0.25, 0.5])
        np.testing.assert_allclose(histogram, expected_histogram, atol=0.1)

    def test_standard_sed_backtrace_matrix(self):

        x = "aaa"
        y = "aa"

        # set up expected count matrix
        expected_K = np.array([[2, 1, 0], [0, 1, 2]]).T
        expected_k = 3

        P, K, k = sed.standard_sed_backtrace_matrix(x, y)

        np.testing.assert_almost_equal(P[: len(x), :][:, : len(y)], K / k, 2)
        np.testing.assert_almost_equal(
            np.sum(P[:, : len(y)], axis=0), np.ones(len(y)), 2
        )
        np.testing.assert_almost_equal(
            np.sum(P[: len(x), :], axis=1), np.ones(len(x)), 2
        )
        np.testing.assert_almost_equal(K, expected_K, 2)
        self.assertEqual(expected_k, k)

        x = "abc"
        y = "aa"

        # set up expected count matrix
        expected_K = np.array([[2, 0, 0], [0, 1, 1]]).T
        expected_k = 2

        P, K, k = sed.standard_sed_backtrace_matrix(x, y)

        np.testing.assert_almost_equal(P[: len(x), :][:, : len(y)], K / k, 2)
        np.testing.assert_almost_equal(
            np.sum(P[:, : len(y)], axis=0), np.ones(len(y)), 2
        )
        np.testing.assert_almost_equal(
            np.sum(P[: len(x), :], axis=1), np.ones(len(x)), 2
        )
        np.testing.assert_almost_equal(K, expected_K, 2)
        self.assertEqual(expected_k, k)

        x = "abc"
        y = "bc"

        # set up expected count matrix
        expected_K = np.array([[0, 1, 0], [0, 0, 1]]).T
        expected_k = 1

        P, K, k = sed.standard_sed_backtrace_matrix(x, y)

        np.testing.assert_almost_equal(P[: len(x), :][:, : len(y)], K / k, 2)
        np.testing.assert_almost_equal(
            np.sum(P[:, : len(y)], axis=0), np.ones(len(y)), 2
        )
        np.testing.assert_almost_equal(
            np.sum(P[: len(x), :], axis=1), np.ones(len(x)), 2
        )
        np.testing.assert_almost_equal(K, expected_K, 2)
        self.assertEqual(expected_k, k)

    def test_speed(self):
        m = 300
        # create a very large tree with maximum number of keyroots
        x = "a" * (2 * m + 1)
        y = "b" * m

        # as character distance, use the kronecker delta
        def kron_distance(x, y):
            if x == y:
                return 0.0
            else:
                return 1.0

        # compare how long the general edit distance takes
        start = time.time()
        d = sed.sed(x, y, kron_distance)
        general_time = time.time() - start

        # with how long the standard edit distance takes
        start = time.time()
        d = sed.standard_sed(x, y)
        std_time = time.time() - start

        # with how long the string edit distance takes
        start = time.time()
        d = sed.sed_string(x, y)
        string_time = time.time() - start

        self.assertTrue(std_time < general_time)
        self.assertTrue(string_time < std_time)


if __name__ == "__main__":
    unittest.main()
