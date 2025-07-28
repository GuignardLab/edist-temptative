#!/usr/bin/python3
"""
Tests the embedding edit distance learning (BEDL) implementation.

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
from scipy.spatial.distance import cdist
import edist.sed as sed
import edist.bedl as bedl

__author__ = "Benjamin Paaßen"
__copyright__ = "Copyright (C) 2019-2021, Benjamin Paaßen"
__license__ = "GPLv3"
__maintainer__ = "Benjamin Paaßen"
__email__ = "bpaassen@techfak.uni-bielefeld.de"


class TestBEDL(unittest.TestCase):

    def test_indexing(self):
        alphabet = ["a", "b", "c"]

        expected_idx = {"a": 0, "b": 1, "c": 2}
        actual_idx = bedl.create_index(alphabet)
        self.assertEqual(expected_idx, actual_idx)

        Xs = ["a", "bac", "bbb"]
        expected_Ys = [[0], [1, 0, 2], [1, 1, 1]]
        actual_Ys = bedl.index_data(Xs, actual_idx)
        self.assertEqual(expected_Ys, actual_Ys)

    def test_initialize_embedding(self):
        n = 8
        # create an embedding with n dimensions
        embedding = bedl.initialize_embedding(n)
        # append the zero vector
        embedding = np.concatenate([embedding, np.zeros((1, n))], axis=0)
        # compute all pairwise distance between embedding vectors
        D = cdist(embedding, embedding)
        # we expect that this should be a matrix with zero on the
        # diagonal and only ones otherwise
        np.testing.assert_allclose(
            D, np.ones((n + 1, n + 1)) - np.eye(n + 1), atol=1e-3
        )

    def test_reduce_backtrace(self):
        # create two index lists
        x = [0, 0, 0]
        y = [1]

        # set up expected reduced matrix
        expected_Phat = np.zeros((3, 3))
        expected_Phat[0, 1] = 1.0
        expected_Phat[0, 2] = 2.0

        # compute actual reduced matrix
        P, _, _ = sed.standard_sed_backtrace_matrix(x, y)
        Phat = bedl.reduce_backtrace(P, x, y, 2)

        np.testing.assert_allclose(Phat, expected_Phat, atol=1e-3)

    def test_fit(self):
        # create a very simple string dataset where we need to learn that
        # a <-> b replacements should be cheap and c <-> d replacements
        # should be cheap
        X = ["ab", "ba", "cd", "dc"]
        y = np.array([0, 0, 1, 1])
        # initialize a BEDL model with one prototype per class
        model = bedl.BEDL(1)
        # fit it to the data
        model.fit(X, y)
        # check the resulting embedding
        Delta = cdist(model._embedding, model._embedding)
        self.assertTrue(np.all(Delta[:2, :2] < 0.1))
        self.assertTrue(np.all(Delta[2:4, 2:4] < 0.1))
        self.assertTrue(np.all(Delta[:2, 2:] > 0.1))
        self.assertTrue(np.all(Delta[2:, :2] > 0.1))


if __name__ == "__main__":
    unittest.main()
