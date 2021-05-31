"""
Implements embedding edit distance learning as described in the paper

Paaßen, B., Gallicchio, C., Micheli, A., and Hammer, B. (2018).
Tree Edit Distance Learning via Adaptive Symbol Embeddings. Proceedings of
the 35th International Conference on Machine Learning (ICML 2018).
URL: http://proceedings.mlr.press/v80/paassen18a.html

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

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClassifierMixin
from proto_dist_ml.mglvq import MGLVQ
import edist.sed as sed
import edist.multiprocess as mp

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

_ERR_CUTOFF = 1E-5
_BFGS_MAX_IT = 100

class BEDL(BaseEstimator, ClassifierMixin):
    """ Implements the embedding edit distance learning (BEDL) scheme for
    metric learning on edit distances.

    In more detail, this learns a median generalized learning vector
    quantization (MGLVQ) classifier on the data and an embedding of the
    input symbols which yields an edit distance that makes classification
    with this classifier easier.

    Attributes
    ----------
    K: int
        The number of prototypes for the MGLVQ classifier.
    T: int
        The number of learning epochs we use at most. Defaults to 5.
    phi: function (default = identity)
        A squashing function to post-process each error term. Defaults to the
        identity.
    phi_grad: function (default = one)
        The gradient function corresponding to phi.
    distance: function (default = sed.sed)
        The edit distance function that shall be learned. Defaults to the
        sequence edit distance sed.sed.
    distance_backtrace: function (default = sed.sed_backtrace_matrix)
        The matrix backtracing function for the distance.
        Defaults to sed.sed_backtrace_matrix. Note that this currently does NOT
        support ADP because ADP returns a different backtracing format.
    _classifier: class proto_dist_ml.MGLVQ
        The learned MGLVQ classifier model.
    _idx: dictionary
        A mapping from alphabet to indices.
    _embedding: array_like
        A len(alphabet) x len(alphabet) - 1 embedding matrix for all symbols in
        the alphabet.
    _delta_obj: class bedl.EmbeddingDelta
        An internal object to make storing of the delta function more
        efficient.
    _delta: function
        The learned delta function.

    """
    def __init__(self, K, T = 5, phi = None, phi_grad = None, distance = None, distance_backtrace = None):
        self.K = K
        self.T = T
        if(phi is None):
            self.phi = lambda mus : mus
            self.phi_grad = lambda mus : np.ones_like(mus)
        else:
            self.phi = phi
        if(distance is None):
            self.distance = sed.sed
            self.distance_backtrace = sed.sed_backtrace_matrix
        else:
            self.distance = distance
            self.distance_backtrace = distance_backtrace

    def fit(self, X, y):
        """ Trains a BEDL model on the given input data.

        In more detail, we iterate the following steps in each training
        epoch:

        1. We re-compute the distance matrix.
        2. We adapt the MGLVQ classifier to the current distance matrix.
        3. We compute the matrix backtraces for the data-to-prototype
           distances.
        4. We optimize the embedding for the current data-to-prototype
           distances.

        For more details, please refer to the ICML 2018 paper.

        Arguments
        ---------
        X: list
            a list of data points, each being either a list or a tree,
            depending on the edit distance that shall be learned.
        y: array_like or list
            an array-like or list-like structure with labels for each
            data point.

        Returns
        -------
        class bedl.BEDL
            self

        """
        # identify the alphabet from the training data
        alphabet = set()
        for i in range(len(X)):
            for t in range(len(X[i])):
                alphabet.add(X[i][t])
        # build an index of the input symbols
        self._idx = create_index(sorted(list(alphabet)))
        # preprocess all input data to have symbol indices as nodes instead
        # of the original symbols
        X = index_data(X, self._idx)
        # initialize the classifier
        self._classifier = MGLVQ(self.K)
        # initialize the embedding
        self._embedding = initialize_embedding(len(self._idx))
        # set up optimizer options
        options = { 'ftol' : _ERR_CUTOFF, 'maxiter' : _BFGS_MAX_IT }
        # set up unique labels
        unique_labels = np.unique(y)
        # keep track of prototype changes
        old_w = None
        # now, start the learning process
        self._loss = []
        for t in range(self.T):
            DeltaObj = EmbeddingDelta(self._embedding)
            # first, compute the current pairwise edit distance matrix
            D = mp.pairwise_distances_symmetric(X, self.distance, DeltaObj.delta)
            # then, train the classifier
            self._classifier.prevent_initialization = t > 0
            self._classifier.fit(D, y)
            self._loss.append(self._classifier._loss[-1])
            # if the prototype locations did not change anymore, stop the
            # optimization early
            if(np.all(old_w == self._classifier._w)):
                break
            # compute the backtracing from datapoints to prototypes
            W = []
            for k in range(len(self._classifier._w)):
                W.append(X[self._classifier._w[k]])
            Ps = mp.pairwise_backtraces(X, W, self.distance_backtrace, DeltaObj.delta)
            # reduce the backtraces to just count the symbol pairings, which
            # speeds up gradient computations later on
            for k in range(len(Ps)):
                for l in range(len(Ps[k])):
                    Ps[k][l] = reduce_backtrace(Ps[k][l][0], X[k], W[l], len(self._embedding))
            # set up the glvq loss and gradient function
            obj = lambda embedding : self._loss_and_grad(embedding, Ps, y, unique_labels)
            # optimize the embedding
            res = minimize(obj, self._embedding.ravel(), method='L-BFGS-B', jac = True, options = options)
            # check optimization result
            if(res.x is None):
                raise ValueError('Optimization returned none and failed with message: %s!' % str(res.message))
            self._loss.append(res.fun)
            self._embedding = res.x.reshape(self._embedding.shape)
            # store current prototypes
            old_w = np.copy(self._classifier._w)
        # store the learned delta function
        self._delta_obj = EmbeddingDelta(self._embedding)
        self._delta_obj._index = self._idx
        self._delta = self._delta_obj.delta_with_indexing
        return self

    def _loss_and_grad(self, embedding, Ps, y, unique_labels):
        """ Computes the GLVQ loss and its gradient with respect to the
        embedding.

        Parameters
        ----------
        embedding: array_like
            the current embedding parameters as a vector.
        Ps: list
            the reduced matrix backtraces between the data and the prototypes.
        y: array_like
            the data labels.
        unique_labels: array_like
            The result of np.unique(y)

        Returns
        -------
        loss: float
            The current GLVQ loss.
        Grad: array_like
            The gradient of the current GLVQ loss with respect to embedding
            in flattened format.

        """
        # reshape embedding back into a matrix if its flattened for optimization
        is_flat = False
        if(len(embedding.shape) == 1):
            is_flat = True
            embedding = embedding.reshape(self._embedding.shape)
        # append zero vector to embedding
        embedding = np.concatenate([embedding, np.zeros((1, embedding.shape[1]))], axis=0)
        # compute the pairwise distances between all embedding elements.
        Delta = squareform(pdist(embedding))
        # compute the datapoint-to-prototype distances based on Ps
        Dp = np.zeros((len(Ps), len(Ps[0])))
        for i in range(len(Ps)):
            for k in range(len(Ps[i])):
                Dp[i, k] = np.sum(Ps[i][k] * Delta)

        # find the closest correct and the closest wrong prototype for all
        # data points
        closest_plus  = np.zeros(len(Ps), dtype=int)
        closest_minus = np.zeros(len(Ps), dtype=int)
        dp = np.zeros(len(Ps))
        dm = np.zeros(len(Ps))
        for l in range(len(unique_labels)):
            # get data points in class l and prototypes in and out of class l.
            inClass_l = np.where(y == unique_labels[l])[0]
            inClass_w_l = np.where(self._classifier._y == unique_labels[l])[0]
            outClass_w_l = np.where(self._classifier._y != unique_labels[l])[0]
            # find the closest prototype in the same class
            closest_plus[inClass_l] = inClass_w_l[np.argmin(Dp[inClass_l, :][:, inClass_w_l], axis=1)]
            dp[inClass_l] = Dp[inClass_l, closest_plus[inClass_l]]
            # find the closest prototype in a different class
            closest_minus[inClass_l] = outClass_w_l[np.argmin(Dp[inClass_l, :][:, outClass_w_l], axis=1)]
            dm[inClass_l] = Dp[inClass_l, closest_minus[inClass_l]]

        # compute the raw loss terms 
        mus = (dp - dm) / (dp + dm + 1E-5)
        # compute the loss
        loss = np.sum(self.phi(mus))
        # compute the gradient for all loss terms
        mus_grad = self.phi_grad(mus) * 2 / np.square(dp + dm + 1E-5)
        # compute the gradients with respect to the embedding
        Grad = np.zeros((embedding.shape[0]-1, embedding.shape[1]))
        for j in range(embedding.shape[0]-1):
            # compute the gradient of the delta function for the current
            # embedding
            delta_grad_j = np.expand_dims(embedding[j, :], 0) - embedding
            nzs = Delta[j, :] > 1E-3
            delta_grad_j[nzs, :] /= np.expand_dims(Delta[j, nzs], 1)
            # multiply that gradient with every P matrix
            for i in range(len(Ps)):
                P_plus = Ps[i][closest_plus[i]]
                Grad[j, :] += mus_grad[i] * dm[i] * np.dot(P_plus[j, :] + P_plus[:, j], delta_grad_j)
                P_minus = Ps[i][closest_minus[i]]
                Grad[j, :] -= mus_grad[i] * dp[i] * np.dot(P_minus[j, :] + P_minus[:, j], delta_grad_j)
        # return the results
        if(is_flat):
            Grad = Grad.flatten()
        return loss, Grad

def create_index(lst):
    """ Creates a map of list elements to indices.

    Parameters
    ----------
    lst: list
        A list.

    Returns
    -------
    idx: dictionary
        A map of list elements to indices.

    """
    idx = {}
    for i in range(len(lst)):
        idx[lst[i]] = i
    return idx


def index_data(Xs, idx):
    """ Indexes all data in the input dataset according to the given index.

    Parameters
    ----------
    Xs: list
        A list of data, each being either a list or a tree in
        node list/adjacency list format.
    idx: dictionary
        A map from symbols to indices.

    Returns
    -------
    Ys: list
        A copy of Xs, where each symbol is replaced by a symbol index.

    """
    Ys = []
    for X in Xs:
        is_tree = isinstance(X, tuple)
        if(is_tree):
            X, adj = X
            if(not isinstance(X, list)):
                raise ValueError('Expected either a list or a tree as data point, but got %s' % str(datum))
        # index the datum
        Y = []
        for x in X:
            if(x not in idx):
                raise ValueError('Symbol not in index: %s' % str(x))
            Y.append(idx[x])
        if(is_tree):
            Ys.append((Y, adj))
        else:
            Ys.append(Y)
    return Ys


def initialize_embedding(size):
    """ Sets up a size-dimensional simplex with side length 1 and size+1
    vertices (i.e. an equilateral hyper-triangle).

    Parameters
    ----------
    size: int
        the dimensionality of the simplex.

    Returns
    -------
    embedding: array_like
        A size x size matrix, where each row contains one vertex of
        the simplex. The origin is the last vertex.

    """
    # set up the initial embedding matrix
    embedding = np.zeros((size, size))
    # set up the rho sequence specifying the vertex positions
    rho = np.zeros(size)
    for j in range(size):
        rho[j] = 1. / (2. * (j+1) * (j+2))
    rho = np.sqrt(rho)
    # set up the embedding itself
    for i in range(size):
        for j in range(i):
            embedding[i, j] = rho[j]
        embedding[i, i] = rho[i] * (i+2)
    return embedding


class EmbeddingDelta:
    """ This class serves as a storage for an embedding to make the
    embedding delta function pickleable.

    Parameters
    ----------
    embedding: array_like
        An embedding matrix.

    Attributes
    ----------
    _Delta: array_like
        The current symbol-to-symbol distance matrix
    """
    def __init__(self, embedding):
        # extend the embedding by a zero vector
        embedding = np.concatenate((embedding, np.zeros((1, embedding.shape[1]))))
        # compute the pairwise distances
        self._Delta = squareform(pdist(embedding))

    def delta(self, x, y):
        """ Computes the distance between two embedding vectors identified
        by their index.

        Parameters
        ----------
        x: int
            the left-hand-side index or None.
        y: int
            the right-hand-side index or None.

        Returns
        -------
        d: float
            The Euclidean distance between the embedding for x and for y.

        """
        if(x is None):
            if(y is None):
                return 0.
            else:
                # return the distance between x and the origin, i.e. the norm
                # of x
                return self._Delta[-1, y]
        if(y is None):
            # return the distance between y and the origin, i.e. the norm of y
            return self._Delta[x, -1]
        return self._Delta[x, y]

    def delta_with_indexing(self, x, y):
        """ Computes the distance between two symbols based on their
        embedding vectors.

        Parameters
        ----------
        x: str
            a symbol.
        y: str
            another symbol.

        Returns
        -------
        d: float
            The Euclidean distance between the embedding for x and for y.
        """
        if(x is None):
            if(y is None):
                return 0.
            else:
                # return the distance between x and the origin, i.e. the norm
                # of x
                return self._Delta[-1, self._index[y]]
        if(y is None):
            # return the distance between y and the origin, i.e. the norm of y
            return self._Delta[self._index[x], -1]
        return self._Delta[self._index[x], self._index[y]]


def reduce_backtrace(P, x, y, size):
    """ Transforms the input backtrace matrix P of size len(x) + 1 x len(y) + 1
    into a reduced matrix Phat of size size + 1 x size + 1 which accumulates the
    probabilities for same symbols being replaced, i.e. Phat[k, l] is the
    sum over all entries P[i, j] where x[i] = k and y[j] = l.

    Parameters
    ----------
    P: array_like
        A len(x) + 1 x len(y) + 1 backtracing matrix between sequence x and y,
        where P[i, j] is the probability of x[i] being replaced with y[j] in a
        co-optimal alignment, P[i, len(y)] is the deletion probability for x[i]
        and P[len(x), j] is the insertion probability for y[j]. The last row
        and column are optional (as in case of DTW).
    X: list
        A list of objects, either sequences or trees.
    Y: list
        A list of objects, either sequences or trees.
    size: int
        The alphabet size.

    Returns
    -------
    Phat: array_like
        A size+1 x size+1 matrix which accumulates the probabilities for same
        symbols being replaced, i.e. Phat[k, l] is the sum over all entries
        P[i, j] where x[i] = k and y[j] = l.

    """
    # extract the node list from trees
    if(isinstance(x, tuple)):
        x = x[0]
    if(isinstance(y, tuple)):
        y = y[0]
    # store where each sequence equals to each symbol
    Loc_x = np.zeros((P.shape[0], size+1), dtype=bool)
    for i in range(len(x)):
        Loc_x[i, x[i]] = True
    if(P.shape[0] > len(x)):
        Loc_x[len(x), size] = True
    Loc_y = np.zeros((P.shape[1], size+1), dtype=bool)
    for j in range(len(y)):
        Loc_y[j, y[j]] = True
    if(P.shape[1] > len(y)):
        Loc_y[len(y), size] = True

    # initialize the accumulation matrix
    Phat = np.zeros((size+1, size+1))
    # accumulate probabilities
    for k in range(size+1):
        for l in range(size+1):
            Phat[k, l] = np.sum(P[Loc_x[:, k], :][:, Loc_y[:, l]])
    return Phat
