"""
Implements tree edits, i.e. functions which take a tree as input and
return a changed tree.

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

import abc
import copy
from edist.ted import outermost_right_leaves

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class Edit(abc.ABC):
    """ An abstract parent class for all edits.

    """
    @abc.abstractmethod
    def apply(self, nodes, adj):
        """ Applies this edit to the given tree and returns a copy of the tree
        with the applied changes. The original tree remains unchanged.

        Parameters
        ----------
        nodes: list
            a node list.
        adj: list
            an adjacency list.

        Returns
        -------
        nodes: list
            a copy of nodes with the applied edit.
        adj: list
            a copy of adj with the applied edit.

        """
        pass

    @abc.abstractmethod
    def apply_in_place(self, nodes, adj):
        """ Applies this edit to the given tree. Note that this changes the
        input arguments.

        Parameters
        ----------
        nodes: list
            a node list.
        adj: list
            an adjacency list.

        """
        pass


class Replacement(Edit):
    """ Replaces the label of node self._index with self._label.

    Attributes
    ----------
    _index: int
        The index of the tree node to which this edit is applied.
    _label: str
        The new label for the self._indexth node.

    """
    def __init__(self, index, label):
        self._index = index
        self._label = label

    def apply(self, nodes_orig, adj):
        """ Replaces the label of node self._index with self._label.

        Parameters
        ----------
        nodes: list
            a node list.
        adj: list
            an adjacency list.

        Returns
        -------
        nodes: list
            a copy of nodes with the applied edit.
        adj: list
            a copy of adj with the applied edit.

        """
        nodes = copy.copy(nodes_orig)
        self.apply_in_place(nodes, adj)
        return nodes, adj

    def apply_in_place(self, nodes, adj):
        """ Replaces the label of node self._index with self._label.


        Parameters
        ----------
        nodes: list
            a node list.
        adj: list
            an adjacency list.

        """
        nodes[self._index] = self._label

    def __repr__(self):
        return 'rep(%d, %s)' % (self._index, str(self._label))

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, Replacement) and self._index == other._index and self._label == other._label


class Deletion(Edit):
    """ Deletes node self._index and raises all its children to the parent
        node. Note that deleting the root node of the tree results in a
        forest instead of a tree.

    Attributes
    ----------
    _index: int
        The index of the tree node to which this edit is applied.

    """
    def __init__(self, index):
        self._index = index

    def apply(self, nodes_orig, adj_orig):
        """ Deletes node self._index and raises all its children to the parent
        node. Note that deleting the root node of the tree results in a
        forest instead of a tree.

        Parameters
        ----------
        nodes: list
            a node list.
        adj: list
            an adjacency list.

        Returns
        -------
        nodes: list
            a copy of nodes with the applied edit.
        adj: list
            a copy of adj with the applied edit.

        """
        nodes = copy.copy(nodes_orig)
        adj   = copy.deepcopy(adj_orig)
        self.apply_in_place(nodes, adj)
        return nodes, adj

    def apply_in_place(self, nodes, adj):
        """ Deletes node self._index and raises all its children to the parent
        node. Note that deleting the root node of the tree results in a
        forest instead of a tree.

        Parameters
        ----------
        nodes: list
            a node list.
        adj: list
            an adjacency list.

        """
        # delete the nodes entry at index
        del nodes[self._index]
        # correct all adjacency list entries. First, we must raise
        # the children of index to the parent of index
        for i in range(len(adj)):
            if(self._index in adj[i]):
                j = adj[i].index(self._index)
                adj[i] = adj[i][:j] + adj[self._index] + adj[i][j+1:]
                break
        # next we must remove the index'th entry of the adjacency list
        del adj[self._index]
        # finally, we must decrement all entries in the adjacency list
        # bigger than index
        for i in range(len(adj)):
            for j in range(len(adj[i])):
                if(adj[i][j] > self._index):
                    adj[i][j] -= 1

    def __repr__(self):
        return 'del(%d)' % (self._index)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, Deletion) and self._index == other._index


def get_roots(adj):
    """ Returns all roots of a forest, described by an adjacency list.

    Parameters
    ----------
    adj: list
        an adjacency list

    Returns
    -------
    roots: list
        a list of roots, ascendingly sorted.

    """
    # build a parent representation of the adjacency list
    p = [-1] * len(adj)
    for i in range(len(adj)):
        for j in adj[i]:
            p[j] = i
    # test which nodes have no parents; those are the roots
    roots = []
    for i in range(len(adj)):
        if(p[i] < 0):
            roots.append(i)
    return roots


class Insertion(Edit):
    """ Inserts a new node with label self._label as self._child_index'th
        child of node self._parent_index and uses the next self._num_children
        right siblings of itself as its children.

    Attributes
    ----------
    _parent_index: int
        The index of the tree node to which we add a new child.
    _label: str
        The label for the new child node.
    _child_index: int
        The new node will be the _child_indexth child of node _parent_index.
    _num_children: int
        The number of right siblings that will be used as new grandchildren.

    """
    def __init__(self, parent_index, child_index, label, num_children = 0):
        self._parent_index = parent_index
        self._label = label
        self._child_index = child_index
        self._num_children = num_children

    def apply(self, nodes_orig, adj_orig):
        """ Inserts a new node with label self._label as self._child_index'th
        child of node self._parent_index and uses the next self._num_children
        right siblings of itself as its children.

        Note that inserting a new child at the root leads to a forest instead
        of a tree structure.

        Parameters
        ----------
        nodes: list
            a node list.
        adj: list
            an adjacency list.

        Returns
        -------
        nodes: list
            a copy of nodes with the applied edit.
        adj: list
            a copy of adj with the applied edit.

        """
        nodes = copy.copy(nodes_orig)
        adj   = copy.deepcopy(adj_orig)
        self.apply_in_place(nodes, adj)
        return nodes, adj

    def apply_in_place(self, nodes, adj):
        """ Inserts a new node with label self._label as self._child_index'th
        child of node self._parent_index and uses the next self._num_children
        right siblings of itself as its children.

        Note that inserting a new child at the root leads to a forest instead
        of a tree structure.

        Parameters
        ----------
        nodes: list
            a node list.
        adj: list
            an adjacency list.

        """
        # introduce some abbreviations for easier typing
        p = self._parent_index
        c = self._child_index
        C = self._num_children

        if(p < 0):
            # first, handle the special case that we want to insert a new node
            # at the root of the tree, resulting in a forest
            # retrieve the current roots
            roots = get_roots(adj)
            # get the index of the current self._child_index'th root, which
            # will be the index of our new node
            if(c < len(roots)):
                idx = roots[c]
                # use the self._num_children next roots as children of our new
                # node
                children = roots[c:(c+C)]
            else:
                # if this child does not exist yet, append at the end
                idx = len(nodes)
                children = []
        else:
            # if our parent exists, get the index of its self._child_index'th
            # child, which will be the index of our new node
            if(c < len(adj[p])):
                idx = adj[p][c]
                # use the self._num_children right siblings as children of our new
                # node
                children = adj[p][c:(c+C)]
                del adj[p][c:(c+C)]
            else:
                # if this child does not exist yet, use the index of the
                # right sibling of the parent node
                # this index can be retrieved by first getting the outermost
                # right leaf of p.
                orl = p
                while(adj[orl]):
                    orl = adj[orl][-1]
                # and then incrementing it by 1
                idx = orl + 1
                children = []
            # insert the new child to the parent
            adj[p].insert(c, idx)
        # insert a new entry in the node list
        nodes.insert(idx, self._label)
        # insert a new entry into the adjacency list
        adj.insert(idx, children)
        # increment all indices >= idx
        for i in range(len(adj)):
            for j in range(len(adj[i])):
                if(adj[i][j] >= idx):
                    adj[i][j] += 1
        # decrement the actual new index again
        if(p >= 0):
            adj[p][c] -= 1

    def __repr__(self):
        return 'ins(%d, %d, %s, %d)' % (self._parent_index, self._child_index, self._label, self._num_children)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, Insertion) and self._parent_index == other._parent_index and self._child_index == other._child_index and self._label == other._label and self._num_children == other._num_children


class Script(list, Edit):
    """ A list of Edits.

    """
    def __init__(self, lst = []):
        list.__init__(self, lst)

    def apply(self, nodes_orig, adj_orig):
        """ Applies all edits in this script.


        Parameters
        ----------
        nodes: list
            a node list.
        adj: list
            an adjacency list.

        Returns
        -------
        nodes: list
            a copy of nodes with the applied edits.
        adj: list
            a copy of adj with the applied edits.

        """
        # if this script is empty, just return the inputs
        if(not self):
            return nodes_orig, adj_orig
        # otherwise make a copy and then apply the script in place
        nodes = copy.copy(nodes_orig)
        adj   = copy.deepcopy(adj_orig)
        self.apply_in_place(nodes, adj)
        return nodes, adj

    def apply_in_place(self, nodes, adj):
        """ Applies all edits in this script.

        Parameters
        ----------
        nodes: list
            a node list.
        adj: list
            an adjacency list.

        """
        for e in range(len(self)):
            self[e].apply_in_place(nodes, adj)


def alignment_to_script(alignment, x_nodes, x_adj, y_nodes, y_adj):
    """ Converts the given alignment into an edit script which maps the given
    tree x to the given tree y such that all tuples in the alignment
    correspond to exactly one edit in the script.

    Note that the order of operations does change because the script will
    first apply replacements (in input order), then deletions
    (in descending order), and finally insertions (in ascending order),
    which simplifies conversion.

    The precise algorithm is described in the paper:
    https://arxiv.org/abs/1805.06869

    Parameters
    ----------
    alignment: class alignment.Alignment
        an Alignment object which maps between the given trees x and y.
    x_nodes: list
        the node list of tree x.
    x_adj: list
        the adjacency list of tree x.
    y_nodes: list
        the node list of tree y.
    y_adj: list
        the adjacency list of tree y.

    Returns
    -------
    script: class tree_edits.Script
        A Script object script such that script.apply(x_nodes, x_adj) =
        (y_nodes, y_adj) where every Tuple in the alignment has one
        corresponding edit.

    """
    # in a first pass over the sequence, we disentangle replacements,
    # deletions, and insertions and convert the replacement right away because
    # they do not change the tree structure
    script = Script()
    dels = []
    inss = []
    for op in alignment:
        if(op._left >= 0):
            if(op._right >= 0):
                # if both left and right index are defined, we have a
                # a replacement that we can convert right away.
                # We ignore replacements that change nothing, though.
                if(x_nodes[op._left] != y_nodes[op._right]):
                    script.append(Replacement(op._left, y_nodes[op._right]))
            else:
                # if only the left index is defined, we have a deletion
                # which we will process later
                dels.append(op._left)
        else:
            # if only the right index is defined, we have an insertion
            # which we will process later
            inss.append(op._right)
    # in a next pass, we convert all deletions in descending order
    dels.reverse()
    dels.sort(reverse=True)
    for i in dels:
        script.append(Deletion(i))
    # the insertion conversion is, unfortunately, somewhat more complex.
    # For preparation, we need to sort the insertion indices ascendingly ...
    inss.sort()
    # then find, for each node, its parent index and its child index, ...
    ps = [-1] * len(y_nodes)
    cs = [-1] * len(y_nodes)
    cs[0] = 0
    for i in range(len(y_nodes)):
        for c in range(len(y_adj[i])):
            j = y_adj[i][c]
            cs[j] = c
            ps[j] = i
    # and, finally, the number of non-inserted descendants.
    Cs = num_descendants(y_adj, inss)
    # after this preparation, convert the insertions
    for i in inss:
        script.append(Insertion(ps[i], cs[i], y_nodes[i], Cs[i]))
    # return the script
    return script


def num_descendants(adj, filter_set):
    """ Counts the number of descendants of each node which are _not_ members
    of the given filter set.

    Parameters
    ----------
    adj: list
        an adjacency list
    filter_set: set_like
        a set excluding some node indices

    Returns
    -------
    out: list
        A list which contains, for each node, the number of descendants
        not from the filter set.

    """
    out = []
    _num_descendants(adj, filter_set, 0, out)
    return out

def _num_descendants(adj, filter_set, i, out):
    # add a counting variable for the number of descendants for the current node
    out.append(0)
    # iterate over all children of the current node
    for j in adj[i]:
        # call this method recursively for the child
        _num_descendants(adj, filter_set, j, out)
        # check if the current child is in the filter set
        if(j in filter_set):
            # if it is, add the number of descendants of the child
            out[i] += out[j]
        else:
            # otherwise, add only 1
            out[i] += 1
