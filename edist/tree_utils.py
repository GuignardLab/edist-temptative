"""
Provides general utility functions to process trees.

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

import json
import os
import numpy as np

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def root(adj):
    """ Returns the root of a tree and raises an error if the input adjacency
    matrix does not correspond to a tree.

    Parameters
    ----------
    adj: list
        a directed graph in adjacency list format.

    Returns
    -------
    root: int
        the index of the root of the tree.

    Raises
    ------
    ValueError
        if the given adjacency list does not form a tree.

    """
    if(not adj):
        raise ValueError("The input tree is empty!")

    par = np.full(len(adj), -1, dtype=int)
    for i in range(len(adj)):
        for j in adj[i]:
            if(par[j] < 0):
                par[j] = i
            else:
                raise ValueError("Input is not a tree because node %d has multiple parents" % j)
    root = -1
    for i in range(len(adj)):
        if(par[i] < 0):
            if(root < 0):
                root = i
            else:
                raise ValueError("Input is not a tree because there is more than one root")
    return root

def check_tree_structure(adj):
    """ Verifies that a given adjacency list describes a tree.

    In particular, we build a parent representation of the tree and throw
    an exception if that fails. This is valid because trees are the subclass
    of directed graphs with a unique parent.

    Parameters
    ----------
    adj: list
        a directed graph in adjacency list format.

    Raises
    ------
    ValueError
        if the given adjacency list does not form a tree.

    """
    m = len(adj)
    # initialize an undefined parent array
    par = np.full(m, -1)
    # iterate through the adjacency list and retrieve the parents for each
    # node
    for i in range(m):
        # j iterates over all children of i
        for j in adj[i]:
            # if the parent of j is already defined, j has multiple parents
            # and this is not a tree
            if(par[j] >= 0):
                raise ValueError("Node %d has multiple parents (%d and %d)." % (j, par[j], i))
            par[j] = i
    return par


def check_dfs_structure(adj, i = 0):
    """ Verifies that a given adjacency list is in depth-first-search order,
    in the sense that the descendants of a node i are all indices i+1, i+2,
    ... orl[i], where orl[i] is the outermost right leaf of i.

    We verify this by performing a depth-first search over the tree and
    checking whether the indices are equivalent

    Parameters
    ----------
    adj: list
        a directed graph in adjacency list format.

    Returns
    -------
    size: int
        the size of the input tree, i.e. the last DFS index.

    Raises
    ------
    ValueError
        if the given adjacency list is not in DFS format.

    """
    if(not adj):
        return 0
    # j iterates over all children of adj i
    for j in adj[i]:
        # check the current child
        if(j != i + 1):
            raise ValueError("Expected %d as next child, but got %d" % (i+1, j))
        # check the dfs structure of the current child
        i = check_dfs_structure(adj, i+1)
    # return the current dfs index
    return i


def to_dfs_structure(nodes, adj):
    """ Re-orders a tree to conform to a depth-first search structure.

    Note that this method performs a copy and leaves the original tree
    untouched.

    Parameters
    ----------
    nodes: list
        A node list.
    adj: list
        An adjacency list.

    Raises
    ------
    ValueError
        if the input is not a tree.

    """
    # initialize tree copy
    nodes_dfs = []
    adj_dfs = []
    # get the root of the tree
    r = root(adj)
    # then, perform a depth-first search through the tree
    # and order the tree accordingly
    _to_dfs_structure(nodes, adj, nodes_dfs, adj_dfs, r)
    return nodes_dfs, adj_dfs


def _to_dfs_structure(nodes_orig, adj_orig, nodes_dfs, adj_dfs, i):
    # append the current node to the tree
    nodes_dfs.append(nodes_orig[i])
    # generate a new adjacency list
    adj_i = []
    adj_dfs.append(adj_i)
    # re-order the children of the current node
    for j in adj_orig[i]:
        # add the current child to the new adjacency list
        adj_i.append(len(nodes_dfs))
        # re-order the current child
        _to_dfs_structure(nodes_orig, adj_orig, nodes_dfs, adj_dfs, j)

def to_json(filename, nodes, adj):
    """ Writes a tree in node list/adjacency list format to a JSON file.

    Parameters
    ----------
    filename: str
        The filename for the resulting JSON file.
    nodes: list
        The node list of the tree.
    adj: list
        The adjacency list of the tree.

    Raises
    ------
    Exception
        if the file is not accessible or the JSON writeout fails.

    """
    with open(filename, 'w') as json_file:
        json.dump({'nodes' : nodes, 'adj' : adj}, json_file, indent='\t')


def from_json(filename):
    """Loads a tree in node list/adjacency list format from a JSON file.

    Parameters
    ----------
    filename: str
        A JSON filename containing tree data as written by the to_json method.

    Returns
    -------
    nodes: list
        The node list of the tree.
    adj: list
        The adjacency list of the tree.

    Raises
    ------
    Exception
        if the given file is not accessible, if the JSON data is malformed,
        if the parsed data is not a tree, or if the parsed tree is not in
        depth first search order.

    """
    with open(filename, 'r') as json_file:
        tree = json.load(json_file)
        nodes = tree['nodes']
        adj = tree['adj']
        check_tree_structure(adj)
        check_dfs_structure(adj)
        return nodes, adj

def dataset_from_json(path):
    """ Reads trees in node list/adjacency list format from all JSON files in
    the given directory.

    Parameters
    ----------
    path: str
        a path to a directory which contains JSON files.

    Returns
    -------
    X: list
        A list of tuples in node list/adjacency list format.
    filenames: list
        A list of filenames from which we read the trees.

    Raises
    ------
    Exception
        if the file access does not work, if some JSON data is malformed,
        if a parsed data is not a tree, or if a parsed tree is not in depth
        first search order.

    """
    if(not os.path.isdir(path)):
        raise OSError('%s is not a directory or was not found' % str(path))

    # list all files in the directory and sort them.
    files = os.listdir(path)
    files.sort()
    # initialize an empty output list.
    X = []
    filenames = []
    # iterate over all files and parse them
    for filename in files:
        # note that 'listdir' returns files relative to the input path, so
        # we have to put the input path in front.
        filepath = path + '/' + filename
        # check if the file is a json file.
        if(not os.path.isfile(filepath) or not filename.endswith('.json')):
            continue
        # load the tree
        X.append(from_json(filepath))
        filenames.append(filename)
    return X, filenames

def tree_to_string(nodes, adj, indent = False, with_indices = False):
    """ Prints a tree in node list/adjacency list format as string.

    Parameters
    ----------
    nodes: list
        The node list of the tree.
    adj: list
        The adjacency list of the tree.
    indent: bool (default = False)
        A boolean flag; if True, each node is printed on a new line.
    with_indices: bool (default = False)
        A boolean flag; if True, each node is printed with its index.

    Raises
    ------
    ValueError
        if the adjacency list does not correspond to a tree.

    """
    r = root(adj)
    if(indent):
        indent = 1
    else:
        indent = None
    # translate recursively
    return _tree_to_string(nodes, adj, r, indent, with_indices = with_indices)

def _tree_to_string(nodes, adj, i, indent = None, with_indices = False):
    # the initial string is just the node label
    if(with_indices):
        tree_string = '%d: %s' % (i, str(nodes[i]))
    else:
        tree_string = str(nodes[i])
    # consider the case where node i has chidlren
    if(adj[i]):
        # first, translate the children to their tree strings
        children_strings = []
        for j in adj[i]:
            if(indent is None):
                children_strings.append(_tree_to_string(nodes, adj, j, with_indices = with_indices))
            else:
                children_strings.append(_tree_to_string(nodes, adj, j, indent + 1, with_indices = with_indices))
        # and then join all these strings and append them in brackets
        if(indent is None):
            tree_string += '(' + ', '.join(children_strings) + ')'
        else:
            tree_string += '(\n' + ('\t' * indent) + (',\n' + '\t' * indent).join(children_strings) + '\n' + ('\t' * (indent - 1)) + ')'
    return tree_string

def subtree(nodes, adj, i):
    """ Returns the subtree rooted at node i in node list/adjacency list
    format.

    Parameters
    ----------
    nodes: list
        The node list of the tree.
    adj: list
        The adjacency list of the tree.
    i: int
        The index of the desired subtree.

    Returns
    -------
    nodes_sub: list
        The node list of the subtree rooted at i.
    adj_sub: list
        The adajency list of the subtree rooted at i.

    """
    # perform a depth-first-search from node i to construct the tree.
    nodes_dfs = []
    adj_dfs = []
    _to_dfs_structure(nodes, adj, nodes_dfs, adj_dfs, i)
    return nodes_dfs, adj_dfs

def parents(adj):
    """ Returns the parent representation of the tree with the given
    adjacency list.

    Parameters
    ----------
    adj: list
        The adjacency list of the tree.

    Returns
    -------
    par: int array
        a numpy integer array with len(adj) elements, where the ith
        element contains the index of the parent of the ith node.
        Nodes without children contain the entry -1.

    """
    par = np.full(len(adj), -1, dtype=int)
    for i in range(len(adj)):
        for j in adj[i]:
            par[j] = i
    return par
