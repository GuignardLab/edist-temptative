# Python Edit Distances

A library for edit distances in Python 3, based on the
[TCS Alignment Toolbox][1]. If you use this library in academic work, please
cite

* Paaßen, B., Mokbel, B., & Hammer, B. (2015). A Toolbox for Adaptive Sequence
    Dissimilarity Measures for Intelligent Tutoring Systems. In O. C. Santos,
    J. G. Boticario, C. Romero, M. Pechenizkiy, A. Merceron, P. Mitros,
    J. M. Luna, et al. (Eds.), Proceedings of the 8th International Conference
    on Educational Data Mining (pp. 632-632). International Educational
    Datamining Society. ([Link][0])

# Quickstart Guide

Once the library is compiled (see below), you can use the following command
to compute the standard edit distance between two lists x and y:

```
from edist.sed import sed
sed(x, y, delta)
```

where `delta` is a function that takes two elements of the lists as input
and returns the local distance. Other distances implemented in this library
are:

* the dynamic time warping distance as `edist.dtw.dtw`
* the tree edit distance as `edist.ted.ted`

Please refer to the respective files for more information.

If you wish to compute pairwise edit distances for a whole dataset of lists
or trees, please use the `multiprocess` module. For example, if Xs ans Ys
are lists of lists, then the following command produces a distance matrix
between all elements of Xs and all elements of Ys:

```
from edist.sed import sed
from edist.multiprocess import pairwise_distances

D = pairwise_distances(Xs, Ys, sed, delta)
```

# Building from source

This library is written in [cython][2], such that some sources need to be
compiled to C. Once you have installed [cython][2], execute the following
commands in this directory:

```
python3 cython_setup.py build_ext --inplace
cp *so edist/.
```

[0]:https://pub.uni-bielefeld.de/publication/2762087
[1]:https://openresearch.cit-ec.de/projects/tcs
[2]:https://cython.org/
