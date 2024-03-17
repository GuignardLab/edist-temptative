import setuptools
from setuptools.extension import Extension

try:
    # try to import CYTHON
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    # if it fails, we use c compilation
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension(
        "edist.adp",
        ["edist/adp" + ext],
    ),
    Extension(
        "edist.dtw",
        ["edist/dtw" + ext],
    ),
    Extension(
        "edist.sed",
        ["edist/sed" + ext],
    ),
    Extension(
        "edist.ted",
        ["edist/ted" + ext],
    ),
    Extension(
        "edist.uted",
        ["edist/uted" + ext],
    ),
    Extension(
        "edist.seted",
        ["edist/seted" + ext],
    ),
]

if(USE_CYTHON):
    extensions = cythonize(extensions)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="edist",
    version="1.2.2",
    author="Benjamin Paassen",
    author_email="bpaassen@techfak.uni-bielefeld.de",
    description="Edit distance implementations in cython",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ub.uni-bielefeld.de/bpaassen/python-edit-distances",
    packages=['edist'],
    install_requires=['numpy', 'scikit-learn', 'scipy', 'proto-dist-ml'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    keywords='levenshtein-distance dynamic-time-warping sequence-edit-distance sequence-alignment tree-edit-distance algebraic-dynamic-programming',
    ext_modules=extensions
)
