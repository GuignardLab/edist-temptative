import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "edist.adp",
        ["edist/adp.pyx"],
    ),
    Extension(
        "edist.dtw",
        ["edist/dtw.pyx"],
    ),
    Extension(
        "edist.sed",
        ["edist/sed.pyx"],
    ),
    Extension(
        "edist.ted",
        ["edist/ted.pyx"],
    ),
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="edist",
    version="1.0.0",
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
    ext_modules=cythonize(extensions)
)
