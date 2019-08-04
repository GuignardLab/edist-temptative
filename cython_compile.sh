#!/bin/bash
python3 cython_setup.py build_ext --inplace
cp *so edist/.
