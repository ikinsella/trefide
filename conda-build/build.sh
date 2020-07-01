#!/usr/bin/env bash

# Build Shared Objects Directly Into $CONDA_PREFIX/lib
make clean; make all # -j"${CPU_COUNT}"

# Remove Previous Cython Extensions
rm -rf build
rm -f trefide/*.so trefide/solvers/*.so trefide/solvers/space/*.so
rm -f trefide/*.cpp trefide/solvers/*.cpp trefide/solvers/space/*.cpp

# Build/Install Cython Extensions & .pyc
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
