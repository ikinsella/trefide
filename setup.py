from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import os
import numpy

# Using Linalg Libraries From Intel MKL
libraries = ["m", "mkl_core", "mkl_intel_lp64", "mkl_intel_thread", "iomp5"]

setup(
    name="trefide",
    ext_modules=cythonize(
        [Extension("trefide.solvers.lagrangian",
                   [os.path.join("trefide", "solvers", "lagrangian.pyx"),
                    os.path.join("trefide", "solvers", "src", "utils.c")],
                   include_dirs=[numpy.get_include()],
                   libraries=libraries,
                   extra_compile_args=["-O3"]),
         Extension("trefide.solvers.constrained",
                   [os.path.join("trefide", "solvers", "constrained.pyx"),
                    os.path.join("trefide", "solvers", "src", "utils.c"),
                    os.path.join("trefide", "solvers", "src", "wpdas.c")],
                   include_dirs=[numpy.get_include()],
                   libraries=libraries,
                   extra_compile_args=["-O3"]),
         Extension("trefide.temporal",
                   [os.path.join("trefide", "temporal.pyx")],
                   include_dirs=[numpy.get_include()])]
    ),
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    install_requires=["numpy", "scipy", "cython", "matplotlib"]
)
