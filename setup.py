from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

setup(
    name="solvers",
    ext_modules=cythonize(
        [Extension("solvers.pdas",
                   ["solvers/pdas.pyx"],
                   include_dirs=[numpy.get_include()],
                   libraries=["m",
                              "mkl_core",
                              "mkl_intel_lp64",
                              "mkl_intel_thread",
                              "iomp5"],
                   extra_compile_args=["-O3", "-qopenmp"],
                   extra_link_args=["-qopenmp"]),
         Extension("solvers.ipm", ["solvers/ipm.pyx"],
                   include_dirs=[numpy.get_include()],
                   libraries=["blas", "lapack", "m"]), ]
    ),
    cmdclass={"build_ext": build_ext},
    packages=["solvers", ],
)
