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
                   libraries=["lapacke", "lapack", "blas"]),
         Extension("solvers.ipm", ["solvers/ipm.pyx"],
                   include_dirs=[numpy.get_include()],
                   libraries=["blas", "lapack", "m"]), ]
    ),
    cmdclass={"build_ext": build_ext},
    packages=["solvers", ],
)
