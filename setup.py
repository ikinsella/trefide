from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import os
import numpy

# setup(
#     name="solvers",
#     ext_modules=cythonize(
#         [Extension("solvers.lagrangian",
#                    [os.path.join("solvers", "lagrangian.pyx"),
#                     os.path.join('solvers', 'src', 'utils.c')],
#                    include_dirs=[numpy.get_include()],
#                    libraries=["m",
#                               "mkl_core",
#                               "mkl_intel_lp64",
#                               "mkl_intel_thread",
#                               "iomp5"],
#                    extra_compile_args=["-O3", "-qopenmp"],
#                    extra_link_args=["-qopenmp"]),
#          Extension("solvers.constrained",
#                    [os.path.join("solvers", "constrained.pyx"),
#                     os.path.join('solvers', 'src', 'utils.c'),
#                     os.path.join('solvers', 'src', 'wpdas.c')],
#                    include_dirs=[numpy.get_include()],
#                    libraries=["m",
#                               "mkl_core",
#                               "mkl_intel_lp64",
#                               "mkl_intel_thread",
#                               "iomp5"],
#                    extra_compile_args=["-O3", "-qopenmp"],
#                    extra_link_args=["-qopenmp"])]
#     ),
#     cmdclass={"build_ext": build_ext},
#     packages=["solvers", ],
# )
#
setup(
    name="temporal",
    ext_modules=cythonize("temporal.pyx"),
    include_dirs=[numpy.get_include()]
)
