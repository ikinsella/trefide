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
        [Extension("trefide.solvers.time.lagrangian",
                   [os.path.join("trefide", "solvers", "time", "lagrangian.pyx"),
                    os.path.join("src", "tf", "utils.c")],
                   include_dirs=[numpy.get_include()],
                   libraries=libraries,
                   extra_compile_args=["-O3"]),
         Extension("trefide.solvers.time.constrained",
                   [os.path.join("trefide", "solvers", "time", "constrained.pyx"),
                    os.path.join("src", "tf", "utils.c"),
                    os.path.join("src", "tf", "wpdas.c")],
                   include_dirs=[numpy.get_include()],
                   package_data={"constrained": "*.pxd"},
                   libraries=libraries,
                   extra_compile_args=["-O3"]),
         Extension("trefide.temporal",
                   [os.path.join("trefide", "temporal.pyx")],
                   include_dirs=[numpy.get_include()],
                   package_data={"temporal": "*.pxd"}),
         Extension("trefide.solvers.space.lagrangian",
                   [os.path.join("trefide", "solvers", "space", "lagrangian.pyx")],
                   include_dirs=[numpy.get_include()],
                   libraries=libraries + ["proxtv"],
                   language="c++",
                   extra_compile_args=["-O3",
                                       "-qopenmp",
                                       "-I/home/ian/devel/trefide/proxTV/src",
                                       "-D NOMATLAB=1"],
                   extra_link_args=["-L/home/ian/devel/trefide/proxTV/src"]),
         Extension("trefide.pmd",
                   [os.path.join("trefide", "pmd.pyx"),
                    os.path.join("src", "tf", "utils.c"),
                    os.path.join("src", "tf", "wpdas.c"),
                    os.path.join("src", "noise", "welch.c")],
                   include_dirs=[numpy.get_include()],
                   language="c++",
                   libraries=libraries + ["proxtv"],
                   extra_compile_args=["-O3",
                                       "-qopenmp",
                                       "-I/home/ian/devel/trefide/proxTV/src",
                                       "-D NOMATLAB=1"],
                   extra_link_args=["-L/home/ian/devel/trefide/proxTV/src"])]
    ),
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    install_requires=["numpy", "scipy", "cython", "matplotlib"]
)
