from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import os
import numpy

# ---------------------- COMPILER ARGS, LIBS, & LOCS -------------------------#

# Using Intel MKL Libs For Lapack, Lapacke, Blas, Vector Math & FFT
LIBRARIES = ["mkl_core",
             "mkl_intel_lp64",
             "trefide",
             "proxtv",
             "iomp5",
             "m"]
            #"iomp5",
            #"mkl_intel_thread",

# Compiled ProxTV CPP Libs From Submodule

# icc optimizations & location of trefide headers
COMPILE_ARGS = ["-O3",
                "-mkl=sequential",
                "-qopenmp",
                "-I/home/ian/devel/trefide/src",
                "-I/home/ian/devel/trefide/proxTV/src",
                "-D NOMATLAB=1"]
                  #"-mkl=parallel",
                  #"-qopenmp",

# location of proxtv headers and ignore mexing for MATLAB

# Location of libtrefide.so
LINK_ARGS = ["-mkl=sequential",
             "-qopenmp",
             "-L/home/ian/devel/trefide/src",
             "-L/home/ian/devel/trefide/proxTV/src"]
            # "-mkl=parallel", "-qopenmp", 


# ---------------------------- SETUP MODULES ---------------------------------#

setup(
    name="trefide",
    ext_modules=cythonize(
        [Extension("trefide.solvers.time.lagrangian",
                   [os.path.join("trefide",
                                 "solvers",
                                 "time",
                                 "lagrangian.pyx")],
                   include_dirs=[numpy.get_include()],
                   language="c++",
                   libraries=LIBRARIES,
                   extra_compile_args=COMPILE_ARGS,
                   extra_link_args=LINK_ARGS),
         Extension("trefide.solvers.time.constrained",
                   [os.path.join("trefide",
                                 "solvers",
                                 "time",
                                 "constrained.pyx")],
                   include_dirs=[numpy.get_include()],
                   language="c++",
                   package_data={"constrained": "*.pxd"},
                   libraries=LIBRARIES,
                   extra_compile_args=COMPILE_ARGS,
                   extra_link_args=LINK_ARGS),
         Extension("trefide.temporal",
                   [os.path.join("trefide", "temporal.pyx")],
                   include_dirs=[numpy.get_include()],
                   language="c++",
                   package_data={"temporal": "*.pxd"},
                   libraries=LIBRARIES,
                   extra_compile_args=COMPILE_ARGS,
                   extra_link_args=LINK_ARGS),
         Extension("trefide.solvers.space.lagrangian",
                   [os.path.join("trefide",
                                 "solvers",
                                 "space",
                                 "lagrangian.pyx")],
                   include_dirs=[numpy.get_include()],
                   language="c++",
                   libraries=LIBRARIES,
                   extra_compile_args=COMPILE_ARGS,
                   extra_link_args=LINK_ARGS),
         Extension("trefide.pmd",
                   [os.path.join("trefide", "pmd.pyx")],
                   include_dirs=[numpy.get_include()],
                   language="c++",
                   libraries=LIBRARIES,
                   extra_compile_args=COMPILE_ARGS,
                   extra_link_args=LINK_ARGS)]
    ),
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    install_requires=["numpy", "scipy", "cython", "matplotlib"]
)


#         Extension("trefide.pmd",
#                   [os.path.join("trefide", "pmd.pyx")],
#                   include_dirs=[numpy.get_include()],
#                   language="c++",
#                   libraries=CPP_LIBRARIES,
#                   extra_compile_args=CPP_COMPILE_ARGS,
#                   extra_link_args=CPP_LINK_ARGS)
