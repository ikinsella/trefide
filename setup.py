import os
import numpy

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext


# -------------------- DEPS, COMPILER ARGS, LIBS, & LOCS ---------------------#

# Additional  (Optional) Packages May Be Needed To MAke Videos In Demos
DEPENDENCIES = ["numpy", "scipy", "cython", "matplotlib"]

TREFIDE = os.getcwd()

# We compile against intel mkl will need an intel MKL Disto:
LIBRARIES = ["mkl_core",          # Used for FFT, Vector Math, CBLAS, Lapacke, ect...
             "mkl_intel_lp64",    # Intel MKL LP64 libs
             "mkl_intel_thread",  # Intel MKL Threading Runtime
             "trefide",           # Our cpp sourcecode which we compile to libtrefide.so
             "proxtv",            # ProxTV's cpp sourcecode which we compile to libproxtv.so
             "glmgen",            # glmgen's c sourcecode which we compile to libglmgen.so
             "iomp5",             # Intel mkl OpenMP runtime
             "m"                  # <math.h>
            ]

# Defaults Set Here Are For using Intel icc/icpc Compilers
COMPILE_ARGS = ["-O3",              # Max compiler optimizations
                "-mkl=sequential",  # Use if processing blocks in parallel (single core/block)
                #"-mkl=parallel",  # Use if processing blocks sequentially (multiple cores/block)
                                    # Remove -mkl args entirely if not using icc/icpc to compile
                "-qopenmp",         # Tell icc/icpc to have OpenMP use OMP_NUM_THREADS when it encounters directives
                # "-fopenmp"        # Tell gcc/g++ to have OpenMP use OMP_NUM_THREADS when it encounters directives
                "-I" + os.path.join(TREFIDE, "src"),              # Location of trefide.h
                "-I" + os.path.join(TREFIDE, "src", "proxtv"),    # Location Of proxtv.h
                "-I"+os.path.join(TREFIDE, "src", "glmgen", "include"), # Location of glmgen.h 
                "-D NOMATLAB=1"                                   # Ignore ProXTV's attempt to mex
                  # Location Of glmgen.h
               ]

# See Above
LINK_ARGS = ["-mkl=sequential",     # See Above
             #"-mkl=parallel",     # See Above
                                    # See Above
             "-qopenmp",            # See Above
             # "-fopenmp",          # See Above
             "-L" + os.path.join(TREFIDE, "src"),            # Location of libtrefide.so (Add to $LD_LIBRARY_PATH)
             "-L" + os.path.join(TREFIDE, "src", "proxtv"),  # Location of libproxtv.so  (Add to $LD_LIBRARY_PATH)
             "-L"+os.path.join(TREFIDE, "src", "glmgen", "lib")  # Location of libglmgen.so (Add to $LD_LIBRARY_PATH)
            ]

# ---------------------------- Build Cythonized Modules ---------------------------------#


setup(
    name="trefide",
    ext_modules=cythonize(
        [Extension("trefide.solvers.temporal",
                   [os.path.join("trefide",
                                 "solvers",
                                 "temporal.pyx")],
                   include_dirs=[numpy.get_include()],
                   language="c++",
                   libraries=LIBRARIES+["glmgen"],
                   package_data={"trefide/solvers/temporal": "*.pxd"},
                   extra_compile_args=COMPILE_ARGS,
                   extra_link_args=LINK_ARGS),
         Extension("trefide.utils",
                   [os.path.join("trefide", "utils.pyx")],
                   include_dirs=[numpy.get_include()],
                   language="c++",
                   libraries=LIBRARIES,
                   extra_compile_args=COMPILE_ARGS,
                   extra_link_args=LINK_ARGS),
         Extension("trefide.decimation",
                   [os.path.join("trefide", "decimation.pyx")],
                   include_dirs=[numpy.get_include()],
                   language="c++",
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
    install_requires=DEPENDENCIES
)
