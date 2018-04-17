from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
 
# Using Intel MKL Libs For Lapack, Lapacke, Blas, Vector Math & FFT
C_LIBRARIES = ["mkl_core",
               "mkl_intel_lp64",
               "mkl_intel_thread",
               "iomp5",
               "trefide",
               "m"]

# Compiled ProxTV CPP Libs From Submodule
CPP_LIBRARIES = C_LIBRARIES + ["proxtv"]

# icc optimizations & location of trefide headers
C_COMPILE_ARGS = ["-O3",
                  "-qopenmp",
                  "-mkl=parallel",
                  "-I/home/ian/devel/trefide/src"]

# location of proxtv headers and ignore mexing for MATLAB
CPP_COMPILE_ARGS = C_COMPILE_ARGS + ["-I/home/ian/devel/trefide/proxTV/src",
                                     "-D NOMATLAB=1"]

# Location of libtrefide.so
C_LINK_ARGS = ["-qopenmp", "-mkl=parallel", "-L/home/ian/devel/trefide/src"]

# Location of libproxtv.so
CPP_LINK_ARGS = C_LINK_ARGS + ["-L/home/ian/devel/trefide/proxTV/src"]
# Compiled ProxTV CPP Libs From Submodule
CPP_LIBRARIES = C_LIBRARIES + ["proxtv"]

setup(
  name = "multithreads",
  cmdclass = {"build_ext": build_ext},
  ext_modules =
  [
    Extension("multithreads",
              ["multithreads.pyx", "/home/ian/devel/trefide/src/pmd/pmd.cpp"],
              extra_compile_args =CPP_COMPILE_ARGS,
              extra_link_args=CPP_LINK_ARGS,
              language="c++",
              libraries=CPP_LIBRARIES
              )
  ]
)

