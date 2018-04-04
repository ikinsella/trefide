from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

# setup(
#     name="shrub",
#     ext_modules=cythonize(
#         [Extension("shrub", ["shrub.pyx"],
#                    include_dirs=[numpy.get_include()],
#                    libraries=["blas", "lapack", "m"])]
#     ),
#     cmdclass={"build_ext": build_ext},
#     packages=["shrub", ],
# )
setup(
    name="temporal",
    ext_modules=cythonize(
        [Extension("temporal", ["temporal.pyx"],
                   include_dirs=[numpy.get_include()],
                   libraries=["blas", "lapack", "m"])]
    ),
    cmdclass={"build_ext": build_ext},
    packages=["temporal", ],
)
