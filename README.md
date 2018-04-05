# TreFiDe - Trend Filtering Denoising
This package contains C source wrapped by Cython which needs to be built. The easiest (and highest performant) way to do this is to ensure you've first installed and setup [Intel MKL](https://software.intel.com/en-us/mkl) which contains a collection of C libraries. After this has been done, all source can be built with the line ```python setup.py build_ext --inplace``` called from the top level directory (where setup.py is located) which will use your system's ```gcc``` compiler by default. Alternatively, I encourage (for ease and performance) that you install the [Intel C Compiler](https://software.intel.com/en-us/c-compilers) ```icc```, which is free for students and academics, and use it in place of ```gcc``` by preferencing the setup line: ```CC=icc python setup.py build_ext --inplace```.

Dependencies:
- Intel MKL (see above)
- Python3
- MacOS or Linux (untested on Windows)
