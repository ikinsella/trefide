# TreFiDe - Trend Filter Denoising

## Dependencies:
- MacOS or Linux (only tested on MacOS High Sierra and Ubuntu 16.04))
- Python3
- Python Packages: Numpy, Scipy, Cython, Matplotlib
- Intel MKL (see below for instructions)
- C compiler (only tested for ```gcc``` and ```icc```, see below for instructions)

This package contains C source wrapped by Cython which needs to be built on your system. 
The easiest (and highest performant) way to ensure all the required libraries are installed is to follow the instructions for installing & setting up [Intel MKL](https://software.intel.com/en-us/mkl). 
Additionally, you will need a C compiler. 
You likely already have ```gcc``` available (used as Cython's default) on MacOS with XCode installed and on your Linux distribution by default. 
Alternatively, the [Intel C Compiler](https://software.intel.com/en-us/c-compilers) ```icc``` is the prefered for ease and performance (free for students and academics). This compiler would be used in place of ```gcc``` by preferencing all calls to 

```python setup.py <options>``` or ```pip install -e <path/to/project>```

in these instructions with ```CC=icc``` on MacOS or ```LDSHARED="icc -shared" CC=icc``` on Linux. For example the source can be built/rebuilt (instructions below) with the lines:

```CC=icc python setup.py build_ext --inplace``` on MacOS 

or 

```LDSHARED="icc -shared" CC=icc python setup.py build_ext --inplace``` on Linux.

## Installation:

There are currently (4) options to install the package into your python environment (i.e. your python distribution should be able to import trefide and it's submodules from any directory regardless of the location in which the package is installed). These options are listed in decreasing order of preference:

1. Run ```pip install -e /path/to/trefide``` this will create an "editable" installation which will change as you modify code in the repository or pull updates. Pip will also download any Python package dependecies you happen to be missing. The project can be uninstalled at any time by running ```pip uninstall trefide```. 
2. Run ```python setup.py install``` in the top level directory of this project. This will have the same functionality as the above, but it will install missing dependencies using ```easy_install``` (less prefereable than pip). It can be uninstalled at any time by running ```python setup.py develop uninstall```.
3. Manually install python package dependencies, build project sources as described in the following section, and add ```/absolute/path/to/trefide``` to your systems ```PYTHONPATH``` environment variable.
4. Last resort: this approach will make a full installation of the package on your system. This is undesirable as the installation will not be updated as you pull changes or modify code. Additionally, there are no nice tools to uninstall, so you will need to force another installation over the first whenever updates are pushed (which will happen frequently since this is still under development). Only do this is you NEED a full installation. Run ```python setup.py install``` in the top level directory. Reinstallation of updates can be forces via ```python setup.py install --force```. 

## Building / Re-building Sources:

The source should be built during installation, but if you modify or pull updates to any Cython/C code int the repository, the project can be rebuilt by running ```python setup.py build_ext --inplace``` from the top level directory (where setup.py is located). 

