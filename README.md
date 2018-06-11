# TreFiDe - Trend Filter Denoising

## Setup

### Dependencies:
- A Linux OS, Ubuntu 16.04 recommended (Installation support for MacOS and Windows coming);
- Python3;
- Required Python Packages: numpy, scipy, cython;
- Recommended Python Packages (To run demos, generate plots, and render videos): matplotlib, jupyter, opencv3; 
- Intel MKL (see below for instructions);
- C compiler (installation currently requires ```icc```, support for ```gcc``` coming);

This package contains C++ source code with Cython wrappers which need to be built on your system. 
The easiest way to ensure all the required libraries are installed is to follow the instructions for installing & setting up [Intel MKL](https://software.intel.com/en-us/mkl) (which is a free product for both personal and commercial applications).
Additionally, you will need a C++ compiler. For ease of implementation, the current installation scripts are setup for the [Intel C Compiler](https://software.intel.com/en-us/c-compilers) ```icc``` (which is free for students and academics). Support to optionally use ```gcc``` (which is more commonly available by default) is coming.

### Installing:
Ensure that the neccessary dependencies are installed and that your the python environment you wish to install trefide into (we highly recommend using ```conda``` contained in the [Anaconda & Miniconda disctributions](https://www.anaconda.com/download/#linux) to manage your python environments) is active.
1. Clone the repository by navigating to the location you wish to install the package and executing```git clone git@github.com:ikinsella/trefide.git```. The absolute path to the location mentioned above will be refered to as ```/path/to/install/directory``` for the remainder of these instructions.
2. Add the location of the C++ libraries to your shared library path by appending the lines
```
export TREFIDE="/path/to/install/directory/trefide"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/src"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/src/proxtv"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/src/glmgen/lib"
```
to your ```.bashrc``` file.
3. Compile the C++ source code by running 
```
cd /path/to/install/directory/trefide/src
make
```
4. Build the Cython wrappers and use pip to create an "editable" installation in your active python environment by running
```
cd /path/to/install/directory/trefide
LDSHARED="icc -shared" CC=icc CXX=icpc pip install -e /path/to/trefide
```
5. Execute PMD demo code using the sample data [here](https://drive.google.com/file/d/1v8E61-mKwyGNVPQFrLabsLsjA-l6D21E/view?usp=sharing) to ensure that the installation worked correctly.

### Rebuilding & Modification
If you modify or pull updates to any C++ &/or Cython code, the C++ &/or Cython code (respectively) will need to be rebuilt for changes to take effect. This can be done by running the following lines
- C++:
  ```
  cd /path/to/install/directory/trefide/src
  make clean
  make
  ```
- Cython:
  ```
  cd /path/to/install/directory/trefide
  LDSHARED="icpc -shared" CXX=icpc CC=icc python setup.py build_ext --inplace
  ``` 

### Uninstalling
The project can be uninstalled from an active python environment at any time by running ```pip uninstall trefide```. If you wish to remove the entire project (all of the files you cloned) from your system, you should also run ```rm -rf /path/to/install/directory/trefide```.

## References:
- [preprint](https://www.biorxiv.org/content/early/2018/06/03/334706.article-info)
- support [slack channel](https://join.slack.com/t/trefide/shared_invite/enQtMzc5NDM4MDk4OTgxLWE0NjNhZGE5N2VlMTcxNGEwODhkMmFlMjcyYmIzYTdkOGVkYThhNjdkMzEyZmM1NzIzYzc0NTZkYmVjMDY5ZTg)
