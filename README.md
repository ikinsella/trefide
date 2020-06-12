# TreFiDe - Trend Filter Denoising

TreFiDe is the software package accompanying the research publication
["Penalized matrix decomposition for denoising, compression, and improved
demixing of functional imaging data"](https://doi.org/10.1101/334706).

TreFiDe is an imporved appproach to compressing and denoising functional image
data. The method is based on a spatially-localized penalized matrix
decomposition (PMD) of the data to separate (low-dimensional) signal from
(temporally-uncorrelated) noise. This approach can be applied in parallel on
local spatial patches and is therefore highly scalable, does not impose
non-negativity constraints or require stringent identifiability assumptions
(leading to significantly more robust results compared to NMF), and estimates
all parameters directly from the data, so no hand-tuning is required. We have
applied the method to a wide range of functional imaging data (including
one-photon, two-photon, three-photon, widefield, somatic, axonal, dendritic,
calcium, and voltage imaging datasets): in all cases, we observe ~2-4x
increases in SNR and compression rates of 20-300x with minimal visible loss of
signal, with no adjustment of hyperparameters; this in turn facilitates the
process of demixing the observed activity into contributions from individual
neurons. We focus on two challenging applications: dendritic calcium imaging
data and voltage imaging data in the context of optogenetic stimulation. In
both cases, we show that our new approach leads to faster and much more robust
extraction of activity from the video data.

## Getting Started

### Docker

1. `docker run -it -p 34000:34000 paninski/trefide:1.2`

2. `localhost:34000` (in a browser of your choise)

### Build from source

#### Prerequisites

- [Anaconda](https://docs.anaconda.com/anaconda/install/) or
  [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/)

- Linux (this package was developed & tested on Ubuntu 18.04)

Note: these instructions will assume that you clone the repo into your home
directory

1. Clone the repository

```Bash
git clone git@github.com:ikinsella/trefide.git
```

2. Add the location of the C++ libraries to your shared library path by
appending the lines to your ```.bashrc``` file.

```Bash
export TREFIDE="${HOME}/trefide"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/external/proxtv"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/external/glmgen/lib"
```

3. Make sure to source your `.bashrc`

```Bash
source ~/.bashrc
```

4. Compile the C++ source code by running

```Bash
cd /path/to/install/directory/trefide/src
make all
```

5. Build the Cython wrappers and use pip to create an "editable" installation
   in your active python environment by running
```
cd /path/to/install/directory/trefide
LDSHARED="icc -shared" CC=icc CXX=icpc pip install -e /path/to/trefide
```

6. Execute PMD demo code using the sample data
   [here](https://drive.google.com/file/d/1v8E61-mKwyGNVPQFrLabsLsjA-l6D21E/view?usp=sharing)
   to ensure that the installation worked correctly.

### Rebuilding & Modification
If you modify or pull updates to any C++ &/or Cython code, the C++ &/or Cython
code (respectively) will need to be rebuilt for changes to take effect. This
can be done by running the following lines

- C++

```Bash
cd /path/to/install/directory/trefide/src
make all
```

- Cython:
```Bash
cd /path/to/install/directory/trefide
LDSHARED="icpc -shared" CXX=icpc CC=icc python setup.py build_ext --inplace
```

### Uninstalling
The project can be uninstalled from an active python environment at any time by
running ```pip uninstall trefide```. If you wish to remove the entire project
(all of the files you cloned) from your system, you should also run:

```Bash
pip uninstall trefide
rm -rf ~/trefide
```

## References
preprint:
```
@article {Buchanan334706,
    author = {Buchanan, E. Kelly and Kinsella, Ian and Zhou, Ding and Zhu, Rong and Zhou, Pengcheng and Gerhard, Felipe and Ferrante, John and Ma, Ying and Kim, Sharon and Shaik, Mohammed and Liang, Yajie and Lu, Rongwen and Reimer, Jacob and Fahey, Paul and Muhammad, Taliah and Dempsey, Graham and Hillman, Elizabeth and Ji, Na and Tolias, Andreas and Paninski, Liam},
    title = {Penalized matrix decomposition for denoising, compression, and improved demixing of functional imaging data},
    elocation-id = {334706},
    year = {2019},
    doi = {10.1101/334706},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2019/01/21/334706},
    eprint = {https://www.biorxiv.org/content/early/2019/01/21/334706.full.pdf},
    journal = {bioRxiv}
}
```

## Troubleshooting

- [slack channel](https://join.slack.com/t/trefide/shared_invite/enQtMzc5NDM4MDk4OTgxLWE0NjNhZGE5N2VlMTcxNGEwODhkMmFlMjcyYmIzYTdkOGVkYThhNjdkMzEyZmM1NzIzYzc0NTZkYmVjMDY5ZTg)
