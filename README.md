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

- Linux (this package was developed & tested on Ubuntu 18.04, Ubuntu 20.04, and Manjaro)

*Note: these instructions will assume that you clone the repo into your home
directory*

1. Clone the repository

```Bash
git clone git@github.com:ikinsella/trefide.git
```

2. Navigate into the trefide repo you just cloned:
```Bash
cd ~/trefide
```

3. Create the conda environment using the provided config:
```Bash
conda env create -f environments/devel.yml
```

4. Activate the conda environment:
```Bash
conda activate trefide_devel
```

5. Compile the underlying source code (written in C++) by running
```Bash
make all -j $(nproc)
```

6. Compile the Cython extensions and install the trefide library:
```Bash
pip install .
```

### Try it out!

1. Execute PMD demo code using a sample dataset:
```Bash
cd ~/trefide
jupyter notebook demos/Matrix_Decomposition/Demo_PMD_Compression_Denoising.ipynb --no-browser --port=34000
```

The aforementioned notebook automatically downloads the sample dataset on your
behalf. If you wish to manually download the sample dataset, it is available
[here](https://drive.google.com/file/d/1v8E61-mKwyGNVPQFrLabsLsjA-l6D21E/view?usp=sharing).

### Rebuilding & Modification
If you modify or pull updates this package will need to be rebuilt for the
changes to take effect. This can be done as follows:

```Bash
make clean && make all -j $(nproc) && pip uninstall trefide -y && pip install .
```

### Uninstalling
If you wish to remove the entire project from your machine, you can run:

```Bash
conda deactivate trefide_devel
conda remove --name trefide_devel --all
rm -rf ~/trefide
```

## References
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
