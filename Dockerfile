FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x ./miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH /opt/conda/bin:$PATH
ENV CONDA_PREFIX=/opt/conda
ENV PATH=$CONDA_PREFIX/bin:$PATH

ENV TREFIDE="/root/trefide"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/external/proxtv"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/external/glmgen/lib"

RUN git -C ~/ clone https://github.com/ikinsella/trefide.git \
    && cd ~/trefide \
    && git checkout optimizations \
    && /opt/conda/bin/conda env update -n root -f environment.yml \
    && /opt/conda/bin/conda clean -fya \
    && ./install_mkl.sh

# ENV MKLROOT /opt/intel/compilers_and_libraries_2020.0.166/linux/mkl

WORKDIR /root/trefide

RUN bash -c ". /opt/intel/bin/compilervars.sh intel64; make all; pip install ."

EXPOSE 34000

# Run Jupyter Notebook
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=34000", "--allow-root", "--no-browser"]
