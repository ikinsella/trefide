FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    gnupg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x ./miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH /opt/conda/bin:$PATH
ENV CONDA_PREFIX=/opt/conda
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN git -C ~/ clone https://github.com/ikinsella/trefide.git \
    && cd ~/trefide \
    && /opt/conda/bin/conda env update -n root -f environment.yml \
    && /opt/conda/bin/conda clean -fya

WORKDIR /root/trefide

RUN bash -c "make clean; make all -j $(nproc); pip install ."

EXPOSE 34000

# Run Jupyter Notebook
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=34000", "--allow-root", "--no-browser"]
