FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
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
    && /opt/conda/bin/conda env update -n root -f environments/deploy.yml \
    && /opt/conda/bin/conda clean -fya

WORKDIR /root/trefide/conda-build

# TODO: possible to securely configure anaconda-client to automate login?
# Build Conda Environments
ENTRYPOINT ["bash", "-c", "~/trefide/conda-build/conda-build.sh"]
