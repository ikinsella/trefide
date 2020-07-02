#!/usr/bin/env bash

# Configure Anaconda Client To Upload Successful Builds
anaconda login
conda config --set anaconda_upload yes

# Run Builds Across Matrix Specified In Config
conda-build . -c intel -c defaults --variant-config-file conda_build_config.yaml
