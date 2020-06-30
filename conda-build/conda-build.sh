#!/usr/bin/env bash

#conda config --set anaconda_upload yes  # automatically upload successful builds
conda-build . -c defaults -c conda-forge --variant-config-file conda_build_config.yaml
