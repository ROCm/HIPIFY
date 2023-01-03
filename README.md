# HIPIFY
HIPIFY is a set of tools to translate CUDA source code into portable [HIP](https://github.com/ROCm-Developer-Tools/HIP) C++ automatically.

## Documentation
Information about HIPIFY and other user topics can be found in the [HIPIFY documentation](https://rocmdocs.amd.com/projects/HIPIFY/en/latest/).
The documentation source is browseable [here](docs/source/index.md).

## Prerequisites
The [ROCm website](https://docs.amd.com) describes how to set up the ROCm repositories and install the required platform dependencies.

## Installing pre-built packages
With the AMD ROCm package repositories installed, the `hipify` package can be retrieved from the system package manager. For example, on Ubuntu:

    sudo apt-get update
    sudo apt-get install hipify
