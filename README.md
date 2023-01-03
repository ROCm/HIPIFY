# HIPIFY
HIPIFY is a set of tools to translate CUDA source code into portable [HIP][2] C++ automatically.

## Documentation
Information about HIPIFY and other user topics can be found in the [HIPIFY documentation][3].

## Prerequisites
The [AMD ROCm install guide][4] describes how to set up the ROCm repositories
and install the required platform dependencies.

## Installing pre-built packages
With the AMD ROCm package repositories installed, the `hipify` package can be
retrieved from the system package manager. For example, on Ubuntu:

    sudo apt-get update
    sudo apt-get install hipify

[1]: https://docs.amd.com
[2]: https://github.com/ROCm-Developer-Tools/HIP
[3]: https://rocmdocs.amd.com/projects/HIPIFY/en/latest/
[4]: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
