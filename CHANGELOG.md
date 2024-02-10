# Changelog for HIPIFY

Documentation for HIPIFY is available at
[https://rocmdocs.amd.com/projects/HIPIFY/en/latest/](https://rocmdocs.amd.com/projects/HIPIFY/en/latest/).

## HIPIFY for ROCm 6.1.0

### Additions

* CUDA 12.3.2 support
* cuDNN 8.9.7 support
* LLVM 17.0.6 support
* Full `hipSOLVER` support
* Full `rocSPARSE` support
* New options:
  * `--amap` to hipify as much as possible, ignoring '--default-preprocessor' behavior

### Fixes

* Do not rewrite tokens in code blocks skipped by Preprocessor (under the '--default-preprocessor' option)

## HIPIFY for ROCm 6.0.2

### Fixes

* Use the new locations of header files of some HIP and ROCm libraries (`hipRAND`, `hipFFT`, `rocSOLVER`)

## HIPIFY for ROCm 6.0.0

### Additions

* CUDA 12.2.2 support
* cuDNN 8.9.5 support
* LLVM 17.0.3 support
* Improved support for Windows and Visual Studio 2019 and 2022
* More rocSPARSE support
* ABI changes are shown in the 'C' ('Changed') column for CUDA, HIP, and ROC API

### Known issues

* [#837] Added a new function to call transformation type "additional non-const arg"
* [#1014] Added a new function to call transformation type "replace argument with a const"

## HIPIFY for ROCm 5.7.0

### Additions

* CUDA 12.2.0 support
* cuDNN 8.9.2 support
* LLVM 16.0.6 support
* Initial rocSPARSE support
* Initial `CUDA2ROC` documentation generation for rocBLAS, rocSPARSE, and MIOpen:
  * In separate files: `hipify-clang --md --doc-format=full --doc-roc=separate`
  * In one file: `hipify-clang --md --doc-format=full --doc-roc=joint`
* New options:
  * `--use-hip-data-types` (Use 'hipDataType' instead of 'hipblasDatatype_t' or 'rocblas_datatype')
  * `--doc-roc=\<value\>` (ROC documentation generation: `skip` (default), `separate`, and `joint`; the
    `--md` or `--csv` option must be included)

### Known issues

* [#822] Added a new function to call transformation type "additional const by value arg"
* [#830] Added a new function to call transformation type "move arg from place X to place Y"

## HIPIFY for ROCm 5.6.0

### Additions

* CUDA 12.1.0 support
* cuDNN 8.8.1 support
* LLVM 16.0.0 support
* New options:
  * `--default-preprocessor` (synonymous with `--skip-excluded-preprocessor-conditional-blocks`)
  * `--no-undocumented-features`
  * `--no-warnings-on-undocumented-features`
  * `--versions`

### Fixes

* Accessing `half2 struct` members (undocumented feature)
* Retargeted `INSTALL` to the `bin` subfolder

## HIPIFY for ROCm 5.5.0

### Additions

* Partial CUDA 12.0.0 support
* cuDNN 8.7.0 support
* Initial MIOpem support
* cuBLAS 64-bit API (CUDA 12.0) initial support
* rocBLAS and MIOpen synthetic tests
* LLVM 15.0.7 support

### Changes

* Synthetic unit tests for `cuBLAS2rocBLAS` and `cuDNN2MIOpen`

## HIPIFY for ROCm 5.4.1

### Additions

* CUDA 11.8.0 support
* cuDNN 8.6.0 support
* Device types support
* LLVM 15.0.4 support

### Fixes

* Removed `RPATH` definitions (Linux)

## HIPIFY for ROCm 5.4.0

### Additions

* hipRTC support
* Error handling API support
* hipBLAS synthetic tests
* LLVM 15.0.0 support

## HIPIFY for ROCm 5.3.0

### Additions

* CUDA 11.7.0 support
* cuDNN 8.4.1 support
* CUB initial support
* More synthetic tests
* New options:
  * `--hip-kernel-execution-syntax`

### Fixes

* Patches for LLVM 14.0.x (Windows only)
* Add `GNUInstallDirs` for CMake on Linux

### Changes

* LLVM 3.8.0 is out of support
* HIPIFY-specific options support in unit testing

## HIPIFY for ROCm 5.2.0

### Additions

* CUDA 11.6.0 support
* cuDNN 8.3.3 support
* LLVM 14.0.0 support

## HIPIFY for ROCm 5.1.0

### Additions

* CUDA 11.5.1 support
* cuDNN 8.3.2 support

### Fixes

* Hipification of `cuOccupancyMaxPotentialBlockSize` and
  `cuOccupancyMaxPotentialBlockSizeWithFlags`

## HIPIFY for ROCm 5.0.0

### Additions

* CUDA 11.4.2 support
* cuDNN 8.3.2 support
* Initial hipRTC support
* GNU C/C++ 11.2 support
* Visual Studio 2022 support
* LLVM 13.0.0 support
* New options:
  * `--experimental`
  * `--cuda-kernel-execution-syntax`

### Fixes

* Packaging for Debian and RPM Linux distributions
* Undo argument typecasting for four driver API functions (`cuStreamWaitValue32`,
  `cuStreamWaitValue64`, `cuStreamWriteValue32`, and `cuStreamWriteValue64`) because the arguments
  in the corresponding HIP functions are now `uint`

### Changes

* Support for different formats of locally generated documentation
* Experimentally supported APIs

## HIPIFY for ROCm 4.5.0

### Additions

* cuDNN 8.2.4 support
* Initial graph API support
* GNU C/C++ 11.1 support
* LLVM 12.0.1 support

### Fixes

* Abandoned `HIP_DYNAMIC_SHARED`

### Changes

* Synthetic unit tests
* `-std=c++14` by default

## HIPIFY for ROCm 4.3.0

### Additions

* CUDA 11.3.0 support
* cuDNN 8.2.0 support
* LLVM 12.0.0 support

### Fixes

* Added missing type casting arguments for `cuStreamWaitValue32(64)` and
  `cuStreamWriteValue32(64)`

## HIPIFY for ROCm 4.2.0

### Additions

* CUDA 11.2.2 support
* cuDNN 8.1.1 support
* Initial device API support
* LLVM 11.1.0 support
* New options:
  * `--doc-format=<value>`, with `full` (default), `strict`, and `compact` options

### Changes

* Tests on kernel launch syntax

## HIPIFY for ROCm 4.1.0

### Additions

* CUDA 11.2.0 support
* Stream-ordered memory API support
* cuDNN 8.1.1 support
* LLVM 11.0.1 support

### Fixes

* Patches for LLVM 10.0.x and 11.0.0 (Windows and Linux)

### Changes

* Initial support for API versioning

## HIPIFY for ROCm 4.0.0

### Additions

* LLVM 11.0.0 support

## HIPIFY for ROCm 3.10.0

### Changes

* Revised CUDA and HIP API and data type versioning
* Revised and removed deprecated CUDA and HIP APIs and data types

## HIPIFY for ROCm 3.9.0

### Additions

* CUDA 11.0.1 support
* `CUDA2HIP` documentation generation in Markdown and CSV formats
* Versioning support for CUDA and HIP APIs
* New options:
  * `--md` (generate Markdown documentation)
  * `--csv` (generate CSV documentation)

  ### Changes

* Improved `hipify-perl` generation

## HIPIFY for ROCm 3.8.0

### Additions

* cuDNN 8.0.2 support
* `compile_commands.json` support (`-p <build-path>`)

### Changes

* Improved `hipify-perl` generation

## HIPIFY for ROCm 3.7.0

### Additions

* CUDA 11.0.0 support
* Linux packaging
* LLVM 10.0.1 support

## HIPIFY for ROCm 3.6.0

### Additions

* `deprecated` flag for all corresponding CUDA and HIP APIs

### Changes

* Added warning for all deprecated APIs

## HIPIFY for ROCm 3.5.0

### Additions

* CUDA 10.2.0 support
* cuDNN 7.6.5 support
* LLVM 10.0.0 support

### Changes

* `hipify-clang` and `clang` options separator (`--`) support
