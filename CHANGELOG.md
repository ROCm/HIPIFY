# Changelog for HIPIFY

Documentation for HIPIFY is available at
[https://rocmdocs.amd.com/projects/HIPIFY/en/latest/](https://rocmdocs.amd.com/projects/HIPIFY/en/latest/).

## HIPIFY for ROCm 6.4.0

### Added

* CUDA 12.6.3 support
* cuDNN 9.7.0 support
* cuTENSOR 2.0.2.1 support
* LLVM 19.1.7 support
* Full support for direct hipification of `cuRAND` into `rocRAND` under the `--roc` option
* [#1617](https://github.com/ROCm/HIPIFY/issues/1617) Support for `fp8` math device/host API

### Resolved issues

* `MIOpen` support in hipify-perl under the `-miopen` option
* Use `const_cast<const char**>` for the last arguments in the `hiprtcCreateProgram` and `hiprtcCompileProgram` function calls, as in CUDA, they are of the `const char* const*` type
* [#1769](https://github.com/ROCm/HIPIFY/issues/1769) Support for `fp16` device/host API
* [#1800](https://github.com/ROCm/HIPIFY/issues/1800) Fix instructions on building LLVM for HIPIFY on Linux

### Known issues

* [#833](https://github.com/ROCm/HIPIFY/issues/833) `hipify-clang` build failure against LLVM 15-18 on `Ubuntu`, `CentOS`, and `Fedora`

## HIPIFY for ROCm 6.3.1

### Added

* CUDA 12.6.2 support
* cuDNN 9.5.1 support
* LLVM 19.1.3 support
* Full `hipBLAS` 64-bit APIs support
* Full `rocBLAS` 64-bit APIs support

### Resolved issues

* Added missing support for device intrinsics and built-ins: `__all_sync`, `__any_sync`, `__ballot_sync`, `__activemask`, `__match_any_sync`, `__match_all_sync`, `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, and `__shfl_xor_sync`

## HIPIFY for ROCm 6.3.0

### Added

* CUDA 12.6.1 support
* cuDNN 9.5.0 support
* LLVM 19.1.1 support
* `rocBLAS` 64-bit APIs support
* Initial support for direct hipification of `cuDNN` into `MIOpen` under the `--roc` option
* Initial support for direct hipification of `cuRAND` into `rocRAND` under the `--roc` option
* [#1650](https://github.com/ROCm/HIPIFY/pull/1650) Added a filtering ability for the supplementary hipification scripts

### Resolved issues

* Correct `roc` header files support

### Known issues

* [#1617](https://github.com/ROCm/HIPIFY/issues/1617) Support for `fp8` data types

## HIPIFY for ROCm 6.2.4

### Added

* cuDNN 9.3.0 support

### Resolved issues

* Removed some post HIP 6.2 APIs from support
* Added hipification support for HIP functions `hipSetValidDevices`, `hipMemcpy2DArrayToArray`, `hipMemcpyAtoA`, `hipMemcpyAtoD`, `hipMemcpyAtoA`, `hipMemcpyAtoHAsync`, and `hipMemcpyHtoAAsync`
* Fixed an issue with `Skipped some replacements` when hipification didn't occur at all

## HIPIFY for ROCm 6.2.1

### Added

* CUDA 12.5.1 support
* cuDNN 9.2.1 support
* LLVM 18.1.8 support
* `hipBLAS` 64-bit APIs support
* Support for Math Constants `math_constants.h`

## HIPIFY for ROCm 6.2.0

### Added

* CUDA 12.4.1 support
* cuDNN 9.1.1 support
* LLVM 18.1.6 support
* Full `hipBLASLt` support

### Resolved issues

* Apply `reinterpret_cast` for an explicit conversion between `pointer-to-function` and `pointer-to-object`;
  affected functions: `hipFuncGetAttributes`, `hipFuncSetAttribute`, `hipFuncSetCacheConfig`, `hipFuncSetSharedMemConfig`, `hipLaunchKernel`, and `hipLaunchCooperativeKernel`

## HIPIFY for ROCm 6.1.2

### Added

* cuDNN 9.0.0 support
* LLVM 18.1.2 support
* New options:
  * `--clang-resource-directory` to specify the clang resource path - the path to the parent folder for the `include` folder that
    contains `__clang_cuda_runtime_wrapper.h` and other header files used during the hipification process

### Resolved issues

* Clang resource files used during hipification are now searchable and also can be specified by the `--clang-resource-directory` option

## HIPIFY for ROCm 6.1.0

### Added

* CUDA 12.3.2 support
* cuDNN 8.9.7 support
* LLVM 17.0.6 support
* Full `hipSOLVER` support
* Full `rocSPARSE` support
* New options:
  * `--amap` to hipify as much as possible, ignoring `--default-preprocessor` behavior

### Resolved issues

* Code blocks skipped by the Preprocessor are not hipified anymore under the `--default-preprocessor` option

## HIPIFY for ROCm 6.0.2

### Resolved issues

* Use the new locations of header files of some HIP and ROCm libraries (`hipRAND`, `hipFFT`, `rocSOLVER`)

## HIPIFY for ROCm 6.0.0

### Added

* CUDA 12.2.2 support
* cuDNN 8.9.5 support
* LLVM 17.0.3 support
* Improved support for Windows and Visual Studio 2019 and 2022
* More rocSPARSE support
* ABI changes are shown in the 'C' ('Changed') column for CUDA, HIP, and ROC API

### Known issues

* [#837](https://github.com/ROCm/HIPIFY/issues/837) Added a new function to call transformation type "additional non-const arg"
* [#1014](https://github.com/ROCm/HIPIFY/issues/1014) Added a new function to call transformation type "replace argument with a const"

## HIPIFY for ROCm 5.7.0

### Added

* CUDA 12.2.0 support
* cuDNN 8.9.2 support
* LLVM 16.0.6 support
* Initial rocSPARSE support
* Initial `CUDA2ROC` documentation generation for rocBLAS, rocSPARSE, and MIOpen:
  * In separate files: `hipify-clang --md --doc-format=full --doc-roc=separate`
  * In one file: `hipify-clang --md --doc-format=full --doc-roc=joint`
* New options:
  * `--use-hip-data-types` (Use `hipDataType` instead of `hipblasDatatype_t` or `rocblas_datatype`)
  * `--doc-roc=\<value\>` (ROC documentation generation: `skip` (default), `separate`, and `joint`; the
    `--md` or `--csv` option must be included)

### Known issues

* [#822](https://github.com/ROCm/HIPIFY/issues/822) Added a new function to call transformation type "additional const by value arg"
* [#830](https://github.com/ROCm/HIPIFY/issues/830) Added a new function to call transformation type "move arg from place X to place Y"

## HIPIFY for ROCm 5.6.0

### Added

* CUDA 12.1.0 support
* cuDNN 8.8.1 support
* LLVM 16.0.0 support
* New options:
  * `--default-preprocessor` (synonymous with `--skip-excluded-preprocessor-conditional-blocks`)
  * `--no-undocumented-features`
  * `--no-warnings-on-undocumented-features`
  * `--versions`

### Resolved issues

* Accessing `half2 struct` members (undocumented feature)
* Retargeted `INSTALL` to the `bin` subfolder

## HIPIFY for ROCm 5.5.0

### Added

* Partial CUDA 12.0.0 support
* cuDNN 8.7.0 support
* Initial MIOpem support
* cuBLAS 64-bit API (CUDA 12.0) initial support
* rocBLAS and MIOpen synthetic tests
* LLVM 15.0.7 support

### Changed

* Synthetic unit tests for `cuBLAS2rocBLAS` and `cuDNN2MIOpen`

## HIPIFY for ROCm 5.4.1

### Added

* CUDA 11.8.0 support
* cuDNN 8.6.0 support
* Device types support
* LLVM 15.0.4 support

### Resolved issues

* Removed `RPATH` definitions (Linux)

## HIPIFY for ROCm 5.4.0

### Added

* hipRTC support
* Error handling API support
* hipBLAS synthetic tests
* LLVM 15.0.0 support

## HIPIFY for ROCm 5.3.0

### Added

* CUDA 11.7.0 support
* cuDNN 8.4.1 support
* CUB initial support
* More synthetic tests
* New options:
  * `--hip-kernel-execution-syntax`

### Changed

* LLVM 3.8.0 is out of support
* HIPIFY-specific options support in unit testing

### Resolved issues

* Patches for LLVM 14.0.x (Windows only)
* Add `GNUInstallDirs` for CMake on Linux

## HIPIFY for ROCm 5.2.0

### Added

* CUDA 11.6.0 support
* cuDNN 8.3.3 support
* LLVM 14.0.0 support

## HIPIFY for ROCm 5.1.0

### Added

* CUDA 11.5.1 support
* cuDNN 8.3.2 support

### Resolved issues

* Hipification of `cuOccupancyMaxPotentialBlockSize` and
  `cuOccupancyMaxPotentialBlockSizeWithFlags`

## HIPIFY for ROCm 5.0.0

### Added

* CUDA 11.4.2 support
* cuDNN 8.3.2 support
* Initial hipRTC support
* GNU C/C++ 11.2 support
* Visual Studio 2022 support
* LLVM 13.0.0 support
* New options:
  * `--experimental`
  * `--cuda-kernel-execution-syntax`

### Changed

* Support for different formats of locally generated documentation
* Experimentally supported APIs

### Resolved issues

* Packaging for Debian and RPM Linux distributions
* Undo argument typecasting for four driver API functions (`cuStreamWaitValue32`,
  `cuStreamWaitValue64`, `cuStreamWriteValue32`, and `cuStreamWriteValue64`) because the arguments
  in the corresponding HIP functions are now `uint`

## HIPIFY for ROCm 4.5.0

### Added

* cuDNN 8.2.4 support
* Initial graph API support
* GNU C/C++ 11.1 support
* LLVM 12.0.1 support

### Changed

* Synthetic unit tests
* `-std=c++14` by default

### Resolved issues

* Abandoned `HIP_DYNAMIC_SHARED`

## HIPIFY for ROCm 4.3.0

### Added

* CUDA 11.3.0 support
* cuDNN 8.2.0 support
* LLVM 12.0.0 support

### Resolved issues

* Added missing type casting arguments for `cuStreamWaitValue32(64)` and
  `cuStreamWriteValue32(64)`

## HIPIFY for ROCm 4.2.0

### Added

* CUDA 11.2.2 support
* cuDNN 8.1.1 support
* Initial device API support
* LLVM 11.1.0 support
* New options:
  * `--doc-format=<value>`, with `full` (default), `strict`, and `compact` options

### Changed

* Tests on kernel launch syntax

## HIPIFY for ROCm 4.1.0

### Added

* CUDA 11.2.0 support
* Stream-ordered memory API support
* cuDNN 8.1.1 support
* LLVM 11.0.1 support

### Changed

* Initial support for API versioning

### Resolved issues

* Patches for LLVM 10.0.x and 11.0.0 (Windows and Linux)

## HIPIFY for ROCm 4.0.0

### Added

* LLVM 11.0.0 support

## HIPIFY for ROCm 3.10.0

### Changed

* Revised CUDA and HIP API and data type versioning
* Revised and removed deprecated CUDA and HIP APIs and data types

## HIPIFY for ROCm 3.9.0

### Added

* CUDA 11.0.1 support
* `CUDA2HIP` documentation generation in Markdown and CSV formats
* Versioning support for CUDA and HIP APIs
* New options:
  * `--md` (generate Markdown documentation)
  * `--csv` (generate CSV documentation)

  ### Changed

* Improved `hipify-perl` generation

## HIPIFY for ROCm 3.8.0

### Added

* cuDNN 8.0.2 support
* `compile_commands.json` support (`-p <build-path>`)

### Changed

* Improved `hipify-perl` generation

## HIPIFY for ROCm 3.7.0

### Added

* CUDA 11.0.0 support
* Linux packaging
* LLVM 10.0.1 support

## HIPIFY for ROCm 3.6.0

### Added

* `deprecated` flag for all corresponding CUDA and HIP APIs

### Changed

* Added warning for all deprecated APIs

## HIPIFY for ROCm 3.5.0

### Added

* CUDA 10.2.0 support
* cuDNN 7.6.5 support
* LLVM 10.0.0 support

### Changed

* `hipify-clang` and `clang` options separator (`--`) support
