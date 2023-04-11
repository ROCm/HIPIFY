# Change Log for HIPIFY

Full documentation for HIPIFY is available at [hipify.readthedocs.io](https://hipify.readthedocs.io/en/latest/).

## HIPIFY for ROCm 5.6.0
### Added
- CUDA 12.1 support
- cuDNN 8.8.1 support
- LLVM 16.0.0 support
- New options:
  - --default-preprocessor (synonymous with '--skip-excluded-preprocessor-conditional-blocks')
  - --no-undocumented-features
  - --no-warnings-on-undocumented-features
  - --versions
### Fixed
- Accessing half2 struct members (undocumented feature)
- INSTALL to 'bin' subfolder

## HIPIFY for ROCm 5.5.0
### Added
- Partial CUDA 12.0 support
- cuDNN 8.7.0 support
- Initial MIOpem support
- cuBLAS 64bit API (CUDA 12.0) initial support
- rocBLAS and MIOpen synthetic tests
- LLVM 15.0.7 support
### Misc
- Synthetic unit tests for cuBLAS2rocBLAS and cuDNN2MIOpen

## HIPIFY for ROCm 5.4.1
### Added
- CUDA 11.8 support
- cuDNN 8.6.0 support
- Device types support
- LLVM 15.0.4 support
### Fixed
- [Linux] Get rid of any RPATH definitions

## HIPIFY for ROCm 5.4.0
### Added
- hipRTC support
- Error Handling API support
- hipBLAS synthetic tests
- LLVM 15.0.0 support

## HIPIFY for ROCm 5.3.0
### Added
- CUDA 11.7 support
- cuDNN 8.4.1 support
- CUB initial support
- More synthetic tests
- New options:
  - --hip-kernel-execution-syntax
### Fixed
- Patches for LLVM 14.0.x (Windows only)
- Add GNUInstallDirs for CMake on Linux
### Misc
- LLVM 3.8.0 is out of support
- HIPIFY-specific options support in unit testing

## HIPIFY for ROCm 5.2.0
### Added
- CUDA 11.6 support
- cuDNN 8.3.3 support
- LLVM 14.0.0 support

## HIPIFY for ROCm 5.1.0
### Added
- CUDA 11.5 support
- cuDNN 8.3.2 support
### Fixed
- hipification of cuOccupancyMaxPotentialBlockSize and cuOccupancyMaxPotentialBlockSizeWithFlags

## HIPIFY for ROCm 5.0.0
### Added
- CUDA 11.4 support
- cuDNN 8.3.2 support
- Initial hipRTC support
- GNU C/C++ 11.2 support
- Visual Studio 2022 support
- LLVM 13.0.0 support
- New options:
  - --experimental
  - --cuda-kernel-execution-syntax
### Fixed
- Packaging for Debian and RPM Linuxes
- Undo args typecasting for 4 Driver API functions cuStreamWaitValue32(64), cuStreamWriteValue32(64) as those args in the corresponding HIP functions become uint finally
### Misc
- Support for different formats of the generated documentation
- Experimentally supported APIs

## HIPIFY for ROCm 4.5.0
### Added
- cuDNN 8.2.4 support
- Initial graph API support
- GNU C/C++ 11.1 support
- LLVM 12.0.1 support
### Fixed
- Abandon HIP_DYNAMIC_SHARED
### Misc
- Synthetic unit tests
- -std=c++14 by default

## HIPIFY for ROCm 4.3.0
### Added
- CUDA 11.3 support
- cuDNN 8.2.0 support
- LLVM 12.0.0 support
### Fixed
- Added the missing type casting of arguments for the following functions:
  cuStreamWaitValue32(64), cuStreamWriteValue32(64)

## HIPIFY for ROCm 4.2.0
### Added
- CUDA 11.2.2 support
- cuDNN 8.1.1 support
- Initial Device API support
- LLVM 11.1.0 support
- New options:
  - --doc-format=<value> - Documentation format: 'full' (default), 'strict', or 'compact'
### Misc
- Tests on kernel launch syntax

## HIPIFY for ROCm 4.1.0
### Added
- CUDA 11.2.0 support
- Stream Ordered Memory API support
- cuDNN 8.1.1 support
- LLVM 11.0.1 support
### Fixed
- Patches for LLVM 10.0.x and 11.0.0 (both Windows and Linux)
### Misc
- Initial support for APIs versioning

## HIPIFY for ROCm 4.0.0
- No changes since 3.10.0
