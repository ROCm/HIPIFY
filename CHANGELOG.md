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
- INSTALL to `bin` subfolder

## HIPIFY for ROCm 5.5.0
### Added
- CUDA 11.8 support
- Partial CUDA 12.0 support
- cuDNN 8.7.0 support
- LLVM 15.0.7 support
- Initial MIOpem support
- Device types support
- cuBLAS 64bit API (CUDA 12.0) initial support
- rocBLAS and MIOpen synthetic tests

## HIPIFY for ROCm 5.4.0
### Added
- LLVM 15.0.0 support
- hipRTC support
- Error Handling API support
- hipBLAS synthetic tests

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
- LLVM 3.8.0 is out ofsupport
- HIPIFY-specific options support in unit testing
