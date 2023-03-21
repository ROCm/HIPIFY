## <a name="cuda-apis"></a> Supported CUDA APIs

- [Runtime API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md)
- [Driver API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/CUDA_Driver_API_functions_supported_by_HIP.md)
- [cuComplex API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/cuComplex_API_supported_by_HIP.md)
- [Device API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/CUDA_Device_API_supported_by_HIP.md)
- [RTC API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/CUDA_RTC_API_supported_by_HIP.md)
- [cuBLAS](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/CUBLAS_API_supported_by_HIP.md)
- [cuRAND](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/CURAND_API_supported_by_HIP.md)
- [cuDNN](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/CUDNN_API_supported_by_HIP.md)
- [cuFFT](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/CUFFT_API_supported_by_HIP.md)
- [cuSPARSE](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/CUSPARSE_API_supported_by_HIP.md)
- [CUB](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/amd-staging/doc/markdown/CUB_API_supported_by_HIP.md)

To generate the above documentation with the actual information about all supported CUDA APIs in Markdown format, run `hipify-clang --md` with or without specifying the output directory (`-o`).
