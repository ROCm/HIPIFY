# Supported CUDA APIs

|      **CUDA**   | **HIP**                                                              |   **ROC**      |        **HIP & ROC**        |
|:----------------|:---------------------------------------------------------------------|:---------------|:----------------------------|
| Runtime API     | [HIP API](tables/CUDA_Runtime_API_functions_supported_by_HIP.md)     |       |        |
| Driver API      | [HIP API](tables/CUDA_Driver_API_functions_supported_by_HIP.md)      |       |        |
| Complex API     | [HIP API](tables/cuComplex_API_supported_by_HIP.md)                  |       |        |
| Device API      | [HIP Device API](tables/CUDA_Device_API_supported_by_HIP.md)         |       |        |
| RTC API         | [HIP RTC API](tables/CUDA_RTC_API_supported_by_HIP.md)               |       |        |
| BLAS API        | [HIP BLAS API](tables/CUBLAS_API_supported_by_HIP.md)                | [ROC BLAS API](tables/CUBLAS_API_supported_by_ROC.md)     | [HIP + ROC BLAS API](tables/CUBLAS_API_supported_by_HIP_and_ROC.md)     |
| SPARSE API      | [HIP SPARSE API](tables/CUSPARSE_API_supported_by_HIP.md)            | [ROC SPARSE API](tables/CUSPARSE_API_supported_by_ROC.md) | [HIP + ROC SPARSE API](tables/CUSPARSE_API_supported_by_HIP_and_ROC.md) |
| SOLVER API      | [HIP SOLVER API](tables/CUSOLVER_API_supported_by_HIP.md)            |       |        |
| RAND API        | [HIP RAND API](tables/CURAND_API_supported_by_HIP.md)                |       |        |
| FFT API         | [HIP FFT API](tables/CUFFT_API_supported_by_HIP.md)                  |       |        |
| DNN API         | [HIP DNN API](tables/CUDNN_API_supported_by_HIP.md)                  |       |        |
| CUB API         | [HIP CUB API](tables/CUB_API_supported_by_HIP.md)                    |       |        |

To generate the above documentation with the actual information about all supported CUDA APIs in Markdown format, run `hipify-clang --md` with or without specifying the output directory (`-o`).
