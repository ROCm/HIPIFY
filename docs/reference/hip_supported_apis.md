<head>
  <meta charset="UTF-8">
  <meta name="description" content="CUDA APIs supported by HIP">
  <meta name="keywords" content="HIPIFY, ROC, ROCm, HIP, CUDA, CUDA2HIP, hipify-clang, hipify-perl">
</head>

# CUDA APIs supported by HIP

|     **CUDA**     | **HIP**                                                           |
|:-----------------|:------------------------------------------------------------------|
| CUDA Runtime API | [HIP API](tables/CUDA_Runtime_API_functions_supported_by_HIP.md)  |
| CUDA Driver API  | [HIP API](tables/CUDA_Driver_API_functions_supported_by_HIP.md)   |
| CUComplex API    | [HIP API](tables/cuComplex_API_supported_by_HIP.md)               |
| CUDA Device API  | [HIP Device API](tables/CUDA_Device_API_supported_by_HIP.md)      |
| CUDA RTC API     | [HIP RTC API](tables/CUDA_RTC_API_supported_by_HIP.md)            |
| CUBLAS API       | [HIP BLAS API](tables/CUBLAS_API_supported_by_HIP.md)             |
| CUSPARSE API     | [HIP SPARSE API](tables/CUSPARSE_API_supported_by_HIP.md)         |
| CUSOLVER API     | [HIP SOLVER API](tables/CUSOLVER_API_supported_by_HIP.md)         |
| CURAND API       | [HIP RAND API](tables/CURAND_API_supported_by_HIP.md)             |
| CUFFT API        | [HIP FFT API](tables/CUFFT_API_supported_by_HIP.md)               |
| CUTENSOR API     | [HIP TENSOR API](tables/CUTENSOR_API_supported_by_HIP.md)         |
| CUB API          | [HIP CUB API](tables/CUB_API_supported_by_HIP.md)                 |

To generate the above documentation with the information about all supported CUDA APIs in Markdown format, run `hipify-clang --md --doc-format=full --doc-roc=separate` with or without specifying the output directory (`-o`).
By running `hipify-clang --csv --doc-format=full --doc-roc=separate`, the documentation will be generated in CSV format.
