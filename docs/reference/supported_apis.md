<head>
  <meta charset="UTF-8">
  <meta name="description" content="Supported CUDA APIs">
  <meta name="keywords" content="HIPIFY, ROC, ROCm, HIP, CUDA, CUDA2HIP, hipify-clang, hipify-perl">
</head>

# Supported CUDA APIs

|     **CUDA**     | **HIP**                                                           |                            **ROC**                        |                            **HIP & ROC**                                |
|:-----------------|:------------------------------------------------------------------|:----------------------------------------------------------|:------------------------------------------------------------------------|
| CUDA Runtime API | [HIP API](tables/CUDA_Runtime_API_functions_supported_by_HIP.md)  |                                                           |                                                                         |
| CUDA Driver API  | [HIP API](tables/CUDA_Driver_API_functions_supported_by_HIP.md)   |                                                           |                                                                         |
| CUComplex API    | [HIP API](tables/cuComplex_API_supported_by_HIP.md)               |                                                           |                                                                         |
| CUDA Device API  | [HIP Device API](tables/CUDA_Device_API_supported_by_HIP.md)      |                                                           |                                                                         |
| CUDA RTC API     | [HIP RTC API](tables/CUDA_RTC_API_supported_by_HIP.md)            |                                                           |                                                                         |
| CUFile API*      | [HIP FILE API](tables/cuFile_API_supported_by_HIP.md)             |                                                           |                                                                         |
| CUBLAS API       | [HIP BLAS API](tables/CUBLAS_API_supported_by_HIP.md)             | [ROC BLAS API](tables/CUBLAS_API_supported_by_ROC.md)     | [HIP + ROC BLAS API](tables/CUBLAS_API_supported_by_HIP_and_ROC.md)     |
| CUSPARSE API     | [HIP SPARSE API](tables/CUSPARSE_API_supported_by_HIP.md)         | [ROC SPARSE API](tables/CUSPARSE_API_supported_by_ROC.md) | [HIP + ROC SPARSE API](tables/CUSPARSE_API_supported_by_HIP_and_ROC.md) |
| CUSOLVER API     | [HIP SOLVER API](tables/CUSOLVER_API_supported_by_HIP.md)         |                                                           |                                                                         |
| CURAND API       | [HIP RAND API](tables/CURAND_API_supported_by_HIP.md)             | [ROC RAND API](tables/CURAND_API_supported_by_ROC.md)     | [HIP + ROC RAND API](tables/CURAND_API_supported_by_HIP_and_ROC.md)     |
| CUFFT API        | [HIP FFT API](tables/CUFFT_API_supported_by_HIP.md)               |                                                           |                                                                         |
| CUDNN API        |                                                                   | [MIOPEN API](tables/CUDNN_API_supported_by_MIOPEN.md)     |                                                                         |
| CUTENSOR API     | [HIP TENSOR API](tables/CUTENSOR_API_supported_by_HIP.md)         |                                                           |                                                                         |
| CUB API          | [HIP CUB API](tables/CUB_API_supported_by_HIP.md)                 |                                                           |                                                                         |

* The CUFile API and HIP FILE API are supported only on Linux.

To generate the above documentation with the information about all supported CUDA APIs in Markdown format, run `hipify-clang --md --doc-format=full --doc-roc=joint` with or without specifying the output directory (`-o`).
By running `hipify-clang --csv --doc-format=full --doc-roc=joint`, the documentation will be generated in CSV format.
