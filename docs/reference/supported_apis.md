<head>
  <meta charset="UTF-8">
  <meta name="description" content="NVIDIA CUDA APIs supported by HIPIFY">

  <meta name="keywords" content="HIPIFY, ROCm, NVIDIA, CUDA, CUDA2HIP, hipify-clang, hipify-perl">
</head>

# Supported NVIDIA CUDA APIs

|     **CUDA**     | **HIP**                                                           |                            **ROC**                        |                            **HIP & ROC**                                |
|:-----------------|:------------------------------------------------------------------|:----------------------------------------------------------|:------------------------------------------------------------------------|
| CUDA Runtime API | [HIP API](tables/CUDA_Runtime_API_functions_supported_by_HIP.md)  |                                                           |                                                                         |
| CUDA Driver API  | [HIP API](tables/CUDA_Driver_API_functions_supported_by_HIP.md)   |                                                           |                                                                         |
| CUComplex API    | [HIP API](tables/cuComplex_API_supported_by_HIP.md)               |                                                           |                                                                         |
| CUDA Device API  | [HIP Device API](tables/CUDA_Device_API_supported_by_HIP.md)      |                                                           |                                                                         |
| CUDA RTC API     | [HIP RTC API](tables/CUDA_RTC_API_supported_by_HIP.md)            |                                                           |                                                                         |
| CUBLAS API       | [HIP BLAS API](tables/CUBLAS_API_supported_by_HIP.md)             | [ROC BLAS API](tables/CUBLAS_API_supported_by_ROC.md)     | [HIP + ROC BLAS API](tables/CUBLAS_API_supported_by_HIP_and_ROC.md)     |
| CUSPARSE API     | [HIP SPARSE API](tables/CUSPARSE_API_supported_by_HIP.md)         | [ROC SPARSE API](tables/CUSPARSE_API_supported_by_ROC.md) | [HIP + ROC SPARSE API](tables/CUSPARSE_API_supported_by_HIP_and_ROC.md) |
| CUSOLVER API     | [HIP SOLVER API](tables/CUSOLVER_API_supported_by_HIP.md)         |                                                           |                                                                         |
| CURAND API       | [HIP RAND API](tables/CURAND_API_supported_by_HIP.md)             | [ROC RAND API](tables/CURAND_API_supported_by_ROC.md)     | [HIP + ROC RAND API](tables/CURAND_API_supported_by_HIP_and_ROC.md)     |
| CUFFT API        | [HIP FFT API](tables/CUFFT_API_supported_by_HIP.md)               |                                                           |                                                                         |
| CUDNN API        | [HIP DNN API](tables/CUDNN_API_supported_by_HIP.md)               | [MIOPEN API](tables/CUDNN_API_supported_by_MIOPEN.md)     | [HIP + MIOPEN API](tables/CUDNN_API_supported_by_HIP_and_MIOPEN.md)     |
| CUTENSOR API     | [HIP TENSOR API](tables/CUTENSOR_API_supported_by_HIP.md)         |                                                           |                                                                         |
| CUB API          | [HIP CUB API](tables/CUB_API_supported_by_HIP.md)                 |                                                           |                                                                         |

To generate the above documentation with the information about all supported CUDA APIs in Markdown format, run `hipify-clang --md --doc-format=full` with or without specifying the output directory (`-o`), for HIP and ROC separately `--doc-roc=separate` or in the joint format (ROC & HIP) `--doc-roc=joint`.
