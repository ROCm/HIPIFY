<head>
  <meta charset="UTF-8">
  <meta name="description" content="NVIDIA CUDA APIs supported by HIPIFY">

  <meta name="keywords" content="HIPIFY, ROCm, NVIDIA, CUDA, CUDA2HIP, hipify-clang, hipify-perl">
</head>

# Supported NVIDIA CUDA APIs

|     **CUDA**     |                            **ROC**                        |
|:-----------------|:----------------------------------------------------------|
| CUBLAS API       | [ROC BLAS API](tables/CUBLAS_API_supported_by_ROC.md)     |
| CUSPARSE API     | [ROC SPARSE API](tables/CUSPARSE_API_supported_by_ROC.md) |
| CURAND API       | [ROC RAND API](tables/CURAND_API_supported_by_ROC.md)     |
| CUDNN API        | [MIOPEN API](tables/CUDNN_API_supported_by_MIOPEN.md)     |

To generate the above documentation with the information about all supported CUDA APIs in Markdown format, run `hipify-clang --md --doc-format=full` with or without specifying the output directory (`-o`), for HIP and ROC separately `--doc-roc=separate` or in the joint format (HIP & ROC) `--doc-roc=joint`.
