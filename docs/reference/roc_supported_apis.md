<head>
  <meta charset="UTF-8">
  <meta name="description" content="CUDA APIs supported by ROC">
  <meta name="keywords" content="HIPIFY, ROC, ROCm, HIP, CUDA, CUDA2HIP, hipify-clang, hipify-perl">
</head>

# CUDA APIs supported by ROC

|     **CUDA**     |                            **ROC**                        |
|:-----------------|:----------------------------------------------------------|
| CUBLAS API       | [ROC BLAS API](tables/CUBLAS_API_supported_by_ROC.md)     |
| CUSPARSE API     | [ROC SPARSE API](tables/CUSPARSE_API_supported_by_ROC.md) |
| CURAND API       | [ROC RAND API](tables/CURAND_API_supported_by_ROC.md)     |
| CUDNN API        | [MIOPEN API](tables/CUDNN_API_supported_by_MIOPEN.md)     |

To generate the above documentation with the information about all supported CUDA APIs in Markdown format, run `hipify-clang --md --doc-format=full --doc-roc=separate` with or without specifying the output directory (`-o`).
By running `hipify-clang --csv --doc-format=full --doc-roc=separate`, the documentation will be generated in CSV format.
