<head>
  <meta charset="UTF-8">
  <meta name="description" content="CUDA APIs supported by HIP and ROC">
  <meta name="keywords" content="HIPIFY, ROC, ROCm, HIP, CUDA, CUDA2HIP, hipify-clang, hipify-perl">
</head>

# CUDA APIs supported by HIP and ROC

|     **CUDA**     |                            **HIP & ROC**                                |
|:-----------------|:------------------------------------------------------------------------|
| CUBLAS API       | [HIP + ROC BLAS API](tables/CUBLAS_API_supported_by_HIP_and_ROC.md)     |
| CUSPARSE API     | [HIP + ROC SPARSE API](tables/CUSPARSE_API_supported_by_HIP_and_ROC.md) |
| CURAND API       | [HIP + ROC RAND API](tables/CURAND_API_supported_by_HIP_and_ROC.md)     |

To generate the above documentation with the information about all supported CUDA APIs in Markdown format, run `hipify-clang --md --doc-format=full --doc-roc=joint` with or without specifying the output directory (`-o`).
By running `hipify-clang --csv --doc-format=full --doc-roc=joint`, the documentation will be generated in CSV format.
