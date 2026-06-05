<head>
    <meta charset="UTF-8">
    <meta name="description" content="CUDA APIs supported by HIPIFY">
    <meta name="keywords" content="HIPIFY, HIP, ROCm, CUDA, CUDA2HIP, hipification, hipify-clang, hipify-perl, SPARSELt, cuSPARSELt, hipSPARSELt">
</head>

# CUSPARSELT API supported by HIP


**Note\:** In the tables that follow the columns marked `A`, `D`, `C`, `R`, `U`, and `E` mean the following:
**A** - Added; **D** - Deprecated; **C** - Changed; **R** - Removed; **U** - Unsupported for CUDA version(s); **E** - Experimental

## **1. CUSPARSELT Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**U**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
|`CUSPARSELT_MATMUL_ALG_CONFIG_ID`|0.0.1| | | |`HIPSPARSELT_MATMUL_ALG_CONFIG_ID`|7.2.0| | | | | |
|`CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID`|0.0.1| | | |`HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID`|7.2.0| | | | | |
|`CUSPARSELT_MATMUL_ALG_DEFAULT`|0.0.1| | | |`HIPSPARSELT_MATMUL_ALG_DEFAULT`|7.2.0| | | | | |
|`CUSPARSELT_MATMUL_SEARCH_ITERATIONS`|0.0.1| | | |`HIPSPARSELT_MATMUL_SEARCH_ITERATIONS`|7.2.0| | | | | |
|`CUSPARSELT_PRUNE_SPMMA_STRIP`|0.0.1| | | |`HIPSPARSELT_PRUNE_SPMMA_STRIP`|7.2.0| | | | | |
|`CUSPARSELT_PRUNE_SPMMA_TILE`|0.0.1| | | |`HIPSPARSELT_PRUNE_SPMMA_TILE`|7.2.0| | | | | |
|`CUSPARSELT_SPARSITY_50_PERCENT`|0.0.1| | | |`HIPSPARSELT_SPARSITY_50_PERCENT`|7.2.0| | | | | |
|`CUSPARSE_COMPUTE_16F`|0.0.1| | | |`HIPSPARSELT_COMPUTE_16F`|7.2.0| | | | | |
|`CUSPARSE_COMPUTE_32I`|0.0.1| | | |`HIPSPARSELT_COMPUTE_32I`|7.2.0| | | | | |
|`cusparseComputeType`|0.0.1| | | |`hipsparseLtComputetype_t`|7.2.0| | | | | |
|`cusparseLtHandle_t`|0.0.1| | | |`hipsparseLtHandle_t`|7.2.0| | | | | |
|`cusparseLtMatDescriptor_t`|0.0.1| | | |`hipsparseLtMatDescriptor_t`|7.2.0| | | | | |
|`cusparseLtMatmulAlgAttribute_t`|0.0.1| | | |`hipsparseLtMatmulAlgAttribute_t`|7.2.0| | | | | |
|`cusparseLtMatmulAlgSelection_t`|0.0.1| | | |`hipsparseLtMatmulAlgSelection_t`|7.2.0| | | | | |
|`cusparseLtMatmulAlg_t`|0.0.1| | | |`hipsparseLtMatmulAlg_t`|7.2.0| | | | | |
|`cusparseLtMatmulDescriptor_t`|0.0.1| | | |`hipsparseLtMatmulDescriptor_t`|7.2.0| | | | | |
|`cusparseLtMatmulPlan_t`|0.0.1| | | |`hipsparseLtMatmulPlan_t`|7.2.0| | | | | |
|`cusparseLtPruneAlg_t`|0.0.1| | | |`hipsparseLtPruneAlg_t`|7.2.0| | | | | |
|`cusparseLtSparsity_t`|0.0.1| | | |`hipsparseLtSparsity_t`|7.2.0| | | | | |

## **2. CUSPARSELT Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**U**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
|`cusparseLtDenseDescriptorInit`|0.0.1| | | |`hipsparseLtDenseDescriptorInit`|7.2.0| | | | | |
|`cusparseLtDestroy`|0.0.1| | | |`hipsparseLtDestroy`|7.2.0| | | | | |
|`cusparseLtInit`|0.0.1| | | |`hipsparseLtInit`|7.2.0| | | | | |
|`cusparseLtMatmul`|0.0.1| | | |`hipsparseLtMatmul`|7.2.0| | | | | |
|`cusparseLtMatmulAlgGetAttribute`|0.0.1| | | |`hipsparseLtMatmulAlgGetAttribute`|7.2.0| | | | | |
|`cusparseLtMatmulAlgSelectionInit`|0.0.1| | | |`hipsparseLtMatmulAlgSelectionInit`|7.2.0| | | | | |
|`cusparseLtMatmulAlgSetAttribute`|0.0.1| | | |`hipsparseLtMatmulAlgSetAttribute`|7.2.0| | | | | |
|`cusparseLtMatmulDescriptorInit`|0.0.1| | | |`hipsparseLtMatmulDescriptorInit`|7.2.0| | | | | |
|`cusparseLtMatmulGetWorkspace`|0.0.1| | | |`hipsparseLtMatmulGetWorkspace`|7.2.0| | | | | |
|`cusparseLtMatmulPlanDestroy`|0.0.1| | | |`hipsparseLtMatmulPlanDestroy`|7.2.0| | | | | |
|`cusparseLtMatmulPlanInit`|0.0.1| | | |`hipsparseLtMatmulPlanInit`|7.2.0| | | | | |
|`cusparseLtMatmulSearch`|0.0.1| | | |`hipsparseLtMatmulSearch`|7.2.0| | | | | |
|`cusparseLtSpMMACompress`|0.0.1| | | |`hipsparseLtSpMMACompress`|7.2.0| | | | | |
|`cusparseLtSpMMACompressedSize`|0.0.1| | | |`hipsparseLtSpMMACompressedSize`|7.2.0| | | | | |
|`cusparseLtSpMMAPrune`|0.0.1| | | |`hipsparseLtSpMMAPrune`|7.2.0| | | | | |
|`cusparseLtSpMMAPruneCheck`|0.0.1| | | |`hipsparseLtSpMMAPruneCheck`|7.2.0| | | | | |
|`cusparseLtStructuredDescriptorInit`|0.0.1| | | |`hipsparseLtStructuredDescriptorInit`|7.2.0| | | | | |

