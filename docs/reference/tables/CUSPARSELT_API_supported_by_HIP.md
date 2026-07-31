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
|`CUSPARSELT_INVALID_MODE`|0.3.0| | | | | | | | | | |
|`CUSPARSELT_MATMUL_ACTIVATION_GELU`|0.2.0| | | |`HIPSPARSELT_MATMUL_ACTIVATION_GELU`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING`|0.3.0| | | |`HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_ACTIVATION_RELU`|0.2.0| | | |`HIPSPARSELT_MATMUL_ACTIVATION_RELU`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD`|0.2.0| | | |`HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND`|0.2.0| | | |`HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_ALG_CONFIG_ID`|0.0.1| | | |`HIPSPARSELT_MATMUL_ALG_CONFIG_ID`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID`|0.0.1| | | |`HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_ALG_DEFAULT`|0.0.1| | | |`HIPSPARSELT_MATMUL_ALG_DEFAULT`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING`|0.3.0| | | |`HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_BETA_VECTOR_SCALING`|0.3.0| | | |`HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_BIAS_POINTER`|0.2.0| | | |`HIPSPARSELT_MATMUL_BIAS_POINTER`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_BIAS_STRIDE`|0.2.0| | | |`HIPSPARSELT_MATMUL_BIAS_STRIDE`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_SEARCH_ITERATIONS`|0.0.1| | | |`HIPSPARSELT_MATMUL_SEARCH_ITERATIONS`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_SPLIT_K`|0.3.0| | | |`HIPSPARSELT_MATMUL_SPLIT_K`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_SPLIT_K_BUFFERS`|0.3.0| | | |`HIPSPARSELT_MATMUL_SPLIT_K_BUFFERS`|7.10.0| | | | | |
|`CUSPARSELT_MATMUL_SPLIT_K_MODE`|0.3.0| | | |`HIPSPARSELT_MATMUL_SPLIT_K_MODE`|7.10.0| | | | | |
|`CUSPARSELT_MAT_BATCH_STRIDE`|0.2.0| | | |`HIPSPARSELT_MAT_BATCH_STRIDE`|7.10.0| | | | | |
|`CUSPARSELT_MAT_NUM_BATCHES`|0.2.0| | | |`HIPSPARSELT_MAT_NUM_BATCHES`|7.10.0| | | | | |
|`CUSPARSELT_PRUNE_SPMMA_STRIP`|0.0.1| | | |`HIPSPARSELT_PRUNE_SPMMA_STRIP`|7.10.0| | | | | |
|`CUSPARSELT_PRUNE_SPMMA_TILE`|0.0.1| | | |`HIPSPARSELT_PRUNE_SPMMA_TILE`|7.10.0| | | | | |
|`CUSPARSELT_SPARSITY_50_PERCENT`|0.0.1| | | |`HIPSPARSELT_SPARSITY_50_PERCENT`|7.10.0| | | | | |
|`CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL`|0.3.0| | | |`HIPSPARSELT_SPLIT_K_MODE_ONE_KERNEL`|7.10.0| | | | | |
|`CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS`|0.3.0| | | |`HIPSPARSELT_SPLIT_K_MODE_TWO_KERNELS`|7.10.0| | | | | |
|`CUSPARSE_COMPUTE_16F`|0.0.1| | | |`HIPSPARSELT_COMPUTE_16F`|7.10.0| | | | | |
|`CUSPARSE_COMPUTE_32I`|0.0.1| | | |`HIPSPARSELT_COMPUTE_32I`|7.10.0| | | | | |
|`CUSPARSE_COMPUTE_TF32`|0.1.0| | | |`HIPSPARSELT_COMPUTE_TF32`|7.10.0| | | | | |
|`CUSPARSE_COMPUTE_TF32_FAST`|0.1.0| | | |`HIPSPARSELT_COMPUTE_TF32_FAST`|7.10.0| | | | | |
|`cusparseComputeType`|0.0.1| | | |`hipsparseLtComputetype_t`|7.10.0| | | | | |
|`cusparseLtHandle_t`|0.0.1| |0.2.0| |`hipsparseLtHandle_t`|7.10.0| | | | | |
|`cusparseLtMatDescAttribute_t`|0.2.0| | | |`hipsparseLtMatDescAttribute_t`|7.10.0| | | | | |
|`cusparseLtMatDescriptor_t`|0.0.1| |0.2.0| |`hipsparseLtMatDescriptor_t`|7.10.0| | | | | |
|`cusparseLtMatmulAlgAttribute_t`|0.0.1| | | |`hipsparseLtMatmulAlgAttribute_t`|7.10.0| | | | | |
|`cusparseLtMatmulAlgSelection_t`|0.0.1| |0.2.0| |`hipsparseLtMatmulAlgSelection_t`|7.10.0| | | | | |
|`cusparseLtMatmulAlg_t`|0.0.1| | | |`hipsparseLtMatmulAlg_t`|7.10.0| | | | | |
|`cusparseLtMatmulDescAttribute_t`|0.2.0| | | |`hipsparseLtMatmulDescAttribute_t`|7.10.0| | | | | |
|`cusparseLtMatmulDescriptor_t`|0.0.1| |0.2.0| |`hipsparseLtMatmulDescriptor_t`|7.10.0| | | | | |
|`cusparseLtMatmulPlan_t`|0.0.1| |0.2.0| |`hipsparseLtMatmulPlan_t`|7.10.0| | | | | |
|`cusparseLtPruneAlg_t`|0.0.1| | | |`hipsparseLtPruneAlg_t`|7.10.0| | | | | |
|`cusparseLtSparsity_t`|0.0.1| | | |`hipsparseLtSparsity_t`|7.10.0| | | | | |
|`cusparseLtSplitKMode_t`|0.3.0| | | |`hipsparseLtSplitKMode_t`|7.10.0| | | | | |

## **2. CUSPARSELT Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**U**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
|`cusparseLtDenseDescriptorInit`|0.0.1| | | |`hipsparseLtDenseDescriptorInit`|7.10.0| | | | | |
|`cusparseLtDestroy`|0.0.1| | | |`hipsparseLtDestroy`|7.10.0| | | | | |
|`cusparseLtInit`|0.0.1| | | |`hipsparseLtInit`|7.10.0| | | | | |
|`cusparseLtMatDescGetAttribute`|0.2.0| | | |`hipsparseLtMatDescGetAttribute`|7.10.0| | | | | |
|`cusparseLtMatDescSetAttribute`|0.2.0| | | |`hipsparseLtMatDescSetAttribute`|7.10.0| | | | | |
|`cusparseLtMatDescriptorDestroy`|0.1.0| | | |`hipsparseLtMatDescriptorDestroy`|7.10.0| | | | | |
|`cusparseLtMatmul`|0.0.1| | | |`hipsparseLtMatmul`|7.10.0| | | | | |
|`cusparseLtMatmulAlgGetAttribute`|0.0.1| | | |`hipsparseLtMatmulAlgGetAttribute`|7.10.0| | | | | |
|`cusparseLtMatmulAlgSelectionInit`|0.0.1| | | |`hipsparseLtMatmulAlgSelectionInit`|7.10.0| | | | | |
|`cusparseLtMatmulAlgSetAttribute`|0.0.1| | | |`hipsparseLtMatmulAlgSetAttribute`|7.10.0| | | | | |
|`cusparseLtMatmulDescGetAttribute`|0.2.0| | | |`hipsparseLtMatmulDescGetAttribute`|7.10.0| | | | | |
|`cusparseLtMatmulDescSetAttribute`|0.2.0| | | |`hipsparseLtMatmulDescSetAttribute`|7.10.0| | | | | |
|`cusparseLtMatmulDescriptorInit`|0.0.1| | | |`hipsparseLtMatmulDescriptorInit`|7.10.0| | | | | |
|`cusparseLtMatmulGetWorkspace`|0.0.1| |0.3.0| |`hipsparseLtMatmulGetWorkspace`|7.10.0| | | | | |
|`cusparseLtMatmulPlanDestroy`|0.0.1| | | |`hipsparseLtMatmulPlanDestroy`|7.10.0| | | | | |
|`cusparseLtMatmulPlanInit`|0.0.1| | | |`hipsparseLtMatmulPlanInit`|7.10.0| | | | | |
|`cusparseLtMatmulSearch`|0.0.1| | | |`hipsparseLtMatmulSearch`|7.10.0| | | | | |
|`cusparseLtSpMMACompress`|0.0.1| | | |`hipsparseLtSpMMACompress`|7.10.0| | | | | |
|`cusparseLtSpMMACompress2`|0.1.0| | | |`hipsparseLtSpMMACompress2`|7.10.0| | | | | |
|`cusparseLtSpMMACompressedSize`|0.0.1| | | |`hipsparseLtSpMMACompressedSize`|7.10.0| | | | | |
|`cusparseLtSpMMACompressedSize2`|0.1.0| | | |`hipsparseLtSpMMACompressedSize2`|7.10.0| | | | | |
|`cusparseLtSpMMAPrune`|0.0.1| | | |`hipsparseLtSpMMAPrune`|7.10.0| | | | | |
|`cusparseLtSpMMAPrune2`|0.1.0| | | |`hipsparseLtSpMMAPrune2`|7.10.0| | | | | |
|`cusparseLtSpMMAPruneCheck`|0.0.1| | | |`hipsparseLtSpMMAPruneCheck`|7.10.0| | | | | |
|`cusparseLtSpMMAPruneCheck2`|0.1.0| | | |`hipsparseLtSpMMAPruneCheck2`|7.10.0| | | | | |
|`cusparseLtStructuredDescriptorInit`|0.0.1| | | |`hipsparseLtStructuredDescriptorInit`|7.10.0| | | | | |

