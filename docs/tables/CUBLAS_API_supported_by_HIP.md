# CUBLAS API supported by HIP

## **2. CUBLAS Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUBLAS_ATOMICS_ALLOWED`| | | | |`HIPBLAS_ATOMICS_ALLOWED`|3.10.0| | | | |
|`CUBLAS_ATOMICS_NOT_ALLOWED`| | | | |`HIPBLAS_ATOMICS_NOT_ALLOWED`|3.10.0| | | | |
|`CUBLAS_COMPUTE_16F`|11.0| | | |`HIPBLAS_COMPUTE_16F`|6.0.0| | | | |
|`CUBLAS_COMPUTE_16F_PEDANTIC`|11.0| | | |`HIPBLAS_COMPUTE_16F_PEDANTIC`|6.0.0| | | | |
|`CUBLAS_COMPUTE_32F`|11.0| | | |`HIPBLAS_COMPUTE_32F`|6.0.0| | | | |
|`CUBLAS_COMPUTE_32F_FAST_16BF`|11.0| | | |`HIPBLAS_COMPUTE_32F_FAST_16BF`|6.0.0| | | | |
|`CUBLAS_COMPUTE_32F_FAST_16F`|11.0| | | |`HIPBLAS_COMPUTE_32F_FAST_16F`|6.0.0| | | | |
|`CUBLAS_COMPUTE_32F_FAST_TF32`|11.0| | | |`HIPBLAS_COMPUTE_32F_FAST_TF32`|6.0.0| | | | |
|`CUBLAS_COMPUTE_32F_PEDANTIC`|11.0| | | |`HIPBLAS_COMPUTE_32F_PEDANTIC`|6.0.0| | | | |
|`CUBLAS_COMPUTE_32I`|11.0| | | |`HIPBLAS_COMPUTE_32I`|6.0.0| | | | |
|`CUBLAS_COMPUTE_32I_PEDANTIC`|11.0| | | |`HIPBLAS_COMPUTE_32I_PEDANTIC`|6.0.0| | | | |
|`CUBLAS_COMPUTE_64F`|11.0| | | |`HIPBLAS_COMPUTE_64F`|6.0.0| | | | |
|`CUBLAS_COMPUTE_64F_PEDANTIC`|11.0| | | |`HIPBLAS_COMPUTE_64F_PEDANTIC`|6.0.0| | | | |
|`CUBLAS_DEFAULT_MATH`|9.0| | | |`HIPBLAS_DEFAULT_MATH`|6.1.0| | | |6.1.0|
|`CUBLAS_DIAG_NON_UNIT`| | | | |`HIPBLAS_DIAG_NON_UNIT`|1.8.2| | | | |
|`CUBLAS_DIAG_UNIT`| | | | |`HIPBLAS_DIAG_UNIT`|1.8.2| | | | |
|`CUBLAS_FILL_MODE_FULL`|10.1| | | |`HIPBLAS_FILL_MODE_FULL`|1.8.2| | | | |
|`CUBLAS_FILL_MODE_LOWER`| | | | |`HIPBLAS_FILL_MODE_LOWER`|1.8.2| | | | |
|`CUBLAS_FILL_MODE_UPPER`| | | | |`HIPBLAS_FILL_MODE_UPPER`|1.8.2| | | | |
|`CUBLAS_GEMM_ALGO0`|8.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO0_TENSOR_OP`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO1`|8.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO10`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO10_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO11`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO11_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO12`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO12_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO13`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO13_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO14`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO14_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO15`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO15_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO16`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO17`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO18`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO19`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO1_TENSOR_OP`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO2`|8.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO20`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO21`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO22`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO23`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO2_TENSOR_OP`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO3`|8.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO3_TENSOR_OP`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO4`|8.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO4_TENSOR_OP`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO5`|8.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO5_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO6`|8.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO6_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO7`|8.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO7_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO8`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO8_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_ALGO9`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_ALGO9_TENSOR_OP`|9.2| | | | | | | | | |
|`CUBLAS_GEMM_DEFAULT`|9.0| | | |`HIPBLAS_GEMM_DEFAULT`|1.8.2| | | | |
|`CUBLAS_GEMM_DEFAULT_TENSOR_OP`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_DFALT`|8.0| | | |`HIPBLAS_GEMM_DEFAULT`|1.8.2| | | | |
|`CUBLAS_GEMM_DFALT_TENSOR_OP`|9.0| | | | | | | | | |
|`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`|11.0| | | |`HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`|6.1.0| | | |6.1.0|
|`CUBLAS_OP_C`| | | | |`HIPBLAS_OP_C`|1.8.2| | | | |
|`CUBLAS_OP_CONJG`|10.1| | | | | | | | | |
|`CUBLAS_OP_HERMITAN`|10.1| | | |`HIPBLAS_OP_C`|1.8.2| | | | |
|`CUBLAS_OP_N`| | | | |`HIPBLAS_OP_N`|1.8.2| | | | |
|`CUBLAS_OP_T`| | | | |`HIPBLAS_OP_T`|1.8.2| | | | |
|`CUBLAS_PEDANTIC_MATH`|11.0| | | |`HIPBLAS_PEDANTIC_MATH`|6.1.0| | | |6.1.0|
|`CUBLAS_POINTER_MODE_DEVICE`| | | | |`HIPBLAS_POINTER_MODE_DEVICE`|1.8.2| | | | |
|`CUBLAS_POINTER_MODE_HOST`| | | | |`HIPBLAS_POINTER_MODE_HOST`|1.8.2| | | | |
|`CUBLAS_SIDE_LEFT`| | | | |`HIPBLAS_SIDE_LEFT`|1.8.2| | | | |
|`CUBLAS_SIDE_RIGHT`| | | | |`HIPBLAS_SIDE_RIGHT`|1.8.2| | | | |
|`CUBLAS_STATUS_ALLOC_FAILED`| | | | |`HIPBLAS_STATUS_ALLOC_FAILED`|1.8.2| | | | |
|`CUBLAS_STATUS_ARCH_MISMATCH`| | | | |`HIPBLAS_STATUS_ARCH_MISMATCH`|1.8.2| | | | |
|`CUBLAS_STATUS_EXECUTION_FAILED`| | | | |`HIPBLAS_STATUS_EXECUTION_FAILED`|1.8.2| | | | |
|`CUBLAS_STATUS_INTERNAL_ERROR`| | | | |`HIPBLAS_STATUS_INTERNAL_ERROR`|1.8.2| | | | |
|`CUBLAS_STATUS_INVALID_VALUE`| | | | |`HIPBLAS_STATUS_INVALID_VALUE`|1.8.2| | | | |
|`CUBLAS_STATUS_LICENSE_ERROR`| | | | |`HIPBLAS_STATUS_UNKNOWN`| | | | | |
|`CUBLAS_STATUS_MAPPING_ERROR`| | | | |`HIPBLAS_STATUS_MAPPING_ERROR`|1.8.2| | | | |
|`CUBLAS_STATUS_NOT_INITIALIZED`| | | | |`HIPBLAS_STATUS_NOT_INITIALIZED`|1.8.2| | | | |
|`CUBLAS_STATUS_NOT_SUPPORTED`| | | | |`HIPBLAS_STATUS_NOT_SUPPORTED`|1.8.2| | | | |
|`CUBLAS_STATUS_SUCCESS`| | | | |`HIPBLAS_STATUS_SUCCESS`|1.8.2| | | | |
|`CUBLAS_TENSOR_OP_MATH`|9.0|11.0| | |`HIPBLAS_TENSOR_OP_MATH`|6.1.0| | | |6.1.0|
|`CUBLAS_TF32_TENSOR_OP_MATH`|11.0| | | |`HIPBLAS_TF32_TENSOR_OP_MATH`|6.1.0| | | |6.1.0|
|`cublasAtomicsMode_t`| | | | |`hipblasAtomicsMode_t`|3.10.0| | | | |
|`cublasComputeType_t`|11.0| | | |`hipblasComputeType_t`|6.0.0| | | | |
|`cublasContext`| | | | | | | | | | |
|`cublasDiagType_t`| | | | |`hipblasDiagType_t`|1.8.2| | | | |
|`cublasFillMode_t`| | | | |`hipblasFillMode_t`|1.8.2| | | | |
|`cublasGemmAlgo_t`|8.0| | | |`hipblasGemmAlgo_t`|1.8.2| | | | |
|`cublasHandle_t`| | | | |`hipblasHandle_t`|3.0.0| | | | |
|`cublasMath_t`|9.0| | | |`hipblasMath_t`|6.1.0| | | |6.1.0|
|`cublasOperation_t`| | | | |`hipblasOperation_t`|1.8.2| | | | |
|`cublasPointerMode_t`| | | | |`hipblasPointerMode_t`|1.8.2| | | | |
|`cublasSideMode_t`| | | | |`hipblasSideMode_t`|1.8.2| | | | |
|`cublasStatus`| | | | |`hipblasStatus_t`|1.8.2| | | | |
|`cublasStatus_t`| | | | |`hipblasStatus_t`|1.8.2| | | | |

## **3. CUDA Library Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUDA_C_16BF`| | | | |`HIP_C_16BF`|5.7.0| | | | |
|`CUDA_C_16F`|8.0| | | |`HIP_C_16F`|5.7.0| | | | |
|`CUDA_C_16I`|11.0| | | | | | | | | |
|`CUDA_C_16U`|11.0| | | | | | | | | |
|`CUDA_C_32F`|8.0| | | |`HIP_C_32F`|5.7.0| | | | |
|`CUDA_C_32I`|8.0| | | |`HIP_C_32I`|5.7.0| | | | |
|`CUDA_C_32U`|8.0| | | |`HIP_C_32U`|5.7.0| | | | |
|`CUDA_C_4I`|11.0| | | | | | | | | |
|`CUDA_C_4U`|11.0| | | | | | | | | |
|`CUDA_C_64F`|8.0| | | |`HIP_C_64F`|5.7.0| | | | |
|`CUDA_C_64I`|11.0| | | | | | | | | |
|`CUDA_C_64U`|11.0| | | | | | | | | |
|`CUDA_C_8I`|8.0| | | |`HIP_C_8I`|5.7.0| | | | |
|`CUDA_C_8U`|8.0| | | |`HIP_C_8U`|5.7.0| | | | |
|`CUDA_R_16BF`| | | | |`HIP_R_16BF`|5.7.0| | | | |
|`CUDA_R_16F`|8.0| | | |`HIP_R_16F`|5.7.0| | | | |
|`CUDA_R_16I`|11.0| | | | | | | | | |
|`CUDA_R_16U`|11.0| | | | | | | | | |
|`CUDA_R_32F`|8.0| | | |`HIP_R_32F`|5.7.0| | | | |
|`CUDA_R_32I`|8.0| | | |`HIP_R_32I`|5.7.0| | | | |
|`CUDA_R_32U`|8.0| | | |`HIP_R_32U`|5.7.0| | | | |
|`CUDA_R_4I`|11.0| | | | | | | | | |
|`CUDA_R_4U`|11.0| | | | | | | | | |
|`CUDA_R_64F`|8.0| | | |`HIP_R_64F`|5.7.0| | | | |
|`CUDA_R_64I`|11.0| | | | | | | | | |
|`CUDA_R_64U`|11.0| | | | | | | | | |
|`CUDA_R_8F_E4M3`|11.8| | | | | | | | | |
|`CUDA_R_8F_E5M2`|11.8| | | | | | | | | |
|`CUDA_R_8I`|8.0| | | |`HIP_R_8I`|5.7.0| | | | |
|`CUDA_R_8U`|8.0| | | |`HIP_R_8U`|5.7.0| | | | |
|`cublasDataType_t`|7.5| | | |`hipDataType`|5.7.0| | | | |
|`cudaDataType`|8.0| | | |`hipDataType`|5.7.0| | | | |
|`cudaDataType_t`|8.0| | | |`hipDataType`|5.7.0| | | | |

## **4. CUBLAS Helper Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasAlloc`| | | | | | | | | | |
|`cublasCreate`| | | | |`hipblasCreate`|1.8.2| | | | |
|`cublasCreate_v2`| | | | |`hipblasCreate`|1.8.2| | | | |
|`cublasDestroy`| | | | |`hipblasDestroy`|1.8.2| | | | |
|`cublasDestroy_v2`| | | | |`hipblasDestroy`|1.8.2| | | | |
|`cublasFree`| | | | | | | | | | |
|`cublasGetAtomicsMode`| | | | |`hipblasGetAtomicsMode`|3.10.0| | | | |
|`cublasGetCudartVersion`|10.1| | | | | | | | | |
|`cublasGetError`| | | | | | | | | | |
|`cublasGetLoggerCallback`|9.2| | | | | | | | | |
|`cublasGetMathMode`|9.0| | | |`hipblasGetMathMode`|6.1.0| | | |6.1.0|
|`cublasGetMatrix`| | | | |`hipblasGetMatrix`|1.8.2| | | | |
|`cublasGetMatrixAsync`| | | | |`hipblasGetMatrixAsync`|3.7.0| | | | |
|`cublasGetMatrixAsync_64`|12.0| | | | | | | | | |
|`cublasGetMatrix_64`|12.0| | | | | | | | | |
|`cublasGetPointerMode`| | | | |`hipblasGetPointerMode`|1.8.2| | | | |
|`cublasGetPointerMode_v2`| | | | |`hipblasGetPointerMode`|1.8.2| | | | |
|`cublasGetProperty`| | | | | | | | | | |
|`cublasGetSmCountTarget`|11.3| | | | | | | | | |
|`cublasGetStatusName`|11.4| | | | | | | | | |
|`cublasGetStatusString`|11.4| | | | | | | | | |
|`cublasGetStream`| | | | |`hipblasGetStream`|1.8.2| | | | |
|`cublasGetStream_v2`| | | | |`hipblasGetStream`|1.8.2| | | | |
|`cublasGetVector`| | | | |`hipblasGetVector`|1.8.2| | | | |
|`cublasGetVectorAsync`| | | | |`hipblasGetVectorAsync`|3.7.0| | | | |
|`cublasGetVectorAsync_64`|12.0| | | | | | | | | |
|`cublasGetVector_64`|12.0| | | | | | | | | |
|`cublasGetVersion`| | | | | | | | | | |
|`cublasGetVersion_v2`| | | | | | | | | | |
|`cublasInit`| | | | | | | | | | |
|`cublasLogCallback`|9.2| | | | | | | | | |
|`cublasLoggerConfigure`|9.2| | | | | | | | | |
|`cublasMigrateComputeType`|11.0| | | | | | | | | |
|`cublasSetAtomicsMode`| | | | |`hipblasSetAtomicsMode`|3.10.0| | | | |
|`cublasSetKernelStream`| | | | | | | | | | |
|`cublasSetLoggerCallback`|9.2| | | | | | | | | |
|`cublasSetMathMode`|9.0| | | |`hipblasSetMathMode`|6.1.0| | | |6.1.0|
|`cublasSetMatrix`| | | | |`hipblasSetMatrix`|1.8.2| | | | |
|`cublasSetMatrixAsync`| | | | |`hipblasSetMatrixAsync`|3.7.0| | | | |
|`cublasSetMatrixAsync_64`|12.0| | | | | | | | | |
|`cublasSetMatrix_64`|12.0| | | | | | | | | |
|`cublasSetPointerMode`| | | | |`hipblasSetPointerMode`|1.8.2| | | | |
|`cublasSetPointerMode_v2`| | | | |`hipblasSetPointerMode`|1.8.2| | | | |
|`cublasSetSmCountTarget`|11.3| | | | | | | | | |
|`cublasSetStream`| | | | |`hipblasSetStream`|1.8.2| | | | |
|`cublasSetStream_v2`| | | | |`hipblasSetStream`|1.8.2| | | | |
|`cublasSetVector`| | | | |`hipblasSetVector`|1.8.2| | | | |
|`cublasSetVectorAsync`| | | | |`hipblasSetVectorAsync`|3.7.0| | | | |
|`cublasSetVectorAsync_64`|12.0| | | | | | | | | |
|`cublasSetVector_64`|12.0| | | | | | | | | |
|`cublasShutdown`| | | | | | | | | | |
|`cublasXerbla`| | | | | | | | | | |

## **5. CUBLAS Level-1 Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasCaxpy`| | | | |`hipblasCaxpy_v2`|6.0.0| | | | |
|`cublasCaxpy_64`|12.0| | | |`hipblasCaxpy_v2_64`|6.1.0| | | |6.1.0|
|`cublasCaxpy_v2`| | | | |`hipblasCaxpy_v2`|6.0.0| | | | |
|`cublasCaxpy_v2_64`|12.0| | | |`hipblasCaxpy_v2_64`|6.1.0| | | |6.1.0|
|`cublasCcopy`| | | | |`hipblasCcopy_v2`|6.0.0| | | | |
|`cublasCcopy_64`|12.0| | | |`hipblasCcopy_v2_64`|6.1.0| | | |6.1.0|
|`cublasCcopy_v2`| | | | |`hipblasCcopy_v2`|6.0.0| | | | |
|`cublasCcopy_v2_64`|12.0| | | |`hipblasCcopy_v2_64`|6.1.0| | | |6.1.0|
|`cublasCdotc`| | | | |`hipblasCdotc_v2`|6.0.0| | | | |
|`cublasCdotc_64`|12.0| | | |`hipblasCdotc_v2_64`|6.1.0| | | |6.1.0|
|`cublasCdotc_v2`| | | | |`hipblasCdotc_v2`|6.0.0| | | | |
|`cublasCdotc_v2_64`|12.0| | | |`hipblasCdotc_v2_64`|6.1.0| | | |6.1.0|
|`cublasCdotu`| | | | |`hipblasCdotu_v2`|6.0.0| | | | |
|`cublasCdotu_64`|12.0| | | |`hipblasCdotu_v2_64`|6.1.0| | | |6.1.0|
|`cublasCdotu_v2`| | | | |`hipblasCdotu_v2`|6.0.0| | | | |
|`cublasCdotu_v2_64`|12.0| | | |`hipblasCdotu_v2_64`|6.1.0| | | |6.1.0|
|`cublasCrot`| | | | |`hipblasCrot_v2`|6.0.0| | | | |
|`cublasCrot_64`|12.0| | | |`hipblasCrot_v2_64`|6.1.0| | | |6.1.0|
|`cublasCrot_v2`| | | | |`hipblasCrot_v2`|6.0.0| | | | |
|`cublasCrot_v2_64`|12.0| | | |`hipblasCrot_v2_64`|6.1.0| | | |6.1.0|
|`cublasCrotg`| | | | |`hipblasCrotg_v2`|6.0.0| | | | |
|`cublasCrotg_v2`| | | | |`hipblasCrotg_v2`|6.0.0| | | | |
|`cublasCscal`| | | | |`hipblasCscal_v2`|6.0.0| | | | |
|`cublasCscal_64`|12.0| | | |`hipblasCscal_v2_64`|6.1.0| | | |6.1.0|
|`cublasCscal_v2`| | | | |`hipblasCscal_v2`|6.0.0| | | | |
|`cublasCscal_v2_64`|12.0| | | |`hipblasCscal_v2_64`|6.1.0| | | |6.1.0|
|`cublasCsrot`| | | | |`hipblasCsrot_v2`|6.0.0| | | | |
|`cublasCsrot_64`|12.0| | | |`hipblasCsrot_v2_64`|6.1.0| | | |6.1.0|
|`cublasCsrot_v2`| | | | |`hipblasCsrot_v2`|6.0.0| | | | |
|`cublasCsrot_v2_64`|12.0| | | |`hipblasCsrot_v2_64`|6.1.0| | | |6.1.0|
|`cublasCsscal`| | | | |`hipblasCsscal_v2`|6.0.0| | | | |
|`cublasCsscal_64`|12.0| | | |`hipblasCsscal_v2_64`|6.1.0| | | |6.1.0|
|`cublasCsscal_v2`| | | | |`hipblasCsscal_v2`|6.0.0| | | | |
|`cublasCsscal_v2_64`|12.0| | | |`hipblasCsscal_v2_64`|6.1.0| | | |6.1.0|
|`cublasCswap`| | | | |`hipblasCswap_v2`|6.0.0| | | | |
|`cublasCswap_64`|12.0| | | |`hipblasCswap_v2_64`|6.1.0| | | |6.1.0|
|`cublasCswap_v2`| | | | |`hipblasCswap_v2`|6.0.0| | | | |
|`cublasCswap_v2_64`|12.0| | | |`hipblasCswap_v2_64`|6.1.0| | | |6.1.0|
|`cublasDasum`| | | | |`hipblasDasum`|1.8.2| | | | |
|`cublasDasum_64`|12.0| | | |`hipblasDasum_64`|6.1.0| | | |6.1.0|
|`cublasDasum_v2`| | | | |`hipblasDasum`|1.8.2| | | | |
|`cublasDasum_v2_64`|12.0| | | |`hipblasDasum_64`|6.1.0| | | |6.1.0|
|`cublasDaxpy`| | | | |`hipblasDaxpy`|1.8.2| | | | |
|`cublasDaxpy_64`|12.0| | | |`hipblasDaxpy_64`|6.1.0| | | |6.1.0|
|`cublasDaxpy_v2`| | | | |`hipblasDaxpy`|1.8.2| | | | |
|`cublasDaxpy_v2_64`|12.0| | | |`hipblasDaxpy_64`|6.1.0| | | |6.1.0|
|`cublasDcopy`| | | | |`hipblasDcopy`|1.8.2| | | | |
|`cublasDcopy_64`|12.0| | | |`hipblasDcopy_64`|6.1.0| | | |6.1.0|
|`cublasDcopy_v2`| | | | |`hipblasDcopy`|1.8.2| | | | |
|`cublasDcopy_v2_64`|12.0| | | |`hipblasDcopy_64`|6.1.0| | | |6.1.0|
|`cublasDdot`| | | | |`hipblasDdot`|3.0.0| | | | |
|`cublasDdot_64`|12.0| | | |`hipblasDdot_64`|6.1.0| | | |6.1.0|
|`cublasDdot_v2`| | | | |`hipblasDdot`|3.0.0| | | | |
|`cublasDdot_v2_64`|12.0| | | |`hipblasDdot_64`|6.1.0| | | |6.1.0|
|`cublasDnrm2`| | | | |`hipblasDnrm2`|1.8.2| | | | |
|`cublasDnrm2_64`|12.0| | | |`hipblasDnrm2_64`|6.1.0| | | |6.1.0|
|`cublasDnrm2_v2`| | | | |`hipblasDnrm2`|1.8.2| | | | |
|`cublasDnrm2_v2_64`|12.0| | | |`hipblasDnrm2_64`|6.1.0| | | |6.1.0|
|`cublasDrot`| | | | |`hipblasDrot`|3.0.0| | | | |
|`cublasDrot_64`|12.0| | | |`hipblasDrot_64`|6.1.0| | | |6.1.0|
|`cublasDrot_v2`| | | | |`hipblasDrot`|3.0.0| | | | |
|`cublasDrot_v2_64`|12.0| | | |`hipblasDrot_64`|6.1.0| | | |6.1.0|
|`cublasDrotg`| | | | |`hipblasDrotg`|3.0.0| | | | |
|`cublasDrotg_v2`| | | | |`hipblasDrotg`|3.0.0| | | | |
|`cublasDrotm`| | | | |`hipblasDrotm`|3.0.0| | | | |
|`cublasDrotm_64`|12.0| | | |`hipblasDrotm_64`|6.1.0| | | |6.1.0|
|`cublasDrotm_v2`| | | | |`hipblasDrotm`|3.0.0| | | | |
|`cublasDrotm_v2_64`|12.0| | | |`hipblasDrotm_64`|6.1.0| | | |6.1.0|
|`cublasDrotmg`| | | | |`hipblasDrotmg`|3.0.0| | | | |
|`cublasDrotmg_v2`| | | | |`hipblasDrotmg`|3.0.0| | | | |
|`cublasDscal`| | | | |`hipblasDscal`|1.8.2| | | | |
|`cublasDscal_64`|12.0| | | |`hipblasDscal_64`|6.1.0| | | |6.1.0|
|`cublasDscal_v2`| | | | |`hipblasDscal`|1.8.2| | | | |
|`cublasDscal_v2_64`|12.0| | | |`hipblasDscal_64`|6.1.0| | | |6.1.0|
|`cublasDswap`| | | | |`hipblasDswap`|3.0.0| | | | |
|`cublasDswap_64`|12.0| | | |`hipblasDswap_64`|6.1.0| | | |6.1.0|
|`cublasDswap_v2`| | | | |`hipblasDswap`|3.0.0| | | | |
|`cublasDswap_v2_64`|12.0| | | |`hipblasDswap_64`|6.1.0| | | |6.1.0|
|`cublasDzasum`| | | | |`hipblasDzasum_v2`|6.0.0| | | | |
|`cublasDzasum_64`|12.0| | | |`hipblasDzasum_v2_64`|6.1.0| | | |6.1.0|
|`cublasDzasum_v2`| | | | |`hipblasDzasum_v2`|6.0.0| | | | |
|`cublasDzasum_v2_64`|12.0| | | |`hipblasDzasum_v2_64`|6.1.0| | | |6.1.0|
|`cublasDznrm2`| | | | |`hipblasDznrm2_v2`|6.0.0| | | | |
|`cublasDznrm2_64`|12.0| | | |`hipblasDznrm2_v2_64`|6.1.0| | | |6.1.0|
|`cublasDznrm2_v2`| | | | |`hipblasDznrm2_v2`|6.0.0| | | | |
|`cublasDznrm2_v2_64`|12.0| | | |`hipblasDznrm2_v2_64`|6.1.0| | | |6.1.0|
|`cublasIcamax`| | | | |`hipblasIcamax_v2`|6.0.0| | | | |
|`cublasIcamax_64`|12.0| | | |`hipblasIcamax_v2_64`|6.1.0| | | |6.1.0|
|`cublasIcamax_v2`| | | | |`hipblasIcamax_v2`|6.0.0| | | | |
|`cublasIcamax_v2_64`|12.0| | | |`hipblasIcamax_v2_64`|6.1.0| | | |6.1.0|
|`cublasIcamin`| | | | |`hipblasIcamin_v2`|6.0.0| | | | |
|`cublasIcamin_64`|12.0| | | |`hipblasIcamin_v2_64`|6.1.0| | | |6.1.0|
|`cublasIcamin_v2`| | | | |`hipblasIcamin_v2`|6.0.0| | | | |
|`cublasIcamin_v2_64`|12.0| | | |`hipblasIcamin_v2_64`|6.1.0| | | |6.1.0|
|`cublasIdamax`| | | | |`hipblasIdamax`|1.8.2| | | | |
|`cublasIdamax_64`|12.0| | | |`hipblasIdamax_64`|6.1.0| | | |6.1.0|
|`cublasIdamax_v2`| | | | |`hipblasIdamax`|1.8.2| | | | |
|`cublasIdamax_v2_64`|12.0| | | |`hipblasIdamax_64`|6.1.0| | | |6.1.0|
|`cublasIdamin`| | | | |`hipblasIdamin`|3.0.0| | | | |
|`cublasIdamin_64`|12.0| | | |`hipblasIdamin_64`|6.1.0| | | |6.1.0|
|`cublasIdamin_v2`| | | | |`hipblasIdamin`|3.0.0| | | | |
|`cublasIdamin_v2_64`|12.0| | | |`hipblasIdamin_64`|6.1.0| | | |6.1.0|
|`cublasIsamax`| | | | |`hipblasIsamax`|1.8.2| | | | |
|`cublasIsamax_64`|12.0| | | |`hipblasIsamax_64`|6.1.0| | | |6.1.0|
|`cublasIsamax_v2`| | | | |`hipblasIsamax`|1.8.2| | | | |
|`cublasIsamax_v2_64`|12.0| | | |`hipblasIsamax_64`|6.1.0| | | |6.1.0|
|`cublasIsamin`| | | | |`hipblasIsamin`|3.0.0| | | | |
|`cublasIsamin_64`|12.0| | | |`hipblasIsamin_64`|6.1.0| | | |6.1.0|
|`cublasIsamin_v2`| | | | |`hipblasIsamin`|3.0.0| | | | |
|`cublasIsamin_v2_64`|12.0| | | |`hipblasIsamin_64`|6.1.0| | | |6.1.0|
|`cublasIzamax`| | | | |`hipblasIzamax_v2`|6.0.0| | | | |
|`cublasIzamax_64`|12.0| | | |`hipblasIzamax_v2_64`|6.1.0| | | |6.1.0|
|`cublasIzamax_v2`| | | | |`hipblasIzamax_v2`|6.0.0| | | | |
|`cublasIzamax_v2_64`|12.0| | | |`hipblasIzamax_v2_64`|6.1.0| | | |6.1.0|
|`cublasIzamin`| | | | |`hipblasIzamin_v2`|6.0.0| | | | |
|`cublasIzamin_64`|12.0| | | |`hipblasIzamin_v2_64`|6.1.0| | | |6.1.0|
|`cublasIzamin_v2`| | | | |`hipblasIzamin_v2`|6.0.0| | | | |
|`cublasIzamin_v2_64`|12.0| | | |`hipblasIzamin_v2_64`|6.1.0| | | |6.1.0|
|`cublasNrm2Ex`|8.0| | | |`hipblasNrm2Ex_v2`|6.0.0| | | | |
|`cublasNrm2Ex_64`|12.0| | | | | | | | | |
|`cublasSasum`| | | | |`hipblasSasum`|1.8.2| | | | |
|`cublasSasum_64`|12.0| | | |`hipblasSasum_64`|6.1.0| | | |6.1.0|
|`cublasSasum_v2`| | | | |`hipblasSasum`|1.8.2| | | | |
|`cublasSasum_v2_64`|12.0| | | |`hipblasSasum_64`|6.1.0| | | |6.1.0|
|`cublasSaxpy`| | | | |`hipblasSaxpy`|1.8.2| | | | |
|`cublasSaxpy_64`|12.0| | | |`hipblasSaxpy_64`|6.1.0| | | |6.1.0|
|`cublasSaxpy_v2`| | | | |`hipblasSaxpy`|1.8.2| | | | |
|`cublasSaxpy_v2_64`|12.0| | | |`hipblasSaxpy_64`|6.1.0| | | |6.1.0|
|`cublasScasum`| | | | |`hipblasScasum_v2`|6.0.0| | | | |
|`cublasScasum_64`|12.0| | | |`hipblasScasum_v2_64`|6.1.0| | | |6.1.0|
|`cublasScasum_v2`| | | | |`hipblasScasum_v2`|6.0.0| | | | |
|`cublasScasum_v2_64`|12.0| | | |`hipblasScasum_v2_64`|6.1.0| | | |6.1.0|
|`cublasScnrm2`| | | | |`hipblasScnrm2_v2`|6.0.0| | | | |
|`cublasScnrm2_64`|12.0| | | |`hipblasScnrm2_v2_64`|6.1.0| | | |6.1.0|
|`cublasScnrm2_v2`| | | | |`hipblasScnrm2_v2`|6.0.0| | | | |
|`cublasScnrm2_v2_64`|12.0| | | |`hipblasScnrm2_v2_64`|6.1.0| | | |6.1.0|
|`cublasScopy`| | | | |`hipblasScopy`|1.8.2| | | | |
|`cublasScopy_64`|12.0| | | |`hipblasScopy_64`|6.1.0| | | |6.1.0|
|`cublasScopy_v2`| | | | |`hipblasScopy`|1.8.2| | | | |
|`cublasScopy_v2_64`|12.0| | | |`hipblasScopy_64`|6.1.0| | | |6.1.0|
|`cublasSdot`| | | | |`hipblasSdot`|3.0.0| | | | |
|`cublasSdot_64`|12.0| | | |`hipblasSdot_64`|6.1.0| | | |6.1.0|
|`cublasSdot_v2`| | | | |`hipblasSdot`|3.0.0| | | | |
|`cublasSdot_v2_64`|12.0| | | |`hipblasSdot_64`|6.1.0| | | |6.1.0|
|`cublasSnrm2`| | | | |`hipblasSnrm2`|1.8.2| | | | |
|`cublasSnrm2_64`|12.0| | | |`hipblasSnrm2_64`|6.1.0| | | |6.1.0|
|`cublasSnrm2_v2`| | | | |`hipblasSnrm2`|1.8.2| | | | |
|`cublasSnrm2_v2_64`|12.0| | | |`hipblasSnrm2_64`|6.1.0| | | |6.1.0|
|`cublasSrot`| | | | |`hipblasSrot`|3.0.0| | | | |
|`cublasSrot_64`|12.0| | | |`hipblasSrot_64`|6.1.0| | | |6.1.0|
|`cublasSrot_v2`| | | | |`hipblasSrot`|3.0.0| | | | |
|`cublasSrot_v2_64`|12.0| | | |`hipblasSrot_64`|6.1.0| | | |6.1.0|
|`cublasSrotg`| | | | |`hipblasSrotg`|3.0.0| | | | |
|`cublasSrotg_v2`| | | | |`hipblasSrotg`|3.0.0| | | | |
|`cublasSrotm`| | | | |`hipblasSrotm`|3.0.0| | | | |
|`cublasSrotm_64`|12.0| | | |`hipblasSrotm_64`|6.1.0| | | |6.1.0|
|`cublasSrotm_v2`| | | | |`hipblasSrotm`|3.0.0| | | | |
|`cublasSrotm_v2_64`|12.0| | | |`hipblasSrotm_64`|6.1.0| | | |6.1.0|
|`cublasSrotmg`| | | | |`hipblasSrotmg`|3.0.0| | | | |
|`cublasSrotmg_v2`| | | | |`hipblasSrotmg`|3.0.0| | | | |
|`cublasSscal`| | | | |`hipblasSscal`|1.8.2| | | | |
|`cublasSscal_64`|12.0| | | |`hipblasSscal_64`|6.1.0| | | |6.1.0|
|`cublasSscal_v2`| | | | |`hipblasSscal`|1.8.2| | | | |
|`cublasSscal_v2_64`|12.0| | | |`hipblasSscal_64`|6.1.0| | | |6.1.0|
|`cublasSswap`| | | | |`hipblasSswap`|3.0.0| | | | |
|`cublasSswap_64`|12.0| | | |`hipblasSswap_64`|6.1.0| | | |6.1.0|
|`cublasSswap_v2`| | | | |`hipblasSswap`|3.0.0| | | | |
|`cublasSswap_v2_64`|12.0| | | |`hipblasSswap_64`|6.1.0| | | |6.1.0|
|`cublasZaxpy`| | | | |`hipblasZaxpy_v2`|6.0.0| | | | |
|`cublasZaxpy_64`|12.0| | | |`hipblasZaxpy_v2_64`|6.1.0| | | |6.1.0|
|`cublasZaxpy_v2`| | | | |`hipblasZaxpy_v2`|6.0.0| | | | |
|`cublasZaxpy_v2_64`|12.0| | | |`hipblasZaxpy_v2_64`|6.1.0| | | |6.1.0|
|`cublasZcopy`| | | | |`hipblasZcopy_v2`|6.0.0| | | | |
|`cublasZcopy_64`|12.0| | | |`hipblasZcopy_v2_64`|6.1.0| | | |6.1.0|
|`cublasZcopy_v2`| | | | |`hipblasZcopy_v2`|6.0.0| | | | |
|`cublasZcopy_v2_64`|12.0| | | |`hipblasZcopy_v2_64`|6.1.0| | | |6.1.0|
|`cublasZdotc`| | | | |`hipblasZdotc_v2`|6.0.0| | | | |
|`cublasZdotc_64`|12.0| | | |`hipblasZdotc_v2_64`|6.1.0| | | |6.1.0|
|`cublasZdotc_v2`| | | | |`hipblasZdotc_v2`|6.0.0| | | | |
|`cublasZdotc_v2_64`|12.0| | | |`hipblasZdotc_v2_64`|6.1.0| | | |6.1.0|
|`cublasZdotu`| | | | |`hipblasZdotu_v2`|6.0.0| | | | |
|`cublasZdotu_64`|12.0| | | |`hipblasZdotu_v2_64`|6.1.0| | | |6.1.0|
|`cublasZdotu_v2`| | | | |`hipblasZdotu_v2`|6.0.0| | | | |
|`cublasZdotu_v2_64`|12.0| | | |`hipblasZdotu_v2_64`|6.1.0| | | |6.1.0|
|`cublasZdrot`| | | | |`hipblasZdrot_v2`|6.0.0| | | | |
|`cublasZdrot_64`|12.0| | | |`hipblasZdrot_v2_64`|6.1.0| | | |6.1.0|
|`cublasZdrot_v2`| | | | |`hipblasZdrot_v2`|6.0.0| | | | |
|`cublasZdrot_v2_64`|12.0| | | |`hipblasZdrot_v2_64`|6.1.0| | | |6.1.0|
|`cublasZdscal`| | | | |`hipblasZdscal_v2`|6.0.0| | | | |
|`cublasZdscal_64`|12.0| | | |`hipblasZdscal_v2_64`|6.1.0| | | |6.1.0|
|`cublasZdscal_v2`| | | | |`hipblasZdscal_v2`|6.0.0| | | | |
|`cublasZdscal_v2_64`|12.0| | | |`hipblasZdscal_v2_64`|6.1.0| | | |6.1.0|
|`cublasZrot`| | | | |`hipblasZrot_v2`|6.0.0| | | | |
|`cublasZrot_64`|12.0| | | |`hipblasZrot_v2_64`|6.1.0| | | |6.1.0|
|`cublasZrot_v2`| | | | |`hipblasZrot_v2`|6.0.0| | | | |
|`cublasZrot_v2_64`|12.0| | | |`hipblasZrot_v2_64`|6.1.0| | | |6.1.0|
|`cublasZrotg`| | | | |`hipblasZrotg_v2`|6.0.0| | | | |
|`cublasZrotg_v2`| | | | |`hipblasZrotg_v2`|6.0.0| | | | |
|`cublasZscal`| | | | |`hipblasZscal_v2`|6.0.0| | | | |
|`cublasZscal_64`|12.0| | | |`hipblasZscal_v2_64`|6.1.0| | | |6.1.0|
|`cublasZscal_v2`| | | | |`hipblasZscal_v2`|6.0.0| | | | |
|`cublasZscal_v2_64`|12.0| | | |`hipblasZscal_v2_64`|6.1.0| | | |6.1.0|
|`cublasZswap`| | | | |`hipblasZswap_v2`|6.0.0| | | | |
|`cublasZswap_64`|12.0| | | |`hipblasZswap_v2_64`|6.1.0| | | |6.1.0|
|`cublasZswap_v2`| | | | |`hipblasZswap_v2`|6.0.0| | | | |
|`cublasZswap_v2_64`|12.0| | | |`hipblasZswap_v2_64`|6.1.0| | | |6.1.0|

## **6. CUBLAS Level-2 Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasCgbmv`| | | | |`hipblasCgbmv_v2`|6.0.0| | | | |
|`cublasCgbmv_64`|12.0| | | | | | | | | |
|`cublasCgbmv_v2`| | | | |`hipblasCgbmv_v2`|6.0.0| | | | |
|`cublasCgbmv_v2_64`|12.0| | | | | | | | | |
|`cublasCgemv`| | | | |`hipblasCgemv_v2`|6.0.0| | | | |
|`cublasCgemv_64`|12.0| | | | | | | | | |
|`cublasCgemv_v2`| | | | |`hipblasCgemv_v2`|6.0.0| | | | |
|`cublasCgemv_v2_64`|12.0| | | | | | | | | |
|`cublasCgerc`| | | | |`hipblasCgerc_v2`|6.0.0| | | | |
|`cublasCgerc_64`|12.0| | | | | | | | | |
|`cublasCgerc_v2`| | | | |`hipblasCgerc_v2`|6.0.0| | | | |
|`cublasCgerc_v2_64`|12.0| | | | | | | | | |
|`cublasCgeru`| | | | |`hipblasCgeru_v2`|6.0.0| | | | |
|`cublasCgeru_64`|12.0| | | | | | | | | |
|`cublasCgeru_v2`| | | | |`hipblasCgeru_v2`|6.0.0| | | | |
|`cublasCgeru_v2_64`|12.0| | | | | | | | | |
|`cublasChbmv`| | | | |`hipblasChbmv_v2`|6.0.0| | | | |
|`cublasChbmv_64`|12.0| | | | | | | | | |
|`cublasChbmv_v2`| | | | |`hipblasChbmv_v2`|6.0.0| | | | |
|`cublasChbmv_v2_64`|12.0| | | | | | | | | |
|`cublasChemv`| | | | |`hipblasChemv_v2`|6.0.0| | | | |
|`cublasChemv_64`|12.0| | | | | | | | | |
|`cublasChemv_v2`| | | | |`hipblasChemv_v2`|6.0.0| | | | |
|`cublasChemv_v2_64`|12.0| | | | | | | | | |
|`cublasCher`| | | | |`hipblasCher_v2`|6.0.0| | | | |
|`cublasCher2`| | | | |`hipblasCher2_v2`|6.0.0| | | | |
|`cublasCher2_64`|12.0| | | | | | | | | |
|`cublasCher2_v2`| | | | |`hipblasCher2_v2`|6.0.0| | | | |
|`cublasCher2_v2_64`|12.0| | | | | | | | | |
|`cublasCher_64`|12.0| | | | | | | | | |
|`cublasCher_v2`| | | | |`hipblasCher_v2`|6.0.0| | | | |
|`cublasCher_v2_64`|12.0| | | | | | | | | |
|`cublasChpmv`| | | | |`hipblasChpmv_v2`|6.0.0| | | | |
|`cublasChpmv_64`|12.0| | | | | | | | | |
|`cublasChpmv_v2`| | | | |`hipblasChpmv_v2`|6.0.0| | | | |
|`cublasChpmv_v2_64`|12.0| | | | | | | | | |
|`cublasChpr`| | | | |`hipblasChpr_v2`|6.0.0| | | | |
|`cublasChpr2`| | | | |`hipblasChpr2_v2`|6.0.0| | | | |
|`cublasChpr2_64`|12.0| | | | | | | | | |
|`cublasChpr2_v2`| | | | |`hipblasChpr2_v2`|6.0.0| | | | |
|`cublasChpr2_v2_64`|12.0| | | | | | | | | |
|`cublasChpr_64`|12.0| | | | | | | | | |
|`cublasChpr_v2`| | | | |`hipblasChpr_v2`|6.0.0| | | | |
|`cublasChpr_v2_64`|12.0| | | | | | | | | |
|`cublasCsymv`| | | | |`hipblasCsymv_v2`|6.0.0| | | | |
|`cublasCsymv_64`|12.0| | | | | | | | | |
|`cublasCsymv_v2`| | | | |`hipblasCsymv_v2`|6.0.0| | | | |
|`cublasCsymv_v2_64`|12.0| | | | | | | | | |
|`cublasCsyr`| | | | |`hipblasCsyr_v2`|6.0.0| | | | |
|`cublasCsyr2`| | | | |`hipblasCsyr2_v2`|6.0.0| | | | |
|`cublasCsyr2_64`|12.0| | | | | | | | | |
|`cublasCsyr2_v2`| | | | |`hipblasCsyr2_v2`|6.0.0| | | | |
|`cublasCsyr2_v2_64`|12.0| | | | | | | | | |
|`cublasCsyr_64`|12.0| | | | | | | | | |
|`cublasCsyr_v2`| | | | |`hipblasCsyr_v2`|6.0.0| | | | |
|`cublasCsyr_v2_64`|12.0| | | | | | | | | |
|`cublasCtbmv`| | | | |`hipblasCtbmv_v2`|6.0.0| | | | |
|`cublasCtbmv_64`|12.0| | | | | | | | | |
|`cublasCtbmv_v2`| | | | |`hipblasCtbmv_v2`|6.0.0| | | | |
|`cublasCtbmv_v2_64`|12.0| | | | | | | | | |
|`cublasCtbsv`| | | | |`hipblasCtbsv_v2`|6.0.0| | | | |
|`cublasCtbsv_64`|12.0| | | | | | | | | |
|`cublasCtbsv_v2`| | | | |`hipblasCtbsv_v2`|6.0.0| | | | |
|`cublasCtbsv_v2_64`|12.0| | | | | | | | | |
|`cublasCtpmv`| | | | |`hipblasCtpmv_v2`|6.0.0| | | | |
|`cublasCtpmv_64`|12.0| | | | | | | | | |
|`cublasCtpmv_v2`| | | | |`hipblasCtpmv_v2`|6.0.0| | | | |
|`cublasCtpmv_v2_64`|12.0| | | | | | | | | |
|`cublasCtpsv`| | | | |`hipblasCtpsv_v2`|6.0.0| | | | |
|`cublasCtpsv_64`|12.0| | | | | | | | | |
|`cublasCtpsv_v2`| | | | |`hipblasCtpsv_v2`|6.0.0| | | | |
|`cublasCtpsv_v2_64`|12.0| | | | | | | | | |
|`cublasCtrmv`| | | | |`hipblasCtrmv_v2`|6.0.0| | | | |
|`cublasCtrmv_64`|12.0| | | | | | | | | |
|`cublasCtrmv_v2`| | | | |`hipblasCtrmv_v2`|6.0.0| | | | |
|`cublasCtrmv_v2_64`|12.0| | | | | | | | | |
|`cublasCtrsv`| | | | |`hipblasCtrsv_v2`|6.0.0| | | | |
|`cublasCtrsv_64`|12.0| | | | | | | | | |
|`cublasCtrsv_v2`| | | | |`hipblasCtrsv_v2`|6.0.0| | | | |
|`cublasCtrsv_v2_64`|12.0| | | | | | | | | |
|`cublasDgbmv`| | | | |`hipblasDgbmv`|3.5.0| | | | |
|`cublasDgbmv_64`|12.0| | | | | | | | | |
|`cublasDgbmv_v2`| | | | |`hipblasDgbmv`|3.5.0| | | | |
|`cublasDgbmv_v2_64`|12.0| | | | | | | | | |
|`cublasDgemv`| | | | |`hipblasDgemv`|1.8.2| | | | |
|`cublasDgemv_64`|12.0| | | | | | | | | |
|`cublasDgemv_v2`| | | | |`hipblasDgemv`|1.8.2| | | | |
|`cublasDgemv_v2_64`|12.0| | | | | | | | | |
|`cublasDger`| | | | |`hipblasDger`|1.8.2| | | | |
|`cublasDger_64`|12.0| | | | | | | | | |
|`cublasDger_v2`| | | | |`hipblasDger`|1.8.2| | | | |
|`cublasDger_v2_64`|12.0| | | | | | | | | |
|`cublasDsbmv`| | | | |`hipblasDsbmv`|3.5.0| | | | |
|`cublasDsbmv_64`|12.0| | | | | | | | | |
|`cublasDsbmv_v2`| | | | |`hipblasDsbmv`|3.5.0| | | | |
|`cublasDsbmv_v2_64`|12.0| | | | | | | | | |
|`cublasDspmv`| | | | |`hipblasDspmv`|3.5.0| | | | |
|`cublasDspmv_64`|12.0| | | | | | | | | |
|`cublasDspmv_v2`| | | | |`hipblasDspmv`|3.5.0| | | | |
|`cublasDspmv_v2_64`|12.0| | | | | | | | | |
|`cublasDspr`| | | | |`hipblasDspr`|3.5.0| | | | |
|`cublasDspr2`| | | | |`hipblasDspr2`|3.5.0| | | | |
|`cublasDspr2_64`|12.0| | | | | | | | | |
|`cublasDspr2_v2`| | | | |`hipblasDspr2`|3.5.0| | | | |
|`cublasDspr2_v2_64`|12.0| | | | | | | | | |
|`cublasDspr_64`|12.0| | | | | | | | | |
|`cublasDspr_v2`| | | | |`hipblasDspr`|3.5.0| | | | |
|`cublasDspr_v2_64`|12.0| | | | | | | | | |
|`cublasDsymv`| | | | |`hipblasDsymv`|3.5.0| | | | |
|`cublasDsymv_64`|12.0| | | | | | | | | |
|`cublasDsymv_v2`| | | | |`hipblasDsymv`|3.5.0| | | | |
|`cublasDsymv_v2_64`|12.0| | | | | | | | | |
|`cublasDsyr`| | | | |`hipblasDsyr`|3.0.0| | | | |
|`cublasDsyr2`| | | | |`hipblasDsyr2`|3.5.0| | | | |
|`cublasDsyr2_64`|12.0| | | | | | | | | |
|`cublasDsyr2_v2`| | | | |`hipblasDsyr2`|3.5.0| | | | |
|`cublasDsyr2_v2_64`|12.0| | | | | | | | | |
|`cublasDsyr_64`|12.0| | | | | | | | | |
|`cublasDsyr_v2`| | | | |`hipblasDsyr`|3.0.0| | | | |
|`cublasDsyr_v2_64`|12.0| | | | | | | | | |
|`cublasDtbmv`| | | | |`hipblasDtbmv`|3.5.0| | | | |
|`cublasDtbmv_64`|12.0| | | | | | | | | |
|`cublasDtbmv_v2`| | | | |`hipblasDtbmv`|3.5.0| | | | |
|`cublasDtbmv_v2_64`|12.0| | | | | | | | | |
|`cublasDtbsv`| | | | |`hipblasDtbsv`|3.6.0| | | | |
|`cublasDtbsv_64`|12.0| | | | | | | | | |
|`cublasDtbsv_v2`| | | | |`hipblasDtbsv`|3.6.0| | | | |
|`cublasDtbsv_v2_64`|12.0| | | | | | | | | |
|`cublasDtpmv`| | | | |`hipblasDtpmv`|3.5.0| | | | |
|`cublasDtpmv_64`|12.0| | | | | | | | | |
|`cublasDtpmv_v2`| | | | |`hipblasDtpmv`|3.5.0| | | | |
|`cublasDtpmv_v2_64`|12.0| | | | | | | | | |
|`cublasDtpsv`| | | | |`hipblasDtpsv`|3.5.0| | | | |
|`cublasDtpsv_64`|12.0| | | | | | | | | |
|`cublasDtpsv_v2`| | | | |`hipblasDtpsv`|3.5.0| | | | |
|`cublasDtpsv_v2_64`|12.0| | | | | | | | | |
|`cublasDtrmv`| | | | |`hipblasDtrmv`|3.5.0| | | | |
|`cublasDtrmv_64`|12.0| | | | | | | | | |
|`cublasDtrmv_v2`| | | | |`hipblasDtrmv`|3.5.0| | | | |
|`cublasDtrmv_v2_64`|12.0| | | | | | | | | |
|`cublasDtrsv`| | | | |`hipblasDtrsv`|3.0.0| | | | |
|`cublasDtrsv_64`|12.0| | | | | | | | | |
|`cublasDtrsv_v2`| | | | |`hipblasDtrsv`|3.0.0| | | | |
|`cublasDtrsv_v2_64`|12.0| | | | | | | | | |
|`cublasSgbmv`| | | | |`hipblasSgbmv`|3.5.0| | | | |
|`cublasSgbmv_64`|12.0| | | | | | | | | |
|`cublasSgbmv_v2`| | | | |`hipblasSgbmv`|3.5.0| | | | |
|`cublasSgbmv_v2_64`|12.0| | | | | | | | | |
|`cublasSgemv`| | | | |`hipblasSgemv`|1.8.2| | | | |
|`cublasSgemv_64`|12.0| | | | | | | | | |
|`cublasSgemv_v2`| | | | |`hipblasSgemv`|1.8.2| | | | |
|`cublasSgemv_v2_64`|12.0| | | | | | | | | |
|`cublasSger`| | | | |`hipblasSger`|1.8.2| | | | |
|`cublasSger_64`|12.0| | | | | | | | | |
|`cublasSger_v2`| | | | |`hipblasSger`|1.8.2| | | | |
|`cublasSger_v2_64`|12.0| | | | | | | | | |
|`cublasSsbmv`| | | | |`hipblasSsbmv`|3.5.0| | | | |
|`cublasSsbmv_64`|12.0| | | | | | | | | |
|`cublasSsbmv_v2`| | | | |`hipblasSsbmv`|3.5.0| | | | |
|`cublasSsbmv_v2_64`|12.0| | | | | | | | | |
|`cublasSspmv`| | | | |`hipblasSspmv`|3.5.0| | | | |
|`cublasSspmv_64`|12.0| | | | | | | | | |
|`cublasSspmv_v2`| | | | |`hipblasSspmv`|3.5.0| | | | |
|`cublasSspmv_v2_64`|12.0| | | | | | | | | |
|`cublasSspr`| | | | |`hipblasSspr`|3.5.0| | | | |
|`cublasSspr2`| | | | |`hipblasSspr2`|3.5.0| | | | |
|`cublasSspr2_64`|12.0| | | | | | | | | |
|`cublasSspr2_v2`| | | | |`hipblasSspr2`|3.5.0| | | | |
|`cublasSspr2_v2_64`|12.0| | | | | | | | | |
|`cublasSspr_64`|12.0| | | | | | | | | |
|`cublasSspr_v2`| | | | |`hipblasSspr`|3.5.0| | | | |
|`cublasSspr_v2_64`|12.0| | | | | | | | | |
|`cublasSsymv`| | | | |`hipblasSsymv`|3.5.0| | | | |
|`cublasSsymv_64`|12.0| | | | | | | | | |
|`cublasSsymv_v2`| | | | |`hipblasSsymv`|3.5.0| | | | |
|`cublasSsymv_v2_64`|12.0| | | | | | | | | |
|`cublasSsyr`| | | | |`hipblasSsyr`|3.0.0| | | | |
|`cublasSsyr2`| | | | |`hipblasSsyr2`|3.5.0| | | | |
|`cublasSsyr2_64`|12.0| | | | | | | | | |
|`cublasSsyr2_v2`| | | | |`hipblasSsyr2`|3.5.0| | | | |
|`cublasSsyr2_v2_64`|12.0| | | | | | | | | |
|`cublasSsyr_64`|12.0| | | | | | | | | |
|`cublasSsyr_v2`| | | | |`hipblasSsyr`|3.0.0| | | | |
|`cublasSsyr_v2_64`|12.0| | | | | | | | | |
|`cublasStbmv`| | | | |`hipblasStbmv`|3.5.0| | | | |
|`cublasStbmv_64`|12.0| | | | | | | | | |
|`cublasStbmv_v2`| | | | |`hipblasStbmv`|3.5.0| | | | |
|`cublasStbmv_v2_64`|12.0| | | | | | | | | |
|`cublasStbsv`| | | | |`hipblasStbsv`|3.6.0| | | | |
|`cublasStbsv_64`|12.0| | | | | | | | | |
|`cublasStbsv_v2`| | | | |`hipblasStbsv`|3.6.0| | | | |
|`cublasStbsv_v2_64`|12.0| | | | | | | | | |
|`cublasStpmv`| | | | |`hipblasStpmv`|3.5.0| | | | |
|`cublasStpmv_64`|12.0| | | | | | | | | |
|`cublasStpmv_v2`| | | | |`hipblasStpmv`|3.5.0| | | | |
|`cublasStpmv_v2_64`|12.0| | | | | | | | | |
|`cublasStpsv`| | | | |`hipblasStpsv`|3.5.0| | | | |
|`cublasStpsv_64`|12.0| | | | | | | | | |
|`cublasStpsv_v2`| | | | |`hipblasStpsv`|3.5.0| | | | |
|`cublasStpsv_v2_64`|12.0| | | | | | | | | |
|`cublasStrmv`| | | | |`hipblasStrmv`|3.5.0| | | | |
|`cublasStrmv_64`|12.0| | | | | | | | | |
|`cublasStrmv_v2`| | | | |`hipblasStrmv`|3.5.0| | | | |
|`cublasStrmv_v2_64`|12.0| | | | | | | | | |
|`cublasStrsv`| | | | |`hipblasStrsv`|3.0.0| | | | |
|`cublasStrsv_64`|12.0| | | | | | | | | |
|`cublasStrsv_v2`| | | | |`hipblasStrsv`|3.0.0| | | | |
|`cublasStrsv_v2_64`|12.0| | | | | | | | | |
|`cublasZgbmv`| | | | |`hipblasZgbmv_v2`|6.0.0| | | | |
|`cublasZgbmv_64`|12.0| | | | | | | | | |
|`cublasZgbmv_v2`| | | | |`hipblasZgbmv_v2`|6.0.0| | | | |
|`cublasZgbmv_v2_64`|12.0| | | | | | | | | |
|`cublasZgemv`| | | | |`hipblasZgemv_v2`|6.0.0| | | | |
|`cublasZgemv_64`|12.0| | | | | | | | | |
|`cublasZgemv_v2`| | | | |`hipblasZgemv_v2`|6.0.0| | | | |
|`cublasZgemv_v2_64`|12.0| | | | | | | | | |
|`cublasZgerc`| | | | |`hipblasZgerc_v2`|6.0.0| | | | |
|`cublasZgerc_64`|12.0| | | | | | | | | |
|`cublasZgerc_v2`| | | | |`hipblasZgerc_v2`|6.0.0| | | | |
|`cublasZgerc_v2_64`|12.0| | | | | | | | | |
|`cublasZgeru`| | | | |`hipblasZgeru_v2`|6.0.0| | | | |
|`cublasZgeru_64`|12.0| | | | | | | | | |
|`cublasZgeru_v2`| | | | |`hipblasZgeru_v2`|6.0.0| | | | |
|`cublasZgeru_v2_64`|12.0| | | | | | | | | |
|`cublasZhbmv`| | | | |`hipblasZhbmv_v2`|6.0.0| | | | |
|`cublasZhbmv_64`|12.0| | | | | | | | | |
|`cublasZhbmv_v2`| | | | |`hipblasZhbmv_v2`|6.0.0| | | | |
|`cublasZhbmv_v2_64`|12.0| | | | | | | | | |
|`cublasZhemv`| | | | |`hipblasZhemv_v2`|6.0.0| | | | |
|`cublasZhemv_64`|12.0| | | | | | | | | |
|`cublasZhemv_v2`| | | | |`hipblasZhemv_v2`|6.0.0| | | | |
|`cublasZhemv_v2_64`|12.0| | | | | | | | | |
|`cublasZher`| | | | |`hipblasZher_v2`|6.0.0| | | | |
|`cublasZher2`| | | | |`hipblasZher2_v2`|6.0.0| | | | |
|`cublasZher2_64`|12.0| | | | | | | | | |
|`cublasZher2_v2`| | | | |`hipblasZher2_v2`|6.0.0| | | | |
|`cublasZher2_v2_64`|12.0| | | | | | | | | |
|`cublasZher_64`|12.0| | | | | | | | | |
|`cublasZher_v2`| | | | |`hipblasZher_v2`|6.0.0| | | | |
|`cublasZher_v2_64`|12.0| | | | | | | | | |
|`cublasZhpmv`| | | | |`hipblasZhpmv_v2`|6.0.0| | | | |
|`cublasZhpmv_64`|12.0| | | | | | | | | |
|`cublasZhpmv_v2`| | | | |`hipblasZhpmv_v2`|6.0.0| | | | |
|`cublasZhpmv_v2_64`|12.0| | | | | | | | | |
|`cublasZhpr`| | | | |`hipblasZhpr_v2`|6.0.0| | | | |
|`cublasZhpr2`| | | | |`hipblasZhpr2_v2`|6.0.0| | | | |
|`cublasZhpr2_64`|12.0| | | | | | | | | |
|`cublasZhpr2_v2`| | | | |`hipblasZhpr2_v2`|6.0.0| | | | |
|`cublasZhpr2_v2_64`|12.0| | | | | | | | | |
|`cublasZhpr_64`|12.0| | | | | | | | | |
|`cublasZhpr_v2`| | | | |`hipblasZhpr_v2`|6.0.0| | | | |
|`cublasZhpr_v2_64`|12.0| | | | | | | | | |
|`cublasZsymv`| | | | |`hipblasZsymv_v2`|6.0.0| | | | |
|`cublasZsymv_64`|12.0| | | | | | | | | |
|`cublasZsymv_v2`| | | | |`hipblasZsymv_v2`|6.0.0| | | | |
|`cublasZsymv_v2_64`|12.0| | | | | | | | | |
|`cublasZsyr`| | | | |`hipblasZsyr_v2`|6.0.0| | | | |
|`cublasZsyr2`| | | | |`hipblasZsyr2_v2`|6.0.0| | | | |
|`cublasZsyr2_64`|12.0| | | | | | | | | |
|`cublasZsyr2_v2`| | | | |`hipblasZsyr2_v2`|6.0.0| | | | |
|`cublasZsyr2_v2_64`|12.0| | | | | | | | | |
|`cublasZsyr_64`|12.0| | | | | | | | | |
|`cublasZsyr_v2`| | | | |`hipblasZsyr_v2`|6.0.0| | | | |
|`cublasZsyr_v2_64`|12.0| | | | | | | | | |
|`cublasZtbmv`| | | | |`hipblasZtbmv_v2`|6.0.0| | | | |
|`cublasZtbmv_64`|12.0| | | | | | | | | |
|`cublasZtbmv_v2`| | | | |`hipblasZtbmv_v2`|6.0.0| | | | |
|`cublasZtbmv_v2_64`|12.0| | | | | | | | | |
|`cublasZtbsv`| | | | |`hipblasZtbsv_v2`|6.0.0| | | | |
|`cublasZtbsv_64`|12.0| | | | | | | | | |
|`cublasZtbsv_v2`| | | | |`hipblasZtbsv_v2`|6.0.0| | | | |
|`cublasZtbsv_v2_64`|12.0| | | | | | | | | |
|`cublasZtpmv`| | | | |`hipblasZtpmv_v2`|6.0.0| | | | |
|`cublasZtpmv_64`|12.0| | | | | | | | | |
|`cublasZtpmv_v2`| | | | |`hipblasZtpmv_v2`|6.0.0| | | | |
|`cublasZtpmv_v2_64`|12.0| | | | | | | | | |
|`cublasZtpsv`| | | | |`hipblasZtpsv_v2`|6.0.0| | | | |
|`cublasZtpsv_64`|12.0| | | | | | | | | |
|`cublasZtpsv_v2`| | | | |`hipblasZtpsv_v2`|6.0.0| | | | |
|`cublasZtpsv_v2_64`|12.0| | | | | | | | | |
|`cublasZtrmv`| | | | |`hipblasZtrmv_v2`|6.0.0| | | | |
|`cublasZtrmv_64`|12.0| | | | | | | | | |
|`cublasZtrmv_v2`| | | | |`hipblasZtrmv_v2`|6.0.0| | | | |
|`cublasZtrmv_v2_64`|12.0| | | | | | | | | |
|`cublasZtrsv`| | | | |`hipblasZtrsv_v2`|6.0.0| | | | |
|`cublasZtrsv_64`|12.0| | | | | | | | | |
|`cublasZtrsv_v2`| | | | |`hipblasZtrsv_v2`|6.0.0| | | | |
|`cublasZtrsv_v2_64`|12.0| | | | | | | | | |

## **7. CUBLAS Level-3 Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasCgemm`| | | | |`hipblasCgemm_v2`|6.0.0| | | | |
|`cublasCgemm3m`|8.0| | | | | | | | | |
|`cublasCgemm3mBatched`|8.0| | | | | | | | | |
|`cublasCgemm3mBatched_64`|12.0| | | | | | | | | |
|`cublasCgemm3mEx`|8.0| | | | | | | | | |
|`cublasCgemm3mEx_64`|12.0| | | | | | | | | |
|`cublasCgemm3mStridedBatched`|8.0| | | | | | | | | |
|`cublasCgemm3mStridedBatched_64`|12.0| | | | | | | | | |
|`cublasCgemm3m_64`|12.0| | | | | | | | | |
|`cublasCgemmBatched`| | | | |`hipblasCgemmBatched_v2`|6.0.0| | | | |
|`cublasCgemmBatched_64`|12.0| | | | | | | | | |
|`cublasCgemmStridedBatched`|8.0| | | |`hipblasCgemmStridedBatched_v2`|6.0.0| | | | |
|`cublasCgemmStridedBatched_64`|12.0| | | | | | | | | |
|`cublasCgemm_64`|12.0| | | | | | | | | |
|`cublasCgemm_v2`| | | | |`hipblasCgemm_v2`|6.0.0| | | | |
|`cublasCgemm_v2_64`|12.0| | | | | | | | | |
|`cublasCgemvBatched`|11.6| | | |`hipblasCgemvBatched_v2`|6.0.0| | | | |
|`cublasCgemvBatched_64`|12.0| | | | | | | | | |
|`cublasCgemvStridedBatched`|11.6| | | |`hipblasCgemvStridedBatched_v2`|6.0.0| | | | |
|`cublasCgemvStridedBatched_64`|12.0| | | | | | | | | |
|`cublasChemm`| | | | |`hipblasChemm_v2`|6.0.0| | | | |
|`cublasChemm_64`|12.0| | | | | | | | | |
|`cublasChemm_v2`| | | | |`hipblasChemm_v2`|6.0.0| | | | |
|`cublasChemm_v2_64`|12.0| | | | | | | | | |
|`cublasCher2k`| | | | |`hipblasCher2k_v2`|6.0.0| | | | |
|`cublasCher2k_64`|12.0| | | | | | | | | |
|`cublasCher2k_v2`| | | | |`hipblasCher2k_v2`|6.0.0| | | | |
|`cublasCher2k_v2_64`|12.0| | | | | | | | | |
|`cublasCherk`| | | | |`hipblasCherk_v2`|6.0.0| | | | |
|`cublasCherk_64`|12.0| | | | | | | | | |
|`cublasCherk_v2`| | | | |`hipblasCherk_v2`|6.0.0| | | | |
|`cublasCherk_v2_64`|12.0| | | | | | | | | |
|`cublasCherkx`| | | | |`hipblasCherkx_v2`|6.0.0| | | | |
|`cublasCherkx_64`|12.0| | | | | | | | | |
|`cublasCsymm`| | | | |`hipblasCsymm_v2`|6.0.0| | | | |
|`cublasCsymm_64`|12.0| | | | | | | | | |
|`cublasCsymm_v2`| | | | |`hipblasCsymm_v2`|6.0.0| | | | |
|`cublasCsymm_v2_64`|12.0| | | | | | | | | |
|`cublasCsyr2k`| | | | |`hipblasCsyr2k_v2`|6.0.0| | | | |
|`cublasCsyr2k_64`|12.0| | | | | | | | | |
|`cublasCsyr2k_v2`| | | | |`hipblasCsyr2k_v2`|6.0.0| | | | |
|`cublasCsyr2k_v2_64`|12.0| | | | | | | | | |
|`cublasCsyrk`| | | | |`hipblasCsyrk_v2`|6.0.0| | | | |
|`cublasCsyrk_64`|12.0| | | | | | | | | |
|`cublasCsyrk_v2`| | | | |`hipblasCsyrk_v2`|6.0.0| | | | |
|`cublasCsyrk_v2_64`|12.0| | | | | | | | | |
|`cublasCsyrkx`| | | | |`hipblasCsyrkx_v2`|6.0.0| | | | |
|`cublasCsyrkx_64`|12.0| | | | | | | | | |
|`cublasCtrmm`| | | | |`hipblasCtrmm_v2`|6.0.0| | | | |
|`cublasCtrmm_64`|12.0| | | | | | | | | |
|`cublasCtrmm_v2`| | | | |`hipblasCtrmm_v2`|6.0.0| | | | |
|`cublasCtrmm_v2_64`|12.0| | | | | | | | | |
|`cublasCtrsm`| | | | |`hipblasCtrsm_v2`|6.0.0| | | | |
|`cublasCtrsm_64`|12.0| | | | | | | | | |
|`cublasCtrsm_v2`| | | | |`hipblasCtrsm_v2`|6.0.0| | | | |
|`cublasCtrsm_v2_64`|12.0| | | | | | | | | |
|`cublasDgemm`| | | | |`hipblasDgemm`|1.8.2| | | | |
|`cublasDgemmBatched`| | | | |`hipblasDgemmBatched`|1.8.2| | | | |
|`cublasDgemmBatched_64`|12.0| | | | | | | | | |
|`cublasDgemmStridedBatched`|8.0| | | |`hipblasDgemmStridedBatched`|1.8.2| | | | |
|`cublasDgemmStridedBatched_64`|12.0| | | | | | | | | |
|`cublasDgemm_64`|12.0| | | | | | | | | |
|`cublasDgemm_v2`| | | | |`hipblasDgemm`|1.8.2| | | | |
|`cublasDgemm_v2_64`|12.0| | | | | | | | | |
|`cublasDgemvBatched`|11.6| | | | | | | | | |
|`cublasDgemvBatched_64`|12.0| | | | | | | | | |
|`cublasDgemvStridedBatched`|11.6| | | | | | | | | |
|`cublasDgemvStridedBatched_64`|12.0| | | | | | | | | |
|`cublasDsymm`| | | | |`hipblasDsymm`|3.6.0| | | | |
|`cublasDsymm_64`|12.0| | | | | | | | | |
|`cublasDsymm_v2`| | | | |`hipblasDsymm`|3.6.0| | | | |
|`cublasDsymm_v2_64`|12.0| | | | | | | | | |
|`cublasDsyr2k`| | | | |`hipblasDsyr2k`|3.5.0| | | | |
|`cublasDsyr2k_64`|12.0| | | | | | | | | |
|`cublasDsyr2k_v2`| | | | |`hipblasDsyr2k`|3.5.0| | | | |
|`cublasDsyr2k_v2_64`|12.0| | | | | | | | | |
|`cublasDsyrk`| | | | |`hipblasDsyrk`|3.5.0| | | | |
|`cublasDsyrk_64`|12.0| | | | | | | | | |
|`cublasDsyrk_v2`| | | | |`hipblasDsyrk`|3.5.0| | | | |
|`cublasDsyrk_v2_64`|12.0| | | | | | | | | |
|`cublasDsyrkx`| | | | |`hipblasDsyrkx`|3.5.0| | | | |
|`cublasDsyrkx_64`|12.0| | | | | | | | | |
|`cublasDtrmm`| | | | |`hipblasDtrmm`|3.2.0| |6.0.0| | |
|`cublasDtrmm_64`|12.0| | | | | | | | | |
|`cublasDtrmm_v2`| | | | |`hipblasDtrmm`|3.2.0| |6.0.0| | |
|`cublasDtrmm_v2_64`|12.0| | | | | | | | | |
|`cublasDtrsm`| | | | |`hipblasDtrsm`|1.8.2| | | | |
|`cublasDtrsm_64`|12.0| | | | | | | | | |
|`cublasDtrsm_v2`| | | | |`hipblasDtrsm`|1.8.2| | | | |
|`cublasDtrsm_v2_64`|12.0| | | | | | | | | |
|`cublasHSHgemvBatched`|11.6| | | | | | | | | |
|`cublasHSHgemvBatched_64`|12.0| | | | | | | | | |
|`cublasHSHgemvStridedBatched`|11.6| | | | | | | | | |
|`cublasHSHgemvStridedBatched_64`|12.0| | | | | | | | | |
|`cublasHSSgemvBatched`|11.6| | | | | | | | | |
|`cublasHSSgemvBatched_64`|12.0| | | | | | | | | |
|`cublasHSSgemvStridedBatched`|11.6| | | | | | | | | |
|`cublasHSSgemvStridedBatched_64`|12.0| | | | | | | | | |
|`cublasHgemm`|7.5| | | |`hipblasHgemm`|1.8.2| | | | |
|`cublasHgemmBatched`|9.0| | | |`hipblasHgemmBatched`|3.0.0| | | | |
|`cublasHgemmBatched_64`|12.0| | | | | | | | | |
|`cublasHgemmStridedBatched`|8.0| | | |`hipblasHgemmStridedBatched`|3.0.0| | | | |
|`cublasHgemmStridedBatched_64`|12.0| | | | | | | | | |
|`cublasHgemm_64`|12.0| | | | | | | | | |
|`cublasSgemm`| | | | |`hipblasSgemm`|1.8.2| | | | |
|`cublasSgemmBatched`| | | | |`hipblasSgemmBatched`|1.8.2| | | | |
|`cublasSgemmBatched_64`|12.0| | | | | | | | | |
|`cublasSgemmStridedBatched`|8.0| | | |`hipblasSgemmStridedBatched`|1.8.2| | | | |
|`cublasSgemmStridedBatched_64`|12.0| | | | | | | | | |
|`cublasSgemm_64`|12.0| | | | | | | | | |
|`cublasSgemm_v2`| | | | |`hipblasSgemm`|1.8.2| | | | |
|`cublasSgemm_v2_64`|12.0| | | | | | | | | |
|`cublasSgemvBatched`|11.6| | | | | | | | | |
|`cublasSgemvBatched_64`|12.0| | | | | | | | | |
|`cublasSgemvStridedBatched`|11.6| | | | | | | | | |
|`cublasSgemvStridedBatched_64`|12.0| | | | | | | | | |
|`cublasSsymm`| | | | |`hipblasSsymm`|3.6.0| | | | |
|`cublasSsymm_64`|12.0| | | | | | | | | |
|`cublasSsymm_v2`| | | | |`hipblasSsymm`|3.6.0| | | | |
|`cublasSsymm_v2_64`|12.0| | | | | | | | | |
|`cublasSsyr2k`| | | | |`hipblasSsyr2k`|3.5.0| | | | |
|`cublasSsyr2k_64`|12.0| | | | | | | | | |
|`cublasSsyr2k_v2`| | | | |`hipblasSsyr2k`|3.5.0| | | | |
|`cublasSsyr2k_v2_64`|12.0| | | | | | | | | |
|`cublasSsyrk`| | | | |`hipblasSsyrk`|3.5.0| | | | |
|`cublasSsyrk_64`|12.0| | | | | | | | | |
|`cublasSsyrk_v2`| | | | |`hipblasSsyrk`|3.5.0| | | | |
|`cublasSsyrk_v2_64`|12.0| | | | | | | | | |
|`cublasSsyrkx`| | | | |`hipblasSsyrkx`|3.5.0| | | | |
|`cublasSsyrkx_64`|12.0| | | | | | | | | |
|`cublasStrmm`| | | | |`hipblasStrmm`|3.2.0| |6.0.0| | |
|`cublasStrmm_64`|12.0| | | | | | | | | |
|`cublasStrmm_v2`| | | | |`hipblasStrmm`|3.2.0| |6.0.0| | |
|`cublasStrmm_v2_64`|12.0| | | | | | | | | |
|`cublasStrsm`| | | | |`hipblasStrsm`|1.8.2| | | | |
|`cublasStrsm_64`|12.0| | | | | | | | | |
|`cublasStrsm_v2`| | | | |`hipblasStrsm`|1.8.2| | | | |
|`cublasStrsm_v2_64`|12.0| | | | | | | | | |
|`cublasTSSgemvBatched`|11.6| | | | | | | | | |
|`cublasTSSgemvBatched_64`|12.0| | | | | | | | | |
|`cublasTSSgemvStridedBatched`|11.6| | | | | | | | | |
|`cublasTSSgemvStridedBatched_64`|12.0| | | | | | | | | |
|`cublasTSTgemvBatched`|11.6| | | | | | | | | |
|`cublasTSTgemvBatched_64`|12.0| | | | | | | | | |
|`cublasTSTgemvStridedBatched`|11.6| | | | | | | | | |
|`cublasTSTgemvStridedBatched_64`|12.0| | | | | | | | | |
|`cublasZgemm`| | | | |`hipblasZgemm_v2`|6.0.0| | | | |
|`cublasZgemm3m`|8.0| | | | | | | | | |
|`cublasZgemm3m_64`|12.0| | | | | | | | | |
|`cublasZgemmBatched`| | | | |`hipblasZgemmBatched_v2`|6.0.0| | | | |
|`cublasZgemmBatched_64`|12.0| | | | | | | | | |
|`cublasZgemmStridedBatched`|8.0| | | |`hipblasZgemmStridedBatched_v2`|6.0.0| | | | |
|`cublasZgemmStridedBatched_64`|12.0| | | | | | | | | |
|`cublasZgemm_64`|12.0| | | | | | | | | |
|`cublasZgemm_v2`| | | | |`hipblasZgemm_v2`|6.0.0| | | | |
|`cublasZgemm_v2_64`|12.0| | | | | | | | | |
|`cublasZgemvBatched`|11.6| | | |`hipblasZgemvBatched_v2`|6.0.0| | | | |
|`cublasZgemvBatched_64`|12.0| | | | | | | | | |
|`cublasZgemvStridedBatched`|11.6| | | |`hipblasZgemvStridedBatched_v2`|6.0.0| | | | |
|`cublasZgemvStridedBatched_64`|12.0| | | | | | | | | |
|`cublasZhemm`| | | | |`hipblasZhemm_v2`|6.0.0| | | | |
|`cublasZhemm_64`|12.0| | | | | | | | | |
|`cublasZhemm_v2`| | | | |`hipblasZhemm_v2`|6.0.0| | | | |
|`cublasZhemm_v2_64`|12.0| | | | | | | | | |
|`cublasZher2k`| | | | |`hipblasZher2k_v2`|6.0.0| | | | |
|`cublasZher2k_64`|12.0| | | | | | | | | |
|`cublasZher2k_v2`| | | | |`hipblasZher2k_v2`|6.0.0| | | | |
|`cublasZher2k_v2_64`|12.0| | | | | | | | | |
|`cublasZherk`| | | | |`hipblasZherk_v2`|6.0.0| | | | |
|`cublasZherk_64`|12.0| | | | | | | | | |
|`cublasZherk_v2`| | | | |`hipblasZherk_v2`|6.0.0| | | | |
|`cublasZherk_v2_64`|12.0| | | | | | | | | |
|`cublasZherkx`| | | | |`hipblasZherkx_v2`|6.0.0| | | | |
|`cublasZherkx_64`|12.0| | | | | | | | | |
|`cublasZsymm`| | | | |`hipblasZsymm_v2`|6.0.0| | | | |
|`cublasZsymm_64`|12.0| | | | | | | | | |
|`cublasZsymm_v2`| | | | |`hipblasZsymm_v2`|6.0.0| | | | |
|`cublasZsymm_v2_64`|12.0| | | | | | | | | |
|`cublasZsyr2k`| | | | |`hipblasZsyr2k_v2`|6.0.0| | | | |
|`cublasZsyr2k_64`|12.0| | | | | | | | | |
|`cublasZsyr2k_v2`| | | | |`hipblasZsyr2k_v2`|6.0.0| | | | |
|`cublasZsyr2k_v2_64`|12.0| | | | | | | | | |
|`cublasZsyrk`| | | | |`hipblasZsyrk_v2`|6.0.0| | | | |
|`cublasZsyrk_64`|12.0| | | | | | | | | |
|`cublasZsyrk_v2`| | | | |`hipblasZsyrk_v2`|6.0.0| | | | |
|`cublasZsyrk_v2_64`|12.0| | | | | | | | | |
|`cublasZsyrkx`| | | | |`hipblasZsyrkx_v2`|6.0.0| | | | |
|`cublasZsyrkx_64`|12.0| | | | | | | | | |
|`cublasZtrmm`| | | | |`hipblasZtrmm_v2`|6.0.0| | | | |
|`cublasZtrmm_64`|12.0| | | | | | | | | |
|`cublasZtrmm_v2`| | | | |`hipblasZtrmm_v2`|6.0.0| | | | |
|`cublasZtrmm_v2_64`|12.0| | | | | | | | | |
|`cublasZtrsm`| | | | |`hipblasZtrsm_v2`|6.0.0| | | | |
|`cublasZtrsm_64`|12.0| | | | | | | | | |
|`cublasZtrsm_v2`| | | | |`hipblasZtrsm_v2`|6.0.0| | | | |
|`cublasZtrsm_v2_64`|12.0| | | | | | | | | |

## **8. BLAS-like Extension**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasAsumEx`|10.1| | | | | | | | | |
|`cublasAsumEx_64`|12.0| | | | | | | | | |
|`cublasAxpyEx`|8.0| | | |`hipblasAxpyEx_v2`|6.0.0| | | | |
|`cublasAxpyEx_64`|12.0| | | | | | | | | |
|`cublasCdgmm`| | | | |`hipblasCdgmm_v2`|6.0.0| | | | |
|`cublasCdgmm_64`|12.0| | | | | | | | | |
|`cublasCgeam`| | | | |`hipblasCgeam_v2`|6.0.0| | | | |
|`cublasCgeam_64`|12.0| | | | | | | | | |
|`cublasCgelsBatched`| | | | |`hipblasCgelsBatched_v2`|6.0.0| | | | |
|`cublasCgemmEx`|8.0| | | | | | | | | |
|`cublasCgemmEx_64`|12.0| | | | | | | | | |
|`cublasCgeqrfBatched`| | | | |`hipblasCgeqrfBatched_v2`|6.0.0| | | | |
|`cublasCgetrfBatched`| | | | |`hipblasCgetrfBatched_v2`|6.0.0| | | | |
|`cublasCgetriBatched`| | | | |`hipblasCgetriBatched_v2`|6.0.0| | | | |
|`cublasCgetrsBatched`| | | | |`hipblasCgetrsBatched_v2`|6.0.0| | | | |
|`cublasCherk3mEx`|8.0| | | | | | | | | |
|`cublasCherk3mEx_64`|12.0| | | | | | | | | |
|`cublasCherkEx`|8.0| | | | | | | | | |
|`cublasCherkEx_64`|12.0| | | | | | | | | |
|`cublasCmatinvBatched`| | | | | | | | | | |
|`cublasCopyEx`|10.1| | | | | | | | | |
|`cublasCopyEx_64`|12.0| | | | | | | | | |
|`cublasCsyrk3mEx`|8.0| | | | | | | | | |
|`cublasCsyrk3mEx_64`|12.0| | | | | | | | | |
|`cublasCsyrkEx`|8.0| | | | | | | | | |
|`cublasCsyrkEx_64`|12.0| | | | | | | | | |
|`cublasCtpttr`| | | | | | | | | | |
|`cublasCtrsmBatched`| | | | |`hipblasCtrsmBatched_v2`|6.0.0| | | | |
|`cublasCtrsmBatched_64`|12.0| | | | | | | | | |
|`cublasCtrttp`| | | | | | | | | | |
|`cublasDdgmm`| | | | |`hipblasDdgmm`|3.6.0| | | | |
|`cublasDdgmm_64`|12.0| | | | | | | | | |
|`cublasDgeam`| | | | |`hipblasDgeam`|1.8.2| | | | |
|`cublasDgeam_64`|12.0| | | | | | | | | |
|`cublasDgelsBatched`| | | | |`hipblasDgelsBatched`|5.4.0| | | | |
|`cublasDgeqrfBatched`| | | | |`hipblasDgeqrfBatched`|3.5.0| | | | |
|`cublasDgetrfBatched`| | | | |`hipblasDgetrfBatched`|3.5.0| | | | |
|`cublasDgetriBatched`| | | | |`hipblasDgetriBatched`|3.7.0| | | | |
|`cublasDgetrsBatched`| | | | |`hipblasDgetrsBatched`|3.5.0| | | | |
|`cublasDmatinvBatched`| | | | | | | | | | |
|`cublasDotEx`|8.0| | | |`hipblasDotEx_v2`|6.0.0| | | | |
|`cublasDotEx_64`|12.0| | | | | | | | | |
|`cublasDotcEx`|8.0| | | |`hipblasDotcEx_v2`|6.0.0| | | | |
|`cublasDotcEx_64`|12.0| | | | | | | | | |
|`cublasDtpttr`| | | | | | | | | | |
|`cublasDtrsmBatched`| | | | |`hipblasDtrsmBatched`|3.2.0| | | | |
|`cublasDtrsmBatched_64`|12.0| | | | | | | | | |
|`cublasDtrttp`| | | | | | | | | | |
|`cublasGemmBatchedEx`|9.1| | | |`hipblasGemmBatchedEx_v2`|6.0.0| | | | |
|`cublasGemmBatchedEx_64`|12.0| | | | | | | | | |
|`cublasGemmEx`|8.0| | | |`hipblasGemmEx_v2`|6.0.0| | | | |
|`cublasGemmEx_64`|12.0| | | | | | | | | |
|`cublasGemmStridedBatchedEx`|9.1| | | |`hipblasGemmStridedBatchedEx_v2`|6.0.0| | | | |
|`cublasGemmStridedBatchedEx_64`|12.0| | | | | | | | | |
|`cublasIamaxEx`|10.1| | | | | | | | | |
|`cublasIamaxEx_64`|12.0| | | | | | | | | |
|`cublasIaminEx`|10.1| | | | | | | | | |
|`cublasIaminEx_64`|12.0| | | | | | | | | |
|`cublasRotEx`|10.1| | | |`hipblasRotEx_v2`|6.0.0| | | | |
|`cublasRotEx_64`|12.0| | | | | | | | | |
|`cublasRotgEx`|10.1| | | | | | | | | |
|`cublasRotmEx`|10.1| | | | | | | | | |
|`cublasRotmEx_64`|12.0| | | | | | | | | |
|`cublasRotmgEx`|10.1| | | | | | | | | |
|`cublasScalEx`|8.0| | | |`hipblasScalEx_v2`|6.0.0| | | | |
|`cublasScalEx_64`|12.0| | | | | | | | | |
|`cublasSdgmm`| | | | |`hipblasSdgmm`|3.6.0| | | | |
|`cublasSdgmm_64`|12.0| | | | | | | | | |
|`cublasSgeam`| | | | |`hipblasSgeam`|1.8.2| | | | |
|`cublasSgeam_64`|12.0| | | | | | | | | |
|`cublasSgelsBatched`| | | | |`hipblasSgelsBatched`|5.4.0| | | | |
|`cublasSgemmEx`|7.5| | | | | | | | | |
|`cublasSgemmEx_64`|12.0| | | | | | | | | |
|`cublasSgeqrfBatched`| | | | |`hipblasSgeqrfBatched`|3.5.0| | | | |
|`cublasSgetrfBatched`| | | | |`hipblasSgetrfBatched`|3.5.0| | | | |
|`cublasSgetriBatched`| | | | |`hipblasSgetriBatched`|3.7.0| | | | |
|`cublasSgetrsBatched`| | | | |`hipblasSgetrsBatched`|3.5.0| | | | |
|`cublasSmatinvBatched`| | | | | | | | | | |
|`cublasStpttr`| | | | | | | | | | |
|`cublasStrsmBatched`| | | | |`hipblasStrsmBatched`|3.2.0| | | | |
|`cublasStrsmBatched_64`|12.0| | | | | | | | | |
|`cublasStrttp`| | | | | | | | | | |
|`cublasSwapEx`|10.1| | | | | | | | | |
|`cublasSwapEx_64`|12.0| | | | | | | | | |
|`cublasUint8gemmBias`|8.0| | | | | | | | | |
|`cublasZdgmm`| | | | |`hipblasZdgmm_v2`|6.0.0| | | | |
|`cublasZdgmm_64`|12.0| | | | | | | | | |
|`cublasZgeam`| | | | |`hipblasZgeam_v2`|6.0.0| | | | |
|`cublasZgeam_64`|12.0| | | | | | | | | |
|`cublasZgelsBatched`| | | | |`hipblasZgelsBatched_v2`|6.0.0| | | | |
|`cublasZgeqrfBatched`| | | | |`hipblasZgeqrfBatched_v2`|6.0.0| | | | |
|`cublasZgetrfBatched`| | | | |`hipblasZgetrfBatched_v2`|6.0.0| | | | |
|`cublasZgetriBatched`| | | | |`hipblasZgetriBatched_v2`|6.0.0| | | | |
|`cublasZgetrsBatched`| | | | |`hipblasZgetrsBatched_v2`|6.0.0| | | | |
|`cublasZmatinvBatched`| | | | | | | | | | |
|`cublasZtpttr`| | | | | | | | | | |
|`cublasZtrsmBatched`| | | | |`hipblasZtrsmBatched_v2`|6.0.0| | | | |
|`cublasZtrsmBatched_64`|12.0| | | | | | | | | |
|`cublasZtrttp`| | | | | | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental