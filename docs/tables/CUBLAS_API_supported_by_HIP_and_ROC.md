# CUBLAS API supported by HIP and ROC

## **2. CUBLAS Data types**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|**E**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`CUBLAS_ATOMICS_ALLOWED`| | | |`HIPBLAS_ATOMICS_ALLOWED`|3.10.0| | | |`rocblas_atomics_allowed`|3.8.0| | | |
|`CUBLAS_ATOMICS_NOT_ALLOWED`| | | |`HIPBLAS_ATOMICS_NOT_ALLOWED`|3.10.0| | | |`rocblas_atomics_not_allowed`|3.8.0| | | |
|`CUBLAS_COMPUTE_16F`|11.0| | | | | | | | | | | | |
|`CUBLAS_COMPUTE_16F_PEDANTIC`|11.0| | | | | | | | | | | | |
|`CUBLAS_COMPUTE_32F`|11.0| | | | | | | | | | | | |
|`CUBLAS_COMPUTE_32F_FAST_16BF`|11.0| | | | | | | | | | | | |
|`CUBLAS_COMPUTE_32F_FAST_16F`|11.0| | | | | | | | | | | | |
|`CUBLAS_COMPUTE_32F_FAST_TF32`|11.0| | | | | | | | | | | | |
|`CUBLAS_COMPUTE_32F_PEDANTIC`|11.0| | | | | | | | | | | | |
|`CUBLAS_COMPUTE_32I`|11.0| | | | | | | | | | | | |
|`CUBLAS_COMPUTE_32I_PEDANTIC`|11.0| | | | | | | | | | | | |
|`CUBLAS_COMPUTE_64F`|11.0| | | | | | | | | | | | |
|`CUBLAS_COMPUTE_64F_PEDANTIC`|11.0| | | | | | | | | | | | |
|`CUBLAS_DEFAULT_MATH`|9.0| | | | | | | | | | | | |
|`CUBLAS_DIAG_NON_UNIT`| | | |`HIPBLAS_DIAG_NON_UNIT`|1.8.2| | | |`rocblas_diagonal_non_unit`|1.5.0| | | |
|`CUBLAS_DIAG_UNIT`| | | |`HIPBLAS_DIAG_UNIT`|1.8.2| | | |`rocblas_diagonal_unit`|1.5.0| | | |
|`CUBLAS_FILL_MODE_FULL`|10.1| | |`HIPBLAS_FILL_MODE_FULL`|1.8.2| | | |`rocblas_fill_full`|1.5.0| | | |
|`CUBLAS_FILL_MODE_LOWER`| | | |`HIPBLAS_FILL_MODE_LOWER`|1.8.2| | | |`rocblas_fill_lower`|1.5.0| | | |
|`CUBLAS_FILL_MODE_UPPER`| | | |`HIPBLAS_FILL_MODE_UPPER`|1.8.2| | | |`rocblas_fill_upper`|1.5.0| | | |
|`CUBLAS_GEMM_ALGO0`|8.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO0_TENSOR_OP`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO1`|8.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO10`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO10_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO11`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO11_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO12`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO12_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO13`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO13_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO14`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO14_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO15`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO15_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO16`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO17`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO18`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO19`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO1_TENSOR_OP`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO2`|8.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO20`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO21`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO22`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO23`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO2_TENSOR_OP`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO3`|8.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO3_TENSOR_OP`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO4`|8.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO4_TENSOR_OP`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO5`|8.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO5_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO6`|8.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO6_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO7`|8.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO7_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO8`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO8_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO9`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_ALGO9_TENSOR_OP`|9.2| | | | | | | | | | | | |
|`CUBLAS_GEMM_DEFAULT`|9.0| | |`HIPBLAS_GEMM_DEFAULT`|1.8.2| | | |`rocblas_gemm_algo_standard`|1.8.2| | | |
|`CUBLAS_GEMM_DEFAULT_TENSOR_OP`|9.0| | | | | | | | | | | | |
|`CUBLAS_GEMM_DFALT`|8.0| | |`HIPBLAS_GEMM_DEFAULT`|1.8.2| | | |`rocblas_gemm_algo_standard`|1.8.2| | | |
|`CUBLAS_GEMM_DFALT_TENSOR_OP`|9.0| | | | | | | | | | | | |
|`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`|11.0| | | | | | | | | | | | |
|`CUBLAS_OP_C`| | | |`HIPBLAS_OP_C`|1.8.2| | | |`rocblas_operation_conjugate_transpose`|1.5.0| | | |
|`CUBLAS_OP_CONJG`|10.1| | | | | | | | | | | | |
|`CUBLAS_OP_HERMITAN`|10.1| | |`HIPBLAS_OP_C`|1.8.2| | | |`rocblas_operation_conjugate_transpose`|1.5.0| | | |
|`CUBLAS_OP_N`| | | |`HIPBLAS_OP_N`|1.8.2| | | |`rocblas_operation_none`|1.5.0| | | |
|`CUBLAS_OP_T`| | | |`HIPBLAS_OP_T`|1.8.2| | | |`rocblas_operation_transpose`|1.5.0| | | |
|`CUBLAS_PEDANTIC_MATH`|11.0| | | | | | | | | | | | |
|`CUBLAS_POINTER_MODE_DEVICE`| | | |`HIPBLAS_POINTER_MODE_DEVICE`|1.8.2| | | |`rocblas_pointer_mode_device`|1.6.0| | | |
|`CUBLAS_POINTER_MODE_HOST`| | | |`HIPBLAS_POINTER_MODE_HOST`|1.8.2| | | |`rocblas_pointer_mode_host`|1.6.0| | | |
|`CUBLAS_SIDE_LEFT`| | | |`HIPBLAS_SIDE_LEFT`|1.8.2| | | |`rocblas_side_left`|1.5.0| | | |
|`CUBLAS_SIDE_RIGHT`| | | |`HIPBLAS_SIDE_RIGHT`|1.8.2| | | |`rocblas_side_right`|1.5.0| | | |
|`CUBLAS_STATUS_ALLOC_FAILED`| | | |`HIPBLAS_STATUS_ALLOC_FAILED`|1.8.2| | | |`rocblas_status_not_implemented`|1.5.0| | | |
|`CUBLAS_STATUS_ARCH_MISMATCH`| | | |`HIPBLAS_STATUS_ARCH_MISMATCH`|1.8.2| | | |`rocblas_status_size_query_mismatch`|3.5.0| | | |
|`CUBLAS_STATUS_EXECUTION_FAILED`| | | |`HIPBLAS_STATUS_EXECUTION_FAILED`|1.8.2| | | |`rocblas_status_memory_error`|1.5.0| | | |
|`CUBLAS_STATUS_INTERNAL_ERROR`| | | |`HIPBLAS_STATUS_INTERNAL_ERROR`|1.8.2| | | |`rocblas_status_internal_error`|1.5.0| | | |
|`CUBLAS_STATUS_INVALID_VALUE`| | | |`HIPBLAS_STATUS_INVALID_VALUE`|1.8.2| | | |`rocblas_status_invalid_pointer`|1.5.0| | | |
|`CUBLAS_STATUS_LICENSE_ERROR`| | | |`HIPBLAS_STATUS_UNKNOWN`| | | | | | | | | |
|`CUBLAS_STATUS_MAPPING_ERROR`| | | |`HIPBLAS_STATUS_MAPPING_ERROR`|1.8.2| | | |`rocblas_status_invalid_size`|1.5.0| | | |
|`CUBLAS_STATUS_NOT_INITIALIZED`| | | |`HIPBLAS_STATUS_NOT_INITIALIZED`|1.8.2| | | |`rocblas_status_invalid_handle`|1.5.0| | | |
|`CUBLAS_STATUS_NOT_SUPPORTED`| | | |`HIPBLAS_STATUS_NOT_SUPPORTED`|1.8.2| | | |`rocblas_status_perf_degraded`|3.5.0| | | |
|`CUBLAS_STATUS_SUCCESS`| | | |`HIPBLAS_STATUS_SUCCESS`|1.8.2| | | |`rocblas_status_success`|1.5.0| | | |
|`CUBLAS_TENSOR_OP_MATH`|9.0|11.0| | | | | | | | | | | |
|`CUBLAS_TF32_TENSOR_OP_MATH`|11.0| | | | | | | | | | | | |
|`cublasAtomicsMode_t`| | | |`hipblasAtomicsMode_t`|3.10.0| | | |`rocblas_atomics_mode`|3.8.0| | | |
|`cublasComputeType_t`|11.0| | |`hipblasDatatype_t`|1.8.2| | | | | | | | |
|`cublasContext`| | | | | | | | |`_rocblas_handle`|1.5.0| | | |
|`cublasDataType_t`|7.5| | |`hipblasDatatype_t`|1.8.2| | | |`rocblas_datatype`|1.8.2| | | |
|`cublasDiagType_t`| | | |`hipblasDiagType_t`|1.8.2| | | |`rocblas_diagonal`|1.5.0| | | |
|`cublasFillMode_t`| | | |`hipblasFillMode_t`|1.8.2| | | |`rocblas_fill`|1.5.0| | | |
|`cublasGemmAlgo_t`|8.0| | |`hipblasGemmAlgo_t`|1.8.2| | | |`rocblas_gemm_algo`|1.8.2| | | |
|`cublasHandle_t`| | | |`hipblasHandle_t`|3.0.0| | | |`rocblas_handle`|1.5.0| | | |
|`cublasMath_t`|9.0| | | | | | | | | | | | |
|`cublasOperation_t`| | | |`hipblasOperation_t`|1.8.2| | | |`rocblas_operation`|1.5.0| | | |
|`cublasPointerMode_t`| | | |`hipblasPointerMode_t`|1.8.2| | | |`rocblas_pointer_mode`|1.6.0| | | |
|`cublasSideMode_t`| | | |`hipblasSideMode_t`|1.8.2| | | |`rocblas_side`|1.5.0| | | |
|`cublasStatus`| | | |`hipblasStatus_t`|1.8.2| | | |`rocblas_status`|1.5.0| | | |
|`cublasStatus_t`| | | |`hipblasStatus_t`|1.8.2| | | |`rocblas_status`|1.5.0| | | |

## **3. CUDA Datatypes Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|**E**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`CUDA_C_16BF`| | | |`HIPBLAS_C_16B`|3.0.0| | | |`rocblas_datatype_bf16_c`|3.5.0| | | |
|`CUDA_C_16F`|8.0| | |`HIPBLAS_C_16F`|1.8.2| | | |`rocblas_datatype_f16_c`|1.8.2| | | |
|`CUDA_C_16I`|11.0| | | | | | | | | | | | |
|`CUDA_C_16U`|11.0| | | | | | | | | | | | |
|`CUDA_C_32F`|8.0| | |`HIPBLAS_C_32F`|1.8.2| | | |`rocblas_datatype_f32_c`|1.8.2| | | |
|`CUDA_C_32I`|8.0| | |`HIPBLAS_C_32I`|3.0.0| | | |`rocblas_datatype_i32_c`|2.0.0| | | |
|`CUDA_C_32U`|8.0| | |`HIPBLAS_C_32U`|3.0.0| | | |`rocblas_datatype_u32_c`|2.0.0| | | |
|`CUDA_C_4I`|11.0| | | | | | | | | | | | |
|`CUDA_C_4U`|11.0| | | | | | | | | | | | |
|`CUDA_C_64F`|8.0| | |`HIPBLAS_C_64F`|1.8.2| | | |`rocblas_datatype_f64_c`|1.8.2| | | |
|`CUDA_C_64I`|11.0| | | | | | | | | | | | |
|`CUDA_C_64U`|11.0| | | | | | | | | | | | |
|`CUDA_C_8I`|8.0| | |`HIPBLAS_C_8I`|3.0.0| | | |`rocblas_datatype_i8_c`|2.0.0| | | |
|`CUDA_C_8U`|8.0| | |`HIPBLAS_C_8U`|3.0.0| | | |`rocblas_datatype_u8_c`|2.0.0| | | |
|`CUDA_R_16BF`| | | |`HIPBLAS_R_16B`|3.0.0| | | |`rocblas_datatype_bf16_r`|3.5.0| | | |
|`CUDA_R_16F`|8.0| | |`HIPBLAS_R_16F`|1.8.2| | | |`rocblas_datatype_f16_r`|1.8.2| | | |
|`CUDA_R_16I`|11.0| | | | | | | | | | | | |
|`CUDA_R_16U`|11.0| | | | | | | | | | | | |
|`CUDA_R_32F`|8.0| | |`HIPBLAS_R_32F`|1.8.2| | | |`rocblas_datatype_f32_r`|1.8.2| | | |
|`CUDA_R_32I`|8.0| | |`HIPBLAS_R_32I`|3.0.0| | | |`rocblas_datatype_i32_r`|2.0.0| | | |
|`CUDA_R_32U`|8.0| | |`HIPBLAS_R_32U`|3.0.0| | | |`rocblas_datatype_u32_r`|2.0.0| | | |
|`CUDA_R_4I`|11.0| | | | | | | | | | | | |
|`CUDA_R_4U`|11.0| | | | | | | | | | | | |
|`CUDA_R_64F`|8.0| | |`HIPBLAS_R_64F`|1.8.2| | | |`rocblas_datatype_f64_r`|1.8.2| | | |
|`CUDA_R_64I`|11.0| | | | | | | | | | | | |
|`CUDA_R_64U`|11.0| | | | | | | | | | | | |
|`CUDA_R_8F_E4M3`|11.8| | | | | | | | | | | | |
|`CUDA_R_8F_E5M2`|11.8| | | | | | | | | | | | |
|`CUDA_R_8I`|8.0| | |`HIPBLAS_R_8I`|3.0.0| | | |`rocblas_datatype_i8_r`|2.0.0| | | |
|`CUDA_R_8U`|8.0| | |`HIPBLAS_R_8U`|3.0.0| | | |`rocblas_datatype_u8_r`|2.0.0| | | |
|`cudaDataType`|8.0| | |`hipblasDatatype_t`|1.8.2| | | |`rocblas_datatype`|1.8.2| | | |
|`cudaDataType_t`|8.0| | |`hipblasDatatype_t`|1.8.2| | | |`rocblas_datatype_`|1.8.2| | | |

## **4. CUBLAS Helper Function Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|**E**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cublasAlloc`| | | | | | | | | | | | | |
|`cublasCreate`| | | |`hipblasCreate`|1.8.2| | | |`rocblas_create_handle`|1.5.0| | | |
|`cublasCreate_v2`| | | |`hipblasCreate`|1.8.2| | | |`rocblas_create_handle`|1.5.0| | | |
|`cublasDestroy`| | | |`hipblasDestroy`|1.8.2| | | |`rocblas_destroy_handle`|1.5.0| | | |
|`cublasDestroy_v2`| | | |`hipblasDestroy`|1.8.2| | | |`rocblas_destroy_handle`|1.5.0| | | |
|`cublasFree`| | | | | | | | | | | | | |
|`cublasGetAtomicsMode`| | | |`hipblasGetAtomicsMode`|3.10.0| | | |`rocblas_get_atomics_mode`|3.8.0| | | |
|`cublasGetCudartVersion`|10.1| | | | | | | | | | | | |
|`cublasGetError`| | | | | | | | | | | | | |
|`cublasGetLoggerCallback`|9.2| | | | | | | | | | | | |
|`cublasGetMathMode`|9.0| | | | | | | | | | | | |
|`cublasGetMatrix`| | | |`hipblasGetMatrix`|1.8.2| | | |`rocblas_get_matrix`|1.6.0| | | |
|`cublasGetMatrixAsync`| | | |`hipblasGetMatrixAsync`|3.7.0| | | |`rocblas_get_matrix_async`|3.5.0| | | |
|`cublasGetMatrixAsync_64`|12.0| | | | | | | | | | | | |
|`cublasGetMatrix_64`|12.0| | | | | | | | | | | | |
|`cublasGetPointerMode`| | | |`hipblasGetPointerMode`|1.8.2| | | |`rocblas_get_pointer_mode`|1.6.0| | | |
|`cublasGetPointerMode_v2`| | | |`hipblasGetPointerMode`|1.8.2| | | |`rocblas_get_pointer_mode`|1.6.0| | | |
|`cublasGetProperty`| | | | | | | | | | | | | |
|`cublasGetSmCountTarget`|11.3| | | | | | | | | | | | |
|`cublasGetStatusName`|11.4| | | | | | | | | | | | |
|`cublasGetStatusString`|11.4| | | | | | | |`rocblas_status_to_string`|3.5.0| | | |
|`cublasGetStream`| | | |`hipblasGetStream`|1.8.2| | | |`rocblas_get_stream`|1.5.0| | | |
|`cublasGetStream_v2`| | | |`hipblasGetStream`|1.8.2| | | |`rocblas_get_stream`|1.5.0| | | |
|`cublasGetVector`| | | |`hipblasGetVector`|1.8.2| | | |`rocblas_get_vector`|1.6.0| | | |
|`cublasGetVectorAsync`| | | |`hipblasGetVectorAsync`|3.7.0| | | |`rocblas_get_vector_async`|3.5.0| | | |
|`cublasGetVectorAsync_64`|12.0| | | | | | | | | | | | |
|`cublasGetVector_64`|12.0| | | | | | | | | | | | |
|`cublasGetVersion`| | | | | | | | | | | | | |
|`cublasGetVersion_v2`| | | | | | | | | | | | | |
|`cublasInit`| | | | | | | | |`rocblas_initialize`|3.5.0| | | |
|`cublasLogCallback`|9.2| | | | | | | | | | | | |
|`cublasLoggerConfigure`|9.2| | | | | | | | | | | | |
|`cublasMigrateComputeType`|11.0| | | | | | | | | | | | |
|`cublasSetAtomicsMode`| | | |`hipblasSetAtomicsMode`|3.10.0| | | |`rocblas_set_atomics_mode`|3.8.0| | | |
|`cublasSetKernelStream`| | | | | | | | | | | | | |
|`cublasSetLoggerCallback`|9.2| | | | | | | | | | | | |
|`cublasSetMathMode`| | | | | | | | | | | | | |
|`cublasSetMatrix`| | | |`hipblasSetMatrix`|1.8.2| | | |`rocblas_set_matrix`|1.6.0| | | |
|`cublasSetMatrixAsync`| | | |`hipblasSetMatrixAsync`|3.7.0| | | |`rocblas_set_matrix_async`|3.5.0| | | |
|`cublasSetMatrixAsync_64`|12.0| | | | | | | | | | | | |
|`cublasSetMatrix_64`|12.0| | | | | | | | | | | | |
|`cublasSetPointerMode`| | | |`hipblasSetPointerMode`|1.8.2| | | |`rocblas_set_pointer_mode`|1.6.0| | | |
|`cublasSetPointerMode_v2`| | | |`hipblasSetPointerMode`|1.8.2| | | |`rocblas_set_pointer_mode`|1.6.0| | | |
|`cublasSetSmCountTarget`|11.3| | | | | | | | | | | | |
|`cublasSetStream`| | | |`hipblasSetStream`|1.8.2| | | |`rocblas_set_stream`|1.5.0| | | |
|`cublasSetStream_v2`| | | |`hipblasSetStream`|1.8.2| | | |`rocblas_set_stream`|1.5.0| | | |
|`cublasSetVector`| | | |`hipblasSetVector`|1.8.2| | | |`rocblas_set_vector`|1.6.0| | | |
|`cublasSetVectorAsync`| | | |`hipblasSetVectorAsync`|3.7.0| | | |`rocblas_set_vector_async`|3.5.0| | | |
|`cublasSetVectorAsync_64`|12.0| | | | | | | | | | | | |
|`cublasSetVector_64`|12.0| | | | | | | | | | | | |
|`cublasShutdown`| | | | | | | | | | | | | |
|`cublasXerbla`| | | | | | | | | | | | | |

## **5. CUBLAS Level-1 Function Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|**E**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cublasCaxpy`| | | |`hipblasCaxpy`|3.0.0| | | |`rocblas_caxpy`|1.5.0| | | |
|`cublasCaxpy_64`|12.0| | | | | | | | | | | | |
|`cublasCaxpy_v2`| | | |`hipblasCaxpy`|3.0.0| | | |`rocblas_caxpy`|1.5.0| | | |
|`cublasCaxpy_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCcopy`| | | |`hipblasCcopy`|3.0.0| | | |`rocblas_ccopy`|1.5.0| | | |
|`cublasCcopy_64`|12.0| | | | | | | | | | | | |
|`cublasCcopy_v2`| | | |`hipblasCcopy`|3.0.0| | | |`rocblas_ccopy`|1.5.0| | | |
|`cublasCcopy_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCdotc`| | | |`hipblasCdotc`|3.0.0| | | |`rocblas_cdotc`|3.5.0| | | |
|`cublasCdotc_64`|12.0| | | | | | | | | | | | |
|`cublasCdotc_v2`| | | |`hipblasCdotc`|3.0.0| | | |`rocblas_cdotc`|3.5.0| | | |
|`cublasCdotc_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCdotu`| | | |`hipblasCdotu`|3.0.0| | | |`rocblas_cdotu`|1.5.0| | | |
|`cublasCdotu_64`|12.0| | | | | | | | | | | | |
|`cublasCdotu_v2`| | | |`hipblasCdotu`|3.0.0| | | |`rocblas_cdotu`|1.5.0| | | |
|`cublasCdotu_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCrot`| | | |`hipblasCrot`|3.0.0| | | |`rocblas_crot`|3.5.0| | | |
|`cublasCrot_64`|12.0| | | | | | | | | | | | |
|`cublasCrot_v2`| | | |`hipblasCrot`|3.0.0| | | |`rocblas_crot`|3.5.0| | | |
|`cublasCrot_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCrotg`| | | |`hipblasCrotg`|3.0.0| | | |`rocblas_crotg`|3.5.0| | | |
|`cublasCrotg_v2`| | | |`hipblasCrotg`|3.0.0| | | |`rocblas_crotg`|3.5.0| | | |
|`cublasCscal`| | | |`hipblasCscal`|1.8.2| | | |`rocblas_cscal`|1.5.0| | | |
|`cublasCscal_64`|12.0| | | | | | | | | | | | |
|`cublasCscal_v2`| | | |`hipblasCscal`|1.8.2| | | |`rocblas_cscal`|1.5.0| | | |
|`cublasCscal_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCsrot`| | | |`hipblasCsrot`|3.0.0| | | |`rocblas_csrot`|3.5.0| | | |
|`cublasCsrot_64`|12.0| | | | | | | | | | | | |
|`cublasCsrot_v2`| | | |`hipblasCsrot`|3.0.0| | | |`rocblas_csrot`|3.5.0| | | |
|`cublasCsrot_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCsscal`| | | |`hipblasCsscal`|3.0.0| | | |`rocblas_csscal`|3.5.0| | | |
|`cublasCsscal_64`|12.0| | | | | | | | | | | | |
|`cublasCsscal_v2`| | | |`hipblasCsscal`|3.0.0| | | |`rocblas_csscal`|3.5.0| | | |
|`cublasCsscal_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCswap`| | | |`hipblasCswap`|3.0.0| | | |`rocblas_cswap`|1.5.0| | | |
|`cublasCswap_64`|12.0| | | | | | | | | | | | |
|`cublasCswap_v2`| | | |`hipblasCswap`|3.0.0| | | |`rocblas_cswap`|1.5.0| | | |
|`cublasCswap_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDasum`| | | |`hipblasDasum`|1.8.2| | | |`rocblas_dasum`|1.5.0| | | |
|`cublasDasum_64`|12.0| | | | | | | | | | | | |
|`cublasDasum_v2`| | | |`hipblasDasum`|1.8.2| | | |`rocblas_dasum`|1.5.0| | | |
|`cublasDasum_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDaxpy`| | | |`hipblasDaxpy`|1.8.2| | | |`rocblas_daxpy`|1.5.0| | | |
|`cublasDaxpy_64`|12.0| | | | | | | | | | | | |
|`cublasDaxpy_v2`| | | |`hipblasDaxpy`|1.8.2| | | |`rocblas_daxpy`|1.5.0| | | |
|`cublasDaxpy_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDcopy`| | | |`hipblasDcopy`|1.8.2| | | |`rocblas_dcopy`|1.5.0| | | |
|`cublasDcopy_64`|12.0| | | | | | | | | | | | |
|`cublasDcopy_v2`| | | |`hipblasDcopy`|1.8.2| | | |`rocblas_dcopy`|1.5.0| | | |
|`cublasDcopy_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDdot`| | | |`hipblasDdot`|3.0.0| | | |`rocblas_ddot`|1.5.0| | | |
|`cublasDdot_64`|12.0| | | | | | | | | | | | |
|`cublasDdot_v2`| | | |`hipblasDdot`|3.0.0| | | |`rocblas_ddot`|1.5.0| | | |
|`cublasDdot_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDnrm2`| | | |`hipblasDnrm2`|1.8.2| | | |`rocblas_dnrm2`|1.5.0| | | |
|`cublasDnrm2_64`|12.0| | | | | | | | | | | | |
|`cublasDnrm2_v2`| | | |`hipblasDnrm2`|1.8.2| | | |`rocblas_dnrm2`|1.5.0| | | |
|`cublasDnrm2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDrot`| | | |`hipblasDrot`|3.0.0| | | |`rocblas_drot`|3.5.0| | | |
|`cublasDrot_64`|12.0| | | | | | | | | | | | |
|`cublasDrot_v2`| | | |`hipblasDrot`|3.0.0| | | |`rocblas_drot`|3.5.0| | | |
|`cublasDrot_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDrotg`| | | |`hipblasDrotg`|3.0.0| | | |`rocblas_drotg`|3.5.0| | | |
|`cublasDrotg_v2`| | | |`hipblasDrotg`|3.0.0| | | |`rocblas_drotg`|3.5.0| | | |
|`cublasDrotm`| | | |`hipblasDrotm`|3.0.0| | | |`rocblas_drotm`|3.5.0| | | |
|`cublasDrotm_64`|12.0| | | | | | | | | | | | |
|`cublasDrotm_v2`| | | |`hipblasDrotm`|3.0.0| | | |`rocblas_drotm`|3.5.0| | | |
|`cublasDrotm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDrotmg`| | | |`hipblasDrotmg`|3.0.0| | | |`rocblas_drotmg`|3.5.0| | | |
|`cublasDrotmg_v2`| | | |`hipblasDrotmg`|3.0.0| | | |`rocblas_drotmg`|3.5.0| | | |
|`cublasDscal`| | | |`hipblasDscal`|1.8.2| | | |`rocblas_dscal`|1.5.0| | | |
|`cublasDscal_64`|12.0| | | | | | | | | | | | |
|`cublasDscal_v2`| | | |`hipblasDscal`|1.8.2| | | |`rocblas_dscal`|1.5.0| | | |
|`cublasDscal_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDswap`| | | |`hipblasDswap`|3.0.0| | | |`rocblas_dswap`|1.5.0| | | |
|`cublasDswap_64`|12.0| | | | | | | | | | | | |
|`cublasDswap_v2`| | | |`hipblasDswap`|3.0.0| | | |`rocblas_dswap`|1.5.0| | | |
|`cublasDswap_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDzasum`| | | |`hipblasDzasum`|3.0.0| | | |`rocblas_dzasum`|1.5.0| | | |
|`cublasDzasum_64`|12.0| | | | | | | | | | | | |
|`cublasDzasum_v2`| | | |`hipblasDzasum`|3.0.0| | | |`rocblas_dzasum`|1.5.0| | | |
|`cublasDzasum_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDznrm2`| | | |`hipblasDznrm2`|3.0.0| | | |`rocblas_dznrm2`|1.5.0| | | |
|`cublasDznrm2_64`|12.0| | | | | | | | | | | | |
|`cublasDznrm2_v2`| | | |`hipblasDznrm2`|3.0.0| | | |`rocblas_dznrm2`|1.5.0| | | |
|`cublasDznrm2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasIcamax`| | | |`hipblasIcamax`|3.0.0| | | |`rocblas_icamax`|3.5.0| | | |
|`cublasIcamax_64`|12.0| | | | | | | | | | | | |
|`cublasIcamax_v2`| | | |`hipblasIcamax`|3.0.0| | | |`rocblas_icamax`|3.5.0| | | |
|`cublasIcamax_v2_64`|12.0| | | | | | | | | | | | |
|`cublasIcamin`| | | |`hipblasIcamin`|3.0.0| | | |`rocblas_icamin`|3.5.0| | | |
|`cublasIcamin_64`|12.0| | | | | | | | | | | | |
|`cublasIcamin_v2`| | | |`hipblasIcamin`|3.0.0| | | |`rocblas_icamin`|3.5.0| | | |
|`cublasIcamin_v2_64`|12.0| | | | | | | | | | | | |
|`cublasIdamax`| | | |`hipblasIdamax`|1.8.2| | | |`rocblas_idamax`|1.6.4| | | |
|`cublasIdamax_64`|12.0| | | | | | | | | | | | |
|`cublasIdamax_v2`| | | |`hipblasIdamax`|1.8.2| | | |`rocblas_idamax`|1.6.4| | | |
|`cublasIdamax_v2_64`|12.0| | | | | | | | | | | | |
|`cublasIdamin`| | | |`hipblasIdamin`|3.0.0| | | |`rocblas_idamin`|1.6.4| | | |
|`cublasIdamin_64`|12.0| | | | | | | | | | | | |
|`cublasIdamin_v2`| | | |`hipblasIdamin`|3.0.0| | | |`rocblas_idamin`|1.6.4| | | |
|`cublasIdamin_v2_64`|12.0| | | | | | | | | | | | |
|`cublasIsamax`| | | |`hipblasIsamax`|1.8.2| | | |`rocblas_isamax`|1.6.4| | | |
|`cublasIsamax_64`|12.0| | | | | | | | | | | | |
|`cublasIsamax_v2`| | | |`hipblasIsamax`|1.8.2| | | |`rocblas_isamax`|1.6.4| | | |
|`cublasIsamax_v2_64`|12.0| | | | | | | | | | | | |
|`cublasIsamin`| | | |`hipblasIsamin`|3.0.0| | | |`rocblas_isamin`|1.6.4| | | |
|`cublasIsamin_64`|12.0| | | | | | | | | | | | |
|`cublasIsamin_v2`| | | |`hipblasIsamin`|3.0.0| | | |`rocblas_isamin`|1.6.4| | | |
|`cublasIsamin_v2_64`|12.0| | | | | | | | | | | | |
|`cublasIzamax`| | | |`hipblasIzamax`|3.0.0| | | |`rocblas_izamax`|3.5.0| | | |
|`cublasIzamax_64`|12.0| | | | | | | | | | | | |
|`cublasIzamax_v2`| | | |`hipblasIzamax`|3.0.0| | | |`rocblas_izamax`|3.5.0| | | |
|`cublasIzamax_v2_64`|12.0| | | | | | | | | | | | |
|`cublasIzamin`| | | |`hipblasIzamin`|3.0.0| | | |`rocblas_izamin`|3.5.0| | | |
|`cublasIzamin_64`|12.0| | | | | | | | | | | | |
|`cublasIzamin_v2`| | | |`hipblasIzamin`|3.0.0| | | |`rocblas_izamin`|3.5.0| | | |
|`cublasIzamin_v2_64`|12.0| | | | | | | | | | | | |
|`cublasNrm2Ex`|8.0| | |`hipblasNrm2Ex`|4.1.0| | | |`rocblas_nrm2_ex`|4.1.0| | | |
|`cublasNrm2Ex_64`|12.0| | | | | | | | | | | | |
|`cublasSasum`| | | |`hipblasSasum`|1.8.2| | | |`rocblas_sasum`|1.5.0| | | |
|`cublasSasum_64`|12.0| | | | | | | | | | | | |
|`cublasSasum_v2`| | | |`hipblasSasum`|1.8.2| | | |`rocblas_sasum`|1.5.0| | | |
|`cublasSasum_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSaxpy`| | | |`hipblasSaxpy`|1.8.2| | | |`rocblas_saxpy`|1.5.0| | | |
|`cublasSaxpy_64`|12.0| | | | | | | | | | | | |
|`cublasSaxpy_v2`| | | |`hipblasSaxpy`|1.8.2| | | |`rocblas_saxpy`|1.5.0| | | |
|`cublasSaxpy_v2_64`|12.0| | | | | | | | | | | | |
|`cublasScasum`| | | |`hipblasScasum`|3.0.0| | | |`rocblas_scasum`|1.5.0| | | |
|`cublasScasum_64`|12.0| | | | | | | | | | | | |
|`cublasScasum_v2`| | | |`hipblasScasum`|3.0.0| | | |`rocblas_scasum`|1.5.0| | | |
|`cublasScasum_v2_64`|12.0| | | | | | | | | | | | |
|`cublasScnrm2`| | | |`hipblasScnrm2`|3.0.0| | | |`rocblas_scnrm2`|1.5.0| | | |
|`cublasScnrm2_64`|12.0| | | | | | | | | | | | |
|`cublasScnrm2_v2`| | | |`hipblasScnrm2`|3.0.0| | | |`rocblas_scnrm2`|1.5.0| | | |
|`cublasScnrm2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasScopy`| | | |`hipblasScopy`|1.8.2| | | |`rocblas_scopy`|1.5.0| | | |
|`cublasScopy_64`|12.0| | | | | | | | | | | | |
|`cublasScopy_v2`| | | |`hipblasScopy`|1.8.2| | | |`rocblas_scopy`|1.5.0| | | |
|`cublasScopy_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSdot`| | | |`hipblasSdot`|3.0.0| | | |`rocblas_sdot`|1.5.0| | | |
|`cublasSdot_64`|12.0| | | | | | | | | | | | |
|`cublasSdot_v2`| | | |`hipblasSdot`|3.0.0| | | |`rocblas_sdot`|1.5.0| | | |
|`cublasSdot_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSnrm2`| | | |`hipblasSnrm2`|1.8.2| | | |`rocblas_snrm2`|1.5.0| | | |
|`cublasSnrm2_64`|12.0| | | | | | | | | | | | |
|`cublasSnrm2_v2`| | | |`hipblasSnrm2`|1.8.2| | | |`rocblas_snrm2`|1.5.0| | | |
|`cublasSnrm2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSrot`| | | |`hipblasSrot`|3.0.0| | | |`rocblas_srot`|3.5.0| | | |
|`cublasSrot_64`|12.0| | | | | | | | | | | | |
|`cublasSrot_v2`| | | |`hipblasSrot`|3.0.0| | | |`rocblas_srot`|3.5.0| | | |
|`cublasSrot_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSrotg`| | | |`hipblasSrotg`|3.0.0| | | |`rocblas_srotg`|3.5.0| | | |
|`cublasSrotg_v2`| | | |`hipblasSrotg`|3.0.0| | | |`rocblas_srotg`|3.5.0| | | |
|`cublasSrotm`| | | |`hipblasSrotm`|3.0.0| | | |`rocblas_srotm`|3.5.0| | | |
|`cublasSrotm_64`|12.0| | | | | | | | | | | | |
|`cublasSrotm_v2`| | | |`hipblasSrotm`|3.0.0| | | |`rocblas_srotm`|3.5.0| | | |
|`cublasSrotm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSrotmg`| | | |`hipblasSrotmg`|3.0.0| | | |`rocblas_srotmg`|3.5.0| | | |
|`cublasSrotmg_v2`| | | |`hipblasSrotmg`|3.0.0| | | |`rocblas_srotmg`|3.5.0| | | |
|`cublasSscal`| | | |`hipblasSscal`|1.8.2| | | |`rocblas_sscal`|1.5.0| | | |
|`cublasSscal_64`|12.0| | | | | | | | | | | | |
|`cublasSscal_v2`| | | |`hipblasSscal`|1.8.2| | | |`rocblas_sscal`|1.5.0| | | |
|`cublasSscal_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSswap`| | | |`hipblasSswap`|3.0.0| | | |`rocblas_sswap`|1.5.0| | | |
|`cublasSswap_64`|12.0| | | | | | | | | | | | |
|`cublasSswap_v2`| | | |`hipblasSswap`|3.0.0| | | |`rocblas_sswap`|1.5.0| | | |
|`cublasSswap_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZaxpy`| | | |`hipblasZaxpy`|3.0.0| | | |`rocblas_zaxpy`|1.5.0| | | |
|`cublasZaxpy_64`|12.0| | | | | | | | | | | | |
|`cublasZaxpy_v2`| | | |`hipblasZaxpy`|3.0.0| | | |`rocblas_zaxpy`|1.5.0| | | |
|`cublasZaxpy_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZcopy`| | | |`hipblasZcopy`|3.0.0| | | |`rocblas_zcopy`|1.5.0| | | |
|`cublasZcopy_64`|12.0| | | | | | | | | | | | |
|`cublasZcopy_v2`| | | |`hipblasZcopy`|3.0.0| | | |`rocblas_zcopy`|1.5.0| | | |
|`cublasZcopy_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZdotc`| | | |`hipblasZdotc`|3.0.0| | | |`rocblas_zdotc`|3.5.0| | | |
|`cublasZdotc_64`|12.0| | | | | | | | | | | | |
|`cublasZdotc_v2`| | | |`hipblasZdotc`|3.0.0| | | |`rocblas_zdotc`|3.5.0| | | |
|`cublasZdotc_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZdotu`| | | |`hipblasZdotu`|3.0.0| | | |`rocblas_zdotu`|1.5.0| | | |
|`cublasZdotu_64`|12.0| | | | | | | | | | | | |
|`cublasZdotu_v2`| | | |`hipblasZdotu`|3.0.0| | | |`rocblas_zdotu`|1.5.0| | | |
|`cublasZdotu_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZdrot`| | | |`hipblasZdrot`|3.0.0| | | |`rocblas_zdrot`|3.5.0| | | |
|`cublasZdrot_64`|12.0| | | | | | | | | | | | |
|`cublasZdrot_v2`| | | |`hipblasZdrot`|3.0.0| | | |`rocblas_zdrot`|3.5.0| | | |
|`cublasZdrot_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZdscal`| | | |`hipblasZdscal`|3.0.0| | | |`rocblas_zdscal`|3.5.0| | | |
|`cublasZdscal_64`|12.0| | | | | | | | | | | | |
|`cublasZdscal_v2`| | | |`hipblasZdscal`|3.0.0| | | |`rocblas_zdscal`|3.5.0| | | |
|`cublasZdscal_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZrot`| | | |`hipblasZrot`|3.0.0| | | |`rocblas_zrot`|3.5.0| | | |
|`cublasZrot_64`|12.0| | | | | | | | | | | | |
|`cublasZrot_v2`| | | |`hipblasZrot`|3.0.0| | | |`rocblas_zrot`|3.5.0| | | |
|`cublasZrot_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZrotg`| | | |`hipblasZrotg`|3.0.0| | | |`rocblas_zrotg`|3.5.0| | | |
|`cublasZrotg_v2`| | | |`hipblasZrotg`|3.0.0| | | |`rocblas_zrotg`|3.5.0| | | |
|`cublasZscal`| | | |`hipblasZscal`|1.8.2| | | |`rocblas_zscal`|1.5.0| | | |
|`cublasZscal_64`|12.0| | | | | | | | | | | | |
|`cublasZscal_v2`| | | |`hipblasZscal`|1.8.2| | | |`rocblas_zscal`|1.5.0| | | |
|`cublasZscal_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZswap`| | | |`hipblasZswap`|3.0.0| | | |`rocblas_zswap`|1.5.0| | | |
|`cublasZswap_64`|12.0| | | | | | | | | | | | |
|`cublasZswap_v2`| | | |`hipblasZswap`|3.0.0| | | |`rocblas_zswap`|1.5.0| | | |
|`cublasZswap_v2_64`|12.0| | | | | | | | | | | | |

## **6. CUBLAS Level-2 Function Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|**E**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cublasCgbmv`| | | |`hipblasCgbmv`|3.5.0| | | |`rocblas_cgbmv`|3.5.0| | | |
|`cublasCgbmv_64`|12.0| | | | | | | | | | | | |
|`cublasCgbmv_v2`| | | |`hipblasCgbmv`|3.5.0| | | |`rocblas_cgbmv`|3.5.0| | | |
|`cublasCgbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCgemv`| | | |`hipblasCgemv`|3.0.0| | | |`rocblas_cgemv`|1.5.0| | | |
|`cublasCgemv_64`|12.0| | | | | | | | | | | | |
|`cublasCgemv_v2`| | | |`hipblasCgemv`|3.0.0| | | |`rocblas_cgemv`|1.5.0| | | |
|`cublasCgemv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCgerc`| | | |`hipblasCgerc`|3.5.0| | | |`rocblas_cgerc`|3.5.0| | | |
|`cublasCgerc_64`|12.0| | | | | | | | | | | | |
|`cublasCgerc_v2`| | | |`hipblasCgerc`|3.5.0| | | |`rocblas_cgerc`|3.5.0| | | |
|`cublasCgerc_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCgeru`| | | |`hipblasCgeru`|3.5.0| | | |`rocblas_cgeru`|3.5.0| | | |
|`cublasCgeru_64`|12.0| | | | | | | | | | | | |
|`cublasCgeru_v2`| | | |`hipblasCgeru`|3.5.0| | | |`rocblas_cgeru`|3.5.0| | | |
|`cublasCgeru_v2_64`|12.0| | | | | | | | | | | | |
|`cublasChbmv`| | | |`hipblasChbmv`|3.5.0| | | |`rocblas_chbmv`|3.5.0| | | |
|`cublasChbmv_64`|12.0| | | | | | | | | | | | |
|`cublasChbmv_v2`| | | |`hipblasChbmv`|3.5.0| | | |`rocblas_chbmv`|3.5.0| | | |
|`cublasChbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasChemv`| | | |`hipblasChemv`|3.5.0| | | |`rocblas_chemv`|1.5.0| | | |
|`cublasChemv_64`|12.0| | | | | | | | | | | | |
|`cublasChemv_v2`| | | |`hipblasChemv`|3.5.0| | | |`rocblas_chemv`|1.5.0| | | |
|`cublasChemv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCher`| | | |`hipblasCher`|3.5.0| | | |`rocblas_cher`|3.5.0| | | |
|`cublasCher2`| | | |`hipblasCher2`|3.5.0| | | |`rocblas_cher2`|3.5.0| | | |
|`cublasCher2_64`|12.0| | | | | | | | | | | | |
|`cublasCher2_v2`| | | |`hipblasCher2`|3.5.0| | | |`rocblas_cher2`|3.5.0| | | |
|`cublasCher2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCher_64`|12.0| | | | | | | | | | | | |
|`cublasCher_v2`| | | |`hipblasCher`|3.5.0| | | |`rocblas_cher`|3.5.0| | | |
|`cublasCher_v2_64`|12.0| | | | | | | | | | | | |
|`cublasChpmv`| | | |`hipblasChpmv`|3.5.0| | | |`rocblas_chpmv`|3.5.0| | | |
|`cublasChpmv_64`|12.0| | | | | | | | | | | | |
|`cublasChpmv_v2`| | | |`hipblasChpmv`|3.5.0| | | |`rocblas_chpmv`|3.5.0| | | |
|`cublasChpmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasChpr`| | | |`hipblasChpr`|3.5.0| | | |`rocblas_chpr`|3.5.0| | | |
|`cublasChpr2`| | | |`hipblasChpr2`|3.5.0| | | |`rocblas_chpr2`|3.5.0| | | |
|`cublasChpr2_64`|12.0| | | | | | | | | | | | |
|`cublasChpr2_v2`| | | |`hipblasChpr2`|3.5.0| | | |`rocblas_chpr2`|3.5.0| | | |
|`cublasChpr2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasChpr_64`|12.0| | | | | | | | | | | | |
|`cublasChpr_v2`| | | |`hipblasChpr`|3.5.0| | | |`rocblas_chpr`|3.5.0| | | |
|`cublasChpr_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCsymv`| | | |`hipblasCsymv`|3.5.0| | | |`rocblas_csymv`|3.5.0| | | |
|`cublasCsymv_64`|12.0| | | | | | | | | | | | |
|`cublasCsymv_v2`| | | |`hipblasCsymv`|3.5.0| | | |`rocblas_csymv`|3.5.0| | | |
|`cublasCsymv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCsyr`| | | |`hipblasCsyr`|3.5.0| | | |`rocblas_csyr`|1.7.1| | | |
|`cublasCsyr2`| | | |`hipblasCsyr2`|3.5.0| | | |`rocblas_csyr2`|3.5.0| | | |
|`cublasCsyr2_64`|12.0| | | | | | | | | | | | |
|`cublasCsyr2_v2`| | | |`hipblasCsyr2`|3.5.0| | | |`rocblas_csyr2`|3.5.0| | | |
|`cublasCsyr2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCsyr_64`|12.0| | | | | | | | | | | | |
|`cublasCsyr_v2`| | | |`hipblasCsyr`|3.5.0| | | |`rocblas_csyr`|1.7.1| | | |
|`cublasCsyr_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCtbmv`| | | |`hipblasCtbmv`|3.5.0| | | |`rocblas_ctbmv`|3.5.0| | | |
|`cublasCtbmv_64`|12.0| | | | | | | | | | | | |
|`cublasCtbmv_v2`| | | |`hipblasCtbmv`|3.5.0| | | |`rocblas_ctbmv`|3.5.0| | | |
|`cublasCtbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCtbsv`| | | |`hipblasCtbsv`|3.6.0| | | |`rocblas_ctbsv`|3.5.0| | | |
|`cublasCtbsv_64`|12.0| | | | | | | | | | | | |
|`cublasCtbsv_v2`| | | |`hipblasCtbsv`|3.6.0| | | |`rocblas_ctbsv`|3.5.0| | | |
|`cublasCtbsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCtpmv`| | | |`hipblasCtpmv`|3.5.0| | | |`rocblas_ctpmv`|3.5.0| | | |
|`cublasCtpmv_64`|12.0| | | | | | | | | | | | |
|`cublasCtpmv_v2`| | | |`hipblasCtpmv`|3.5.0| | | |`rocblas_ctpmv`|3.5.0| | | |
|`cublasCtpmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCtpsv`| | | |`hipblasCtpsv`|3.5.0| | | |`rocblas_ctpsv`|3.5.0| | | |
|`cublasCtpsv_64`|12.0| | | | | | | | | | | | |
|`cublasCtpsv_v2`| | | |`hipblasCtpsv`|3.5.0| | | |`rocblas_ctpsv`|3.5.0| | | |
|`cublasCtpsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCtrmv`| | | |`hipblasCtrmv`|3.5.0| | | |`rocblas_ctrmv`|3.5.0| | | |
|`cublasCtrmv_64`|12.0| | | | | | | | | | | | |
|`cublasCtrmv_v2`| | | |`hipblasCtrmv`|3.5.0| | | |`rocblas_ctrmv`|3.5.0| | | |
|`cublasCtrmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCtrsv`| | | |`hipblasCtrsv`|3.5.0| | | |`rocblas_ctrsv`|3.5.0| | | |
|`cublasCtrsv_64`|12.0| | | | | | | | | | | | |
|`cublasCtrsv_v2`| | | |`hipblasCtrsv`|3.5.0| | | |`rocblas_ctrsv`|3.5.0| | | |
|`cublasCtrsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDgbmv`| | | |`hipblasDgbmv`|3.5.0| | | |`rocblas_dgbmv`|3.5.0| | | |
|`cublasDgbmv_64`|12.0| | | | | | | | | | | | |
|`cublasDgbmv_v2`| | | |`hipblasDgbmv`|3.5.0| | | |`rocblas_dgbmv`|3.5.0| | | |
|`cublasDgbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDgemv`| | | |`hipblasDgemv`|1.8.2| | | |`rocblas_dgemv`|1.5.0| | | |
|`cublasDgemv_64`|12.0| | | | | | | | | | | | |
|`cublasDgemv_v2`| | | |`hipblasDgemv`|1.8.2| | | |`rocblas_dgemv`|1.5.0| | | |
|`cublasDgemv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDger`| | | |`hipblasDger`|1.8.2| | | |`rocblas_dger`|1.5.0| | | |
|`cublasDger_64`|12.0| | | | | | | | | | | | |
|`cublasDger_v2`| | | |`hipblasDger`|1.8.2| | | |`rocblas_dger`|1.5.0| | | |
|`cublasDger_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDsbmv`| | | |`hipblasDsbmv`|3.5.0| | | |`rocblas_dsbmv`|3.5.0| | | |
|`cublasDsbmv_64`|12.0| | | | | | | | | | | | |
|`cublasDsbmv_v2`| | | |`hipblasDsbmv`|3.5.0| | | |`rocblas_dsbmv`|3.5.0| | | |
|`cublasDsbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDspmv`| | | |`hipblasDspmv`|3.5.0| | | |`rocblas_dspmv`|3.5.0| | | |
|`cublasDspmv_64`|12.0| | | | | | | | | | | | |
|`cublasDspmv_v2`| | | |`hipblasDspmv`|3.5.0| | | |`rocblas_dspmv`|3.5.0| | | |
|`cublasDspmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDspr`| | | |`hipblasDspr`|3.5.0| | | |`rocblas_dspr`|3.5.0| | | |
|`cublasDspr2`| | | |`hipblasDspr2`|3.5.0| | | |`rocblas_dspr2`|3.5.0| | | |
|`cublasDspr2_64`|12.0| | | | | | | | | | | | |
|`cublasDspr2_v2`| | | |`hipblasDspr2`|3.5.0| | | |`rocblas_dspr2`|3.5.0| | | |
|`cublasDspr2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDspr_64`|12.0| | | | | | | | | | | | |
|`cublasDspr_v2`| | | |`hipblasDspr`|3.5.0| | | |`rocblas_dspr`|3.5.0| | | |
|`cublasDspr_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDsymv`| | | |`hipblasDsymv`|3.5.0| | | |`rocblas_dsymv`|1.5.0| | | |
|`cublasDsymv_64`|12.0| | | | | | | | | | | | |
|`cublasDsymv_v2`| | | |`hipblasDsymv`|3.5.0| | | |`rocblas_dsymv`|1.5.0| | | |
|`cublasDsymv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDsyr`| | | |`hipblasDsyr`|3.0.0| | | |`rocblas_dsyr`|1.7.1| | | |
|`cublasDsyr2`| | | |`hipblasDsyr2`|3.5.0| | | |`rocblas_dsyr2`|3.5.0| | | |
|`cublasDsyr2_64`|12.0| | | | | | | | | | | | |
|`cublasDsyr2_v2`| | | |`hipblasDsyr2`|3.5.0| | | |`rocblas_dsyr2`|3.5.0| | | |
|`cublasDsyr2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDsyr_64`|12.0| | | | | | | | | | | | |
|`cublasDsyr_v2`| | | |`hipblasDsyr`|3.0.0| | | |`rocblas_dsyr`|1.7.1| | | |
|`cublasDsyr_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDtbmv`| | | |`hipblasDtbmv`|3.5.0| | | |`rocblas_dtbmv`|3.5.0| | | |
|`cublasDtbmv_64`|12.0| | | | | | | | | | | | |
|`cublasDtbmv_v2`| | | |`hipblasDtbmv`|3.5.0| | | |`rocblas_dtbmv`|3.5.0| | | |
|`cublasDtbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDtbsv`| | | |`hipblasDtbsv`|3.6.0| | | |`rocblas_dtbsv`|3.5.0| | | |
|`cublasDtbsv_64`|12.0| | | | | | | | | | | | |
|`cublasDtbsv_v2`| | | |`hipblasDtbsv`|3.6.0| | | |`rocblas_dtbsv`|3.5.0| | | |
|`cublasDtbsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDtpmv`| | | |`hipblasDtpmv`|3.5.0| | | |`rocblas_dtpmv`|3.5.0| | | |
|`cublasDtpmv_64`|12.0| | | | | | | | | | | | |
|`cublasDtpmv_v2`| | | |`hipblasDtpmv`|3.5.0| | | |`rocblas_dtpmv`|3.5.0| | | |
|`cublasDtpmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDtpsv`| | | |`hipblasDtpsv`|3.5.0| | | |`rocblas_dtpsv`|3.5.0| | | |
|`cublasDtpsv_64`|12.0| | | | | | | | | | | | |
|`cublasDtpsv_v2`| | | |`hipblasDtpsv`|3.5.0| | | |`rocblas_dtpsv`|3.5.0| | | |
|`cublasDtpsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDtrmv`| | | |`hipblasDtrmv`|3.5.0| | | |`rocblas_dtrmv`|3.5.0| | | |
|`cublasDtrmv_64`|12.0| | | | | | | | | | | | |
|`cublasDtrmv_v2`| | | |`hipblasDtrmv`|3.5.0| | | |`rocblas_dtrmv`|3.5.0| | | |
|`cublasDtrmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDtrsv`| | | |`hipblasDtrsv`|3.0.0| | | |`rocblas_dtrsv`|3.5.0| | | |
|`cublasDtrsv_64`|12.0| | | | | | | | | | | | |
|`cublasDtrsv_v2`| | | |`hipblasDtrsv`|3.0.0| | | |`rocblas_dtrsv`|3.5.0| | | |
|`cublasDtrsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSgbmv`| | | |`hipblasSgbmv`|3.5.0| | | |`rocblas_sgbmv`|3.5.0| | | |
|`cublasSgbmv_64`|12.0| | | | | | | | | | | | |
|`cublasSgbmv_v2`| | | |`hipblasSgbmv`|3.5.0| | | |`rocblas_sgbmv`|3.5.0| | | |
|`cublasSgbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSgemv`| | | |`hipblasSgemv`|1.8.2| | | |`rocblas_sgemv`|1.5.0| | | |
|`cublasSgemv_64`|12.0| | | | | | | | | | | | |
|`cublasSgemv_v2`| | | |`hipblasSgemv`|1.8.2| | | |`rocblas_sgemv`|1.5.0| | | |
|`cublasSgemv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSger`| | | |`hipblasSger`|1.8.2| | | |`rocblas_sger`|1.5.0| | | |
|`cublasSger_64`|12.0| | | | | | | | | | | | |
|`cublasSger_v2`| | | |`hipblasSger`|1.8.2| | | |`rocblas_sger`|1.5.0| | | |
|`cublasSger_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSsbmv`| | | |`hipblasSsbmv`|3.5.0| | | |`rocblas_ssbmv`|3.5.0| | | |
|`cublasSsbmv_64`|12.0| | | | | | | | | | | | |
|`cublasSsbmv_v2`| | | |`hipblasSsbmv`|3.5.0| | | |`rocblas_ssbmv`|3.5.0| | | |
|`cublasSsbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSspmv`| | | |`hipblasSspmv`|3.5.0| | | |`rocblas_sspmv`|3.5.0| | | |
|`cublasSspmv_64`|12.0| | | | | | | | | | | | |
|`cublasSspmv_v2`| | | |`hipblasSspmv`|3.5.0| | | |`rocblas_sspmv`|3.5.0| | | |
|`cublasSspmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSspr`| | | |`hipblasSspr`|3.5.0| | | |`rocblas_sspr`|3.5.0| | | |
|`cublasSspr2`| | | |`hipblasSspr2`|3.5.0| | | |`rocblas_sspr2`|3.5.0| | | |
|`cublasSspr2_64`|12.0| | | | | | | | | | | | |
|`cublasSspr2_v2`| | | |`hipblasSspr2`|3.5.0| | | |`rocblas_sspr2`|3.5.0| | | |
|`cublasSspr2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSspr_64`|12.0| | | | | | | | | | | | |
|`cublasSspr_v2`| | | |`hipblasSspr`|3.5.0| | | |`rocblas_sspr`|3.5.0| | | |
|`cublasSspr_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSsymv`| | | |`hipblasSsymv`|3.5.0| | | |`rocblas_ssymv`|1.5.0| | | |
|`cublasSsymv_64`|12.0| | | | | | | | | | | | |
|`cublasSsymv_v2`| | | |`hipblasSsymv`|3.5.0| | | |`rocblas_ssymv`|1.5.0| | | |
|`cublasSsymv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSsyr`| | | |`hipblasSsyr`|3.0.0| | | |`rocblas_ssyr`|1.7.1| | | |
|`cublasSsyr2`| | | |`hipblasSsyr2`|3.5.0| | | |`rocblas_ssyr2`|3.5.0| | | |
|`cublasSsyr2_64`|12.0| | | | | | | | | | | | |
|`cublasSsyr2_v2`| | | |`hipblasSsyr2`|3.5.0| | | |`rocblas_ssyr2`|3.5.0| | | |
|`cublasSsyr2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSsyr_64`|12.0| | | | | | | | | | | | |
|`cublasSsyr_v2`| | | |`hipblasSsyr`|3.0.0| | | |`rocblas_ssyr`|1.7.1| | | |
|`cublasSsyr_v2_64`|12.0| | | | | | | | | | | | |
|`cublasStbmv`| | | |`hipblasStbmv`|3.5.0| | | |`rocblas_stbmv`|3.5.0| | | |
|`cublasStbmv_64`|12.0| | | | | | | | | | | | |
|`cublasStbmv_v2`| | | |`hipblasStbmv`|3.5.0| | | |`rocblas_stbmv`|3.5.0| | | |
|`cublasStbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasStbsv`| | | |`hipblasStbsv`|3.6.0| | | |`rocblas_stbsv`|3.5.0| | | |
|`cublasStbsv_64`|12.0| | | | | | | | | | | | |
|`cublasStbsv_v2`| | | |`hipblasStbsv`|3.6.0| | | |`rocblas_stbsv`|3.5.0| | | |
|`cublasStbsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasStpmv`| | | |`hipblasStpmv`|3.5.0| | | |`rocblas_stpmv`|3.5.0| | | |
|`cublasStpmv_64`|12.0| | | | | | | | | | | | |
|`cublasStpmv_v2`| | | |`hipblasStpmv`|3.5.0| | | |`rocblas_stpmv`|3.5.0| | | |
|`cublasStpmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasStpsv`| | | |`hipblasStpsv`|3.5.0| | | |`rocblas_stpsv`|3.5.0| | | |
|`cublasStpsv_64`|12.0| | | | | | | | | | | | |
|`cublasStpsv_v2`| | | |`hipblasStpsv`|3.5.0| | | |`rocblas_stpsv`|3.5.0| | | |
|`cublasStpsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasStrmv`| | | |`hipblasStrmv`|3.5.0| | | |`rocblas_strmv`|3.5.0| | | |
|`cublasStrmv_64`|12.0| | | | | | | | | | | | |
|`cublasStrmv_v2`| | | |`hipblasStrmv`|3.5.0| | | |`rocblas_strmv`|3.5.0| | | |
|`cublasStrmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasStrsv`| | | |`hipblasStrsv`|3.0.0| | | |`rocblas_strsv`|3.5.0| | | |
|`cublasStrsv_64`|12.0| | | | | | | | | | | | |
|`cublasStrsv_v2`| | | |`hipblasStrsv`|3.0.0| | | |`rocblas_strsv`|3.5.0| | | |
|`cublasStrsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZgbmv`| | | |`hipblasZgbmv`|3.5.0| | | |`rocblas_zgbmv`|3.5.0| | | |
|`cublasZgbmv_64`|12.0| | | | | | | | | | | | |
|`cublasZgbmv_v2`| | | |`hipblasZgbmv`|3.5.0| | | |`rocblas_zgbmv`|3.5.0| | | |
|`cublasZgbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZgemv`| | | |`hipblasZgemv`|3.0.0| | | |`rocblas_zgemv`|1.5.0| | | |
|`cublasZgemv_64`|12.0| | | | | | | | | | | | |
|`cublasZgemv_v2`| | | |`hipblasZgemv`|3.0.0| | | |`rocblas_zgemv`|1.5.0| | | |
|`cublasZgemv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZgerc`| | | |`hipblasZgerc`|3.5.0| | | |`rocblas_zgerc`|3.5.0| | | |
|`cublasZgerc_64`|12.0| | | | | | | | | | | | |
|`cublasZgerc_v2`| | | |`hipblasZgerc`|3.5.0| | | |`rocblas_zgerc`|3.5.0| | | |
|`cublasZgerc_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZgeru`| | | |`hipblasZgeru`|3.5.0| | | |`rocblas_zgeru`|3.5.0| | | |
|`cublasZgeru_64`|12.0| | | | | | | | | | | | |
|`cublasZgeru_v2`| | | |`hipblasZgeru`|3.5.0| | | |`rocblas_zgeru`|3.5.0| | | |
|`cublasZgeru_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZhbmv`| | | |`hipblasZhbmv`|3.5.0| | | |`rocblas_zhbmv`|3.5.0| | | |
|`cublasZhbmv_64`|12.0| | | | | | | | | | | | |
|`cublasZhbmv_v2`| | | |`hipblasZhbmv`|3.5.0| | | |`rocblas_zhbmv`|3.5.0| | | |
|`cublasZhbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZhemv`| | | |`hipblasZhemv`|3.5.0| | | |`rocblas_zhemv`|1.5.0| | | |
|`cublasZhemv_64`|12.0| | | | | | | | | | | | |
|`cublasZhemv_v2`| | | |`hipblasZhemv`|3.5.0| | | |`rocblas_zhemv`|1.5.0| | | |
|`cublasZhemv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZher`| | | |`hipblasZher`|3.5.0| | | |`rocblas_zher`|3.5.0| | | |
|`cublasZher2`| | | |`hipblasZher2`|3.5.0| | | |`rocblas_zher2`|3.5.0| | | |
|`cublasZher2_64`|12.0| | | | | | | | | | | | |
|`cublasZher2_v2`| | | |`hipblasZher2`|3.5.0| | | |`rocblas_zher2`|3.5.0| | | |
|`cublasZher2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZher_64`|12.0| | | | | | | | | | | | |
|`cublasZher_v2`| | | |`hipblasZher`|3.5.0| | | |`rocblas_zher`|3.5.0| | | |
|`cublasZher_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZhpmv`| | | |`hipblasZhpmv`|3.5.0| | | |`rocblas_zhpmv`|3.5.0| | | |
|`cublasZhpmv_64`|12.0| | | | | | | | | | | | |
|`cublasZhpmv_v2`| | | |`hipblasZhpmv`|3.5.0| | | |`rocblas_zhpmv`|3.5.0| | | |
|`cublasZhpmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZhpr`| | | |`hipblasZhpr`|3.5.0| | | |`rocblas_zhpr`|3.5.0| | | |
|`cublasZhpr2`| | | |`hipblasZhpr2`|3.5.0| | | |`rocblas_zhpr2`|3.5.0| | | |
|`cublasZhpr2_64`|12.0| | | | | | | | | | | | |
|`cublasZhpr2_v2`| | | |`hipblasZhpr2`|3.5.0| | | |`rocblas_zhpr2`|3.5.0| | | |
|`cublasZhpr2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZhpr_64`|12.0| | | | | | | | | | | | |
|`cublasZhpr_v2`| | | |`hipblasZhpr`|3.5.0| | | |`rocblas_zhpr`|3.5.0| | | |
|`cublasZhpr_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZsymv`| | | |`hipblasZsymv`|3.5.0| | | |`rocblas_zsymv`|3.5.0| | | |
|`cublasZsymv_64`|12.0| | | | | | | | | | | | |
|`cublasZsymv_v2`| | | |`hipblasZsymv`|3.5.0| | | |`rocblas_zsymv`|3.5.0| | | |
|`cublasZsymv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZsyr`| | | |`hipblasZsyr`|3.5.0| | | |`rocblas_zsyr`|1.7.1| | | |
|`cublasZsyr2`| | | |`hipblasZsyr2`|3.5.0| | | |`rocblas_zsyr2`|3.5.0| | | |
|`cublasZsyr2_64`|12.0| | | | | | | | | | | | |
|`cublasZsyr2_v2`| | | |`hipblasZsyr2`|3.5.0| | | |`rocblas_zsyr2`|3.5.0| | | |
|`cublasZsyr2_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZsyr_64`|12.0| | | | | | | | | | | | |
|`cublasZsyr_v2`| | | |`hipblasZsyr`|3.5.0| | | |`rocblas_zsyr`|1.7.1| | | |
|`cublasZsyr_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZtbmv`| | | |`hipblasZtbmv`|3.5.0| | | |`rocblas_ztbmv`|3.5.0| | | |
|`cublasZtbmv_64`|12.0| | | | | | | | | | | | |
|`cublasZtbmv_v2`| | | |`hipblasZtbmv`|3.5.0| | | |`rocblas_ztbmv`|3.5.0| | | |
|`cublasZtbmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZtbsv`| | | |`hipblasZtbsv`|3.6.0| | | |`rocblas_ztbsv`|3.5.0| | | |
|`cublasZtbsv_64`|12.0| | | | | | | | | | | | |
|`cublasZtbsv_v2`| | | |`hipblasZtbsv`|3.6.0| | | |`rocblas_ztbsv`|3.5.0| | | |
|`cublasZtbsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZtpmv`| | | |`hipblasZtpmv`|3.5.0| | | |`rocblas_ztpmv`|3.5.0| | | |
|`cublasZtpmv_64`|12.0| | | | | | | | | | | | |
|`cublasZtpmv_v2`| | | |`hipblasZtpmv`|3.5.0| | | |`rocblas_ztpmv`|3.5.0| | | |
|`cublasZtpmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZtpsv`| | | |`hipblasZtpsv`|3.5.0| | | |`rocblas_ztpsv`|3.5.0| | | |
|`cublasZtpsv_64`|12.0| | | | | | | | | | | | |
|`cublasZtpsv_v2`| | | |`hipblasZtpsv`|3.5.0| | | |`rocblas_ztpsv`|3.5.0| | | |
|`cublasZtpsv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZtrmv`| | | |`hipblasZtrmv`|3.5.0| | | |`rocblas_ztrmv`|3.5.0| | | |
|`cublasZtrmv_64`|12.0| | | | | | | | | | | | |
|`cublasZtrmv_v2`| | | |`hipblasZtrmv`|3.5.0| | | |`rocblas_ztrmv`|3.5.0| | | |
|`cublasZtrmv_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZtrsv`| | | |`hipblasZtrsv`|3.5.0| | | |`rocblas_ztrsv`|3.5.0| | | |
|`cublasZtrsv_64`|12.0| | | | | | | | | | | | |
|`cublasZtrsv_v2`| | | |`hipblasZtrsv`|3.5.0| | | |`rocblas_ztrsv`|3.5.0| | | |
|`cublasZtrsv_v2_64`|12.0| | | | | | | | | | | | |

## **7. CUBLAS Level-3 Function Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|**E**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cublasCgemm`| | | |`hipblasCgemm`|1.8.2| | | |`rocblas_cgemm`|1.5.0| | | |
|`cublasCgemm3m`|8.0| | | | | | | | | | | | |
|`cublasCgemm3mBatched`|8.0| | | | | | | | | | | | |
|`cublasCgemm3mBatched_64`|12.0| | | | | | | | | | | | |
|`cublasCgemm3mEx`|8.0| | | | | | | | | | | | |
|`cublasCgemm3mEx_64`|12.0| | | | | | | | | | | | |
|`cublasCgemm3mStridedBatched`|8.0| | | | | | | | | | | | |
|`cublasCgemm3mStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasCgemm3m_64`|12.0| | | | | | | | | | | | |
|`cublasCgemmBatched`| | | |`hipblasCgemmBatched`|3.0.0| | | |`rocblas_cgemm_batched`|3.5.0| | | |
|`cublasCgemmBatched_64`|12.0| | | | | | | | | | | | |
|`cublasCgemmStridedBatched`|8.0| | |`hipblasCgemmStridedBatched`|3.0.0| | | |`rocblas_cgemm_strided_batched`|1.5.0| | | |
|`cublasCgemmStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasCgemm_64`|12.0| | | | | | | | | | | | |
|`cublasCgemm_v2`| | | |`hipblasCgemm`|1.8.2| | | |`rocblas_cgemm`|1.5.0| | | |
|`cublasCgemm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCgemvBatched`|11.6| | | | | | | | | | | | |
|`cublasCgemvBatched_64`|12.0| | | | | | | | | | | | |
|`cublasCgemvStridedBatched`|11.6| | | | | | | | | | | | |
|`cublasCgemvStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasChemm`| | | |`hipblasChemm`|3.6.0| | | |`rocblas_chemm`|3.5.0| | | |
|`cublasChemm_64`|12.0| | | | | | | | | | | | |
|`cublasChemm_v2`| | | |`hipblasChemm`|3.6.0| | | |`rocblas_chemm`|3.5.0| | | |
|`cublasChemm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCher2k`| | | |`hipblasCher2k`|3.5.0| | | |`rocblas_cher2k`|3.5.0| | | |
|`cublasCher2k_64`|12.0| | | | | | | | | | | | |
|`cublasCher2k_v2`| | | |`hipblasCher2k`|3.5.0| | | |`rocblas_cher2k`|3.5.0| | | |
|`cublasCher2k_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCherk`| | | |`hipblasCherk`|3.5.0| | | |`rocblas_cherk`|3.5.0| | | |
|`cublasCherk_64`|12.0| | | | | | | | | | | | |
|`cublasCherk_v2`| | | |`hipblasCherk`|3.5.0| | | |`rocblas_cherk`|3.5.0| | | |
|`cublasCherk_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCherkx`| | | |`hipblasCherkx`|3.5.0| | | |`rocblas_cherkx`|3.5.0| | | |
|`cublasCherkx_64`|12.0| | | | | | | | | | | | |
|`cublasCsymm`| | | |`hipblasCsymm`|3.6.0| | | |`rocblas_csymm`|3.5.0| | | |
|`cublasCsymm_64`|12.0| | | | | | | | | | | | |
|`cublasCsymm_v2`| | | |`hipblasCsymm`|3.6.0| | | |`rocblas_csymm`|3.5.0| | | |
|`cublasCsymm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCsyr2k`| | | |`hipblasCsyr2k`|3.5.0| | | |`rocblas_csyr2k`|3.5.0| | | |
|`cublasCsyr2k_64`|12.0| | | | | | | | | | | | |
|`cublasCsyr2k_v2`| | | |`hipblasCsyr2k`|3.5.0| | | |`rocblas_csyr2k`|3.5.0| | | |
|`cublasCsyr2k_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCsyrk`| | | |`hipblasCsyrk`|3.5.0| | | |`rocblas_csyrk`|3.5.0| | | |
|`cublasCsyrk_64`|12.0| | | | | | | | | | | | |
|`cublasCsyrk_v2`| | | |`hipblasCsyrk`|3.5.0| | | |`rocblas_csyrk`|3.5.0| | | |
|`cublasCsyrk_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCsyrkx`| | | |`hipblasCsyrkx`|3.5.0| | | |`rocblas_csyrkx`|3.5.0| | | |
|`cublasCsyrkx_64`|12.0| | | | | | | | | | | | |
|`cublasCtrmm`| | | |`hipblasCtrmm`|3.5.0|5.6.0| | |`rocblas_ctrmm_outofplace`|5.0.0|5.6.0| | |
|`cublasCtrmm_64`|12.0| | | | | | | | | | | | |
|`cublasCtrmm_v2`| | | |`hipblasCtrmm`|3.5.0|5.6.0| | |`rocblas_ctrmm_outofplace`|5.0.0|5.6.0| | |
|`cublasCtrmm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasCtrsm`| | | |`hipblasCtrsm`|3.5.0| | | |`rocblas_ctrsm`|3.5.0| | | |
|`cublasCtrsm_64`|12.0| | | | | | | | | | | | |
|`cublasCtrsm_v2`| | | |`hipblasCtrsm`|3.5.0| | | |`rocblas_ctrsm`|3.5.0| | | |
|`cublasCtrsm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDgemm`| | | |`hipblasDgemm`|1.8.2| | | |`rocblas_dgemm`|1.5.0| | | |
|`cublasDgemmBatched`| | | |`hipblasDgemmBatched`|1.8.2| | | |`rocblas_dgemm_batched`|3.5.0| | | |
|`cublasDgemmBatched_64`|12.0| | | | | | | | | | | | |
|`cublasDgemmStridedBatched`|8.0| | |`hipblasDgemmStridedBatched`|1.8.2| | | |`rocblas_dgemm_strided_batched`|1.5.0| | | |
|`cublasDgemmStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasDgemm_64`|12.0| | | | | | | | | | | | |
|`cublasDgemm_v2`| | | |`hipblasDgemm`|1.8.2| | | |`rocblas_dgemm`|1.5.0| | | |
|`cublasDgemm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDgemvBatched`|11.6| | | | | | | | | | | | |
|`cublasDgemvBatched_64`|12.0| | | | | | | | | | | | |
|`cublasDgemvStridedBatched`|11.6| | | | | | | | | | | | |
|`cublasDgemvStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasDsymm`| | | |`hipblasDsymm`|3.6.0| | | |`rocblas_dsymm`|3.5.0| | | |
|`cublasDsymm_64`|12.0| | | | | | | | | | | | |
|`cublasDsymm_v2`| | | |`hipblasDsymm`|3.6.0| | | |`rocblas_dsymm`|3.5.0| | | |
|`cublasDsymm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDsyr2k`| | | |`hipblasDsyr2k`|3.5.0| | | |`rocblas_dsyr2k`|3.5.0| | | |
|`cublasDsyr2k_64`|12.0| | | | | | | | | | | | |
|`cublasDsyr2k_v2`| | | |`hipblasDsyr2k`|3.5.0| | | |`rocblas_dsyr2k`|3.5.0| | | |
|`cublasDsyr2k_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDsyrk`| | | |`hipblasDsyrk`|3.5.0| | | |`rocblas_dsyrk`|3.5.0| | | |
|`cublasDsyrk_64`|12.0| | | | | | | | | | | | |
|`cublasDsyrk_v2`| | | |`hipblasDsyrk`|3.5.0| | | |`rocblas_dsyrk`|3.5.0| | | |
|`cublasDsyrk_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDsyrkx`| | | |`hipblasDsyrkx`|3.5.0| | | |`rocblas_dsyrkx`|3.5.0| | | |
|`cublasDsyrkx_64`|12.0| | | | | | | | | | | | |
|`cublasDtrmm`| | | |`hipblasDtrmm`|3.2.0|5.6.0| | |`rocblas_dtrmm_outofplace`|5.0.0|5.6.0| | |
|`cublasDtrmm_64`|12.0| | | | | | | | | | | | |
|`cublasDtrmm_v2`| | | |`hipblasDtrmm`|3.2.0|5.6.0| | |`rocblas_dtrmm_outofplace`|5.0.0|5.6.0| | |
|`cublasDtrmm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasDtrsm`| | | |`hipblasDtrsm`|1.8.2| | | |`rocblas_dtrsm`|1.5.0| | | |
|`cublasDtrsm_64`|12.0| | | | | | | | | | | | |
|`cublasDtrsm_v2`| | | |`hipblasDtrsm`|1.8.2| | | |`rocblas_dtrsm`|1.5.0| | | |
|`cublasDtrsm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasHSHgemvBatched`|11.6| | | | | | | | | | | | |
|`cublasHSHgemvBatched_64`|12.0| | | | | | | | | | | | |
|`cublasHSHgemvStridedBatched`|11.6| | | | | | | | | | | | |
|`cublasHSHgemvStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasHSSgemvBatched`|11.6| | | | | | | | | | | | |
|`cublasHSSgemvBatched_64`|12.0| | | | | | | | | | | | |
|`cublasHSSgemvStridedBatched`|11.6| | | | | | | | | | | | |
|`cublasHSSgemvStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasHgemm`|7.5| | |`hipblasHgemm`|1.8.2| | | |`rocblas_hgemm`|1.5.0| | | |
|`cublasHgemmBatched`|9.0| | |`hipblasHgemmBatched`|3.0.0| | | |`rocblas_hgemm_batched`|3.5.0| | | |
|`cublasHgemmBatched_64`|12.0| | | | | | | | | | | | |
|`cublasHgemmStridedBatched`|8.0| | |`hipblasHgemmStridedBatched`|3.0.0| | | |`rocblas_hgemm_strided_batched`|1.5.0| | | |
|`cublasHgemmStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasHgemm_64`|12.0| | | | | | | | | | | | |
|`cublasSgemm`| | | |`hipblasSgemm`|1.8.2| | | |`rocblas_sgemm`|1.5.0| | | |
|`cublasSgemmBatched`| | | |`hipblasSgemmBatched`|1.8.2| | | |`rocblas_sgemm_batched`|3.5.0| | | |
|`cublasSgemmBatched_64`|12.0| | | | | | | | | | | | |
|`cublasSgemmStridedBatched`|8.0| | |`hipblasSgemmStridedBatched`|1.8.2| | | |`rocblas_sgemm_strided_batched`|1.5.0| | | |
|`cublasSgemmStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasSgemm_64`|12.0| | | | | | | | | | | | |
|`cublasSgemm_v2`| | | |`hipblasSgemm`|1.8.2| | | |`rocblas_sgemm`|1.5.0| | | |
|`cublasSgemm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSgemvBatched`|11.6| | | | | | | | | | | | |
|`cublasSgemvBatched_64`|12.0| | | | | | | | | | | | |
|`cublasSgemvStridedBatched`|11.6| | | | | | | | | | | | |
|`cublasSgemvStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasSsymm`| | | |`hipblasSsymm`|3.6.0| | | |`rocblas_ssymm`|3.5.0| | | |
|`cublasSsymm_64`|12.0| | | | | | | | | | | | |
|`cublasSsymm_v2`| | | |`hipblasSsymm`|3.6.0| | | |`rocblas_ssymm`|3.5.0| | | |
|`cublasSsymm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSsyr2k`| | | |`hipblasSsyr2k`|3.5.0| | | |`rocblas_ssyr2k`|3.5.0| | | |
|`cublasSsyr2k_64`|12.0| | | | | | | | | | | | |
|`cublasSsyr2k_v2`| | | |`hipblasSsyr2k`|3.5.0| | | |`rocblas_ssyr2k`|3.5.0| | | |
|`cublasSsyr2k_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSsyrk`| | | |`hipblasSsyrk`|3.5.0| | | |`rocblas_ssyrk`|3.5.0| | | |
|`cublasSsyrk_64`|12.0| | | | | | | | | | | | |
|`cublasSsyrk_v2`| | | |`hipblasSsyrk`|3.5.0| | | |`rocblas_ssyrk`|3.5.0| | | |
|`cublasSsyrk_v2_64`|12.0| | | | | | | | | | | | |
|`cublasSsyrkx`| | | |`hipblasSsyrkx`|3.5.0| | | |`rocblas_ssyrkx`|3.5.0| | | |
|`cublasSsyrkx_64`|12.0| | | | | | | | | | | | |
|`cublasStrmm`| | | |`hipblasStrmm`|3.2.0|5.6.0| | |`rocblas_strmm_outofplace`|5.0.0|5.6.0| | |
|`cublasStrmm_64`|12.0| | | | | | | | | | | | |
|`cublasStrmm_v2`| | | |`hipblasStrmm`|3.2.0|5.6.0| | |`rocblas_strmm_outofplace`|5.0.0|5.6.0| | |
|`cublasStrmm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasStrsm`| | | |`hipblasStrsm`|1.8.2| | | |`rocblas_strsm`|1.5.0| | | |
|`cublasStrsm_64`|12.0| | | | | | | | | | | | |
|`cublasStrsm_v2`| | | |`hipblasStrsm`|1.8.2| | | |`rocblas_strsm`|1.5.0| | | |
|`cublasStrsm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasTSSgemvBatched`|11.6| | | | | | | | | | | | |
|`cublasTSSgemvBatched_64`|12.0| | | | | | | | | | | | |
|`cublasTSSgemvStridedBatched`|11.6| | | | | | | | | | | | |
|`cublasTSSgemvStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasTSTgemvBatched`|11.6| | | | | | | | | | | | |
|`cublasTSTgemvBatched_64`|12.0| | | | | | | | | | | | |
|`cublasTSTgemvStridedBatched`|11.6| | | | | | | | | | | | |
|`cublasTSTgemvStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasZgemm`| | | |`hipblasZgemm`|1.8.2| | | |`rocblas_zgemm`|1.5.0| | | |
|`cublasZgemm3m`|8.0| | | | | | | | | | | | |
|`cublasZgemm3m_64`|12.0| | | | | | | | | | | | |
|`cublasZgemmBatched`| | | |`hipblasZgemmBatched`|3.0.0| | | |`rocblas_zgemm_batched`|3.5.0| | | |
|`cublasZgemmBatched_64`|12.0| | | | | | | | | | | | |
|`cublasZgemmStridedBatched`|8.0| | |`hipblasZgemmStridedBatched`|3.0.0| | | |`rocblas_zgemm_strided_batched`|1.5.0| | | |
|`cublasZgemmStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasZgemm_64`|12.0| | | | | | | | | | | | |
|`cublasZgemm_v2`| | | |`hipblasZgemm`|1.8.2| | | |`rocblas_zgemm`|1.5.0| | | |
|`cublasZgemm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZgemvBatched`|11.6| | | | | | | | | | | | |
|`cublasZgemvBatched_64`|12.0| | | | | | | | | | | | |
|`cublasZgemvStridedBatched`|11.6| | | | | | | | | | | | |
|`cublasZgemvStridedBatched_64`|12.0| | | | | | | | | | | | |
|`cublasZhemm`| | | |`hipblasZhemm`|3.6.0| | | |`rocblas_zhemm`|3.5.0| | | |
|`cublasZhemm_64`|12.0| | | | | | | | | | | | |
|`cublasZhemm_v2`| | | |`hipblasZhemm`|3.6.0| | | |`rocblas_zhemm`|3.5.0| | | |
|`cublasZhemm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZher2k`| | | |`hipblasZher2k`|3.5.0| | | |`rocblas_zher2k`|3.5.0| | | |
|`cublasZher2k_64`|12.0| | | | | | | | | | | | |
|`cublasZher2k_v2`| | | |`hipblasZher2k`|3.5.0| | | |`rocblas_zher2k`|3.5.0| | | |
|`cublasZher2k_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZherk`| | | |`hipblasZherk`|3.5.0| | | |`rocblas_zherk`|3.5.0| | | |
|`cublasZherk_64`|12.0| | | | | | | | | | | | |
|`cublasZherk_v2`| | | |`hipblasZherk`|3.5.0| | | |`rocblas_zherk`|3.5.0| | | |
|`cublasZherk_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZherkx`| | | |`hipblasZherkx`|3.5.0| | | |`rocblas_zherkx`|3.5.0| | | |
|`cublasZherkx_64`|12.0| | | | | | | | | | | | |
|`cublasZsymm`| | | |`hipblasZsymm`|3.6.0| | | |`rocblas_zsymm`|3.5.0| | | |
|`cublasZsymm_64`|12.0| | | | | | | | | | | | |
|`cublasZsymm_v2`| | | |`hipblasZsymm`|3.6.0| | | |`rocblas_zsymm`|3.5.0| | | |
|`cublasZsymm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZsyr2k`| | | |`hipblasZsyr2k`|3.5.0| | | |`rocblas_zsyr2k`|3.5.0| | | |
|`cublasZsyr2k_64`|12.0| | | | | | | | | | | | |
|`cublasZsyr2k_v2`| | | |`hipblasZsyr2k`|3.5.0| | | |`rocblas_zsyr2k`|3.5.0| | | |
|`cublasZsyr2k_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZsyrk`| | | |`hipblasZsyrk`|3.5.0| | | |`rocblas_zsyrk`|3.5.0| | | |
|`cublasZsyrk_64`|12.0| | | | | | | | | | | | |
|`cublasZsyrk_v2`| | | |`hipblasZsyrk`|3.5.0| | | |`rocblas_zsyrk`|3.5.0| | | |
|`cublasZsyrk_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZsyrkx`| | | |`hipblasZsyrkx`|3.5.0| | | |`rocblas_zsyrkx`|3.5.0| | | |
|`cublasZsyrkx_64`|12.0| | | | | | | | | | | | |
|`cublasZtrmm`| | | |`hipblasZtrmm`|3.5.0|5.6.0| | |`rocblas_ztrmm_outofplace`|5.0.0|5.6.0| | |
|`cublasZtrmm_64`|12.0| | | | | | | | | | | | |
|`cublasZtrmm_v2`| | | |`hipblasZtrmm`|3.5.0|5.6.0| | |`rocblas_ztrmm_outofplace`|5.0.0|5.6.0| | |
|`cublasZtrmm_v2_64`|12.0| | | | | | | | | | | | |
|`cublasZtrsm`| | | |`hipblasZtrsm`|3.5.0| | | |`rocblas_ztrsm`|3.5.0| | | |
|`cublasZtrsm_64`|12.0| | | | | | | | | | | | |
|`cublasZtrsm_v2`| | | |`hipblasZtrsm`|3.5.0| | | |`rocblas_ztrsm`|3.5.0| | | |
|`cublasZtrsm_v2_64`|12.0| | | | | | | | | | | | |

## **8. BLAS-like Extension**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|**E**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cublasAsumEx`|10.1| | | | | | | | | | | | |
|`cublasAsumEx_64`|12.0| | | | | | | | | | | | |
|`cublasAxpyEx`|8.0| | |`hipblasAxpyEx`|4.1.0| | | |`rocblas_axpy_ex`|3.9.0| | | |
|`cublasAxpyEx_64`|12.0| | | | | | | | | | | | |
|`cublasCdgmm`| | | |`hipblasCdgmm`|3.6.0| | | |`rocblas_cdgmm`|3.5.0| | | |
|`cublasCdgmm_64`|12.0| | | | | | | | | | | | |
|`cublasCgeam`| | | |`hipblasCgeam`|3.6.0| | | |`rocblas_cgeam`|3.5.0| | | |
|`cublasCgeam_64`|12.0| | | | | | | | | | | | |
|`cublasCgelsBatched`| | | |`hipblasCgelsBatched`|5.4.0| | | | | | | | |
|`cublasCgemmEx`|8.0| | | | | | | | | | | | |
|`cublasCgemmEx_64`|12.0| | | | | | | | | | | | |
|`cublasCgeqrfBatched`| | | |`hipblasCgeqrfBatched`|3.5.0| | | | | | | | |
|`cublasCgetrfBatched`| | | |`hipblasCgetrfBatched`|3.5.0| | | | | | | | |
|`cublasCgetriBatched`| | | |`hipblasCgetriBatched`|3.7.0| | | | | | | | |
|`cublasCgetrsBatched`| | | |`hipblasCgetrsBatched`|3.5.0| | | | | | | | |
|`cublasCherk3mEx`|8.0| | | | | | | | | | | | |
|`cublasCherk3mEx_64`|12.0| | | | | | | | | | | | |
|`cublasCherkEx`|8.0| | | | | | | | | | | | |
|`cublasCherkEx_64`|12.0| | | | | | | | | | | | |
|`cublasCmatinvBatched`| | | | | | | | | | | | | |
|`cublasCopyEx`|10.1| | | | | | | | | | | | |
|`cublasCopyEx_64`|12.0| | | | | | | | | | | | |
|`cublasCsyrk3mEx`|8.0| | | | | | | | | | | | |
|`cublasCsyrk3mEx_64`|12.0| | | | | | | | | | | | |
|`cublasCsyrkEx`|8.0| | | | | | | | | | | | |
|`cublasCsyrkEx_64`|12.0| | | | | | | | | | | | |
|`cublasCtpttr`| | | | | | | | | | | | | |
|`cublasCtrsmBatched`| | | |`hipblasCtrsmBatched`|3.5.0| | | |`rocblas_ctrsm_batched`|3.5.0| | | |
|`cublasCtrsmBatched_64`|12.0| | | | | | | | | | | | |
|`cublasCtrttp`| | | | | | | | | | | | | |
|`cublasDdgmm`| | | |`hipblasDdgmm`|3.6.0| | | |`rocblas_ddgmm`|3.5.0| | | |
|`cublasDdgmm_64`|12.0| | | | | | | | | | | | |
|`cublasDgeam`| | | |`hipblasDgeam`|1.8.2| | | |`rocblas_dgeam`|1.6.4| | | |
|`cublasDgeam_64`|12.0| | | | | | | | | | | | |
|`cublasDgelsBatched`| | | |`hipblasDgelsBatched`|5.4.0| | | | | | | | |
|`cublasDgeqrfBatched`| | | |`hipblasDgeqrfBatched`|3.5.0| | | | | | | | |
|`cublasDgetrfBatched`| | | |`hipblasDgetrfBatched`|3.5.0| | | | | | | | |
|`cublasDgetriBatched`| | | |`hipblasDgetriBatched`|3.7.0| | | | | | | | |
|`cublasDgetrsBatched`| | | |`hipblasDgetrsBatched`|3.5.0| | | | | | | | |
|`cublasDmatinvBatched`| | | | | | | | | | | | | |
|`cublasDotEx`|8.0| | |`hipblasDotEx`|4.1.0| | | |`rocblas_dot_ex`|4.1.0| | | |
|`cublasDotEx_64`|12.0| | | | | | | | | | | | |
|`cublasDotcEx`|8.0| | |`hipblasDotcEx`|4.1.0| | | |`rocblas_dotc_ex`|4.1.0| | | |
|`cublasDotcEx_64`|12.0| | | | | | | | | | | | |
|`cublasDtpttr`| | | | | | | | | | | | | |
|`cublasDtrsmBatched`| | | |`hipblasDtrsmBatched`|3.2.0| | | |`rocblas_dtrsm_batched`|3.5.0| | | |
|`cublasDtrsmBatched_64`|12.0| | | | | | | | | | | | |
|`cublasDtrttp`| | | | | | | | | | | | | |
|`cublasGemmBatchedEx`|9.1| | |`hipblasGemmBatchedEx`|3.6.0| | | |`rocblas_gemm_batched_ex`|3.5.0| | | |
|`cublasGemmBatchedEx_64`|12.0| | | | | | | | | | | | |
|`cublasGemmEx`|8.0| | |`hipblasGemmEx`|1.8.2| | | |`rocblas_gemm_ex`|1.8.2| | | |
|`cublasGemmEx_64`|12.0| | | | | | | | | | | | |
|`cublasGemmStridedBatchedEx`|9.1| | |`hipblasGemmStridedBatchedEx`|3.6.0| | | |`rocblas_gemm_strided_batched_ex`|1.9.0| | | |
|`cublasGemmStridedBatchedEx_64`|12.0| | | | | | | | | | | | |
|`cublasIamaxEx`|10.1| | | | | | | | | | | | |
|`cublasIamaxEx_64`|12.0| | | | | | | | | | | | |
|`cublasIaminEx`|10.1| | | | | | | | | | | | |
|`cublasIaminEx_64`|12.0| | | | | | | | | | | | |
|`cublasRotEx`|10.1| | |`hipblasRotEx`|4.1.0| | | |`rocblas_rot_ex`|4.1.0| | | |
|`cublasRotEx_64`|12.0| | | | | | | | | | | | |
|`cublasRotgEx`|10.1| | | | | | | | | | | | |
|`cublasRotmEx`|10.1| | | | | | | | | | | | |
|`cublasRotmEx_64`|12.0| | | | | | | | | | | | |
|`cublasRotmgEx`|10.1| | | | | | | | | | | | |
|`cublasScalEx`|8.0| | |`hipblasScalEx`|4.1.0| | | |`rocblas_scal_ex`|4.0.0| | | |
|`cublasScalEx_64`|12.0| | | | | | | | | | | | |
|`cublasSdgmm`| | | |`hipblasSdgmm`|3.6.0| | | |`rocblas_sdgmm`|3.5.0| | | |
|`cublasSdgmm_64`|12.0| | | | | | | | | | | | |
|`cublasSgeam`| | | |`hipblasSgeam`|1.8.2| | | |`rocblas_sgeam`|1.6.4| | | |
|`cublasSgeam_64`|12.0| | | | | | | | | | | | |
|`cublasSgelsBatched`| | | |`hipblasSgelsBatched`|5.4.0| | | | | | | | |
|`cublasSgemmEx`|7.5| | | | | | | | | | | | |
|`cublasSgemmEx_64`|12.0| | | | | | | | | | | | |
|`cublasSgeqrfBatched`| | | |`hipblasSgeqrfBatched`|3.5.0| | | | | | | | |
|`cublasSgetrfBatched`| | | |`hipblasSgetrfBatched`|3.5.0| | | | | | | | |
|`cublasSgetriBatched`| | | |`hipblasSgetriBatched`|3.7.0| | | | | | | | |
|`cublasSgetrsBatched`| | | |`hipblasSgetrsBatched`|3.5.0| | | | | | | | |
|`cublasSmatinvBatched`| | | | | | | | | | | | | |
|`cublasStpttr`| | | | | | | | | | | | | |
|`cublasStrsmBatched`| | | |`hipblasStrsmBatched`|3.2.0| | | |`rocblas_strsm_batched`|3.5.0| | | |
|`cublasStrsmBatched_64`|12.0| | | | | | | | | | | | |
|`cublasStrttp`| | | | | | | | | | | | | |
|`cublasSwapEx`|10.1| | | | | | | | | | | | |
|`cublasSwapEx_64`|12.0| | | | | | | | | | | | |
|`cublasUint8gemmBias`|8.0| | | | | | | | | | | | |
|`cublasZdgmm`| | | |`hipblasZdgmm`|3.6.0| | | |`rocblas_zdgmm`|3.5.0| | | |
|`cublasZdgmm_64`|12.0| | | | | | | | | | | | |
|`cublasZgeam`| | | |`hipblasZgeam`|3.6.0| | | |`rocblas_zgeam`|3.5.0| | | |
|`cublasZgeam_64`|12.0| | | | | | | | | | | | |
|`cublasZgelsBatched`| | | |`hipblasZgelsBatched`|5.4.0| | | | | | | | |
|`cublasZgeqrfBatched`| | | |`hipblasZgeqrfBatched`|3.5.0| | | | | | | | |
|`cublasZgetrfBatched`| | | |`hipblasZgetrfBatched`|3.5.0| | | | | | | | |
|`cublasZgetriBatched`| | | |`hipblasZgetriBatched`|3.7.0| | | | | | | | |
|`cublasZgetrsBatched`| | | |`hipblasZgetrsBatched`|3.5.0| | | | | | | | |
|`cublasZmatinvBatched`| | | | | | | | | | | | | |
|`cublasZtpttr`| | | | | | | | | | | | | |
|`cublasZtrsmBatched`| | | |`hipblasZtrsmBatched`|3.5.0| | | |`rocblas_ztrsm_batched`|3.5.0| | | |
|`cublasZtrsmBatched_64`|12.0| | | | | | | | | | | | |
|`cublasZtrttp`| | | | | | | | | | | | | |


\*A - Added; D - Deprecated; R - Removed; E - Experimental