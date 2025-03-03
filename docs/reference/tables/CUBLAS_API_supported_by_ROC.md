# CUBLAS API supported by ROC


**Note\:** In the tables that follow the columns marked `A`, `D`, `C`, `R`, and `E` mean the following:
**A** - Added; **D** - Deprecated; **C** - Changed; **R** - Removed; **E** - Experimental

## **1. CUBLAS Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUBLAS_ATOMICS_ALLOWED`| | | | |`rocblas_atomics_allowed`|3.8.0| | | | |
|`CUBLAS_ATOMICS_NOT_ALLOWED`| | | | |`rocblas_atomics_not_allowed`|3.8.0| | | | |
|`CUBLAS_COMPUTE_16F`|11.0| | | | | | | | | |
|`CUBLAS_COMPUTE_16F_PEDANTIC`|11.0| | | | | | | | | |
|`CUBLAS_COMPUTE_32F`|11.0| | | |`rocblas_compute_type_f32`|5.7.0| | | | |
|`CUBLAS_COMPUTE_32F_FAST_16BF`|11.0| | | | | | | | | |
|`CUBLAS_COMPUTE_32F_FAST_16F`|11.0| | | | | | | | | |
|`CUBLAS_COMPUTE_32F_FAST_TF32`|11.0| | | | | | | | | |
|`CUBLAS_COMPUTE_32F_PEDANTIC`|11.0| | | | | | | | | |
|`CUBLAS_COMPUTE_32I`|11.0| | | | | | | | | |
|`CUBLAS_COMPUTE_32I_PEDANTIC`|11.0| | | | | | | | | |
|`CUBLAS_COMPUTE_64F`|11.0| | | | | | | | | |
|`CUBLAS_COMPUTE_64F_PEDANTIC`|11.0| | | | | | | | | |
|`CUBLAS_DEFAULT_MATH`|9.0| | | |`rocblas_default_math`|5.7.0| | | | |
|`CUBLAS_DIAG_NON_UNIT`| | | | |`rocblas_diagonal_non_unit`|1.5.0| | | | |
|`CUBLAS_DIAG_UNIT`| | | | |`rocblas_diagonal_unit`|1.5.0| | | | |
|`CUBLAS_FILL_MODE_FULL`|10.1| | | |`rocblas_fill_full`|1.5.0| | | | |
|`CUBLAS_FILL_MODE_LOWER`| | | | |`rocblas_fill_lower`|1.5.0| | | | |
|`CUBLAS_FILL_MODE_UPPER`| | | | |`rocblas_fill_upper`|1.5.0| | | | |
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
|`CUBLAS_GEMM_DEFAULT`|9.0| | | |`rocblas_gemm_algo_standard`|1.8.2| | | | |
|`CUBLAS_GEMM_DEFAULT_TENSOR_OP`|9.0| | | | | | | | | |
|`CUBLAS_GEMM_DFALT`|8.0| | | |`rocblas_gemm_algo_standard`|1.8.2| | | | |
|`CUBLAS_GEMM_DFALT_TENSOR_OP`|9.0| | | | | | | | | |
|`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`|11.0| | | | | | | | | |
|`CUBLAS_OP_C`| | | | |`rocblas_operation_conjugate_transpose`|1.5.0| | | | |
|`CUBLAS_OP_CONJG`|10.1| | | | | | | | | |
|`CUBLAS_OP_HERMITAN`|10.1| | | |`rocblas_operation_conjugate_transpose`|1.5.0| | | | |
|`CUBLAS_OP_N`| | | | |`rocblas_operation_none`|1.5.0| | | | |
|`CUBLAS_OP_T`| | | | |`rocblas_operation_transpose`|1.5.0| | | | |
|`CUBLAS_PEDANTIC_MATH`|11.0| | | | | | | | | |
|`CUBLAS_POINTER_MODE_DEVICE`| | | | |`rocblas_pointer_mode_device`|1.6.0| | | | |
|`CUBLAS_POINTER_MODE_HOST`| | | | |`rocblas_pointer_mode_host`|1.6.0| | | | |
|`CUBLAS_SIDE_LEFT`| | | | |`rocblas_side_left`|1.5.0| | | | |
|`CUBLAS_SIDE_RIGHT`| | | | |`rocblas_side_right`|1.5.0| | | | |
|`CUBLAS_STATUS_ALLOC_FAILED`| | | | |`rocblas_status_not_implemented`|1.5.0| | | | |
|`CUBLAS_STATUS_ARCH_MISMATCH`| | | | |`rocblas_status_arch_mismatch`|5.7.0| | | | |
|`CUBLAS_STATUS_EXECUTION_FAILED`| | | | |`rocblas_status_memory_error`|1.5.0| | | | |
|`CUBLAS_STATUS_INTERNAL_ERROR`| | | | |`rocblas_status_internal_error`|1.5.0| | | | |
|`CUBLAS_STATUS_INVALID_VALUE`| | | | |`rocblas_status_invalid_value`|3.5.0| | | | |
|`CUBLAS_STATUS_LICENSE_ERROR`| | | | | | | | | | |
|`CUBLAS_STATUS_MAPPING_ERROR`| | | | |`rocblas_status_invalid_size`|1.5.0| | | | |
|`CUBLAS_STATUS_NOT_INITIALIZED`| | | | |`rocblas_status_invalid_handle`|1.5.0| | | | |
|`CUBLAS_STATUS_NOT_SUPPORTED`| | | | |`rocblas_status_perf_degraded`|3.5.0| | | | |
|`CUBLAS_STATUS_SUCCESS`| | | | |`rocblas_status_success`|1.5.0| | | | |
|`CUBLAS_TENSOR_OP_MATH`|9.0|11.0| | | | | | | | |
|`CUBLAS_TF32_TENSOR_OP_MATH`|11.0| | | | | | | | | |
|`cublasAtomicsMode_t`| | | | |`rocblas_atomics_mode`|3.8.0| | | | |
|`cublasComputeType_t`|11.0| | | |`rocblas_computetype`|5.7.0| | | | |
|`cublasContext`| | | | |`_rocblas_handle`|1.5.0| | | | |
|`cublasDiagType_t`| | | | |`rocblas_diagonal`|1.5.0| | | | |
|`cublasFillMode_t`| | | | |`rocblas_fill`|1.5.0| | | | |
|`cublasGemmAlgo_t`|8.0| | | |`rocblas_gemm_algo`|1.8.2| | | | |
|`cublasHandle_t`| | | | |`rocblas_handle`|1.5.0| | | | |
|`cublasMath_t`|9.0| | | |`rocblas_math_mode`|5.7.0| | | | |
|`cublasOperation_t`| | | | |`rocblas_operation`|1.5.0| | | | |
|`cublasPointerMode_t`| | | | |`rocblas_pointer_mode`|1.6.0| | | | |
|`cublasSideMode_t`| | | | |`rocblas_side`|1.5.0| | | | |
|`cublasStatus`| | | | |`rocblas_status`|1.5.0| | | | |
|`cublasStatus_t`| | | | |`rocblas_status`|1.5.0| | | | |

## **2. CUDA Library Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUDA_C_16BF`| | | | |`rocblas_datatype_bf16_c`|3.5.0| | | | |
|`CUDA_C_16F`|8.0| | | |`rocblas_datatype_f16_c`|1.8.2| | | | |
|`CUDA_C_16I`|11.0| | | | | | | | | |
|`CUDA_C_16U`|11.0| | | | | | | | | |
|`CUDA_C_32F`|8.0| | | |`rocblas_datatype_f32_c`|1.8.2| | | | |
|`CUDA_C_32I`|8.0| | | |`rocblas_datatype_i32_c`|2.0.0| | | | |
|`CUDA_C_32U`|8.0| | | |`rocblas_datatype_u32_c`|2.0.0| | | | |
|`CUDA_C_4I`|11.0| | | | | | | | | |
|`CUDA_C_4U`|11.0| | | | | | | | | |
|`CUDA_C_64F`|8.0| | | |`rocblas_datatype_f64_c`|1.8.2| | | | |
|`CUDA_C_64I`|11.0| | | | | | | | | |
|`CUDA_C_64U`|11.0| | | | | | | | | |
|`CUDA_C_8I`|8.0| | | |`rocblas_datatype_i8_c`|2.0.0| | | | |
|`CUDA_C_8U`|8.0| | | |`rocblas_datatype_u8_c`|2.0.0| | | | |
|`CUDA_R_16BF`| | | | |`rocblas_datatype_bf16_r`|3.5.0| | | | |
|`CUDA_R_16F`|8.0| | | |`rocblas_datatype_f16_r`|1.8.2| | | | |
|`CUDA_R_16I`|11.0| | | | | | | | | |
|`CUDA_R_16U`|11.0| | | | | | | | | |
|`CUDA_R_32F`|8.0| | | |`rocblas_datatype_f32_r`|1.8.2| | | | |
|`CUDA_R_32I`|8.0| | | |`rocblas_datatype_i32_r`|2.0.0| | | | |
|`CUDA_R_32U`|8.0| | | |`rocblas_datatype_u32_r`|2.0.0| | | | |
|`CUDA_R_4I`|11.0| | | | | | | | | |
|`CUDA_R_4U`|11.0| | | | | | | | | |
|`CUDA_R_64F`|8.0| | | |`rocblas_datatype_f64_r`|1.8.2| | | | |
|`CUDA_R_64I`|11.0| | | | | | | | | |
|`CUDA_R_64U`|11.0| | | | | | | | | |
|`CUDA_R_8F_E4M3`|11.8| | | | | | | | | |
|`CUDA_R_8F_E5M2`|11.8| | | | | | | | | |
|`CUDA_R_8I`|8.0| | | |`rocblas_datatype_i8_r`|2.0.0| | | | |
|`CUDA_R_8U`|8.0| | | |`rocblas_datatype_u8_r`|2.0.0| | | | |
|`cublasDataType_t`|7.5| | | |`rocblas_datatype`|1.8.2| | | | |
|`cudaDataType`|8.0| | | |`rocblas_datatype`|1.8.2| | | | |
|`cudaDataType_t`|8.0| | | |`rocblas_datatype_`|1.8.2| | | | |

## **3. CUBLASLt Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUBLASLT_ALGO_CAP_ATOMIC_SYNC`|12.2| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_EPILOGUE_MASK`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_LD_NEGATIVE`|11.0| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES`|11.0| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES`|11.0| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES`|11.0| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES`|11.0| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS`|11.0| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_POINTER_MODE_MASK`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_SPLITK_SUPPORT`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_STAGES_IDS`|11.0| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_TILE_IDS`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CAP_UPLO_SUPPORT`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID`|11.8| | | | | | | | | |
|`CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CONFIG_ID`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID`|11.8| | | | | | | | | |
|`CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CONFIG_SPLITK_NUM`|10.1| | | | | | | | | |
|`CUBLASLT_ALGO_CONFIG_STAGES_ID`|11.0| | | | | | | | | |
|`CUBLASLT_ALGO_CONFIG_TILE_ID`|10.1| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_10x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_11x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_12x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_13x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_14x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_15x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_16x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x10x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x11x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x12x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x13x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x14x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x15x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x16x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x2x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x3x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x4x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x5x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x6x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x7x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x8x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_1x9x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_2x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_2x2x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_2x3x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_2x4x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_2x5x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_2x6x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_2x7x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_2x8x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_3x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_3x2x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_3x3x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_3x4x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_3x5x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_4x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_4x2x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_4x3x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_4x4x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_5x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_5x2x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_5x3x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_6x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_6x2x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_7x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_7x2x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_8x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_8x2x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_9x1x1`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_AUTO`|11.8| | | | | | | | | |
|`CUBLASLT_CLUSTER_SHAPE_END`|11.8| | | | | | | | | |
|`CUBLASLT_EPILOGUE_BGRADA`|11.4| | | | | | | | | |
|`CUBLASLT_EPILOGUE_BGRADB`|11.4| | | | | | | | | |
|`CUBLASLT_EPILOGUE_BIAS`|10.1| | | | | | | | | |
|`CUBLASLT_EPILOGUE_DEFAULT`|10.1| | | | | | | | | |
|`CUBLASLT_EPILOGUE_DGELU`|11.6| | | | | | | | | |
|`CUBLASLT_EPILOGUE_DGELU_BGRAD`|11.3| | | | | | | | | |
|`CUBLASLT_EPILOGUE_DRELU`|11.6| | | | | | | | | |
|`CUBLASLT_EPILOGUE_DRELU_BGRAD`|11.3| | | | | | | | | |
|`CUBLASLT_EPILOGUE_GELU`|11.3| | | | | | | | | |
|`CUBLASLT_EPILOGUE_GELU_AUX`|11.3| | | | | | | | | |
|`CUBLASLT_EPILOGUE_GELU_AUX_BIAS`|11.3| | | | | | | | | |
|`CUBLASLT_EPILOGUE_GELU_BIAS`|11.3| | | | | | | | | |
|`CUBLASLT_EPILOGUE_RELU`|10.1| | | | | | | | | |
|`CUBLASLT_EPILOGUE_RELU_AUX`|11.3| | | | | | | | | |
|`CUBLASLT_EPILOGUE_RELU_AUX_BIAS`|11.3| | | | | | | | | |
|`CUBLASLT_EPILOGUE_RELU_BIAS`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE`|11.4| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_AMAX_D_POINTER`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER`|12.2| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS`|12.2| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS`|12.2| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER`|12.2| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE`|11.3| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_BIAS_POINTER`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_B_SCALE_POINTER`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_COMPUTE_TYPE`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_C_SCALE_POINTER`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_D_SCALE_POINTER`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_EPILOGUE`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE`|11.3| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD`|11.3| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`|11.3| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_FAST_ACCUM`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_FILL_MODE`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_POINTER_MODE`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_SCALE_TYPE`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET`|11.5| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_TRANSA`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_TRANSB`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_DESC_TRANSC`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_INNER_SHAPE_END`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_INNER_SHAPE_MMA16816`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_INNER_SHAPE_MMA1684`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_INNER_SHAPE_MMA1688`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_INNER_SHAPE_MMA884`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_INNER_SHAPE_UNDEFINED`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_PREF_IMPL_MASK`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_PREF_SEARCH_MODE`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_128x1`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_128x2`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_128x3`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_128x4`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_128x5`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_128x6`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_128xAUTO`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_16x1`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_16x10`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_16x2`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_16x3`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_16x4`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_16x5`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_16x6`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_16xAUTO`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_32x1`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_32x10`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_32x2`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_32x3`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_32x4`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_32x5`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_32x6`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_32xAUTO`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_64x1`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_64x2`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_64x3`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_64x4`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_64x5`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_64x6`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_64xAUTO`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_8x3`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_8x4`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_8x5`|11.2| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_8xAUTO`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_END`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_STAGES_UNDEFINED`|11.0| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_104x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_104x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_104x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_104x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_104x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_104x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_104x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_112x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_112x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_112x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_112x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_112x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_112x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_120x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_120x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_120x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_120x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_120x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_120x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x104`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x112`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x120`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x128`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x136`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x144`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x152`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x160`|11.3| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x168`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x176`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x184`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x192`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x200`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x208`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x216`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x224`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x232`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x240`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x248`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x256`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x264`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x272`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x280`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x288`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x296`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x304`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x312`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x32`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x328`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x336`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x344`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x352`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x360`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x368`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x376`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x392`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x400`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x408`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x416`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x424`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x432`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x440`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x456`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x464`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x472`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x480`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x488`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x496`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x504`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x64`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_128x96`|11.8| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_136x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_136x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_136x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_136x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_136x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_144x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_144x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_144x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_144x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_144x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_152x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_152x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_152x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_152x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_152x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_160x128`|11.3| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_160x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_160x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_160x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_168x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_168x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_168x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_168x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x16`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x32`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x576`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x640`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x704`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x768`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_16x8`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_176x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_176x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_176x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_176x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_184x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_184x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_184x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_184x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x104`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x112`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x120`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x128`|11.3| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x136`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x144`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x152`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x160`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x168`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x176`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x184`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x200`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x208`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x216`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x224`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x232`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x240`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x248`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x264`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x272`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x280`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x288`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x296`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x304`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x312`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x32`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x328`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x336`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_192x96`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_200x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_200x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_200x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_208x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_208x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_208x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_216x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_216x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_216x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_224x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_224x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_224x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_232x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_232x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_232x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_240x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_240x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_240x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_248x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_248x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_248x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x576`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x640`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x704`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_24x768`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x104`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x112`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x120`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x128`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x136`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x144`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x152`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x160`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x168`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x176`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x184`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x200`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x208`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x216`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x224`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x232`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x240`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x248`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x32`|12.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x64`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_256x96`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_264x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_264x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_272x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_272x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_280x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_280x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_288x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_288x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_296x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_296x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_304x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_304x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_312x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_312x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x104`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x112`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x120`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x136`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x144`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x152`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x160`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x168`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x176`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x184`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x200`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x32`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_320x96`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_328x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_328x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x128`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x16`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x256`|12.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x32`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x576`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x64`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x640`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x704`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x768`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_32x8`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_336x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_336x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_344x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_344x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_352x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_352x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_360x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_360x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_368x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_368x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_376x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_376x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x104`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x112`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x120`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x136`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x144`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x152`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x160`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x168`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x32`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_384x96`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_392x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_400x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_408x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x576`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x640`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x704`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_40x768`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_416x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_424x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_432x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_440x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x104`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x112`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x120`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x136`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x144`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x32`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_448x96`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_456x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_464x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_472x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_480x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_488x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x576`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x640`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x704`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_48x768`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_496x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_504x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x104`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x112`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x120`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x32`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x64`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_512x96`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_520x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_528x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_536x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_544x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_552x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_560x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_568x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x576`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x640`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x704`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_56x768`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x104`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x112`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x32`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_576x96`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_584x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_592x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_600x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_608x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_616x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_624x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_632x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x32`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_640x96`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_648x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x104`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x112`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x120`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x128`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x136`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x144`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x152`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x160`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x168`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x176`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x184`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x200`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x208`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x216`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x224`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x232`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x240`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x248`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x256`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x264`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x272`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x280`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x288`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x296`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x304`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x312`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x32`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x328`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x336`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x344`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x352`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x360`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x368`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x376`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x392`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x400`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x408`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x416`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x424`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x432`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x440`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x456`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x464`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x472`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x480`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x488`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x496`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x504`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x512`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x520`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x528`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x536`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x544`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x552`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x560`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x568`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x576`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x584`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x592`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x600`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x608`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x616`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x624`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x632`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x64`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x640`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x648`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x656`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x664`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x672`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x680`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x688`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x696`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x704`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x712`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x720`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x728`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x736`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x744`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x752`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x760`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x768`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x8`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_64x96`|11.3| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_656x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_664x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_672x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_680x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_688x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_696x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x32`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_704x88`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_712x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_720x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_728x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_72x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_72x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_72x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_72x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_72x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_72x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_72x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_72x576`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_72x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_72x640`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_736x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_744x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_752x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_760x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_768x16`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_768x24`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_768x32`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_768x40`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_768x48`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_768x56`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_768x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_768x72`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_768x8`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_768x80`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_80x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_80x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_80x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_80x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_80x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_80x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_80x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_80x576`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_80x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_88x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_88x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_88x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_88x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_88x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_88x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_88x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_88x64`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x128`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x16`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x32`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x576`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x64`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x640`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x704`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x768`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_8x8`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_96x128`|11.3| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_96x192`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_96x256`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_96x320`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_96x384`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_96x448`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_96x512`|12.6| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_96x64`|11.3| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_END`|10.1| | | | | | | | | |
|`CUBLASLT_MATMUL_TILE_UNDEFINED`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_LAYOUT_COLS`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_LAYOUT_LD`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_LAYOUT_ORDER`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_LAYOUT_ROWS`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_LAYOUT_TYPE`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA`|10.1| | | | | | | | | |
|`CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB`|10.1| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_64F`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_DMMA`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_GAUSSIAN`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_64F`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E4M3`|11.8| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E5M2`|11.8| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK`|11.0| | | | | | | | | |
|`CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK`|11.0| | | | | | | | | |
|`CUBLASLT_ORDER_COL`|10.1| | | | | | | | | |
|`CUBLASLT_ORDER_COL32`|10.1| | | | | | | | | |
|`CUBLASLT_ORDER_COL32_2R_4R4`|11.0| | | | | | | | | |
|`CUBLASLT_ORDER_COL4_4R2_8C`|10.1| | | | | | | | | |
|`CUBLASLT_ORDER_ROW`|10.1| | | | | | | | | |
|`CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST`|11.4| | | | | | | | | |
|`CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO`|10.1| | | | | | | | | |
|`CUBLASLT_POINTER_MODE_DEVICE`| | | | | | | | | | |
|`CUBLASLT_POINTER_MODE_DEVICE_VECTOR`|10.1| | | | | | | | | |
|`CUBLASLT_POINTER_MODE_HOST`|10.1| | | | | | | | | |
|`CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST`|11.4| | | | | | | | | |
|`CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO`|10.1| | | | | | | | | |
|`CUBLASLT_POINTER_MODE_MASK_DEVICE`|10.1| | | | | | | | | |
|`CUBLASLT_POINTER_MODE_MASK_DEVICE_VECTOR`|10.1| | | | | | | | | |
|`CUBLASLT_POINTER_MODE_MASK_HOST`|10.1| | | | | | | | | |
|`CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE`|10.1| | | | | | | | | |
|`CUBLASLT_REDUCTION_SCHEME_INPLACE`|10.1| | | | | | | | | |
|`CUBLASLT_REDUCTION_SCHEME_MASK`|10.1| | | | | | | | | |
|`CUBLASLT_REDUCTION_SCHEME_NONE`|10.1| | | | | | | | | |
|`CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE`|10.1| | | | | | | | | |
|`CUBLASLT_SEARCH_BEST_FIT`|10.1| | | | | | | | | |
|`CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID`|10.1| | | | | | | | | |
|`CUBLASLT_SEARCH_RESERVED_02`|11.0| | | | | | | | | |
|`CUBLASLT_SEARCH_RESERVED_03`|11.0| | | | | | | | | |
|`CUBLASLT_SEARCH_RESERVED_04`|11.0| | | | | | | | | |
|`CUBLASLT_SEARCH_RESERVED_05`|11.0| | | | | | | | | |
|`CUBLASLT_SEARCH_RESERVED_06`|12.6| | | | | | | | | |
|`CUBLASLT_SEARCH_RESERVED_07`|12.6| | | | | | | | | |
|`CUBLASLT_SEARCH_RESERVED_08`|12.6| | | | | | | | | |
|`CUBLASLT_SEARCH_RESERVED_09`|12.6| | | | | | | | | |
|`cublasLtClusterShape_t`|11.8| | | | | | | | | |
|`cublasLtContext`|10.1| | | | | | | | | |
|`cublasLtEpilogue_t`|10.1| | | | | | | | | |
|`cublasLtHandle_t`|10.1| | | | | | | | | |
|`cublasLtLoggerCallback_t`|11.0| | | | | | | | | |
|`cublasLtMatmulAlgoCapAttributes_t`|10.1| | | | | | | | | |
|`cublasLtMatmulAlgoConfigAttributes_t`|10.1| | | | | | | | | |
|`cublasLtMatmulAlgo_t`|10.1| | | | | | | | | |
|`cublasLtMatmulDescAttributes_t`|10.1| | | | | | | | | |
|`cublasLtMatmulDescOpaque_t`|11.0| | | | | | | | | |
|`cublasLtMatmulDesc_t`|10.1| | | | | | | | | |
|`cublasLtMatmulHeuristicResult_t`|10.1| | | | | | | | | |
|`cublasLtMatmulInnerShape_t`|11.8| | | | | | | | | |
|`cublasLtMatmulPreferenceAttributes_t`|10.1| | | | | | | | | |
|`cublasLtMatmulPreferenceOpaque_t`|11.0| | | | | | | | | |
|`cublasLtMatmulPreference_t`|10.1| | | | | | | | | |
|`cublasLtMatmulSearch_t`|10.1| | | | | | | | | |
|`cublasLtMatmulStages_t`|11.0| | | | | | | | | |
|`cublasLtMatmulTile_t`|10.1| | | | | | | | | |
|`cublasLtMatrixLayoutAttribute_t`|10.1| | | | | | | | | |
|`cublasLtMatrixLayoutOpaque_t`|11.0| | | | | | | | | |
|`cublasLtMatrixLayoutStruct`|10.1| | |10.2| | | | | | |
|`cublasLtMatrixLayout_t`|10.1| | | | | | | | | |
|`cublasLtMatrixTransformDescAttributes_t`|10.1| | | | | | | | | |
|`cublasLtMatrixTransformDescOpaque_t`|11.0| | | | | | | | | |
|`cublasLtMatrixTransformDesc_t`|10.1| | | | | | | | | |
|`cublasLtNumericalImplFlags_t`|11.0| | | | | | | | | |
|`cublasLtOrder_t`|10.1| | | | | | | | | |
|`cublasLtPointerModeMask_t`|10.1| | | | | | | | | |
|`cublasLtPointerMode_t`|10.1| | | | | | | | | |
|`cublasLtReductionScheme_t`|10.1| | | | | | | | | |

## **4. CUBLAS Helper Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasAlloc`| | | | | | | | | | |
|`cublasCreate`| | | | |`rocblas_create_handle`|1.5.0| | | | |
|`cublasCreate_v2`| | | | |`rocblas_create_handle`|1.5.0| | | | |
|`cublasDestroy`| | | | |`rocblas_destroy_handle`|1.5.0| | | | |
|`cublasDestroy_v2`| | | | |`rocblas_destroy_handle`|1.5.0| | | | |
|`cublasFree`| | | | | | | | | | |
|`cublasGetAtomicsMode`| | | | |`rocblas_get_atomics_mode`|3.8.0| | | | |
|`cublasGetCudartVersion`|10.1| | | | | | | | | |
|`cublasGetError`| | | | | | | | | | |
|`cublasGetLoggerCallback`|9.2| | | | | | | | | |
|`cublasGetMathMode`|9.0| | | |`rocblas_get_math_mode`|5.7.0| | | | |
|`cublasGetMatrix`| | | | |`rocblas_get_matrix`|1.6.0| | | | |
|`cublasGetMatrixAsync`| | | | |`rocblas_get_matrix_async`|3.5.0| | | | |
|`cublasGetMatrixAsync_64`|12.0| | | | | | | | | |
|`cublasGetMatrix_64`|12.0| | | | | | | | | |
|`cublasGetPointerMode`| | | | |`rocblas_get_pointer_mode`|1.6.0| | | | |
|`cublasGetPointerMode_v2`| | | | |`rocblas_get_pointer_mode`|1.6.0| | | | |
|`cublasGetProperty`| | | | | | | | | | |
|`cublasGetSmCountTarget`|11.3| | | | | | | | | |
|`cublasGetStatusName`|11.4| | | | | | | | | |
|`cublasGetStatusString`|11.4| | | |`rocblas_status_to_string`|3.5.0| | | | |
|`cublasGetStream`| | | | |`rocblas_get_stream`|1.5.0| | | | |
|`cublasGetStream_v2`| | | | |`rocblas_get_stream`|1.5.0| | | | |
|`cublasGetVector`| | | | |`rocblas_get_vector`|1.6.0| | | | |
|`cublasGetVectorAsync`| | | | |`rocblas_get_vector_async`|3.5.0| | | | |
|`cublasGetVectorAsync_64`|12.0| | | | | | | | | |
|`cublasGetVector_64`|12.0| | | | | | | | | |
|`cublasGetVersion`| | | | | | | | | | |
|`cublasGetVersion_v2`| | | | | | | | | | |
|`cublasInit`| | | | |`rocblas_initialize`|3.5.0| | | | |
|`cublasLogCallback`|9.2| | | | | | | | | |
|`cublasLoggerConfigure`|9.2| | | | | | | | | |
|`cublasMigrateComputeType`|11.0| | | | | | | | | |
|`cublasSetAtomicsMode`| | | | |`rocblas_set_atomics_mode`|3.8.0| | | | |
|`cublasSetKernelStream`| | | | | | | | | | |
|`cublasSetLoggerCallback`|9.2| | | | | | | | | |
|`cublasSetMathMode`|9.0| | | |`rocblas_set_math_mode`|5.7.0| | | | |
|`cublasSetMatrix`| | | | |`rocblas_set_matrix`|1.6.0| | | | |
|`cublasSetMatrixAsync`| | | | |`rocblas_set_matrix_async`|3.5.0| | | | |
|`cublasSetMatrixAsync_64`|12.0| | | | | | | | | |
|`cublasSetMatrix_64`|12.0| | | | | | | | | |
|`cublasSetPointerMode`| | | | |`rocblas_set_pointer_mode`|1.6.0| | | | |
|`cublasSetPointerMode_v2`| | | | |`rocblas_set_pointer_mode`|1.6.0| | | | |
|`cublasSetSmCountTarget`|11.3| | | | | | | | | |
|`cublasSetStream`| | | | |`rocblas_set_stream`|1.5.0| | | | |
|`cublasSetStream_v2`| | | | |`rocblas_set_stream`|1.5.0| | | | |
|`cublasSetVector`| | | | |`rocblas_set_vector`|1.6.0| | | | |
|`cublasSetVectorAsync`| | | | |`rocblas_set_vector_async`|3.5.0| | | | |
|`cublasSetVectorAsync_64`|12.0| | | | | | | | | |
|`cublasSetVector_64`|12.0| | | | | | | | | |
|`cublasShutdown`| | | | | | | | | | |
|`cublasXerbla`| | | | | | | | | | |

## **5. CUBLAS Level-1 Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasCaxpy`| | | | |`rocblas_caxpy`|1.5.0| | | | |
|`cublasCaxpy_64`|12.0| | | |`rocblas_caxpy_64`|6.1.0| | | | |
|`cublasCaxpy_v2`| | | | |`rocblas_caxpy`|1.5.0| | | | |
|`cublasCaxpy_v2_64`|12.0| | | |`rocblas_caxpy_64`|6.1.0| | | | |
|`cublasCcopy`| | | | |`rocblas_ccopy`|1.5.0| | | | |
|`cublasCcopy_64`|12.0| | | |`rocblas_ccopy_64`|6.1.0| | | | |
|`cublasCcopy_v2`| | | | |`rocblas_ccopy`|1.5.0| | | | |
|`cublasCcopy_v2_64`|12.0| | | |`rocblas_ccopy_64`|6.1.0| | | | |
|`cublasCdotc`| | | | |`rocblas_cdotc`|3.5.0| | | | |
|`cublasCdotc_64`|12.0| | | |`rocblas_cdotc_64`|6.1.0| | | | |
|`cublasCdotc_v2`| | | | |`rocblas_cdotc`|3.5.0| | | | |
|`cublasCdotc_v2_64`|12.0| | | |`rocblas_cdotc_64`|6.1.0| | | | |
|`cublasCdotu`| | | | |`rocblas_cdotu`|1.5.0| | | | |
|`cublasCdotu_64`|12.0| | | |`rocblas_cdotu_64`|6.1.0| | | | |
|`cublasCdotu_v2`| | | | |`rocblas_cdotu`|1.5.0| | | | |
|`cublasCdotu_v2_64`|12.0| | | |`rocblas_cdotu_64`|6.1.0| | | | |
|`cublasCrot`| | | | |`rocblas_crot`|3.5.0| | | | |
|`cublasCrot_64`|12.0| | | |`rocblas_crot_64`|6.1.0| | | | |
|`cublasCrot_v2`| | | | |`rocblas_crot`|3.5.0| | | | |
|`cublasCrot_v2_64`|12.0| | | |`rocblas_crot_64`|6.1.0| | | | |
|`cublasCrotg`| | | | |`rocblas_crotg`|3.5.0| | | | |
|`cublasCrotg_v2`| | | | |`rocblas_crotg`|3.5.0| | | | |
|`cublasCscal`| | | | |`rocblas_cscal`|1.5.0| | | | |
|`cublasCscal_64`|12.0| | | |`rocblas_cscal_64`|6.1.0| | | | |
|`cublasCscal_v2`| | | | |`rocblas_cscal`|1.5.0| | | | |
|`cublasCscal_v2_64`|12.0| | | |`rocblas_cscal_64`|6.1.0| | | | |
|`cublasCsrot`| | | | |`rocblas_csrot`|3.5.0| | | | |
|`cublasCsrot_64`|12.0| | | |`rocblas_csrot_64`|6.1.0| | | | |
|`cublasCsrot_v2`| | | | |`rocblas_csrot`|3.5.0| | | | |
|`cublasCsrot_v2_64`|12.0| | | |`rocblas_csrot_64`|6.1.0| | | | |
|`cublasCsscal`| | | | |`rocblas_csscal`|3.5.0| | | | |
|`cublasCsscal_64`|12.0| | | |`rocblas_csscal_64`|6.1.0| | | | |
|`cublasCsscal_v2`| | | | |`rocblas_csscal`|3.5.0| | | | |
|`cublasCsscal_v2_64`|12.0| | | |`rocblas_csscal_64`|6.1.0| | | | |
|`cublasCswap`| | | | |`rocblas_cswap`|1.5.0| | | | |
|`cublasCswap_64`|12.0| | | |`rocblas_cswap_64`|6.1.0| | | | |
|`cublasCswap_v2`| | | | |`rocblas_cswap`|1.5.0| | | | |
|`cublasCswap_v2_64`|12.0| | | |`rocblas_cswap_64`|6.1.0| | | | |
|`cublasDasum`| | | | |`rocblas_dasum`|1.5.0| | | | |
|`cublasDasum_64`|12.0| | | |`rocblas_dasum_64`|6.1.0| | | | |
|`cublasDasum_v2`| | | | |`rocblas_dasum`|1.5.0| | | | |
|`cublasDasum_v2_64`|12.0| | | |`rocblas_dasum_64`|6.1.0| | | | |
|`cublasDaxpy`| | | | |`rocblas_daxpy`|1.5.0| | | | |
|`cublasDaxpy_64`|12.0| | | |`rocblas_daxpy_64`|6.1.0| | | | |
|`cublasDaxpy_v2`| | | | |`rocblas_daxpy`|1.5.0| | | | |
|`cublasDaxpy_v2_64`|12.0| | | |`rocblas_daxpy_64`|6.1.0| | | | |
|`cublasDcopy`| | | | |`rocblas_dcopy`|1.5.0| | | | |
|`cublasDcopy_64`|12.0| | | |`rocblas_dcopy_64`|6.1.0| | | | |
|`cublasDcopy_v2`| | | | |`rocblas_dcopy`|1.5.0| | | | |
|`cublasDcopy_v2_64`|12.0| | | |`rocblas_dcopy_64`|6.1.0| | | | |
|`cublasDdot`| | | | |`rocblas_ddot`|1.5.0| | | | |
|`cublasDdot_64`|12.0| | | |`rocblas_ddot_64`|6.1.0| | | | |
|`cublasDdot_v2`| | | | |`rocblas_ddot`|1.5.0| | | | |
|`cublasDdot_v2_64`|12.0| | | |`rocblas_ddot_64`|6.1.0| | | | |
|`cublasDnrm2`| | | | |`rocblas_dnrm2`|1.5.0| | | | |
|`cublasDnrm2_64`|12.0| | | |`rocblas_dnrm2_64`|6.1.0| | | | |
|`cublasDnrm2_v2`| | | | |`rocblas_dnrm2`|1.5.0| | | | |
|`cublasDnrm2_v2_64`|12.0| | | |`rocblas_dnrm2_64`|6.1.0| | | | |
|`cublasDrot`| | | | |`rocblas_drot`|3.5.0| | | | |
|`cublasDrot_64`|12.0| | | |`rocblas_drot_64`|6.1.0| | | | |
|`cublasDrot_v2`| | | | |`rocblas_drot`|3.5.0| | | | |
|`cublasDrot_v2_64`|12.0| | | |`rocblas_drot_64`|6.1.0| | | | |
|`cublasDrotg`| | | | |`rocblas_drotg`|3.5.0| | | | |
|`cublasDrotg_v2`| | | | |`rocblas_drotg`|3.5.0| | | | |
|`cublasDrotm`| | | | |`rocblas_drotm`|3.5.0| | | | |
|`cublasDrotm_64`|12.0| | | |`rocblas_drotm_64`|6.1.0| | | | |
|`cublasDrotm_v2`| | | | |`rocblas_drotm`|3.5.0| | | | |
|`cublasDrotm_v2_64`|12.0| | | |`rocblas_drotm_64`|6.1.0| | | | |
|`cublasDrotmg`| | | | |`rocblas_drotmg`|3.5.0| | | | |
|`cublasDrotmg_v2`| | | | |`rocblas_drotmg`|3.5.0| | | | |
|`cublasDscal`| | | | |`rocblas_dscal`|1.5.0| | | | |
|`cublasDscal_64`|12.0| | | |`rocblas_dscal_64`|6.1.0| | | | |
|`cublasDscal_v2`| | | | |`rocblas_dscal`|1.5.0| | | | |
|`cublasDscal_v2_64`|12.0| | | |`rocblas_dscal_64`|6.1.0| | | | |
|`cublasDswap`| | | | |`rocblas_dswap`|1.5.0| | | | |
|`cublasDswap_64`|12.0| | | |`rocblas_dswap_64`|6.1.0| | | | |
|`cublasDswap_v2`| | | | |`rocblas_dswap`|1.5.0| | | | |
|`cublasDswap_v2_64`|12.0| | | |`rocblas_dswap_64`|6.1.0| | | | |
|`cublasDzasum`| | | | |`rocblas_dzasum`|1.5.0| | | | |
|`cublasDzasum_64`|12.0| | | |`rocblas_dzasum_64`|6.1.0| | | | |
|`cublasDzasum_v2`| | | | |`rocblas_dzasum`|1.5.0| | | | |
|`cublasDzasum_v2_64`|12.0| | | |`rocblas_dzasum_64`|6.1.0| | | | |
|`cublasDznrm2`| | | | |`rocblas_dznrm2`|1.5.0| | | | |
|`cublasDznrm2_64`|12.0| | | |`rocblas_dznrm2_64`|6.1.0| | | | |
|`cublasDznrm2_v2`| | | | |`rocblas_dznrm2`|1.5.0| | | | |
|`cublasDznrm2_v2_64`|12.0| | | |`rocblas_dznrm2_64`|6.1.0| | | | |
|`cublasIcamax`| | | | |`rocblas_icamax`|3.5.0| | | | |
|`cublasIcamax_64`|12.0| | | |`rocblas_icamax_64`|6.1.0| | | | |
|`cublasIcamax_v2`| | | | |`rocblas_icamax`|3.5.0| | | | |
|`cublasIcamax_v2_64`|12.0| | | |`rocblas_icamax_64`|6.1.0| | | | |
|`cublasIcamin`| | | | |`rocblas_icamin`|3.5.0| | | | |
|`cublasIcamin_64`|12.0| | | |`rocblas_icamin_64`|6.1.0| | | | |
|`cublasIcamin_v2`| | | | |`rocblas_icamin`|3.5.0| | | | |
|`cublasIcamin_v2_64`|12.0| | | |`rocblas_icamin_64`|6.1.0| | | | |
|`cublasIdamax`| | | | |`rocblas_idamax`|1.6.4| | | | |
|`cublasIdamax_64`|12.0| | | |`rocblas_idamax_64`|6.1.0| | | | |
|`cublasIdamax_v2`| | | | |`rocblas_idamax`|1.6.4| | | | |
|`cublasIdamax_v2_64`|12.0| | | |`rocblas_idamax_64`|6.1.0| | | | |
|`cublasIdamin`| | | | |`rocblas_idamin`|1.6.4| | | | |
|`cublasIdamin_64`|12.0| | | |`rocblas_idamin_64`|6.1.0| | | | |
|`cublasIdamin_v2`| | | | |`rocblas_idamin`|1.6.4| | | | |
|`cublasIdamin_v2_64`|12.0| | | |`rocblas_idamin_64`|6.1.0| | | | |
|`cublasIsamax`| | | | |`rocblas_isamax`|1.6.4| | | | |
|`cublasIsamax_64`|12.0| | | |`rocblas_isamax_64`|6.1.0| | | | |
|`cublasIsamax_v2`| | | | |`rocblas_isamax`|1.6.4| | | | |
|`cublasIsamax_v2_64`|12.0| | | |`rocblas_isamax_64`|6.1.0| | | | |
|`cublasIsamin`| | | | |`rocblas_isamin`|1.6.4| | | | |
|`cublasIsamin_64`|12.0| | | |`rocblas_isamin_64`|6.1.0| | | | |
|`cublasIsamin_v2`| | | | |`rocblas_isamin`|1.6.4| | | | |
|`cublasIsamin_v2_64`|12.0| | | |`rocblas_isamin_64`|6.1.0| | | | |
|`cublasIzamax`| | | | |`rocblas_izamax`|3.5.0| | | | |
|`cublasIzamax_64`|12.0| | | |`rocblas_izamax_64`|6.1.0| | | | |
|`cublasIzamax_v2`| | | | |`rocblas_izamax`|3.5.0| | | | |
|`cublasIzamax_v2_64`|12.0| | | |`rocblas_izamax_64`|6.1.0| | | | |
|`cublasIzamin`| | | | |`rocblas_izamin`|3.5.0| | | | |
|`cublasIzamin_64`|12.0| | | |`rocblas_izamin_64`|6.1.0| | | | |
|`cublasIzamin_v2`| | | | |`rocblas_izamin`|3.5.0| | | | |
|`cublasIzamin_v2_64`|12.0| | | |`rocblas_izamin_64`|6.1.0| | | | |
|`cublasNrm2Ex`|8.0| | | |`rocblas_nrm2_ex`|4.1.0| | | | |
|`cublasNrm2Ex_64`|12.0| | | |`rocblas_nrm2_ex_64`|6.1.0| | | | |
|`cublasSasum`| | | | |`rocblas_sasum`|1.5.0| | | | |
|`cublasSasum_64`|12.0| | | |`rocblas_sasum_64`|6.1.0| | | | |
|`cublasSasum_v2`| | | | |`rocblas_sasum`|1.5.0| | | | |
|`cublasSasum_v2_64`|12.0| | | |`rocblas_sasum_64`|6.1.0| | | | |
|`cublasSaxpy`| | | | |`rocblas_saxpy`|1.5.0| | | | |
|`cublasSaxpy_64`|12.0| | | |`rocblas_saxpy_64`|6.1.0| | | | |
|`cublasSaxpy_v2`| | | | |`rocblas_saxpy`|1.5.0| | | | |
|`cublasSaxpy_v2_64`|12.0| | | |`rocblas_saxpy_64`|6.1.0| | | | |
|`cublasScasum`| | | | |`rocblas_scasum`|1.5.0| | | | |
|`cublasScasum_64`|12.0| | | |`rocblas_scasum_64`|6.1.0| | | | |
|`cublasScasum_v2`| | | | |`rocblas_scasum`|1.5.0| | | | |
|`cublasScasum_v2_64`|12.0| | | |`rocblas_scasum_64`|6.1.0| | | | |
|`cublasScnrm2`| | | | |`rocblas_scnrm2`|1.5.0| | | | |
|`cublasScnrm2_64`|12.0| | | |`rocblas_scnrm2_64`|6.1.0| | | | |
|`cublasScnrm2_v2`| | | | |`rocblas_scnrm2`|1.5.0| | | | |
|`cublasScnrm2_v2_64`|12.0| | | |`rocblas_scnrm2_64`|6.1.0| | | | |
|`cublasScopy`| | | | |`rocblas_scopy`|1.5.0| | | | |
|`cublasScopy_64`|12.0| | | |`rocblas_scopy_64`|6.1.0| | | | |
|`cublasScopy_v2`| | | | |`rocblas_scopy`|1.5.0| | | | |
|`cublasScopy_v2_64`|12.0| | | |`rocblas_scopy_64`|6.1.0| | | | |
|`cublasSdot`| | | | |`rocblas_sdot`|1.5.0| | | | |
|`cublasSdot_64`|12.0| | | |`rocblas_sdot_64`|6.1.0| | | | |
|`cublasSdot_v2`| | | | |`rocblas_sdot`|1.5.0| | | | |
|`cublasSdot_v2_64`|12.0| | | |`rocblas_sdot_64`|6.1.0| | | | |
|`cublasSnrm2`| | | | |`rocblas_snrm2`|1.5.0| | | | |
|`cublasSnrm2_64`|12.0| | | |`rocblas_snrm2_64`|6.1.0| | | | |
|`cublasSnrm2_v2`| | | | |`rocblas_snrm2`|1.5.0| | | | |
|`cublasSnrm2_v2_64`|12.0| | | |`rocblas_snrm2_64`|6.1.0| | | | |
|`cublasSrot`| | | | |`rocblas_srot`|3.5.0| | | | |
|`cublasSrot_64`|12.0| | | |`rocblas_srot_64`|6.1.0| | | | |
|`cublasSrot_v2`| | | | |`rocblas_srot`|3.5.0| | | | |
|`cublasSrot_v2_64`|12.0| | | |`rocblas_srot_64`|6.1.0| | | | |
|`cublasSrotg`| | | | |`rocblas_srotg`|3.5.0| | | | |
|`cublasSrotg_v2`| | | | |`rocblas_srotg`|3.5.0| | | | |
|`cublasSrotm`| | | | |`rocblas_srotm`|3.5.0| | | | |
|`cublasSrotm_64`|12.0| | | |`rocblas_srotm_64`|6.1.0| | | | |
|`cublasSrotm_v2`| | | | |`rocblas_srotm`|3.5.0| | | | |
|`cublasSrotm_v2_64`|12.0| | | |`rocblas_srotm_64`|6.1.0| | | | |
|`cublasSrotmg`| | | | |`rocblas_srotmg`|3.5.0| | | | |
|`cublasSrotmg_v2`| | | | |`rocblas_srotmg`|3.5.0| | | | |
|`cublasSscal`| | | | |`rocblas_sscal`|1.5.0| | | | |
|`cublasSscal_64`|12.0| | | |`rocblas_sscal_64`|6.1.0| | | | |
|`cublasSscal_v2`| | | | |`rocblas_sscal`|1.5.0| | | | |
|`cublasSscal_v2_64`|12.0| | | |`rocblas_sscal_64`|6.1.0| | | | |
|`cublasSswap`| | | | |`rocblas_sswap`|1.5.0| | | | |
|`cublasSswap_64`|12.0| | | |`rocblas_sswap_64`|6.1.0| | | | |
|`cublasSswap_v2`| | | | |`rocblas_sswap`|1.5.0| | | | |
|`cublasSswap_v2_64`|12.0| | | |`rocblas_sswap_64`|6.1.0| | | | |
|`cublasZaxpy`| | | | |`rocblas_zaxpy`|1.5.0| | | | |
|`cublasZaxpy_64`|12.0| | | |`rocblas_zaxpy_64`|6.1.0| | | | |
|`cublasZaxpy_v2`| | | | |`rocblas_zaxpy`|1.5.0| | | | |
|`cublasZaxpy_v2_64`|12.0| | | |`rocblas_zaxpy_64`|6.1.0| | | | |
|`cublasZcopy`| | | | |`rocblas_zcopy`|1.5.0| | | | |
|`cublasZcopy_64`|12.0| | | |`rocblas_zcopy_64`|6.1.0| | | | |
|`cublasZcopy_v2`| | | | |`rocblas_zcopy`|1.5.0| | | | |
|`cublasZcopy_v2_64`|12.0| | | |`rocblas_zcopy_64`|6.1.0| | | | |
|`cublasZdotc`| | | | |`rocblas_zdotc`|3.5.0| | | | |
|`cublasZdotc_64`|12.0| | | |`rocblas_zdotc_64`|6.1.0| | | | |
|`cublasZdotc_v2`| | | | |`rocblas_zdotc`|3.5.0| | | | |
|`cublasZdotc_v2_64`|12.0| | | |`rocblas_zdotc_64`|6.1.0| | | | |
|`cublasZdotu`| | | | |`rocblas_zdotu`|1.5.0| | | | |
|`cublasZdotu_64`|12.0| | | |`rocblas_zdotu_64`|6.1.0| | | | |
|`cublasZdotu_v2`| | | | |`rocblas_zdotu`|1.5.0| | | | |
|`cublasZdotu_v2_64`|12.0| | | |`rocblas_zdotu_64`|6.1.0| | | | |
|`cublasZdrot`| | | | |`rocblas_zdrot`|3.5.0| | | | |
|`cublasZdrot_64`|12.0| | | |`rocblas_zdrot_64`|6.1.0| | | | |
|`cublasZdrot_v2`| | | | |`rocblas_zdrot`|3.5.0| | | | |
|`cublasZdrot_v2_64`|12.0| | | |`rocblas_zdrot_64`|6.1.0| | | | |
|`cublasZdscal`| | | | |`rocblas_zdscal`|3.5.0| | | | |
|`cublasZdscal_64`|12.0| | | |`rocblas_zdscal_64`|6.1.0| | | | |
|`cublasZdscal_v2`| | | | |`rocblas_zdscal`|3.5.0| | | | |
|`cublasZdscal_v2_64`|12.0| | | |`rocblas_zdscal_64`|6.1.0| | | | |
|`cublasZrot`| | | | |`rocblas_zrot`|3.5.0| | | | |
|`cublasZrot_64`|12.0| | | |`rocblas_zrot_64`|6.1.0| | | | |
|`cublasZrot_v2`| | | | |`rocblas_zrot`|3.5.0| | | | |
|`cublasZrot_v2_64`|12.0| | | |`rocblas_zrot_64`|6.1.0| | | | |
|`cublasZrotg`| | | | |`rocblas_zrotg`|3.5.0| | | | |
|`cublasZrotg_v2`| | | | |`rocblas_zrotg`|3.5.0| | | | |
|`cublasZscal`| | | | |`rocblas_zscal`|1.5.0| | | | |
|`cublasZscal_64`|12.0| | | |`rocblas_zscal_64`|6.1.0| | | | |
|`cublasZscal_v2`| | | | |`rocblas_zscal`|1.5.0| | | | |
|`cublasZscal_v2_64`|12.0| | | |`rocblas_zscal_64`|6.1.0| | | | |
|`cublasZswap`| | | | |`rocblas_zswap`|1.5.0| | | | |
|`cublasZswap_64`|12.0| | | |`rocblas_zswap_64`|6.1.0| | | | |
|`cublasZswap_v2`| | | | |`rocblas_zswap`|1.5.0| | | | |
|`cublasZswap_v2_64`|12.0| | | |`rocblas_zswap_64`|6.1.0| | | | |

## **6. CUBLAS Level-2 Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasCgbmv`| | | | |`rocblas_cgbmv`|3.5.0| | | | |
|`cublasCgbmv_64`|12.0| | | |`rocblas_cgbmv_64`|6.2.0| | | | |
|`cublasCgbmv_v2`| | | | |`rocblas_cgbmv`|3.5.0| | | | |
|`cublasCgbmv_v2_64`|12.0| | | |`rocblas_cgbmv_64`|6.2.0| | | | |
|`cublasCgemv`| | | | |`rocblas_cgemv`|1.5.0| | | | |
|`cublasCgemv_64`|12.0| | | |`rocblas_cgemv_64`|6.2.0| | | | |
|`cublasCgemv_v2`| | | | |`rocblas_cgemv`|1.5.0| | | | |
|`cublasCgemv_v2_64`|12.0| | | |`rocblas_cgemv_64`|6.2.0| | | | |
|`cublasCgerc`| | | | |`rocblas_cgerc`|3.5.0| | | | |
|`cublasCgerc_64`|12.0| | | |`rocblas_cgerc_64`|6.2.0| | | | |
|`cublasCgerc_v2`| | | | |`rocblas_cgerc`|3.5.0| | | | |
|`cublasCgerc_v2_64`|12.0| | | |`rocblas_cgerc_64`|6.2.0| | | | |
|`cublasCgeru`| | | | |`rocblas_cgeru`|3.5.0| | | | |
|`cublasCgeru_64`|12.0| | | |`rocblas_cgeru_64`|6.2.0| | | | |
|`cublasCgeru_v2`| | | | |`rocblas_cgeru`|3.5.0| | | | |
|`cublasCgeru_v2_64`|12.0| | | |`rocblas_cgeru_64`|6.2.0| | | | |
|`cublasChbmv`| | | | |`rocblas_chbmv`|3.5.0| | | | |
|`cublasChbmv_64`|12.0| | | |`rocblas_chbmv_64`|6.2.0| | | | |
|`cublasChbmv_v2`| | | | |`rocblas_chbmv`|3.5.0| | | | |
|`cublasChbmv_v2_64`|12.0| | | |`rocblas_chbmv_64`|6.2.0| | | | |
|`cublasChemv`| | | | |`rocblas_chemv`|1.5.0| | | | |
|`cublasChemv_64`|12.0| | | |`rocblas_chemv_64`|6.2.0| | | | |
|`cublasChemv_v2`| | | | |`rocblas_chemv`|1.5.0| | | | |
|`cublasChemv_v2_64`|12.0| | | |`rocblas_chemv_64`|6.2.0| | | | |
|`cublasCher`| | | | |`rocblas_cher`|3.5.0| | | | |
|`cublasCher2`| | | | |`rocblas_cher2`|3.5.0| | | | |
|`cublasCher2_64`|12.0| | | |`rocblas_cher2_64`|6.2.0| | | | |
|`cublasCher2_v2`| | | | |`rocblas_cher2`|3.5.0| | | | |
|`cublasCher2_v2_64`|12.0| | | |`rocblas_cher2_64`|6.2.0| | | | |
|`cublasCher_64`|12.0| | | |`rocblas_cher_64`|6.2.0| | | | |
|`cublasCher_v2`| | | | |`rocblas_cher`|3.5.0| | | | |
|`cublasCher_v2_64`|12.0| | | |`rocblas_cher_64`|6.2.0| | | | |
|`cublasChpmv`| | | | |`rocblas_chpmv`|3.5.0| | | | |
|`cublasChpmv_64`|12.0| | | |`rocblas_chpmv_64`|6.2.0| | | | |
|`cublasChpmv_v2`| | | | |`rocblas_chpmv`|3.5.0| | | | |
|`cublasChpmv_v2_64`|12.0| | | |`rocblas_chpmv_64`|6.2.0| | | | |
|`cublasChpr`| | | | |`rocblas_chpr`|3.5.0| | | | |
|`cublasChpr2`| | | | |`rocblas_chpr2`|3.5.0| | | | |
|`cublasChpr2_64`|12.0| | | |`rocblas_chpr2_64`|6.2.0| | | | |
|`cublasChpr2_v2`| | | | |`rocblas_chpr2`|3.5.0| | | | |
|`cublasChpr2_v2_64`|12.0| | | |`rocblas_chpr2_64`|6.2.0| | | | |
|`cublasChpr_64`|12.0| | | |`rocblas_chpr_64`|6.2.0| | | | |
|`cublasChpr_v2`| | | | |`rocblas_chpr`|3.5.0| | | | |
|`cublasChpr_v2_64`|12.0| | | |`rocblas_chpr_64`|6.2.0| | | | |
|`cublasCsymv`| | | | |`rocblas_csymv`|3.5.0| | | | |
|`cublasCsymv_64`|12.0| | | |`rocblas_csymv_64`|6.2.0| | | | |
|`cublasCsymv_v2`| | | | |`rocblas_csymv`|3.5.0| | | | |
|`cublasCsymv_v2_64`|12.0| | | |`rocblas_csymv_64`|6.2.0| | | | |
|`cublasCsyr`| | | | |`rocblas_csyr`|1.7.1| | | | |
|`cublasCsyr2`| | | | |`rocblas_csyr2`|3.5.0| | | | |
|`cublasCsyr2_64`|12.0| | | |`rocblas_csyr2_64`|6.2.0| | | | |
|`cublasCsyr2_v2`| | | | |`rocblas_csyr2`|3.5.0| | | | |
|`cublasCsyr2_v2_64`|12.0| | | |`rocblas_csyr2_64`|6.2.0| | | | |
|`cublasCsyr_64`|12.0| | | |`rocblas_csyr_64`|6.2.0| | | | |
|`cublasCsyr_v2`| | | | |`rocblas_csyr`|1.7.1| | | | |
|`cublasCsyr_v2_64`|12.0| | | |`rocblas_csyr_64`|6.2.0| | | | |
|`cublasCtbmv`| | | | |`rocblas_ctbmv`|3.5.0| | | | |
|`cublasCtbmv_64`|12.0| | | |`rocblas_ctbmv_64`|6.2.0| | | | |
|`cublasCtbmv_v2`| | | | |`rocblas_ctbmv`|3.5.0| | | | |
|`cublasCtbmv_v2_64`|12.0| | | |`rocblas_ctbmv_64`|6.2.0| | | | |
|`cublasCtbsv`| | | | |`rocblas_ctbsv`|3.5.0| | | | |
|`cublasCtbsv_64`|12.0| | | |`rocblas_ctbsv_64`|6.2.0| | | | |
|`cublasCtbsv_v2`| | | | |`rocblas_ctbsv`|3.5.0| | | | |
|`cublasCtbsv_v2_64`|12.0| | | |`rocblas_ctbsv_64`|6.2.0| | | | |
|`cublasCtpmv`| | | | |`rocblas_ctpmv`|3.5.0| | | | |
|`cublasCtpmv_64`|12.0| | | |`rocblas_ctpmv_64`|6.2.0| | | | |
|`cublasCtpmv_v2`| | | | |`rocblas_ctpmv`|3.5.0| | | | |
|`cublasCtpmv_v2_64`|12.0| | | |`rocblas_ctpmv_64`|6.2.0| | | | |
|`cublasCtpsv`| | | | |`rocblas_ctpsv`|3.5.0| | | | |
|`cublasCtpsv_64`|12.0| | | |`rocblas_ctpsv_64`|6.2.0| | | | |
|`cublasCtpsv_v2`| | | | |`rocblas_ctpsv`|3.5.0| | | | |
|`cublasCtpsv_v2_64`|12.0| | | |`rocblas_ctpsv_64`|6.2.0| | | | |
|`cublasCtrmv`| | | | |`rocblas_ctrmv`|3.5.0| | | | |
|`cublasCtrmv_64`|12.0| | | |`rocblas_ctrmv_64`|6.2.0| | | | |
|`cublasCtrmv_v2`| | | | |`rocblas_ctrmv`|3.5.0| | | | |
|`cublasCtrmv_v2_64`|12.0| | | |`rocblas_ctrmv_64`|6.2.0| | | | |
|`cublasCtrsv`| | | | |`rocblas_ctrsv`|3.5.0| | | | |
|`cublasCtrsv_64`|12.0| | | |`rocblas_ctrsv_64`|6.2.0| | | | |
|`cublasCtrsv_v2`| | | | |`rocblas_ctrsv`|3.5.0| | | | |
|`cublasCtrsv_v2_64`|12.0| | | |`rocblas_ctrsv_64`|6.2.0| | | | |
|`cublasDgbmv`| | | | |`rocblas_dgbmv`|3.5.0| | | | |
|`cublasDgbmv_64`|12.0| | | |`rocblas_dgbmv_64`|6.2.0| | | | |
|`cublasDgbmv_v2`| | | | |`rocblas_dgbmv`|3.5.0| | | | |
|`cublasDgbmv_v2_64`|12.0| | | |`rocblas_dgbmv_64`|6.2.0| | | | |
|`cublasDgemv`| | | | |`rocblas_dgemv`|1.5.0| | | | |
|`cublasDgemv_64`|12.0| | | |`rocblas_dgemv_64`|6.2.0| | | | |
|`cublasDgemv_v2`| | | | |`rocblas_dgemv`|1.5.0| | | | |
|`cublasDgemv_v2_64`|12.0| | | |`rocblas_dgemv_64`|6.2.0| | | | |
|`cublasDger`| | | | |`rocblas_dger`|1.5.0| | | | |
|`cublasDger_64`|12.0| | | |`rocblas_dger_64`|6.2.0| | | | |
|`cublasDger_v2`| | | | |`rocblas_dger`|1.5.0| | | | |
|`cublasDger_v2_64`|12.0| | | |`rocblas_dger_64`|6.2.0| | | | |
|`cublasDsbmv`| | | | |`rocblas_dsbmv`|3.5.0| | | | |
|`cublasDsbmv_64`|12.0| | | |`rocblas_dsbmv_64`|6.2.0| | | | |
|`cublasDsbmv_v2`| | | | |`rocblas_dsbmv`|3.5.0| | | | |
|`cublasDsbmv_v2_64`|12.0| | | |`rocblas_dsbmv_64`|6.2.0| | | | |
|`cublasDspmv`| | | | |`rocblas_dspmv`|3.5.0| | | | |
|`cublasDspmv_64`|12.0| | | |`rocblas_dspmv_64`|6.2.0| | | | |
|`cublasDspmv_v2`| | | | |`rocblas_dspmv`|3.5.0| | | | |
|`cublasDspmv_v2_64`|12.0| | | |`rocblas_dspmv_64`|6.2.0| | | | |
|`cublasDspr`| | | | |`rocblas_dspr`|3.5.0| | | | |
|`cublasDspr2`| | | | |`rocblas_dspr2`|3.5.0| | | | |
|`cublasDspr2_64`|12.0| | | |`rocblas_dspr2_64`|6.2.0| | | | |
|`cublasDspr2_v2`| | | | |`rocblas_dspr2`|3.5.0| | | | |
|`cublasDspr2_v2_64`|12.0| | | |`rocblas_dspr2_64`|6.2.0| | | | |
|`cublasDspr_64`|12.0| | | |`rocblas_dspr_64`|6.2.0| | | | |
|`cublasDspr_v2`| | | | |`rocblas_dspr`|3.5.0| | | | |
|`cublasDspr_v2_64`|12.0| | | |`rocblas_dspr_64`|6.2.0| | | | |
|`cublasDsymv`| | | | |`rocblas_dsymv`|1.5.0| | | | |
|`cublasDsymv_64`|12.0| | | |`rocblas_dsymv_64`|6.2.0| | | | |
|`cublasDsymv_v2`| | | | |`rocblas_dsymv`|1.5.0| | | | |
|`cublasDsymv_v2_64`|12.0| | | |`rocblas_dsymv_64`|6.2.0| | | | |
|`cublasDsyr`| | | | |`rocblas_dsyr`|1.7.1| | | | |
|`cublasDsyr2`| | | | |`rocblas_dsyr2`|3.5.0| | | | |
|`cublasDsyr2_64`|12.0| | | |`rocblas_dsyr2_64`|6.2.0| | | | |
|`cublasDsyr2_v2`| | | | |`rocblas_dsyr2`|3.5.0| | | | |
|`cublasDsyr2_v2_64`|12.0| | | |`rocblas_dsyr2_64`|6.2.0| | | | |
|`cublasDsyr_64`|12.0| | | |`rocblas_dsyr_64`|6.2.0| | | | |
|`cublasDsyr_v2`| | | | |`rocblas_dsyr`|1.7.1| | | | |
|`cublasDsyr_v2_64`|12.0| | | |`rocblas_dsyr_64`|6.2.0| | | | |
|`cublasDtbmv`| | | | |`rocblas_dtbmv`|3.5.0| | | | |
|`cublasDtbmv_64`|12.0| | | |`rocblas_dtbmv_64`|6.2.0| | | | |
|`cublasDtbmv_v2`| | | | |`rocblas_dtbmv`|3.5.0| | | | |
|`cublasDtbmv_v2_64`|12.0| | | |`rocblas_dtbmv_64`|6.2.0| | | | |
|`cublasDtbsv`| | | | |`rocblas_dtbsv`|3.5.0| | | | |
|`cublasDtbsv_64`|12.0| | | |`rocblas_dtbsv_64`|6.2.0| | | | |
|`cublasDtbsv_v2`| | | | |`rocblas_dtbsv`|3.5.0| | | | |
|`cublasDtbsv_v2_64`|12.0| | | |`rocblas_dtbsv_64`|6.2.0| | | | |
|`cublasDtpmv`| | | | |`rocblas_dtpmv`|3.5.0| | | | |
|`cublasDtpmv_64`|12.0| | | |`rocblas_dtpmv_64`|6.2.0| | | | |
|`cublasDtpmv_v2`| | | | |`rocblas_dtpmv`|3.5.0| | | | |
|`cublasDtpmv_v2_64`|12.0| | | |`rocblas_dtpmv_64`|6.2.0| | | | |
|`cublasDtpsv`| | | | |`rocblas_dtpsv`|3.5.0| | | | |
|`cublasDtpsv_64`|12.0| | | |`rocblas_dtpsv_64`|6.2.0| | | | |
|`cublasDtpsv_v2`| | | | |`rocblas_dtpsv`|3.5.0| | | | |
|`cublasDtpsv_v2_64`|12.0| | | |`rocblas_dtpsv_64`|6.2.0| | | | |
|`cublasDtrmv`| | | | |`rocblas_dtrmv`|3.5.0| | | | |
|`cublasDtrmv_64`|12.0| | | |`rocblas_dtrmv_64`|6.2.0| | | | |
|`cublasDtrmv_v2`| | | | |`rocblas_dtrmv`|3.5.0| | | | |
|`cublasDtrmv_v2_64`|12.0| | | |`rocblas_dtrmv_64`|6.2.0| | | | |
|`cublasDtrsv`| | | | |`rocblas_dtrsv`|3.5.0| | | | |
|`cublasDtrsv_64`|12.0| | | |`rocblas_dtrsv_64`|6.2.0| | | | |
|`cublasDtrsv_v2`| | | | |`rocblas_dtrsv`|3.5.0| | | | |
|`cublasDtrsv_v2_64`|12.0| | | |`rocblas_dtrsv_64`|6.2.0| | | | |
|`cublasSgbmv`| | | | |`rocblas_sgbmv`|3.5.0| | | | |
|`cublasSgbmv_64`|12.0| | | |`rocblas_sgbmv_64`|6.2.0| | | | |
|`cublasSgbmv_v2`| | | | |`rocblas_sgbmv`|3.5.0| | | | |
|`cublasSgbmv_v2_64`|12.0| | | |`rocblas_sgbmv_64`|6.2.0| | | | |
|`cublasSgemv`| | | | |`rocblas_sgemv`|1.5.0| | | | |
|`cublasSgemv_64`|12.0| | | |`rocblas_sgemv_64`|6.2.0| | | | |
|`cublasSgemv_v2`| | | | |`rocblas_sgemv`|1.5.0| | | | |
|`cublasSgemv_v2_64`|12.0| | | |`rocblas_sgemv_64`|6.2.0| | | | |
|`cublasSger`| | | | |`rocblas_sger`|1.5.0| | | | |
|`cublasSger_64`|12.0| | | |`rocblas_sger_64`|6.2.0| | | | |
|`cublasSger_v2`| | | | |`rocblas_sger`|1.5.0| | | | |
|`cublasSger_v2_64`|12.0| | | |`rocblas_sger_64`|6.2.0| | | | |
|`cublasSsbmv`| | | | |`rocblas_ssbmv`|3.5.0| | | | |
|`cublasSsbmv_64`|12.0| | | |`rocblas_ssbmv_64`|6.2.0| | | | |
|`cublasSsbmv_v2`| | | | |`rocblas_ssbmv`|3.5.0| | | | |
|`cublasSsbmv_v2_64`|12.0| | | |`rocblas_ssbmv_64`|6.2.0| | | | |
|`cublasSspmv`| | | | |`rocblas_sspmv`|3.5.0| | | | |
|`cublasSspmv_64`|12.0| | | |`rocblas_sspmv_64`|6.2.0| | | | |
|`cublasSspmv_v2`| | | | |`rocblas_sspmv`|3.5.0| | | | |
|`cublasSspmv_v2_64`|12.0| | | |`rocblas_sspmv_64`|6.2.0| | | | |
|`cublasSspr`| | | | |`rocblas_sspr`|3.5.0| | | | |
|`cublasSspr2`| | | | |`rocblas_sspr2`|3.5.0| | | | |
|`cublasSspr2_64`|12.0| | | |`rocblas_sspr2_64`|6.2.0| | | | |
|`cublasSspr2_v2`| | | | |`rocblas_sspr2`|3.5.0| | | | |
|`cublasSspr2_v2_64`|12.0| | | |`rocblas_sspr2_64`|6.2.0| | | | |
|`cublasSspr_64`|12.0| | | |`rocblas_sspr_64`|6.2.0| | | | |
|`cublasSspr_v2`| | | | |`rocblas_sspr`|3.5.0| | | | |
|`cublasSspr_v2_64`|12.0| | | |`rocblas_sspr_64`|6.2.0| | | | |
|`cublasSsymv`| | | | |`rocblas_ssymv`|1.5.0| | | | |
|`cublasSsymv_64`|12.0| | | |`rocblas_ssymv_64`|6.2.0| | | | |
|`cublasSsymv_v2`| | | | |`rocblas_ssymv`|1.5.0| | | | |
|`cublasSsymv_v2_64`|12.0| | | |`rocblas_ssymv_64`|6.2.0| | | | |
|`cublasSsyr`| | | | |`rocblas_ssyr`|1.7.1| | | | |
|`cublasSsyr2`| | | | |`rocblas_ssyr2`|3.5.0| | | | |
|`cublasSsyr2_64`|12.0| | | |`rocblas_ssyr2_64`|6.2.0| | | | |
|`cublasSsyr2_v2`| | | | |`rocblas_ssyr2`|3.5.0| | | | |
|`cublasSsyr2_v2_64`|12.0| | | |`rocblas_ssyr2_64`|6.2.0| | | | |
|`cublasSsyr_64`|12.0| | | |`rocblas_ssyr_64`|6.2.0| | | | |
|`cublasSsyr_v2`| | | | |`rocblas_ssyr`|1.7.1| | | | |
|`cublasSsyr_v2_64`|12.0| | | |`rocblas_ssyr_64`|6.2.0| | | | |
|`cublasStbmv`| | | | |`rocblas_stbmv`|3.5.0| | | | |
|`cublasStbmv_64`|12.0| | | |`rocblas_stbmv_64`|6.2.0| | | | |
|`cublasStbmv_v2`| | | | |`rocblas_stbmv`|3.5.0| | | | |
|`cublasStbmv_v2_64`|12.0| | | |`rocblas_stbmv_64`|6.2.0| | | | |
|`cublasStbsv`| | | | |`rocblas_stbsv`|3.5.0| | | | |
|`cublasStbsv_64`|12.0| | | |`rocblas_stbsv_64`|6.2.0| | | | |
|`cublasStbsv_v2`| | | | |`rocblas_stbsv`|3.5.0| | | | |
|`cublasStbsv_v2_64`|12.0| | | |`rocblas_stbsv_64`|6.2.0| | | | |
|`cublasStpmv`| | | | |`rocblas_stpmv`|3.5.0| | | | |
|`cublasStpmv_64`|12.0| | | |`rocblas_stpmv_64`|6.2.0| | | | |
|`cublasStpmv_v2`| | | | |`rocblas_stpmv`|3.5.0| | | | |
|`cublasStpmv_v2_64`|12.0| | | |`rocblas_stpmv_64`|6.2.0| | | | |
|`cublasStpsv`| | | | |`rocblas_stpsv`|3.5.0| | | | |
|`cublasStpsv_64`|12.0| | | |`rocblas_stpsv_64`|6.2.0| | | | |
|`cublasStpsv_v2`| | | | |`rocblas_stpsv`|3.5.0| | | | |
|`cublasStpsv_v2_64`|12.0| | | |`rocblas_stpsv_64`|6.2.0| | | | |
|`cublasStrmv`| | | | |`rocblas_strmv`|3.5.0| | | | |
|`cublasStrmv_64`|12.0| | | |`rocblas_strmv_64`|6.2.0| | | | |
|`cublasStrmv_v2`| | | | |`rocblas_strmv`|3.5.0| | | | |
|`cublasStrmv_v2_64`|12.0| | | |`rocblas_strmv_64`|6.2.0| | | | |
|`cublasStrsv`| | | | |`rocblas_strsv`|3.5.0| | | | |
|`cublasStrsv_64`|12.0| | | |`rocblas_strsv_64`|6.2.0| | | | |
|`cublasStrsv_v2`| | | | |`rocblas_strsv`|3.5.0| | | | |
|`cublasStrsv_v2_64`|12.0| | | |`rocblas_strsv_64`|6.2.0| | | | |
|`cublasZgbmv`| | | | |`rocblas_zgbmv`|3.5.0| | | | |
|`cublasZgbmv_64`|12.0| | | |`rocblas_zgbmv_64`|6.2.0| | | | |
|`cublasZgbmv_v2`| | | | |`rocblas_zgbmv`|3.5.0| | | | |
|`cublasZgbmv_v2_64`|12.0| | | |`rocblas_zgbmv_64`|6.2.0| | | | |
|`cublasZgemv`| | | | |`rocblas_zgemv`|1.5.0| | | | |
|`cublasZgemv_64`|12.0| | | |`rocblas_zgemv_64`|6.2.0| | | | |
|`cublasZgemv_v2`| | | | |`rocblas_zgemv`|1.5.0| | | | |
|`cublasZgemv_v2_64`|12.0| | | |`rocblas_zgemv_64`|6.2.0| | | | |
|`cublasZgerc`| | | | |`rocblas_zgerc`|3.5.0| | | | |
|`cublasZgerc_64`|12.0| | | |`rocblas_zgerc_64`|6.2.0| | | | |
|`cublasZgerc_v2`| | | | |`rocblas_zgerc`|3.5.0| | | | |
|`cublasZgerc_v2_64`|12.0| | | |`rocblas_zgerc_64`|6.2.0| | | | |
|`cublasZgeru`| | | | |`rocblas_zgeru`|3.5.0| | | | |
|`cublasZgeru_64`|12.0| | | |`rocblas_zgeru_64`|6.2.0| | | | |
|`cublasZgeru_v2`| | | | |`rocblas_zgeru`|3.5.0| | | | |
|`cublasZgeru_v2_64`|12.0| | | |`rocblas_zgeru_64`|6.2.0| | | | |
|`cublasZhbmv`| | | | |`rocblas_zhbmv`|3.5.0| | | | |
|`cublasZhbmv_64`|12.0| | | |`rocblas_zhbmv_64`|6.2.0| | | | |
|`cublasZhbmv_v2`| | | | |`rocblas_zhbmv`|3.5.0| | | | |
|`cublasZhbmv_v2_64`|12.0| | | |`rocblas_zhbmv_64`|6.2.0| | | | |
|`cublasZhemv`| | | | |`rocblas_zhemv`|1.5.0| | | | |
|`cublasZhemv_64`|12.0| | | |`rocblas_zhemv_64`|6.2.0| | | | |
|`cublasZhemv_v2`| | | | |`rocblas_zhemv`|1.5.0| | | | |
|`cublasZhemv_v2_64`|12.0| | | |`rocblas_zhemv_64`|6.2.0| | | | |
|`cublasZher`| | | | |`rocblas_zher`|3.5.0| | | | |
|`cublasZher2`| | | | |`rocblas_zher2`|3.5.0| | | | |
|`cublasZher2_64`|12.0| | | |`rocblas_zher2_64`|6.2.0| | | | |
|`cublasZher2_v2`| | | | |`rocblas_zher2`|3.5.0| | | | |
|`cublasZher2_v2_64`|12.0| | | |`rocblas_zher2_64`|6.2.0| | | | |
|`cublasZher_64`|12.0| | | |`rocblas_zher_64`|6.2.0| | | | |
|`cublasZher_v2`| | | | |`rocblas_zher`|3.5.0| | | | |
|`cublasZher_v2_64`|12.0| | | |`rocblas_zher_64`|6.2.0| | | | |
|`cublasZhpmv`| | | | |`rocblas_zhpmv`|3.5.0| | | | |
|`cublasZhpmv_64`|12.0| | | |`rocblas_zhpmv_64`|6.2.0| | | | |
|`cublasZhpmv_v2`| | | | |`rocblas_zhpmv`|3.5.0| | | | |
|`cublasZhpmv_v2_64`|12.0| | | |`rocblas_zhpmv_64`|6.2.0| | | | |
|`cublasZhpr`| | | | |`rocblas_zhpr`|3.5.0| | | | |
|`cublasZhpr2`| | | | |`rocblas_zhpr2`|3.5.0| | | | |
|`cublasZhpr2_64`|12.0| | | |`rocblas_zhpr2_64`|6.2.0| | | | |
|`cublasZhpr2_v2`| | | | |`rocblas_zhpr2`|3.5.0| | | | |
|`cublasZhpr2_v2_64`|12.0| | | |`rocblas_zhpr2_64`|6.2.0| | | | |
|`cublasZhpr_64`|12.0| | | |`rocblas_zhpr_64`|6.2.0| | | | |
|`cublasZhpr_v2`| | | | |`rocblas_zhpr`|3.5.0| | | | |
|`cublasZhpr_v2_64`|12.0| | | |`rocblas_zhpr_64`|6.2.0| | | | |
|`cublasZsymv`| | | | |`rocblas_zsymv`|3.5.0| | | | |
|`cublasZsymv_64`|12.0| | | |`rocblas_zsymv_64`|6.2.0| | | | |
|`cublasZsymv_v2`| | | | |`rocblas_zsymv`|3.5.0| | | | |
|`cublasZsymv_v2_64`|12.0| | | |`rocblas_zsymv_64`|6.2.0| | | | |
|`cublasZsyr`| | | | |`rocblas_zsyr`|1.7.1| | | | |
|`cublasZsyr2`| | | | |`rocblas_zsyr2`|3.5.0| | | | |
|`cublasZsyr2_64`|12.0| | | |`rocblas_zsyr2_64`|6.2.0| | | | |
|`cublasZsyr2_v2`| | | | |`rocblas_zsyr2`|3.5.0| | | | |
|`cublasZsyr2_v2_64`|12.0| | | |`rocblas_zsyr2_64`|6.2.0| | | | |
|`cublasZsyr_64`|12.0| | | |`rocblas_zsyr_64`|6.2.0| | | | |
|`cublasZsyr_v2`| | | | |`rocblas_zsyr`|1.7.1| | | | |
|`cublasZsyr_v2_64`|12.0| | | |`rocblas_zsyr_64`|6.2.0| | | | |
|`cublasZtbmv`| | | | |`rocblas_ztbmv`|3.5.0| | | | |
|`cublasZtbmv_64`|12.0| | | |`rocblas_ztbmv_64`|6.2.0| | | | |
|`cublasZtbmv_v2`| | | | |`rocblas_ztbmv`|3.5.0| | | | |
|`cublasZtbmv_v2_64`|12.0| | | |`rocblas_ztbmv_64`|6.2.0| | | | |
|`cublasZtbsv`| | | | |`rocblas_ztbsv`|3.5.0| | | | |
|`cublasZtbsv_64`|12.0| | | |`rocblas_ztbsv_64`|6.2.0| | | | |
|`cublasZtbsv_v2`| | | | |`rocblas_ztbsv`|3.5.0| | | | |
|`cublasZtbsv_v2_64`|12.0| | | |`rocblas_ztbsv_64`|6.2.0| | | | |
|`cublasZtpmv`| | | | |`rocblas_ztpmv`|3.5.0| | | | |
|`cublasZtpmv_64`|12.0| | | |`rocblas_ztpmv_64`|6.2.0| | | | |
|`cublasZtpmv_v2`| | | | |`rocblas_ztpmv`|3.5.0| | | | |
|`cublasZtpmv_v2_64`|12.0| | | |`rocblas_ztpmv_64`|6.2.0| | | | |
|`cublasZtpsv`| | | | |`rocblas_ztpsv`|3.5.0| | | | |
|`cublasZtpsv_64`|12.0| | | |`rocblas_ztpsv_64`|6.2.0| | | | |
|`cublasZtpsv_v2`| | | | |`rocblas_ztpsv`|3.5.0| | | | |
|`cublasZtpsv_v2_64`|12.0| | | |`rocblas_ztpsv_64`|6.2.0| | | | |
|`cublasZtrmv`| | | | |`rocblas_ztrmv`|3.5.0| | | | |
|`cublasZtrmv_64`|12.0| | | |`rocblas_ztrmv_64`|6.2.0| | | | |
|`cublasZtrmv_v2`| | | | |`rocblas_ztrmv`|3.5.0| | | | |
|`cublasZtrmv_v2_64`|12.0| | | |`rocblas_ztrmv_64`|6.2.0| | | | |
|`cublasZtrsv`| | | | |`rocblas_ztrsv`|3.5.0| | | | |
|`cublasZtrsv_64`|12.0| | | |`rocblas_ztrsv_64`|6.2.0| | | | |
|`cublasZtrsv_v2`| | | | |`rocblas_ztrsv`|3.5.0| | | | |
|`cublasZtrsv_v2_64`|12.0| | | |`rocblas_ztrsv_64`|6.2.0| | | | |

## **7. CUBLAS Level-3 Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasCgemm`| | | | |`rocblas_cgemm`|1.5.0| | | | |
|`cublasCgemm3m`|8.0| | | | | | | | | |
|`cublasCgemm3mBatched`|8.0| | | | | | | | | |
|`cublasCgemm3mBatched_64`|12.0| | | | | | | | | |
|`cublasCgemm3mEx`|8.0| | | | | | | | | |
|`cublasCgemm3mEx_64`|12.0| | | | | | | | | |
|`cublasCgemm3mStridedBatched`|8.0| | | | | | | | | |
|`cublasCgemm3mStridedBatched_64`|12.0| | | | | | | | | |
|`cublasCgemm3m_64`|12.0| | | | | | | | | |
|`cublasCgemmBatched`| | | | |`rocblas_cgemm_batched`|3.5.0| | | | |
|`cublasCgemmBatched_64`|12.0| | | |`rocblas_cgemm_batched_64`|6.3.0| | | | |
|`cublasCgemmStridedBatched`|8.0| | | |`rocblas_cgemm_strided_batched`|1.5.0| | | | |
|`cublasCgemmStridedBatched_64`|12.0| | | |`rocblas_cgemm_strided_batched_64`|6.3.0| | | | |
|`cublasCgemm_64`|12.0| | | |`rocblas_cgemm_64`|6.3.0| | | | |
|`cublasCgemm_v2`| | | | |`rocblas_cgemm`|1.5.0| | | | |
|`cublasCgemm_v2_64`|12.0| | | |`rocblas_cgemm_64`|6.3.0| | | | |
|`cublasCgemvBatched`|11.6| | | |`rocblas_cgemv_batched`|3.5.0| | | | |
|`cublasCgemvBatched_64`|12.0| | | |`rocblas_cgemv_batched_64`|6.2.0| | | | |
|`cublasCgemvStridedBatched`|11.6| | | |`rocblas_cgemv_strided_batched`|3.5.0| | | | |
|`cublasCgemvStridedBatched_64`|12.0| | | |`rocblas_cgemv_strided_batched_64`|6.2.0| | | | |
|`cublasChemm`| | | | |`rocblas_chemm`|3.5.0| | | | |
|`cublasChemm_64`|12.0| | | |`rocblas_chemm_64`|6.3.0| | | | |
|`cublasChemm_v2`| | | | |`rocblas_chemm`|3.5.0| | | | |
|`cublasChemm_v2_64`|12.0| | | |`rocblas_chemm_64`|6.3.0| | | | |
|`cublasCher2k`| | | | |`rocblas_cher2k`|3.5.0| | | | |
|`cublasCher2k_64`|12.0| | | |`rocblas_cher2k_64`|6.3.0| | | | |
|`cublasCher2k_v2`| | | | |`rocblas_cher2k`|3.5.0| | | | |
|`cublasCher2k_v2_64`|12.0| | | |`rocblas_cher2k_64`|6.3.0| | | | |
|`cublasCherk`| | | | |`rocblas_cherk`|3.5.0| | | | |
|`cublasCherk_64`|12.0| | | |`rocblas_cherk_64`|6.3.0| | | | |
|`cublasCherk_v2`| | | | |`rocblas_cherk`|3.5.0| | | | |
|`cublasCherk_v2_64`|12.0| | | |`rocblas_cherk_64`|6.3.0| | | | |
|`cublasCherkx`| | | | |`rocblas_cherkx`|3.5.0| | | | |
|`cublasCherkx_64`|12.0| | | |`rocblas_cherkx_64`|6.3.0| | | | |
|`cublasCsymm`| | | | |`rocblas_csymm`|3.5.0| | | | |
|`cublasCsymm_64`|12.0| | | |`rocblas_csymm_64`|6.3.0| | | | |
|`cublasCsymm_v2`| | | | |`rocblas_csymm`|3.5.0| | | | |
|`cublasCsymm_v2_64`|12.0| | | |`rocblas_csymm_64`|6.3.0| | | | |
|`cublasCsyr2k`| | | | |`rocblas_csyr2k`|3.5.0| | | | |
|`cublasCsyr2k_64`|12.0| | | |`rocblas_csyr2k_64`|6.3.0| | | | |
|`cublasCsyr2k_v2`| | | | |`rocblas_csyr2k`|3.5.0| | | | |
|`cublasCsyr2k_v2_64`|12.0| | | |`rocblas_csyr2k_64`|6.3.0| | | | |
|`cublasCsyrk`| | | | |`rocblas_csyrk`|3.5.0| | | | |
|`cublasCsyrk_64`|12.0| | | |`rocblas_csyrk_64`|6.3.0| | | | |
|`cublasCsyrk_v2`| | | | |`rocblas_csyrk`|3.5.0| | | | |
|`cublasCsyrk_v2_64`|12.0| | | |`rocblas_csyrk_64`|6.3.0| | | | |
|`cublasCsyrkx`| | | | |`rocblas_csyrkx`|3.5.0| | | | |
|`cublasCsyrkx_64`|12.0| | | |`rocblas_csyrkx_64`|6.3.0| | | | |
|`cublasCtrmm`| | | | |`rocblas_ctrmm`|3.5.0| |6.0.0| | |
|`cublasCtrmm_64`|12.0| | | |`rocblas_ctrmm_64`|6.3.0| | | | |
|`cublasCtrmm_v2`| | | | |`rocblas_ctrmm`|3.5.0| |6.0.0| | |
|`cublasCtrmm_v2_64`|12.0| | | |`rocblas_ctrmm_64`|6.3.0| | | | |
|`cublasCtrsm`| | | | |`rocblas_ctrsm`|3.5.0| | | | |
|`cublasCtrsm_64`|12.0| | | |`rocblas_ctrsm_64`|6.2.0| | | | |
|`cublasCtrsm_v2`| | | | |`rocblas_ctrsm`|3.5.0| | | | |
|`cublasCtrsm_v2_64`|12.0| | | |`rocblas_ctrsm_64`|6.2.0| | | | |
|`cublasDgemm`| | | | |`rocblas_dgemm`|1.5.0| | | | |
|`cublasDgemmBatched`| | | | |`rocblas_dgemm_batched`|3.5.0| | | | |
|`cublasDgemmBatched_64`|12.0| | | |`rocblas_dgemm_batched_64`|6.3.0| | | | |
|`cublasDgemmGroupedBatched`|12.4| | | | | | | | | |
|`cublasDgemmGroupedBatched_64`|12.4| | | | | | | | | |
|`cublasDgemmStridedBatched`|8.0| | | |`rocblas_dgemm_strided_batched`|1.5.0| | | | |
|`cublasDgemmStridedBatched_64`|12.0| | | |`rocblas_dgemm_strided_batched_64`|6.3.0| | | | |
|`cublasDgemm_64`|12.0| | | |`rocblas_dgemm_64`|6.3.0| | | | |
|`cublasDgemm_v2`| | | | |`rocblas_dgemm`|1.5.0| | | | |
|`cublasDgemm_v2_64`|12.0| | | |`rocblas_dgemm_64`|6.3.0| | | | |
|`cublasDgemvBatched`|11.6| | | |`rocblas_dgemv_batched`|3.5.0| | | | |
|`cublasDgemvBatched_64`|12.0| | | |`rocblas_dgemv_batched_64`|6.2.0| | | | |
|`cublasDgemvStridedBatched`|11.6| | | |`rocblas_dgemv_strided_batched`|3.5.0| | | | |
|`cublasDgemvStridedBatched_64`|12.0| | | |`rocblas_dgemv_strided_batched_64`|6.2.0| | | | |
|`cublasDsymm`| | | | |`rocblas_dsymm`|3.5.0| | | | |
|`cublasDsymm_64`|12.0| | | |`rocblas_dsymm_64`|6.3.0| | | | |
|`cublasDsymm_v2`| | | | |`rocblas_dsymm`|3.5.0| | | | |
|`cublasDsymm_v2_64`|12.0| | | |`rocblas_dsymm_64`|6.3.0| | | | |
|`cublasDsyr2k`| | | | |`rocblas_dsyr2k`|3.5.0| | | | |
|`cublasDsyr2k_64`|12.0| | | |`rocblas_dsyr2k_64`|6.3.0| | | | |
|`cublasDsyr2k_v2`| | | | |`rocblas_dsyr2k`|3.5.0| | | | |
|`cublasDsyr2k_v2_64`|12.0| | | |`rocblas_dsyr2k_64`|6.3.0| | | | |
|`cublasDsyrk`| | | | |`rocblas_dsyrk`|3.5.0| | | | |
|`cublasDsyrk_64`|12.0| | | |`rocblas_dsyrk_64`|6.3.0| | | | |
|`cublasDsyrk_v2`| | | | |`rocblas_dsyrk`|3.5.0| | | | |
|`cublasDsyrk_v2_64`|12.0| | | |`rocblas_dsyrk_64`|6.3.0| | | | |
|`cublasDsyrkx`| | | | |`rocblas_dsyrkx`|3.5.0| | | | |
|`cublasDsyrkx_64`|12.0| | | |`rocblas_dsyrkx_64`|6.3.0| | | | |
|`cublasDtrmm`| | | | |`rocblas_dtrmm`|3.5.0| |6.0.0| | |
|`cublasDtrmm_64`|12.0| | | |`rocblas_dtrmm_64`|6.3.0| | | | |
|`cublasDtrmm_v2`| | | | |`rocblas_dtrmm`|3.5.0| |6.0.0| | |
|`cublasDtrmm_v2_64`|12.0| | | |`rocblas_dtrmm_64`|6.3.0| | | | |
|`cublasDtrsm`| | | | |`rocblas_dtrsm`|1.5.0| | | | |
|`cublasDtrsm_64`|12.0| | | |`rocblas_dtrsm_64`|6.2.0| | | | |
|`cublasDtrsm_v2`| | | | |`rocblas_dtrsm`|1.5.0| | | | |
|`cublasDtrsm_v2_64`|12.0| | | |`rocblas_dtrsm_64`|6.2.0| | | | |
|`cublasGemmGroupedBatchedEx`|12.5| | | | | | | | | |
|`cublasGemmGroupedBatchedEx_64`|12.5| | | | | | | | | |
|`cublasHSHgemvBatched`|11.6| | | |`rocblas_hshgemv_batched`|6.0.0| | | | |
|`cublasHSHgemvBatched_64`|12.0| | | |`rocblas_hshgemv_batched_64`|6.2.0| | | | |
|`cublasHSHgemvStridedBatched`|11.6| | | |`rocblas_hshgemv_strided_batched`|6.0.0| | | | |
|`cublasHSHgemvStridedBatched_64`|12.0| | | |`rocblas_hshgemv_strided_batched_64`|6.2.0| | | | |
|`cublasHSSgemvBatched`|11.6| | | |`rocblas_hssgemv_batched`|6.0.0| | | | |
|`cublasHSSgemvBatched_64`|12.0| | | |`rocblas_hssgemv_batched_64`|6.2.0| | | | |
|`cublasHSSgemvStridedBatched`|11.6| | | |`rocblas_hssgemv_strided_batched`|6.0.0| | | | |
|`cublasHSSgemvStridedBatched_64`|12.0| | | |`rocblas_hssgemv_strided_batched_64`|6.2.0| | | | |
|`cublasHgemm`|7.5| | | |`rocblas_hgemm`|1.5.0| | | | |
|`cublasHgemmBatched`|9.0| | | |`rocblas_hgemm_batched`|3.5.0| | | | |
|`cublasHgemmBatched_64`|12.0| | | |`rocblas_hgemm_batched_64`|6.3.0| | | | |
|`cublasHgemmStridedBatched`|8.0| | | |`rocblas_hgemm_strided_batched`|1.5.0| | | | |
|`cublasHgemmStridedBatched_64`|12.0| | | |`rocblas_hgemm_strided_batched_64`|6.3.0| | | | |
|`cublasHgemm_64`|12.0| | | |`rocblas_hgemm_64`|6.3.0| | | | |
|`cublasSgemm`| | | | |`rocblas_sgemm`|1.5.0| | | | |
|`cublasSgemmBatched`| | | | |`rocblas_sgemm_batched`|3.5.0| | | | |
|`cublasSgemmBatched_64`|12.0| | | |`rocblas_sgemm_batched_64`|6.3.0| | | | |
|`cublasSgemmGroupedBatched`|12.4| | | | | | | | | |
|`cublasSgemmGroupedBatched_64`|12.4| | | | | | | | | |
|`cublasSgemmStridedBatched`|8.0| | | |`rocblas_sgemm_strided_batched`|1.5.0| | | | |
|`cublasSgemmStridedBatched_64`|12.0| | | |`rocblas_sgemm_strided_batched_64`|6.3.0| | | | |
|`cublasSgemm_64`|12.0| | | |`rocblas_sgemm_64`|6.3.0| | | | |
|`cublasSgemm_v2`| | | | |`rocblas_sgemm`|1.5.0| | | | |
|`cublasSgemm_v2_64`|12.0| | | |`rocblas_sgemm_64`|6.3.0| | | | |
|`cublasSgemvBatched`|11.6| | | |`rocblas_sgemv_batched`|3.5.0| | | | |
|`cublasSgemvBatched_64`|12.0| | | |`rocblas_sgemv_batched_64`|6.2.0| | | | |
|`cublasSgemvStridedBatched`|11.6| | | |`rocblas_sgemv_strided_batched`|3.5.0| | | | |
|`cublasSgemvStridedBatched_64`|12.0| | | |`rocblas_sgemv_strided_batched_64`|6.2.0| | | | |
|`cublasSsymm`| | | | |`rocblas_ssymm`|3.5.0| | | | |
|`cublasSsymm_64`|12.0| | | |`rocblas_ssymm_64`|6.3.0| | | | |
|`cublasSsymm_v2`| | | | |`rocblas_ssymm`|3.5.0| | | | |
|`cublasSsymm_v2_64`|12.0| | | |`rocblas_ssymm_64`|6.3.0| | | | |
|`cublasSsyr2k`| | | | |`rocblas_ssyr2k`|3.5.0| | | | |
|`cublasSsyr2k_64`|12.0| | | |`rocblas_ssyr2k_64`|6.3.0| | | | |
|`cublasSsyr2k_v2`| | | | |`rocblas_ssyr2k`|3.5.0| | | | |
|`cublasSsyr2k_v2_64`|12.0| | | |`rocblas_ssyr2k_64`|6.3.0| | | | |
|`cublasSsyrk`| | | | |`rocblas_ssyrk`|3.5.0| | | | |
|`cublasSsyrk_64`|12.0| | | |`rocblas_ssyrk_64`|6.3.0| | | | |
|`cublasSsyrk_v2`| | | | |`rocblas_ssyrk`|3.5.0| | | | |
|`cublasSsyrk_v2_64`|12.0| | | |`rocblas_ssyrk_64`|6.3.0| | | | |
|`cublasSsyrkx`| | | | |`rocblas_ssyrkx`|3.5.0| | | | |
|`cublasSsyrkx_64`|12.0| | | |`rocblas_ssyrkx_64`|6.3.0| | | | |
|`cublasStrmm`| | | | |`rocblas_strmm`|3.5.0| |6.0.0| | |
|`cublasStrmm_64`|12.0| | | |`rocblas_strmm_64`|6.3.0| | | | |
|`cublasStrmm_v2`| | | | |`rocblas_strmm`|3.5.0| |6.0.0| | |
|`cublasStrmm_v2_64`|12.0| | | |`rocblas_strmm_64`|6.3.0| | | | |
|`cublasStrsm`| | | | |`rocblas_strsm`|1.5.0| | | | |
|`cublasStrsm_64`|12.0| | | |`rocblas_strsm_64`|6.2.0| | | | |
|`cublasStrsm_v2`| | | | |`rocblas_strsm`|1.5.0| | | | |
|`cublasStrsm_v2_64`|12.0| | | |`rocblas_strsm_64`|6.2.0| | | | |
|`cublasTSSgemvBatched`|11.6| | | |`rocblas_tssgemv_batched`|6.0.0| | | | |
|`cublasTSSgemvBatched_64`|12.0| | | |`rocblas_tssgemv_batched_64`|6.2.0| | | | |
|`cublasTSSgemvStridedBatched`|11.6| | | |`rocblas_tssgemv_strided_batched`|6.0.0| | | | |
|`cublasTSSgemvStridedBatched_64`|12.0| | | |`rocblas_tssgemv_strided_batched_64`|6.2.0| | | | |
|`cublasTSTgemvBatched`|11.6| | | |`rocblas_tstgemv_batched`|6.0.0| | | | |
|`cublasTSTgemvBatched_64`|12.0| | | |`rocblas_tstgemv_batched_64`|6.2.0| | | | |
|`cublasTSTgemvStridedBatched`|11.6| | | |`rocblas_tstgemv_strided_batched`|6.0.0| | | | |
|`cublasTSTgemvStridedBatched_64`|12.0| | | |`rocblas_tstgemv_strided_batched_64`|6.2.0| | | | |
|`cublasZgemm`| | | | |`rocblas_zgemm`|1.5.0| | | | |
|`cublasZgemm3m`|8.0| | | | | | | | | |
|`cublasZgemm3m_64`|12.0| | | | | | | | | |
|`cublasZgemmBatched`| | | | |`rocblas_zgemm_batched`|3.5.0| | | | |
|`cublasZgemmBatched_64`|12.0| | | |`rocblas_zgemm_batched_64`|6.3.0| | | | |
|`cublasZgemmStridedBatched`|8.0| | | |`rocblas_zgemm_strided_batched`|1.5.0| | | | |
|`cublasZgemmStridedBatched_64`|12.0| | | |`rocblas_zgemm_strided_batched_64`|6.3.0| | | | |
|`cublasZgemm_64`|12.0| | | |`rocblas_zgemm_64`|6.3.0| | | | |
|`cublasZgemm_v2`| | | | |`rocblas_zgemm`|1.5.0| | | | |
|`cublasZgemm_v2_64`|12.0| | | |`rocblas_zgemm_64`|6.3.0| | | | |
|`cublasZgemvBatched`|11.6| | | |`rocblas_zgemv_batched`|3.5.0| | | | |
|`cublasZgemvBatched_64`|12.0| | | |`rocblas_zgemv_batched_64`|6.2.0| | | | |
|`cublasZgemvStridedBatched`|11.6| | | |`rocblas_zgemv_strided_batched`|3.5.0| | | | |
|`cublasZgemvStridedBatched_64`|12.0| | | |`rocblas_zgemv_strided_batched_64`|6.2.0| | | | |
|`cublasZhemm`| | | | |`rocblas_zhemm`|3.5.0| | | | |
|`cublasZhemm_64`|12.0| | | |`rocblas_zhemm_64`|6.3.0| | | | |
|`cublasZhemm_v2`| | | | |`rocblas_zhemm`|3.5.0| | | | |
|`cublasZhemm_v2_64`|12.0| | | |`rocblas_zhemm_64`|6.3.0| | | | |
|`cublasZher2k`| | | | |`rocblas_zher2k`|3.5.0| | | | |
|`cublasZher2k_64`|12.0| | | |`rocblas_zher2k_64`|6.3.0| | | | |
|`cublasZher2k_v2`| | | | |`rocblas_zher2k`|3.5.0| | | | |
|`cublasZher2k_v2_64`|12.0| | | |`rocblas_zher2k_64`|6.3.0| | | | |
|`cublasZherk`| | | | |`rocblas_zherk`|3.5.0| | | | |
|`cublasZherk_64`|12.0| | | |`rocblas_zherk_64`|6.3.0| | | | |
|`cublasZherk_v2`| | | | |`rocblas_zherk`|3.5.0| | | | |
|`cublasZherk_v2_64`|12.0| | | |`rocblas_zherk_64`|6.3.0| | | | |
|`cublasZherkx`| | | | |`rocblas_zherkx`|3.5.0| | | | |
|`cublasZherkx_64`|12.0| | | |`rocblas_zherkx_64`|6.3.0| | | | |
|`cublasZsymm`| | | | |`rocblas_zsymm`|3.5.0| | | | |
|`cublasZsymm_64`|12.0| | | |`rocblas_zsymm_64`|6.3.0| | | | |
|`cublasZsymm_v2`| | | | |`rocblas_zsymm`|3.5.0| | | | |
|`cublasZsymm_v2_64`|12.0| | | |`rocblas_zsymm_64`|6.3.0| | | | |
|`cublasZsyr2k`| | | | |`rocblas_zsyr2k`|3.5.0| | | | |
|`cublasZsyr2k_64`|12.0| | | |`rocblas_zsyr2k_64`|6.3.0| | | | |
|`cublasZsyr2k_v2`| | | | |`rocblas_zsyr2k`|3.5.0| | | | |
|`cublasZsyr2k_v2_64`|12.0| | | |`rocblas_zsyr2k_64`|6.3.0| | | | |
|`cublasZsyrk`| | | | |`rocblas_zsyrk`|3.5.0| | | | |
|`cublasZsyrk_64`|12.0| | | |`rocblas_zsyrk_64`|6.3.0| | | | |
|`cublasZsyrk_v2`| | | | |`rocblas_zsyrk`|3.5.0| | | | |
|`cublasZsyrk_v2_64`|12.0| | | |`rocblas_zsyrk_64`|6.3.0| | | | |
|`cublasZsyrkx`| | | | |`rocblas_zsyrkx`|3.5.0| | | | |
|`cublasZsyrkx_64`|12.0| | | |`rocblas_zsyrkx_64`|6.3.0| | | | |
|`cublasZtrmm`| | | | |`rocblas_ztrmm`|3.5.0| |6.0.0| | |
|`cublasZtrmm_64`|12.0| | | |`rocblas_ztrmm_64`|6.3.0| | | | |
|`cublasZtrmm_v2`| | | | |`rocblas_ztrmm`|3.5.0| |6.0.0| | |
|`cublasZtrmm_v2_64`|12.0| | | |`rocblas_ztrmm_64`|6.3.0| | | | |
|`cublasZtrsm`| | | | |`rocblas_ztrsm`|3.5.0| | | | |
|`cublasZtrsm_64`|12.0| | | |`rocblas_ztrsm_64`|6.2.0| | | | |
|`cublasZtrsm_v2`| | | | |`rocblas_ztrsm`|3.5.0| | | | |
|`cublasZtrsm_v2_64`|12.0| | | |`rocblas_ztrsm_64`|6.2.0| | | | |

## **8. BLAS-like Extension**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasAsumEx`|10.1| | | | | | | | | |
|`cublasAsumEx_64`|12.0| | | | | | | | | |
|`cublasAxpyEx`|8.0| | | |`rocblas_axpy_ex`|3.9.0| | | | |
|`cublasAxpyEx_64`|12.0| | | |`rocblas_axpy_ex_64`|6.1.0| | | | |
|`cublasCdgmm`| | | | |`rocblas_cdgmm`|3.5.0| | | | |
|`cublasCdgmm_64`|12.0| | | |`rocblas_cdgmm_64`|6.3.0| | | | |
|`cublasCgeam`| | | | |`rocblas_cgeam`|3.5.0| | | | |
|`cublasCgeam_64`|12.0| | | |`rocblas_cgeam_64`|6.3.0| | | | |
|`cublasCgelsBatched`| | | | | | | | | | |
|`cublasCgemmEx`|8.0| | | | | | | | | |
|`cublasCgemmEx_64`|12.0| | | | | | | | | |
|`cublasCgeqrfBatched`| | | | | | | | | | |
|`cublasCgetrfBatched`| | | | | | | | | | |
|`cublasCgetriBatched`| | | | | | | | | | |
|`cublasCgetrsBatched`| | | | | | | | | | |
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
|`cublasCtrsmBatched`| | | | |`rocblas_ctrsm_batched`|3.5.0| | | | |
|`cublasCtrsmBatched_64`|12.0| | | |`rocblas_ctrsm_batched_64`|6.2.0| | | | |
|`cublasCtrttp`| | | | | | | | | | |
|`cublasDdgmm`| | | | |`rocblas_ddgmm`|3.5.0| | | | |
|`cublasDdgmm_64`|12.0| | | |`rocblas_ddgmm_64`|6.3.0| | | | |
|`cublasDgeam`| | | | |`rocblas_dgeam`|1.6.4| | | | |
|`cublasDgeam_64`|12.0| | | |`rocblas_dgeam_64`|6.3.0| | | | |
|`cublasDgelsBatched`| | | | | | | | | | |
|`cublasDgeqrfBatched`| | | | | | | | | | |
|`cublasDgetrfBatched`| | | | | | | | | | |
|`cublasDgetriBatched`| | | | | | | | | | |
|`cublasDgetrsBatched`| | | | | | | | | | |
|`cublasDmatinvBatched`| | | | | | | | | | |
|`cublasDotEx`|8.0| | | |`rocblas_dot_ex`|4.1.0| | | | |
|`cublasDotEx_64`|12.0| | | |`rocblas_dot_ex_64`|6.1.0| | | | |
|`cublasDotcEx`|8.0| | | |`rocblas_dotc_ex`|4.1.0| | | | |
|`cublasDotcEx_64`|12.0| | | |`rocblas_dotc_ex_64`|6.1.0| | | | |
|`cublasDtpttr`| | | | | | | | | | |
|`cublasDtrsmBatched`| | | | |`rocblas_dtrsm_batched`|3.5.0| | | | |
|`cublasDtrsmBatched_64`|12.0| | | |`rocblas_dtrsm_batched_64`|6.2.0| | | | |
|`cublasDtrttp`| | | | | | | | | | |
|`cublasGemmBatchedEx`|9.1| |11.0| |`rocblas_gemm_batched_ex`|3.5.0| | | | |
|`cublasGemmBatchedEx_64`|12.0| | | | | | | | | |
|`cublasGemmEx`|8.0| |11.0| |`rocblas_gemm_ex`|1.8.2| | | | |
|`cublasGemmEx_64`|12.0| | | | | | | | | |
|`cublasGemmStridedBatchedEx`|9.1| |11.0| |`rocblas_gemm_strided_batched_ex`|1.9.0| | | | |
|`cublasGemmStridedBatchedEx_64`|12.0| | | | | | | | | |
|`cublasIamaxEx`|10.1| | | | | | | | | |
|`cublasIamaxEx_64`|12.0| | | | | | | | | |
|`cublasIaminEx`|10.1| | | | | | | | | |
|`cublasIaminEx_64`|12.0| | | | | | | | | |
|`cublasRotEx`|10.1| | | |`rocblas_rot_ex`|4.1.0| | | | |
|`cublasRotEx_64`|12.0| | | |`rocblas_rot_ex_64`|6.1.0| | | | |
|`cublasRotgEx`|10.1| | | | | | | | | |
|`cublasRotmEx`|10.1| | | | | | | | | |
|`cublasRotmEx_64`|12.0| | | | | | | | | |
|`cublasRotmgEx`|10.1| | | | | | | | | |
|`cublasScalEx`|8.0| | | |`rocblas_scal_ex`|4.0.0| | | | |
|`cublasScalEx_64`|12.0| | | |`rocblas_scal_ex_64`|6.1.0| | | | |
|`cublasSdgmm`| | | | |`rocblas_sdgmm`|3.5.0| | | | |
|`cublasSdgmm_64`|12.0| | | |`rocblas_sdgmm_64`|6.3.0| | | | |
|`cublasSgeam`| | | | |`rocblas_sgeam`|1.6.4| | | | |
|`cublasSgeam_64`|12.0| | | |`rocblas_sgeam_64`|6.3.0| | | | |
|`cublasSgelsBatched`| | | | | | | | | | |
|`cublasSgemmEx`|7.5| | | | | | | | | |
|`cublasSgemmEx_64`|12.0| | | | | | | | | |
|`cublasSgeqrfBatched`| | | | | | | | | | |
|`cublasSgetrfBatched`| | | | | | | | | | |
|`cublasSgetriBatched`| | | | | | | | | | |
|`cublasSgetrsBatched`| | | | | | | | | | |
|`cublasSmatinvBatched`| | | | | | | | | | |
|`cublasStpttr`| | | | | | | | | | |
|`cublasStrsmBatched`| | | | |`rocblas_strsm_batched`|3.5.0| | | | |
|`cublasStrsmBatched_64`|12.0| | | |`rocblas_strsm_batched_64`|6.2.0| | | | |
|`cublasStrttp`| | | | | | | | | | |
|`cublasSwapEx`|10.1| | | | | | | | | |
|`cublasSwapEx_64`|12.0| | | | | | | | | |
|`cublasUint8gemmBias`|8.0| | | | | | | | | |
|`cublasZdgmm`| | | | |`rocblas_zdgmm`|3.5.0| | | | |
|`cublasZdgmm_64`|12.0| | | |`rocblas_zdgmm_64`|6.3.0| | | | |
|`cublasZgeam`| | | | |`rocblas_zgeam`|3.5.0| | | | |
|`cublasZgeam_64`|12.0| | | |`rocblas_zgeam_64`|6.3.0| | | | |
|`cublasZgelsBatched`| | | | | | | | | | |
|`cublasZgeqrfBatched`| | | | | | | | | | |
|`cublasZgetrfBatched`| | | | | | | | | | |
|`cublasZgetriBatched`| | | | | | | | | | |
|`cublasZgetrsBatched`| | | | | | | | | | |
|`cublasZmatinvBatched`| | | | | | | | | | |
|`cublasZtpttr`| | | | | | | | | | |
|`cublasZtrsmBatched`| | | | |`rocblas_ztrsm_batched`|3.5.0| | | | |
|`cublasZtrsmBatched_64`|12.0| | | |`rocblas_ztrsm_batched_64`|6.2.0| | | | |
|`cublasZtrttp`| | | | | | | | | | |

## **9. BLASLt Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cublasLtCreate`|10.1| | | | | | | | | |
|`cublasLtDestroy`|10.1| | | | | | | | | |
|`cublasLtDisableCpuInstructionsSetMask`|12.1| | | | | | | | | |
|`cublasLtGetCudartVersion`|10.1| | | | | | | | | |
|`cublasLtGetProperty`|10.1| | | | | | | | | |
|`cublasLtGetStatusName`|11.4| | | | | | | | | |
|`cublasLtGetStatusString`|11.4| | | | | | | | | |
|`cublasLtGetVersion`|10.1| | | | | | | | | |
|`cublasLtHeuristicsCacheGetCapacity`|11.8| | | | | | | | | |
|`cublasLtHeuristicsCacheSetCapacity`|11.8| | | | | | | | | |
|`cublasLtLoggerForceDisable`|11.0| | | | | | | | | |
|`cublasLtLoggerOpenFile`|11.0| | | | | | | | | |
|`cublasLtLoggerSetCallback`|11.0| | | | | | | | | |
|`cublasLtLoggerSetFile`|11.0| | | | | | | | | |
|`cublasLtLoggerSetLevel`|11.0| | | | | | | | | |
|`cublasLtLoggerSetMask`|11.0| | | | | | | | | |
|`cublasLtMatmul`|10.1| | | | | | | | | |
|`cublasLtMatmulAlgoCapGetAttribute`|10.1| | | | | | | | | |
|`cublasLtMatmulAlgoCheck`|10.1| | | | | | | | | |
|`cublasLtMatmulAlgoConfigGetAttribute`|10.1| | | | | | | | | |
|`cublasLtMatmulAlgoConfigSetAttribute`|10.1| | | | | | | | | |
|`cublasLtMatmulAlgoGetHeuristic`|10.1| | | | | | | | | |
|`cublasLtMatmulAlgoGetIds`|10.1| | | | | | | | | |
|`cublasLtMatmulAlgoInit`|10.1| | | | | | | | | |
|`cublasLtMatmulDescCreate`|10.1| |11.0| | | | | | | |
|`cublasLtMatmulDescDestroy`|10.1| | | | | | | | | |
|`cublasLtMatmulDescGetAttribute`|10.1| | | | | | | | | |
|`cublasLtMatmulDescInit`|11.0| | | | | | | | | |
|`cublasLtMatmulDescSetAttribute`|10.1| | | | | | | | | |
|`cublasLtMatmulPreferenceCreate`|10.1| | | | | | | | | |
|`cublasLtMatmulPreferenceDestroy`|10.1| | | | | | | | | |
|`cublasLtMatmulPreferenceGetAttribute`|10.1| | | | | | | | | |
|`cublasLtMatmulPreferenceInit`|11.0| | | | | | | | | |
|`cublasLtMatmulPreferenceSetAttribute`|10.1| | | | | | | | | |
|`cublasLtMatrixLayoutCreate`|10.1| | | | | | | | | |
|`cublasLtMatrixLayoutDestroy`|10.1| | | | | | | | | |
|`cublasLtMatrixLayoutGetAttribute`|10.1| | | | | | | | | |
|`cublasLtMatrixLayoutInit`|11.0| | | | | | | | | |
|`cublasLtMatrixLayoutSetAttribute`|10.1| | | | | | | | | |
|`cublasLtMatrixTransform`|10.1| | | | | | | | | |
|`cublasLtMatrixTransformDescCreate`|10.1| | | | | | | | | |
|`cublasLtMatrixTransformDescDestroy`|10.1| | | | | | | | | |
|`cublasLtMatrixTransformDescGetAttribute`|10.1| | | | | | | | | |
|`cublasLtMatrixTransformDescInit`|11.0| | | | | | | | | |
|`cublasLtMatrixTransformDescSetAttribute`|10.1| | | | | | | | | |

