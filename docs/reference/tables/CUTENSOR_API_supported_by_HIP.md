# CUTENSOR API supported by HIP


**Note\:** In the tables that follow the columns marked `A`, `D`, `C`, `R`, and `E` mean the following:
**A** - Added; **D** - Deprecated; **C** - Changed; **R** - Removed; **E** - Experimental

## **1. CUTENSOR Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUTENSORMG_ALGO_DEFAULT`|1.4.0.0| | | | | | | | | |
|`CUTENSORMG_CONTRACTION_FIND_ATTRIBUTE_MAX`|1.5.0.0| | | | | | | | | |
|`CUTENSOR_ALGO_DEFAULT`|1.0.1.0| | | |`HIPTENSOR_ALGO_DEFAULT`|5.7.0| | | | |
|`CUTENSOR_ALGO_DEFAULT_PATIENT`|1.4.0.0| | | |`HIPTENSOR_ALGO_DEFAULT_PATIENT`|5.7.0| | | | |
|`CUTENSOR_ALGO_GETT`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_ALGO_TGETT`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_ALGO_TTGT`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_AUTOTUNE_MODE_INCREMENTAL`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_AUTOTUNE_MODE_NONE`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_CACHE_MODE_NONE`|1.2.0.0| | | | | | | | | |
|`CUTENSOR_CACHE_MODE_PEDANTIC`|1.2.0.0| | | | | | | | | |
|`CUTENSOR_COMPUTE_16BF`|1.0.1.0| | |2.0.0.0|`HIPTENSOR_COMPUTE_16BF`|5.7.0| | | | |
|`CUTENSOR_COMPUTE_16F`|1.0.1.0| | |2.0.0.0|`HIPTENSOR_COMPUTE_16F`|5.7.0| | | | |
|`CUTENSOR_COMPUTE_32F`|1.0.1.0| | |2.0.0.0|`HIPTENSOR_COMPUTE_32F`|5.7.0| | | | |
|`CUTENSOR_COMPUTE_32I`|1.0.1.0| | |2.0.0.0|`HIPTENSOR_COMPUTE_32I`|5.7.0| | | | |
|`CUTENSOR_COMPUTE_32U`|1.0.1.0| | |2.0.0.0|`HIPTENSOR_COMPUTE_32U`|5.7.0| | | | |
|`CUTENSOR_COMPUTE_3XTF32`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_COMPUTE_64F`|1.0.1.0| | |2.0.0.0|`HIPTENSOR_COMPUTE_64F`|5.7.0| | | | |
|`CUTENSOR_COMPUTE_8I`|1.0.1.0| | |2.0.0.0|`HIPTENSOR_COMPUTE_8I`|5.7.0| | | | |
|`CUTENSOR_COMPUTE_8U`|1.0.1.0| | |2.0.0.0|`HIPTENSOR_COMPUTE_8U`|5.7.0| | | | |
|`CUTENSOR_COMPUTE_TF32`|1.0.1.0| | |2.0.0.0| | | | | | |
|`CUTENSOR_C_16BF`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_16F`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_16I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_16U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_32F`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_C32F`|6.1.0| | | | |
|`CUTENSOR_C_32I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_32U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_4I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_4U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_64F`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_C64F`|5.7.0| | | | |
|`CUTENSOR_C_64I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_64U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_8I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_8U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_MIN_16F`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_C_MIN_32F`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_C_MIN_64F`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_C_MIN_TF32`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_JIT_MODE_DEFAULT`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_JIT_MODE_NONE`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_MG_DEVICE_HOST`|1.4.0.0| | | | | | | | | |
|`CUTENSOR_MG_DEVICE_HOST_PINNED`|1.4.0.0| | | | | | | | | |
|`CUTENSOR_OPERATION_DESCRIPTOR_FLOPS`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OPERATION_DESCRIPTOR_TAG`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OP_ABS`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ACOS`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ACOSH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ADD`|1.0.1.0| | | |`HIPTENSOR_OP_ADD`|6.3.0| | | | |
|`CUTENSOR_OP_ASIN`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ASINH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ATAN`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ATANH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_CEIL`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_CONJ`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_COS`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_COSH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_EXP`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_FLOOR`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_IDENTITY`|1.0.1.0| | | |`HIPTENSOR_OP_IDENTITY`|5.7.0| | | | |
|`CUTENSOR_OP_LOG`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_MAX`|1.0.1.0| | | |`HIPTENSOR_OP_MAX`|6.3.0| | | | |
|`CUTENSOR_OP_MIN`|1.0.1.0| | | |`HIPTENSOR_OP_MIN`|6.3.0| | | | |
|`CUTENSOR_OP_MISH`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OP_MUL`|1.0.1.0| | | |`HIPTENSOR_OP_MUL`|6.3.0| | | | |
|`CUTENSOR_OP_NEG`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_RCP`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_RELU`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_SIGMOID`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_SIN`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_SINH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_SOFT_PLUS`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OP_SOFT_SIGN`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OP_SQRT`|1.0.1.0| | | |`HIPTENSOR_OP_SQRT`|6.2.0| | | | |
|`CUTENSOR_OP_SWISH`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OP_TAN`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_TANH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_UNKNOWN`|1.0.1.0| | | |`HIPTENSOR_OP_UNKNOWN`|5.7.0| | | | |
|`CUTENSOR_PLAN_PREFERENCE_ALGO`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_PLAN_PREFERENCE_CACHE_MODE`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_PLAN_PREFERENCE_JIT`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_PLAN_REQUIRED_WORKSPACE`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_16BF`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_16BF`|5.7.0| | | | |
|`CUTENSOR_R_16F`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_16F`|5.7.0| | | | |
|`CUTENSOR_R_16I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_16U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_32F`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_32F`|5.7.0| | | | |
|`CUTENSOR_R_32I`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_32I`|5.7.0| | | | |
|`CUTENSOR_R_32U`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_32U`|5.7.0| | | | |
|`CUTENSOR_R_4I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_4U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_64F`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_64F`|5.7.0| | | | |
|`CUTENSOR_R_64I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_64U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_8I`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_8I`|5.7.0| | | | |
|`CUTENSOR_R_8U`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_8U`|5.7.0| | | | |
|`CUTENSOR_R_MIN_16BF`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_R_MIN_16F`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_R_MIN_32F`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_R_MIN_32I`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_R_MIN_32U`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_R_MIN_64F`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_R_MIN_8I`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_R_MIN_8U`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_R_MIN_TF32`|1.0.1.0|1.2.0.0| |2.0.0.0| | | | | | |
|`CUTENSOR_STATUS_ALLOC_FAILED`|1.0.1.0| | | |`HIPTENSOR_STATUS_ALLOC_FAILED`|5.7.0| | | | |
|`CUTENSOR_STATUS_ARCH_MISMATCH`|1.0.1.0| | | |`HIPTENSOR_STATUS_ARCH_MISMATCH`|5.7.0| | | | |
|`CUTENSOR_STATUS_CUBLAS_ERROR`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_STATUS_CUDA_ERROR`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_STATUS_EXECUTION_FAILED`|1.0.1.0| | | |`HIPTENSOR_STATUS_EXECUTION_FAILED`|5.7.0| | | | |
|`CUTENSOR_STATUS_INSUFFICIENT_DRIVER`|1.0.1.0| | | |`HIPTENSOR_STATUS_INSUFFICIENT_DRIVER`|5.7.0| | | | |
|`CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE`|1.0.1.0| | | |`HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE`|5.7.0| | | | |
|`CUTENSOR_STATUS_INTERNAL_ERROR`|1.0.1.0| | | |`HIPTENSOR_STATUS_INTERNAL_ERROR`|5.7.0| | | | |
|`CUTENSOR_STATUS_INVALID_VALUE`|1.0.1.0| | | |`HIPTENSOR_STATUS_INVALID_VALUE`|5.7.0| | | | |
|`CUTENSOR_STATUS_IO_ERROR`|1.2.0.0| | | |`HIPTENSOR_STATUS_IO_ERROR`|5.7.0| | | | |
|`CUTENSOR_STATUS_LICENSE_ERROR`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_STATUS_MAPPING_ERROR`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_STATUS_NOT_INITIALIZED`|1.0.1.0| | | |`HIPTENSOR_STATUS_NOT_INITIALIZED`|5.7.0| | | | |
|`CUTENSOR_STATUS_NOT_SUPPORTED`|1.0.1.0| | | |`HIPTENSOR_STATUS_NOT_SUPPORTED`|5.7.0| | | | |
|`CUTENSOR_STATUS_SUCCESS`|1.0.1.0| | | |`HIPTENSOR_STATUS_SUCCESS`|5.7.0| | | | |
|`CUTENSOR_WORKSPACE_DEFAULT`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_WORKSPACE_MAX`|1.0.1.0| | | |`HIPTENSOR_WORKSPACE_MAX`|5.7.0| | | | |
|`CUTENSOR_WORKSPACE_MIN`|1.0.1.0| | | |`HIPTENSOR_WORKSPACE_MIN`|5.7.0| | | | |
|`CUTENSOR_WORKSPACE_RECOMMENDED`|1.0.1.0| | |2.0.0.0|`HIPTENSOR_WORKSPACE_RECOMMENDED`|5.7.0| | | | |
|`cutensorAlgo_t`|1.0.1.0| | | |`hiptensorAlgo_t`|5.7.0| | | | |
|`cutensorAutotuneMode_t`|1.2.0.0| | | | | | | | | |
|`cutensorCacheMode_t`|1.2.0.0| | | | | | | | | |
|`cutensorComputeType_t`| | | | |`hiptensorComputeType_t`|5.7.0| | | | |
|`cutensorContractionPlan_t`|1.0.1.0| | |2.0.0.0|`hiptensorContractionPlan_t`|5.7.0| | | | |
|`cutensorDataType_t`|2.0.0.0| | | |`hiptensorComputeType_t`|5.7.0| | | | |
|`cutensorHandle`|2.0.0.0| | | | | | | | | |
|`cutensorHandle_t`|1.0.1.0| | | |`hiptensorHandle_t`|5.7.0| | | | |
|`cutensorJitMode_t`|2.0.0.0| | | | | | | | | |
|`cutensorLoggerCallback_t`|1.3.2.0| | | |`hiptensorLoggerCallback_t`|5.7.0| | | | |
|`cutensorMgAlgo_t`|1.4.0.0| | | | | | | | | |
|`cutensorMgContractionDescriptor_s`|1.4.0.0| | | | | | | | | |
|`cutensorMgContractionDescriptor_t`|1.4.0.0| | | | | | | | | |
|`cutensorMgContractionFindAttribute_t`|1.5.0.0| | | | | | | | | |
|`cutensorMgContractionFind_s`|1.4.0.0| | | | | | | | | |
|`cutensorMgContractionFind_t`|1.4.0.0| | | | | | | | | |
|`cutensorMgContractionPlan_s`|1.4.0.0| | | | | | | | | |
|`cutensorMgContractionPlan_t`|1.4.0.0| | | | | | | | | |
|`cutensorMgCopyDescriptor_s`|1.4.0.0| | | | | | | | | |
|`cutensorMgCopyDescriptor_t`|1.4.0.0| | | | | | | | | |
|`cutensorMgCopyPlan_s`|1.4.0.0| | | | | | | | | |
|`cutensorMgCopyPlan_t`|1.4.0.0| | | | | | | | | |
|`cutensorMgHandle_s`|1.4.0.0| | | | | | | | | |
|`cutensorMgHandle_t`|1.4.0.0| | | | | | | | | |
|`cutensorMgHostDevice_t`|1.4.0.0| | | | | | | | | |
|`cutensorMgTensorDescriptor_s`|1.4.0.0| | | | | | | | | |
|`cutensorMgTensorDescriptor_t`|1.4.0.0| | | | | | | | | |
|`cutensorOperationDescriptorAttribute_t`|2.0.0.0| | | | | | | | | |
|`cutensorOperator_t`|1.0.1.0| | | |`hiptensorOperator_t`|5.7.0| | | | |
|`cutensorPlan`|2.0.0.0| | | | | | | | | |
|`cutensorPlanAttribute_t`|2.0.0.0| | | | | | | | | |
|`cutensorPlanPreferenceAttribute_t`|2.0.0.0| | | | | | | | | |
|`cutensorPlan_t`|2.0.0.0| | | |`hiptensorContractionPlan_t`|5.7.0| | | | |
|`cutensorStatus_t`|1.0.1.0| | | |`hiptensorStatus_t`|5.7.0| | | | |
|`cutensorTensorDescriptor`|2.0.0.0| | | | | | | | | |
|`cutensorTensorDescriptor_t`|1.0.1.0| | | |`hiptensorTensorDescriptor_t`|5.7.0| | | | |
|`cutensorWorksizePreference_t`|1.0.1.0| | | |`hiptensorWorksizePreference_t`|5.7.0| | | | |

## **2. CUTENSOR Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cutensorContract`|2.0.0.0| | | |`hiptensorContraction`|6.1.0| | | | |
|`cutensorContraction`|1.0.1.0| | |2.0.0.0|`hiptensorContraction`|6.1.0| | | | |
|`cutensorCreate`|1.7.0.0| | | |`hiptensorCreate`|5.7.0| | | | |
|`cutensorCreateContraction`|2.0.0.0| | | | | | | | | |
|`cutensorCreateElementwiseBinary`|2.0.0.0| | | | | | | | | |
|`cutensorCreateElementwiseTrinary`|2.0.0.0| | | | | | | | | |
|`cutensorCreatePermutation`|2.0.0.0| | | | | | | | | |
|`cutensorCreatePlan`|2.0.0.0| | | | | | | | | |
|`cutensorCreatePlanPreference`|2.0.0.0| | | | | | | | | |
|`cutensorCreateReduction`|2.0.0.0| | | | | | | | | |
|`cutensorCreateTensorDescriptor`|2.0.0.0| | | | | | | | | |
|`cutensorDestroy`|1.7.0.0| | | |`hiptensorDestroy`|5.7.0| | | | |
|`cutensorDestroyOperationDescriptor`|2.0.0.0| | | | | | | | | |
|`cutensorDestroyPlan`|2.0.0.0| | | | | | | | | |
|`cutensorDestroyPlanPreference`|2.0.0.0| | | | | | | | | |
|`cutensorDestroyTensorDescriptor`|2.0.0.0| | | | | | | | | |
|`cutensorElementwiseBinaryExecute`|2.0.0.0| | | | | | | | | |
|`cutensorElementwiseTrinaryExecute`|2.0.0.0| | | | | | | | | |
|`cutensorEstimateWorkspaceSize`|2.0.0.0| | | | | | | | | |
|`cutensorGetCudartVersion`|1.0.1.0| | | |`hiptensorGetHiprtVersion`|5.7.0| | | | |
|`cutensorGetErrorString`|1.0.1.0| | | |`hiptensorGetErrorString`|5.7.0| | | | |
|`cutensorGetVersion`|1.0.1.0| | | | | | | | | |
|`cutensorHandleReadPlanCacheFromFile`|2.0.0.0| | | | | | | | | |
|`cutensorHandleResizePlanCache`|2.0.0.0| | | | | | | | | |
|`cutensorHandleWritePlanCacheToFile`|2.0.0.0| | | | | | | | | |
|`cutensorInitTensorDescriptor`|1.0.1.0| | |2.0.0.0|`hiptensorInitTensorDescriptor`|5.7.0| | | | |
|`cutensorLoggerForceDisable`|1.3.2.0| | | |`hiptensorLoggerForceDisable`|5.7.0| | | | |
|`cutensorLoggerOpenFile`|1.3.2.0| | | |`hiptensorLoggerOpenFile`|5.7.0| | | | |
|`cutensorLoggerSetCallback`|1.3.2.0| | | |`hiptensorLoggerSetCallback`|5.7.0| | | | |
|`cutensorLoggerSetFile`|1.3.2.0| | | |`hiptensorLoggerSetFile`|5.7.0| | | | |
|`cutensorLoggerSetLevel`|1.3.2.0| | | |`hiptensorLoggerSetLevel`|5.7.0| | | | |
|`cutensorLoggerSetMask`|1.3.2.0| | | |`hiptensorLoggerSetMask`|5.7.0| | | | |
|`cutensorMgContraction`|1.4.0.0| | | | | | | | | |
|`cutensorMgContractionFindSetAttribute`|1.5.0.0| | | | | | | | | |
|`cutensorMgContractionGetWorkspace`|1.4.0.0| | | | | | | | | |
|`cutensorMgCopy`|1.4.0.0| | | | | | | | | |
|`cutensorMgCopyGetWorkspace`|1.4.0.0| | | | | | | | | |
|`cutensorMgCreate`|1.4.0.0| | | | | | | | | |
|`cutensorMgCreateContractionDescriptor`|1.4.0.0| | | | | | | | | |
|`cutensorMgCreateContractionFind`|1.4.0.0| | | | | | | | | |
|`cutensorMgCreateContractionPlan`|1.4.0.0| | | | | | | | | |
|`cutensorMgCreateCopyDescriptor`|1.4.0.0| | | | | | | | | |
|`cutensorMgCreateCopyPlan`|1.4.0.0| | | | | | | | | |
|`cutensorMgCreateTensorDescriptor`|1.4.0.0| | | | | | | | | |
|`cutensorMgDestroy`|1.4.0.0| | | | | | | | | |
|`cutensorMgDestroyContractionDescriptor`|1.4.0.0| | | | | | | | | |
|`cutensorMgDestroyContractionFind`|1.4.0.0| | | | | | | | | |
|`cutensorMgDestroyContractionPlan`|1.4.0.0| | | | | | | | | |
|`cutensorMgDestroyCopyDescriptor`|1.4.0.0| | | | | | | | | |
|`cutensorMgDestroyCopyPlan`|1.4.0.0| | | | | | | | | |
|`cutensorMgDestroyTensorDescriptor`|1.4.0.0| | | | | | | | | |
|`cutensorOperationDescriptorGetAttribute`|2.0.0.0| | | | | | | | | |
|`cutensorOperationDescriptorSetAttribute`|2.0.0.0| | | | | | | | | |
|`cutensorPermutation`|1.0.1.0| | |2.0.0.0|`hiptensorPermutation`|6.1.0| | | | |
|`cutensorPermute`|2.0.0.0| | | | | | | | | |
|`cutensorPlanGetAttribute`|2.0.0.0| | | | | | | | | |
|`cutensorPlanPreferenceSetAttribute`|2.0.0.0| | | | | | | | | |
|`cutensorReadKernelCacheFromFile`|2.0.0.0| | | | | | | | | |
|`cutensorReduce`|2.0.0.0| | | | | | | | | |
|`cutensorReduction`|1.0.1.0| | |2.0.0.0|`hiptensorReduction`|6.3.0| | | | |
|`cutensorWriteKernelCacheToFile`|2.0.0.0| | | | | | | | | |

