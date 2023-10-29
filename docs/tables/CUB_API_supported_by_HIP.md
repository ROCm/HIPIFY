# CUB API supported by HIP

## **1. CUB Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUB_ALIGN`| | | | | | | | | | |
|`CUB_CAT`| | | | | | | | | | |
|`CUB_CAT_`| | | | | | | | | | |
|`CUB_COMPILER_DEPRECATION`| | | | | | | | | | |
|`CUB_COMPILER_DEPRECATION_SOFT`| | | | | | | | | | |
|`CUB_COMP_DEPR_IMPL`| | | | | | | | | | |
|`CUB_COMP_DEPR_IMPL0`| | | | | | | | | | |
|`CUB_COMP_DEPR_IMPL1`| | | | | | | | | | |
|`CUB_CPLUSPLUS`| | | | | | | | | | |
|`CUB_CPP_DIALECT`| | | | | | | | | | |
|`CUB_DEFINE_DETECT_NESTED_TYPE`| | | | | | | | | | |
|`CUB_DEFINE_VECTOR_TYPE`| | | | | | | | | | |
|`CUB_DEPRECATED`| | | | | | | | | | |
|`CUB_DEVICE_COMPILER`| | | | | | | | | | |
|`CUB_DEVICE_COMPILER_CLANG`| | | | | | | | | | |
|`CUB_DEVICE_COMPILER_GCC`| | | | | | | | | | |
|`CUB_DEVICE_COMPILER_MSVC`| | | | | | | | | | |
|`CUB_DEVICE_COMPILER_NVCC`| | | | | | | | | | |
|`CUB_DEVICE_COMPILER_UNKNOWN`| | | | | | | | | | |
|`CUB_HOST_COMPILER`| | | | | | | | | | |
|`CUB_HOST_COMPILER_CLANG`| | | | | | | | | | |
|`CUB_HOST_COMPILER_GCC`| | | | | | | | | | |
|`CUB_HOST_COMPILER_MSVC`| | | | | | | | | | |
|`CUB_HOST_COMPILER_UNKNOWN`| | | | | | | | | | |
|`CUB_IGNORE_DEPRECATED_API`| | | | | | | | | | |
|`CUB_IGNORE_DEPRECATED_COMPILER`| | | | | | | | | | |
|`CUB_IGNORE_DEPRECATED_CPP_11`| | | | | | | | | | |
|`CUB_IGNORE_DEPRECATED_CPP_DIALECT`| | | | | | | | | | |
|`CUB_IGNORE_DEPRECATED_DIALECT`| | | | | | | | | | |
|`CUB_INCLUDE_DEVICE_CODE`| | | | | | | | | | |
|`CUB_INCLUDE_HOST_CODE`| | | | | | | | | | |
|`CUB_IS_DEVICE_CODE`| | | | | | | | | | |
|`CUB_IS_HOST_CODE`| | | | | | | | | | |
|`CUB_LOG_SMEM_BANKS`| | | | | | | | | | |
|`CUB_LOG_WARP_THREADS`| | | | | | | | | | |
|`CUB_MAX`| | | | |`CUB_MAX`|4.5.0| | | | |
|`CUB_MAX_DEVICES`| | | | | | | | | | |
|`CUB_MIN`| | | | |`CUB_MIN`|4.5.0| | | | |
|`CUB_MSVC_VERSION`| | | | | | | | | | |
|`CUB_MSVC_VERSION_FULL`| | | | | | | | | | |
|`CUB_NAMESPACE_BEGIN`| | | | |`BEGIN_HIPCUB_NAMESPACE`|2.5.0| | | | |
|`CUB_NAMESPACE_END`| | | | |`END_HIPCUB_NAMESPACE`|2.5.0| | | | |
|`CUB_PREFER_CONFLICT_OVER_PADDING`| | | | | | | | | | |
|`CUB_PREVENT_MACRO_SUBSTITUTION`| | | | | | | | | | |
|`CUB_PTX_ARCH`| | | | |`HIPCUB_ARCH`|2.5.0| | | | |
|`CUB_PTX_LOG_SMEM_BANKS`| | | | | | | | | | |
|`CUB_PTX_LOG_WARP_THREADS`| | | | | | | | | | |
|`CUB_PTX_PREFER_CONFLICT_OVER_PADDING`| | | | | | | | | | |
|`CUB_PTX_SMEM_BANKS`| | | | | | | | | | |
|`CUB_PTX_SUBSCRIPTION_FACTOR`| | | | | | | | | | |
|`CUB_PTX_WARP_THREADS`| | | | |`HIPCUB_WARP_THREADS`|2.5.0| | | | |
|`CUB_QUOTIENT_CEILING`| | | | | | | | | | |
|`CUB_QUOTIENT_FLOOR`| | | | | | | | | | |
|`CUB_ROUND_DOWN_NEAREST`| | | | | | | | | | |
|`CUB_ROUND_UP_NEAREST`| | | | | | | | | | |
|`CUB_RUNTIME_ENABLED`| | | | | | | | | | |
|`CUB_RUNTIME_FUNCTION`| | | | |`HIPCUB_RUNTIME_FUNCTION`|2.5.0| | | | |
|`CUB_SMEM_BANKS`| | | | | | | | | | |
|`CUB_STATIC_ASSERT`| | | | | | | | | | |
|`CUB_STDERR`| | | | |`HIPCUB_STDERR`|2.5.0| | | | |
|`CUB_SUBSCRIPTION_FACTOR`| | | | | | | | | | |
|`CUB_USE_COOPERATIVE_GROUPS`| | | | | | | | | | |
|`CubDebug`| | | | |`HipcubDebug`|2.5.0| | | | |
|`CubDebugExit`| | | | | | | | | | |
|`CubVector`| | | | | | | | | | |
|`_CUB_ASM_PTR_`| | | | | | | | | | |
|`_CUB_ASM_PTR_SIZE_`| | | | | | | | | | |
|`_CubLog`| | | | |`_HipcubLog`|2.5.0| | | | |
|`__CUB_ALIGN_BYTES`| | | | |`__HIPCUB_ALIGN_BYTES`|4.5.0| | | | |
|`__CUB_LP64__`| | | | | | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental