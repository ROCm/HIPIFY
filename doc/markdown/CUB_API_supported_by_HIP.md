# CUB API supported by HIP

## **1. CUB Data types**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`CUB_CPP_DIALECT`| | | | | | | | |
|`CUB_INCLUDE_DEVICE_CODE`| | | | | | | | |
|`CUB_INCLUDE_HOST_CODE`| | | | | | | | |
|`CUB_IS_DEVICE_CODE`| | | | | | | | |
|`CUB_IS_HOST_CODE`| | | | | | | | |
|`CUB_LOG_SMEM_BANKS`| | | | | | | | |
|`CUB_LOG_WARP_THREADS`| | | | | | | | |
|`CUB_MAX`| | | |`CUB_MAX`| | | | |
|`CUB_MAX_DEVICES`| | | | | | | | |
|`CUB_MIN`| | | |`CUB_MIN`|2.5.0| | | |
|`CUB_NAMESPACE_BEGIN`| | | |`BEGIN_HIPCUB_NAMESPACE`|2.5.0| | | |
|`CUB_NAMESPACE_END`| | | |`END_HIPCUB_NAMESPACE`|2.5.0| | | |
|`CUB_PREFER_CONFLICT_OVER_PADDING`| | | | | | | | |
|`CUB_PTX_ARCH`| | | |`HIPCUB_ARCH`|2.5.0| | | |
|`CUB_PTX_LOG_SMEM_BANKS`| | | | | | | | |
|`CUB_PTX_LOG_WARP_THREADS`| | | | | | | | |
|`CUB_PTX_PREFER_CONFLICT_OVER_PADDING`| | | | | | | | |
|`CUB_PTX_SMEM_BANKS`| | | | | | | | |
|`CUB_PTX_SUBSCRIPTION_FACTOR`| | | | | | | | |
|`CUB_PTX_WARP_THREADS`| | | |`HIPCUB_WARP_THREADS`|2.5.0| | | |
|`CUB_RUNTIME_ENABLED`| | | | | | | | |
|`CUB_RUNTIME_FUNCTION`| | | |`HIPCUB_RUNTIME_FUNCTION`|2.5.0| | | |
|`CUB_SMEM_BANKS`| | | | | | | | |
|`CUB_STDERR`| | | |`HIPCUB_STDERR`|2.5.0| | | |
|`CUB_SUBSCRIPTION_FACTOR`| | | | | | | | |
|`CUB_USE_COOPERATIVE_GROUPS`| | | | | | | | |
|`CubDebug`| | | |`HipcubDebug`|2.5.0| | | |
|`CubDebugExit`| | | | | | | | |
|`_CubLog`| | | |`_HipcubLog`|2.5.0| | | |


\*A - Added; D - Deprecated; R - Removed; E - Experimental