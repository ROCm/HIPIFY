# CUDA RTC API supported by HIP

## **1. RTC Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`NVRTC_ERROR_BUILTIN_OPERATION_FAILURE`| | | | |`HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE`|2.6.0| | | | |
|`NVRTC_ERROR_COMPILATION`| | | | |`HIPRTC_ERROR_COMPILATION`|2.6.0| | | | |
|`NVRTC_ERROR_INTERNAL_ERROR`|8.0| | | |`HIPRTC_ERROR_INTERNAL_ERROR`|2.6.0| | | | |
|`NVRTC_ERROR_INVALID_INPUT`| | | | |`HIPRTC_ERROR_INVALID_INPUT`|2.6.0| | | | |
|`NVRTC_ERROR_INVALID_OPTION`| | | | |`HIPRTC_ERROR_INVALID_OPTION`|2.6.0| | | | |
|`NVRTC_ERROR_INVALID_PROGRAM`| | | | |`HIPRTC_ERROR_INVALID_PROGRAM`|2.6.0| | | | |
|`NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID`|8.0| | | |`HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID`|2.6.0| | | | |
|`NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION`|8.0| | | |`HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION`|2.6.0| | | | |
|`NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION`|8.0| | | |`HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION`|2.6.0| | | | |
|`NVRTC_ERROR_OUT_OF_MEMORY`| | | | |`HIPRTC_ERROR_OUT_OF_MEMORY`|2.6.0| | | | |
|`NVRTC_ERROR_PROGRAM_CREATION_FAILURE`| | | | |`HIPRTC_ERROR_PROGRAM_CREATION_FAILURE`|2.6.0| | | | |
|`NVRTC_ERROR_TIME_FILE_WRITE_FAILED`|12.1| | | | | | | | | |
|`NVRTC_SUCCESS`| | | | |`HIPRTC_SUCCESS`|2.6.0| | | | |
|`nvrtcProgram`| | | | |`hiprtcProgram`|2.6.0| | | | |
|`nvrtcResult`| | | | |`hiprtcResult`|2.6.0| | | | |

## **2. RTC API functions**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`nvrtcAddNameExpression`|8.0| | | |`hiprtcAddNameExpression`|2.6.0| | | | |
|`nvrtcCompileProgram`| | |8.0| |`hiprtcCompileProgram`|2.6.0| | | | |
|`nvrtcCreateProgram`| | |8.0| |`hiprtcCreateProgram`|2.6.0| | | | |
|`nvrtcDestroyProgram`| | | | |`hiprtcDestroyProgram`|2.6.0| | | | |
|`nvrtcGetCUBIN`|11.1| | | |`hiprtcGetBitcode`|5.3.0| | | | |
|`nvrtcGetCUBINSize`|11.1| | | |`hiprtcGetBitcodeSize`|5.3.0| | | | |
|`nvrtcGetErrorString`| | | | |`hiprtcGetErrorString`|2.6.0| | | | |
|`nvrtcGetLTOIR`|12.0| | | | | | | | | |
|`nvrtcGetLTOIRSize`|12.0| | | | | | | | | |
|`nvrtcGetLoweredName`|8.0| | | |`hiprtcGetLoweredName`|2.6.0| | | | |
|`nvrtcGetNVVM`|11.4|12.0| | | | | | | | |
|`nvrtcGetNVVMSize`|11.4|12.0| | | | | | | | |
|`nvrtcGetNumSupportedArchs`|11.2| | | | | | | | | |
|`nvrtcGetOptiXIR`|12.0| | | | | | | | | |
|`nvrtcGetOptiXIRSize`|12.0| | | | | | | | | |
|`nvrtcGetPTX`| | | | |`hiprtcGetCode`|2.6.0| | | | |
|`nvrtcGetPTXSize`| | | | |`hiprtcGetCodeSize`|2.6.0| | | | |
|`nvrtcGetProgramLog`| | | | |`hiprtcGetProgramLog`|2.6.0| | | | |
|`nvrtcGetProgramLogSize`| | | | |`hiprtcGetProgramLogSize`|2.6.0| | | | |
|`nvrtcGetSupportedArchs`|11.2| | | | | | | | | |
|`nvrtcVersion`| | | | |`hiprtcVersion`|2.6.0| | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental