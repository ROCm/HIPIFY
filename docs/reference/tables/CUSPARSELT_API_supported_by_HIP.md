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
|`cusparseLtHandle_t`|0.0.1| | | |`hipsparseLtHandle_t`|7.2.0| | | | | |
|`cusparseLtMatDescriptor_t`|0.0.1| | | |`hipsparseLtMatDescriptor_t`|7.2.0| | | | | |
|`cusparseLtMatmulAlgSelection_t`|0.0.1| | | |`hipsparseLtMatmulAlgSelection_t`|7.2.0| | | | | |
|`cusparseLtMatmulDescriptor_t`|0.0.1| | | |`hipsparseLtMatmulDescriptor_t`|7.2.0| | | | | |
|`cusparseLtMatmulPlan_t`|0.0.1| | | |`hipsparseLtMatmulPlan_t`|7.2.0| | | | | |

## **2. CUSPARSELT Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**U**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
|`cusparseLtInit`|0.0.1| | | |`hipsparseLtInit`|7.2.0| | | | | |

