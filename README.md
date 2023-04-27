# HIPIFY

HIPIFY is a set of tools to translate CUDA source code into portable [HIP](https://github.com/ROCm-Developer-Tools/HIP) C++ automatically.

## Table of Contents

<!-- toc -->

- [hipify-clang](./docs/hipify-clang.md#hipify-clang)
     * [Dependencies](./docs/hipify-clang.md#hipify-clang-dependencies)
     * [Usage](./docs/hipify-clang.md#hipify-clang-usage)
     * [Building](./docs/hipify-clang.md#hipify-clang-building)
     * [Testing](./docs/hipify-clang.md#hipify-clang-testing)
        + [Linux](./docs/hipify-clang.md#hipify-clang-linux-testing)
        + [Windows](./docs/hipify-clang.md#hipify-clang-windows-testing)
- [hipify-perl](./docs/hipify-perl.md#hipify-perl)
     * [Usage](./docs/hipify-perl.md#hipify-perl-usage)
     * [Building](./docs/hipify-perl.md#hipify-perl-building)
- [Supported CUDA APIs](./docs/supported_apis.md#supported-cuda-apis)
- [Documentation](#documentation)
- [Disclaimer](#disclaimer)

<!-- tocstop -->

## <a name="documentation"></a>Documentation

Information about HIPIFY and other user topics is found in the [HIPIFY documentation](https://rocmdocs.amd.com/projects/HIPIFY/en/latest/).

### How to build documentation

Run the steps below to build documentation locally.

```
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## <a name="disclaimer"></a>Disclaimer

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright &copy; 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
