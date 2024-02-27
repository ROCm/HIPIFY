# HIPIFY

HIPIFY is a set of tools that you can use to automatically translate CUDA source code into portable
[HIP](https://github.com/ROCm/HIP) C++.

## Table of contents

<!-- toc -->

* [hipify-clang](./docs/hipify-clang.md#hipify-clang)
  * [Dependencies](./docs/hipify-clang.md#hipify-clang-dependencies)
  * [Usage](./docs/hipify-clang.md#hipify-clang-usage)
    * [JSON Compilation Database](./docs/hipify-clang.md#hipify-clang-using-json-compilation-database)
    * [Hipification Statistics](./docs/hipify-clang.md#hipify-clang-hipification-statistics)
  * [Building](./docs/hipify-clang.md#hipify-clang-building)
  * [Testing](./docs/hipify-clang.md#hipify-clang-testing)
    * [Linux](./docs/hipify-clang.md#hipify-clang-linux-testing)
    * [Windows](./docs/hipify-clang.md#hipify-clang-windows-testing)
* [hipify-perl](./docs/hipify-perl.md#hipify-perl)
  * [Usage](./docs/hipify-perl.md#hipify-perl-usage)
  * [Building](./docs/hipify-perl.md#hipify-perl-building)
* Related: [hipify_torch](https://github.com/ROCmSoftwarePlatform/hipify_torch)
* [Supported CUDA APIs](./docs/supported_apis.md#supported-cuda-apis)
* [Documentation](#documentation)

<!-- tocstop -->

## Documentation

Documentation for HIPIFY is available at
[https://rocmdocs.amd.com/projects/HIPIFY/en/latest/](https://rocmdocs.amd.com/projects/HIPIFY/en/latest/).

To build our documentation locally, run the following code.

```bash
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

To build `CUDA2HIP` (CUDA APIs supported by HIP) documentation, run the following `hipify-clang`
command. This builds the same content as
[Supported CUDA APIs](./docs/supported_apis.md#supported-cuda-apis).

```bash
hipify-clang --md --doc-format=full --doc-roc=joint

# Alternatively, you can use:

hipify-clang --md --doc-format=full --doc-roc=separate
```

To generate this documentation in CSV, use the `--csv` option instead of `--md`. Instead of using
the `full` format, you can also build in `strict` or `compact` format.

To see all available options, use the `--help` or `--help-hidden` `hipify-clang` option.
