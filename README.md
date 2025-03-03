# HIPIFY

HIPIFY is a set of tools that you can use to automatically translate CUDA source code into portable
[HIP](https://github.com/ROCm/HIP) C++.

## Documentation

The published documentation is available at [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `HIPIFY/docs` folder of this GitHub repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

To build our documentation locally, run the following code.

```bash
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

To build `CUDA2HIP` (CUDA APIs supported by HIP) documentation, run the following `hipify-clang`
command. This builds the same content as
[Supported CUDA APIs](./docs/reference/supported_apis.md#supported-cuda-apis).

```bash
hipify-clang --md --doc-format=full --doc-roc=joint

# Alternatively, you can use:

hipify-clang --md --doc-format=full --doc-roc=separate
```

To generate this documentation in CSV, use the `--csv` option instead of `--md`. Instead of using
the `full` format, you can also build in `strict` or `compact` format.

To see all available options, use the `--help` or `--help-hidden` `hipify-clang` option.
