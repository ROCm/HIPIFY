# HIPIFY

HIPIFY is a set of tools that you can use to automatically translate CUDA source code into portable
[HIP](https://github.com/ROCm/HIP) C++.

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
