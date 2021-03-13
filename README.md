# HIPIFY

### Tools to translate CUDA source code into portable HIP C++ automatically
## Table of Contents

<!-- toc -->

- [hipify-clang](#clang)
     * [Dependencies](#dependencies)
     * [Usage](#hipify-clang-usage)
     * [Building](#building)
     * [Testing](#testing)
     * [Linux](#linux)
     * [Windows](#windows)
- [hipify-perl](#perl)
     * [Usage](#hipify-perl-usage)
     * [Building](#hipify-perl-building)
- [Supported CUDA APIs](#cuda-apis)
- [Disclaimer](#disclaimer)

<!-- tocstop -->

## <a name="clang"></a> hipify-clang

`hipify-clang` is a clang-based tool for translating CUDA sources into HIP sources.
It translates CUDA source into an abstract syntax tree, which is traversed by transformation matchers.
After applying all the matchers, the output HIP source is produced.

**Advantages:**

1. It is a translator; thus, any even very complicated constructs will be parsed successfully, or an error will be reported.
2. It supports clang options like [`-I`](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-i-dir), [`-D`](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-d-macro), [`--cuda-path`](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-cuda-path), etc.
3. Seamless support of new CUDA versions as it is clang's responsibility.
4. Ease of support.

**Disadvantages:**

1. The main advantage is also the main disadvantage: the input CUDA code should be correct; incorrect code wouldn't be translated to HIP.
2. CUDA should be installed and provided in case of multiple installations by `--cuda-path` option.
3. All the includes and defines should be provided to transform code successfully.

### <a name="dependencies"></a> hipify-clang: dependencies

`hipify-clang` requires:

1. [**LLVM+CLANG**](http://releases.llvm.org) of at least version [3.8.0](http://releases.llvm.org/download.html#3.8.0); the latest stable and recommended release: [**11.1.0**](https://github.com/llvm/llvm-project/releases/tag/llvmorg-11.1.0).

2. [**CUDA**](https://developer.nvidia.com/cuda-downloads) of at least version [7.0](https://developer.nvidia.com/cuda-toolkit-70), the latest supported version is [**11.2.2**](https://developer.nvidia.com/cuda-downloads).

<table align="center">
  <thead>
     <tr align="center">
       <th>LLVM release version</th>
       <th>CUDA latest supported version</th>
       <th>Windows</th>
       <th>Linux</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td><a href="http://releases.llvm.org/download.html#3.8.0">3.8.0</a>*,
          <a href="http://releases.llvm.org/download.html#3.8.1">3.8.1</a>*,<br>
          <a href="http://releases.llvm.org/download.html#3.9.0">3.9.0</a>*,
          <a href="http://releases.llvm.org/download.html#3.9.1">3.9.1</a>*</td>
      <td><a href="https://developer.nvidia.com/cuda-75-downloads-archive">7.5</a></td>
      <td>+</td>
      <td>+</td>
    </tr>
    <tr align="center">
      <td><a href="http://releases.llvm.org/download.html#4.0.0">4.0.0</a>,
          <a href="http://releases.llvm.org/download.html#4.0.1">4.0.1</a>,<br>
          <a href="http://releases.llvm.org/download.html#5.0.0">5.0.0</a>,
          <a href="http://releases.llvm.org/download.html#5.0.1">5.0.1</a>,
          <a href="http://releases.llvm.org/download.html#5.0.2">5.0.2</a></td>
      <td><a href="https://developer.nvidia.com/cuda-80-ga2-download-archive">8.0</a></td>
      <td>+</td>
      <td>+</td>
    </tr>
    <tr align="center">
      <td><a href="http://releases.llvm.org/download.html#6.0.0">6.0.0</a>,
          <a href="http://releases.llvm.org/download.html#6.0.1">6.0.1</a></td>
      <td><a href="https://developer.nvidia.com/cuda-90-download-archive">9.0</a></td>
      <td>+</td>
      <td>+</td>
    </tr>
    <tr align="center">
      <td><a href="http://releases.llvm.org/download.html#7.0.0">7.0.0</a>,
          <a href="http://releases.llvm.org/download.html#7.0.1">7.0.1</a>,
          <a href="http://releases.llvm.org/download.html#7.1.0">7.1.0</a></td>
      <td><a href="https://developer.nvidia.com/cuda-92-download-archive">9.2</a></td>
      <td>works only with the patch <br> due to the clang's bug <a href="https://bugs.llvm.org/show_bug.cgi?id=38811">38811</a><br>
          <a href="patches/patch_for_clang_7.0.0_bug_38811.zip">patch for 7.0.0</a>**<br>
          <a href="patches/patch_for_clang_7.0.1_bug_38811.zip">patch for 7.0.1</a>**<br>
          <a href="patches/patch_for_clang_7.1.0_bug_38811.zip">patch for 7.1.0</a>**<br></td>
      <td>-<br> not working due to <br> the clang's bug <a href="https://bugs.llvm.org/show_bug.cgi?id=36384">36384</a></td>
    </tr>
    <tr align="center">
      <td><a href="http://releases.llvm.org/download.html#8.0.0">8.0.0</a>,
          <a href="http://releases.llvm.org/download.html#8.0.1">8.0.1</a></td>
      <td><a href="https://developer.nvidia.com/cuda-10.0-download-archive">10.0</a></td>
      <td>works only with the patch <br> due to the clang's bug <a href="https://bugs.llvm.org/show_bug.cgi?id=38811">38811</a><br>
          <a href="patches/patch_for_clang_8.0.0_bug_38811.zip">patch for 8.0.0</a>**<br>
          <a href="patches/patch_for_clang_8.0.1_bug_38811.zip">patch for 8.0.1</a>**<br></td>
      <td>+</td>
    </tr>
    <tr align="center">
      <td><a href="http://releases.llvm.org/download.html#9.0.0">9.0.0</a>,
          <a href="http://releases.llvm.org/download.html#9.0.1">9.0.1</a></td>
      <td><a href="https://developer.nvidia.com/cuda-10.1-download-archive-base">10.1</a></td>
      <td>+</td>
      <td>+</td>
    </tr>
    <tr align="center">
      <td rowspan=2><a href="http://releases.llvm.org/download.html#10.0.0">10.0.0</a>,
          <a href="http://releases.llvm.org/download.html#10.0.1">10.0.1</a></td>
      <td><a href="https://developer.nvidia.com/cuda-11.0-download-archive">11.0</a></td>
      <td>+</td>
      <td>+</td>
    </tr>
    <tr align="center">
      <td><a href="https://developer.nvidia.com/cuda-11.0-update1-download-archive">11.0.1</a>,
          <a href="https://developer.nvidia.com/cuda-11.1.0-download-archive">11.1.0</a>,
          <a href="https://developer.nvidia.com/cuda-11.1.1-download-archive">11.1.1</a></td>
      <td colspan=2>works only with the patch <br> due to the clang's bug <a href="https://bugs.llvm.org/show_bug.cgi?id=47332">47332</a><br>
          <a href="patches/patch_for_clang_10.0.0_bug_47332.zip">patch for 10.0.0</a>***<br>
          <a href="patches/patch_for_clang_10.0.1_bug_47332.zip">patch for 10.0.1</a>***<br></td>
    </tr>
    <tr align="center">
      <td rowspan=2><a href="http://releases.llvm.org/download.html#11.0.0">11.0.0</a></td>
      <td><a href="https://developer.nvidia.com/cuda-11.0-download-archive">11.0</a></td>
      <td>+</td>
      <td>+</td>
    </tr>
    <tr align="center">
      <td><a href="https://developer.nvidia.com/cuda-11.0-update1-download-archive">11.0.1</a>,
          <a href="https://developer.nvidia.com/cuda-11.1.0-download-archive">11.1.0</a>,
          <a href="https://developer.nvidia.com/cuda-11.1.1-download-archive">11.1.1</a></td>
      <td colspan=2>works only with the patch <br> due to the clang's bug <a href="https://bugs.llvm.org/show_bug.cgi?id=47332">47332</a><br>
          <a href="patches/patch_for_clang_11.0.0_bug_47332.zip">patch for 11.0.0</a>***</td>
    </tr>
    <tr align="center">
      <td bgcolor="eefaeb"><a href="https://releases.llvm.org/download.html#11.0.1">11.0.1</a>,
                           <a href="https://github.com/llvm/llvm-project/releases/tag/llvmorg-11.1.0"><b>11.1.0</b></a>
      <td bgcolor="eefaeb"><a href="https://developer.nvidia.com/cuda-downloads"><b>11.2.2</b></a></td>
      <td colspan=2 bgcolor="eefaeb"><font color="green"><b>LATEST STABLE CONFIG</b></font></td>
    </tr>
  </tbody>
</table>

`*`   `LLVM 3.x` is not supported anymore but might still work.

`**`  Download the patch and unpack it into your `LLVM distributive directory`: a few header files will be overwritten; rebuilding of `LLVM` is not needed.

`***` Download the patch and unpack it into your `LLVM source directory`: the file `Cuda.cpp` will be overwritten; needs further rebuilding of `LLVM`.

In most cases, you can get a suitable version of `LLVM+CLANG` with your package manager.

Failing that or having multiple versions of `LLVM`, you can [download a release archive](http://releases.llvm.org/), build or install it, and set
[CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/v3.5/variable/CMAKE_PREFIX_PATH.html) so `cmake` can find it; for instance: `-DCMAKE_PREFIX_PATH=d:\LLVM\11.1.0\dist`

### <a name="hipify-clang-usage"></a> hipify-clang: usage

To process a file, `hipify-clang` needs access to the same headers that would be required to compile it with clang.

For example:

```shell
./hipify-clang square.cu --cuda-path=/usr/local/cuda-11.2 -I /usr/local/cuda-11.2/samples/common/inc
```

`hipify-clang` arguments are given first, followed by a separator `'--'`, and then the arguments you'd pass to `clang` if you
were compiling the input file. For example:

```bash
./hipify-clang cpp17.cu --cuda-path=/usr/local/cuda-11.2 -- -std=c++17
```

The [Clang manual for compiling CUDA](https://llvm.org/docs/CompileCudaWithLLVM.html#compiling-cuda-code) may be useful.

For some hipification automation (starting from clang 8.0.0), it is also possible to provide a [Compilation Database in JSON format](https://clang.llvm.org/docs/JSONCompilationDatabase.html) in the `compile_commands.json` file:

```bash
-p <folder containing compile_commands.json> or
-p=<folder containing compile_commands.json>
```

The compilation database should be provided in the `compile_commands.json` file or generated by clang based on cmake; options separator `'--'` must not be used.


For a list of `hipify-clang` options, run `hipify-clang --help`.

### <a name="building"></a> hipify-clang: building

```bash
mkdir build dist
cd build

cmake \
 -DCMAKE_INSTALL_PREFIX=../dist \
 -DCMAKE_BUILD_TYPE=Release \
 ..

make -j install
```
On Windows, the following option should be specified for `cmake` at first place: `-G "Visual Studio 16 2019 Win64"`; the generated `hipify-clang.sln` should be built by `Visual Studio 16 2019` instead of `make.`
Please, see [hipify-clang: Windows](#windows) for the supported tools for building.

Debug build type `-DCMAKE_BUILD_TYPE=Debug` is also supported and tested; `LLVM+CLANG` should be built in `Debug` mode as well.
64-bit build mode (`-Thost=x64` on Windows) is also supported; `LLVM+CLANG` should be built in 64-bit mode as well.

The binary can then be found at `./dist/bin/hipify-clang`.

### <a name="testing"></a> hipify-clang: testing

`hipify-clang` has unit tests using `LLVM` [`lit`](https://llvm.org/docs/CommandGuide/lit.html)/[`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html).

`LLVM+CLANG` should be built from sources, pre-built binaries are not exhaustive for testing. Before building ensure that the [software required for building](https://releases.llvm.org/11.0.0/docs/GettingStarted.html#software) is of an appropriate version.

**LLVM 9.0.1 or older:**

1. download [`LLVM`](https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/llvm-9.0.1.src.tar.xz)+[`CLANG`](https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/clang-9.0.1.src.tar.xz) sources; 
2. build [`LLVM+CLANG`](http://releases.llvm.org/9.0.0/docs/CMake.html):

 **Linux**:
   ```bash
        cmake \
         -DCMAKE_INSTALL_PREFIX=../dist \
         -DLLVM_SOURCE_DIR=../llvm \
         -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
         -DCMAKE_BUILD_TYPE=Release \
         ../llvm
        make -j install
   ```
 **Windows**:
   ```shell
        cmake \
         -G "Visual Studio 16 2019" \
         -A x64 \
         -DCMAKE_INSTALL_PREFIX=../dist \
         -DLLVM_SOURCE_DIR=../llvm \
         -DLLVM_TARGETS_TO_BUILD="NVPTX" \
         -DCMAKE_BUILD_TYPE=Release \
         -Thost=x64 \
         ../llvm
   ```
Run `Visual Studio 16 2019`, open the generated `LLVM.sln`, build all, build project `INSTALL`.

**LLVM 10.0.0 or newer:**

1. download [`LLVM project`](https://releases.llvm.org/download.html#11.1.0) sources;
2. build [`LLVM project`](http://llvm.org/docs/CMake.html):

 **Linux**:
   ```bash
        cmake \
         -DCMAKE_INSTALL_PREFIX=../dist \
         -DLLVM_SOURCE_DIR=../llvm-project \
         -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
         -DLLVM_ENABLE_PROJECTS="clang" \
         -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON \
         -DCMAKE_BUILD_TYPE=Release \
         ../llvm-project/llvm
        make -j install
   ```
 **Windows**:
   ```shell
        cmake \
         -G "Visual Studio 16 2019" \
         -A x64 \
         -DCMAKE_INSTALL_PREFIX=../dist \
         -DLLVM_SOURCE_DIR=../llvm-project \
         -DLLVM_TARGETS_TO_BUILD="NVPTX" \
         -DLLVM_ENABLE_PROJECTS="clang" \
         -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -Thost=x64 \
         ../llvm-project/llvm
   ```
Run `Visual Studio 16 2019`, open the generated `LLVM.sln`, build all, build project `INSTALL`.

3. Ensure [`CUDA`](https://developer.nvidia.com/cuda-toolkit-archive) of minimum version 7.0 is installed.

    * Having multiple CUDA installations to choose a particular version the `DCUDA_TOOLKIT_ROOT_DIR` option should be specified:

        - ***Linux***: `-DCUDA_TOOLKIT_ROOT_DIR=/usr/include`

        - ***Windows***: `-DCUDA_TOOLKIT_ROOT_DIR="c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"`

          `-DCUDA_SDK_ROOT_DIR="c:/ProgramData/NVIDIA Corporation/CUDA Samples/v11.2"`

4. Ensure [`cuDNN`](https://developer.nvidia.com/rdp/cudnn-archive) of the version corresponding to CUDA's version is installed.

    * Path to [`cuDNN`](https://developer.nvidia.com/rdp/cudnn-download) should be specified by the `CUDA_DNN_ROOT_DIR` option:

        - ***Linux***: `-DCUDA_DNN_ROOT_DIR=/usr/include`

        - ***Windows***: `-DCUDA_DNN_ROOT_DIR=d:/CUDNN/cudnn-11.2-windows-x64-v8.1.1`

5. Ensure [`CUB`](https://github.com/NVlabs/cub) of the version corresponding to CUDA's version is installed.

    * Path to CUB should be specified by the `CUDA_CUB_ROOT_DIR` option:

        - ***Linux***: `-DCUDA_CUB_ROOT_DIR=/srv/git/CUB`

        - ***Windows***: `-DCUDA_CUB_ROOT_DIR=d:/GIT/cub`

5. Ensure [`python`](https://www.python.org/downloads) of minimum required version 2.7 is installed.

6. Ensure `lit` and `FileCheck` are installed - these are distributed with `LLVM`.

    * Install `lit` into `python`:

        - ***Linux***: `python /usr/llvm/11.1.0/llvm-project/llvm/utils/lit/setup.py install`

        - ***Windows***: `python d:/LLVM/11.1.0/llvm-project/llvm/utils/lit/setup.py install`

    * Starting with LLVM 6.0.1 path to `llvm-lit` python script should be specified by the `LLVM_EXTERNAL_LIT` option:

        - ***Linux***: `-DLLVM_EXTERNAL_LIT=/usr/llvm/11.1.0/build/bin/llvm-lit`

        - ***Windows***: `-DLLVM_EXTERNAL_LIT=d:/LLVM/11.1.0/build/Release/bin/llvm-lit.py`

    * `FileCheck`:

        - ***Linux***: copy from `/usr/llvm/11.1.0/build/bin/` to `CMAKE_INSTALL_PREFIX/dist/bin`

        - ***Windows***: copy from `d:/LLVM/11.1.0/build/Release/bin` to `CMAKE_INSTALL_PREFIX/dist/bin`

        - Or specify the path to `FileCheck` in `CMAKE_INSTALL_PREFIX` option

7. Set `HIPIFY_CLANG_TESTS` option turned on: `-DHIPIFY_CLANG_TESTS=1`.

8. Build and run tests:

### <a name="Linux"></a > hipify-clang: Linux

On Linux the following configurations are tested:

Ubuntu 14: LLVM 4.0.0 - 7.1.0, CUDA 7.0 - 9.0, cuDNN 5.0.5 - 7.6.5.32

Ubuntu 16-18: LLVM 8.0.0 - 11.1.0, CUDA 8.0 - 10.2, cuDNN 5.1.10 - 8.0.5.39

Ubuntu 20: LLVM 9.0.0 - 11.1.0, CUDA 8.0 - 11.2.2, cuDNN 5.1.10 - 8.1.1.33

Minimum build system requirements for the above configurations:

Python 2.7, cmake 3.5.1, GNU C/C++ 5.4.0.

Here is an example of building `hipify-clang` with testing support on `Ubuntu 20.04.1`:

```bash
cmake
 -DHIPIFY_CLANG_TESTS=1 \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../dist \
 -DCMAKE_PREFIX_PATH=/usr/llvm/11.1.0/dist \
 -DCUDA_TOOLKIT_ROOT_DIR=/usr/include \
 -DCUDA_DNN_ROOT_DIR=/usr/include \
 -DCUDA_CUB_ROOT_DIR=/usr/CUB \
 -DLLVM_EXTERNAL_LIT=/usr/llvm/11.1.0/build/bin/llvm-lit \
 ..
```
*A corresponding successful output:*
```shell
-- The C compiler identification is GNU 9.3.0
-- The CXX compiler identification is GNU 9.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found ZLIB: /usr/lib/x86_64-linux-gnu/libz.so (found version "1.2.11")
-- Found LLVM 11.1.0:
--    - CMake module path: /usr/llvm/11.1.0/dist/lib/cmake/llvm
--    - Include path     : /usr/llvm/11.1.0/dist/include
--    - Binary path      : /usr/llvm/11.1.0/dist/bin
-- Linker detection: GNU ld
-- Found PythonInterp: /usr/bin/python3.8 (found suitable version "3.8.5", minimum required is "2.7")
-- Found lit: /usr/local/bin/lit
-- Found FileCheck: /usr/llvm/11.1.0/dist/bin/FileCheck
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Found CUDA: /usr/include (found version "11.2")
-- Configuring done
-- Generating done
-- Build files have been written to: /usr/hipify/build
```
```shell
make test-hipify
```
*A corresponding successful output:*
```shell
Running HIPify regression tests
========================================
CUDA 11.2 - will be used for testing
LLVM 11.1.0 - will be used for testing
x86_64 - Platform architecture
Linux 5.4.0-51-generic - Platform OS
64 - hipify-clang binary bitness
64 - python 3.8.5 binary bitness
========================================
-- Testing: 67 tests, 12 threads --
PASS: hipify :: unit_tests/casts/reinterpret_cast.cu (1 of 67)
PASS: hipify :: unit_tests/device/atomics.cu (2 of 67)
PASS: hipify :: unit_tests/compilation_database/cd_intro.cu (3 of 67)
PASS: hipify :: unit_tests/device/device_symbols.cu (4 of 67)
PASS: hipify :: unit_tests/device/math_functions.cu (5 of 67)
PASS: hipify :: unit_tests/headers/headers_test_01.cu (6 of 67)
PASS: hipify :: unit_tests/headers/headers_test_02.cu (7 of 67)
PASS: hipify :: unit_tests/headers/headers_test_03.cu (8 of 67)
PASS: hipify :: unit_tests/headers/headers_test_05.cu (9 of 67)
PASS: hipify :: unit_tests/headers/headers_test_06.cu (10 of 67)
PASS: hipify :: unit_tests/headers/headers_test_04.cu (11 of 67)
PASS: hipify :: unit_tests/headers/headers_test_07.cu (12 of 67)
PASS: hipify :: unit_tests/headers/headers_test_10.cu (13 of 67)
PASS: hipify :: unit_tests/headers/headers_test_11.cu (14 of 67)
PASS: hipify :: unit_tests/headers/headers_test_08.cu (15 of 67)
PASS: hipify :: unit_tests/kernel_launch/kernel_launch_01.cu (16 of 67)
PASS: hipify :: unit_tests/headers/headers_test_09.cu (17 of 67)
PASS: hipify :: unit_tests/libraries/CAFFE2/caffe2_02.cu (18 of 67)
PASS: hipify :: unit_tests/libraries/CAFFE2/caffe2_01.cu (19 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/cublas_0_based_indexing.cu (20 of 67)
PASS: hipify :: unit_tests/libraries/CUB/cub_03.cu (21 of 67)
PASS: hipify :: unit_tests/libraries/CUB/cub_01.cu (22 of 67)
PASS: hipify :: unit_tests/libraries/CUB/cub_02.cu (23 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/cublas_sgemm_matrix_multiplication.cu (24 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/rocBLAS/cublas_0_based_indexing_rocblas.cu (25 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/rocBLAS/cublas_1_based_indexing_rocblas.cu (26 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/cublas_1_based_indexing.cu (27 of 67)
PASS: hipify :: unit_tests/libraries/cuComplex/cuComplex_Julia.cu (28 of 67)
PASS: hipify :: unit_tests/libraries/cuDNN/cudnn_softmax.cu (29 of 67)
PASS: hipify :: unit_tests/libraries/cuFFT/simple_cufft.cu (30 of 67)
PASS: hipify :: unit_tests/libraries/cuBLAS/rocBLAS/cublas_sgemm_matrix_multiplication_rocblas.cu (31 of 67)
PASS: hipify :: unit_tests/libraries/cuRAND/poisson_api_example.cu (32 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_03.cu (33 of 67)
PASS: hipify :: unit_tests/libraries/cuRAND/benchmark_curand_generate.cpp (34 of 67)
PASS: hipify :: unit_tests/libraries/cuRAND/benchmark_curand_kernel.cpp (35 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_04.cu (36 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_05.cu (37 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_06.cu (38 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_07.cu (39 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_08.cu (40 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_09.cu (41 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_11.cu (42 of 67)
PASS: hipify :: unit_tests/namespace/ns_kernel_launch.cu (43 of 67)
PASS: hipify :: unit_tests/libraries/cuSPARSE/cuSPARSE_10.cu (44 of 67)
PASS: hipify :: unit_tests/pp/pp_if_else_conditionals.cu (45 of 67)
PASS: hipify :: unit_tests/pp/pp_if_else_conditionals_01.cu (46 of 67)
PASS: hipify :: unit_tests/pp/pp_if_else_conditionals_01_LLVM_10.cu (47 of 67)
PASS: hipify :: unit_tests/pp/pp_if_else_conditionals_LLVM_10.cu (48 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/11_texture_driver/tex2dKernel.cpp (49 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/0_MatrixTranspose/MatrixTranspose.cpp (50 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/11_texture_driver/texture2dDrv.cpp (51 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/13_occupancy/occupancy.cpp (52 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/1_hipEvent/hipEvent.cpp (53 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/2_Profiler/Profiler.cpp (54 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/7_streams/stream.cpp (55 of 67)
PASS: hipify :: unit_tests/samples/MallocManaged.cpp (56 of 67)
PASS: hipify :: unit_tests/samples/2_Cookbook/8_peer2peer/peer2peer.cpp (57 of 67)
PASS: hipify :: unit_tests/samples/allocators.cu (58 of 67)
PASS: hipify :: unit_tests/samples/coalescing.cu (59 of 67)
PASS: hipify :: unit_tests/samples/dynamic_shared_memory.cu (60 of 67)
PASS: hipify :: unit_tests/samples/axpy.cu (61 of 67)
PASS: hipify :: unit_tests/samples/cudaRegister.cu (62 of 67)
PASS: hipify :: unit_tests/samples/intro.cu (63 of 67)
PASS: hipify :: unit_tests/samples/square.cu (64 of 67)
PASS: hipify :: unit_tests/samples/static_shared_memory.cu (65 of 67)
PASS: hipify :: unit_tests/samples/vec_add.cu (66 of 67)
PASS: hipify :: unit_tests/kernel_launch/kernel_launch_syntax.cu (67 of 67)
Testing Time: 2.91s
  Expected Passes    : 67
[100%] Built target test-hipify
```
### <a name="windows"></a > hipify-clang: Windows

*Tested configurations:*

|      **LLVM**   | **CUDA**     |      **cuDNN**      | **Visual Studio (latest)**|   **cmake**    |  **Python**  |
|----------------:|-------------:|--------------------:|--------------------------:|---------------:|-------------:|
| 4.0.0 - 5.0.2   | 8.0          | 5.1.10   - 7.1.4.18 | 2015.14.0, 2017.15.5.2    | 3.5.1, 3.18.0  | 3.6.4, 3.8.5 |
| 6.0.0 - 6.0.1   | 9.0          | 7.0.5.15 - 7.6.5.32 | 2015.14.0, 2017.15.5.5    | 3.6.0, 3.18.0  | 3.7.2, 3.8.5 |
| 7.0.0 - 7.1.0   | 9.2          | 7.6.5.32            | 2017.15.9.11              | 3.13.3, 3.18.0 | 3.7.3, 3.8.5 |
| 8.0.0 - 8.0.1   | 10.0         | 7.6.5.32            | 2017.15.9.15              | 3.14.2, 3.18.0 | 3.7.4, 3.8.5 |
| 9.0.0 - 9.0.1   | 10.1         | 7.6.5.32            | 2017.15.9.20, 2019.16.4.5 | 3.16.4, 3.18.0 | 3.8.0, 3.8.5 |
| 10.0.0 - 11.0.0 | 8.0 - 11.1   | 7.6.5.32 - 8.0.5.39 | 2017.15.9.30, 2019.16.8.3 | 3.19.2         | 3.9.1        |
| 11.0.1 - 11.1.0 | 8.0 - 11.2.2 | 7.6.5.32 - 8.0.5.39 | 2017.15.9.31, 2019.16.8.4 | 3.19.3         | 3.9.2        |
| 13.0.0git       | 8.0 - 11.2.2 | 7.6.5.32 - 8.1.1.33 | 2017.15.9.34, 2019.16.9.1 | 3.19.4         | 3.9.2        |

*Building with testing support by `Visual Studio 16 2019` on `Windows 10`:*

```shell
cmake
 -G "Visual Studio 16 2019" \
 -A x64 \
 -DHIPIFY_CLANG_TESTS=1 \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../dist \
 -DCMAKE_PREFIX_PATH=d:/LLVM/11.1.0/dist \
 -DCUDA_TOOLKIT_ROOT_DIR="c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2" \
 -DCUDA_SDK_ROOT_DIR="c:/ProgramData/NVIDIA Corporation/CUDA Samples/v11.2" \
 -DCUDA_DNN_ROOT_DIR=d:/CUDNN/cudnn-11.2-windows-x64-v8.1.1 \
 -DCUDA_CUB_ROOT_DIR=d:/GIT/cub \
 -DLLVM_EXTERNAL_LIT=d:/LLVM/11.1.0/build/Release/bin/llvm-lit.py \
 -Thost=x64
 ..
```
*A corresponding successful output:*
```shell
-- Found LLVM 11.1.0:
--    - CMake module path: d:/LLVM/11.1.0/dist/lib/cmake/llvm
--    - Include path     : d:/LLVM/11.1.0/dist/include
--    - Binary path      : d:/LLVM/11.1.0/dist/bin
-- Found PythonInterp: c:/Program Files/Python39/python.exe (found suitable version "3.9.2", minimum required is "3.6")
-- Found lit: c:/Program Files/Python39/Scripts/lit.exe
-- Found FileCheck: d:/LLVM/11.1.0/dist/bin/FileCheck.exe
-- Found CUDA: c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2 (found version "11.2")
-- Configuring done
-- Generating done
-- Build files have been written to: d:/hipify/build
```

Run `Visual Studio 16 2019`, open the generated `hipify-clang.sln`, build project `test-hipify`.

## <a name="perl"></a> hipify-perl

`hipify-perl` is an autogenerated perl-based script which heavily uses regular expressions.

**Advantages:**

1. Ease of use.

2. It doesn't check the input source CUDA code for correctness.

3. It doesn't have dependencies on 3rd party tools, including CUDA.

**Disadvantages:**

1. Current disability (and difficulty in implementing) of transforming the following constructs:

    * macros expansion;

    * namespaces:

        - redefines of CUDA entities in user namespaces;

        - using directive;

    * templates (some cases);

    * device/host function calls distinguishing;

    * header files correct injection;

    * complicated argument lists parsing.

2. Difficulties in supporting.

### <a name="hipify-perl-usage"></a> hipify-perl: usage

```shell
perl hipify-perl square.cu > square.cu.hip
```

### <a name="hipify-perl-building"></a> hipify-perl: building

To generate `hipify-perl`, run `hipify-clang --perl`. Output directory for the generated `hipify-perl` file might be specified by `--o-hipify-perl-dir` option.

## <a name="cuda-apis"></a> Supported CUDA APIs

- [Runtime API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md)
- [Driver API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDA_Driver_API_functions_supported_by_HIP.md)
- [cuComplex API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/cuComplex_API_supported_by_HIP.md)
- [Device API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDA_Device_API_supported_by_HIP.md)
- [cuBLAS](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUBLAS_API_supported_by_HIP.md)
- [cuRAND](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CURAND_API_supported_by_HIP.md)
- [cuDNN](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDNN_API_supported_by_HIP.md)
- [cuFFT](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUFFT_API_supported_by_HIP.md)
- [cuSPARSE](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUSPARSE_API_supported_by_HIP.md)

To generate the above documentation with the actual information about all supported CUDA APIs in Markdown format, run `hipify-clang --md` with or without output directory specifying (`-o`).

## <a name="disclaimer"></a> Disclaimer

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
