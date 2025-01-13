.. meta::
   :description: Tools to automatically translate CUDA source code into portable HIP C++
   :keywords: HIPIFY, ROCm, library, tool, CUDA, CUDA2HIP, hipify-clang, hipify-perl

.. _build-hipify-clang:

**************************************************************************
Building hipify-clang
**************************************************************************

After cloning the HIPIFY repository (``git clone https://github.com/ROCm/HIPIFY.git``), run the following commands from the HIPIFY root folder.

.. code-block:: bash

  cd .. \
  mkdir build dist \
  cd build

  cmake \
  -DCMAKE_INSTALL_PREFIX=../dist \
  -DCMAKE_BUILD_TYPE=Release \
  ../hipify

  make -j install

To ensure LLVM is found, or in case of multiple LLVM instances, specify the path to the root folder containing the LLVM distribution:

.. code-block:: bash

  -DCMAKE_PREFIX_PATH=/usr/llvm/19.1.6/dist

On Windows, specify the following option for CMake:
``-G "Visual Studio 17 2022"``

Build the generated ``hipify-clang.sln`` using ``Visual Studio 17 2022`` instead of ``Make``. See :ref:`Windows testing` for the
supported tools for building.

As debug build type ``-DCMAKE_BUILD_TYPE=Debug`` is supported and tested, it is recommended to build ``LLVM+Clang``
in ``debug`` mode.

Also, 64-bit build mode (``-Thost=x64`` on Windows) is supported, hence it is recommended to build ``LLVM+Clang`` in
64-bit mode.

You can find the binary at ``./dist/hipify-clang`` or at the folder specified by the
``-DCMAKE_INSTALL_PREFIX`` option.

Testing hipify-clang
================================================

``hipify-clang`` is equipped with unit tests using LLVM
`lit <https://llvm.org/docs/CommandGuide/lit.html>`_ or `FileCheck <https://llvm.org/docs/CommandGuide/FileCheck.html>`_.

Build ``LLVM+Clang`` from sources, as prebuilt binaries are not exhaustive for testing. Before
building, ensure that the
`software required for building <https://releases.llvm.org/11.0.0/docs/GettingStarted.html#software>`_
belongs to an appropriate version.

LLVM >= 10.0.0
-----------------

1. Download `LLVM project <https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.6>`_ sources.

2. Build `LLVM project <http://llvm.org/docs/CMake.html>`_:

   .. code-block:: bash

    cd .. \
    mkdir build dist \
    cd build

   **Linux**:

   .. code-block:: bash

    cmake \
      -DCMAKE_INSTALL_PREFIX=../dist \
      -DLLVM_TARGETS_TO_BUILD="X86" \
      -DLLVM_ENABLE_PROJECTS="clang" \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      ../llvm-project/llvm
    make -j install

   **Windows**:

   .. code-block:: shell

    cmake \
      -G "Visual Studio 17 2022" \
      -A x64 \
      -Thost=x64 \
      -DCMAKE_INSTALL_PREFIX=../dist \
      -DLLVM_TARGETS_TO_BUILD="" \
      -DLLVM_ENABLE_PROJECTS="clang" \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      ../llvm-project/llvm

   Run ``Visual Studio 17 2022``, open the generated ``LLVM.sln``, build all, and build project ``INSTALL``.

3. Install `CUDA <https://developer.nvidia.com/cuda-toolkit-archive>`_ version 7.0 or
   greater.

   * In case of multiple CUDA installations, specify the particular version using ``DCUDA_TOOLKIT_ROOT_DIR`` option:

     **Linux**:

     .. code-block:: bash

      -DCUDA_TOOLKIT_ROOT_DIR=/usr/include

     **Windows**:

     .. code-block:: shell

      -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"

      -DCUDA_SDK_ROOT_DIR="C:/ProgramData/NVIDIA Corporation/CUDA Samples/v12.6"

4. [Optional] Install `cuTensor <https://developer.nvidia.com/cutensor-downloads>`_:

   * To specify the path to `cuTensor <https://developer.nvidia.com/cutensor-downloads>`_, use the ``CUDA_TENSOR_ROOT_DIR`` option:

     **Linux**:

     .. code-block:: bash

      -DCUDA_TENSOR_ROOT_DIR=/usr/include

     **Windows**:

     .. code-block:: shell

      -DCUDA_TENSOR_ROOT_DIR=D:/CUDA/cuTensor/2.0.2.1

5. [Optional] Install `cuDNN <https://developer.nvidia.com/rdp/cudnn-archive>`_ belonging to the version corresponding
   to the CUDA version:

   * To specify the path to `cuDNN <https://developer.nvidia.com/cudnn-downloads>`_, use the ``CUDA_DNN_ROOT_DIR`` option:

     **Linux**:

     .. code-block:: bash

      -DCUDA_DNN_ROOT_DIR=/usr/include

     **Windows**:

     .. code-block:: shell

      -DCUDA_DNN_ROOT_DIR=D:/CUDA/cuDNN/9.6.0

6. [Optional] Install `CUB 1.9.8 <https://github.com/NVIDIA/cub/releases/tag/1.9.8>`_ for ``CUDA < 11.0`` only;
   for ``CUDA >= 11.0``, the CUB shipped with CUDA will be used for testing.

   * To specify the path to CUB, use the ``CUDA_CUB_ROOT_DIR`` option (only for ``CUDA < 11.0``):

     **Linux**:

     .. code-block:: bash

      -DCUDA_CUB_ROOT_DIR=/srv/git/CUB

     **Windows**:

     .. code-block:: shell

      -DCUDA_CUB_ROOT_DIR=D:/CUDA/CUB

7. Install `Python <https://www.python.org/downloads>`_ version 3.0 or greater.

8. Install ``lit`` and ``FileCheck``; these are distributed with LLVM.

   * Install ``lit`` into ``Python``:

     **Linux**:

     .. code-block:: bash

      python /usr/llvm/19.1.6/llvm-project/llvm/utils/lit/setup.py install
      
     **Windows**:

     .. code-block:: shell

      python D:/LLVM/19.1.6/llvm-project/llvm/utils/lit/setup.py install

     In case of errors similar to ``ModuleNotFoundError: No module named 'setuptools'``, upgrade the ``setuptools`` package:

     .. code-block:: bash

      python -m pip install --upgrade pip setuptools
      
   * Starting with LLVM 6.0.1, specify the path to the ``llvm-lit`` Python script using the ``LLVM_EXTERNAL_LIT`` option:

     **Linux**:

     .. code-block:: bash

      -DLLVM_EXTERNAL_LIT=/usr/llvm/19.1.6/build/bin/llvm-lit

     **Windows**:

     .. code-block:: shell

      -DLLVM_EXTERNAL_LIT=D:/LLVM/19.1.6/build/Release/bin/llvm-lit.py

   * ``FileCheck``:

     **Linux**:

     Copy from ``/usr/llvm/19.1.6/build/bin/`` to ``CMAKE_INSTALL_PREFIX/dist/bin``.

     **Windows**:

     Copy from ``D:/LLVM/19.1.6/build/Release/bin`` to ``CMAKE_INSTALL_PREFIX/dist/bin``.

     Alternatively, specify the path to ``FileCheck`` in the ``CMAKE_INSTALL_PREFIX`` option.

9. To run OpenGL tests successfully on:

   **Linux**:

   Install GL headers.

   On Ubuntu, use: ``sudo apt-get install mesa-common-dev``

   **Windows**:

   No installation required. All the required headers are shipped with the Windows SDK.

10. Set the ``HIPIFY_CLANG_TESTS`` option to ``ON``: ``-DHIPIFY_CLANG_TESTS=ON``

11. Build and run tests.

LLVM <= 9.0.1
---------------------------------------------------------------------

1. Download `LLVM <https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/llvm-9.0.1.src.tar.xz>`_ \+ `Clang <https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/clang-9.0.1.src.tar.xz>`_ sources

2. Build `LLVM+Clang <http://releases.llvm.org/9.0.0/docs/CMake.html>`_:

   .. code-block:: bash

    cd .. \
    mkdir build dist \
    cd build

   **Linux**:

   .. code-block:: bash

    cmake \
      -DCMAKE_INSTALL_PREFIX=../dist \
      -DLLVM_SOURCE_DIR=../llvm \
      -DLLVM_TARGETS_TO_BUILD="X86" \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      ../llvm
    make -j install

   **Windows**:

   .. code-block:: shell

    cmake \
      -G "Visual Studio 16 2019" \
      -A x64 \
      -Thost=x64 \
      -DCMAKE_INSTALL_PREFIX=../dist \
      -DLLVM_SOURCE_DIR=../llvm \
      -DLLVM_TARGETS_TO_BUILD="" \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      ../llvm

3. Run ``Visual Studio 16 2019``, open the generated ``LLVM.sln``, build all, and build the ``INSTALL`` project.

Linux testing
======================================================

On Linux, the following configurations are tested:

* Ubuntu 22-23: LLVM 13.0.0 - 19.1.6, CUDA 7.0 - 12.6.3, cuDNN 8.0.5 - 9.6.0, cuTensor 1.0.1.0 - 2.0.2.1
* Ubuntu 20-21: LLVM 9.0.0 - 19.1.6, CUDA 7.0 - 12.6.3, cuDNN 5.1.10 - 9.6.0, cuTensor 1.0.1.0 - 2.0.2.1
* Ubuntu 16-19: LLVM 8.0.0 - 14.0.6, CUDA 7.0 - 10.2, cuDNN 5.1.10 - 8.0.5
* Ubuntu 14: LLVM 4.0.0 - 7.1.0, CUDA 7.0 - 9.0, cuDNN 5.0.5 - 7.6.5

Minimum build system requirements for the above configurations:

* CMake 3.16.8, GNU C/C++ 9.2, Python 3.0.

Recommended build system requirements:

* CMake 3.31.2, GNU C/C++ 13.2, Python 3.13.1.

Here's how to build ``hipify-clang`` with testing support on ``Ubuntu 23.10.01``:

.. code-block:: bash

  cmake
  -DHIPIFY_CLANG_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../dist \
  -DCMAKE_PREFIX_PATH=/usr/llvm/19.1.6/dist \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6.3 \
  -DCUDA_DNN_ROOT_DIR=/usr/local/cudnn-9.6.0 \
  -DCUDA_TENSOR_ROOT_DIR=/usr/local/cutensor-2.0.2.1 \
  -DLLVM_EXTERNAL_LIT=/usr/llvm/19.1.6/build/bin/llvm-lit \
  ../hipify

The corresponding successful output is:

.. code-block:: shell

  -- The C compiler identification is GNU 13.2.0
  -- The CXX compiler identification is GNU 13.2.0
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
  -- HIPIFY config:
  --    - Build hipify-clang    : ON
  --    - Test hipify-clang     : ON
  --    - Is part of HIP SDK    : OFF
  --    - Install clang headers : ON
  -- Found ZLIB: /usr/lib/x86_64-linux-gnu/libz.so (found version "1.2.13")
  -- Found LLVM 19.1.6:
  --    - CMake module path     : /usr/llvm/19.1.6/dist/lib/cmake/llvm
  --    - Clang include path    : /usr/llvm/19.1.6/dist/include
  --    - LLVM Include path     : /usr/llvm/19.1.6/dist/include
  --    - Binary path           : /usr/llvm/19.1.6/dist/bin
  -- Linker detection: GNU ld
  -- ---- The below configuring for hipify-clang testing only ----
  -- Found Python: /usr/bin/python3.13 (found suitable version "3.13.1", required range is "3.0...3.14") found components: Interpreter
  -- Found lit: /usr/local/bin/lit
  -- Found FileCheck: /GIT/LLVM/trunk/dist/FileCheck
  -- Initial CUDA to configure:
  --    - CUDA Toolkit path     : /usr/local/cuda-12.6.3
  --    - CUDA Samples path     :
  --    - cuDNN path            : /usr/local/cudnn-9.6.0
  --    - cuTENSOR path         : /usr/local/cuTensor/2.0.2.1
  --    - CUB path              :
  -- Found CUDAToolkit: /usr/local/cuda-12.6.3/targets/x86_64-linux/include (found version "12.6.85")
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
  -- Found Threads: TRUE
  -- Found CUDA config:
  --    - CUDA Toolkit path     : /usr/local/cuda-12.6.3
  --    - CUDA Samples path     : OFF
  --    - cuDNN path            : /usr/local/cudnn-9.6.0
  --    - CUB path              : /usr/local/cuda-12.6.3/include/cub
  --    - cuTENSOR path         : /usr/local/cuTensor/2.0.2.1
  -- Configuring done (0.6s)
  -- Generating done (0.0s)
  -- Build files have been written to: /usr/hipify/build

.. code-block:: shell

  make test-hipify

The corresponding successful output is:

.. code-block:: shell

  Running HIPify regression tests
  ===============================================================
  CUDA 12.6.85 - will be used for testing
  LLVM 19.1.6 - will be used for testing
  x86_64 - Platform architecture
  Linux 6.5.0-15-generic - Platform OS
  64 - hipify-clang binary bitness
  64 - python 3.13.1 binary bitness
  ===============================================================
  -- Testing: 106 tests, 12 threads --
  Testing Time: 6.91s

  Total Discovered Tests: 106
    Passed: 106 (100.00%)

.. _Windows testing:

Windows testing
=====================================================

Tested configurations:

.. list-table::
  :header-rows: 1

  * - LLVM
    - CUDA
    - cuDNN
    - Visual Studio
    - CMake
    - Python
  * - ``4.0.0 - 5.0.2``
    - ``7.0 - 8.0``
    - ``5.1.10 - 7.1.4``
    - ``2015.14.0, 2017.15.5.2``
    - ``3.5.1  - 3.18.0``
    - ``3.6.4 - 3.8.5``
  * - ``6.0.0 - 6.0.1``
    - ``7.0 - 9.0``
    - ``7.0.5  - 7.6.5``
    - ``2015.14.0, 2017.15.5.5``
    - ``3.6.0  - 3.18.0``
    - ``3.7.2 - 3.8.5``
  * - ``7.0.0 - 7.1.0``
    - ``7.0 - 9.2``
    - ``7.0.5  - 7.6.5``
    - ``2017.15.9.11``
    - ``3.13.3 - 3.18.0``
    - ``3.7.3 - 3.8.5``
  * - ``8.0.0 - 8.0.1``
    - ``7.0 - 10.0``
    - ``7.6.5``
    - ``2017.15.9.15``
    - ``3.14.2 - 3.18.0``
    - ``3.7.4 - 3.8.5``
  * - ``9.0.0 - 9.0.1``
    - ``7.0 - 10.1``
    - ``7.6.5``
    - ``2017.15.9.20, 2019.16.4.5``
    - ``3.16.4 - 3.18.0``
    - ``3.8.0 - 3.8.5``
  * - ``10.0.0 - 11.0.0``
    - ``7.0 - 11.1``
    - ``7.6.5  - 8.0.5``
    - ``2017.15.9.30, 2019.16.8.3``
    - ``3.19.2``
    - ``3.9.1``
  * - ``11.0.1 - 11.1.0``
    - ``7.0 - 11.2.2``
    - ``7.6.5  - 8.0.5``
    - ``2017.15.9.31, 2019.16.8.4``
    - ``3.19.3``
    - ``3.9.2``
  * - ``12.0.0 - 13.0.1``
    - ``7.0 - 11.5.1``
    - ``7.6.5  - 8.3.2``
    - ``2017.15.9.43, 2019.16.11.9``
    - ``3.22.2``
    - ``3.10.2``
  * - ``14.0.0 - 14.0.6``
    - ``7.0 - 11.7.1``
    - ``8.0.5  - 8.4.1``
    - ``2017.15.9.57,`` :sup:`5` ``2019.16.11.17, 2022.17.2.6``
    - ``3.24.0``
    - ``3.10.6``
  * - ``15.0.0 - 15.0.7``
    - ``7.0 - 11.8.0``
    - ``8.0.5  - 8.8.1``
    - ``2019.16.11.25, 2022.17.5.2``
    - ``3.26.0``
    - ``3.11.2``
  * - ``16.0.0 - 16.0.6``
    - ``7.0 - 12.2.2``
    - ``8.0.5  - 8.9.5``
    - ``2019.16.11.29, 2022.17.7.1``
    - ``3.27.3``
    - ``3.11.4``
  * - ``17.0.1`` :sup:`6` - ``18.1.8`` :sup:`7`
    - ``7.0 - 12.3.2``
    - ``8.0.5  - 9.6.0``
    - ``2019.16.11.42, 2022.17.12.3``
    - ``3.31.2``
    - ``3.13.1``
  * - ``19.1.0 - 19.1.6``
    - ``7.0 - 12.6.3``
    - ``8.0.5  - 9.6.0``
    - ``2019.16.11.42, 2022.17.12.3``
    - ``3.31.2``
    - ``3.13.1``

:sup:`5` LLVM 14.x.x is the latest major release supporting Visual Studio 2017.

To build LLVM 14.x.x correctly using Visual Studio 2017, add ``-DLLVM_FORCE_USE_OLD_TOOLCHAIN=ON``
to corresponding CMake command line.

You can also build LLVM \< 14.x.x correctly using Visual Studio 2017 without the
``LLVM_FORCE_USE_OLD_TOOLCHAIN`` option.

:sup:`6` Note that LLVM 17.0.0 was withdrawn due to an issue; use 17.0.1 or newer instead.

:sup:`7` Note that LLVM 18.0.0 has never been released; use 18.1.0 or newer instead.

Building with testing support using ``Visual Studio 17 2022`` on ``Windows 11``:

.. code-block:: shell

  cmake
  -G "Visual Studio 17 2022" \
  -A x64 \
  -Thost=x64 \
  -DHIPIFY_CLANG_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../dist \
  -DCMAKE_PREFIX_PATH=D:/LLVM/19.1.6/dist \
  -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" \
  -DCUDA_SDK_ROOT_DIR="C:/ProgramData/NVIDIA Corporation/CUDA Samples/v12.5" \
  -DCUDA_DNN_ROOT_DIR=D:/CUDA/cuDNN/9.6.0 \
  -DCUDA_TENSOR_ROOT_DIR=D:/CUDA/cuTensor/2.0.2.1 \
  -DLLVM_EXTERNAL_LIT=D:/LLVM/19.1.6/build/Release/bin/llvm-lit.py \
  ../hipify

The corresponding successful output is:

.. code-block:: shell

  -- Selecting Windows SDK version 10.0.22621.0 to target Windows 10.0.22631.
  -- The C compiler identification is MSVC 19.42.34435.0
  -- The CXX compiler identification is MSVC 19.42.34435.0
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Check for working C compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe - skipped
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Check for working CXX compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe - skipped
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- HIPIFY config:
  --    - Build hipify-clang    : ON
  --    - Test hipify-clang     : ON
  --    - Is part of HIP SDK    : OFF
  --    - Install clang headers : ON
  -- Found LLVM 19.1.6:
  --    - CMake module path     : D:/LLVM/19.1.6/dist/lib/cmake/llvm
  --    - Clang include path    : D:/LLVM/19.1.6/dist/include
  --    - LLVM Include path     : D:/LLVM/19.1.6/dist/include
  --    - Binary path           : D:/LLVM/19.1.6/dist/bin
  -- ---- The below configuring for hipify-clang testing only ----
  -- Found Python: C:/Users/TT/AppData/Local/Programs/Python/Python313/python.exe (found suitable version "3.13.1", required range is "3.0...3.14") found components: Interpreter
  -- Found lit: C:/Users/TT/AppData/Local/Programs/Python/Python313/Scripts/lit.exe
  -- Found FileCheck: D:/LLVM/19.1.6/dist/bin/FileCheck.exe
  -- Initial CUDA to configure:
  --    - CUDA Toolkit path     : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6
  --    - CUDA Samples path     : C:/ProgramData/NVIDIA Corporation/CUDA Samples/v12.5
  --    - cuDNN path            : D:/CUDA/cuDNN/9.6.0
  --    - cuTENSOR path         : D:/CUDA/cuTensor/2.0.2.1
  --    - CUB path              :
  -- Found CUDAToolkit: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include (found version "12.6.85")
  -- Found CUDA config:
  --    - CUDA Toolkit path     : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6
  --    - CUDA Samples path     : C:/ProgramData/NVIDIA Corporation/CUDA Samples/v12.5
  --    - cuDNN path            : D:/CUDA/cuDNN/9.6.0
  --    - cuTENSOR path         : D:/CUDA/cuTensor/2.0.2.1
  --    - CUB path              : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include/cub
  -- Configuring done (4.4s)
  -- Generating done (0.1s)
  -- Build files have been written to: D:/HIPIFY/build

Run ``Visual Studio 17 2022``, open the generated ``hipify-clang.sln``, and build the project ``test-hipify``.
