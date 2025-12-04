.. meta::
   :description: Tools to automatically translate CUDA source code into portable HIP C++
   :keywords: HIPIFY, ROCm, library, tool, CUDA, CUDA2HIP, hipify-clang, hipify-perl, Linux

.. _build-hipify-clang-linux:

===================
Building hipify-clang on Linux
===================

Building LLVM
~~~~~~~~~~~~~~

.. code-block:: bash

  # Create a root directory for building LLVM, Clang and HIPIFY 
  export ROOT_DIR=$(pwd)

  # If you would like to clone LLVM with the full git history, remove the `--depth 1` option.
  git clone --depth 1 https://github.com/llvm/llvm-project.git

  mkdir build dist
  cd build

  cmake \
    -DCMAKE_INSTALL_PREFIX=../dist \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    ../llvm-project/llvm
  make -j install

.. note::
  If LLVM and Clang are built in ``Debug`` mode (with ``-DCMAKE_BUILD_TYPE=Debug``), please build ``HIPIFY`` in ``Debug`` mode as well.

  We support 64-bit build mode (``-Thost=x64``). Please build LLVM and Clang in 64-bit mode.

Building HIPIFY
~~~~~~~~~~~~~~~

.. code-block:: bash

  cd $ROOT_DIR

  git clone https://github.com/ROCm/HIPIFY.git

  cd build

  # To ensure LLVM is found, or in the case of multiple LLVM instances, 
  # specify the path to the root folder containing the LLVM distribution.
  cmake \
    -DCMAKE_INSTALL_PREFIX=../dist \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$ROOT_DIR/dist \
    ../hipify

  make -j install

You can find the binary at ``./dist/hipify-clang`` or at the folder specified by the ``-DCMAKE_INSTALL_PREFIX`` option.

Testing hipify-clang
~~~~~~~~~~~~~~~~~~~~

``hipify-clang`` is equipped with unit tests using LLVM
`lit <https://llvm.org/docs/CommandGuide/lit.html>`_ or `FileCheck <https://llvm.org/docs/CommandGuide/FileCheck.html>`_.

We recommend that you build ``LLVM+Clang`` from sources, as prebuilt binaries are not exhaustive for testing.
Before building, ensure that the `software required for building <https://llvm.org/docs/GettingStarted.html#software>`_ 
belongs to an appropriate version.

- Install `CUDA <https://developer.nvidia.com/cuda-toolkit-archive>`_ version 7.0 or greater.

  In case of multiple CUDA installations, specify the particular version using ``DCUDA_TOOLKIT_ROOT_DIR`` option:

  .. code-block:: bash

    -DCUDA_TOOLKIT_ROOT_DIR=/usr/include

- [Optional] Install `cuTensor <https://developer.nvidia.com/cutensor-downloads>`_:

  To specify the path to `cuTensor <https://developer.nvidia.com/cutensor-downloads>`_, use the ``CUDA_TENSOR_ROOT_DIR`` option:

  .. code-block:: bash

   -DCUDA_TENSOR_ROOT_DIR=/usr/include

- [Optional] Install `cuDNN <https://developer.nvidia.com/rdp/cudnn-archive>`_ belonging to the version corresponding to the CUDA version:

  To specify the path to `cuDNN <https://developer.nvidia.com/cudnn-downloads>`_, use the ``CUDA_DNN_ROOT_DIR`` option:

  .. code-block:: bash

   -DCUDA_DNN_ROOT_DIR=/usr/include

- [Optional] Install `CUB 1.9.8 <https://github.com/NVIDIA/cub/releases/tag/1.9.8>`_ for ``CUDA < 11.0`` only; for ``CUDA >= 11.0``, the CUB shipped with CUDA will be used for testing.

  To specify the path to CUB, use the ``CUDA_CUB_ROOT_DIR`` option (only for ``CUDA < 11.0``):

  .. code-block:: bash

   -DCUDA_CUB_ROOT_DIR=/srv/git/CUB

- Install `Python <https://www.python.org/downloads>`_ version 3.0 or greater.

- Install ``lit`` and ``FileCheck``; these are distributed with LLVM.

  ``lit``:

  .. code-block:: bash

   python $(ROOT_DIR)/llvm-project/llvm/utils/lit/setup.py install

  Starting with LLVM 6.0.1, specify the path to the ``llvm-lit`` Python script using the ``LLVM_EXTERNAL_LIT`` option:

  .. code-block:: bash

   -DLLVM_EXTERNAL_LIT=$ROOT_DIR/build/bin/llvm-lit

  ``FileCheck``:

  Copy from ``$ROOT_DIR/build/bin/`` to ``CMAKE_INSTALL_PREFIX/dist/bin``.

- To run OpenGL tests successfully, you need to install OpenGL headers and libraries.

  On Ubuntu, use: ``sudo apt-get install mesa-common-dev``

- Set the ``HIPIFY_CLANG_TESTS`` option to ``ON``: ``-DHIPIFY_CLANG_TESTS=ON``.

- Build and run tests.

Linux testing
=============

On Linux, the following configurations are tested:

* Ubuntu 22-24: LLVM 13.0.0 - 21.1.7, CUDA 7.0 - 12.9.1, cuDNN 8.0.5 - 9.15.0, cuTensor 1.0.1.0 - 2.4.0.0
* Ubuntu 20-21: LLVM 9.0.0 - 20.1.8, CUDA 7.0 - 12.8.1, cuDNN 5.1.10 - 9.15.0, cuTensor 1.0.1.0 - 2.4.0.0
* Ubuntu 16-19: LLVM 8.0.0 - 14.0.6, CUDA 7.0 - 10.2, cuDNN 5.1.10 - 8.0.5
* Ubuntu 14: LLVM 4.0.0 - 7.1.0, CUDA 7.0 - 9.0, cuDNN 5.0.5 - 7.6.5

Minimum build system requirements for the above configurations:

* CMake 3.16.8, GNU C/C++ 9.2, Python 3.0.

Recommended build system requirements:

* CMake 4.1.1, GNU C/C++ 13.3, Python 3.14.0.

Here's how to build ``hipify-clang`` with testing support on ``Ubuntu 24.04.02``:

.. code-block:: bash

  cd $ROOT_DIR/build

  cmake \
    -DHIPIFY_CLANG_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../dist \
    -DCMAKE_PREFIX_PATH=$ROOT_DIR/dist \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.9.1 \
    -DCUDA_DNN_ROOT_DIR=/usr/local/cudnn-9.15.0 \
    -DCUDA_TENSOR_ROOT_DIR=/usr/local/cutensor-2.4.0.0 \
    -DLLVM_EXTERNAL_LIT=$ROOT_DIR/build/bin/llvm-lit \
    ../hipify

The corresponding successful output is (assuming ROOT_DIR is ``/usr/llvm/21.1.7``):

.. code-block:: shell

  -- The C compiler identification is GNU 13.3.0
  -- The CXX compiler identification is GNU 13.3.0
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
  -- Found ZLIB: /usr/lib/x86_64-linux-gnu/libz.so (found version "1.3")
  -- Found LLVM 21.1.7:
  --    - CMake module path     : /usr/llvm/21.1.7/dist/lib/cmake/llvm
  --    - Clang include path    : /usr/llvm/21.1.7/dist/include
  --    - LLVM Include path     : /usr/llvm/21.1.7/dist/include
  --    - Binary path           : /usr/llvm/21.1.7/dist/bin
  -- Linker detection: GNU ld
  -- ---- The below configuring for hipify-clang testing only ----
  -- Found Python: /usr/bin/python3.14 (found suitable version "3.14.0", required range is "3.0...3.15") found components: Interpreter
  -- Found lit: /usr/local/bin/lit
  -- Found FileCheck: /GIT/LLVM/trunk/dist/FileCheck
  -- Initial CUDA to configure:
  --    - CUDA Toolkit path     : /usr/local/cuda-12.9.1
  --    - CUDA Samples path     :
  --    - cuDNN path            : /usr/local/cudnn-9.15.0
  --    - cuTENSOR path         : /usr/local/cuTensor/2.4.0.0
  --    - CUB path              :
  -- Found CUDAToolkit: /usr/local/cuda-12.9.1/targets/x86_64-linux/include (found version "12.9.86")
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
  -- Found Threads: TRUE
  -- Found CUDA config:
  --    - CUDA Toolkit path     : /usr/local/cuda-12.9.1
  --    - CUDA Samples path     : OFF
  --    - cuDNN path            : /usr/local/cudnn-9.15.0/include
  --    - cuTENSOR path         : /usr/local/cuTensor/2.4.0.0/include
  --    - CUB path              : /usr/local/cuda-12.9.1/include/cub
  -- Configuring done (0.6s)
  -- Generating done (0.0s)
  -- Build files have been written to: /usr/hipify/build

.. code-block:: shell

  make test-hipify

The corresponding successful output is:

.. code-block:: shell

  Running HIPify regression tests
  ===============================================================
  CUDA 12.9.86 - will be used for testing
  LLVM 21.1.7 - will be used for testing
  x86_64 - Platform architecture
  Linux 6.5.0-15-generic - Platform OS
  64 - hipify-clang binary bitness
  64 - python 3.14.0 binary bitness
  ===============================================================
  -- Testing: 106 tests, 12 threads --
  Testing Time: 6.91s

  Total Discovered Tests: 106
    Passed: 106 (100.00%)
