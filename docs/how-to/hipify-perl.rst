.. meta::
   :description: Tools to automatically translate CUDA source code into portable HIP C++
   :keywords: HIPIFY, ROCm, library, tool, CUDA, CUDA2HIP, hipify-clang, hipify-perl

.. _hipify-perl:

===================
Using hipify-perl
===================

``hipify-perl`` is perl-based script that heavily uses regular expressions, that is automatically generated from ``hipify-clang``.

**Advantages:**

- Ease of use
- No checks for input source NVIDIA CUDA code for correctness required
- No dependency on third party tools, including CUDA

**Disadvantages:**

- Inability or difficulty in implementing the following constructs:

  - Macros expansion
  - Namespaces:

    - Redefinition of CUDA entities in user namespaces
    - Using directive

  - Templates (some cases)
  - Device or host function calls differentiation
  - Correct injection of header files
  - Parsing complicated argument lists

Example
=======

For additional details on the following ``hipify-perl`` command options, see :ref:`hipify_perl-command`. For more advanced translation needs use ``hipify-clang`` as it is more comprehensive and accurate. 

Convert a simple CUDA file (``square.cu``) to HIP using ``hipify-perl``:

.. code-block:: shell

    hipify-perl square.cu -o square.cu.hip

This command translates the input file and writes the result to ``square.cu.hip``.
