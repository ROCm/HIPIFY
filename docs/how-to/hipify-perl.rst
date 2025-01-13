.. meta::
   :description: Tools to automatically translate CUDA source code into portable HIP C++
   :keywords: HIPIFY, ROCm, library, tool, CUDA, CUDA2HIP, hipify-clang, hipify-perl

.. _hipify-perl:

===================
Using hipify-perl
===================

``hipify-perl`` is perl-based script that heavily uses regular expressions, which is automatically generated from ``hipify-clang``.

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

Convert a simple CUDA file (``square.cu``) to HIP using ``hipify-perl``:

.. code-block:: shell

    perl hipify-perl square.cu > square.cu.hip

This command translates the input file and writes the result to ``square.cu.hip``.

.. note:: 
    For advanced translation needs use ``hipify-clang`` as it is more comprehensive and accurate. 


hipify-perl command
===================

Run the Perl script with the ``--help`` flag to display its usage and options:

.. code-block:: cpp

  hipify-perl --help


Output Summary:
---------------

1.	Usage:

.. code-block:: cpp

    hipify-perl [options] <input_file> [output_file]

  * <input_file>: The CUDA source file to be translated
  * [output_file]: Optional. The name of the output file. If not specified, the modified code is printed to the console

2.	Options:

  * ``--help``: Displays the help message.
  * ``--quiet``: Suppresses non-critical messages and warnings during the translation process.
  * ``--print-stats``: Prints a summary of the translation statistics, such as the number of API calls replaced.
  * ``--cuda-path=<path>``: Specifies the path to the CUDA installation (used to resolve header paths, if needed).
  * ``--hip-path=<path>``: Specifies the HIP installation path (optional; defaults to the ROCm installation path).

3.	Translation Scope:

  * The Perl script focuses on basic translation tasks: 

    - Replacing CUDA runtime API calls (e.g., cudaMalloc → hipMalloc).
    - Adjusting kernel launch syntax (e.g., <<<blocks, threads>>>).
    - Replacing CUDA-specific constants and macros with HIP equivalents.

  * It does not support advanced syntax transformations, such as handling device-specific intrinsics or complex template-based CUDA code.

4.	Diagnostics:

  * The script typically outputs a summary of what was replaced, unless the ``--quiet`` option is used.

5.	Limitations:

  * The help text often notes that the script does not handle complex code structures or full compliance with the HIP API.
