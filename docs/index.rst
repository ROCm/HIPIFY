.. meta::
   :description: Tools to automatically translate CUDA source code into portable HIP C++
   :keywords: HIPIFY, ROCm, library, tool, CUDA, CUDA2HIP, hipify-clang, hipify-perl

.. _index:

=====================
HIPIFY documentation
=====================

HIPIFY is a ROCm tool to help developers migrate GPU programming from NVIDIA's CUDA language to AMD's HIP C++ programming language for use on AMD GPUs. HIPIFY includes two tools offering different levels of capability: 

•	``hipify-clang``: A clang-based tool that parses CUDA code and converts it to HIP code. It handles syntax changes, API calls, and kernel launch differences.
•	``hipify-perl``: A simpler tool generated from ``hipify-clang`` that replaces CUDA API calls with HIP equivalents for basic code translation needs. ``hipify-perl`` is useful for simple CUDA programs, but offers less error detection when running into issues during translation. 

.. note::
    
    `hipify_torch <https://github.com/ROCm/hipify_torch>`_ is a related tool that also translates CUDA source code into portable HIP C++. It was developed as part of the PyTorch project to cater to the project's unique requirements, was found to be useful for PyTorch-related projects, and released as an independent utility.

HIPIFY does not automatically convert all CUDA code into HIP code seamlessly. While it is a powerful tool for translating CUDA code to HIP, there are some limitations and areas where manual intervention is often required. HIPIFY can automatically convert many CUDA  runtime API calls, kernel launch syntax, standard CUDA library functions where there is a HIP library equivalent, specific keywords like ``__global__`` and ``__device__``. However, HIP is not a complete replacement for CUDA, and HIPIFY cannot automatically translate all code. CUDA libraries, or third-party libraries that have no HIP equivalent cannot be translated. In addition, code which is optimized for performance on NVIDIA GPUs might require additional rework to optimize performance on AMD GPUs. 

After migrating code through HIPIFY, you should perform a code review to ensure functional correctness, replace any unsupported libraries or constructs with HIP or ROCm features. Debug and test the new HIP program, and optimize the performance on the target AMD GPUs. 

HIPIFY is open-source and freely available as part of the ROCm ecosystem. You can find the HIPIFY code on AMD's `GitHub HIPIFY repository <https://github.com/ROCm/HIPIFY>`_.

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Building

    * :ref:`build-hipify-clang`
    * :ref:`build-hipify-perl`
    
  .. grid-item-card:: How to

    * :doc:`Use hipify-clang <./how-to/hipify-clang>`
    * :doc:`Use hipify-perl <./how-to/hipify-perl>`
    
  .. grid-item-card:: API reference

    * :ref:`hipify_clang-command`
    * :ref:`hipify_perl-command`
    * :doc:`Supported APIs <./reference/supported_apis>`
     
To contribute to the documentation, refer to
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
