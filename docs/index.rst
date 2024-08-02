.. meta::
   :description: Tools to automatically translate CUDA source code into portable HIP C++
   :keywords: HIPIFY, ROCm, library, tool, CUDA, CUDA2HIP, hipify-clang, hipify-perl

.. _index:

=====================
HIPIFY documentation
=====================

``hipify-clang`` and ``hipify-perl`` are tools that automatically translate NVIDIA CUDA source code into portable HIP C++.

.. note::
    
    `hipify_torch <https://github.com/ROCmSoftwarePlatform/hipify_torch>`_ is a related tool that also translates CUDA source code into portable HIP C++. It was initially developed as part of the PyTorch project to cater to the project's unique requirements but was found to be useful for PyTorch-related projects and thus was released as an independent utility.

You can access HIPIFY code on our `GitHub repository <https://github.com/ROCm/HIPIFY>`_.

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :ref:`hipify-clang`
    * :ref:`hipify-perl`
    
  .. grid-item-card:: API reference

    * :doc:`Supported APIs <supported_apis>`
     
To contribute to the documentation, refer to
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
