.. meta::
   :description: Tools to automatically translate CUDA source code into portable HIP C++
   :keywords: HIPIFY, ROCm, library, tool, CUDA, CUDA2HIP, hipify-clang, hipify-perl

.. _build-hipify-perl:

===================
Building hipify-perl
===================

``hipify-perl`` is a perl-based script that heavily uses regular expressions, which is automatically generated from ``hipify-clang``. To generate ``hipify-perl``, run: 

.. code-block:: shell
    
    hipify-clang --perl
    
You can choose to specify the output directory for the generated ``hipify-perl`` file using ``--o-hipify-perl-dir`` option.
