.. meta::
   :description: Tools to automatically translate CUDA source code into portable HIP C++
   :keywords: HIPIFY, ROCm, library, tool, CUDA, CUDA2HIP, hipify-clang, hipify-perl

.. _hipify_perl-command:

**************************************************************************
hipify-perl command
**************************************************************************

For a list of ``hipify-perl`` options, run: 

.. code-block:: cpp

  hipify-perl --help

Output:
=======

Usage
-----

.. code-block:: cpp

  hipify-perl [options] <source0> [... <sourceN>]

Options
-------

.. # COMMENT: The following lines define a break for use in the table below. 
.. |br| raw:: html 

    <br />

.. list-table::
    :widths: 2 5

    * - **Options**
      - **Description**

    * - ``-cuda-kernel-execution-syntax`` 
      - Keep CUDA kernel launch syntax (default)

    * - ``-examine``                      
      - Combines ``-no-output`` and ``-print-stats`` options

    * - ``-exclude-dirs=<string>``               
      - Exclude directories

    * - ``-exclude-files=<string>``              
      - Exclude files

    * - ``-experimental``                 
      - HIPIFY experimentally supported APIs

    * - ``-help``                         
      - Display available options

    * - ``-hip-kernel-execution-syntax``  
      - Transform CUDA kernel launch syntax to a regular HIP function call (overrides ``--cuda-kernel-execution-syntax``)

    * - ``-inplace``                      
      - Backs up the input file in ``.prehip`` file, and modifies the input file in-place

    * - ``-no-output``                    
      - Don't write any translated output to stdout

    * - ``-o=<string>``                          
      - Output filename

    * - ``-print-stats``                  
      - Print translation statistics as described in :ref:`hipify-stats`

    * - ``-quiet-warnings``                
      - Don't print warnings on unknown CUDA identifiers

    * - ``-roc``                          
      - Translate to ``roc`` libraries instead of ``hip`` libraries where possible

    * - ``-version``                      
      - The supported HIP version

    * - ``-whitelist=<string>``                  
      - Whitelist of identifiers

