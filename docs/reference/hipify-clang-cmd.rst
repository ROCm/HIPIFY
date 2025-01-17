.. meta::
   :description: Tools to automatically translate CUDA source code into portable HIP C++
   :keywords: HIPIFY, ROCm, library, tool, CUDA, CUDA2HIP, hipify-clang, hipify-perl

.. _hipify_clang-command:

**************************************************************************
hipify-clang command
**************************************************************************

For a list of ``hipify-clang`` options, run: 

.. code-block:: cpp

  hipify-clang --help

Output:
=======

Usage
-----

.. code-block:: cpp

  hipify-clang [options] <source0> [... <sourceN>]

Options
-------

.. # COMMENT: The following lines define a break for use in the table below. 
.. |br| raw:: html 

    <br />

.. list-table::
    :widths: 2 5

    * - **Options**
      - **Description**

    * - ``--``
      - Separator between ``hipify-clang`` and ``clang`` options. Don't specify if there are no ``clang`` options. Not all ``clang`` options are supported by ``hipify-clang``

    * - ``-D <macro>=<value>``                              
      - Define ``<macro>`` to ``<value>`` or 1 if ``<value>`` is omitted

    * - ``-I <directory>``                                  
      - Add directory to include search path

    * - ``--amap``                                          
      - Try to hipify as much as possible; ignores ``default-preprocessor``

    * - ``--clang-resource-directory=<directory>``          
      - Defines the path to the parent folder for the ``include`` folder, containing ``__clang_cuda_runtime_wrapper.h`` and other header files used on runtime

    * - ``--csv``                                           
      - Generate documentation in CSV format

    * - ``--cuda-gpu-arch=<value>``                         
      - CUDA GPU architecture (e.g. sm_35); may be specified more than once

    * - ``--cuda-kernel-execution-syntax``                  
      - Keep CUDA kernel launch syntax (default)

    * - ``--cuda-path=<directory>``                         
      - CUDA installation path. The CUDA path is required for ``hipify-clang``

    * - ``--default-preprocessor``                          
      - Enable default preprocessor behavior (synonymous with ``--skip-excluded-preprocessor-conditional-blocks``)

    * - ``--doc-format=<value>``                            
      - Documentation format: ``full`` (default), ``strict``, or ``compact``. Either the ``--md`` or ``--csv`` option must also be specified to generate the documentation.

    * - ``--doc-roc=<value>``                               
      - ROC documentation generation: ``skip`` (default), ``separate``, or ``joint``. Either the ``--md`` or ``--csv`` option must also be specified to generate the documentation.

    * - ``--examine``                                       
      - Combine the ``--no-output`` and ``--print-stats`` options

    * - ``--experimental``                                  
      - Hipify HIP APIs that are experimentally supported, otherwise, the corresponding warnings will be emitted

    * - ``--extra-arg=<string>``                            
      - Additional argument to append to the compiler command line

    * - ``--extra-arg-before=<string>``                     
      - Additional argument to prepend to the compiler command line

    * - ``--help``                                          
      - Display available options (Use ``--help-hidden`` to include hidden options)

    * - ``--help-list``                                     
      - Display list of available options (Use ``--help-list-hidden`` to include hidden options)

    * - ``--hip-kernel-execution-syntax``                   
      - Transform CUDA kernel launch syntax to a regular HIP function call (overrides ``--cuda-kernel-execution-syntax``)

    * - ``--inplace``                                       
      - Modify input file in-place. This will overwrite the input file with the hipify output

    * - ``--md``                                            
      - Generate documentation in Markdown format

    * - ``--miopen``                                        
      - Translate to ``miopen`` libraries instead of ``hip`` libraries where it is possible. Cannot be used with ``--roc``

    * - ``--no-backup``                                     
      - Don't create a backup file for the hipified source

    * - ``--no-output``                                     
      - Don't write any translated output to stdout

    * - ``--no-undocumented-features``                      
      - Don't rely on undocumented features in code transformation

    * - ``--no-warnings-on-undocumented-features``          
      - Suppress warnings on undocumented features in code transformation

    * - ``-o <filename>``                                   
      - Output filename

    * - ``--o-dir=<directory>``                             
      - Output directory

    * - ``--o-hipify-perl-dir=<directory>``                 
      - Output directory for hipify-perl script

    * - ``--o-python-map-dir=<directory>``                  
      - Output directory for Python map

    * - ``--o-stats=<filename>``                            
      - Output filename for statistics

    * - ``-p <build-path>`` 
      - Used to read a compile command database as described in :ref:`hipify-json`. For example, it can be a CMake build directory in which a file named ``compile_commands.json`` exists (use ``-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`` CMake option to get this output). When no build path is specified, a search for ``compile_commands.json`` will be attempted through all parent paths of the first input file . See: https://clang.llvm.org/docs/HowToSetupToolingForLLVM.html for an example of setting up Clang Tooling on a source tree

    * - ``--perl``                                          
      - Generate ``hipify-perl`` script. See :ref:`build-hipify-perl` for more information. 

    * - ``--print-stats``                                   
      - Print translation statistics. See :ref:`hipify-stats` for more information

    * - ``--print-stats-csv``                               
      - Print translation statistics in a CSV file. See :ref:`hipify-stats` for more information

    * - ``--python``                                        
      - Generate ``hipify-python`` command

    * - ``--roc``                                           
      - Translate to ``roc`` libraries instead of ``hip`` libraries where possible. Cannot be used with ``--miopen``

    * - ``--save-temps``                                    
      - Save temporary files

    * - ``--skip-excluded-preprocessor-\`` |br| ``conditional-blocks`` 
      - Enable default preprocessor behavior by skipping undefined conditional blocks. This has the same effect as ``--default-preprocessor``

    * - ``--temp-dir=<directory>``                          
      - Temporary directory

    * - ``--use-hip-data-types``                            
      - Use ``hipDataType`` instead of ``hipblasDatatype_t`` or ``rocblas_datatype``

    * - ``-v``                                              
      - Show commands to run and use verbose output

    * - ``--version``                                       
      - Display the version of this program

    * - ``--versions``                                      
      - Display the versions of the supported 3rd-party software

    * - ``<source0> ...`` 
      - Specify the file paths and names of one or more source files. These paths are looked up in the compile command database. If the path of a file is absolute, it needs to point into CMake's source tree. If the path is relative, the current working directory needs to be in the CMake source tree and the file must be in a subdirectory of the current working directory. ``./`` prefixes in the relative files will be automatically removed, but the rest of a relative path must be a suffix of a path in the compile command database

Option uses:
------------

1.	Common Options:

  * ``--help``: Displays the help message
  * ``-o <file>``: Specifies the output file for the converted source
  * ``-I <dir>``: Adds the specified directory to the include search paths
  * ``--cuda-path=<path>``: Specifies the path to the CUDA installation. Required
  * ``--hip-path=<path>``: Specifies the path to the HIP installation (optional; defaults to the ROCm installation path)

2.	Preprocessor and Compilation Options:

  * ``-D<macro>``: Defines macros for the preprocessor
  * ``-U<macro>``: Undefines macros
  * ``--save-temps``: Keeps intermediate files generated during processing

3.	Diagnostics and Debugging:

  * ``-v``: Enables verbose output to provide detailed diagnostic information
  * ``--version``: Displays the version of HIPIFY-Clang
  * ``--show-progress``: Displays progress during the translation process
  * ``--print-stats`` | ``--print-stats-csv``: Prints statistics about the translation process (e.g., the number of functions or API calls converted) into either text or CSV form

4.	Include and Exclude Rules:

  * ``--exclude-path=<path>``: Specifies paths to exclude from translation
  * ``--include-path=<path>``: Specifies paths to explicitly include during translation
