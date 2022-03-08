# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.16.8)

set(HIPIFY_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(HIPIFY_WRAPPER_DIR ${HIPIFY_BUILD_DIR}/wrapper_dir)
set(HIPIFY_WRAPPER_INC_DIR ${HIPIFY_WRAPPER_DIR}/include)
set(HIPIFY_WRAPPER_BIN_DIR ${HIPIFY_WRAPPER_DIR}/bin)

#Function to generate header template file
function(create_header_template)
    file(WRITE ${HIPIFY_WRAPPER_DIR}/header.hpp.in "/*
    Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the \"Software\"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   */\n\n#ifndef @include_guard@\n#define @include_guard@ \n\n#pragma message(\"This file is deprecated. Use file from include path /opt/rocm-ver/bin/include\")\n@include_statements@ \n\n#endif")
endfunction()

#use header template file and generate wrapper header files
function(generate_wrapper_header)
  file(MAKE_DIRECTORY ${HIPIFY_WRAPPER_INC_DIR}/cuda_wrappers)
  file(MAKE_DIRECTORY ${HIPIFY_WRAPPER_INC_DIR}/fuzzer)
  file(MAKE_DIRECTORY ${HIPIFY_WRAPPER_INC_DIR}/xray)
  file(MAKE_DIRECTORY ${HIPIFY_WRAPPER_INC_DIR}/sanitizer)
  #Empty profile folder is packaged, so doing the same here as well: TBD
  file(MAKE_DIRECTORY ${HIPIFY_WRAPPER_INC_DIR}/profile)

  #find all header files from CL folder
  file(GLOB_RECURSE include_files ${LLVM_DIR}/../../clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/include/*)
  #Generate wrapper header files
  foreach(header_file ${include_files})
     #set include  guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "${include_guard}ROCPROF_WRAPPER_INCLUDE_${INC_GAURD_NAME}_H")
     #set include statement
    get_filename_component(file_name ${header_file} NAME)
    get_filename_component ( header_subdir ${header_file} DIRECTORY )
    get_filename_component ( subdir_name ${header_subdir} NAME)
    #Create wrapper header files for all *.h files
    if("${subdir_name}" STREQUAL "include")
      #module.modulemap is not a header file , so using symlink
      if(${file_name} STREQUAL "module.modulemap")
        add_custom_target(link_${file_name} ALL
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ../../../bin/include/${file_name} ${HIPIFY_WRAPPER_INC_DIR}/${file_name})
      else()
        set(include_statements "${include_statements}#include \"../../../bin/include/${file_name}\"\n")
        configure_file(${HIPIFY_WRAPPER_DIR}/header.hpp.in ${HIPIFY_WRAPPER_INC_DIR}/${file_name})
      endif()#end of modulemap check

    # These subdirectory contains header files
    elseif(("${subdir_name}" STREQUAL "xray") OR
            ("${subdir_name}" STREQUAL "sanitizer") OR
            ("${subdir_name}" STREQUAL "fuzzer") )
      set(include_statements "${include_statements}#include \"../../../../bin/include/${subdir_name}/${file_name}\"\n")
      configure_file(${HIPIFY_WRAPPER_DIR}/header.hpp.in ${HIPIFY_WRAPPER_INC_DIR}/${subdir_name}/${file_name})

    #cuda wrapper contains files , but not header files, so creating soft link
    elseif("${subdir_name}" STREQUAL "cuda_wrappers")
      add_custom_target(link_${file_name} ALL
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ../../../../bin/include/${subdir_name}/${file_name} ${HIPIFY_WRAPPER_INC_DIR}/${subdir_name}/${file_name})

    endif()#end of subdir name check
    unset(include_guard)
    unset(include_statements)
  endforeach()

endfunction()

#function to create symlink to binaries
function(create_binary_symlink)
  file(MAKE_DIRECTORY ${HIPIFY_WRAPPER_BIN_DIR})
  file(GLOB binary_files ${CMAKE_SOURCE_DIR}/bin/*)
  foreach(binary_file ${binary_files})
    get_filename_component(file_name ${binary_file} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../bin/${file_name} ${HIPIFY_WRAPPER_BIN_DIR}/${file_name})
  endforeach()
  #symlink for hipify-clang
  set(file_name "hipify-clang")
  add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../bin/${file_name} ${HIPIFY_WRAPPER_BIN_DIR}/${file_name})
endfunction()
#Creater a template for header file
create_header_template()
#Use template header file and generater wrapper header files
generate_wrapper_header()
install(DIRECTORY ${HIPIFY_WRAPPER_INC_DIR} DESTINATION hip/bin)
# Create symlink to binaries
create_binary_symlink()
install(DIRECTORY ${HIPIFY_WRAPPER_BIN_DIR}  DESTINATION hip)
