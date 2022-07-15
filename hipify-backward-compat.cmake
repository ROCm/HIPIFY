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
set(HIPIFY_WRAPPER_BIN_DIR ${HIPIFY_WRAPPER_DIR}/bin)

#With File Reorganization , hipify(and hip)  will be installed in /opt/rocm-ver
#instead of  /opt/rocm-ver/hip/. For maintaining backward  compatibility
# the previous location(/opt/rocm-ver/hip/) will have soft link.
#This file is for creating  soft link to binary files and install it in the  previous location
#Note: soft link added for binary files.

#function to create symlink to binaries
function(create_binary_symlink)
  file(MAKE_DIRECTORY ${HIPIFY_WRAPPER_BIN_DIR})
  #create softlink for public scripts
  file(GLOB binary_files ${CMAKE_SOURCE_DIR}/bin/hip*)
  foreach(binary_file ${binary_files})
    get_filename_component(file_name ${binary_file} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_BINDIR}/${file_name} ${HIPIFY_WRAPPER_BIN_DIR}/${file_name})
  endforeach()
  #create softlink for private scripts
  file(GLOB binary_files ${CMAKE_SOURCE_DIR}/bin/find*)
  foreach(binary_file ${binary_files})
    get_filename_component(file_name ${binary_file} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBEXECDIR}/hipify/${file_name} ${HIPIFY_WRAPPER_BIN_DIR}/${file_name})
  endforeach()

  #symlink for hipify-clang
  set(file_name "hipify-clang")
  add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_BINDIR}/${file_name} ${HIPIFY_WRAPPER_BIN_DIR}/${file_name})
endfunction()
# Create symlink to binaries
create_binary_symlink()
#install symlink in hip folder
install(DIRECTORY ${HIPIFY_WRAPPER_BIN_DIR}  DESTINATION hip)
