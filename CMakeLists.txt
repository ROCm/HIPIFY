cmake_minimum_required(VERSION 3.16.8)

project(hipify-clang)

include(GNUInstallDirs)

option(HIPIFY_CLANG_TESTS "Build HIPIFY tests, if lit is installed" OFF)
option(HIPIFY_CLANG_TESTS_ONLY "Build HIPIFY tests only, if lit is installed and hipify-clang binary is already produced" OFF)
option(HIPIFY_INCLUDE_IN_HIP_SDK "Include HIPIFY in HIP SDK" OFF)

if(HIPIFY_CLANG_TESTS OR HIPIFY_CLANG_TESTS_ONLY)
  set(HIPIFY_TEST "ON")
else()
  set(HIPIFY_TEST "OFF")
endif()

if(HIPIFY_CLANG_TESTS_ONLY)
  set(HIPIFY_BUILD "OFF")
else()
  set(HIPIFY_BUILD "ON")
endif()

message(STATUS "HIPIFY config:")
message(STATUS "   - Build hipify-clang : ${HIPIFY_BUILD}")
message(STATUS "   - Test  hipify-clang : ${HIPIFY_TEST}")
message(STATUS "   - Is part of HIP SDK : ${HIPIFY_INCLUDE_IN_HIP_SDK}")

if(HIPIFY_INCLUDE_IN_HIP_SDK)
  if(NOT WIN32)
    message(FATAL_ERROR "HIPIFY_INCLUDE_IN_HIP_SDK is only supported on Windows")
  elseif(CMAKE_GENERATOR MATCHES "Visual Studio")
    message(FATAL_ERROR "HIPIFY_INCLUDE_IN_HIP_SDK is not targeting Visual Studio")
  endif()
else()
  find_package(Clang REQUIRED CONFIG PATHS ${CMAKE_PREFIX_PATH})
  find_package(LLVM REQUIRED CONFIG PATHS ${CMAKE_PREFIX_PATH})

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}:")
  message(STATUS "   - CMake module path  : ${LLVM_CMAKE_DIR}")
  message(STATUS "   - Clang include path : ${CLANG_INCLUDE_DIRS}")
  message(STATUS "   - LLVM Include path  : ${LLVM_INCLUDE_DIRS}")
  message(STATUS "   - Binary path        : ${LLVM_TOOLS_BINARY_DIR}")
endif()

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

if (NOT HIPIFY_CLANG_TESTS_ONLY)
  if(MSVC AND MSVC_VERSION VERSION_LESS "1900")
    message(SEND_ERROR "hipify-clang could be built by Visual Studio 14 2015 or higher.")
    return()
  endif()

  include_directories(${CLANG_INCLUDE_DIRS})
  include_directories(${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})

  file(GLOB_RECURSE HIPIFY_SOURCES src/*.cpp)
  file(GLOB_RECURSE HIPIFY_HEADERS src/*.h)

  add_llvm_executable(hipify-clang ${HIPIFY_SOURCES} ${HIPIFY_HEADERS})
  target_link_directories(hipify-clang PRIVATE ${LLVM_LIBRARY_DIRS})

  if(HIPIFY_INCLUDE_IN_HIP_SDK)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
      message(FATAL_ERROR "In order to include HIPIFY in HIP SDK, HIPIFY needs to be built with LLVM_EXTERNAL_PROJECTS")
    endif()

    # Need to add clang include directories explicitly if building as part of llvm.
    if(LLVM_EXTERNAL_CLANG_SOURCE_DIR)
      target_include_directories(hipify-clang
        PRIVATE
          ${LLVM_BINARY_DIR}/tools/clang/include
          ${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/include)
    endif()

    # Need to add lld include directories explicitly if building as part of llvm.
    if(LLVM_EXTERNAL_LLD_SOURCE_DIR)
      target_include_directories(hipify-clang
        PRIVATE
          ${LLVM_BINARY_DIR}/tools/lld/include
          ${LLVM_EXTERNAL_LLD_SOURCE_DIR}/include)
    endif()
  else()
    set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
    set(CMAKE_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)
  endif()

  # Link against LLVM and CLANG libraries.
  target_link_libraries(hipify-clang PRIVATE
    clangASTMatchers
    clangFrontend
    clangTooling
    clangParse
    clangSerialization
    clangSema
    clangEdit
    clangFormat
    clangLex
    clangAnalysis
    clangDriver
    clangAST
    clangToolingCore
    clangRewrite
    clangBasic
    LLVMProfileData
    LLVMSupport
    LLVMMCParser
    LLVMMC
    LLVMBitReader
    LLVMOption
    LLVMCore)

  if(LLVM_PACKAGE_VERSION VERSION_GREATER "6.0.1")
    target_link_libraries(hipify-clang PRIVATE clangToolingInclusions)
  endif()

  if(LLVM_PACKAGE_VERSION VERSION_GREATER "9.0.1")
    target_link_libraries(hipify-clang PRIVATE LLVMFrontendOpenMP)
  endif()

  if(LLVM_PACKAGE_VERSION VERSION_EQUAL "15.0.0" OR LLVM_PACKAGE_VERSION VERSION_GREATER "15.0.0")
    target_link_libraries(hipify-clang PRIVATE LLVMWindowsDriver clangSupport)
  endif()

  if(LLVM_PACKAGE_VERSION VERSION_EQUAL "16.0.0" OR LLVM_PACKAGE_VERSION VERSION_GREATER "16.0.0")
    if(MSVC)
      set(STD "/std:c++17")
    else()
      set(STD "-std=c++17")
    endif()
  else()
    if(MSVC)
      set(STD "/std:c++14")
    else()
      set(STD "-std=c++14")
    endif()
  endif()

  if(MSVC)
    target_link_libraries(hipify-clang PRIVATE version)
    target_compile_options(hipify-clang PRIVATE ${STD} /Od /GR- /EHs- /EHc- /MP)
    if((LLVM_PACKAGE_VERSION VERSION_EQUAL "18.0.0" OR LLVM_PACKAGE_VERSION VERSION_GREATER "18.0.0") AND MSVC_VERSION VERSION_GREATER "1919")
      target_compile_options(hipify-clang PRIVATE /Zc:preprocessor)
    endif()
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} /SUBSYSTEM:WINDOWS")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STD} -pthread -fno-rtti -fvisibility-inlines-hidden")
  endif()

  # Address Sanitize Flag.
  if(ADDRESS_SANITIZER)
    set(addr_var -fsanitize=address)
  else()
    set(addr_var )
  endif()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS} ${addr_var}")

  if(LLVM_PACKAGE_VERSION VERSION_EQUAL "16.0.0" OR LLVM_PACKAGE_VERSION VERSION_GREATER "16.0.0")
    set(LIB_CLANG_RES ${LLVM_VERSION_MAJOR})
  else()
    set(LIB_CLANG_RES ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH})
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CFLAGS} -DLIB_CLANG_RES=${LIB_CLANG_RES} ${addr_var}")

  set(INSTALL_PATH_DOC_STRING "hipify-clang Installation Path")
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/dist" CACHE PATH ${INSTALL_PATH_DOC_STRING} FORCE)
  endif()

  set(HIPIFY_BIN_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/bin")

  install(TARGETS hipify-clang DESTINATION bin)
  # Install bin directory in CMAKE_INSTALL_PREFIX path.
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    DESTINATION .
    USE_SOURCE_PERMISSIONS
    PATTERN "hipify-perl"
    PATTERN "*.sh")

  # Headers are already included in HIP SDK, so skip those if including HIPIFY in HIP SDK.
  if(NOT HIPIFY_INCLUDE_IN_HIP_SDK)
    # Install all folders under clang/version/ in CMAKE_INSTALL_PREFIX path.
    install(
      DIRECTORY ${LLVM_DIR}/../../clang/${LIB_CLANG_RES}/
      DESTINATION .
      COMPONENT clang-resource-headers
      FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.modulemap"
      PATTERN "algorithm"
      PATTERN "complex"
      PATTERN "new"
      PATTERN "ppc_wrappers" EXCLUDE
      PATTERN "omp-tools.h" EXCLUDE
      PATTERN "omp.h" EXCLUDE
      PATTERN "ompt.h" EXCLUDE
      PATTERN "openmp_wrappers" EXCLUDE)
  endif()

  if(UNIX)
    # Get rid of any RPATH definations already.
    set_target_properties(hipify-clang PROPERTIES INSTALL_RPATH "")
    # Set RPATH for the binary.
    set_target_properties(hipify-clang PROPERTIES LINK_FLAGS "-Wl,--enable-new-dtags -Wl,--rpath,$ORIGIN/../lib" )

    option(FILE_REORG_BACKWARD_COMPATIBILITY "Enable File Reorg with backward compatibility" OFF)

    if(FILE_REORG_BACKWARD_COMPATIBILITY)
      include(hipify-backward-compat.cmake)
    endif()

    set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE PATH "HIP Package Installation Path")
    set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/hipify-clang)

    configure_file(packaging/hipify-clang.txt ${BUILD_DIR}/CMakeLists.txt @ONLY)
    configure_file(${CMAKE_SOURCE_DIR}/LICENSE.txt ${BUILD_DIR}/LICENSE.txt @ONLY)

    add_custom_target(package_hipify-clang COMMAND ${CMAKE_COMMAND} .
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMAND rm -rf *.deb *.rpm *.tar.gz
      COMMAND make package
      COMMAND cp *.deb ${PROJECT_BINARY_DIR}
      COMMAND cp *.rpm ${PROJECT_BINARY_DIR}
      COMMAND cp *.tar.gz ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${BUILD_DIR})
  endif()

endif() # if (NOT HIPIFY_CLANG_TESTS_ONLY)

if(HIPIFY_CLANG_TESTS OR HIPIFY_CLANG_TESTS_ONLY)

  message(STATUS "---- The below configuring for hipify-clang testing only ----")

  set (HIPIFY_CLANG_RES "${LLVM_LIBRARY_DIRS}/clang/${LIB_CLANG_RES}")

  if(${CMAKE_VERSION} VERSION_LESS "3.27.0")
    find_package(PythonInterp 2.7 REQUIRED)
  else()
    find_package(Python 2.7...3.13 REQUIRED)
  endif()

  function (require_program PROGRAM_NAME)
    find_program(FOUND_${PROGRAM_NAME} ${PROGRAM_NAME})
    if(FOUND_${PROGRAM_NAME})
       message(STATUS "Found ${PROGRAM_NAME}: ${FOUND_${PROGRAM_NAME}}")
    else()
       message(SEND_ERROR "Can't find ${PROGRAM_NAME}. Either set HIPIFY_CLANG_TESTS(_ONLY) to OFF to disable HIPIFY tests, or install the missing program.")
    endif()
  endfunction()

  require_program(lit)
  require_program(FileCheck)

  set(CUDA_TOOLKIT_ROOT_DIR OFF CACHE PATH "Path to CUDA Toolkit to use in hipify-clang unit testing")
  set(CUDA_SDK_ROOT_DIR OFF CACHE PATH "Path to CUDA Toolkit Samples")
  set(CUDA_DNN_ROOT_DIR OFF CACHE PATH "Path to cuDNN")
  set(CUDA_CUB_ROOT_DIR OFF CACHE PATH "Path to CUB")

  if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT CUDA_TOOLKIT_ROOT_DIR MATCHES "OFF")
    if(NOT DEFINED CUDAToolkit_ROOT OR CUDAToolkit_ROOT MATCHES "OFF")
      set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
    endif()
  else()
    if(DEFINED CUDAToolkit_ROOT AND NOT CUDAToolkit_ROOT MATCHES "OFF")
      set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}")
    endif()
  endif()

  message(STATUS "Initial CUDA to configure:")
  message(STATUS "   - CUDA Toolkit path  : ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "   - CUDA Samples path  : ${CUDA_SDK_ROOT_DIR}")
  message(STATUS "   - cuDNN path         : ${CUDA_DNN_ROOT_DIR}")
  message(STATUS "   - CUB path           : ${CUDA_CUB_ROOT_DIR}")

  if(${CMAKE_VERSION} VERSION_LESS "3.27.0")
    find_package(CUDA REQUIRED)
    set(CUDA_VERSION_FULL     "${CUDA_VERSION}")
  else()
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_VERSION_MAJOR    "${CUDAToolkit_VERSION_MAJOR}")
    set(CUDA_VERSION_MINOR    "${CUDAToolkit_VERSION_MINOR}")
    set(CUDA_VERSION_PATCH    "${CUDAToolkit_VERSION_PATCH}")
    set(CUDA_VERSION_FULL     "${CUDAToolkit_VERSION}")
    set(CUDA_VERSION          "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
    set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_LIBRARY_ROOT}")
  endif()

  if(WIN32 AND CUDA_VERSION VERSION_LESS "11.6" AND CUDA_SDK_ROOT_DIR STREQUAL "OFF" AND NOT CUDA_TOOLKIT_ROOT_DIR MATCHES "OFF")
    set(NVCUDASAMPLES "NVCUDASAMPLES${CUDA_VERSION_MAJOR}_${CUDA_VERSION_MINOR}_ROOT")
    if(DEFINED ENV{${NVCUDASAMPLES}})
      set(CUDA_SDK_ROOT_DIR "$ENV{${NVCUDASAMPLES}}")
      string(REPLACE "\\" "/" CUDA_SDK_ROOT_DIR ${CUDA_SDK_ROOT_DIR})
    endif()
  endif()

  message(STATUS "Found CUDA config:")
  message(STATUS "   - CUDA Toolkit path  : ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "   - CUDA Samples path  : ${CUDA_SDK_ROOT_DIR}")
  message(STATUS "   - cuDNN path         : ${CUDA_DNN_ROOT_DIR}")
  message(STATUS "   - CUB path           : ${CUDA_CUB_ROOT_DIR}")

  if((CUDA_VERSION VERSION_LESS "7.0") OR (LLVM_PACKAGE_VERSION VERSION_LESS "3.8") OR
     (CUDA_VERSION VERSION_GREATER "7.5" AND LLVM_PACKAGE_VERSION VERSION_LESS "4.0") OR
     (CUDA_VERSION VERSION_GREATER "8.0" AND LLVM_PACKAGE_VERSION VERSION_LESS "6.0") OR
     (CUDA_VERSION VERSION_GREATER "9.0" AND LLVM_PACKAGE_VERSION VERSION_LESS "7.0") OR
     (CUDA_VERSION VERSION_GREATER "9.2" AND LLVM_PACKAGE_VERSION VERSION_LESS "8.0") OR
     (CUDA_VERSION VERSION_GREATER "10.0" AND LLVM_PACKAGE_VERSION VERSION_LESS "9.0") OR
     (CUDA_VERSION VERSION_GREATER "10.1" AND LLVM_PACKAGE_VERSION VERSION_LESS "10.0"))
    message(SEND_ERROR "CUDA ${CUDA_VERSION} is not supported by LLVM ${LLVM_PACKAGE_VERSION}.")
    if(CUDA_VERSION_MAJOR VERSION_LESS "7")
      message(STATUS "Please install CUDA 7.0 or higher.")
    elseif(CUDA_VERSION_MAJOR VERSION_LESS "8")
      message(STATUS "Please install LLVM + clang 3.8 or higher.")
    elseif(CUDA_VERSION_MAJOR VERSION_LESS "9")
      message(STATUS "Please install LLVM + clang 4.0 or higher.")
    elseif(CUDA_VERSION VERSION_EQUAL "9.0")
      message(STATUS "Please install LLVM + clang 6.0 or higher.")
    elseif(CUDA_VERSION_MAJOR VERSION_LESS "10")
      message(STATUS "Please install LLVM + clang 7.0 or higher.")
    elseif(CUDA_VERSION VERSION_EQUAL "10.0")
      message(STATUS "Please install LLVM + clang 8.0 or higher.")
    elseif(CUDA_VERSION VERSION_EQUAL "10.1")
      message(STATUS "Please install LLVM + clang 9.0 or higher.")
    elseif(CUDA_VERSION VERSION_EQUAL "10.2" OR CUDA_VERSION VERSION_EQUAL "11.0")
      message(STATUS "Please install LLVM + clang 10.0 or higher.")
    endif()
  endif()

  configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/tests/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/tests/lit.site.cfg
    @ONLY)

  if(HIPIFY_CLANG_TESTS_ONLY)
    add_lit_testsuite(test-hipify "Running HIPIFY regression tests"
      ${CMAKE_CURRENT_LIST_DIR}/tests
      PARAMS site_config=${CMAKE_CURRENT_BINARY_DIR}/tests/lit.site.cfg
      ARGS -v)
  else()
    add_lit_testsuite(test-hipify "Running HIPIFY regression tests"
      ${CMAKE_CURRENT_LIST_DIR}/tests
      PARAMS site_config=${CMAKE_CURRENT_BINARY_DIR}/tests/lit.site.cfg
      ARGS -v
      DEPENDS hipify-clang)
  endif()

  add_custom_target(test-hipify-clang)
  add_dependencies(test-hipify-clang test-hipify)
  set_target_properties(test-hipify-clang PROPERTIES FOLDER "Tests")

endif() # if(HIPIFY_CLANG_TESTS OR HIPIFY_CLANG_TESTS_ONLY)
