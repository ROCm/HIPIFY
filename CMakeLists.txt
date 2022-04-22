cmake_minimum_required(VERSION 3.16.8)

project(hipify-clang)

find_package(LLVM REQUIRED CONFIG PATHS ${CMAKE_PREFIX_PATH})

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}:")
message(STATUS "   - CMake module path: ${LLVM_CMAKE_DIR}")
message(STATUS "   - Include path     : ${LLVM_INCLUDE_DIRS}")
message(STATUS "   - Binary path      : ${LLVM_TOOLS_BINARY_DIR}")

option(HIPIFY_CLANG_TESTS "Build HIPIFY tests, if lit is installed" OFF)
option(HIPIFY_CLANG_TESTS_ONLY "Build HIPIFY tests only, if lit is installed and hipify-clang binary is already produced" OFF)

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

if (NOT HIPIFY_CLANG_TESTS_ONLY)

if(MSVC AND MSVC_VERSION VERSION_LESS "1900")
    message(SEND_ERROR "hipify-clang could be built by Visual Studio 14 2015 or higher.")
    return()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

file(GLOB_RECURSE HIPIFY_SOURCES src/*.cpp)
file(GLOB_RECURSE HIPIFY_HEADERS src/*.h)
add_llvm_executable(hipify-clang ${HIPIFY_SOURCES} ${HIPIFY_HEADERS})

set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
set(CMAKE_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)

# Link against LLVM and CLANG libraries
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
    target_link_libraries(hipify-clang PRIVATE LLVMWindowsDriver)
endif()

if(MSVC)
    target_link_libraries(hipify-clang PRIVATE version)
    target_compile_options(hipify-clang PRIVATE /std:c++14 /Od /GR- /EHs- /EHc-)
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} /SUBSYSTEM:WINDOWS")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fno-rtti -fvisibility-inlines-hidden")
endif()

# Address Sanitize Flag
if(ADDRESS_SANITIZER)
    set(addr_var -fsanitize=address)
else()
    set(addr_var )
endif()

# [ToDo] Remove SWDEV_331863 related guards from CMakeLists.txt and HipifyAction.cpp when the blocker SWDEV_331863 is overcome
option (SWDEV_331863 "Enables SWDEV-331863 blocker workaround" OFF)
if(SWDEV_331863)
    add_definitions(-DSWDEV_331863)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS} ${addr_var}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CFLAGS} -DHIPIFY_CLANG_RES=\\\"${LLVM_LIBRARY_DIRS}/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}\\\" ${addr_var}")

set(INSTALL_PATH_DOC_STRING "hipify-clang Installation Path")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/dist" CACHE PATH ${INSTALL_PATH_DOC_STRING} FORCE)
endif()

set(HIPIFY_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})

install(TARGETS hipify-clang DESTINATION ${CMAKE_INSTALL_PREFIX})
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/bin/
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    USE_SOURCE_PERMISSIONS
    PATTERN "hipify-perl"
    PATTERN "*.sh")
install(
    DIRECTORY ${LLVM_DIR}/../../clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    COMPONENT clang-resource-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.modulemap"
    PATTERN "algorithm"
    PATTERN "complex"
    PATTERN "new"
    PATTERN "ppc_wrappers" EXCLUDE
    PATTERN "openmp_wrappers" EXCLUDE)

option(FILE_REORG_BACKWARD_COMPATIBILITY "Enable File Reorg with backward compatibility" ON)
if(UNIX)
    if(FILE_REORG_BACKWARD_COMPATIBILITY)
        include(hipify-backward-compat.cmake)
    endif()
    set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE PATH "HIP Package Installation Path")
    #TODO: To be removed
    #In jenkins and docker build , hipify need to be installed in /opt/rocm.
    #Currently build script passes /opt/rocm/hip as install path
    #Workaround for removing same, till prototype changes are merged
    if(CPACK_PACKAGING_INSTALL_PREFIX)
        string(FIND ${CPACK_PACKAGING_INSTALL_PREFIX} "/opt/rocm" ROCMDIR_FOUND)
        if(NOT ${ROCMDIR_FOUND} MATCHES "-1")
            get_filename_component(DEST_DIR ${CPACK_PACKAGING_INSTALL_PREFIX} NAME)
            if(DEST_DIR STREQUAL "hip")
                get_filename_component(CPACK_PACKAGING_INSTALL_PREFIX ${CPACK_PACKAGING_INSTALL_PREFIX} DIRECTORY)
            endif()#end  of DEST_DIR check
        endif()#end of rocm dir check
    endif()
    set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/hipify-clang)
    configure_file(packaging/hipify-clang.txt ${BUILD_DIR}/CMakeLists.txt @ONLY)
    configure_file(${CMAKE_SOURCE_DIR}/LICENSE.txt ${BUILD_DIR}/LICENSE.txt @ONLY)

    add_custom_target(package_hipify-clang  COMMAND ${CMAKE_COMMAND} .
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMAND rm -rf *.deb *.rpm *.tar.gz
        COMMAND make package
        COMMAND cp *.deb ${PROJECT_BINARY_DIR}
        COMMAND cp *.rpm ${PROJECT_BINARY_DIR}
        COMMAND cp *.tar.gz ${PROJECT_BINARY_DIR}
        WORKING_DIRECTORY ${BUILD_DIR})
endif()

endif()

if(HIPIFY_CLANG_TESTS OR HIPIFY_CLANG_TESTS_ONLY)
    find_package(PythonInterp 2.7 REQUIRED)

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

    find_package(CUDA REQUIRED)
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
endif()
