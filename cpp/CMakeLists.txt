##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

##########################################################################
#	CMake Build Rules for the Cylon                               #
##########################################################################
# Basic Usage:                                                           #
#   cmake .								 #
#   make  								 #
#   make test                                                            #
# For more information about CMake, see http://www.cmake.org             #
##########################################################################
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(CYLON VERSION 0.4.0)

set(CYLON_VERSION 0.4.0)

## defaults to release build
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# cmake modules directories
set(CYLON_ARROW_VERSION 4.0.1)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/Modules/" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${CYLON_SOURCE_DIR}/CMake)

# Compiler specific flags
IF(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
ELSE()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
ENDIF()

if (NOT DEFINED CYLON_SIMD_LEVEL)
    set("CYLON_SIMD_LEVEL" "SSE4_2") # options: SSE4_2| AVX512 | AVX2
endif ()

# Add common flags
include(SetupCxxFlags)
message("CXX_COMMON_FLAGS: ${CXX_COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COMMON_FLAGS}")
set(CMAKE_INSTALL_INCLUDEDIR "include")

string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)
if (UPPERCASE_CMAKE_BUILD_TYPE MATCHES "DEBUG")
    message("Running on debug mode...")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer ")
elseif (UPPERCASE_CMAKE_BUILD_TYPE MATCHES "RELEASE")
    message("Running on Release mode...")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
else ()
    message(FATAL_ERROR "Unknown CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}! Exiting...")
endif ()

## PyCylon build
if (PYCYLON_BUILD)
    message("Cylon Python Build [READ] ${PYCYLON_BUILD}")
else ()
    message("Cylon Python Build [UNREAD] ${PYTCYLON_BUILD}")
    set(PYCYLON_BUILD OFF)
endif ()

# C++ standard
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(GCC_ABI_COMPILE_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
if (NOT HAVE_EXCLUSIVE_SCAN)
    add_definitions(-DNEED_EXCLUSIVE_SCAN)
endif ()

# conda environment
if ("$ENV{CONDA_BUILD}" STREQUAL "1")
    message("CONDA_BUILD detected. BUILD_PREFIX: $ENV{BUILD_PREFIX} PREFIX: $ENV{PREFIX}")

    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{BUILD_PREFIX}/lib" "$ENV{PREFIX}/lib")
    set(ARROW_BUILD_TYPE "SYSTEM")
    set(CMAKE_INSTALL_INCLUDEDIR "$ENV{BUILD_PREFIX}/include")

    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
    message(STATUS "CONDA_INCLUDE_DIRS set to: ${CONDA_INCLUDE_DIRS}")
    message(STATUS "CONDA_LINK_DIRS set to: ${CONDA_LINK_DIRS}")


    #    if conda build is on, set Python Root DIR to conda
    set(Python3_ROOT_DIR "$ENV{BUILD_PREFIX}")
elseif (DEFINED ENV{CONDA_PREFIX})
    message("CONDA_PREFIX detected: $ENV{CONDA_PREFIX}")

    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib $ENV{CONDA_PREFIX}/Library/lib")
    set(ARROW_BUILD_TYPE "SYSTEM")

    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
    message(STATUS "CONDA_INCLUDE_DIRS set to: ${CONDA_INCLUDE_DIRS}")
    message(STATUS "CONDA_LINK_DIRS set to: ${CONDA_LINK_DIRS}")

    #    if conda prefix is on, set Python Root DIR to conda
    set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}")
endif ("$ENV{CONDA_BUILD}" STREQUAL "1")

## PyCylon build
if (PYCYLON_BUILD)
    message("Cylon Python Build [READ] ${PYCYLON_BUILD}")

    # LocationByValue Python
    find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
    message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
    message(STATUS "Python3 include dir: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3 NumPy include dir: ${Python3_NumPy_INCLUDE_DIRS}")

    include_directories(${Python3_INCLUDE_DIRS})
else ()
    message("Cylon Python Build [UNREAD] ${PYTCYLON_BUILD}")
    set(PYCYLON_BUILD OFF)
endif (PYCYLON_BUILD)

# Put the libaries and binaries that get built into directories at the
# top of the build.
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(TEST_DATA_DIR ${CYLON_SOURCE_DIR}/data)

# root include directory
#include_directories(src/cylon)
include_directories(src/)
if(CONDA_INCLUDE_DIRS)
    include_directories("${CONDA_INCLUDE_DIRS}")
endif(CONDA_INCLUDE_DIRS)

if(CONDA_LINK_DIRS)
    link_directories("${CONDA_LINK_DIRS}")
endif(CONDA_LINK_DIRS)

# LocationByValue MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# glog, if conda build, we will get glog from conda instead of building ourselves
if (CONDA_INCLUDE_DIRS)
    find_package (glog 0.5.0 REQUIRED)
    set(GLOG_LIBRARIES glog::glog)
else()
    option(WITH_GFLAGS "Use gflags" ON)
    SET(WITH_GFLAGS OFF)
    include(ConfigureGlog)
    include_directories(SYSTEM "${GLOG_INCLUDE_DIR}")
endif (CONDA_INCLUDE_DIRS)


# this is required on windows to prevent clashes between ERROR abbreviation in logging.h and windows.h
if(WIN32)
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
endif()

include(Build)

# if building gcylon, no need to build cylon
if (GCYLON_BUILD)
    add_subdirectory(src/gcylon)
    add_subdirectory(src/examples/gcylon)
    if (CYLON_WITH_TEST)
        message("Tests enabled!")
        enable_testing()
        add_subdirectory(test/gcylon)
    endif ()
    return()
endif ()

# parquet
option(CYLON_PARQUET "Build Cylon with Parquet functionalities" OFF)
if (CYLON_PARQUET)
    add_definitions(-DBUILD_CYLON_PARQUET)
endif (CYLON_PARQUET)

# ucx
option(CYLON_UCX "Build Cylon with UCX" OFF)
if (CYLON_UCX)
    # Check if UCX include and lib paths are given
    if (NOT UCX_INCLUDEDIR)
        message(FATAL_ERROR "CYLON_UCX is set, UCX_INCLUDEDIR should also be set")
    endif ()
    if (NOT UCX_LIBDIR)
        message(FATAL_ERROR "CYLON_UCX is set, UCX_LIBDIR should also be set")
    endif ()

    # Definition used for checking
    add_definitions(-DBUILD_CYLON_UCX)

    # Set UCX found as true
    set(UCX_FOUND TRUE)

    # Include UCX
    include_directories(${UCX_INCLUDEDIR})
    # Set library directory for later use
    set(UCX_LIBRARIES
            ${UCX_LIBDIR}/ucx/libuct_cma.so
            ${UCX_LIBDIR}/ucx/libuct_ib.so
            ${UCX_LIBDIR}/libuct.so
            ${UCX_LIBDIR}/libucs.so
            ${UCX_LIBDIR}/libucm.so
            ${UCX_LIBDIR}/libucp.so)
endif (CYLON_UCX)

if (NOT ARROW_BUILD_TYPE)
    set(ARROW_BUILD_TYPE "SOURCE")
endif()

if (${ARROW_BUILD_TYPE} STREQUAL "SYSTEM")
    message("Finding Arrow from SYSTEM")
    find_package(Arrow REQUIRED)
    message(STATUS "Arrow ver: ${ARROW_FULL_SO_VERSION}")
    message(STATUS "Arrow include dir: ${ARROW_INCLUDE_DIR}")
    message(STATUS "Arrow lib dir: ${ARROW_LIB_DIR}")
    set(ARROW_LIB arrow_shared)

    if (CYLON_PARQUET)
        find_library(PARQUET_LIB parquet REQUIRED)
        message(STATUS "Parquet lib: ${PARQUET_LIB}")
    endif ()
elseif (${ARROW_BUILD_TYPE} STREQUAL "SOURCE")
    message("Building Arrow from SOURCE")
    include(ConfigureArrow)
    include_directories(SYSTEM "${ARROW_INCLUDE_DIR}")
elseif (${ARROW_BUILD_TYPE} STREQUAL "CUSTOM")
    message("Using CUSTOM Arrow installation")

    if (NOT ARROW_LIB_DIR)
        message(FATAL_ERROR "ARROW_BUILD_TYPE is set to CUSTOM, ARROW_LIB_DIR should be set")
    endif ()

    if (NOT ARROW_INCLUDE_DIR)
        message(FATAL_ERROR "ARROW_BUILD_TYPE is set to CUSTOM, ARROW_INCLUDE_DIR should be set")
    endif ()
    add_definitions(${GCC_ABI_COMPILE_FLAGS})
    LINK_DIRECTORIES(${ARROW_LIB_DIR})

    string(REPLACE "." "" ARROW_SO_VERSION ${CYLON_ARROW_VERSION})
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".so.${ARROW_SO_VERSION}")
    find_library(ARROW_LIB arrow REQUIRED HINTS "${ARROW_LIB_DIR}")
    message("ARROW_LIB: ${ARROW_LIB}")

    if (CYLON_PARQUET)
        find_library(PARQUET_LIB parquet REQUIRED HINTS "${ARROW_LIB_DIR}")
        message(STATUS "Parquet lib: ${PARQUET_LIB}")
    endif ()

    include_directories(SYSTEM ${ARROW_INCLUDE_DIR})
else ()
    message(FATAL_ERROR "ARROW_BUILD_TYPE should be SYSTEM, SOURCE or CUSTOM")
endif()

# Off if you dont want to build tests -- ON default
option(CYLON_BUILD_TESTS "Build tests." ON)

add_subdirectory(src/cylon)
add_subdirectory(src/examples)
add_subdirectory(src/tutorial)

# installing arrow files
if (${ARROW_BUILD_TYPE} STREQUAL "SOURCE")
    #    Following variables will be populated by FindPackage(Arrow...) in ConfigureArrow.cmake
    message(STATUS "Arrow include dir: ${ARROW_INCLUDE_DIR}")
    message(STATUS "Arrow libs: ${ARROW_SHARED_LIB} ${ARROW_STATIC_LIB}")

    # install locally built arrow files
    install(DIRECTORY ${ARROW_INCLUDE_DIR}/arrow DESTINATION include)
    # install libarrow.so, libarrow.so.xxx, libarrow.so.xxx.x.x, and libarrow.a
    install(FILES ${ARROW_SHARED_LIB} ${ARROW_SHARED_LIB}.${ARROW_SO_VERSION}
            ${ARROW_SHARED_LIB}.${ARROW_FULL_SO_VERSION} ${ARROW_STATIC_LIB}
            DESTINATION lib)

    # glog.a file
    install(FILES ${GLOG_LIBRARIES} DESTINATION lib)

    # install parquet from arrow files
    if (CYLON_PARQUET)
        message(STATUS "parquet include dir: ${PARQUET_INCLUDE_DIR}")
        message(STATUS "parquet libs: ${PARQUET_SHARED_LIB} ${PARQUET_STATIC_LIB}")

        install(DIRECTORY ${PARQUET_INCLUDE_DIR}/parquet DESTINATION include)
        install(FILES ${PARQUET_SHARED_LIB} ${PARQUET_SHARED_LIB}.${ARROW_SO_VERSION}
                ${PARQUET_SHARED_LIB}.${ARROW_FULL_SO_VERSION} ${PARQUET_STATIC_LIB}
                DESTINATION lib)
    endif ()

    if (PYCYLON_BUILD)
        message(STATUS "arrow_python include dir: ${ARROW_PYTHON_INCLUDE_DIR}")
        message(STATUS "arrow_python libs: ${ARROW_PYTHON_SHARED_LIB} ${ARROW_PYTHON_STATIC_LIB}")

        # install locally built arrow_python files
        install(DIRECTORY ${ARROW_PYTHON_INCLUDE_DIR}/arrow DESTINATION include)
        install(FILES ${ARROW_PYTHON_SHARED_LIB} ${ARROW_PYTHON_SHARED_LIB}.${ARROW_SO_VERSION}
                ${ARROW_PYTHON_SHARED_LIB}.${ARROW_FULL_SO_VERSION} ${ARROW_PYTHON_STATIC_LIB}
                DESTINATION lib)
    endif ()

endif ()

if (CYLON_WITH_TEST)
    message("Tests enabled!")
    enable_testing()
    add_subdirectory(test)
endif ()
