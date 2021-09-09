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
    set(CMAKE_CXX_FLAGS  "-Wall -Wextra -Wno-error=redundant-move")
ENDIF()

if (NOT DEFINED CYLON_SIMD_LEVEL)
    set("CYLON_SIMD_LEVEL" "SSE4_2") # options: SSE4_2| AVX512 | AVX2
endif ()

# Add common flags
include(SetupCxxFlags)
message("CXX_COMMON_FLAGS: ${CXX_COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COMMON_FLAGS}")
set(CMAKE_INSTALL_INCLUDEDIR "include")

# Check build type (debug/ release) and set flags
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

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(GCC_ABI_COMPILE_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
if (NOT HAVE_EXCLUSIVE_SCAN)
    add_definitions(-DNEED_EXCLUSIVE_SCAN)
endif ()

# check for conda environment
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

    set(CYLON_CONDA_BUILD ON)
elseif (DEFINED ENV{CONDA_PREFIX})
    message("CONDA_PREFIX detected: $ENV{CONDA_PREFIX}")

    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/Library/lib")
    set(ARROW_BUILD_TYPE "SYSTEM")

    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
    message(STATUS "CONDA_INCLUDE_DIRS set to: ${CONDA_INCLUDE_DIRS}")
    message(STATUS "CONDA_LINK_DIRS set to: ${CONDA_LINK_DIRS}")

    #    if conda prefix is on, set Python Root DIR to conda
    set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}")

    set(CYLON_CONDA_BUILD ON)
endif ("$ENV{CONDA_BUILD}" STREQUAL "1")

## PyCylon build
option(PYCYLON_BUILD "Enable PyCylon build" OFF)
if (PYCYLON_BUILD)
    message("Cylon Python Build enabled")

    # LocationByValue Python
    find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
    message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
    message(STATUS "Python3 include dir: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3 NumPy include dir: ${Python3_NumPy_INCLUDE_DIRS}")

    include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
endif (PYCYLON_BUILD)

# Put the libaries and binaries that get built into directories at the
# top of the build.
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(TEST_DATA_DIR ${CYLON_SOURCE_DIR}/data)

# set cmake include directory
if (CYLON_CONDA_BUILD)
    include_directories(SYSTEM "${CONDA_INCLUDE_DIRS}")
endif (CYLON_CONDA_BUILD)
include_directories(src/)

# set cmake link dirs
if (CYLON_CONDA_BUILD)
    link_directories("${CONDA_LINK_DIRS}")
endif (CYLON_CONDA_BUILD)

# LocationByValue MPI
message("Finding MPI")
find_package(MPI REQUIRED COMPONENTS CXX)
message(STATUS "MPI include dir: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI libs: ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI executable: ${MPIEXEC_EXECUTABLE}")

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# Glog
# if conda build, we will get glog from conda instead of building ourselves
if (CYLON_CONDA_BUILD)
    message("Using Conda Glog")
    find_package(glog 0.5.0 REQUIRED)
    set(GLOG_LIBRARIES glog::glog)
else ()
    message("Building Glog locally")
    include(ConfigureGlog)
    include_directories(SYSTEM "${GLOG_INCLUDE_DIR}")
    message(STATUS "Glog include dir: ${GLOG_INCLUDE_DIR}")
endif (CYLON_CONDA_BUILD)
message(STATUS "Glog libs: ${GLOG_LIBRARIES}")

# this is required on windows to prevent clashes between ERROR abbreviation in logging.h and windows.h
if (WIN32)
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
endif ()

# include Modules/Build.cmake
include(Build)

# if building gcylon, no need to build cylon
option(GCYLON_BUILD "Build GCylon" OFF)
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

# ucx
option(CYLON_UCX "Build Cylon with UCX" OFF)
if (CYLON_UCX)
    # Definition used for checking
    add_definitions(-DBUILD_CYLON_UCX)

    if (CONDA_LINK_DIRS)
        message("Using Conda UCX")
        find_library(UCX_UCT uct REQUIRED HINTS ${CONDA_LINK_DIRS})
        find_library(UCX_UCS ucs REQUIRED HINTS ${CONDA_LINK_DIRS})
        find_library(UCX_UCM ucm REQUIRED HINTS ${CONDA_LINK_DIRS})
        find_library(UCX_UCP ucp REQUIRED HINTS ${CONDA_LINK_DIRS})
        set(UCX_LIBRARIES ${UCX_UCT} ${UCX_UCS} ${UCX_UCM} ${UCX_UCP})
    else (CONDA_LINK_DIRS)
        message("Using locally installed UCX")

        # Check if UCX include and lib paths are given
        if (NOT UCX_INCLUDEDIR)
            message(FATAL_ERROR "CYLON_UCX is set, UCX_INCLUDEDIR should also be set")
        endif ()
        if (NOT UCX_LIBDIR)
            message(FATAL_ERROR "CYLON_UCX is set, UCX_LIBDIR should also be set")
        endif ()

        # Set UCX found as true
        set(UCX_FOUND TRUE)

        # Include UCX
        include_directories(SYSTEM ${UCX_INCLUDEDIR})
        # Set library directory for later use
        set(UCX_LIBRARIES
                ${UCX_LIBDIR}/ucx/libuct_cma.so
                ${UCX_LIBDIR}/ucx/libuct_ib.so
                ${UCX_LIBDIR}/libuct.so
                ${UCX_LIBDIR}/libucs.so
                ${UCX_LIBDIR}/libucm.so
                ${UCX_LIBDIR}/libucp.so)
    endif (CONDA_LINK_DIRS)
    message(STATUS "UCX libs: ${UCX_LIBRARIES}")
endif (CYLON_UCX)

# Arrow
if (NOT ARROW_BUILD_TYPE)
    set(ARROW_BUILD_TYPE "SOURCE")
endif()

if (${ARROW_BUILD_TYPE} STREQUAL "SYSTEM")
    message("Finding Arrow from SYSTEM")
    find_package(Arrow REQUIRED)
    message(STATUS "Arrow ver: ${ARROW_FULL_SO_VERSION}")
    message(STATUS "Arrow include dir: ${ARROW_INCLUDE_DIR}")
    message(STATUS "Arrow lib dir: ${ARROW_LIB_DIR}")

    find_library(ARROW_LIB arrow ${CYLON_ARROW_VERSION} REQUIRED)
    message(STATUS "Arrow lib: ${ARROW_LIB}")

    if (PYCYLON_BUILD)
        find_library(ARROW_PY_LIB arrow_python ${CYLON_ARROW_VERSION} REQUIRED)
        message(STATUS "Arrow py lib: ${ARROW_PY_LIB}")
    endif (PYCYLON_BUILD)

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

    string(REPLACE "." "" ARROW_SO_VERSION ${CYLON_ARROW_VERSION})
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".so.${ARROW_SO_VERSION}")

    include_directories(SYSTEM ${ARROW_INCLUDE_DIR})
    link_directories(${ARROW_LIB_DIR})

    find_library(ARROW_LIB arrow ${CYLON_ARROW_VERSION} REQUIRED HINTS "${ARROW_LIB_DIR}")
    message(STATUS "Arrow lib: ${ARROW_LIB}")

    if (PYCYLON_BUILD)
        find_library(ARROW_PY_LIB arrow_python ${CYLON_ARROW_VERSION} REQUIRED HINTS "${ARROW_LIB_DIR}")
        message(STATUS "Arrow py lib: ${ARROW_PY_LIB}")
    endif (PYCYLON_BUILD)
else ()
    message(FATAL_ERROR "ARROW_BUILD_TYPE should be SYSTEM, SOURCE or CUSTOM. Given ${ARROW_BUILD_TYPE}")
endif ()

# parquet
option(CYLON_PARQUET "Build Cylon with Parquet functionalities" OFF)
if (CYLON_PARQUET)
    message("Cylon Parquet enabled")
    add_definitions(-DBUILD_CYLON_PARQUET)
    find_library(PARQUET_LIB parquet REQUIRED HINTS "${ARROW_LIB_DIR}")
    message(STATUS "Parquet lib: ${PARQUET_LIB}")
endif (CYLON_PARQUET)

add_subdirectory(src/cylon)
add_subdirectory(src/examples)
add_subdirectory(src/tutorial)

# installing lib files if built locally
if (${ARROW_BUILD_TYPE} STREQUAL "SOURCE")
    #    Following variables will be populated by FindPackage(Arrow...) in ConfigureArrow.cmake
    message(STATUS "Arrow include dir: ${ARROW_INCLUDE_DIR}")
    message(STATUS "Arrow libs: ${ARROW_SHARED_LIB} ${ARROW_STATIC_LIB}")

    # install locally built arrow files
    install(DIRECTORY ${ARROW_INCLUDE_DIR}/arrow DESTINATION include)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/arrow/install/lib/ DESTINATION lib
            PATTERN ${ARROW_INCLUDE_DIR}/cmake EXCLUDE
            PATTERN ${ARROW_INCLUDE_DIR}/pkgconfig EXCLUDE
            )

    # glog.a file
    install(FILES ${GLOG_LIBRARIES} DESTINATION lib)
endif ()

# Off if you dont want to build tests -- ON default
option(CYLON_WITH_TEST "Build Cylon C++ tests." OFF)
if (CYLON_WITH_TEST)
    message("C++ tests enabled")
    enable_testing()
    add_subdirectory(test)
endif ()
