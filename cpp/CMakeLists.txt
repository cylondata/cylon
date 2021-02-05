# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

##########################################################################
#	CMake Build Rules for the Cylon                               #
##########################################################################
# Basic Usage:                                                           #
#   cmake .								 #
#   make  								 #
#   make test                                                            #
# For more information about CMake, see http://www.cmake.org             #
##########################################################################
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(CYLON)

set(CYLON_VERSION 0.2.0)

## defaults to release build
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# cmake modules directories
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/Modules/" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${CYLON_SOURCE_DIR}/CMake)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")

set("CYLON_SIMD_LEVEL" "SSE4_2") # options: SSE4_2| AVX512 | AVX2

# Add common flags
include(SetupCxxFlags)
message("CXX_COMMON_FLAGS: ${CXX_COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COMMON_FLAGS}")

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

## Python Executable Path
if (PYTHON_EXEC_PATH)
    message("Python Executable Path ${PYTHON_EXEC_PATH}")
else ()
    set(PYTHON_EXEC_PATH /usr)
    message("Python Executable Path ${PYTHON_EXEC_PATH}")
endif ()

# C++ standard
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_REQUIRED_FLAGS "-std=c++17")
add_compile_options(-Wall -Wextra -W)
if (NOT HAVE_EXCLUSIVE_SCAN)
    add_definitions(-DNEED_EXCLUSIVE_SCAN)
endif ()

# Put the libaries and binaries that get built into directories at the
# top of the build.
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(TEST_DATA_DIR ${CYLON_SOURCE_DIR}/data)

# root include directory
include_directories(src/cylon)

# LocationByValue MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# glog
option(WITH_GFLAGS "Use gflags" ON)
SET(WITH_GFLAGS OFF)
include(ConfigureGlog)
include_directories(SYSTEM "${GLOG_INCLUDE_DIR}")

# parquet
option(CYLON_PARQUET "Build Cylon with Parquet functionalities" OFF)
if (CYLON_PARQUET)
    add_definitions(-DBUILD_CYLON_PARQUET)
endif (CYLON_PARQUET)

include(Build)

if (NOT ARROW_BUILD_TYPE)
    set(ARROW_BUILD_TYPE "SOURCE")
endif()

if (${ARROW_BUILD_TYPE} STREQUAL "SYSTEM")
    find_package(Arrow REQUIRED)
    set(ARROW_LIB libarrow.so)
elseif (${ARROW_BUILD_TYPE} STREQUAL "SOURCE")
    include(ConfigureArrow)
    include_directories(SYSTEM "${ARROW_INCLUDE_DIR}")
elseif (${ARROW_BUILD_TYPE} STREQUAL "CUSTOM")
    if (NOT ARROW_LIB_DIR)
        message(FATAL_ERROR "ARROW_BUILD_TYPE is set to CUSTOM, ARROW_LIB_DIR should be set")
    endif()

    if (NOT ARROW_INCLUDE_DIR)
        message(FATAL_ERROR "ARROW_BUILD_TYPE is set to CUSTOM, ARROW_INCLUDE_DIR should be set")
    endif()
    set(ARROW_LIB ${ARROW_LIB_DIR}/libarrow.so)
    include_directories(SYSTEM ${ARROW_INCLUDE_DIR})
else ()
    message(FATAL_ERROR "ARROW_BUILD_TYPE should be SYSTEM or SOURCE")
endif()

## LocationByValue Python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Off if you dont want to build tests -- ON default
option(CYLON_BUILD_TESTS "Build tests." ON)

add_subdirectory(src/cylon)
add_subdirectory(src/examples)
add_subdirectory(src/tutorial)

## installing arrow files
install(DIRECTORY ${CMAKE_BINARY_DIR}/arrow/install/include/arrow DESTINATION include)
install(FILES ${CMAKE_BINARY_DIR}/arrow/install/lib/libarrow.so DESTINATION lib)
install(FILES ${CMAKE_BINARY_DIR}/arrow/install/lib/libarrow.a DESTINATION lib)

if (CYLON_WITH_TEST)
    message("Tests enabled!")
    enable_testing()
    add_subdirectory(test)
endif ()
