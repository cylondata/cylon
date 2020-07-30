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
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(CYLON)

set(CYLON_VERSION 0.1.0)

## defaults to debug build
IF (NOT (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RELEASE"))
    SET(CMAKE_BUILD_TYPE "Debug")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}") #  -fsanitize=address
ENDIF (NOT (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RELEASE"))

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

# cmake modules directories
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/Modules/" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${CYLON_SOURCE_DIR}/CMake)

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

# Find MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# glog
option(WITH_GFLAGS "Use gflags" ON)
SET(WITH_GFLAGS OFF)
include(ConfigureGlog)
include_directories(SYSTEM "${GLOG_INCLUDE_DIR}")

# arrow
include(ConfigureArrow)
include(Build)
if (ARROW_FOUND)
    message(STATUS "Apache Arrow found in ${ARROW_INCLUDE_DIR}")
else ()
    message(FATAL_ERROR "Apache Arrow not found, please check your settings.")
endif (ARROW_FOUND)
include_directories(SYSTEM "${ARROW_INCLUDE_DIR}")

## Find Python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Off if you dont want to build tests -- ON default
option(CYLON_BUILD_TESTS "Build tests." ON)

add_subdirectory(src/cylon)
add_subdirectory(src/examples)

## installing arrow files
install(DIRECTORY ${CMAKE_BINARY_DIR}/arrow/install/include/arrow DESTINATION include)
install(FILES ${CMAKE_BINARY_DIR}/arrow/install/lib/libarrow.so.16.0.0 DESTINATION lib)
install(FILES ${CMAKE_BINARY_DIR}/arrow/install/lib/libarrow.a DESTINATION lib)

if (CYLON_WITH_TEST)
    message("Tests enabled!")
    enable_testing()
    add_subdirectory(test)
endif()
