##########################################################################
#	CMake Build Rules for the TwisterX                               #
##########################################################################
# Basic Usage:                                                           #
#   cmake .								 #
#   make  								 #
#   make test                                                            #
# For more information about CMake, see http://www.cmake.org             #
##########################################################################
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(TWISTERX)
set(TWISTERX_VERSION 0.1)

# custom module path
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/Modules/" ${CMAKE_MODULE_PATH})

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
endif ()

IF (NOT (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RELEASE"))
    SET(CMAKE_BUILD_TYPE "Debug")
ENDIF (NOT (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RELEASE"))

IF (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
ENDIF (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG")

# Put the libaries and binaries that get built into directories at the
# top of the build.
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(TEST_DATA_DIR ${TWISTERX_SOURCE_DIR}/data)

# C++ standard
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_REQUIRED_FLAGS "-std=c++17")
add_compile_options(-Wall -Wextra -W)

# glog
option(WITH_GFLAGS "Use gflags" ON)
SET(WITH_GFLAGS OFF)
add_subdirectory(thirdparty/glog)

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# arrow include
include(ConfigureArrow)
include_directories("${ARROW_INCLUDE_DIR}")

include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
        "#include<numeric>
    int main() {
        int* ptr;
        std::exclusive_scan(ptr, ptr, ptr, 0);
    }"
        HAVE_EXCLUSIVE_SCAN
)

if (NOT HAVE_EXCLUSIVE_SCAN)
    add_definitions(-DNEED_EXCLUSIVE_SCAN)
endif ()

# module directories
add_subdirectory(src/twisterx)
add_subdirectory(src/examples)