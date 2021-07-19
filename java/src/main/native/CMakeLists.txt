#
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
#

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(CYLON_JNI)
set(CYLON_VERSION 0.0)

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(TEST_DATA_DIR ${CYLON_SOURCE_DIR}/data)

SET(CYLON_CORE_LIBS "../../../../cpp/build/lib" CACHE STRING "Path to cylon core libs")

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
endif ()

SET(CMAKE_REQUIRED_FLAGS "-std=c++17")
add_compile_options(-Wall -Wextra -W)

# C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

message("${PROJECT_SOURCE_DIR}/../../../../cpp/src")
set(CYLON_INCLUDE "${PROJECT_SOURCE_DIR}/../../../../cpp/src/")
find_library(CYLON_LIBRARY "cylon" "${CYLON_CORE_LIBS}")
find_library(ARROW_LIBRARY "arrow" "${CYLON_ARROW_DIR}/build/release")
message(STATUS "CYLON_LIBRARY set to ${CYLON_LIBRARY}")
message(STATUS "ARROW_LIBRARY set to ${ARROW_LIBRARY}")

add_library(cylon SHARED IMPORTED ${CYLON_LIBRARY})
add_library(arrow SHARED IMPORTED ${ARROW_LIBRARY})
set_target_properties(cylon PROPERTIES IMPORTED_LOCATION ${CYLON_LIBRARY})

find_package(JNI REQUIRED)
if (JNI_FOUND)
    message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif (JNI_FOUND)

message(STATUS "Inclduing ${CYLON_ARROW_DIR}/install/include/")
include_directories("include"
        "${JNI_INCLUDE_DIRS}" ${CYLON_INCLUDE} "${CYLON_ARROW_DIR}/install/include/")

set(SOURCE_FILES
        src/Table.cpp
        src/TwisterXContext.cpp
        src/ConversionUtils.h
        src/ConversionUtils.cpp
        src/ArrowTable.cpp
        src/Utils.hpp
        src/Row.cpp
        src/Utils.cpp)
add_library(cylonjni SHARED ${SOURCE_FILES})

include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(cylonjni ${MPI_LIBRARIES})
target_link_libraries(cylonjni cylon)
target_link_libraries(cylonjni ${JNI_LIBRARIES})
