cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
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
set(CYLON_INCLUDE "${PROJECT_SOURCE_DIR}/../../../../cpp/src/cylon"
        "${PROJECT_SOURCE_DIR}/../../../../cpp/thirdparty/cub")
find_library(CYLON_LIBRARY "cylon" "${CYLON_CORE_LIBS}")
message(STATUS "CYLON_LIBRARY set to ${CYLON_LIBRARY}")
add_library(cylon SHARED IMPORTED ${CYLON_LIBRARY})
set_target_properties(cylon PROPERTIES IMPORTED_LOCATION ${CYLON_LIBRARY})


find_package(JNI REQUIRED)
if (JNI_FOUND)
    message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif (JNI_FOUND)

include_directories("include"
        "${JNI_INCLUDE_DIRS}" ${CYLON_INCLUDE})

set(SOURCE_FILES
        src/Table.cpp src/TwisterXContext.cpp src/ConversionUtils.h src/ConversionUtils.cpp src/ArrowTable.cpp src/Utils.hpp src/Row.cpp src/Utils.cpp)
add_library(cylonjni SHARED ${SOURCE_FILES})

include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(cylonjni ${MPI_LIBRARIES})
target_link_libraries(cylonjni cylon)
target_link_libraries(cylonjni ${JNI_LIBRARIES})
