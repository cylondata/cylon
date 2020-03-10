cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(TWISTERX_JNI)
set(TWISTERX_VERSION 0.0)

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(TEST_DATA_DIR ${TWISTERX_SOURCE_DIR}/data)

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
endif()

SET(CMAKE_REQUIRED_FLAGS "-std=c++17")
add_compile_options(-Wall -Wextra -W)

# C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

set(TWISTERX_INCLUDE "${PROJECT_SOURCE_DIR}/../../../../cpp/src"
        "${PROJECT_SOURCE_DIR}/../../../../cpp/thirdparty/cub")
find_library(TWISTERX_LIBRARY "twisterx" "../../../../cpp/build/lib")
message(STATUS "TWISTERX_LIBRARY set to ${TWISTERX_LIBRARY}")
add_library(twisterx SHARED IMPORTED ${TWISTERX_LIBRARY})
set_target_properties(twisterx PROPERTIES IMPORTED_LOCATION ${TWISTERX_LIBRARY})


find_package(JNI REQUIRED)
if(JNI_FOUND)
    message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif(JNI_FOUND)

include_directories("include"
                    "${JNI_INCLUDE_DIRS}")

set(SOURCE_FILES
        "src/AllToAll.cpp"
        "src/AllToAll.h" src/Join.cpp)
add_library(twisterxjni SHARED ${SOURCE_FILES})

include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(twisterxjni ${MPI_LIBRARIES})
target_link_libraries(twisterxjni twisterx)
target_link_libraries(twisterxjni ${JNI_LIBRARIES})
