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

# Find the CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# we enable all archs
# can be limited
set(CMAKE_CUDA_ARCHITECTURES 60 61 75)

# this must come after find_package and setting CMAKE_CUDA_ARCHITECTURES
# according to cudf build files
# https://github.com/rapidsai/cudf/blob/branch-21.10/cpp/cmake/Modules/ConfigureCUDA.cmake
enable_language(CUDA)
message("CUDAToolkit_INCLUDE_DIRS: " ${CUDAToolkit_INCLUDE_DIRS})

# If executed on github actions, set below flag to avoid linker error for cuda
# ref: https://github.com/NVIDIA/nvidia-docker/issues/775
if(DEFINED ENV{GITHUB_ACTIONS})
    set(CMAKE_CXX_FLAGS "-Wl,--allow-shlib-undefined")
endif()

message("CONDA_PREFIX: " $ENV{CONDA_PREFIX})
if ((NOT DEFINED ENV{CONDA_PREFIX}) AND (NOT DEFINED ENV{CONDA_BUILD}) )
    message(SEND_ERROR "Neither CONDA_PREFIX nor CONDA_BUILD is set. Activate conda environment or use conda-build")
endif ()

## defaults to release build
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (NOT DEFINED GCYLON_SIMD_LEVEL)
    set("GCYLON_SIMD_LEVEL" "SSE4_2") # options: SSE4_2| AVX512 | AVX2
endif ()

# C++ standard
# with cudf v21.08, at least c++ 17 required
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_REQUIRED_FLAGS "-std=c++17")

if (NOT HAVE_EXCLUSIVE_SCAN)
    add_definitions(-DNEED_EXCLUSIVE_SCAN)
endif ()

# if conda build is used
if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CUDF_CUDA_INCLUDE_DIR "$ENV{BUILD_PREFIX}/include/libcudf/libcudacxx/" "$ENV{PREFIX}/include/libcudf/libcudacxx/")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CUDF_CUDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include/libcudf/libcudacxx/")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

# Put the libaries and binaries that get built into directories at the top of the build.
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(TEST_DATA_DIR ${GCYLON_SOURCE_DIR}/data)

set(CYLON_INCLUDE_DIR "../")

include_directories(
        ${CYLON_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CONDA_INCLUDE_DIRS}
        ${CUDF_CUDA_INCLUDE_DIR}
)

add_library(gcylon SHARED
        cudf_buffer.hpp
        cudf_buffer.cpp
        all2all/cudf_all_to_all.cuh
        all2all/cudf_all_to_all.cu
        all2all/cudf_all_to_all.hpp
        all2all/cudf_all_to_all.cpp
        net/cudf_net_ops.hpp
        net/cudf_bcast.cpp
        net/cudf_serialize.hpp
        net/cudf_serialize.cpp
        net/cudf_gather.cpp
        sorting/deserialize.hpp
        sorting/deserialize.cpp
        sorting/sorting.hpp
        sorting/sorting.cpp
        utils/util.hpp
        utils/util.cpp
        gtable.hpp
        gtable.cpp
        gtable_api.hpp
        gtable_api.cpp
        )

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

set_target_properties(gcylon PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET gcylon PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gcylon PROPERTIES VERSION ${CYLON_VERSION})

target_link_libraries(gcylon
        ${GLOG_LIBRARIES}
        cylon
        cudf
        CUDA::cudart
        ${MPI_CXX_LIBRARIES}
)

cylon_install_all_headers("gcylon")

add_subdirectory(all2all)
add_subdirectory(utils)

install(TARGETS gcylon DESTINATION lib)