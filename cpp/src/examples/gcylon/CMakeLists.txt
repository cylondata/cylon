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

add_definitions(-DEXAMPLE_CMAKE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")

set(CMAKE_CUDA_COMPILER $ENV{CUDACXX})

# project(GCYLON LANGUAGES CXX CUDA)
enable_language(CUDA)

set("GCYLON_SIMD_LEVEL" "SSE4_2") # options: SSE4_2| AVX512 | AVX2
set(CMAKE_CUDA_ARCHITECTURES 60 61 75)

# C++ standard
# with cudf v21.08, at least c++ 17 required
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_REQUIRED_FLAGS "-std=c++17")

# find CUDA
find_package(CUDAToolkit REQUIRED)

# if conda build is used
if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CUDF_CUDA_INCLUDE_DIR "$ENV{BUILD_PREFIX}/include/libcudf/libcudacxx/" "$ENV{PREFIX}/include/libcudf/libcudacxx/")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CUDF_CUDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include/libcudf/libcudacxx/")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

include_directories(../../
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CONDA_INCLUDE_DIRS}
        ${CUDF_CUDA_INCLUDE_DIR}
        )

#build a binary with cuda
macro(cylon_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp ../../gcylon/utils/construct.hpp print.hpp)
    target_link_libraries(${EXENAME} gcylon)
    install(TARGETS ${EXENAME} RUNTIME DESTINATION gcylon/examples)
endmacro(cylon_add_exe)

cylon_add_exe(join)
cylon_add_exe(join_csv)
cylon_add_exe(shuffle_perf)
