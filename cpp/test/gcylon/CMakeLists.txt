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

# If executed on github actions, set below flag to avoid linker error for cuda
# ref: https://github.com/NVIDIA/nvidia-docker/issues/775
if(DEFINED ENV{GITHUB_ACTIONS})
  set(CMAKE_CXX_FLAGS "-Wl,--allow-shlib-undefined")
endif()

# if conda build is used
if("$ENV{CONDA_BUILD}" STREQUAL "1")
  set(CUDF_CUDA_INCLUDE_DIR "$ENV{BUILD_PREFIX}/include/libcudf/libcudacxx/" "$ENV{PREFIX}/include/libcudf/libcudacxx/")
elseif(DEFINED ENV{CONDA_PREFIX})
  set(CUDF_CUDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include/libcudf/libcudacxx/")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

SET(MPI_RUN_CMD mpirun)
file(COPY ${CYLON_SOURCE_DIR}/../data DESTINATION ${CMAKE_BINARY_DIR})

include_directories(
        ${CYLON_SOURCE_DIR}/src
        ${CYLON_SOURCE_DIR}/test
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CONDA_INCLUDE_DIRS}
        ${CUDF_CUDA_INCLUDE_DIR}
        )

include(CTest)

#build a binary
macro(tx_add_exe EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

# macro to add a test
function(cylon_add_test TESTNAME)
  set(exec_name "${TESTNAME}")
  add_executable(${exec_name} ${TESTNAME}.cpp test_gutils.hpp)
  target_link_libraries(${exec_name} ${MPI_LIBRARIES} )
  target_link_libraries(${exec_name} cylon)
  target_link_libraries(${exec_name} gcylon)
  target_link_libraries(${exec_name} cudf)
  target_link_libraries(${exec_name} CUDA::cudart)
  target_link_libraries(${exec_name} ${ARROW_LIB})
endfunction(cylon_add_test)

# macro to run executable
function(cylon_run_test TESTNAME no_mpi_proc)
  set(exec_name "${TESTNAME}_${no_mpi_proc}")
  set(mpi_params --allow-run-as-root --oversubscribe --mca opal_cuda_support 1 --mca pml ucx --mca btl_openib_allow_ib true)
  add_test(NAME ${exec_name} COMMAND ${MPI_RUN_CMD} ${mpi_params} -np ${no_mpi_proc} ${CMAKE_BINARY_DIR}/bin/${TESTNAME})
endfunction(cylon_run_test)

#Add tests as follows ...
# param 1 -- name of the test, param 2 -- number of processes

# create table test
cylon_add_test(create_cudf_table_test)
cylon_run_test(create_cudf_table_test 4)

# shuffle table test
cylon_add_test(shuffle_gcylon_table_test)
cylon_run_test(shuffle_gcylon_table_test 4)

# gather table test
cylon_add_test(gather_gcylon_table_test)
cylon_run_test(gather_gcylon_table_test 4)

# broadcast table test
cylon_add_test(bcast_gcylon_table_test)
cylon_run_test(bcast_gcylon_table_test 4)

