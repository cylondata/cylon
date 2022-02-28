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

file(COPY ${CYLON_SOURCE_DIR}/../data DESTINATION ${CMAKE_BINARY_DIR})

include_directories(${CYLON_SOURCE_DIR}/src)
include_directories(${CYLON_SOURCE_DIR}/test)
include_directories(${CMAKE_BINARY_DIR}/thirdparty/glog)
include_directories(${CYLON_SOURCE_DIR}/thirdparty/glog/src)

include(CTest)

#build a binary
macro(tx_add_exe EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

# macro to add a test
function(cylon_add_test TESTNAME)
  set(exec_name "${TESTNAME}")
  add_executable(${exec_name} ${TESTNAME}.cpp)

  target_sources(${exec_name} PRIVATE
          test_macros.hpp
          test_utils.hpp
          test_index_utils.hpp
          test_arrow_utils.hpp)

  target_link_libraries(${exec_name} ${MPI_CXX_LIBRARIES})
  target_link_libraries(${exec_name} ${ARROW_LIB})
  target_link_libraries(${exec_name} cylon)
  target_link_libraries(${exec_name} ${GLOG_LIBRARIES})
endfunction(cylon_add_test)

# macro to run executable
function(cylon_run_test TESTNAME no_mpi_proc)
  set(exec_name "${TESTNAME}_${no_mpi_proc}")
  if (WIN32)
    set(test_params -np ${no_mpi_proc} "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/${TESTNAME}")
  else()
    set(test_params --allow-run-as-root --oversubscribe -np ${no_mpi_proc} "${CMAKE_BINARY_DIR}/bin/${TESTNAME}")
  endif()
  add_test(NAME ${exec_name} COMMAND ${MPIEXEC_EXECUTABLE} ${test_params})
endfunction(cylon_run_test)

#Add tests as follows ...
# param 1 -- name of the test, param 2 -- number of processes

# join tests
cylon_add_test(join_test)
cylon_run_test(join_test 1)
cylon_run_test(join_test 2)
cylon_run_test(join_test 4)

# join parquet tests
cylon_add_test(parquet_join_test)
cylon_run_test(parquet_join_test 1)
cylon_run_test(parquet_join_test 2)
cylon_run_test(parquet_join_test 4)

# set operation tests
cylon_add_test(set_op_test)
cylon_run_test(set_op_test 1)
cylon_run_test(set_op_test 2)
cylon_run_test(set_op_test 4)

# create table test
cylon_add_test(create_table_test)
cylon_run_test(create_table_test 1)

#util tests
cylon_add_test(quick_sort_test)
cylon_run_test(quick_sort_test 1)

# aggergate tests
cylon_add_test(aggregate_test)
cylon_run_test(aggregate_test 1)
cylon_run_test(aggregate_test 2)
cylon_run_test(aggregate_test 4)

#groupby tests
cylon_add_test(groupby_test)
cylon_run_test(groupby_test 1)
cylon_run_test(groupby_test 2)
cylon_run_test(groupby_test 4)

#table op tests
cylon_add_test(table_op_test)
cylon_run_test(table_op_test 1)
cylon_run_test(table_op_test 2)
cylon_run_test(table_op_test 4)

#partition tests
cylon_add_test(partition_test)
cylon_run_test(partition_test 1)
cylon_run_test(partition_test 2)
cylon_run_test(partition_test 4)

#indexing tests
cylon_add_test(indexing_test)
cylon_run_test(indexing_test 1)

# create table test
cylon_add_test(sorting_test)
cylon_run_test(sorting_test 1)

# hash utils test
cylon_add_test(hash_utils_test)
cylon_run_test(hash_utils_test 1)

# utils test
cylon_add_test(utils_test)
cylon_run_test(utils_test 1)

# equal test
cylon_add_test(equal_test)
cylon_run_test(equal_test 1)
cylon_run_test(equal_test 2)

# flatten array test
cylon_add_test(flatten_array_test)
cylon_run_test(flatten_array_test 1)

# comparator test
cylon_add_test(comparator_test)
cylon_run_test(comparator_test 1)

# repartition test
cylon_add_test(repartition_test)
cylon_run_test(repartition_test 1)
cylon_run_test(repartition_test 4)

# serialization test
cylon_add_test(serialize_test)
cylon_run_test(serialize_test 1)

# synchronous communications test
cylon_add_test(sync_comms_test)
cylon_run_test(sync_comms_test 1)
cylon_run_test(sync_comms_test 4)

# distributed sort test
cylon_add_test(dist_sort_test)
cylon_run_test(dist_sort_test 1)
cylon_run_test(dist_sort_test 2)
cylon_run_test(dist_sort_test 4)

# custom mpi comm test
cylon_add_test(custom_mpi_comm_test)
cylon_run_test(custom_mpi_comm_test 1)
cylon_run_test(custom_mpi_comm_test 4)
