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

  if (CYLON_GLOO)
    target_link_libraries(${exec_name} ${GLOO_LIBRARIES})
  endif (CYLON_GLOO)

endfunction(cylon_add_test)

# macro to run executable
function(cylon_run_test TESTNAME parallelism comm_args)
  set(exec_name "${TESTNAME}_${parallelism}_${comm_args}")
  if (WIN32)
    set(test_params -np ${parallelism} "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/${TESTNAME}" --comm "${comm_args}")
  else ()
    set(test_params --allow-run-as-root --oversubscribe -np ${parallelism} "${CMAKE_BINARY_DIR}/bin/${TESTNAME}" --comm "${comm_args}")
  endif ()
  add_test(NAME ${exec_name} COMMAND ${MPIEXEC_EXECUTABLE} ${test_params})
endfunction(cylon_run_test)

#Add tests as follows ...
# param 1 -- name of the test, param 2 -- number of processes

# repartition test
cylon_add_test(repartition_test)
cylon_run_test(repartition_test 1 mpi)
cylon_run_test(repartition_test 4 mpi)