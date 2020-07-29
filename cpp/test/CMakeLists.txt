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

SET(MPI_RUN_CMD mpirun)

file(COPY ${CYLON_SOURCE_DIR}/../data DESTINATION ${CMAKE_BINARY_DIR})

include_directories (${CYLON_SOURCE_DIR}/src)
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories (${CMAKE_BINARY_DIR}/thirdparty/glog)
include_directories (${CYLON_SOURCE_DIR}/thirdparty/glog/src)
include_directories (${CYLON_SOURCE_DIR}/src/cylon)

include(CTest)

#build a binary
macro(tx_add_exe EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

#a macro to add a test
function(cylon_add_test TESTNAME no_mpi_proc)
  set(exec_name "${TESTNAME}_${no_mpi_proc}")
  add_executable(${exec_name} ${TESTNAME}.cpp test_utils.hpp)
  target_link_libraries(${exec_name} ${MPI_LIBRARIES})
  target_link_libraries(${exec_name} cylon)
  target_link_libraries(${exec_name} ${ARROW_LIB})
  set(test_params -np ${no_mpi_proc} "${CMAKE_BINARY_DIR}/bin/${exec_name}")
  add_test(NAME ${exec_name} COMMAND ${MPI_RUN_CMD} ${test_params})
endfunction(cylon_add_test)

#Add tests as follows ...
# param 1 -- name of the test, param 2 -- number of processes

# join tests
cylon_add_test(join_test 1)
cylon_add_test(join_test 2)
cylon_add_test(join_test 4)

# set operation tests
cylon_add_test(set_op_test 1)
cylon_add_test(set_op_test 2)
cylon_add_test(set_op_test 4)
