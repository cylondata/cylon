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

file(MAKE_DIRECTORY ${TEST_DATA_DIR})
file(COPY ${TEST_DATA_DIR} DESTINATION ${CMAKE_BINARY_DIR})

include_directories (${CYLON_SOURCE_DIR}/src)
include(CTest)

#build a binary
macro(tx_add_exe EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

#a macro to add a test
function(tx_add_test TESTNAME no_mpi_proc)
  add_executable(${TESTNAME} ${TESTNAME}.cpp)
  target_link_libraries(${TESTNAME} Catch)
  target_link_libraries(${TESTNAME} ${MPI_LIBRARIES})
  catch_discover_tests(${TESTNAME})
  # The important lines:
  set (test_parameters -np ${no_mpi_proc} "./${TESTNAME}")
  add_test(NAME ${TESTNAME} COMMAND "mpirun-mpich-gcc8" ${test_parameters})
endfunction(tx_add_test)

#Add tests as follows ...
# param 1 -- name of the test, param 2 -- number of processes
