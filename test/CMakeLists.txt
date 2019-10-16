SET(MPI_RUN_CMD mpirun-mpich-gcc8)

file(COPY ${TEST_DATA_DIR} DESTINATION ${CMAKE_BINARY_DIR})

include_directories (${TWISTERX_SOURCE_DIR}/src)
include(CTest)

#build a binary
macro(tx_add_exe EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

#a macro to add a test
function(tx_add_test TESTNAME no_mpi_proc)
  add_executable(${TESTNAME} ${TESTNAME}.cpp)
  target_link_libraries(${TESTNAME} Catch)
  catch_discover_tests(${TESTNAME})
  # The important lines:
  set (test_parameters -np ${no_mpi_proc} "./${TESTNAME}")
  add_test(NAME ${TESTNAME} COMMAND "mpirun-mpich-gcc8" ${test_parameters})
endfunction(tx_add_test)

#Add tests as follows ...
# param 1 -- name of the test, param 2 -- number of processes
tx_add_test(test_example 1)
