#build a binary
macro(tx_add_exe EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

include_directories(../src)

tx_add_exe(small_twisterx)
tx_add_exe(arrow_twisterx)
tx_add_exe(arrow_test)
tx_add_exe(join_test)
tx_add_exe(mpi_test)

include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(small_twisterx ${MPI_LIBRARIES})
target_link_libraries(small_twisterx twisterx)
target_link_libraries(small_twisterx ${ARROW_LIB})

target_link_libraries(arrow_twisterx ${MPI_LIBRARIES})
target_link_libraries(arrow_twisterx twisterx)
target_link_libraries(arrow_twisterx ${ARROW_LIB})

target_link_libraries(arrow_test ${MPI_LIBRARIES})
target_link_libraries(arrow_test twisterx)
target_link_libraries(arrow_test ${ARROW_LIB})
target_link_libraries(arrow_test glog::glog)

target_link_libraries(join_test ${MPI_LIBRARIES})
target_link_libraries(join_test twisterx)
target_link_libraries(join_test ${ARROW_LIB})
target_link_libraries(join_test glog::glog)

target_link_libraries(mpi_test ${MPI_LIBRARIES})
target_link_libraries(mpi_test twisterx)
target_link_libraries(mpi_test ${ARROW_LIB})
target_link_libraries(mpi_test glog::glog)


