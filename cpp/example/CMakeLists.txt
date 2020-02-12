#build a binary
macro(tx_add_exe EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

include_directories(../src)

tx_add_exe(small_twisterx)
tx_add_exe(arrow_twisterx)
tx_add_exe(arrow_test)

if(MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
  target_link_libraries(small_twisterx ${MPI_LIBRARIES})
  target_link_libraries(small_twisterx twisterxlib)

  target_link_libraries(arrow_twisterx ${MPI_LIBRARIES})
  target_link_libraries(arrow_twisterx twisterxlib)
  target_link_libraries(arrow_twisterx ${ARROW_LIB})

  target_link_libraries(arrow_test ${MPI_LIBRARIES})
  target_link_libraries(arrow_test twisterxlib)
  target_link_libraries(arrow_test ${ARROW_LIB})
endif()

